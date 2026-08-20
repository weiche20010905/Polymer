import argparse, copy, json, os, sys, time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import load_data
from diffusion import diffusion_loss, sample
from model import DenoiserDiT
from utils import generation_metrics, tokens_to_smiles


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for a, b in zip(self.shadow.parameters(), model.parameters()):
            a.lerp_(b.detach(), 1 - self.decay)
        for a, b in zip(self.shadow.buffers(), model.buffers()):
            a.copy_(b)


def lr_at(step, total, base, warmup, min_ratio=0.05):
    if step < warmup:
        return base * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + np.cos(np.pi * min(p, 1.0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pi1m_processed.npz")
    ap.add_argument("--out", default="checkpoints/mdlm_pi1m")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.03)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=2000, help="steps between val+sample")
    ap.add_argument("--sample-n", type=int, default=256)
    ap.add_argument("--sample-steps", type=int, default=128)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    tr, va, meta = load_data(args.data, os.path.join(args.out, "scaler.pkl"))
    itos = meta["itos"]
    print(f"[train] train={len(tr)} val={len(va)} vocab={meta['vocab_size']} "
          f"len={meta['max_len']} n_feat={meta['n_feat']}", flush=True)
    train_smiles = set(np.load(args.data, allow_pickle=True)["smiles"].tolist())

    dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    dlv = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                     num_workers=2, pin_memory=True)

    model = DenoiserDiT(meta["vocab_size"], meta["max_len"], meta["n_feat"],
                        dim=args.dim, depth=args.depth, heads=args.heads).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[train] params {n_par/1e6:.1f}M", flush=True)
    ema = EMA(model, args.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))
    net = torch.compile(model) if args.compile else model

    total_steps = args.epochs * len(dl)
    json.dump(vars(args) | {"vocab_size": meta["vocab_size"], "max_len": meta["max_len"],
                            "n_feat": meta["n_feat"], "params": n_par},
              open(os.path.join(args.out, "config.json"), "w"), indent=2)
    log = open(os.path.join(args.out, "log.csv"), "a")
    if log.tell() == 0:
        log.write("step,epoch,train_loss,val_loss,validity,polymer_valid,uniqueness,novelty,lr,sec\n")

    step, best, t0 = 0, float("inf"), time.time()
    for ep in range(args.epochs):
        model.train()
        run, cnt = 0.0, 0
        for x, c, m in dl:
            lr = lr_at(step, total_steps, args.lr, args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            x, c, m = x.to(dev, non_blocking=True), c.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                loss, _ = diffusion_loss(net, x, c, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            run += loss.item(); cnt += 1; step += 1

            if step % 200 == 0:
                print(f"[{ep} {step}/{total_steps}] loss {run/cnt:.4f} lr {lr:.2e} "
                      f"{(time.time()-t0)/60:.1f}m", flush=True)
            if step % args.eval_every == 0 or step == total_steps:
                vl = evaluate_val(net, dlv, dev)
                gm = quick_sample(ema.shadow, va, meta, args, dev, train_smiles)
                print(f"  >> val {vl:.4f} | " + " ".join(f"{k} {v:.3f}" for k, v in gm.items()),
                      flush=True)
                log.write(f"{step},{ep},{run/max(cnt,1):.5f},{vl:.5f},{gm['validity']:.4f},"
                          f"{gm['polymer_valid']:.4f},{gm['uniqueness']:.4f},"
                          f"{gm.get('novelty',0):.4f},{lr:.3e},{time.time()-t0:.0f}\n")
                log.flush()
                if vl < best:
                    best = vl
                    torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict(),
                                "args": vars(args), "meta": {k: meta[k] for k in
                                ("itos", "desc_names", "vocab_size", "max_len", "n_feat")},
                                "step": step, "val_loss": vl},
                               os.path.join(args.out, "best.pt"))
                model.train()
        torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict(),
                    "args": vars(args), "meta": {k: meta[k] for k in
                    ("itos", "desc_names", "vocab_size", "max_len", "n_feat")},
                    "step": step}, os.path.join(args.out, "last.pt"))
    print(f"[train] done, best val {best:.4f}", flush=True)


@torch.no_grad()
def evaluate_val(net, dlv, dev, n_batches=8, repeats=4):
    net.eval()
    tot, n = 0.0, 0
    for i, (x, c, m) in enumerate(dlv):
        if i >= n_batches:
            break
        x, c, m = x.to(dev), c.to(dev), m.to(dev)
        for _ in range(repeats):   # average over t draws
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                l, _ = diffusion_loss(net, x, c, m)
            tot += l.item(); n += 1
    return tot / max(n, 1)


@torch.no_grad()
def quick_sample(model, va, meta, args, dev, train_smiles):
    idx = np.random.choice(len(va), args.sample_n, replace=len(va) < args.sample_n)
    cond = torch.from_numpy(va.cond[idx]).to(dev)
    cm = torch.ones_like(cond, dtype=torch.bool)
    toks = sample(model, args.sample_n, meta["max_len"], cond, cm,
                  steps=args.sample_steps, guidance=args.guidance, device=dev)
    return generation_metrics(tokens_to_smiles(toks.cpu().numpy(), meta["itos"]), train_smiles)


if __name__ == "__main__":
    main()
