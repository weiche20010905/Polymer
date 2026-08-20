"""Generate polymers conditioned on target RDKit descriptor values.

  python src/sample.py --ckpt checkpoints/mdlm_pi1m/best.pt -n 500 \
      --target MolWt=350 MolLogP=2.5 TPSA=60 --guidance 2.0

Unspecified descriptors are left free (their learned null embedding is used),
so the model only has to honour the constraints you actually care about.
Achieved values are recomputed with RDKit and reported against the targets.
"""
import argparse, os, pickle, sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffusion import sample
from model import DenoiserDiT
from preprocess import DESC_NAMES, descriptors
from utils import generation_metrics, tokens_to_smiles
from rdkit import Chem


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a, meta = ck["args"], ck["meta"]
    model = DenoiserDiT(meta["vocab_size"], meta["max_len"], meta["n_feat"],
                        dim=a["dim"], depth=a["depth"], heads=a["heads"]).to(device)
    model.load_state_dict(ck.get("ema") or ck["model"])
    model.eval()
    scaler = pickle.load(open(os.path.join(os.path.dirname(ckpt_path), "scaler.pkl"), "rb"))
    return model, meta, scaler


def build_condition(targets, scaler, n, device, jitter=0.0):
    """targets: {name: value} in RAW descriptor units -> (cond, cond_mask)."""
    names = scaler["desc_names"]
    raw = np.zeros((1, len(names)), dtype=np.float64)
    mask = np.zeros(len(names), dtype=bool)
    med = scaler["qt"].inverse_transform(np.zeros((1, len(names))))  # median fallback
    raw[0] = med[0]
    for k, v in targets.items():
        if k not in names:
            raise SystemExit(f"unknown descriptor {k}; choose from {names}")
        raw[0, names.index(k)] = v
        mask[names.index(k)] = True
    z = np.clip(scaler["qt"].transform(raw), -scaler["clip"], scaler["clip"])
    cond = np.repeat(z.astype(np.float32), n, axis=0)
    if jitter > 0:
        cond = cond + np.random.randn(*cond.shape).astype(np.float32) * jitter
    cm = np.repeat(mask[None], n, axis=0)
    return torch.from_numpy(cond).to(device), torch.from_numpy(cm).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/mdlm_pi1m/best.pt")
    ap.add_argument("-n", "--n-samples", type=int, default=500)
    ap.add_argument("--target", nargs="*", default=[], metavar="NAME=VALUE",
                    help=f"raw-unit targets, any subset of {DESC_NAMES}")
    ap.add_argument("--guidance", type=float, default=0.0,
                    help="CFG weight; 0 is best here -- see outputs/guidance_sweep.csv")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="noise on the standardized condition; adds diversity")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default="outputs/generated.csv")
    ap.add_argument("--train-smiles", default="data/pi1m_processed.npz")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta, scaler = load_model(args.ckpt, dev)
    targets = {}
    for kv in args.target:
        k, v = kv.split("=")
        targets[k] = float(v)
    print(f"[sample] targets: {targets or '(unconditional)'}  guidance={args.guidance}", flush=True)

    smis = []
    left = args.n_samples
    while left > 0:
        b = min(args.batch, left)
        cond, cm = build_condition(targets, scaler, b, dev, args.jitter)
        toks = sample(model, b, meta["max_len"], cond, cm, steps=args.steps,
                      guidance=args.guidance, temperature=args.temperature,
                      top_p=args.top_p, device=dev)
        smis += tokens_to_smiles(toks.cpu().numpy(), meta["itos"])
        left -= b
        print(f"  {args.n_samples-left}/{args.n_samples}", flush=True)

    train_set = set(np.load(args.train_smiles, allow_pickle=True)["smiles"].tolist()) \
        if os.path.exists(args.train_smiles) else None
    m = generation_metrics(smis, train_set)
    print("[sample] " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)

    good = [s for s in smis if s and s.count("*") == 2]
    rows = []
    for s in good:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        rows.append([s] + descriptors(mol))
    df = pd.DataFrame(rows, columns=["SMILES"] + DESC_NAMES).drop_duplicates("SMILES")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[sample] wrote {len(df)} unique polymers -> {args.out}", flush=True)

    if targets:
        print("\n[condition adherence]  target   mean    median     MAE     within10%")
        for k, v in targets.items():
            got = df[k].to_numpy()
            err = np.abs(got - v)
            tol = max(abs(v) * 0.1, 1e-6)
            print(f"  {k:20s} {v:8.3f} {got.mean():8.3f} {np.median(got):8.3f} "
                  f"{err.mean():8.3f} {(err <= tol).mean():9.1%}")


if __name__ == "__main__":
    main()
