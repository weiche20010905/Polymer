"""Conditioning-adherence sweep: does the model actually hit the requested values?

For each descriptor, sweeps the target across PI1M percentiles, generates samples
conditioned only on that descriptor, and reports achieved median + Spearman rho
between requested and achieved. Also writes a target-vs-achieved figure.
"""
import argparse, os, pickle, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffusion import sample
from preprocess import DESC_NAMES, descriptors
from sample import build_condition, load_model
from utils import generation_metrics, tokens_to_smiles


def plot(raw, features, guidance, path):
    k = len(features)
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 3.6), squeeze=False)
    for ax, feat in zip(axes[0], features):
        g = raw[raw.feature == feat]
        if g.empty:
            continue
        tg = sorted(g["target"].unique())
        ax.violinplot([g.achieved[g.target == t].values for t in tg],
                      positions=range(len(tg)), showmedians=True)
        ax.plot(range(len(tg)), tg, "r--o", ms=4, label="target")
        ax.set_xticks(range(len(tg)))
        ax.set_xticklabels([f"{t:g}" for t in tg])
        ax.set_title(feat); ax.set_xlabel("target"); ax.set_ylabel("achieved")
        ax.legend(fontsize=7)
    fig.suptitle(f"Condition adherence (CFG w={guidance})")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/mdlm_pi1m/best.pt")
    ap.add_argument("--data", default="data/pi1m_processed.npz")
    ap.add_argument("--features", nargs="*", default=["MolWt", "MolLogP", "TPSA", "RingCount"])
    ap.add_argument("--percentiles", nargs="*", type=float, default=[10, 30, 50, 70, 90])
    ap.add_argument("-n", "--n-per-target", type=int, default=256)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--outdir", default="outputs/eval")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta, scaler = load_model(args.ckpt, dev)
    d = np.load(args.data, allow_pickle=True)
    ref = pd.DataFrame(d["desc"], columns=list(d["desc_names"]))
    train_set = set(d["smiles"].tolist())

    rows, raw_rows = [], []
    for feat in args.features:
        for p in args.percentiles:
            tgt = float(np.percentile(ref[feat], p))
            cond, cm = build_condition({feat: tgt}, scaler, args.n_per_target, dev)
            toks = sample(model, args.n_per_target, meta["max_len"], cond, cm,
                          steps=args.steps, guidance=args.guidance, device=dev)
            smis = tokens_to_smiles(toks.cpu().numpy(), meta["itos"])
            gm = generation_metrics(smis, train_set)
            got = []
            for s in {x for x in smis if x and x.count("*") == 2}:
                mol = Chem.MolFromSmiles(s)
                if mol is not None:
                    got.append(descriptors(mol)[DESC_NAMES.index(feat)])
            got = np.array(got, dtype=float)
            if got.size == 0:
                continue
            rows.append(dict(feature=feat, pct=p, target=tgt, n=got.size,
                             achieved_mean=got.mean(), achieved_median=np.median(got),
                             mae=np.abs(got - tgt).mean(), std=got.std(),
                             polymer_valid=gm["polymer_valid"], uniqueness=gm["uniqueness"],
                             novelty=gm.get("novelty", np.nan)))
            raw_rows += [(feat, tgt, v) for v in got]
            print(f"  {feat:16s} p{p:>4.0f} target {tgt:9.3f} -> median {np.median(got):9.3f} "
                  f"MAE {np.abs(got-tgt).mean():8.3f} valid {gm['polymer_valid']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "adherence.csv"), index=False)
    raw = pd.DataFrame(raw_rows, columns=["feature", "target", "achieved"])
    raw.to_csv(os.path.join(args.outdir, "adherence_raw.csv"), index=False)

    print("\n[summary] Spearman(target, achieved) per feature:")
    for feat, g in raw.groupby("feature"):
        rho = spearmanr(g["target"], g["achieved"]).statistic
        print(f"  {feat:16s} rho = {rho:.3f}   (n={len(g)})")

    plot(raw, args.features, args.guidance,
         os.path.join(args.outdir, "adherence.png"))
    print(f"\n[eval] wrote {args.outdir}/adherence.{{csv,png}}")


if __name__ == "__main__":
    main()
