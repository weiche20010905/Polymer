"""How CFG strength trades condition adherence against diversity/validity."""
import argparse, os, sys

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffusion import sample
from preprocess import DESC_NAMES, descriptors
from sample import build_condition, load_model
from utils import generation_metrics, tokens_to_smiles

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/mdlm_pi1m/best.pt")
ap.add_argument("--data", default="data/pi1m_processed.npz")
ap.add_argument("--target", nargs="*", default=["MolWt=350", "MolLogP=2.5", "TPSA=60"])
ap.add_argument("--guidances", nargs="*", type=float, default=[0, 0.5, 1, 2, 3, 5])
ap.add_argument("-n", type=int, default=512)
ap.add_argument("--steps", type=int, default=128)
ap.add_argument("--out", default="outputs/guidance_sweep.csv")
a = ap.parse_args()

dev = "cuda"
model, meta, scaler = load_model(a.ckpt, dev)
train_set = set(np.load(a.data, allow_pickle=True)["smiles"].tolist())
targets = {k: float(v) for k, v in (t.split("=") for t in a.target)}

rows = []
for g in a.guidances:
    cond, cm = build_condition(targets, scaler, a.n, dev)
    toks = sample(model, a.n, meta["max_len"], cond, cm, steps=a.steps, guidance=g, device=dev)
    smis = tokens_to_smiles(toks.cpu().numpy(), meta["itos"])
    m = generation_metrics(smis, train_set)
    got = {k: [] for k in targets}
    uniq = {s for s in smis if s and s.count("*") == 2}
    for s in uniq:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        d = descriptors(mol)
        for k in targets:
            got[k].append(d[DESC_NAMES.index(k)])
    r = dict(guidance=g, n_unique=len(uniq), **{k: round(v, 4) for k, v in m.items()})
    for k, v in targets.items():
        arr = np.array(got[k])
        r[f"{k}_median"] = round(float(np.median(arr)), 3)
        r[f"{k}_mae"] = round(float(np.abs(arr - v).mean()), 3)
        r[f"{k}_std"] = round(float(arr.std()), 3)
    rows.append(r)
    print(r, flush=True)

df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
print(f"\nwrote {a.out}")
print(df.to_string(index=False))
