"""PI1M -> SELFIES tokens + RDKit descriptor conditions.

Polymer endpoints `*` are swapped to [Au] for SELFIES encoding (SELFIES has no
wildcard atom); descriptors are computed on the original `*` form so the dummy
atoms contribute ~0 mass instead of 197 g/mol each.
"""
import argparse, os, pickle, sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# SA score lives in RDKit's Contrib dir, not the main package.
try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
except Exception:
    sascorer = None

DESC_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "FractionCSP3", "QED", "SAscore",
]
PLACEHOLDER = "[Au]"


def descriptors(mol):
    d = [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
    ]
    try:
        d.append(QED.qed(mol))
    except Exception:
        d.append(np.nan)
    d.append(sascorer.calculateScore(mol) if sascorer is not None else np.nan)
    return d


def process(smiles):
    """-> (selfies_string, descriptor_list, canonical_smiles) or None."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() < 3:
            return None
        desc = descriptors(mol)
        if not np.all(np.isfinite(desc)):
            return None
        can = Chem.MolToSmiles(mol)
        au = Chem.MolToSmiles(Chem.MolFromSmiles(smiles.replace("*", PLACEHOLDER)))
        s = sf.encoder(au)
        if s is None or sf.decoder(s) is None:
            return None
        return s, desc, can
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/PI1M.csv")
    ap.add_argument("--out", default="data/pi1m_processed.npz")
    ap.add_argument("--max-len", type=int, default=72, help="max SELFIES tokens")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    smiles = pd.read_csv(args.csv)["SMILES"].astype(str).tolist()
    if args.limit:
        smiles = smiles[: args.limit]
    print(f"[preprocess] {len(smiles)} input SMILES, {args.workers} workers", flush=True)

    with Pool(args.workers) as pool:
        res = pool.map(process, smiles, chunksize=512)
    ok = [r for r in res if r is not None]
    print(f"[preprocess] parsed {len(ok)} ({len(ok)/len(smiles):.1%})", flush=True)

    selfies_list = [r[0] for r in ok]
    lens = np.array([sf.len_selfies(s) for s in selfies_list])
    print("[preprocess] SELFIES token length pct:",
          {p: int(np.percentile(lens, p)) for p in (50, 90, 95, 99, 100)}, flush=True)

    keep = lens <= args.max_len
    print(f"[preprocess] keep len<={args.max_len}: {keep.sum()} ({keep.mean():.1%})", flush=True)
    selfies_list = [s for s, k in zip(selfies_list, keep) if k]
    desc = np.array([r[1] for r, k in zip(ok, keep) if k], dtype=np.float32)
    can = [r[2] for r, k in zip(ok, keep) if k]

    # vocabulary: 0 = [PAD] (also the absorbing/eos filler), 1 = [MASK]
    alphabet = sorted(sf.get_alphabet_from_selfies(selfies_list))
    itos = ["[PAD]", "[MASK]"] + alphabet
    stoi = {t: i for i, t in enumerate(itos)}
    print(f"[preprocess] vocab {len(itos)} (incl PAD/MASK)", flush=True)

    max_len = args.max_len
    tokens = np.zeros((len(selfies_list), max_len), dtype=np.int16)
    for i, s in enumerate(selfies_list):
        ids = [stoi[t] for t in sf.split_selfies(s)]
        tokens[i, : len(ids)] = ids

    np.savez_compressed(
        args.out, tokens=tokens, desc=desc,
        desc_names=np.array(DESC_NAMES), itos=np.array(itos),
        smiles=np.array(can, dtype=object),
    )
    with open(os.path.splitext(args.out)[0] + "_vocab.pkl", "wb") as f:
        pickle.dump({"itos": itos, "stoi": stoi, "max_len": max_len,
                     "desc_names": DESC_NAMES}, f)
    print(f"[preprocess] wrote {args.out}  tokens={tokens.shape} desc={desc.shape}", flush=True)
    for i, n in enumerate(DESC_NAMES):
        print(f"  {n:20s} mean={desc[:, i].mean():9.3f} std={desc[:, i].std():8.3f} "
              f"min={desc[:, i].min():9.3f} max={desc[:, i].max():9.3f}", flush=True)


if __name__ == "__main__":
    main()
