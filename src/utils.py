"""Token <-> SMILES conversion and generation metrics."""
import re

import numpy as np
import selfies as sf
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
PAD_ID, MASK_ID = 0, 1
_AU = re.compile(r"\[[^\]]*Au[^\]]*\]")  # [Au], [AuH1], [Au+] ... -> polymer endpoint


def tokens_to_smiles(tokens, itos):
    """(B,L) int array -> list of polymer SMILES ('*' endpoints) or None."""
    out = []
    for row in np.asarray(tokens):
        toks = []
        for i in row:
            if i in (PAD_ID, MASK_ID):
                break
            toks.append(itos[int(i)])
        if not toks:
            out.append(None)
            continue
        try:
            smi = sf.decoder("".join(toks))
            smi = _AU.sub("*", smi)
            mol = Chem.MolFromSmiles(smi)
            out.append(Chem.MolToSmiles(mol) if mol is not None else None)
        except Exception:
            out.append(None)
    return out


def generation_metrics(smiles_list, train_set=None):
    n = len(smiles_list)
    valid = [s for s in smiles_list if s]
    poly = [s for s in valid if s.count("*") == 2]
    uniq = set(poly)
    m = {
        "validity": len(valid) / max(n, 1),
        "polymer_valid": len(poly) / max(n, 1),   # valid AND exactly 2 endpoints
        "uniqueness": len(uniq) / max(len(poly), 1),
    }
    if train_set is not None:
        m["novelty"] = len(uniq - train_set) / max(len(uniq), 1)
    return m
