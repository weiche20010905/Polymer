"""PI1M token/condition dataset with per-feature condition dropout for CFG."""
import pickle

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import Dataset


class PI1MDataset(Dataset):
    def __init__(self, tokens, cond, p_uncond=0.10, p_subset=0.30, p_feat_drop=0.5,
                 train=True):
        self.tokens = tokens
        self.cond = cond
        self.p_uncond = p_uncond
        self.p_subset = p_subset
        self.p_feat_drop = p_feat_drop
        self.train = train

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        x = torch.from_numpy(self.tokens[i].astype(np.int64))
        c = torch.from_numpy(self.cond[i])
        m = torch.ones(c.shape[0], dtype=torch.bool)
        if self.train:
            u = np.random.rand()
            if u < self.p_uncond:                       # fully unconditional (CFG)
                m[:] = False
            elif u < self.p_uncond + self.p_subset:     # random subset of features
                m = torch.from_numpy(np.random.rand(c.shape[0]) >= self.p_feat_drop)
        return x, c, m


def load_data(npz_path, scaler_path, val_frac=0.01, seed=0, clip=5.0):
    d = np.load(npz_path, allow_pickle=True)
    tokens = d["tokens"].astype(np.int16)
    desc = d["desc"].astype(np.float64)
    itos = list(d["itos"])
    desc_names = list(d["desc_names"])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tokens))
    n_val = max(1, int(len(tokens) * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                             subsample=200_000, random_state=seed)
    qt.fit(desc[tr_idx])
    cond = np.clip(qt.transform(desc), -clip, clip).astype(np.float32)
    with open(scaler_path, "wb") as f:
        pickle.dump({"qt": qt, "clip": clip, "desc_names": desc_names, "itos": itos}, f)

    return (PI1MDataset(tokens[tr_idx], cond[tr_idx], train=True),
            PI1MDataset(tokens[val_idx], cond[val_idx], train=False),
            {"itos": itos, "desc_names": desc_names, "vocab_size": len(itos),
             "max_len": tokens.shape[1], "n_feat": desc.shape[1],
             "raw_desc_val": desc[val_idx]})
