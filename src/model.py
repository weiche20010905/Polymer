"""Conditional Transformer denoiser (DiT-style AdaLN-Zero) for masked diffusion."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000.0):
    """t: (B,) float in [0,1] -> (B, dim) sinusoidal."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * 1000.0 * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class FourierFeatures(nn.Module):
    """Random Fourier features for a scalar; keeps fine resolution on conditions."""

    def __init__(self, out_dim, scale=2.0):
        super().__init__()
        self.register_buffer("W", torch.randn(out_dim // 2) * scale)

    def forward(self, v):  # v: (B,) -> (B, out_dim)
        proj = 2 * math.pi * v[:, None] * self.W[None]
        return torch.cat([proj.cos(), proj.sin()], dim=-1)


class ConditionEmbedder(nn.Module):
    """Embeds K standardized scalar descriptors, with per-feature CFG dropout.

    Each feature gets its own Fourier+MLP branch and its own learned null token,
    so a model trained this way can be conditioned on any subset at sample time.
    """

    def __init__(self, n_feat, dim, fourier_dim=64):
        super().__init__()
        self.n_feat = n_feat
        self.fourier = nn.ModuleList(FourierFeatures(fourier_dim) for _ in range(n_feat))
        self.proj = nn.ModuleList(
            nn.Sequential(nn.Linear(fourier_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
            for _ in range(n_feat)
        )
        self.null = nn.Parameter(torch.zeros(n_feat, dim))
        nn.init.normal_(self.null, std=0.02)
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, cond, cond_mask):
        """cond: (B,K) standardized. cond_mask: (B,K) bool, True = use this feature."""
        acc = 0
        for j in range(self.n_feat):
            e = self.proj[j](self.fourier[j](cond[:, j]))
            keep = cond_mask[:, j : j + 1].to(e.dtype)
            acc = acc + keep * e + (1 - keep) * self.null[j][None]
        return self.out(acc)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), sh1, sc1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1[:, None] * h
        h = modulate(self.norm2(x), sh2, sc2)
        x = x + g2[:, None] * self.mlp(h)
        return x


class DenoiserDiT(nn.Module):
    def __init__(self, vocab_size, max_len, n_feat, dim=512, depth=8, heads=8,
                 mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.t_dim = dim
        self.cond = ConditionEmbedder(n_feat, dim)
        self.blocks = nn.ModuleList(DiTBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth))
        self.norm_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.ada_f[1].weight)
        nn.init.zeros_(self.ada_f[1].bias)
        self.head = nn.Linear(dim, vocab_size)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_t, t, cond, cond_mask):
        """x_t: (B,L) long. t: (B,) in (0,1]. cond/(cond_mask): (B,K)."""
        h = self.tok(x_t) + self.pos
        c = self.t_mlp(timestep_embedding(t, self.t_dim)) + self.cond(cond, cond_mask)
        for blk in self.blocks:
            h = blk(h, c)
        sh, sc = self.ada_f(c).chunk(2, dim=-1)
        return self.head(modulate(self.norm_f(h), sh, sc))
