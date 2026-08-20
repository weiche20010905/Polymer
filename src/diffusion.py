"""Absorbing-state (masked) discrete diffusion, MDLM-style continuous time.

Forward:  each token is independently replaced by [MASK] with probability t,
          using the linear schedule alpha_t = 1 - t.
Loss:     E_t [ (1/t) * sum_{masked i} CE(x0_i, p_theta) ]  -- an NLL bound.
Reverse:  ancestral "carry-over unmasking"; a token, once unmasked, is frozen.
"""
import torch
import torch.nn.functional as F

MASK_ID = 1
PAD_ID = 0


def sample_t(batch, device, eps=1e-3, antithetic=True):
    """Low-discrepancy t in [eps, 1] to cut loss variance."""
    u = torch.rand(1 if antithetic else batch, device=device)
    if antithetic:
        offs = torch.arange(batch, device=device) / batch
        u = (u + offs) % 1.0
    return (1 - eps) * u + eps


def q_sample(x0, t):
    """Mask each position independently with probability t."""
    mask = torch.rand(x0.shape, device=x0.device) < t[:, None]
    return torch.where(mask, torch.full_like(x0, MASK_ID), x0), mask


def diffusion_loss(model, x0, cond, cond_mask, eps=1e-3):
    t = sample_t(x0.shape[0], x0.device, eps)
    x_t, mask = q_sample(x0, t)
    logits = model(x_t, t, cond, cond_mask)
    # never let the model put probability on [MASK] itself
    logits[..., MASK_ID] = -1e9
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), x0.reshape(-1), reduction="none"
    ).view(x0.shape)
    per_seq = (ce * mask).sum(dim=1) / t          # 1/t weighting
    return per_seq.mean() / x0.shape[1], (ce * mask).sum() / mask.sum().clamp(min=1)


def _sample_tokens(logits, temperature, top_p):
    logits = logits.float()
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-5)
    if top_p < 1.0:
        srt, idx = logits.sort(dim=-1, descending=True)
        cum = srt.softmax(-1).cumsum(-1)
        drop = cum - srt.softmax(-1) > top_p
        srt = srt.masked_fill(drop, -1e9)
        logits = torch.empty_like(logits).scatter_(-1, idx, srt)
    probs = logits.softmax(-1)
    flat = probs.view(-1, probs.shape[-1])
    return torch.multinomial(flat, 1).view(probs.shape[:-1])


@torch.no_grad()
def sample(model, n, seq_len, cond, cond_mask, steps=128, guidance=0.0,
           temperature=1.0, top_p=1.0, device="cuda", dtype=torch.bfloat16):
    """cond/(cond_mask): (n, K). guidance = classifier-free guidance weight."""
    model.eval()
    x = torch.full((n, seq_len), MASK_ID, dtype=torch.long, device=device)
    null_mask = torch.zeros_like(cond_mask)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    for i in range(steps):
        t, s = ts[i], ts[i + 1]
        tb = t.expand(n)
        with torch.autocast("cuda", dtype=dtype, enabled=device.startswith("cuda")):
            if guidance > 0:
                xx = torch.cat([x, x], 0)
                tt = torch.cat([tb, tb], 0)
                cc = torch.cat([cond, cond], 0)
                mm = torch.cat([cond_mask, null_mask], 0)
                lg = model(xx, tt, cc, mm)
                lc, lu = lg.chunk(2, 0)
                logits = lu + (1 + guidance) * (lc - lu)
            else:
                logits = model(x, tb, cond, cond_mask)
        logits = logits.float()
        logits[..., MASK_ID] = -1e9
        x0 = _sample_tokens(logits, temperature, top_p)

        is_masked = x == MASK_ID
        if i == steps - 1:
            unmask = is_masked
        else:
            p_unmask = ((t - s) / t).clamp(0, 1)
            unmask = is_masked & (torch.rand(x.shape, device=device) < p_unmask)
        x = torch.where(unmask, x0, x)
    return x
