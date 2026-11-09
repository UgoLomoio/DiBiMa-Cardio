"""Deterministic DDIM sampling for every UniCardio conditioning pattern.

Upstream UniCardio (UniCardio/base_model/diffusion_model_no_compress_final.py) ships a
DDIM sampler only for one-condition flags (two digits, e.g. "02"); two- and
three-condition flags ("012", "0123", ...) are always sampled with 50-step ancestral
DDPM, and the one-condition branch for flag "21" (ECG->BP) has a defect (it writes
x0_hat instead of x_next into the state).

`ddim_sample` re-implements upstream `one_condition_DDIM_imputation` for ANY flag:

  * same time grid: arange(0, num_steps, num_steps // sample_steps) reversed
    (50 steps, 6 sample steps -> 48, 40, 32, 24, 16, 8, 0 -> six updates),
  * same deterministic (eta = 0) update with alpha = cumulative alpha_bar:
        x0_hat = (x_t - sqrt(1 - a_t) * eps) / sqrt(a_t)
        x_next = sqrt(a_next) * x0_hat + sqrt(1 - a_next) * eps
  * same initial state, masks and conditioning as the upstream DDPM samplers:
    observed source slots are copied from the input, every other slot of the
    4 x L/4 state starts as one shared N(0, I) draw per sample, and only the target
    slot (last flag digit) is updated,
  * same output: the target slot after the last update, n_samples draws.

For one-condition flags other than "21" it is numerically identical to upstream
(checked by `validate_against_upstream`). Nothing in UniCardio/ is modified;
`install(model)` only wraps the instance's `generate` method.
"""
from __future__ import annotations

import types

import torch

SLOT_MASK = {"0": "mask1", "1": "mask2", "2": "mask3",
             "01": "mask12", "02": "mask13", "12": "mask23", "012": "mask123"}


def _slots(L):
    q = L // 4
    return [slice(i * q, (i + 1) * q) for i in range(4)]


def ddim_sample(model, observed_data, n_samples, sample_steps, model_flag, borrow_mode,
                ratio=1.0):
    flag = str(model_flag)
    sources, target = flag[:-1], int(flag[-1])
    if sources not in SLOT_MASK:
        raise ValueError(f"Unsupported UniCardio model_flag {flag!r}")
    mask = getattr(model, SLOT_MASK[sources]).to(model.device)
    with torch.no_grad():
        B, K, L = observed_data.shape
        sl = _slots(L)
        q = L // 4
        total = model.num_steps
        skip = total // sample_steps
        grid = list(range(0, total, skip))[::-1]
        out = torch.ones(B, n_samples, K, q, device=model.device)
        for i in range(n_samples):
            noise = torch.randn(B, K, q).to(model.device)
            parts = [observed_data[:, :, sl[s]] if str(s) in sources else noise for s in range(4)]
            x = torch.concat(parts, dim=-1)
            for k in range(len(grid) - 1):
                t, t_next = grid[k], grid[k + 1]
                eps = model.diffmodel(x, torch.tensor([t]).to(model.device), mask,
                                      mode=target, borrow_mode=borrow_mode) / ratio
                a_t, a_next = model.alpha[t], model.alpha[t_next]
                x0_hat = (x[:, :, sl[target]] - ((1 - a_t) ** 0.5) * eps) / a_t ** 0.5
                x[:, :, sl[target]] = (a_next ** 0.5) * x0_hat + ((1 - a_next) ** 0.5) * eps
            out[:, i] = x[:, :, sl[target]].detach()
        return out


def install(model, use_upstream_single=True, defective=("21",)):
    """Route every DDIM_flag=1 request through DDIM, for all flag lengths.

    use_upstream_single: keep upstream one_condition_DDIM_imputation for two-digit
    flags that are not known-defective (bit-identical to ddim_sample anyway).
    """
    original = model.generate

    def generate(self, observed_data, n_samples, model_flag, borrow_mode, sample_steps=6,
                 DDIM_flag=0, ratio=1, improved=0):
        flag = str(model_flag)
        if DDIM_flag != 1:
            return original(observed_data, n_samples, model_flag, borrow_mode,
                            sample_steps=sample_steps, DDIM_flag=DDIM_flag, ratio=ratio,
                            improved=improved)
        if use_upstream_single and len(flag) == 2 and flag not in defective:
            return original(observed_data, n_samples, model_flag, borrow_mode,
                            sample_steps=sample_steps, DDIM_flag=1, ratio=ratio,
                            improved=improved)
        return ddim_sample(self, observed_data, n_samples, sample_steps, flag, borrow_mode,
                           ratio)

    model.generate = types.MethodType(generate, model)
    model._ddim_all_installed = True
    return model


def validate_against_upstream(model, observed_data, flags, n_samples=2, sample_steps=6,
                              borrow=None, seed=0):
    """Max |difference| between ddim_sample and upstream DDIM on one-condition flags."""
    res = {}
    for f in flags:
        b = borrow[f] if borrow else int(f[-1]) % 3
        torch.manual_seed(seed)
        up = model.one_condition_DDIM_imputation(observed_data, n_samples, sample_steps, f, b, 1)
        up = up[0] if isinstance(up, tuple) else up
        torch.manual_seed(seed)
        mine = ddim_sample(model, observed_data, n_samples, sample_steps, f, b)
        res[f] = float((up - mine).abs().max())
    return res
