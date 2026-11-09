"""DiBiMa-Cardio inference engine (no Gradio dependency).

Models
------
* DiBiMa-Cardio   - diffusion-trained bidirectional-Mamba (Mamba2) model, 1 network pass per window
                    (t = T-1 of a 20-step schedule), template prior for gaps and forecasts.
* DiBiMa-Cardio-M - multimodal fine-tune of DiBiMa-Cardio (gated Bi-Mamba fusion bottleneck, early fusion,
                    decoder fusion adapters, time/modality/mask-aware cross-attention).
* UniCardio-M     - comparison baseline: the released UniCardio checkpoint with deterministic DDIM sampling
                    (6 steps x 6 draws, scored on the mean of the draws).

Data
----
`Bundle` loads data/test256_inputs.npz: 256 held-out 4-s windows (125 Hz) of PPG, ECG and BP from the UCI
cuff-less blood-pressure dataset, with the exact references, gap masks and noise realisations of the paper.

Corruption
----------
`Corruption` selects the gap length, forecast horizon and noise scale. HARD (2.0 s gap centred on the stored
gap position, last 2.0 s hidden, 2x the stored noise residual) is the primary protocol of the paper;
STANDARD uses the stored benchmark corruption (1.2 s gap, ~1.17 s horizon, 1x noise) bit for bit.

Input contract (DiBiMa-Cardio / DiBiMa-Cardio-M)
-----------------------------------------------
[B, 6+3K, L] input = [ppg, ecg, bp, mask_ppg, mask_ecg, mask_bp, prior block], t = T-1 (T = 20); self routes
[0_0,1_1,2_2] for restoration, [s_0,s_1,s_2] for single-source translation; the template prior is given only
for the corrupted target of imputation/forecast. Batches of 8 (as in the paper's evaluation).

Metrics
-------
Per window: MAE, MSE, RMSE, PCC, SNR, PSNR, KS statistic, on the scales native / ref_zscore / legacy_shape
(each window min-max to [-1, 1]), over the full window or the gap / horizon only. Each model is scored against
its OWN reference preprocessing (DiBiMa-Cardio: record-level z-scored windows; UniCardio: its preprocessed
windows), so cross-model comparisons use the common-unit scales.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

MODALITIES = ("ppg", "ecg", "bp")
KINDS = ("imputation", "denoising", "forecast", "sr")
METRICS = ("MAE", "MSE", "RMSE", "PCC", "SNR", "PSNR", "KS_stat")
SCALES = ("native", "ref_zscore", "legacy_shape")
L = 500
FS = 125
BATCH = 8
INFERENCE_SEED = 42
UNI_SEGMENT_ORDER = ("ppg", "bp", "ecg", "aux")


# ----------------------------------------------------------------------------- tasks
@dataclasses.dataclass(frozen=True)
class Task:
    kind: str
    target: str
    sources: tuple
    primary: str | None = None

    @property
    def name(self):
        return (f"{'+'.join(self.sources)}_to_{self.target}"
                if self.kind == "translation" else f"{self.target}_{self.kind}")

    @property
    def variant(self):
        if self.kind == "translation":
            return "+".join(m.upper() for m in self.sources)
        suffix = {"imputation": "I", "denoising": "N", "forecast": "F", "sr": "SR"}[self.kind]
        return "+".join([f"{self.target.upper()}-{suffix}"] +
                        [m.upper() for m in self.sources if m != self.target])

    @property
    def key(self):
        return f"{self.name}::{self.variant}"

    @property
    def n_inputs(self):
        return len(self.sources)


def task_roster():
    """All 57 task-variant combinations of the evaluator (multi_modal=True), same order."""
    tasks = []
    for target in MODALITIES:
        other = [m for m in MODALITIES if m != target]
        aux = [c for n in range(3) for c in itertools.combinations(other, n)]
        for kind in KINDS:
            tasks.extend(Task(kind, target, (target, *a), target) for a in aux)
    for target in MODALITIES:
        other = [m for m in MODALITIES if m != target]
        for n in (1, 2):
            tasks.extend(Task("translation", target, tuple(s)) for s in itertools.combinations(other, n))
    return tasks


def make_task(kind, target, inputs):
    """Build the evaluator Task from UI choices. inputs: iterable of modality names."""
    inputs = [m for m in MODALITIES if m in set(inputs)]
    if kind == "translation":
        src = tuple(m for m in inputs if m != target)
        if not src:
            raise ValueError("Translation needs at least one input signal different from the output.")
        return Task("translation", target, src)
    aux = tuple(m for m in inputs if m != target)
    return Task(kind, target, (target, *aux), target)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- data
class Bundle:
    """The 256 test windows with the evaluator's exact corrupted inputs and masks."""

    def __init__(self, path=HERE / "data" / "test256_inputs.npz"):
        z = np.load(path, allow_pickle=False)
        self.d = {k: z[k] for k in z.files}
        self.n = len(self.d["sample_id"])
        self.sample_id = [str(s) for s in self.d["sample_id"]]
        self.record_id = [str(s) for s in self.d["record_id"]]

    def native(self, block, m, idx):
        return self.d[f"native_{block}_{m}"][idx]

    def mask(self, kind, m, idx):
        return self.d[f"mask_{kind}_{m}"][idx].astype(np.float32)

    def uni_clean(self, m, idx):
        return self.d[f"uni_clean_{m}"][idx]


CORRUPTED_BLOCK = {"imputation": "masked", "denoising": "noised", "sr": "masked_sr", "forecast": "masked_forecast"}


@dataclasses.dataclass(frozen=True)
class Corruption:
    """gap_s / horizon_s = None keep the stored benchmark masks; noise_alpha scales the stored noise residual."""
    gap_s: float | None = 2.0
    horizon_s: float | None = 2.0
    noise_alpha: float = 2.0

    @property
    def label(self):
        if self == STANDARD:
            return "standard benchmark corruption (1.2 s gap, ~1.17 s horizon, 1x noise)"
        if self == HARD:
            return "hard corruption (2.0 s gap/horizon, 2x noise)"
        g = "stored 1.2 s" if self.gap_s is None else f"{self.gap_s:g} s"
        h = "stored ~1.17 s" if self.horizon_s is None else f"{self.horizon_s:g} s"
        return f"custom corruption ({g} gap, {h} horizon, {self.noise_alpha:g}x noise)"


HARD = Corruption(2.0, 2.0, 2.0)
STANDARD = Corruption(None, None, 1.0)


def task_mask(bundle, kind, m, idx, corr=None):
    """(N,L) float32 observed-mask for imputation/forecast under `corr` (None = STANDARD).
    Imputation: n = round(gap_s*FS) samples centred on the stored gap's centre;
    forecast: the last round(horizon_s*FS) samples hidden."""
    corr = corr or STANDARD
    old = bundle.mask(kind, m, idx)
    length = corr.gap_s if kind == "imputation" else corr.horizon_s
    if length is None:
        return old
    n = int(round(length * FS))
    if not 0 < n < L:
        raise ValueError(f"{kind} length must be between 1 sample and the 4 s window (got {length} s).")
    new = np.ones_like(old, dtype=np.float32)
    if kind == "imputation":
        miss = (old == 0).astype(np.float32)
        centre = np.round((miss * np.arange(L, dtype=np.float32)).sum(1, dtype=np.float32)
                          / np.maximum(miss.sum(1, dtype=np.float32), np.float32(1))).astype(np.int64)
        start = np.clip(centre - n // 2, 0, L - n)
        for i, s0 in enumerate(start):
            new[i, s0:s0 + n] = 0
    else:
        new[:, L - n:] = 0
    return new


def corrupted_block(bundle, kind, m, idx, corr=None):
    """Corrupted native input of modality m for task kind under `corr` (None = STANDARD)."""
    corr = corr or STANDARD
    if kind in ("imputation", "forecast") and (corr.gap_s if kind == "imputation" else corr.horizon_s) is not None:
        return (bundle.native("targets", m, idx) * task_mask(bundle, kind, m, idx, corr)).astype(np.float32)
    if kind == "denoising" and float(corr.noise_alpha) != 1.0:
        t = bundle.native("targets", m, idx).astype(np.float32)
        return (t + np.float32(corr.noise_alpha) * (bundle.native("noised", m, idx).astype(np.float32) - t)).astype(np.float32)
    return bundle.native(CORRUPTED_BLOCK[kind], m, idx)


def dibima_inputs(bundle, task, idx, corr=None):
    """Model inputs (paper evaluation contract) for windows idx -> signals/masks dicts (N,L)."""
    ref = bundle.native("targets", task.target, idx)
    signals = {m: np.zeros_like(ref) for m in MODALITIES}
    masks = {m: np.zeros_like(ref) for m in MODALITIES}
    for m in task.sources:
        primary = task.kind != "translation" and m == task.primary
        signals[m] = (corrupted_block(bundle, task.kind, m, idx, corr) if primary
                      else bundle.native("targets", m, idx)).copy()
        if primary and task.kind in ("imputation", "forecast"):
            masks[m] = task_mask(bundle, task.kind, m, idx, corr)
        else:
            masks[m] = np.ones_like(ref)
    return signals, masks


def train_consistent_flags(task):
    if task.kind != "translation":
        return ["0_0", "1_1", "2_2"]
    if len(task.sources) == 1:
        s = MODALITIES.index(task.sources[0])
        return [f"{s}_{j}" for j in range(3)]
    return [f"{MODALITIES.index(m)}_{MODALITIES.index(task.target)}" for m in task.sources]


def scoring_regions(bundle, task, idx, corr=None):
    regs = {"full": None}
    if task.kind in ("imputation", "forecast"):
        regs["missing" if task.kind == "imputation" else "future"] = task_mask(bundle, task.kind, task.target, idx, corr) == 0
    return regs


# ----------------------------------------------------------------------------- UniCardio contract
def load_uni_tasks(path=HERE / "data" / "uni_tasks_ddim.json"):
    return json.load(open(path))


def uni_entry(config, task):
    e = config.get(task.key)
    if e is None and len(task.sources) == 1:
        e = config.get(task.name)
    return e


def uni_supports(config, task):
    if task.kind == "sr":
        return False, ("UniCardio has no super-resolution mode for this reference preprocessing "
                       "(not evaluable, as in the manuscript).")
    e = uni_entry(config, task)
    if not isinstance(e, dict) or e.get("verified") is not True or e.get("model_flag") is None:
        return False, "No verified UniCardio flag for this input/output combination."
    return True, ""


def unicardio_inputs(bundle, task, idx, entry, corr=None):
    """UniCardioHook.evaluate input construction (derived_shared_standardized_noise protocol).
    Returns x (N,1,2000) float32 and the UniCardio reference y (N,L)."""
    clean = {m: bundle.uni_clean(m, idx).astype(np.float32) for m in MODALITIES}
    native_t = {m: bundle.native("targets", m, idx) for m in MODALITIES}
    fill, auxv = float(entry["fill_value"]), float(entry["aux_value"])
    available = {}
    for m in task.sources:
        primary = task.kind != "translation" and m == task.primary
        if not primary:
            available[m] = clean[m]
        elif task.kind == "denoising":
            residual = corrupted_block(bundle, "denoising", m, idx, corr) - native_t[m]
            native_sd = native_t[m].std(axis=1, keepdims=True)
            uni_sd = clean[m].std(axis=1, keepdims=True)
            if np.any(native_sd == 0) or np.any(uni_sd == 0):
                raise ValueError("Cannot map standardized noise for a constant target.")
            slope = bundle.d[f"uni_slope_{m}"][idx].astype(np.float32)[:, None]
            valid = bundle.d[f"uni_valid_{m}"][idx][:, None]
            sign = np.where(np.isfinite(slope) & (slope != 0), np.sign(slope), 1.0)
            ratio = np.where(valid, slope, (uni_sd / native_sd) * sign)
            available[m] = (clean[m] + residual * ratio).astype(np.float32)
        elif task.kind in ("imputation", "forecast"):
            observed = task_mask(bundle, task.kind, m, idx, corr).astype(bool)
            available[m] = np.where(observed, clean[m], np.float32(fill)).astype(np.float32)
        else:
            raise ValueError("UniCardio SR is not supported.")
    blank = np.full_like(clean[task.target], fill)
    segs = [np.full_like(blank, auxv) if m == "aux" else available.get(m, blank) for m in UNI_SEGMENT_ORDER]
    x = np.concatenate(segs, axis=1)[:, None, :].astype(np.float32)
    return x, clean[task.target]


# ----------------------------------------------------------------------------- models
def _set_device_attrs(module, device):
    """UniCardio stores `.device` on several submodules and builds tensors with it at call time."""
    import torch
    for mod in module.modules():
        if isinstance(getattr(mod, "device", None), (torch.device, str)) and not isinstance(
                type(mod).__dict__.get("device"), property):
            try:
                mod.device = device
            except Exception:
                pass


def configure_torch():
    """Global numerics used for training and for the paper's evaluation (TF32 matmuls)."""
    import torch
    try:
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except Exception:
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")


DIBIMA_VARIANTS = {
    # key: (display name, weights, params YAML); both use the v5 template-prior file
    "v5": ("DiBiMa-Cardio", "dibima_v5.pt", "dibima_v5_params.yaml"),
    "m": ("DiBiMa-Cardio-M", "dibima_m.pt", "dibima_m_params.yaml"),
}
TEMPLATES_FILE = "dibima_v5_ext_templates.npz"
DIBIMA_FILES = tuple(sorted({f for v in DIBIMA_VARIANTS.values() for f in v[1:]} | {TEMPLATES_FILE}))


DEFAULT_WEIGHTS_REPO = "Ugo96/dibima-cardio-v5-weights"


def weight_file(name):
    """weights/<name> if present (run `python scripts/download_weights.py` once); otherwise download it from
    the Hugging Face model repo $DIBIMA_WEIGHTS_REPO (default Ugo96/dibima-cardio-v5-weights) with $HF_TOKEN."""
    local = HERE / "weights" / name
    if local.is_file():
        return local
    repo = os.environ.get("DIBIMA_WEIGHTS_REPO", DEFAULT_WEIGHTS_REPO)
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo, filename=name, repo_type="model",
                                token=os.environ.get("HF_TOKEN") or None))


def weights_cache_dir():
    try:
        from huggingface_hub import constants
        return constants.HF_HUB_CACHE
    except Exception:
        return os.path.expanduser("~/.cache/huggingface")


def import_mamba_safely():
    """Import mamba_ssm; if an optional compiled/Mamba3 part fails, stub it (the v5 model only
    uses Mamba2, whose kernels are Triton; Mamba1's selective_scan_cuda is never called)."""
    import types
    try:
        import mamba_ssm  # noqa: F401
        return "mamba_ssm " + mamba_ssm.__version__
    except Exception as first:
        for k in [k for k in sys.modules if k == "mamba_ssm" or k.startswith("mamba_ssm.")]:
            del sys.modules[k]
        if "selective_scan_cuda" not in sys.modules:
            sys.modules["selective_scan_cuda"] = types.ModuleType("selective_scan_cuda")
        m3 = types.ModuleType("mamba_ssm.modules.mamba3")
        m3.Mamba3 = None
        sys.modules["mamba_ssm.modules.mamba3"] = m3
        import mamba_ssm  # noqa: F401
        return f"mamba_ssm {mamba_ssm.__version__} (stubbed optional parts after: {type(first).__name__}: {first})"


def mamba2_path():
    """'fused' (causal_conv1d CUDA kernel + split-conv1d-scan) or 'triton' (nn.Conv1d + Triton SSD).
    DIBIMA_MAMBA_FUSED=1/0 forces it; default: fused iff causal_conv1d imports."""
    flag = os.environ.get("DIBIMA_MAMBA_FUSED", "auto").lower()
    if flag in ("1", "true", "fused"):
        return "fused"
    if flag in ("0", "false", "triton"):
        return "triton"
    try:
        import causal_conv1d  # noqa: F401
        from causal_conv1d.cpp_functions import causal_conv1d_fwd_function  # noqa: F401
        return "fused"
    except Exception:
        return "triton"


def backbone_kwargs(cfg):
    """BiMa_nn kwargs exactly as used in training (absent keys = original v5 architecture)."""
    v3 = cfg["mamba_version"] == 3
    return dict(
        target_channels=3, num_channels=3, seconds=4, residual_internal=cfg["internal_residual"],
        use_mamba=cfg["use_mamba"], use_diffusion=cfg["use_diffusion"], n_mamba_layers=cfg["n_mamba_layers"],
        mamba_dim=cfg["mamba_dim"], mamba_d_state=cfg["mamba_d_state"], mamba_version=cfg["mamba_version"],
        n_mamba_blocks=cfg["n_mamba_blocks"], use_positional_encoding=False, merge_type=cfg["merge_type"],
        do=cfg["do"], cross=cfg["cross"], norm_layers=cfg["norm_layers"],
        headdim=cfg["headdim"] if v3 else None, is_mimo=cfg["is_mimo"] if v3 else None,
        mimo_rank=cfg["mimo_rank"] if v3 else None, chunk_size=cfg["chunk_size"] if v3 else None,
        is_outproj_norm=cfg["is_outproj_norm"] if v3 else None, dtype=cfg["dtype"] if v3 else None,
        mask_cond=True, use_mamba_decoder=cfg["use_mamba_decoder"],
        prior_channels=cfg.get("prior_K", 1) if cfg.get("template_prior", False) else 0,
        attention=bool(cfg.get("attention", False)) and bool(cfg.get("attention_wired", False)),
        attention_wired=cfg.get("attention_wired", False), bottleneck=cfg.get("bottleneck", None),
        latent_concat=cfg.get("latent_concat", "sample"), cross_mask=cfg.get("cross_mask", "none"),
        bottleneck_blocks=cfg.get("bottleneck_blocks"), bottleneck_layers=cfg.get("bottleneck_layers"),
        bottleneck_expand=cfg.get("bottleneck_expand"), bottleneck_attention=cfg.get("bottleneck_attention"),
        early_fusion=cfg.get("early_fusion", False),
        decoder_fusion=cfg.get("decoder_fusion", False), decoder_fusion_kernel=cfg.get("decoder_fusion_kernel", 9),
        decoder_fusion_hidden=cfg.get("decoder_fusion_hidden", 64), cross_aware=cfg.get("cross_aware", False),
        corruption_aware=cfg.get("corruption_aware", False), decoder_local_attn=cfg.get("decoder_local_attn", False),
        decoder_local_window=cfg.get("decoder_local_window", 62), decoder_local_heads=cfg.get("decoder_local_heads", 4),
        decoder_local_dim=cfg.get("decoder_local_dim", 32), residual_filter=cfg.get("residual_filter", False),
        residual_filter_kernel=cfg.get("residual_filter_kernel", 15),
        residual_filter_hidden=cfg.get("residual_filter_hidden", 16))


def build_dibima(device="cuda", variant="v5"):
    """DiBiMa-Cardio (variant 'v5', 3.87 M params) or DiBiMa-Cardio-M (variant 'm', multimodal fine-tune) with the
    template prior, strict state_dict load."""
    import torch
    import yaml
    from diffusers import DDPMScheduler
    mamba_info = import_mamba_safely()
    from mamba_ssm import Mamba2
    from src.models import BiMa_nn, DiBiMa
    from src.template_prior import TemplatePrior
    display, wfile, cfile = DIBIMA_VARIANTS[variant]
    cfg = yaml.safe_load(open(weight_file(cfile)))
    if cfg.get("corruption_aware", False):
        raise NotImplementedError("corruption-aware DiBiMa-Cardio models are not served by this Space.")
    backbone = BiMa_nn(**backbone_kwargs(cfg))
    sched = DDPMScheduler(num_train_timesteps=20, beta_start=1e-5, beta_end=2e-2, beta_schedule="scaled_linear",
                          prediction_type="sample", clip_sample=False, clip_sample_range=None)
    model = DiBiMa(backbone, train_scheduler=sched, total_steps=None)
    sd = torch.load(weight_file(wfile), map_location="cpu", weights_only=True)
    if all(k.startswith("module.") for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.K = int(getattr(backbone, "prior_channels", 0) or 0)
    model.prior = TemplatePrior.load(str(weight_file(TEMPLATES_FILE))) if model.K else None
    model.n_params = sum(p.numel() for p in model.parameters())
    model.variant, model.display_name = variant, display
    model.mamba_path = mamba2_path()
    if model.mamba_path == "triton":
        # Non-fused Mamba2 still calls causal_conv1d_fn when that package imports; disable it so the
        # conv runs as nn.Conv1d (identical math; the Space does not install causal_conv1d).
        import mamba_ssm.modules.mamba2 as _m2
        _m2.causal_conv1d_fn, _m2.causal_conv1d_update = None, None
    n_m2 = 0
    for mod in model.modules():
        if isinstance(mod, Mamba2):
            mod.use_mem_eff_path = model.mamba_path == "fused"
            n_m2 += 1
    model.mamba_info = f"{mamba_info}; {n_m2} Mamba2 blocks, {model.mamba_path} path"
    return model.to(device).eval()


def build_unicardio(device="cuda"):
    import torch
    import yaml
    from unicardio.diffusion_model_no_compress_final import CSDI_base
    import unicardio_ddim
    cfg = yaml.safe_load(open(HERE / "unicardio" / "base_no_compress_original.yaml"))
    model = CSDI_base(cfg, torch.device("cpu"), L=4 * L)
    sd = torch.load(HERE / "weights" / "unicardio_no_compress799.pth", map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and not all(hasattr(v, "shape") for v in sd.values()):
        sd = sd.get("state_dict", sd.get("model_state_dict", sd))
    if all(k.startswith("module.") for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    unicardio_ddim.install(model)
    model.n_params = sum(p.numel() for p in model.parameters())
    model = model.to(device).eval()
    _set_device_attrs(model, torch.device(device))
    return model


def dibima_prior_block(model, signals, masks, task):
    N = signals["ppg"].shape[0]
    K = model.K
    blk = np.zeros((N, 3 * K, L), np.float32)
    if K and task.kind in ("imputation", "forecast"):
        i = MODALITIES.index(task.target)
        for b in range(N):
            blk[b, i * K:(i + 1) * K] = model.prior(signals[task.target][b], masks[task.target][b],
                                                   task.target, augment=False, K=K)
    return blk


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def dibima_forward(model, x_np, flags, target, device):
    """x_np: (B, 6+3K, L). Returns (B,L) prediction of `target`."""
    import torch
    x = torch.from_numpy(x_np).to(device)
    t = torch.full((x.shape[0],), int(model.T) - 1, dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(x, t=t, model_flags=flags)
    return out[MODALITIES.index(target)].float().reshape(x.shape[0], -1)


def _pad(a):
    return np.concatenate([a, np.repeat(a[-1:], BATCH - len(a), 0)], 0) if len(a) < BATCH else a


def dibima_x(model, bundle, task, idx, corr=None):
    signals, masks = dibima_inputs(bundle, task, idx, corr)
    prior = dibima_prior_block(model, signals, masks, task)
    return np.concatenate([np.stack([signals[m] for m in MODALITIES], 1),
                           np.stack([masks[m] for m in MODALITIES], 1), prior], 1).astype(np.float32)


_DRAW_GROUP = [6]   # draws per forward pass; reduced automatically after a CUDA OOM


def dibima_draws(model, x_all, task, device="cuda", n=6, seed=INFERENCE_SEED):
    """Uncertainty draws: x0 predictions from n
    independently diffused inputs at t = T-1 (training forward process); only the three signal
    channels are diffused (eps from torch.Generator seeded seed+i per batch of 8), masks and prior
    unchanged. Returns (N, n, L)."""
    import torch
    flags = train_consistent_flags(task)
    out = []
    for s in range(0, len(x_all), BATCH):
        chunk = x_all[s:s + BATCH]
        xb = torch.from_numpy(_pad(chunk))
        t_cpu = torch.full((xb.shape[0],), int(model.T) - 1, dtype=torch.long)
        xs = []
        for i in range(n):
            g = torch.Generator().manual_seed(seed + i)
            eps = torch.randn(xb[:, :3].shape, generator=g)
            x_t, _ = model.q_sample(xb[:, :3].float(), t_cpu, noise=eps)
            xs.append(torch.cat([x_t.cpu(), xb[:, 3:]], dim=1))
        # several draws of the 8-window batch per pass (batch g*8, even as in the evaluator); g shrinks on OOM
        ys, i = [], 0
        while i < n:
            g = min(_DRAW_GROUP[0], n - i)
            try:
                y = dibima_forward(model, torch.cat(xs[i:i + g]).numpy().astype(np.float32), flags, task.target, device)
            except torch.cuda.OutOfMemoryError:
                if g == 1:
                    raise
                torch.cuda.empty_cache()
                _DRAW_GROUP[0] = max(1, g // 3)
                continue
            ys.append(y.cpu().numpy().reshape(g, BATCH, -1)[:, :len(chunk)])
            i += g
        out.append(np.transpose(np.concatenate(ys), (1, 0, 2)))
    return np.concatenate(out).astype(np.float32)


MULTI_SOURCE_MODES = ("ensemble", "fused")


def dibima_passes(task, multi_source="ensemble"):
    """[(Task used to build the input, route flags)] for one DiBiMa-Cardio prediction.

    Restoration (with or without clean auxiliaries) and single-source translation: one pass with the
    training routes. Two-source translation: 'ensemble' (default) = two passes, each EXACTLY the trained
    single-source contract (only that source present, routes [s_0,s_1,s_2]), outputs averaged;
    'fused' = one pass with both sources and routes [a_t, b_t] (z_t <- (z_t + z_a + z_b)/3), the
    v5 manuscript evaluation contract, which training never used."""
    if task.kind == "translation" and len(task.sources) > 1 and multi_source == "ensemble":
        return [(Task("translation", task.target, (s,)), train_consistent_flags(Task("translation", task.target, (s,))))
                for s in task.sources]
    return [(task, train_consistent_flags(task))]


def run_dibima(model, bundle, task, idx, device="cuda", latency_reps=5, n_draws=6, multi_source="ensemble", corr=None):
    """Predict windows idx in padded batches of 8. Returns predictions (N,L) native z units, timings
    and (if n_draws) the uncertainty draws (N, n_draws, L), computed after the timed pass.
    Two-source translation uses `multi_source` (see dibima_passes)."""
    passes = dibima_passes(task, multi_source)
    if len(passes) > 1:
        outs = [run_dibima(model, bundle, t_, idx, device, latency_reps, n_draws, multi_source="fused", corr=corr)
                for t_, _ in passes]
        pred = np.mean([o[0] for o in outs], 0).astype(np.float32)
        tms = [o[1] for o in outs]
        timing = dict(tms[0], total_s=sum(t["total_s"] for t in tms), latency_b1_ms=sum(t["latency_b1_ms"] for t in tms),
                      warmup_s=sum(t["warmup_s"] for t in tms), forward_passes_per_window=len(passes))
        timing["ms_per_window"] = 1000 * timing["total_s"] / len(list(idx))
        if not n_draws:
            return pred, timing
        timing["uncertainty_s"] = sum(t["uncertainty_s"] for t in tms)
        return pred, timing, np.mean([o[2] for o in outs], 0).astype(np.float32)   # paired draws averaged
    idx = list(idx)
    x_all = dibima_x(model, bundle, task, idx, corr)
    flags = passes[0][1]
    pad = _pad
    # warm-up (kernel selection / Triton JIT), excluded from timing
    t0 = time.perf_counter()
    dibima_forward(model, pad(x_all[:BATCH]), flags, task.target, device); _sync()
    warmup = time.perf_counter() - t0
    # single-window latency (B = 1, streaming use); output not used for scoring (batch-parity effect)
    lat = []
    for _ in range(latency_reps):
        _sync(); t0 = time.perf_counter()
        dibima_forward(model, x_all[:1], flags, task.target, device); _sync()
        lat.append(time.perf_counter() - t0)
    preds = []
    _sync(); t0 = time.perf_counter()
    for s in range(0, len(idx), BATCH):
        chunk = x_all[s:s + BATCH]
        p = dibima_forward(model, pad(chunk), flags, task.target, device)[:len(chunk)]
        preds.append(p.cpu().numpy())
    _sync()
    total = time.perf_counter() - t0
    timing = dict(model=f"{getattr(model, 'display_name', 'DiBiMa-Cardio')} 1-step", n_windows=len(idx), batch_size=BATCH, total_s=total,
                  ms_per_window=1000 * total / len(idx), latency_b1_ms=1000 * float(np.median(lat)),
                  warmup_s=warmup, forward_passes_per_window=1)
    pred = np.concatenate(preds).astype(np.float32)
    if not n_draws:
        return pred, timing
    t0 = time.perf_counter()
    draws = dibima_draws(model, x_all, task, device, n=n_draws); _sync()
    timing["uncertainty_s"] = time.perf_counter() - t0
    return pred, timing, draws


def unicardio_forward(model, x_np, entry, device, n_samples=6, sample_steps=6):
    import torch
    x = torch.from_numpy(x_np).to(device)
    with torch.inference_mode():
        out = model(x, n_samples=n_samples, model_flag=entry["model_flag"], borrow_mode=entry["borrow_mode"],
                    DDIM_flag=int(entry.get("DDIM_flag", 1)), sample_steps=int(entry.get("sample_steps", sample_steps)),
                    train_gen_flag=1, improved=0)
    if isinstance(out, tuple):
        out = out[int(entry.get("tuple_index", 0))]
    out = torch.as_tensor(out).detach().float()
    if out.ndim == 4:      # B,S,1,L
        out = out[:, :, 0]
    if out.ndim == 2:
        out = out[:, None]
    if out.shape[-1] == 4 * L:
        seg = entry.get("output_segment")
        out = out[..., seg * L:(seg + 1) * L]
    return out.cpu().numpy()   # (B, S, L) draws; the scored prediction is their mean


def get_rng_state():
    import torch
    cuda = [s.numpy() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None
    return {"cpu": torch.get_rng_state().numpy(), "cuda": cuda}


def set_rng_state(state):
    import torch
    torch.set_rng_state(torch.from_numpy(np.asarray(state["cpu"], np.uint8)))
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.from_numpy(np.asarray(s, np.uint8)) for s in state["cuda"]])


def run_unicardio(model, bundle, task, idx, entry, device="cuda", seed=None, measure_latency=True,
                  n_samples=6, sample_steps=6, rng_state=None, return_rng=False, corr=None):
    """UniCardio DDIM-6 (mean of 6 draws) for windows idx in batches of 8.

    The RNG starts from the evaluator's per-task seed (seed_all(digest([42, 'unicardio', key]))),
    or from `rng_state` returned by a previous chunk, so chunked runs reproduce one uninterrupted
    pass over the same windows."""
    idx = list(idx)
    x_all, y = unicardio_inputs(bundle, task, idx, entry, corr)
    if rng_state is not None:
        set_rng_state(rng_state)
    else:
        import torch
        seed_all(int(digest([INFERENCE_SEED, "unicardio", task.key])[:8], 16) if seed is None else seed)
        # The evaluator iterates a torch DataLoader after seeding; creating the iterator draws one int64
        # from the global CPU generator (worker base seed). Consume it too, so the draws match the evaluator.
        torch.empty((), dtype=torch.int64).random_()
    warmup, lat = 0.0, float("nan")
    rng_state = None
    if measure_latency:
        import torch
        rng_state = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        t0 = time.perf_counter()
        unicardio_forward(model, x_all[:1], entry, device, n_samples=1, sample_steps=sample_steps); _sync()
        warmup = time.perf_counter() - t0
        _sync(); t0 = time.perf_counter()
        unicardio_forward(model, x_all[:1], entry, device, n_samples, sample_steps); _sync()
        lat = time.perf_counter() - t0
        torch.set_rng_state(rng_state[0])            # timing passes do not change the scored draws
        if rng_state[1] is not None:
            torch.cuda.set_rng_state_all(rng_state[1])
    draws = []
    _sync(); t0 = time.perf_counter()
    for s in range(0, len(idx), BATCH):
        draws.append(unicardio_forward(model, x_all[s:s + BATCH], entry, device, n_samples, sample_steps))
    _sync()
    total = time.perf_counter() - t0
    steps = int(entry.get("sample_steps", sample_steps))
    timing = dict(model="UniCardio-M DDIM-6", n_windows=len(idx), batch_size=BATCH, total_s=total,
                  ms_per_window=1000 * total / len(idx), latency_b1_ms=1000 * lat, warmup_s=warmup,
                  forward_passes_per_window=n_samples * steps)
    draws = np.concatenate(draws).astype(np.float32)          # (N, S, L)
    out = (draws.mean(1), y, timing, draws)
    return (*out, get_rng_state()) if return_rng else out


# ----------------------------------------------------------------------------- metrics
def metric_values(p, y, select=None):
    from scipy.stats import ks_2samp
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    select = np.ones(y.shape, bool) if select is None else np.asarray(select, bool)
    res = dict.fromkeys(METRICS, float("nan"))
    n = int(select.sum())
    if not n:
        return res, 0
    ps, ys = p[select], y[select]
    err = ps - ys
    mse = float(np.mean(err ** 2))
    res.update(MAE=float(np.mean(np.abs(err))), MSE=mse, RMSE=math.sqrt(mse),
               KS_stat=float(ks_2samp(ps, ys, method="auto").statistic))
    if n > 1 and np.ptp(ps) > 0 and np.ptp(ys) > 0:
        res["PCC"] = float(np.clip(np.corrcoef(ps, ys)[0, 1], -1, 1))
    e, ee = float(np.sum(ys ** 2)), float(np.sum(err ** 2))
    res["SNR"] = (float("inf") if e > 0 else float("nan")) if ee == 0 else (
        10 * math.log10(e / ee) if e > 0 else float("-inf"))
    dr = float(np.ptp(y))
    if dr > 0:
        res["PSNR"] = float("inf") if mse == 0 else 20 * math.log10(dr / math.sqrt(mse))
    return res, n


def ref_zscore(p, y):
    sd = float(np.std(y))
    if not np.isfinite(sd) or sd == 0:
        return None, None
    mu = float(np.mean(y))
    return (np.asarray(p, np.float64) - mu) / sd, (np.asarray(y, np.float64) - mu) / sd


def minmax(x):
    x = np.asarray(x, np.float64)
    w = np.ptp(x)
    return None if w == 0 else 2 * (x - x.min()) / w - 1


def band(draws, ref, mode="ref_zscore", center=None):
    """Draws (S,L) and reference (L,) -> dict of band arrays. One affine map is applied to every draw,
    so the band keeps the draws' real spread (per-draw renormalisation would add spurious spread).
    mode 'ref_zscore': map from the reference window's mean/SD (reference uses the same map).
    mode 'minmax': map that sends `center` (the plotted prediction; default the draw mean) to [-1, 1],
    the reference is min-maxed independently (= the legacy_shape scale, as plot_predictions.draw_stats)."""
    d = np.asarray(draws, np.float64)
    if mode == "minmax":
        c = d.mean(0) if center is None else np.asarray(center, np.float64)
        lo_c, w = float(c.min()), float(np.ptp(c)) or 1.0
        z = 2 * (d - lo_c) / w - 1
        y = minmax(ref)
        y = np.zeros_like(c) if y is None else y
    else:
        mu, sd = float(np.mean(ref)), float(np.std(ref)) or 1.0
        z = (d - mu) / sd
        y = (np.asarray(ref, np.float64) - mu) / sd
    lo, hi = np.percentile(z, [2.5, 97.5], axis=0)
    return dict(mean=z.mean(0), sd=z.std(0), lo=lo, hi=hi, y=y)


def uncertainty_rows(draws, target, regions, model_name, sample_ids):
    """Per-window uncertainty summaries (ref_zscore units): mean predictive SD across draws, mean
    width of the 2.5-97.5 % draw interval, and the fraction of reference samples inside it."""
    rows = []
    for i, (d, y) in enumerate(zip(draws, target)):
        b = band(d, y)
        for region, sel in regions.items():
            s = np.ones(len(y), bool) if sel is None else np.asarray(sel[i], bool)
            if not s.any():
                continue
            inside = (b["y"][s] >= b["lo"][s]) & (b["y"][s] <= b["hi"][s])
            rows.append(dict(model=model_name, region=region, sample_id=sample_ids[i], n_draws=d.shape[0],
                             pred_SD=float(b["sd"][s].mean()), PI95_width=float((b["hi"] - b["lo"])[s].mean()),
                             PI95_coverage=float(inside.mean())))
    return rows


def score(pred, target, regions, model_name, task, sample_ids):
    rows = []
    for i, (p, y) in enumerate(zip(pred, target)):
        for scale in SCALES:
            if scale == "native":
                pp, yy = p, y
            elif scale == "ref_zscore":
                pp, yy = ref_zscore(p, y)
            else:
                pp, yy = minmax(p), minmax(y)
            for region, sel in regions.items():
                s = None if sel is None else sel[i]
                if pp is None or yy is None:
                    vals, n = dict.fromkeys(METRICS, float("nan")), 0
                else:
                    vals, n = metric_values(pp, yy, s)
                rows.append(dict(model=model_name, task=task.name, variant=task.variant, scale=scale,
                                 region=region, sample_id=sample_ids[i], n_points=n, **vals))
    return rows
