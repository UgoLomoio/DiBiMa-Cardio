"""
Augmented template prior for imputation and forecasting (PulseDiff-style).

Reference: Jenkins, Chen, Ng, Mandic, "Improving Diffusion Models for ECG
Imputation with an Augmented Template Prior", arXiv:2310.15742v2, Sec. III-B.

The paper's procedure, adapted to DiBiMa-Cardio windows (4 s, 125 Hz, per-window
z-score) and applied to every modality (PPG, ECG, BP):

1. External (global) template p_ext per modality: median beat from N_ext fully
   observed training windows. Beats are located at a fiducial peak (ECG: R peak,
   Christov detector when neurokit2 is available; PPG/BP: systolic peak) and the
   segment starts `pre_s` seconds before it (paper: 250 ms before the QRS, to
   include the P wave).
2. Per window, the observed values x_ob are linearly interpolated (x_ob-l).
   Beats are detected with a matched filter (normalised cross-correlation with
   p_ext, 400 ms refractory period), an intermediate subject template p'_s is
   built as the median of the detected segments, and detection is repeated with
   p'_s to give beat locations b_s and the final subject template p_s.
   Only beats whose segment is mostly observed are used.
3. Rhythm confidence R_s = std(IBI) / median(IBI). If R_s < r (paper: r = 1),
   missing beat locations are filled at the median IBI, inside gaps and in the
   forecast horizon.
4. Prior p(t) = sum_i [a_i + p_s(t - b_i - s_i)], zeros between beats, with
   s_i ~ U(-M, M) samples and a_i ~ U(-A, A) during training (augmented prior),
   and s_i = a_i = 0 for the fixed prior. K priors can be stacked.

Deviations from the paper (mostly because windows are 4 s instead of 10 s):
- ECG detection is polarity-agnostic (matched filter with +p_ext and -p_ext),
  because part of the training ECG remains inverted after rectification; the
  subject template is built from the window itself, so it keeps that polarity;
- the matched-filter threshold is an absolute normalised correlation
  (`corr_threshold`) combined with a percentile, instead of the 97th percentile
  alone, which would keep only ~15 of 500 samples;
- IBIs that span a missing beat (> 1.5 x median IBI) are excluded from R_s.
If fewer than `min_beats` beats are found, the prior is all zeros (no prior).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.signal import find_peaks

MODALITIES = ("ppg", "ecg", "bp")


# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------
def interpolate_observed(x: np.ndarray, m: np.ndarray) -> np.ndarray | None:
    """Linear interpolation of missing samples (m == 0); edges hold the nearest value."""
    obs = m > 0.5
    if obs.all():
        return x.astype(np.float64, copy=True)
    if obs.sum() < 2:
        return None
    idx = np.arange(x.shape[-1])
    return np.interp(idx, idx[obs], x[obs].astype(np.float64))


def normalized_xcorr(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Pearson correlation between h and every len(h) segment of x ('valid' mode).

    Output index k corresponds to the segment x[k : k + len(h)], i.e. the template
    start. Length is len(x) - len(h) + 1.
    """
    n = h.shape[0]
    if x.shape[0] < n:
        return np.zeros(0)
    h0 = h - h.mean()
    h0 = h0 / (np.linalg.norm(h0) + 1e-8)
    num = np.correlate(x, h0, mode="valid")
    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
    s = cs[n:] - cs[:-n]
    s2 = cs2[n:] - cs2[:-n]
    var = np.maximum(s2 - s * s / n, 1e-8)
    return num / np.sqrt(var)


def _median_segment(x: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray | None:
    segs = [x[b:b + length] for b in starts if 0 <= b and b + length <= x.shape[0]]
    if not segs:
        return None
    return np.median(np.stack(segs), axis=0)


def _fiducial_peaks(x: np.ndarray, modality: str, fs: int) -> np.ndarray:
    """Fiducial beat locations on a fully observed z-scored window."""
    if modality == "ecg":
        try:
            import neurokit2 as nk  # Christov detector, as in the paper
            _, info = nk.ecg_peaks(x, sampling_rate=fs, method="christov")
            pk = np.asarray(info["ECG_R_Peaks"], dtype=int)
            # Christov marks the QRS onset region; snap to the local maximum (R peak)
            w = int(0.06 * fs)
            pk = np.array([max(0, p - w) + int(np.argmax(x[max(0, p - w):p + w + 1])) for p in pk], dtype=int)
            if len(pk) >= 2:
                return np.unique(pk)
        except Exception:
            pass
        pk, _ = find_peaks(x, distance=int(0.4 * fs), prominence=1.0)
        return pk
    # PPG / BP: systolic peaks
    pk, _ = find_peaks(x, distance=int(0.4 * fs), prominence=0.5)
    return pk


# --------------------------------------------------------------------------
# Config + main class
# --------------------------------------------------------------------------
@dataclass
class TemplatePriorConfig:
    fs: int = 125
    n_ext: int = 256                  # windows used for the external template (paper: 10 x 10-s records)
    pre_s: dict = field(default_factory=lambda: {"ppg": 0.20, "ecg": 0.25, "bp": 0.20})
    min_dist_s: float = 0.4           # 400 ms refractory period (max 150 bpm)
    corr_threshold: float = 0.5       # absolute normalised-correlation threshold
    corr_percentile: float = 90.0     # and percentile threshold; the larger of the two is used
    min_observed_frac: float = 0.9    # a beat segment must be >= 90% observed to enter the template
    min_beats: int = 2                # fewer beats -> zero prior
    rhythm_r: float = 1.0             # paper: r = 1
    max_shift: float = 3.0            # M, beat-location shift in samples (U(-M, M))
    max_amp: float = 0.1              # A, additive amplitude shift in z units (U(-A, A))
    K: int = 1                        # number of augmented priors per modality
    ibi_bounds_s: tuple = (0.3, 1.6)  # clamp for the subject template length
    polarity_agnostic: tuple = ("ecg",)  # also match -p_ext (inverted ECG); the subject template keeps the window's polarity

    def to_dict(self):
        d = asdict(self)
        d["ibi_bounds_s"] = list(self.ibi_bounds_s)
        d["polarity_agnostic"] = list(self.polarity_agnostic)
        return d


class TemplatePrior:
    """Builds PulseDiff-style template priors for PPG, ECG and BP windows."""

    def __init__(self, external_templates: dict | None = None, config: TemplatePriorConfig | None = None):
        self.cfg = config or TemplatePriorConfig()
        self.ext = external_templates or {}

    # ---------------- external templates ----------------
    @classmethod
    def from_windows(cls, signals: dict, config: TemplatePriorConfig | None = None, seed: int = 42):
        """signals: {"ppg": (N, L) or (N, 1, L) array/tensor, ...}; fully observed windows."""
        cfg = config or TemplatePriorConfig()
        rng = np.random.default_rng(seed)
        ext = {}
        for mod in MODALITIES:
            X = signals[mod]
            X = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
            X = X.reshape(X.shape[0], -1).astype(np.float64)
            idx = rng.choice(X.shape[0], size=min(cfg.n_ext, X.shape[0]), replace=False)
            pre = int(round(cfg.pre_s[mod] * cfg.fs))
            ibis, peaks = [], []
            for i in idx:
                x = X[i]
                if not np.all(np.isfinite(x)) or x.std() < 1e-6:
                    continue
                pk = _fiducial_peaks(x, mod, cfg.fs)
                if len(pk) >= 2:
                    ibis.extend(np.diff(pk).tolist())
                    peaks.append((i, pk))
            if not ibis:
                raise RuntimeError(f"No beats found for the external {mod} template")
            lo, hi = (int(b * cfg.fs) for b in cfg.ibi_bounds_s)
            length = int(np.clip(np.median(ibis), lo, hi))
            segs = []
            for i, pk in peaks:
                for p in pk:
                    s = p - pre
                    if s >= 0 and s + length <= X.shape[1]:
                        segs.append(X[i, s:s + length])
            ext[mod] = np.median(np.stack(segs), axis=0).astype(np.float32)
        return cls(ext, cfg)

    def save(self, path: str):
        np.savez(path, **{f"ext_{k}": v for k, v in self.ext.items()},
                 config=np.array(repr(self.cfg.to_dict())))

    @classmethod
    def load(cls, path: str):
        import ast
        d = np.load(path, allow_pickle=False)
        cfg_d = ast.literal_eval(str(d["config"]))
        cfg_d["ibi_bounds_s"] = tuple(cfg_d["ibi_bounds_s"])
        cfg_d["polarity_agnostic"] = tuple(cfg_d.get("polarity_agnostic", ("ecg",)))
        ext = {k[4:]: d[k] for k in d.files if k.startswith("ext_")}
        return cls(ext, TemplatePriorConfig(**cfg_d))

    # ---------------- per-window detection ----------------
    def _detect(self, xl: np.ndarray, m: np.ndarray, h: np.ndarray) -> np.ndarray:
        c = normalized_xcorr(xl, h)
        if c.size == 0:
            return np.zeros(0, dtype=int)
        thr = max(self.cfg.corr_threshold, float(np.percentile(c, self.cfg.corr_percentile)))
        pk, _ = find_peaks(c, height=thr, distance=int(self.cfg.min_dist_s * self.cfg.fs))
        n = h.shape[0]
        # keep beats whose segment is (mostly) observed
        csm = np.concatenate([[0.0], np.cumsum(m)])
        frac = (csm[pk + n] - csm[pk]) / n
        return pk[frac >= self.cfg.min_observed_frac].astype(int)

    def subject_template(self, x: np.ndarray, m: np.ndarray, modality: str):
        """Return (template p_s, beat starts b_s, interpolated x_ob-l) or (None, [], xl)."""
        xl = interpolate_observed(x, m)
        if xl is None or modality not in self.ext:
            return None, np.zeros(0, dtype=int), xl
        h = self.ext[modality].astype(np.float64)
        b1 = self._detect(xl, m, h)
        if modality in self.cfg.polarity_agnostic:
            b1n = self._detect(xl, m, -h)
            if len(b1n) > len(b1):
                b1 = b1n
        if len(b1) < self.cfg.min_beats:
            return None, b1, xl
        # subject template length from the observed rhythm (bounded)
        lo, hi = (int(b * self.cfg.fs) for b in self.cfg.ibi_bounds_s)
        ibi = np.diff(b1)
        ibi = ibi[ibi <= 1.5 * np.median(ibi)] if len(ibi) else ibi
        length = int(np.clip(np.median(ibi), lo, hi)) if len(ibi) else h.shape[0]
        # intermediate template p'_s, aligned to the matched-filter starts
        p1 = _median_segment(xl, b1, length)
        if p1 is None:
            return None, b1, xl
        # second pass with the subject template
        b2 = self._detect(xl, m, p1)
        bs = b2 if len(b2) >= self.cfg.min_beats else b1
        ps = _median_segment(xl, bs, p1.shape[0])
        if ps is None:
            ps = p1
        return ps, bs, xl

    def rhythm(self, bs: np.ndarray):
        """Return (R_s, median IBI) from detected beats (IBIs spanning a missing beat excluded)."""
        if len(bs) < 2:
            return np.inf, None
        ibi = np.diff(bs).astype(float)
        med = np.median(ibi)
        ibi_ok = ibi[ibi <= 1.5 * med]
        if len(ibi_ok) == 0:
            return np.inf, None
        med = float(np.median(ibi_ok))
        return float(np.std(ibi_ok) / med), med

    def fill_beats(self, bs: np.ndarray, med: float, L: int, Lt: int) -> np.ndarray:
        """Impute missing beat locations at the median IBI (gaps, both window edges, forecast horizon)."""
        bs = np.sort(bs)
        out = list(bs)
        for a, b in zip(bs[:-1], bs[1:]):
            n_missing = int(round((b - a) / med)) - 1
            for k in range(1, n_missing + 1):
                out.append(int(round(a + k * (b - a) / (n_missing + 1))))
        t = bs[-1] + med
        while t < L:
            out.append(int(round(t)))
            t += med
        t = bs[0] - med
        while t > -Lt:
            out.append(int(round(t)))
            t -= med
        return np.array(sorted(set(out)), dtype=int)

    # ---------------- prior synthesis ----------------
    @staticmethod
    def place(template: np.ndarray, starts: np.ndarray, L: int, shifts=None, amps=None) -> np.ndarray:
        p = np.zeros(L, dtype=np.float64)
        n = template.shape[0]
        for i, b in enumerate(starts):
            s = b + (int(round(shifts[i])) if shifts is not None else 0)
            seg = template + (amps[i] if amps is not None else 0.0)
            lo, hi = max(0, s), min(L, s + n)
            if hi > lo:
                p[lo:hi] = seg[lo - s:hi - s]
        return p

    def __call__(self, x, m, modality: str, augment: bool = False, K: int | None = None, rng=None) -> np.ndarray:
        """Prior of shape (K, L) for one window. x, m: (L,) or (1, L) arrays/tensors."""
        K = K or self.cfg.K
        x = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
        x = x.reshape(-1).astype(np.float64)
        m = m.reshape(-1).astype(np.float64)
        L = x.shape[0]
        out = np.zeros((K, L), dtype=np.float32)
        ps, bs, _ = self.subject_template(x, m, modality)
        if ps is None or len(bs) < self.cfg.min_beats:
            return out
        R, med = self.rhythm(bs)
        starts = self.fill_beats(bs, med, L, ps.shape[0]) if (med is not None and R < self.cfg.rhythm_r) else np.sort(bs)
        if rng is None:
            rng = np.random.default_rng(random.getrandbits(63))
        for k in range(K):
            if augment or k > 0:   # row 0 is the fixed prior at evaluation
                sh = rng.uniform(-self.cfg.max_shift, self.cfg.max_shift, size=len(starts))
                am = rng.uniform(-self.cfg.max_amp, self.cfg.max_amp, size=len(starts))
                out[k] = self.place(ps, starts, L, sh, am)
            else:
                out[k] = self.place(ps, starts, L)
        return out
