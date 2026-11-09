"""DiBiMa-Cardio interactive demo (Gradio): real-time cardiovascular signal restoration and translation.

DiBiMa-Cardio and DiBiMa-Cardio-M run in one network pass per window on the 256-window test subset (4 s of
PPG / ECG / BP at 125 Hz) shipped in data/test256_inputs.npz, with the paper's seeded corruptions (gap masks,
noise residuals, low-resolution versions) and references. The default corruption is the paper's hard protocol
(2.0 s gap, 2.0 s forecast horizon, 2x noise); the standard benchmark corruption and custom values can be
selected. UniCardio-M (UniCardio, DDIM-6) runs alongside as the comparison model.

Run:  python app.py   ->  http://127.0.0.1:7860  (CUDA GPU required)
The same app runs on Hugging Face ZeroGPU when the `spaces` package is installed.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # also under PYTHONSAFEPATH=1

import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import spaces  # ZeroGPU
except ImportError:  # local run
    spaces = None

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E  # noqa: E402

# ----------------------------------------------------------------------------- setup
NAME = {"m": "DiBiMa-Cardio-M 1-step", "v5": "DiBiMa-Cardio 1-step", "uni": "UniCardio-M DDIM-6"}
SHORT = {"m": "DiBiMa-Cardio-M", "v5": "DiBiMa-Cardio", "uni": "UniCardio-M"}
COL = {NAME["m"]: "#009E73", NAME["v5"]: "#0072B2", NAME["uni"]: "#999999", "ref": "#222222",
       "ppg": "#56B4E9", "ecg": "#CC79A7", "bp": "#E69F00"}
RGBA = {NAME["m"]: "0,158,115", NAME["v5"]: "0,114,178", NAME["uni"]: "120,120,120"}
LABEL = {"ppg": "PPG", "ecg": "ECG", "bp": "BP"}
KIND_LABEL = {"imputation": "Imputation", "denoising": "Denoising", "forecast": "Forecasting",
              "sr": "Super-resolution", "translation": "Translation"}
DIBIMA_ORDER = ("m", "v5")       # DiBiMa-Cardio-M first: the recommended model
UNI_CHUNK = 32                   # UniCardio-M windows per GPU call (results stream after each chunk)
N_DRAWS = 6                      # uncertainty draws per model
N_WIN = 256
PRESETS = {"hard": E.HARD, "standard": E.STANDARD}


def gpu(duration):
    if spaces is None:
        return lambda f: f
    return spaces.GPU(duration=duration)


E.configure_torch()
DEVICE = "cuda" if (torch.cuda.is_available() or spaces is not None) else "cpu"
LOAD_ERROR = None
MODELS = {}
try:
    BUNDLE = E.Bundle()
    UNI_CFG = E.load_uni_tasks()
    TASKS = {t.key: t for t in E.task_roster()}
    for v in DIBIMA_ORDER:
        MODELS[v] = E.build_dibima(DEVICE, v)
    UNICARDIO = E.build_unicardio(DEVICE)
    MODEL_INFO = (f"DiBiMa-Cardio-M: {MODELS['m'].n_params / 1e6:.2f} M parameters; DiBiMa-Cardio: {MODELS['v5'].n_params / 1e6:.2f} M "
                  f"parameters ({MODELS['v5'].mamba_info}). UniCardio-M: {UNICARDIO.n_params / 1e6:.2f} M parameters.")
except Exception as exc:  # surfaced in the UI instead of crashing the Space
    LOAD_ERROR = "".join(traceback.format_exception(exc))
    MODEL_INFO = "Model loading failed."
    print(LOAD_ERROR)


def corruption(preset, gap_s, horizon_s, alpha):
    if preset in PRESETS:
        return PRESETS[preset]
    return E.Corruption(float(gap_s), float(horizon_s), float(alpha))


def _corr_tuple(c):
    return (c.gap_s, c.horizon_s, c.noise_alpha)


def default_multi(variant, multi_source):
    """Two-source translation: DiBiMa-Cardio-M was fine-tuned on the fused route; DiBiMa-Cardio (v5) is best as the mean of
    its two trained single-source passes."""
    if multi_source in E.MULTI_SOURCE_MODES:
        return multi_source
    return "fused" if variant == "m" else "ensemble"


# ----------------------------------------------------------------------------- GPU calls
def _dur_dibima(variant, task_key, idx, multi_source, corr):
    return int(20 + 0.1 * len(idx))


@gpu(_dur_dibima)
def gpu_dibima(variant, task_key, idx, multi_source, corr):
    pred, tm, draws = E.run_dibima(MODELS[variant], BUNDLE, TASKS[task_key], idx, DEVICE, n_draws=N_DRAWS,
                                   multi_source=multi_source, corr=E.Corruption(*corr))
    tm["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return pred, tm, draws


def _dur_uni(task_key, idx, rng_state, measure_latency, corr):
    return int(min(120, 25 + 1.0 * len(idx)))


@gpu(_dur_uni)
def gpu_unicardio(task_key, idx, rng_state, measure_latency, corr):
    task = TASKS[task_key]
    pred, y, tm, draws, rng = E.run_unicardio(UNICARDIO, BUNDLE, task, idx, E.uni_entry(UNI_CFG, task), DEVICE,
                                              measure_latency=measure_latency, rng_state=rng_state,
                                              return_rng=True, corr=E.Corruption(*corr))
    tm["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return pred, y, tm, draws, rng


# ----------------------------------------------------------------------------- task selection
def resolve(kind, target, inputs):
    """UI choices -> Task. Restoration always includes the corrupted output signal."""
    inputs = list(inputs or [])
    if kind != "translation" and target not in inputs:
        inputs.append(target)
    return E.make_task(kind, target, inputs)


def kind_desc(kind, corr):
    if kind == "imputation":
        g = "a 1.2 s gap (150 samples)" if corr.gap_s is None else f"a {corr.gap_s:g} s gap ({int(round(corr.gap_s * E.FS))} samples)"
        return f"{g} of the output signal is masked and must be filled in"
    if kind == "forecast":
        h = "the last ~1.17 s (1.0-1.3 s)" if corr.horizon_s is None else f"the last {corr.horizon_s:g} s"
        return f"{h} of the output signal is hidden and must be predicted"
    if kind == "denoising":
        a = float(corr.noise_alpha)
        snr = "clean input" if a == 0 else f"about {9.0 - 20 * np.log10(a):.0f} dB SNR"
        return (f"the output signal carries additive Gaussian noise, {a:g}x the standard test noise "
                f"(noise SD {0.355 * a:.2f} x signal SD, {snr})")
    if kind == "sr":
        return "the output signal is given at 4x lower resolution and must be reconstructed at 125 Hz"
    return "the output signal is absent and is generated from the other input signal(s)"


def task_info(kind, target, inputs, multi_source="auto", models=("m", "v5", "uni"), preset="hard", gap_s=2.0,
              horizon_s=2.0, alpha=2.0):
    models = list(models or [])
    try:
        task = resolve(kind, target, inputs)
        corr = corruption(preset, gap_s, horizon_s, alpha)
    except ValueError as e:
        return f"**Invalid selection:** {e}"
    lines = [f"**Configuration** `{task.key}`: {KIND_LABEL[kind]} of **{LABEL[target]}** from "
             f"{' + '.join(LABEL[m] + (' (corrupted)' if m == task.primary else '') for m in task.sources)}.",
             f"*{kind_desc(kind, corr)}.*"]
    if kind in ("imputation", "forecast", "denoising"):
        lines.append(f"Corruption: {corr.label}.")
    if task.n_inputs == 1:
        lines.append("Single-input setting.")
    else:
        lines.append("Multi-input setting: **DiBiMa-Cardio-M** (multimodal fine-tune) is the recommended model. "
                     "DiBiMa-Cardio (v5) was pretrained without a clean-auxiliary curriculum.")
    for v in [k for k in DIBIMA_ORDER if k in models]:
        passes = E.dibima_passes(task, default_multi(v, multi_source))
        if len(passes) == 1:
            lines.append(f"{SHORT[v]}: routes `{passes[0][1]}`, one forward pass at t = T-1.")
        else:
            lines.append(f"{SHORT[v]}: mean of the two single-source passes ("
                         + "; ".join(f"{LABEL[p.sources[0]]} only, routes `{f}`" for p, f in passes) + "), 2 forward passes.")
    if "uni" in models:
        ok, why = E.uni_supports(UNI_CFG, task)
        entry = E.uni_entry(UNI_CFG, task) or {}
        lines.append(f"UniCardio-M: flag `{entry.get('model_flag')}`, borrow mode {entry.get('borrow_mode')}, "
                     "DDIM 6 steps x 6 draws." if ok else f"UniCardio-M: not evaluable. {why}")
    return "\n\n".join(lines)


def sync_inputs(kind, target, inputs, *rest):
    inputs = list(inputs or [])
    if kind == "translation":
        inputs = [m for m in inputs if m != target] or [next(m for m in E.MODALITIES if m != target)]
    elif target not in inputs:
        inputs = [target] + inputs
    inputs = [m for m in E.MODALITIES if m in inputs]
    return gr.update(value=inputs), task_info(kind, target, inputs, *rest)


def preset_changed(preset, gap_s, horizon_s, alpha):
    custom = preset == "custom"
    vals = {"hard": (2.0, 2.0, 2.0), "standard": (1.2, 1.17, 1.0)}.get(preset, (gap_s, horizon_s, alpha))
    return [gr.update(value=v, interactive=custom) for v in vals]


def window_set(scope, w):
    w = int(w)
    if scope == "one":
        return [w]
    return sorted(set(range(int(scope))) | {w})


# ----------------------------------------------------------------------------- metrics tables
def _key(model):
    return next(k for k, v in NAME.items() if v == model)


def summarize(rows, scale, region, urows=None):
    if rows is None or not len(rows):
        return pd.DataFrame()
    df = rows[(rows.scale == scale) & (rows.region == region)]
    fmt = lambda v: (f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else
                     (f"{v.mean():.4f}" if len(v) else "n/a"))
    out = []
    for model, g in df.groupby("model", sort=False):
        rec = {"Model": model, "Windows": len(g)}
        for m in ("MAE", "RMSE", "PCC", "SNR", "PSNR", "KS_stat"):
            rec[m] = fmt(g[m].replace([np.inf, -np.inf], np.nan).dropna())
        if urows is not None and len(urows):
            u = urows[(urows.model == model) & (urows.region == region)]
            if len(u):
                rec["Pred. SD (z)"] = fmt(u.pred_SD)
                rec["95% interval width (z)"] = fmt(u.PI95_width)
                rec["95% interval coverage"] = fmt(u.PI95_coverage)
        out.append(rec)
    return pd.DataFrame(out)


def main_table(st, w, scale, region):
    """Compact live table under the signal plot: example window + mean over evaluated windows,
    accuracy, uncertainty and speed."""
    rows, urows, tms = (st or {}).get("rows"), (st or {}).get("urows"), (st or {}).get("timings", {})
    if rows is None or not len(rows):
        return pd.DataFrame()
    sid = BUNDLE.sample_id[int(w)]
    df = rows[(rows.scale == scale) & (rows.region == region)]
    out = []
    for model, g in df.groupby("model", sort=False):
        ex = g[g.sample_id == sid]
        f = lambda s: s.replace([np.inf, -np.inf], np.nan).dropna()
        rec = {"Model": model,
               f"Window {int(w)} PCC": round(float(ex.PCC.iloc[0]), 3) if len(ex) else None,
               f"Window {int(w)} MAE": round(float(ex.MAE.iloc[0]), 3) if len(ex) else None,
               "Windows": len(g),
               "PCC mean": round(float(f(g.PCC).mean()), 3),
               "MAE mean": round(float(f(g.MAE).mean()), 3),
               "RMSE mean": round(float(f(g.RMSE).mean()), 3),
               "SNR (dB)": round(float(f(g.SNR).mean()), 2)}
        if urows is not None and len(urows):
            u = urows[(urows.model == model) & (urows.region == region)]
            if len(u):
                rec["Pred. SD (z)"] = round(float(u.pred_SD.mean()), 3)
                rec["95% interval coverage"] = f"{100 * u.PI95_coverage.mean():.0f}%"
        k = _key(model)
        if k in tms:
            rec["ms / window (batch 8)"] = round(tms[k]["ms_per_window"], 1)
            rec["Latency B=1 (ms)"] = round(tms[k]["latency_b1_ms"], 1)
        out.append(rec)
    return pd.DataFrame(out)


def timing_table(timings):
    out = []
    for tm in timings.values():
        out.append({"Model": tm["model"], "GPU": tm.get("device", ""), "Windows": tm["n_windows"],
                    "Network passes / window": tm["forward_passes_per_window"],
                    "Batch-8 throughput (ms / window)": round(tm["ms_per_window"], 2),
                    "Single-window latency, B=1 (ms)": round(tm["latency_b1_ms"], 2),
                    "Real-time factor (4 s window / latency)": round(4000 / tm["latency_b1_ms"], 1)
                    if np.isfinite(tm["latency_b1_ms"]) and tm["latency_b1_ms"] > 0 else None,
                    "Total GPU compute (s)": round(tm["total_s"], 2),
                    "Uncertainty draws, extra (s)": (round(tm["uncertainty_s"], 2) if "uncertainty_s" in tm
                                                     else "included (the 6 draws are the prediction)"),
                    "Warm-up excluded (s)": round(tm["warmup_s"], 2)})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- figures
def _l(a, nd=5):
    """Plain JSON list (NaN -> None); avoids Plotly 6 base64 typed arrays."""
    a = np.asarray(a, dtype=float).round(nd)
    return [None if not np.isfinite(v) else float(v) for v in a]


def _segments(mask_missing):
    m = np.asarray(mask_missing, bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _empty(msg, height=300):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                       font=dict(size=14, color="#666"))
    fig.update_layout(template="plotly_white", height=height, xaxis_visible=False, yaxis_visible=False,
                      margin=dict(l=20, r=20, t=30, b=20))
    return fig


def _mm(x):
    """Independent min-max to [-1, 1] (legacy_shape scale), NaN-aware (gaps stay gaps)."""
    x = np.asarray(x, np.float64)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if not np.isfinite(hi - lo) or hi == lo else 2 * (x - lo) / (hi - lo) - 1


def fig_waveform(st, w):
    """All traces in min-max [-1, 1] units (legacy_shape): each reference, input and prediction is min-maxed
    independently; uncertainty bands use the affine map of the plotted prediction."""
    if not st or "task" not in st:
        return _empty("Choose a configuration and press Run.", 560)
    task = TASKS[st["task"]]
    corr = E.Corruption(*st["corr"])
    w = int(w)
    if w not in st["idx"]:
        return _empty(f"Window {w} was not evaluated in the last run (evaluated: {len(st['idx'])} windows). "
                      "Press Run again to include it.", 560)
    pos = st["idx"].index(w)
    t = np.arange(E.L) / E.FS
    keys = list(st["models"]) + (["uni"] if st.get("uni") else [])
    titles = ["Model input (each signal min-max to [-1, 1])"]
    titles += [f"{NAME[k]} vs its reference" + (" (UniCardio preprocessing)" if k == "uni" else "") for k in keys]
    titles += ["Overlay: predictions and reference"]
    nrow = len(titles)
    fig = make_subplots(rows=nrow, cols=1, shared_xaxes=True, vertical_spacing=0.05 if nrow <= 4 else 0.04,
                        subplot_titles=titles)
    sig, msk = E.dibima_inputs(BUNDLE, task, [w], corr)
    for m in task.sources:
        x = sig[m][0].astype(float).copy()
        if task.kind in ("imputation", "forecast") and m == task.primary:
            x[msk[m][0] == 0] = np.nan
        name = f"{LABEL[m]} input" + (" (corrupted)" if m == task.primary else "")
        fig.add_trace(go.Scatter(x=_l(t), y=_l(_mm(x)), name=name, line=dict(color=COL[m], width=1.4)), 1, 1)

    lab = LABEL[task.target]
    ref_drawn = set()
    for r, k in enumerate(keys, start=2):
        model = NAME[k]
        pred = st["pred"].get(k)
        if pred is None or pos >= len(pred):
            fig.add_annotation(text=st.get("msg", {}).get(k, f"{model} pending ..."), showarrow=False, x=2, y=0,
                               row=r, col=1, font=dict(color="#888"))
            continue
        y, p, dr = st["ref"][k][pos], pred[pos], st["draws"].get(k)
        if dr is not None:
            bnd = E.band(dr[pos], y, mode="minmax", center=p)
            for lo, hi, a, nm in ((bnd["lo"], bnd["hi"], 0.16, "95% draw interval"),
                                  (bnd["mean"] - bnd["sd"], bnd["mean"] + bnd["sd"], 0.34, "±1 SD")):
                fig.add_trace(go.Scatter(x=_l(np.concatenate([t, t[::-1]])), y=_l(np.concatenate([hi, lo[::-1]])),
                                         fill="toself", fillcolor=f"rgba({RGBA[model]},{a})", line=dict(width=0),
                                         hoverinfo="skip", name=f"{SHORT[k]} {nm} ({dr.shape[1]} draws)",
                                         legendgroup=f"{k}-{nm}"), r, 1)
        refname = "baseline" if k == "uni" else "dibima"
        fig.add_trace(go.Scatter(x=_l(t), y=_l(_mm(y)),
                                 name=f"{lab} reference" + (" (UniCardio preprocessing)" if k == "uni" else ""),
                                 showlegend=refname not in ref_drawn, legendgroup=f"ref-{refname}",
                                 line=dict(color=COL["ref"] if k != "uni" else "#666", width=1.6)), r, 1)
        fig.add_trace(go.Scatter(x=_l(t), y=_l(_mm(p)), name=model, legendgroup=k,
                                 line=dict(color=COL[model], width=1.6)), r, 1)
        if refname not in ref_drawn:
            fig.add_trace(go.Scatter(x=_l(t), y=_l(_mm(y)), showlegend=False, legendgroup=f"ref-{refname}",
                                     line=dict(color=COL["ref"] if k != "uni" else "#666", width=1.3,
                                               dash="solid" if k != "uni" else "dot")), nrow, 1)
            ref_drawn.add(refname)
        fig.add_trace(go.Scatter(x=_l(t), y=_l(_mm(p)), showlegend=False, legendgroup=k,
                                 line=dict(color=COL[model], width=1.4)), nrow, 1)
    region = {"imputation": "missing", "forecast": "future"}.get(task.kind, "full")
    if region != "full":
        miss = E.task_mask(BUNDLE, task.kind, task.target, [w], corr)[0] == 0
        for a, b in _segments(miss):
            for r in range(1, nrow + 1):
                fig.add_vrect(x0=a / E.FS, x1=b / E.FS, fillcolor="#999", opacity=0.13, line_width=0, row=r, col=1)
    rows = st.get("rows")
    if rows is not None and len(rows):
        for r, k in enumerate(keys, start=2):
            g = rows[(rows.model == NAME[k]) & (rows.sample_id == BUNDLE.sample_id[w]) & (rows.scale == "legacy_shape")]
            parts = []
            for rg in (("full", region) if region != "full" else ("full",)):
                q = g[g.region == rg]
                if len(q):
                    parts.append(f"{rg}: PCC {q.PCC.iloc[0]:.3f}, MAE {q.MAE.iloc[0]:.3f}")
            if parts:
                fig.layout.annotations[r - 1].text += "  |  " + "; ".join(parts)
    fig.update_xaxes(title_text="Time (s)", row=nrow, col=1)
    for r in range(1, nrow + 1):
        fig.update_yaxes(title_text="[-1, 1]", range=[-1.5, 1.5], row=r, col=1)
    fig.update_layout(template="plotly_white", height=210 * nrow + 100, hovermode="x unified",
                      title=dict(text=f"{task.key} | window {w} (record {BUNDLE.record_id[w]}) | {corr.label}", x=0.01),
                      legend=dict(orientation="h", y=-0.06), margin=dict(l=55, r=20, t=70, b=40))
    return fig


def _ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (v.mean() if len(v) else np.nan), 0.0
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def fig_metrics(rows, scale, region):
    if rows is None or not len(rows):
        return _empty("Metrics appear here after each model has run.")
    df = rows[(rows.scale == scale) & (rows.region == region)]
    if not len(df):
        return _empty(f"No rows for region '{region}' in this task.")
    fig = make_subplots(rows=1, cols=3, subplot_titles=("MAE (lower is better)", "RMSE (lower is better)",
                                                        "PCC (higher is better)"))
    for j, metric in enumerate(("MAE", "RMSE", "PCC"), start=1):
        for model, g in df.groupby("model", sort=False):
            mu, ci = _ci(g[metric])
            fig.add_trace(go.Bar(x=[model], y=[float(mu)], error_y=dict(type="data", array=[float(ci)], visible=True),
                                 marker_color=COL.get(model, "#888"), name=model, showlegend=(j == 1),
                                 text=[f"{mu:.3f}"], textposition="outside",
                                 hovertemplate=f"{model}<br>{metric} %{{y:.4f}} ± {ci:.4f} (95% CI)<br>n = {len(g)}<extra></extra>"),
                          1, j)
    fig.update_layout(template="plotly_white", height=380, barmode="group",
                      title=dict(text=f"Mean ± 95% CI over evaluated windows | scale {scale}, region {region}", x=0.01),
                      margin=dict(l=40, r=20, t=80, b=30), legend=dict(orientation="h", y=-0.12))
    return fig


def fig_distribution(rows, scale, region):
    if rows is None or not len(rows):
        return _empty("Per-window distributions appear here.")
    df = rows[(rows.scale == scale) & (rows.region == region)]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Per-window PCC", "Per-window MAE",
                                                        "Paired PCC (each dot = one window)"))
    for model, g in df.groupby("model", sort=False):
        for j, metric in ((1, "PCC"), (2, "MAE")):
            fig.add_trace(go.Box(y=_l(g[metric]), name=model, marker_color=COL.get(model), boxpoints="all",
                                 jitter=0.4, pointpos=0, showlegend=False, customdata=g.sample_id.tolist(),
                                 hovertemplate="window %{customdata}<br>%{y:.4f}<extra></extra>"), 1, j)
    piv = df.pivot_table(index="sample_id", columns="model", values="PCC")
    present = [NAME[k] for k in ("m", "v5", "uni") if NAME[k] in piv.columns]
    if len(present) >= 2:
        ym = present[0]                                   # DiBiMa-Cardio-M (or DiBiMa-Cardio)
        xm = NAME["uni"] if NAME["uni"] in present and ym != NAME["uni"] else present[1]   # vs UniCardio-M if run
        p = piv[[xm, ym]].dropna()
        fig.add_trace(go.Scatter(x=_l(p[xm]), y=_l(p[ym]), mode="markers",
                                 marker=dict(color=COL[ym], size=6, opacity=0.6), text=list(p.index), showlegend=False,
                                 hovertemplate=f"window %{{text}}<br>{xm} %{{x:.3f}}<br>{ym} %{{y:.3f}}<extra></extra>"), 1, 3)
        lo = float(np.nanmin(p.values)) if len(p) else -1
        fig.add_trace(go.Scatter(x=[lo, 1], y=[lo, 1], mode="lines", line=dict(dash="dash", color="#aaa"),
                                 showlegend=False, hoverinfo="skip"), 1, 3)
        win = int((p[ym] > p[xm]).sum())
        fig.layout.annotations[2].text = f"Paired PCC: {SHORT[_key(ym)]} higher in {win}/{len(p)} windows"
        fig.update_xaxes(title_text=f"{SHORT[_key(xm)]} PCC", row=1, col=3)
        fig.update_yaxes(title_text=f"{SHORT[_key(ym)]} PCC", row=1, col=3)
    fig.update_layout(template="plotly_white", height=400, margin=dict(l=40, r=20, t=60, b=30))
    return fig


def fig_timing(timings):
    if not timings:
        return _empty("Inference times appear here after each model has run.")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Batch-8 throughput (ms per window)",
                                                        "Single-window latency, batch 1 (ms)"))
    for tm in timings.values():
        c = COL.get(tm["model"], "#888")
        fig.add_trace(go.Bar(x=[tm["model"]], y=[float(tm["ms_per_window"])], marker_color=c, name=tm["model"],
                             text=[f"{tm['ms_per_window']:.1f} ms"], textposition="outside"), 1, 1)
        if np.isfinite(tm["latency_b1_ms"]):
            fig.add_trace(go.Bar(x=[tm["model"]], y=[float(tm["latency_b1_ms"])], marker_color=c, showlegend=False,
                                 text=[f"{tm['latency_b1_ms']:.1f} ms"], textposition="outside"), 1, 2)
    fig.add_hline(y=4000, line_dash="dot", line_color="#999", row=1, col=2,
                  annotation_text="4 s window (real-time limit)", annotation_position="bottom right")
    fig.update_yaxes(type="log", title_text="ms (log scale)")
    title = "Inference time on " + next(iter(timings.values())).get("device", "GPU")
    lead = next((k for k in ("m", "v5") if k in timings), None)
    if lead:
        d = timings[lead]
        title += (f" | {SHORT[lead]}: {d['ms_per_window']:.1f} ms/window, "
                  f"real-time factor {4000 / d['latency_b1_ms']:.0f}x at batch 1")
        if "uni" in timings:
            u = timings["uni"]
            title += (f" | {u['ms_per_window'] / d['ms_per_window']:.0f}x faster than UniCardio-M (batch 8), "
                      f"{u['latency_b1_ms'] / d['latency_b1_ms']:.0f}x (batch 1)")
    fig.update_layout(template="plotly_white", height=400, showlegend=False,
                      title=dict(text=title, x=0.01), margin=dict(l=50, r=20, t=80, b=30))
    return fig


# ----------------------------------------------------------------------------- run (streaming)
def _rows_frame(rows_list):
    df = pd.DataFrame(rows_list)
    if len(df):
        rid = dict(zip(BUNDLE.sample_id, BUNDLE.record_id))
        df.insert(df.columns.get_loc("sample_id") + 1, "record_id", df.sample_id.map(rid))
    return df


def _csv(st):
    rows, urows = st.get("rows"), st.get("urows")
    if rows is None or not len(rows):
        return None
    df = rows
    if urows is not None and len(urows):
        df = rows.merge(urows, on=["model", "region", "sample_id"], how="left")
    df = df.assign(corruption=E.Corruption(*st["corr"]).label)
    path = os.path.join(tempfile.gettempdir(), f"per_window_metrics_{int(time.time() * 1000)}.csv")
    df.to_csv(path, index=False)
    return path


def _outputs(status, st, w, scale, region):
    rows, urows = st.get("rows"), st.get("urows")
    return (status, fig_waveform(st, w), main_table(st, w, scale, region), fig_metrics(rows, scale, region),
            fig_distribution(rows, scale, region), summarize(rows, scale, region, urows),
            fig_timing(st.get("timings", {})), timing_table(st.get("timings", {})),
            _csv(st) if st.get("done") else None, st)


def run(kind, target, inputs, models, preset, gap_s, horizon_s, alpha, w, scope, scale, region,
        multi_source="auto"):
    baseline = "uni" in (models or [])
    if LOAD_ERROR:
        yield _outputs(f"**Model loading failed**\n\n```\n{LOAD_ERROR[-2000:]}\n```", {}, w, scale, region)
        return
    try:
        task = resolve(kind, target, inputs)
        corr = corruption(preset, gap_s, horizon_s, alpha)
        if task.kind == "imputation" and corr.gap_s is not None:
            E.task_mask(BUNDLE, "imputation", task.target, [0], corr)
        if task.kind == "forecast" and corr.horizon_s is not None:
            E.task_mask(BUNDLE, "forecast", task.target, [0], corr)
    except ValueError as e:
        yield _outputs(f"**Invalid selection:** {e}", {}, w, scale, region)
        return
    models = [k for k in DIBIMA_ORDER if k in (models or [])]
    if not models and not baseline:
        yield _outputs("**Select at least one model.**", {}, w, scale, region)
        return
    region = region if region == "full" or task.kind in ("imputation", "forecast") else "full"
    idx = window_set(scope, w)
    st = {"task": task.key, "idx": idx, "corr": _corr_tuple(corr), "models": models, "uni": bool(baseline),
          "timings": {}, "rows": None, "urows": None, "pred": {}, "ref": {}, "draws": {}, "msg": {}}
    regs = E.scoring_regions(BUNDLE, task, idx, corr)
    sids = [BUNDLE.sample_id[i] for i in idx]
    rows_all, urows_all = [], []
    n_steps = len(models) + (1 if baseline else 0)
    for step, v in enumerate(models, start=1):
        yield _outputs(f"**{step}/{n_steps} Running {NAME[v]}** on {len(idx)} window(s) ({corr.label}) ...",
                       st, w, scale, region)
        t0 = time.perf_counter()
        try:
            pred, tm, draws = gpu_dibima(v, task.key, idx, default_multi(v, multi_source), _corr_tuple(corr))
        except Exception as e:
            st["msg"][v] = f"{NAME[v]} failed: {e}"
            st.setdefault("failed", []).append(f"{NAME[v]} failed ({type(e).__name__}: {e}); press Run again")
            yield _outputs(f"**{NAME[v]} failed:** {type(e).__name__}: {e}", st, w, scale, region)
            continue
        tm["model"] = NAME[v]
        st["pred"][v], st["draws"][v] = pred, draws
        st["ref"][v] = BUNDLE.native("targets", task.target, idx)
        rows_all += E.score(pred, st["ref"][v], regs, NAME[v], task, sids)
        urows_all += E.uncertainty_rows(draws, st["ref"][v], regs, NAME[v], sids)
        st["rows"], st["urows"] = _rows_frame(rows_all), pd.DataFrame(urows_all)
        st["timings"][v] = tm
        st["done"] = step == n_steps
        yield _outputs(f"**{NAME[v]} done**: {tm['ms_per_window']:.1f} ms/window (batch 8), "
                       f"{tm['latency_b1_ms']:.1f} ms at batch 1, {tm['total_s']:.2f} s GPU compute "
                       f"({time.perf_counter() - t0:.1f} s wall incl. GPU allocation).", st, w, scale, region)
    if not baseline:
        st["done"] = True
        yield _outputs("**Done.** " + ", ".join(f"{SHORT[k]} {st['timings'][k]['ms_per_window']:.1f} ms/window"
                                                for k in models if k in st["timings"]) + f" ({corr.label})."
                       + "".join(f" **{f}.**" for f in st.get("failed", [])),
                       st, w, scale, region)
        return
    ok, why = E.uni_supports(UNI_CFG, task)
    if not ok:
        st["msg"]["uni"] = f"UniCardio-M not evaluable: {why}"
        st["done"] = True
        yield _outputs(f"**Done.** UniCardio-M not evaluable for this configuration: {why}"
                       + "".join(f" **{f}.**" for f in st.get("failed", [])), st, w, scale, region)
        return
    preds_u, refs_u, draws_u, rng, tm_u = [], [], [], None, None
    chunks = [idx[s:s + UNI_CHUNK] for s in range(0, len(idx), UNI_CHUNK)]
    for k, ch in enumerate(chunks):
        done = sum(len(c) for c in chunks[:k])
        st["msg"]["uni"] = f"UniCardio-M pending ({done}/{len(idx)} windows done) ..."
        yield _outputs(f"**{n_steps}/{n_steps} Running UniCardio-M DDIM-6**: {done}/{len(idx)} windows ...",
                       st, w, scale, region)
        try:
            p, y, tm, dr, rng = gpu_unicardio(task.key, ch, rng, k == 0, _corr_tuple(corr))
        except Exception as e:
            st["msg"]["uni"] = f"UniCardio-M failed: {e}"
            st["done"] = True
            yield _outputs(f"**UniCardio-M failed** on chunk {k + 1}: {type(e).__name__}: {e}", st, w, scale, region)
            return
        preds_u.append(p); refs_u.append(y); draws_u.append(dr)
        regs_c, sids_c = E.scoring_regions(BUNDLE, task, ch, corr), [BUNDLE.sample_id[i] for i in ch]
        rows_all += E.score(p, y, regs_c, NAME["uni"], task, sids_c)
        urows_all += E.uncertainty_rows(dr, y, regs_c, NAME["uni"], sids_c)
        if tm_u is None:
            tm_u = dict(tm)
        else:
            tm_u["total_s"] += tm["total_s"]
            tm_u["n_windows"] += tm["n_windows"]
            tm_u["ms_per_window"] = 1000 * tm_u["total_s"] / tm_u["n_windows"]
        tm_u["model"] = NAME["uni"]
        st["pred"]["uni"], st["ref"]["uni"], st["draws"]["uni"] = (np.concatenate(preds_u), np.concatenate(refs_u),
                                                                   np.concatenate(draws_u))
        st["rows"], st["urows"] = _rows_frame(rows_all), pd.DataFrame(urows_all)
        st["timings"]["uni"] = tm_u
        st["done"] = k == len(chunks) - 1
    lead = next((k for k in models if k in st["timings"]), None)
    msg = "**Done.** " + ", ".join(f"{SHORT[k]} {st['timings'][k]['ms_per_window']:.1f} ms/window" for k in models
                                   if k in st["timings"])
    msg += (", " if lead else "") + f"UniCardio-M {tm_u['ms_per_window']:.1f} ms/window (batch 8) on {tm_u.get('device', '')}."
    msg += "".join(f" **{f}.**" for f in st.get("failed", []))
    yield _outputs(msg, st, w, scale, region)


def rerender_wave(st, w, scale, region):
    return fig_waveform(st, w), main_table(st, w, scale, region)


def rerender_metrics(st, w, scale, region):
    rows, urows = (st or {}).get("rows"), (st or {}).get("urows")
    return (fig_metrics(rows, scale, region), fig_distribution(rows, scale, region),
            summarize(rows, scale, region, urows), main_table(st, w, scale, region))


# ----------------------------------------------------------------------------- UI
INTRO = f"""
# DiBiMa-Cardio: real-time cardiovascular signal restoration and translation

**DiBiMa-Cardio** is a diffusion-trained bidirectional-Mamba model that restores and translates photoplethysmography
(PPG), electrocardiogram (ECG) and arterial blood pressure (BP) waveforms in **one network pass** per 4-s window.
One model covers **imputation** of gaps, **denoising**, **forecasting**, **4x super-resolution** and **translation**
between signals, for each of the three outputs, with any subset of the other signals as extra inputs.

- **DiBiMa-Cardio-M** (recommended): multimodal fine-tune that uses the extra clean signals through a gated Bi-Mamba fusion
  bottleneck, early fusion, decoder fusion and time/modality-aware cross-attention.
- **DiBiMa-Cardio**: the base model (v5 full-data weights).
- **UniCardio-M**: the comparison model (UniCardio, DDIM with 6 steps x 6 draws = 36 network passes per window).

All selected models run **live on the GPU** for every request, on the same **256 test windows** (4 s at 125 Hz, held-out
records of the UCI cuff-less BP dataset) with the same corruptions. The DiBiMa-Cardio models are drawn as soon as they finish;
UniCardio-M then runs in chunks of {UNI_CHUNK} windows and the figures update after every chunk. The default corruption is
the hard protocol (**2.0 s gap, 2.0 s forecast horizon, 2x noise**); the standard benchmark corruption and custom values can
be selected.

{MODEL_INFO}
"""

NOTES = """
**Method notes.**
1. **Inputs.** The windows, references, gap positions and noise realisations are exported from the project's evaluator
   (`train_consistent` protocol, record-level preprocessing). *Hard corruption* rebuilds the masks as a 2.0 s gap centred on
   the stored gap position and the last 2.0 s for forecasting (zero fill), and doubles the stored noise residual
   (noise SD 0.71 x signal SD, about 3 dB SNR). *Standard benchmark corruption* uses the stored 1.2 s gap, ~1.17 s horizon and
   1x noise (0.355 x signal SD, about 9 dB SNR). *Custom* rebuilds gap/horizon at the chosen length and scales the noise.
   Super-resolution (factor 4) and translation are not affected by the corruption setting.
2. **Preprocessing** (train.py `preprocess_dibima`, applied to the whole record, then cut into windows): PPG and ECG 0.5 Hz
   high-pass + 50 Hz low-pass (Butterworth, order 5, zero-phase) + 50 Hz notch; ECG polarity corrected per record; BP
   (BP-100)/50; every window z-scored. Plots and the default `legacy_shape` scale put each window in min-max [-1, 1] units;
   `ref_zscore` keeps amplitude in units of the reference window's SD.
3. **Regions.** `full` = whole window; `missing` / `future` = the gap / forecast horizon only (imputation / forecasting).
4. **Inference time** is measured inside the GPU call with CUDA synchronisation after a warm-up pass. *Batch-8 throughput*
   is the compute time for the evaluated windows in batches of 8, divided by the number of windows. *Single-window latency*
   is one window at batch 1 (streaming use). On ZeroGPU the GPU is shared, so times vary between requests; GPU allocation
   and queueing are not included.
5. **Reproducibility.** DiBiMa-Cardio and DiBiMa-Cardio-M are deterministic. Under the standard benchmark corruption they reproduce the
   manuscript's per-window scores on all 57 input/task settings; under the hard corruption they reproduce the paper's
   hard-condition results.
6. **Batch size.** Outputs depend slightly on batch-size parity on some GPUs, so batches are padded to 8 as in the evaluator.
   The batch-1 latency pass is timed only and is not scored.
7. **Routes.** Restoration uses the routes `[0_0,1_1,2_2]` (extra clean signals enter through the shared encoder, the fusion
   modules and cross-modal attention); single-source translation uses `[s_0,s_1,s_2]`. Two-source translation: DiBiMa-Cardio-M
   uses the fused route it was fine-tuned on; DiBiMa-Cardio uses the mean of its two trained single-source predictions (its fused
   route was never trained). The setting below can force either mode for both models.
8. **Uncertainty.** 6 extra passes on independently diffused inputs at t = T-1 (training forward process, noise on the three
   signal channels only, torch seeds 42+i). The scored prediction stays the deterministic 1-step output and the draws are
   not counted in its inference time. Bands are ±1 SD and the 2.5-97.5 % range of the draws, drawn in min-max units with the
   affine map of the plotted prediction. *Coverage* (fraction of reference samples inside that range, reference-z units) is
   a coarse calibration indicator with 6 draws, not a calibrated 95 % interval.
"""

BASELINE_NOTES = """
**UniCardio-M notes (DDIM-6).** UniCardio with DDIM sampling, 6 steps x 6 draws (36 network passes per window);
the scored prediction is the mean of the 6 draws and the band is their spread. It uses its own released preprocessing
(PPG/ECG min-max per window, BP (BP-100)/50) and is scored against its own reference windows, with the same gap masks and the
same standardized noise residual (scaled by the noise setting). Its draws start from the evaluator's per-task seed and the
random state is carried across chunks of 32 windows, so windows 0..N-1 follow the evaluator's draw sequence. It has no
super-resolution mode, and some input combinations have no verified task flag. Cross-model comparisons are descriptive
(common-unit scales only).
"""

with gr.Blocks(title="DiBiMa-Cardio: real-time cardiovascular signal restoration and translation") as demo:
    gr.Markdown(INTRO)
    if LOAD_ERROR:
        gr.Markdown(f"**Model loading failed** (see logs):\n```\n{LOAD_ERROR[-1500:]}\n```")
    st = gr.State({})
    with gr.Row(equal_height=False):
        # ---------------- left: outputs
        with gr.Column(scale=3):
            status = gr.Markdown("Ready. Configure on the right and press **Run**.")
            with gr.Accordion("Signals: input, predictions and references", open=True):
                wave = gr.Plot(fig_waveform({}, 0), label="Signals, min-max [-1, 1] (shaded: ±1 SD and 95% interval over draws)")
            with gr.Accordion("Main metrics", open=True):
                main = gr.Dataframe(label="Main metrics (scale / region as selected in the Accuracy tab)",
                                    interactive=False, wrap=True)
            with gr.Tabs():
                with gr.Tab("Accuracy"):
                    with gr.Row():
                        scale = gr.Radio([("ref_zscore (common units)", "ref_zscore"),
                                          ("legacy_shape (min-max)", "legacy_shape"), ("native (model units)", "native")],
                                         value="legacy_shape", label="Scale",
                                         info="legacy_shape = each window min-max to [-1, 1] (the plotted units)")
                        region = gr.Radio([("Whole window", "full"), ("Gap only (imputation)", "missing"),
                                           ("Horizon only (forecast)", "future")], value="missing", label="Region")
                    metrics_plot = gr.Plot(fig_metrics(None, "legacy_shape", "full"), label="Metrics")
                    dist_plot = gr.Plot(fig_distribution(None, "legacy_shape", "full"), label="Per-window")
                    summary = gr.Dataframe(label="Summary (mean ± SD over windows)", interactive=False)
                with gr.Tab("Inference time"):
                    time_plot = gr.Plot(fig_timing({}), label="Inference time")
                    timing_df = gr.Dataframe(label="Timing", interactive=False)
                with gr.Tab("Download"):
                    csv = gr.File(label="Per-window metrics (all scales and regions, CSV)")
        # ---------------- right: all configuration, help text and the Run button
        with gr.Column(scale=1, min_width=320):
            btn = gr.Button("Run", variant="primary")
            with gr.Accordion("Models", open=True):
                models = gr.CheckboxGroup([("DiBiMa-Cardio-M (multimodal fine-tune, recommended)", "m"), ("DiBiMa-Cardio (base)", "v5"),
                                           ("UniCardio-M (DDIM-6)", "uni")], value=["m", "v5", "uni"], label="Models",
                                          info="UniCardio-M runs last (36 network passes per window).")
            with gr.Accordion("Task", open=True):
                kind = gr.Radio([(v, k) for k, v in KIND_LABEL.items()], value="imputation", label="Task")
                target = gr.Radio([("PPG", "ppg"), ("ECG", "ecg"), ("BP", "bp")], value="ecg", label="Output signal")
                inputs = gr.CheckboxGroup([("PPG", "ppg"), ("ECG", "ecg"), ("BP", "bp")], value=["ecg", "ppg"],
                                          label="Input signals",
                                          info="Restoration tasks always use the (corrupted) output signal; tick others "
                                               "to add them as clean extra inputs.")
                multi = gr.Radio([("Per model (recommended)", "auto"), ("Fused route", "fused"),
                                  ("Mean of single-source passes", "ensemble")], value="auto",
                                 label="DiBiMa-Cardio two-source translation", info="Only affects translation from two inputs.")
            with gr.Accordion("Corruption", open=True):
                preset = gr.Radio([("Hard (2.0 s gap / horizon, 2x noise)", "hard"),
                                   ("Standard benchmark (1.2 s gap, ~1.17 s horizon, 1x noise)", "standard"),
                                   ("Custom", "custom")], value="hard", label="Corruption",
                                  info="Sliders are editable with Custom.")
                gap_s = gr.Slider(0.5, 3.0, value=2.0, step=0.1, label="Gap length (s)", interactive=False)
                horizon_s = gr.Slider(0.5, 3.0, value=2.0, step=0.01, label="Forecast horizon (s)", interactive=False)
                alpha = gr.Slider(0.0, 3.0, value=2.0, step=0.25, label="Noise scale (x standard)", interactive=False)
            with gr.Accordion("Example window", open=True):
                window = gr.Slider(0, N_WIN - 1, value=0, step=1, label="Example window (0-255)")
                scope = gr.Radio([("This window only", "one"), ("Windows 0-31", "32"), ("Windows 0-63", "64"),
                                  ("All 256 windows", "256")], value="32", label="Windows to evaluate",
                                 info="The example window is always included. All 256 takes about 1-2 min for UniCardio-M.")
            with gr.Accordion("Configuration details", open=True):
                info = gr.Markdown(task_info("imputation", "ecg", ["ecg", "ppg"]) if not LOAD_ERROR else "")
            with gr.Accordion("Method notes", open=False):
                gr.Markdown(NOTES)
                gr.Markdown(BASELINE_NOTES)

    info_in = [kind, target, inputs, multi, models, preset, gap_s, horizon_s, alpha]
    for comp in (kind, target):
        comp.change(sync_inputs, info_in, [inputs, info])
    preset.change(preset_changed, [preset, gap_s, horizon_s, alpha], [gap_s, horizon_s, alpha]).then(
        task_info, info_in, info)
    for comp in (inputs, multi, models, gap_s, horizon_s, alpha):
        comp.input(task_info, info_in, info)
    outs = [status, wave, main, metrics_plot, dist_plot, summary, time_plot, timing_df, csv, st]
    btn.click(run, [kind, target, inputs, models, preset, gap_s, horizon_s, alpha, window, scope, scale,
                    region, multi], outs, concurrency_limit=2)
    window.release(rerender_wave, [st, window, scale, region], [wave, main])
    for comp in (scale, region):
        comp.change(rerender_metrics, [st, window, scale, region], [metrics_plot, dist_plot, summary, main])

if __name__ == "__main__":
    demo.queue(max_size=20).launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
                                   server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
                                   # weights (local folder or HF cache) are never served to visitors
                                   blocked_paths=[E.weights_cache_dir(), os.path.join(os.path.dirname(
                                       os.path.abspath(__file__)), "weights")])
