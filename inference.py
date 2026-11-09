"""Command-line inference for DiBiMa-Cardio, DiBiMa-Cardio-M and the UniCardio-M baseline.

Runs the selected models on the 256-window test subset (data/test256_inputs.npz) for one task or for all 57
task/input combinations of the paper, and writes per-window metrics, summary metrics, timings and (optionally)
the predicted waveforms.

Examples
--------
    # BP imputation from a corrupted BP window plus a clean PPG window, hard corruption, both DiBiMa-Cardio models
    python inference.py --kind imputation --target bp --inputs bp ppg

    # PPG -> ECG translation, all three models, first 32 windows, save waveforms
    python inference.py --kind translation --target ecg --inputs ppg --models m v5 uni --windows 32 --save-predictions

    # every task of the paper under the standard benchmark corruption
    python inference.py --kind all --corruption standard --out results/standard
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402

MODEL_NAMES = {"m": "DiBiMa-Cardio-M", "v5": "DiBiMa-Cardio", "uni": "UniCardio-M"}
KIND_CHOICES = ("imputation", "denoising", "forecast", "sr", "translation", "all")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--models", nargs="+", default=["m", "v5"], choices=list(MODEL_NAMES),
                   help="m = DiBiMa-Cardio-M, v5 = DiBiMa-Cardio, uni = UniCardio-M (DDIM-6 baseline, slower)")
    p.add_argument("--kind", default="imputation", choices=KIND_CHOICES,
                   help="task type; 'all' runs the 57 task/input combinations of the paper")
    p.add_argument("--target", default="bp", choices=E.MODALITIES, help="output signal")
    p.add_argument("--inputs", nargs="+", default=None, choices=E.MODALITIES,
                   help="input signals (default: the target alone for restoration; required for translation). "
                        "For restoration the target is the corrupted input, the others are clean extra inputs")
    p.add_argument("--corruption", default="hard", choices=["hard", "standard", "custom"],
                   help="hard = 2.0 s gap / 2.0 s horizon / 2x noise (paper primary); "
                        "standard = stored 1.2 s gap / ~1.17 s horizon / 1x noise")
    p.add_argument("--gap", type=float, default=2.0, help="custom imputation gap in seconds (0.5-3.0)")
    p.add_argument("--horizon", type=float, default=2.0, help="custom forecast horizon in seconds (0.5-3.0)")
    p.add_argument("--noise", type=float, default=2.0, help="custom noise scale (x the standard test noise)")
    p.add_argument("--windows", type=int, default=256, help="number of test windows (first N of 256)")
    p.add_argument("--draws", type=int, default=0,
                   help="DiBiMa-Cardio uncertainty draws per window (0 = point prediction only)")
    p.add_argument("--multi-source", default="auto", choices=["auto", "fused", "ensemble"],
                   help="two-source translation: auto = fused for DiBiMa-Cardio-M, ensemble for DiBiMa-Cardio")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-predictions", action="store_true", help="also save predicted waveforms (.npz)")
    p.add_argument("--out", default="results", help="output folder")
    return p.parse_args(argv)


def corruption(args):
    if args.corruption == "hard":
        return E.HARD
    if args.corruption == "standard":
        return E.STANDARD
    return E.Corruption(args.gap, args.horizon, args.noise)


def tasks(args):
    if args.kind == "all":
        return E.task_roster()
    inputs = args.inputs or ([args.target] if args.kind != "translation" else None)
    if not inputs:
        sys.exit("--inputs is required for translation (e.g. --target ecg --inputs ppg)")
    return [E.make_task(args.kind, args.target, inputs)]


def summarise(rows):
    df = pd.DataFrame(rows)
    return (df.groupby(["model", "task", "variant", "scale", "region"], sort=False)[list(E.METRICS)]
              .mean().reset_index())


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("A CUDA GPU is required (the Mamba2 kernels are Triton/CUDA).")
    E.configure_torch()
    corr = corruption(args)
    bundle = E.Bundle()
    idx = list(range(min(args.windows, bundle.n)))
    sids = [bundle.sample_id[i] for i in idx]
    todo = tasks(args)
    print(f"{len(todo)} task(s), {len(idx)} window(s), {corr.label}")

    models = {}
    for k in args.models:
        t0 = time.perf_counter()
        models[k] = E.build_unicardio(args.device) if k == "uni" else E.build_dibima(args.device, k)
        print(f"loaded {MODEL_NAMES[k]} ({models[k].n_params / 1e6:.2f} M params, {time.perf_counter() - t0:.1f} s)")
    uni_cfg = E.load_uni_tasks() if "uni" in models else None

    rows, timings, preds = [], [], {}
    for task in todo:
        regions = E.scoring_regions(bundle, task, idx, corr)
        for k, model in models.items():
            name = MODEL_NAMES[k]
            if k == "uni":
                ok, why = E.uni_supports(uni_cfg, task)
                if not ok:
                    print(f"  {task.key:40s} {name:16s} skipped: {why}")
                    continue
                pred, ref, tm, _ = E.run_unicardio(model, bundle, task, idx, E.uni_entry(uni_cfg, task),
                                                   args.device, corr=corr)
            else:
                multi = args.multi_source if args.multi_source != "auto" else ("fused" if k == "m" else "ensemble")
                res = E.run_dibima(model, bundle, task, idx, args.device, n_draws=args.draws,
                                   multi_source=multi, corr=corr)
                pred, tm = res[0], res[1]
                ref = bundle.native("targets", task.target, idx)
                if args.draws:
                    preds[f"{k}__{task.name}__{task.variant}__draws"] = res[2]
            tm = dict(tm, model=name, task=task.name, variant=task.variant)
            timings.append(tm)
            rows += E.score(pred, ref, regions, name, task, sids)
            if args.save_predictions:
                preds[f"{k}__{task.name}__{task.variant}__pred"] = pred
                preds[f"{k}__{task.name}__{task.variant}__ref"] = ref
            r = pd.DataFrame(E.score(pred, ref, {"full": None}, name, task, sids))
            r = r[r.scale == "legacy_shape"]
            print(f"  {task.key:40s} {name:16s} MAE {r.MAE.mean():.4f}  PCC {r.PCC.mean():.3f}  "
                  f"{tm['ms_per_window']:.2f} ms/window")

    if not rows:
        sys.exit("Nothing was run.")
    pd.DataFrame(rows).to_csv(out / "per_window_metrics.csv", index=False)
    summarise(rows).to_csv(out / "summary_metrics.csv", index=False)
    pd.DataFrame(timings).to_csv(out / "timings.csv", index=False)
    meta = dict(models=[MODEL_NAMES[k] for k in models], corruption=corr.label, n_windows=len(idx),
                tasks=[t.key for t in todo], draws=args.draws, device=torch.cuda.get_device_name(0))
    (out / "run_config.json").write_text(json.dumps(meta, indent=1))
    if preds:
        np.savez_compressed(out / "predictions.npz", sample_id=np.array(sids), **preds)
    print(f"results written to {out.resolve()}")


if __name__ == "__main__":
    main()
