# DiBiMa-Cardio

Inference code and test data for **"Real-time restoration and translation of multimodal cardiovascular signals
with one-step bidirectional state-space diffusion"** (U. Lomoio, P. Liò, P. Veltri, P. H. Guzzi).

DiBiMa-Cardio is a diffusion-trained bidirectional-Mamba (Mamba2) model for photoplethysmography (PPG),
electrocardiogram (ECG) and arterial blood pressure (BP) waveforms. One model handles five tasks for each of the
three output signals:

| Task | Input | Output |
| --- | --- | --- |
| Imputation | window with a zero-filled gap | completed window |
| Denoising | noisy window | clean window |
| Forecasting | window with the last seconds hidden | completed window |
| Super-resolution (4x) | low-resolution window | full-resolution window |
| Translation | one or two other signals | the missing signal |

Any subset of the other two signals can be added as clean extra inputs. Inference is **one network pass per 4-s
window** (t = T-1 of a 20-step diffusion schedule), so a window takes milliseconds on a GPU.

Live demo: <https://huggingface.co/spaces/Ugo96/dibima-cardio>

## Models

| Model | Description | Parameters | Weights |
| --- | --- | --- | --- |
| **DiBiMa-Cardio-M** | Multimodal fine-tune of DiBiMa-Cardio: gated Bi-Mamba fusion bottleneck, early fusion, decoder fusion adapters and time/modality/mask-aware cross-attention. Recommended, especially with extra input signals | 5.36 M | Hugging Face (`scripts/download_weights.py`) |
| **DiBiMa-Cardio** | Base model: Bi-Mamba encoder/decoder with self- and cross-modal attention and a template prior for gaps and forecasts | 3.87 M | Hugging Face (`scripts/download_weights.py`) |
| **UniCardio-M** (baseline) | The released [UniCardio](https://github.com/thu-ml/UniCardio) checkpoint with deterministic DDIM sampling (6 steps x 6 draws, mean of the draws). No super-resolution mode | 8.12 M | included (`weights/`, MIT) |

## Repository layout

```
DiBiMa-Cardio/
├── inference.py              # command-line inference and scoring
├── app.py                    # interactive Gradio demo (same app as the Hugging Face Space)
├── engine.py                 # model loading, input contracts, corruptions, inference, metrics
├── src/                      # DiBiMa-Cardio model code (models.py, template_prior.py, utils.py)
├── unicardio/                # UniCardio model code and config (MIT, see unicardio/LICENSE)
├── unicardio_ddim.py         # deterministic DDIM sampler for UniCardio
├── data/
│   ├── test256_inputs.npz        # 256 test windows with references, gap masks and noise realisations
│   ├── test256_inputs_meta.json  # protocol metadata
│   └── uni_tasks_ddim.json       # UniCardio task flags (verified contract)
├── weights/                  # model weights (see weights/README.md)
├── scripts/download_weights.py
├── requirements.txt
├── environment.yml
└── INSTALL.md                # environment installation guide
```

## Installation

You need Linux and an NVIDIA GPU (the Mamba2 kernels are Triton/CUDA). The tested setup is Python 3.13.2,
PyTorch 2.9.0 + CUDA 12.8 and mamba_ssm 2.3.1. **See [INSTALL.md](INSTALL.md) for the full guide** and for
troubleshooting. In short:

```bash
git clone https://github.com/UgoLomoio/DiBiMa-Cardio.git
cd DiBiMa-Cardio
conda env create -f environment.yml
conda activate dibima-cardio
pip install https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.9cxx11abiTRUE-cp313-cp313-linux_x86_64.whl
export HF_TOKEN=hf_xxx                # only needed while the weights repository is private
python scripts/download_weights.py
```

## Data

`data/test256_inputs.npz` holds the 256 held-out 4-s windows (125 Hz, 500 samples) of the paper's test subset. They
come from the [UCI Cuff-Less Blood Pressure Estimation dataset](https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation)
(Kachuee et al., derived from MIMIC-II) and are the same windows as UniCardio's test subset. For every window and signal
the file stores:

* the reference window, and UniCardio's own preprocessed reference. DiBiMa-Cardio preprocessing is applied to the
  whole record before windowing: PPG and ECG 0.5 Hz high-pass, 50 Hz low-pass (Butterworth, order 5, zero-phase) and
  50 Hz notch; ECG polarity corrected per record; BP scaled as (BP - 100) / 50. Each window is then z-scored;
* the corrupted inputs of the standard benchmark: 1.2 s gap, ~1.17 s forecast horizon, Gaussian noise
  (SD 0.355 x signal SD, about 9 dB SNR) and the 4x low-resolution version;
* the matching observation masks, sample IDs and record IDs.

The corruption is rebuilt from these arrays at run time:

| Setting | Imputation gap | Forecast horizon | Denoising noise |
| --- | --- | --- | --- |
| `hard` (default, primary protocol of the paper) | 2.0 s, centred on the stored gap | last 2.0 s | 2x the stored noise (about 3 dB SNR) |
| `standard` (benchmark) | stored 1.2 s gap | stored last ~1.17 s | 1x |
| `custom` | `--gap` s | `--horizon` s | `--noise` x |

Super-resolution and translation do not depend on the corruption setting.

## Inference

### Command line

```bash
# BP imputation (hard corruption) with DiBiMa-Cardio-M and DiBiMa-Cardio
python inference.py --kind imputation --target bp --inputs bp

# same task with a clean PPG window as extra input
python inference.py --kind imputation --target bp --inputs bp ppg

# PPG -> ECG translation, all three models, first 32 windows, save the waveforms and 6 uncertainty draws
python inference.py --kind translation --target ecg --inputs ppg --models m v5 uni --windows 32 \
                    --draws 6 --save-predictions --out results/ppg2ecg

# all 57 task/input combinations of the paper under the standard benchmark corruption
python inference.py --kind all --corruption standard --out results/standard
```

On an RTX 4000 Ada, DiBiMa-Cardio and DiBiMa-Cardio-M take about 13 ms per window at batch 8, and `--kind all` with both
models (57 tasks x 256 windows) runs in about 10 minutes, mostly CPU time for the template prior. UniCardio-M needs
36 network passes per window (about 0.8 s per window).

The main options are listed below; `python inference.py -h` shows all of them.

| Option | Values | Default |
| --- | --- | --- |
| `--models` | `m` (DiBiMa-Cardio-M), `v5` (DiBiMa-Cardio), `uni` (UniCardio-M) | `m v5` |
| `--kind` | `imputation`, `denoising`, `forecast`, `sr`, `translation`, `all` | `imputation` |
| `--target` | `ppg`, `ecg`, `bp` | `bp` |
| `--inputs` | input signals; for restoration the target is the corrupted input and the others are clean extras | target only |
| `--corruption` | `hard`, `standard`, `custom` (`--gap`, `--horizon`, `--noise`) | `hard` |
| `--windows` | number of test windows (1-256) | 256 |
| `--draws` | uncertainty draws per window for the DiBiMa-Cardio models | 0 |
| `--multi-source` | two-source translation: `fused` (one pass, both sources) or `ensemble` (mean of the two single-source passes); `auto` = fused for DiBiMa-Cardio-M, ensemble for DiBiMa-Cardio | `auto` |

Each run writes the following files to `--out`:

| File | Content |
| --- | --- |
| `per_window_metrics.csv` | MAE, MSE, RMSE, PCC, SNR, PSNR and KS statistic for each window, scale and region |
| `summary_metrics.csv` | the same metrics averaged over windows |
| `timings.csv` | throughput (ms/window at batch 8), batch-1 latency and passes per window |
| `run_config.json` | models, corruption, windows and tasks |
| `predictions.npz` | predicted and reference waveforms (with `--save-predictions`) and uncertainty draws (with `--draws`) |

There are three metric scales. `legacy_shape` min-max scales each window to [-1, 1]; it is the common scale for
comparing with UniCardio. `ref_zscore` z-scores with the reference statistics, and `native` uses the model's own
units. Each metric is reported over the `full` window and, for imputation and forecasting, over the `missing` gap
or the `future` horizon only. Each model is scored against its own reference preprocessing.

### Python API

```python
import engine as E

E.configure_torch()
bundle = E.Bundle()                                     # data/test256_inputs.npz
model = E.build_dibima("cuda", variant="m")             # "m" = DiBiMa-Cardio-M, "v5" = DiBiMa-Cardio
task = E.make_task("imputation", "bp", ["bp", "ppg"])   # BP gap filling with a clean PPG window as extra input
idx = range(32)
pred, timing = E.run_dibima(model, bundle, task, idx, "cuda", n_draws=0, multi_source="fused", corr=E.HARD)
ref = bundle.native("targets", "bp", list(idx))
rows = E.score(pred, ref, E.scoring_regions(bundle, task, list(idx), E.HARD), "DiBiMa-Cardio-M", task,
               [bundle.sample_id[i] for i in idx])
```

To run your own signals, build the model input with the same layout as `engine.dibima_x`:
`[B, 6 + 3K, 500]` = PPG, ECG, BP (4 s at 125 Hz, preprocessed as above and z-scored per window), their observation masks (1 = observed),
and the template-prior block from `model.prior`. Then call `engine.dibima_forward` with the routes from
`engine.train_consistent_flags(task)`.

### Interactive demo

```bash
python app.py        # http://127.0.0.1:7860
```

This runs the same app as the [Hugging Face Space](https://huggingface.co/spaces/Ugo96/dibima-cardio). You can choose
the models, task, signals, corruption and windows, and the app shows the waveforms with uncertainty bands,
per-window and summary metrics, and inference times.

## Citation

```bibtex
@article{lomoio2026dibimacardio,
  title   = {Real-time restoration and translation of multimodal cardiovascular signals with one-step
             bidirectional state-space diffusion},
  author  = {Lomoio, Ugo and Li{\`o}, Pietro and Veltri, Pierangelo and Guzzi, Pietro Hiram},
  year    = {2026}
}
```

## License and credits

Code: MIT (see [LICENSE](LICENSE)). UniCardio model code and checkpoint: MIT, TSAIL group (see
[unicardio/LICENSE](unicardio/LICENSE)). Data: derived from the UCI Cuff-Less Blood Pressure Estimation dataset
(MIMIC-II). Mamba2: [state-spaces/mamba](https://github.com/state-spaces/mamba).

**Research use only. Not a medical device.**
