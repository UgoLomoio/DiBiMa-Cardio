# Model weights

| File | Model | Source |
| --- | --- | --- |
| `dibima_v5.pt`, `dibima_v5_params.yaml` | DiBiMa-Cardio (3.87 M parameters) | Hugging Face, `python scripts/download_weights.py` |
| `dibima_m.pt`, `dibima_m_params.yaml` | DiBiMa-Cardio-M (5.36 M parameters) | Hugging Face, `python scripts/download_weights.py` |
| `dibima_v5_ext_templates.npz` | Template prior (used by both DiBiMa-Cardio models) | Hugging Face, `python scripts/download_weights.py` |
| `unicardio_no_compress799.pth` | UniCardio released checkpoint (baseline, 8.12 M parameters, MIT) | included in this repository |

If a file is missing from this folder, `engine.py` falls back to downloading it from
`$DIBIMA_WEIGHTS_REPO` (default `Ugo96/dibima-cardio-v5-weights`) with `$HF_TOKEN`.
