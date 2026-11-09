# Environment installation guide

DiBiMa-Cardio runs on Linux with an NVIDIA GPU. The Mamba2 layers use Triton/CUDA kernels from
[`mamba_ssm`](https://github.com/state-spaces/mamba), which have no CPU path.

## Requirements

| Component | Tested version | Notes |
| --- | --- | --- |
| OS | Ubuntu 22.04 / 24.04 (x86_64) | other Linux distributions should work |
| GPU | NVIDIA, compute capability >= 7.0 | tested on RTX 4000 Ada (20 GB) |
| NVIDIA driver | supporting CUDA 12.8 | check with `nvidia-smi` |
| Conda | Miniconda / Anaconda | |
| Python | 3.13.2 | 3.10-3.13 work if matching `mamba_ssm` wheels exist |
| PyTorch | 2.9.0 + cu128 | |
| mamba_ssm | 2.3.1 | |
| causal-conv1d | 1.6.2.post1 | optional, see step 4 |
| triton | 3.5.0 | installed with PyTorch |

## 1. Clone the repository

```bash
git clone https://github.com/UgoLomoio/DiBiMa-Cardio.git
cd DiBiMa-Cardio
```

## 2. Create the conda environment

**Option A: `environment.yml`** (installs Python, the CUDA build of PyTorch and `requirements.txt`)

```bash
conda env create -f environment.yml
conda activate dibima-cardio
```

**Option B: step by step**

```bash
conda create -n dibima-cardio python=3.13.2 -y
conda activate dibima-cardio
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Check that PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 2.9.0+cu128 12.8 True
```

## 3. Install `mamba_ssm`

Install the prebuilt wheel that matches your Python, PyTorch and CUDA versions (all wheels are listed on the
[mamba v2.3.1 release page](https://github.com/state-spaces/mamba/releases/tag/v2.3.1)). For the tested setup
(Python 3.13, torch 2.9, CUDA 12):

```bash
pip install https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.9cxx11abiTRUE-cp313-cp313-linux_x86_64.whl
```

For another configuration, replace `cp313` with your Python tag (`cp310`-`cp313`) and `torch2.9` with your PyTorch
version. Recent PyTorch wheels from pip use the C++11 ABI (`cxx11abiTRUE`); you can check with
`python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"`.

If no wheel matches, build from source. This needs the CUDA toolkit (`nvcc`) at the same major/minor version as
PyTorch's CUDA, and takes 10-30 minutes:

```bash
conda install -c nvidia cuda-toolkit=12.8 -y      # provides nvcc 12.8 inside the environment
pip install ninja packaging
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --no-build-isolation "mamba_ssm==2.3.1"
```

The build is faster if you limit it to your GPU architecture, for example `TORCH_CUDA_ARCH_LIST="8.9"` on Ada GPUs
or `"8.6"` on RTX 30xx GPUs.

## 4. (Optional) Install `causal-conv1d`

`causal-conv1d` enables the fused Mamba2 kernel path. Without it, `engine.py` uses the Triton-only path
(`nn.Conv1d` + Triton SSD scan), which gives the same results within floating-point tolerance (largest per-window MAE difference
below 1e-3 over the 57 tasks of the test subset).

```bash
pip install --no-build-isolation "causal-conv1d>=1.4.0"
```

Set `DIBIMA_MAMBA_FUSED=0` or `DIBIMA_MAMBA_FUSED=1` to force either path.

## 5. Download the model weights

The DiBiMa-Cardio and DiBiMa-Cardio-M weights are hosted on Hugging Face. If the weights repository is private,
request access from the authors and create a read token at <https://huggingface.co/settings/tokens>.

```bash
export HF_TOKEN=hf_xxx              # only needed while the weights repository is private
python scripts/download_weights.py  # writes 5 files (~38 MB) into weights/
```

The UniCardio baseline checkpoint (`weights/unicardio_no_compress799.pth`, MIT license) ships with the repository.

## 6. Check the installation

```bash
python -c "import mamba_ssm, engine; print('mamba_ssm', mamba_ssm.__version__, '|', engine.mamba2_path(), 'path')"
python inference.py --kind imputation --target bp --inputs bp --windows 16
```

The second command loads both DiBiMa-Cardio models and prints the mean MAE/PCC of BP imputation on 16 test windows.
It writes its outputs to `results/`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ImportError: ... selective_scan_cuda` or `Mamba3` | `engine.import_mamba_safely()` stubs these optional parts (only Mamba2 is used). If the import still fails, reinstall `mamba_ssm` for your torch/CUDA versions |
| `undefined symbol` when importing `mamba_ssm` or `causal_conv1d` | the package was built against a different torch version: reinstall it with `--no-cache-dir --force-reinstall --no-build-isolation` |
| `nvcc` not found or wrong CUDA version during the build | `conda install -c nvidia cuda-toolkit=12.8`, then `export CUDA_HOME=$CONDA_PREFIX` |
| `401` / `403` / `GatedRepoError` from `download_weights.py` | the token has no access to the weights repository: set `HF_TOKEN` or pass `--token` |
| Out of GPU memory | use `--draws 0` (the default) and fewer `--windows` |
