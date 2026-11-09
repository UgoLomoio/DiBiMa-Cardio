"""Download the DiBiMa-Cardio and DiBiMa-Cardio-M weights from Hugging Face into weights/.

    python scripts/download_weights.py                 # uses $HF_TOKEN if set
    python scripts/download_weights.py --token hf_xxx  # or pass the token explicitly

The weights repository (default Ugo96/dibima-cardio-v5-weights) may be private: request access from the
authors and create a read token at https://huggingface.co/settings/tokens. The UniCardio checkpoint
(weights/unicardio_no_compress799.pth, MIT) is already part of this repository.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("dibima_v5.pt", "dibima_v5_params.yaml", "dibima_v5_ext_templates.npz",
         "dibima_m.pt", "dibima_m_params.yaml")


def main():
    p = argparse.ArgumentParser(description="Download DiBiMa-Cardio weights into weights/")
    p.add_argument("--repo", default=os.environ.get("DIBIMA_WEIGHTS_REPO", "Ugo96/dibima-cardio-v5-weights"))
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--out", default=str(ROOT / "weights"))
    p.add_argument("--force", action="store_true", help="re-download files that already exist")
    a = p.parse_args()
    from huggingface_hub import hf_hub_download
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dst = out / name
        if dst.is_file() and not a.force:
            print(f"exists     {dst}")
            continue
        src = hf_hub_download(repo_id=a.repo, filename=name, repo_type="model", token=a.token or None)
        shutil.copyfile(src, dst)
        print(f"downloaded {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    print("done")


if __name__ == "__main__":
    main()
