"""
Step 0 — Pre-download models to the local HuggingFace cache.

Run this once before starting the captioning or training steps.
Models are cached in ~/.cache/huggingface/hub and reused automatically.

Usage:
    uv run scripts/00_download_models.py
    uv run scripts/00_download_models.py --skip-vision   # fine-tuning base only
    uv run scripts/00_download_models.py --skip-finetune # vision model only
"""

import argparse
from huggingface_hub import snapshot_download, hf_hub_download
import os

VISION_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
FINETUNE_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"


def download(repo_id: str, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"Downloading: {label}")
    print(f"  Repo : {repo_id}")
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    print(f"  Cache: {cache}")
    print(f"{'='*60}")
    local_dir = snapshot_download(
        repo_id=repo_id,
        local_files_only=False,  # fetch from HF if not cached
    )
    print(f"\nDone. Cached at:\n  {local_dir}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-vision", action="store_true", help="Skip vision model download")
    parser.add_argument("--skip-finetune", action="store_true", help="Skip fine-tuning base model download")
    args = parser.parse_args()

    if not args.skip_vision:
        download(VISION_MODEL, "Vision model (captioning) — Qwen2.5-VL-7B 4-bit (~4.5 GB)")
    else:
        print(f"Skipping vision model ({VISION_MODEL})")

    if not args.skip_finetune:
        download(FINETUNE_MODEL, "Fine-tuning base — Qwen3.5-9B 4-bit (~5 GB)")
    else:
        print(f"Skipping fine-tuning model ({FINETUNE_MODEL})")

    print("All downloads complete.")
    print("Next step: uv run scripts/02_caption.py")


if __name__ == "__main__":
    main()
