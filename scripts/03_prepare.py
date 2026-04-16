"""
Step 3 — Prepare the training dataset in ChatML format for mlx-lm LoRA.

Reads data/icons_captioned.jsonl and produces:
  data/train.jsonl   — 90 % of icons
  data/valid.jsonl   — 5  %
  data/test.jsonl    — 5  %

Each record is formatted as:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

This is the native format consumed by `mlx_lm.lora --train`.

Curriculum ordering (default ON):
  Icons are sorted from simple (few paths, monochrome) to complex (many
  paths, multicolor) before splitting. The training loop therefore sees
  easy examples first, which mirrors the SVGen paper's approach.

Data augmentation (default ON):
  For each icon we generate up to 3 training variants with different user
  prompts pointing to the same SVG output:
    1. short caption  → SVG  (primary)
    2. long caption   → SVG
    3. raw icon name  → SVG  (e.g. "credit-card")
  This teaches the model to handle prompts of varying specificity.

Usage:
    uv run scripts/03_prepare.py
    uv run scripts/03_prepare.py --no-augment     # one variant per icon
    uv run scripts/03_prepare.py --no-curriculum  # random order
    uv run scripts/03_prepare.py --val-ratio 0.1  # larger validation set
"""

import argparse
import json
import random
import sys
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an SVG icon generator. "
    "Given a description, output clean, valid SVG code for a single icon. "
    "Use currentColor for fill and stroke so the icon inherits its parent's color. "
    "Output only the SVG element — no explanation, no markdown fences."
)

# Complexity score: used for curriculum ordering (lower = simpler)
def complexity(icon: dict) -> tuple:
    return (int(icon["is_multicolor"]), icon["path_count"])


def make_record(user_content: str, svg: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
            {"role": "assistant", "content": svg},
        ]
    }


def augment(icon: dict) -> list[dict]:
    """Return up to 3 training records for a single icon."""
    svg = icon["svg"]
    records = []

    short = icon.get("caption_short", "").strip()
    long_ = icon.get("caption_long", "").strip()
    name  = icon["icon_name"].replace("-", " ").replace("_", " ")

    # Variant 1 — short caption (always present)
    if short and short != name:
        records.append(make_record(f"Generate an SVG icon: {short}", svg))
    else:
        records.append(make_record(f"Generate an SVG icon: {name}", svg))

    # Variant 2 — long caption (if meaningfully different from short)
    if long_ and long_ != short and len(long_) > len(short) + 10:
        records.append(make_record(f"Generate an SVG icon: {long_}", svg))

    # Variant 3 — raw icon name (teaches the model to handle terse queries)
    if name and name != short:
        records.append(make_record(f"Generate an SVG icon: {name}", svg))

    return records


def split(items: list, val_ratio: float, test_ratio: float):
    n = len(items)
    n_val  = max(1, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio))
    n_train = n - n_val - n_test
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",       type=Path,  default=Path("data/icons_captioned.jsonl"))
    parser.add_argument("--output-dir",  type=Path,  default=Path("data"))
    parser.add_argument("--val-ratio",   type=float, default=0.05)
    parser.add_argument("--test-ratio",  type=float, default=0.05)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--no-augment",  action="store_true", help="One training record per icon (no augmentation)")
    parser.add_argument("--no-curriculum", action="store_true", help="Shuffle instead of curriculum ordering")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"ERROR: {args.input} not found. Run scripts/02_caption.py first.")

    # --- Load ---
    with open(args.input) as f:
        icons = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(icons):,} captioned icons.")

    # --- Order ---
    if args.no_curriculum:
        random.seed(args.seed)
        random.shuffle(icons)
        print("Order: random shuffle (curriculum disabled)")
    else:
        icons.sort(key=complexity)
        print("Order: curriculum (simple → complex by path_count and color)")

    # --- Split on icons first, then augment within each split ---
    # Important: split BEFORE augmenting so the same icon never appears in
    # both train and val/test (even as a different prompt variant).
    train_icons, val_icons, test_icons = split(icons, args.val_ratio, args.test_ratio)

    print(f"Split: {len(train_icons):,} train / {len(val_icons):,} val / {len(test_icons):,} test icons")

    # --- Augment ---
    if args.no_augment:
        train_records = [make_record(f"Generate an SVG icon: {ic.get('caption_short') or ic['icon_name']}", ic["svg"]) for ic in train_icons]
    else:
        train_records = [rec for ic in train_icons for rec in augment(ic)]

    # Val and test: one record per icon (short caption only — clean evaluation)
    val_records  = [make_record(f"Generate an SVG icon: {ic.get('caption_short') or ic['icon_name']}", ic["svg"]) for ic in val_icons]
    test_records = [make_record(f"Generate an SVG icon: {ic.get('caption_short') or ic['icon_name']}", ic["svg"]) for ic in test_icons]

    # --- Write ---
    write_jsonl(args.output_dir / "train.jsonl", train_records)
    write_jsonl(args.output_dir / "valid.jsonl", val_records)   # mlx-lm expects "valid.jsonl"
    write_jsonl(args.output_dir / "test.jsonl",  test_records)

    print(f"\n=== Dataset prepared ===")
    print(f"  train.jsonl : {len(train_records):,} records  (~{len(train_records)/len(train_icons):.1f}x per icon)")
    print(f"  valid.jsonl : {len(val_records):,} records")
    print(f"  test.jsonl  : {len(test_records):,} records")
    print(f"  Output dir  : {args.output_dir}/")
    print(f"\nEstimated training time on M4 (Qwen3.5-9B, 1 epoch):")
    avg_tokens = 400  # ~30 caption + ~370 SVG tokens average
    total_tokens = len(train_records) * avg_tokens
    print(f"  ~{total_tokens/1e6:.0f}M tokens × 2–3 epochs")
    print(f"  M4 Pro  (~1200 tok/s): ~{total_tokens * 2.5 / 1200 / 3600:.0f} hours total")
    print(f"  A100    (~20000 tok/s): ~{total_tokens * 2.5 / 20000 / 3600:.1f} hours total")
    print(f"\nNext step: uv run -m mlx_lm.lora --help  (then adapt scripts/04_train.sh)")


if __name__ == "__main__":
    main()
