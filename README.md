# IconGenAI

[![Code License](https://img.shields.io/badge/Code%20License-MIT-blue)](LICENSE)
[![Dataset License](https://img.shields.io/badge/Dataset%20License-CC%20BY--NC%204.0-red)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Weights License](https://img.shields.io/badge/Weights%20License-CC%20BY--NC%204.0-red)](https://creativecommons.org/licenses/by-nc/4.0/)
![Model](https://img.shields.io/badge/Base%20Model-Qwen3--Coder--30B--A3B-blue)
![LoRA](https://img.shields.io/badge/Fine--Tuning-QLoRA-blueviolet)
![Training](https://img.shields.io/badge/Training-H100%2080GB-orange)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

Fine-tuning a code language model for high-quality **monochrome** text-to-SVG icon generation via QLoRA on the Iconify corpus. Generates pixel-perfect, instantly rebrandable icons using `currentColor`.

---

## Prerequisites

- macOS with Apple Silicon (captioning and Mac pilot run use MLX)
- [uv](https://docs.astral.sh/uv/) (`brew install uv`)
- Cairo for SVG rendering (`brew install cairo`)
- ~20 GB free disk space (models + data)
- MacTeX for building the paper (`brew install --cask mactex`)
- Cloud GPU with ≥80 GB VRAM (H100/A100) for full training

## Setup

```bash
git clone https://github.com/yauheniya-adesso/icongenai.git
cd icongenai
uv sync                   # core dependencies
uv sync --extra dev       # Jupyter, pandas, matplotlib, etc.
```

---

## Pipeline

```
01_collect  →  02_caption  →  [02b_fix_captions]  →  03_filter
                                                           ↓
                                                       04_merge
                                                           ↓
                                                       05_prepare
                                                           ↓
                                                       06_train
                                                           ↓
                                                       07_generate        ← also run on base model (zero-shot baseline)
                                                           ↓
                                                       08_evaluate
                                                           ↓
                                                       09_compare
```

---

### Step 0 — Download models

Pre-cache models to avoid repeated downloads during captioning and training.

```bash
uv run scripts/00_download_models.py
# downloads mlx-community/Qwen2.5-VL-7B-Instruct-4bit  (~4.5 GB, captioning)
# downloads mlx-community/Qwen3.5-9B-MLX-4bit           (~5 GB, pilot training)

uv run scripts/00_download_models.py --skip-vision    # skip captioning model
uv run scripts/00_download_models.py --skip-finetune  # skip training model
```

Models are cached to `~/.cache/huggingface/hub/`.

---

### Step 1 — Collect icons

```bash
git clone https://github.com/iconify/icon-sets.git   # ~1.15 GB

uv run scripts/01_collect.py
# → data/icons_filtered.jsonl  (275,912 icons)
```

```
--icon-sets-dir PATH   path to cloned icon-sets repo (default: ../icon-sets)
--output PATH          output path (default: data/icons_filtered.jsonl)
```

Aggregates all Iconify icon sets and filters by license. Included licenses: MIT, Apache-2.0, OFL-1.1, CC0-1.0, ISC, BSD-3-Clause, Unlicense, MPL-2.0, CC-BY-4.0, CC-BY-3.0, CC-BY-NC-4.0. Output fields: `icon_id`, `icon_name`, `svg`, `path_count`.

<p align="center">
  <img src="notebooks/output/final_sample.png" width="100%" alt="Icon Examples by Path">
  <br>
  <em>Fig. 1: Sample icons stratified by structural complexity (1–5 path elements).</em>
</p>

---

### Step 2 — Caption icons

Render each icon to PNG and generate short + long captions via Qwen2.5-VL-7B. Fully resumable — safe to kill and restart.

```bash
caffeinate -i uv run scripts/02_caption.py
# → data/icons_captioned.jsonl
```

```
--input PATH         (default: data/icons_filtered.jsonl)
--output PATH        (default: data/icons_captioned.jsonl)
--model STR          (default: mlx-community/Qwen2.5-VL-7B-Instruct-4bit)
--offset N           skip first N icons — use for parallel captioning on a second machine
--limit N            process at most N icons (for testing)
--render-size INT    PNG render size in pixels (default: 256)
```

**Parallel captioning on two machines:**

```bash
# machine 1 (icons 0–225911)
caffeinate -i uv run scripts/02_caption.py

# machine 2 (icons 225912–end)
caffeinate -i uv run scripts/02_caption.py --offset 225912 --output data/icons_captioned_2.jsonl
```

Merge afterwards: `cat data/icons_captioned.jsonl data/icons_captioned_2.jsonl > data/icons_captioned_merged.jsonl`

<p align="center">
  <img src="notebooks/output/icon_caption_examples.png" width="100%" alt="Icon Caption Examples">
  <br>
  <em>Fig. 2: Sample icons with automatically generated captions, stratified by structural complexity.</em>
</p>

---

### Step 2b — Fix bad captions _(optional)_

Re-captions icons where the model fell back to the icon name or produced bloated output. Dry-run by default (lists bad IDs without re-captioning).

```bash
# dry run — list bad caption IDs
uv run scripts/02b_fix_captions.py --merged data/icons_captioned_merged.jsonl

# actually re-caption bad icons
uv run scripts/02b_fix_captions.py \
    --merged data/icons_captioned_merged.jsonl \
    --recaption
# → overwrites data/icons_captioned_merged.jsonl
```

```
--merged PATH       input merged caption file (default: data/icons_captioned_merged.jsonl)
--fixes-out PATH    checkpoint for resumable re-captioning (default: data/icons_caption_fixes.jsonl)
--output PATH       corrected output (default: overwrite --merged)
--model STR         (default: mlx-community/Qwen3-VL-4B-Instruct-4bit)
--short-max INT     max chars for SHORT caption (default: 100)
--long-max INT      max chars for LONG caption (default: 500)
--limit N           re-caption at most N icons
--force             ignore checkpoint, re-caption all bad icons
```

Flags a caption as bad if: `SHORT == normalized icon_name`, `LONG == copy of SHORT`, or either field exceeds the char limit. Resumable via `--fixes-out` checkpoint.

---

### Step 3 — Filter & normalize

Apply training-quality filters and normalize SVG format.

```bash
uv run scripts/03_filter.py
# → data/icons_training.jsonl  (~216k icons)
```

```
--input PATH       (default: data/icons_filtered.jsonl)
--output PATH      (default: data/icons_training.jsonl)
--max-paths INT    max path elements per icon (default: 5)
--svg-pct FLOAT    upper percentile cap for SVG string length (default: 99.0)
```

Filtering steps applied in order:

1. Keep monochrome + single-color only; exclude multicolor
2. Exclude animated icons (`<animate>`, `@keyframes`, etc.)
3. Convert explicit fill/stroke colors → `currentColor`
4. Inject `fill="currentColor"` on `<svg>` root if missing
5. Exclude logo, brand, flag, emoji, and cryptocurrency collections
6. Exclude icons with more than `--max-paths` path elements
7. Exclude SVG strings above the `--svg-pct` percentile length
8. Normalize: strip boilerplate/metadata, round floats to 2 dp, collapse whitespace
9. Rescale `viewBox` to `0 0 24 24` via `<g transform="matrix(…)">`

Output fields: `icon_id`, `icon_name`, `svg` (normalized), `path_count`, `color_class`, `svg_len`.

<p align="center">
  <img src="notebooks/output/final_sample_svg_inspection.png" width="100%" alt="SVG Examples by Path">
  <br>
  <em>Fig. 3: Normalized SVG code stratified by structural complexity.</em>
</p>

---

### Step 4 — Merge captions

Left-join captions onto the filtered training SVGs (matched on `icon_id`).

```bash
uv run scripts/04_merge.py
# → data/icons_training_captioned.jsonl
```

No CLI flags — paths are hardcoded:

| Input | File |
|---|---|
| Filtered training SVGs | `data/icons_training.jsonl` |
| Merged captions | `data/icons_captioned_merged.jsonl` |

Output: `data/icons_training_captioned.jsonl`. Uncaptioned rows are kept with `icon_name` used as fallback. Typical match rate: ~81% captioned.

---

### Step 5 — Prepare training splits

Split into train/val/test, augment training examples, and format as ChatML messages.

```bash
uv run scripts/05_prepare.py
# → data/train.jsonl   (90%, augmented up to 3× per icon)
# → data/valid.jsonl   (5%,  1 record per icon)
# → data/test.jsonl    (5%,  1 record per icon)
```

```
--input PATH          (default: data/icons_training_captioned.jsonl)
--output-dir PATH     (default: data/)
--val-ratio FLOAT     (default: 0.05)
--test-ratio FLOAT    (default: 0.05)
--seed INT            (default: 42)
--no-augment          single record per icon, no augmentation
--no-curriculum       random order instead of simple→complex
```

Each record is a ChatML conversation:

```json
{"messages": [
  {"role": "system",    "content": "You are an SVG icon generator…"},
  {"role": "user",      "content": "Generate an SVG icon: <prompt>"},
  {"role": "assistant", "content": "<svg …>…</svg>"}
]}
```

Augmentation generates up to 3 variants per icon using: short caption, long caption, and raw icon name. Curriculum ordering sorts by `(path_count, svg_len)` so the model trains on simpler icons first.

---

### Step 6 — Train

#### Mac pilot run (~6-7 h total on M4 24 GB)

Full fine-tune of Qwen2.5-Coder-1.5B on a 10k-record representative subset (3 epochs), followed by generation on 3,000 test icons and evaluation. Produces a before/after comparison table (VR, RSR, MPC) suitable for a GPU access application.

| Phase | Time (M4 24 GB) |
|---|---|
| Training (100k records × 3 epochs) | ~4 h |
| Generation (100 icons × 2 runs) | ~2–3 h |

```bash
bash scripts/06_train_test_mac.sh
# → models/pilot-lora/           LoRA adapter (checkpoint every 250 iters)
# → results/pilot/generated.jsonl
# → results/pilot/zeroshot.jsonl
# → results/pilot/generated.metrics.json
# → results/pilot/zeroshot.metrics.json
# → logs/train_mac_<timestamp>.log
```

The script runs end-to-end: data subset → train → generate (fine-tuned + zero-shot) → evaluate → comparison table. Generation is resumable — safe to interrupt and rerun.

```bash
# resume training after a restart
bash scripts/06_train_test_mac.sh --resume-adapter-file models/pilot-lora/adapters.npz
```

#### Cloud GPU training (H100 80 GB)

QLoRA fine-tune of Qwen3-Coder-30B-A3B-Instruct. Requires H100 or A100 80 GB.

```bash
# install cloud dependencies first
pip install torch transformers peft trl bitsandbytes accelerate flash-attn

python scripts/06_train.py
# → models/icongenai-qlora/
# checkpoints saved every 500 steps
```

```
--model_name STR                   (default: Qwen/Qwen3-Coder-30B-A3B-Instruct)
--data_dir PATH                    (default: data/)
--output_dir PATH                  (default: models/icongenai-qlora/)
--per_device_train_batch_size INT  (default: 4)
--gradient_accumulation_steps INT  (default: 4)
--lora_r INT                       (default: 64)
--lora_alpha INT                   (default: 128)
--lora_dropout FLOAT               (default: 0.05)
--max_seq_length INT               (default: 4096)
--num_train_epochs INT             (default: 3)
--learning_rate FLOAT              (default: 2e-4)
--save_steps INT                   (default: 500)
--warmup_ratio FLOAT               (default: 0.05)
```

Training uses 4-bit NF4 quantization, Flash Attention 2, and completion-only loss masking (gradients computed only on assistant/SVG tokens, not on the repeated system/user prompt).

**Resume after restart:**

```bash
# resume from a specific checkpoint
python scripts/06_train.py --resume_from_checkpoint models/icongenai-qlora/checkpoint-1000

# resume from the latest checkpoint automatically
python scripts/06_train.py --resume_from_checkpoint true
```

The trainer keeps the last 3 checkpoints (`save_total_limit=3`), so at most ~1500 steps of work can be lost.

---

### Step 7 — Generate

Batch-generate SVGs from test prompts using the fine-tuned model. Fully resumable.

```bash
# fine-tuned model
uv run scripts/07_generate.py --adapter models/icongenai-lora
# → results/generated.jsonl

# zero-shot baseline (no adapter)
uv run scripts/07_generate.py --no-adapter --output results/zeroshot.jsonl

# quick sanity check on first 200 prompts
uv run scripts/07_generate.py --adapter models/icongenai-lora --n 200
```

```
--model STR        base model (default: mlx-community/Qwen3.5-9B-MLX-4bit)
--adapter PATH     path to LoRA adapter directory
--no-adapter       zero-shot baseline
--test PATH        (default: data/test.jsonl)
--output PATH      (default: results/generated.jsonl)
--n INT            limit to first N prompts
--max-tokens INT   (default: 1024)
--temp FLOAT       sampling temperature (default: 0.0, greedy)
```

Output fields: `prompt`, `reference_svg`, `generated_svg`.

<p align="center">
  <img src="notebooks/output/model_comparison_grid.png" width="100%" alt="Model Comparison Grid">
  <br>
  <em>Fig. 4: Qualitative comparison of code language models on monochrome SVG icon generation. IconGenAI column is a placeholder pending fine-tuning.</em>
</p>

---

### Step 8 — Evaluate

Compute evaluation metrics on generated SVGs.

```bash
# fast — no PyTorch required
uv run scripts/08_evaluate.py --skip-clip --skip-fid
# → results/generated.metrics.json

# full metrics (requires PyTorch + open-clip-torch + cleanfid)
uv run scripts/08_evaluate.py
```

```
--input PATH    (default: results/generated.jsonl)
--skip-clip     skip CLIP similarity
--skip-fid      skip FID
--device STR    device for CLIP/FID (default: cpu; use mps on Apple Silicon)
```

| Metric | Description | Extra deps |
|---|---|---|
| **VR** — Validity Rate | fraction that parse as well-formed XML with `<svg>` root | none |
| **RSR** — Render Success Rate | fraction that render via CairoSVG without error | `cairosvg` |
| **MPC** — Mean Path Count | avg `<path>` elements in valid SVGs | none |
| **CLIP** | text ↔ rendered-PNG cosine similarity (ViT-B/32) | `torch open-clip-torch` |
| **FID** | Fréchet Inception Distance, generated vs. reference renders | `torch cleanfid` |

---

### Step 9 — Compare models

Aggregate `.metrics.json` files from multiple runs into a comparison table.

```bash
uv run scripts/09_compare.py \
    "Qwen3.5-9B zero-shot:results/zeroshot.metrics.json" \
    "IconGenAI (ours):results/generated.metrics.json"

# LaTeX table output
uv run scripts/09_compare.py \
    "Qwen3.5-9B zero-shot:results/zeroshot.metrics.json" \
    "IconGenAI (ours):results/generated.metrics.json" \
    --latex
```

Outputs a table to stdout with best values marked. Pass any number of `"Label:path"` arguments.
