"""
Step 2 — Caption icons with a vision LLM.

Reads data/icons_filtered.jsonl, renders each SVG to PNG, passes it to
Qwen2.5-VL-7B-Instruct-4bit, and writes SHORT + LONG natural language
descriptions to data/icons_captioned.jsonl.

The script is fully resumable: it skips any icon_id already present in the
output file, so it is safe to interrupt and re-run at any time.

Usage:
    uv run scripts/02_caption.py
    uv run scripts/02_caption.py --limit 100          # test run on first 100 icons
    uv run scripts/02_caption.py --limit 100 --offset 500  # icons 500..599
"""

import argparse
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import cairosvg
from PIL import Image
from tqdm import tqdm

VISION_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
RENDER_SIZE = 256  # pixels — enough detail for icon captioning, not wasteful

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

CAPTION_TASK = """\
Describe this SVG icon for a visual search index.

Respond in EXACTLY this format — two lines, nothing else:
SHORT: <a natural search query, 5–12 words, describing the concept and visual style>
LONG: <a detailed description, 20–35 words, covering visual style (outline/filled/flat/solid/line), all visible elements, any decorations or modifiers, and likely UI context>

Examples of good SHORT descriptions:
  "outlined credit card with sparkles"
  "filled house with chimney, real estate icon"
  "line-style shopping cart with checkmark badge"
"""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_svg(svg: str, size: int = RENDER_SIZE) -> str:
    """
    Render an SVG string to a PNG, write it to a temp file, return the path.
    The caller is responsible for deleting the file when done.
    Raises RuntimeError if rendering fails.
    """
    png_bytes = cairosvg.svg2png(
        bytestring=svg.encode(),
        output_width=size,
        output_height=size,
        background_color="white",  # icons with currentColor look better on white
    )
    # Write to a named temp file so mlx-vlm can read it by path
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(png_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Caption parsing
# ---------------------------------------------------------------------------

def parse_captions(raw: str, icon_name: str) -> tuple[str, str]:
    """
    Extract SHORT and LONG captions from the model output.
    Falls back gracefully if the model did not follow the format.
    """
    short, long_ = "", ""
    for line in raw.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("SHORT:"):
            short = line[6:].strip()
        elif line.upper().startswith("LONG:"):
            long_ = line[5:].strip()

    # Strip outer quotes the model sometimes adds: "outlined card" → outlined card
    short = short.strip('"\'')
    long_ = long_.strip('"\'')

    # Fallback: use icon name (normalised) so the record is never empty
    if not short:
        short = icon_name.replace("-", " ").replace("_", " ")
    if not long_:
        long_ = short

    return short, long_


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/icons_filtered.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/icons_captioned.jsonl"))
    parser.add_argument("--model", default=VISION_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N icons (for testing)")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N icons in the input")
    parser.add_argument("--render-size", type=int, default=RENDER_SIZE)
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"ERROR: input file not found: {args.input}\nRun scripts/01_collect.py first.")

    # --- Load checkpoint (already-captioned icon_ids) ---
    done_ids: set[str] = set()
    if args.output.exists():
        with open(args.output) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["icon_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming: {len(done_ids):,} icons already captioned, skipping them.")

    # --- Load all icons ---
    with open(args.input) as f:
        all_icons = [json.loads(line) for line in f]

    icons = all_icons[args.offset:]
    if args.limit is not None:
        icons = icons[:args.limit]

    to_process = [ic for ic in icons if ic["icon_id"] not in done_ids]
    print(f"Icons to caption: {len(to_process):,}  (of {len(icons):,} in range, {len(done_ids):,} already done)")

    if not to_process:
        print("Nothing to do.")
        return

    # --- Load vision model (once — expensive) ---
    print(f"\nLoading vision model: {args.model}")
    print("(first load may take 20–30 seconds while mlx compiles kernels)\n")

    # Patch: transformers >=5.1 tries to load Qwen2-VL's video processor, which
    # requires PyTorch/Torchvision. We only need the image processor for icons.
    # Removing "video_processor" from the lazy mapping prevents that code path.
    from transformers.processing_utils import MODALITY_TO_AUTOPROCESSOR_MAPPING
    MODALITY_TO_AUTOPROCESSOR_MAPPING._MAPPING_NAMES.pop("video_processor", None)

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    model, processor = load(args.model)
    config = load_config(args.model)

    # --- Caption loop ---
    stats = {"written": 0, "render_errors": 0, "parse_fallbacks": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "a") as out:
        for icon in tqdm(to_process, desc="Captioning", unit="icon"):
            tmp_path = None
            try:
                # 1. Render SVG → temp PNG
                tmp_path = render_svg(icon["svg"], size=args.render_size)

                # 2. Build prompt with image token
                user_msg = f"This SVG icon is named \"{icon['icon_name']}\" from the \"{icon['collection_name']}\" icon set.\n\n{CAPTION_TASK}"
                messages = [{"role": "user", "content": user_msg}]
                prompt = apply_chat_template(processor, config, messages, num_images=1)

                # 3. Generate captions (greedy, deterministic)
                result = generate(
                    model,
                    processor,
                    prompt,
                    image=tmp_path,
                    max_tokens=150,
                    temp=0.0,       # greedy — deterministic, factual
                    verbose=False,
                )
                raw_text = result.text

                # 4. Parse output
                short, long_ = parse_captions(raw_text, icon["icon_name"])
                if short == icon["icon_name"].replace("-", " ").replace("_", " "):
                    stats["parse_fallbacks"] += 1

                # 5. Write record
                record = {
                    "icon_id": icon["icon_id"],
                    "collection_prefix": icon["collection_prefix"],
                    "collection_name": icon["collection_name"],
                    "license_spdx": icon["license_spdx"],
                    "icon_name": icon["icon_name"],
                    "svg": icon["svg"],
                    "width": icon["width"],
                    "height": icon["height"],
                    "path_count": icon["path_count"],
                    "is_multicolor": icon["is_multicolor"],
                    "caption_short": short,
                    "caption_long": long_,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                stats["written"] += 1

            except Exception as e:
                tqdm.write(f"  WARN [{icon['icon_id']}]: {e}")
                stats["render_errors"] += 1
            finally:
                if tmp_path:
                    Path(tmp_path).unlink(missing_ok=True)

    print(f"\n=== Captioning complete ===")
    print(f"  Written         : {stats['written']:,}")
    print(f"  Render errors   : {stats['render_errors']:,}")
    print(f"  Parse fallbacks : {stats['parse_fallbacks']:,}  (used icon_name as caption)")
    print(f"  Output          : {args.output}")
    print(f"\nNext step: uv run scripts/03_prepare.py")


if __name__ == "__main__":
    main()
