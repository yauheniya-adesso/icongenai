"""
Step 2b — Fix bad captions in the merged caption file.

Identifies records in data/icons_captioned_merged.jsonl where the captioner
produced unusable output, re-captions them with the same VLM, and writes a
corrected merged file (all icons, bad ones replaced).

Bad-caption criteria (all configurable via flags):
  case 2  SHORT ok, LONG missing  → long_ was silently copied from short
  case 3  SHORT missing           → caption_short == normalised icon_name
  bloat   caption_short > --short-max chars  (model rambled on short)
  bloat   caption_long  > --long-max  chars  (model rambled on long)

The input file already contains the SVG field, so icons_filtered.jsonl is
not required.  The script is fully resumable: re-run it after an interruption
and it will skip any icon_id already present in --fixes-out.

Usage:
    uv run scripts/02b_fix_captions.py                        # dry run: list bad IDs only
    uv run scripts/02b_fix_captions.py --recaption            # actually fix them
    uv run scripts/02b_fix_captions.py --recaption --limit 10 # test on 10 icons
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cairosvg
from tqdm import tqdm

# Defaults matching 02_caption.py
VISION_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
RENDER_SIZE  = 256
SHORT_MAX    = 100   # chars; anything longer is model bloat
LONG_MAX     = 500   # chars; anything longer is model bloat
MAX_TOKENS   = 512   # higher than 02_caption.py: Qwen3-VL may emit <think> blocks first

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
# Quality checks — mirror the 4-case analysis in notebook 04
# ---------------------------------------------------------------------------

def _norm_name(icon_name: str) -> str:
    return icon_name.replace("-", " ").replace("_", " ").strip().lower()


def bad_reason(r: dict, short_max: int, long_max: int) -> str | None:
    """Return a short string describing why this record is bad, or None if ok."""
    s = r["caption_short"].strip()
    lo = r["caption_long"].strip()
    if s.lower() == _norm_name(r["icon_name"]):
        return "short=icon_name (case 3/4)"
    if s.lower() == lo.lower():
        return "long=copy_of_short (case 2)"
    if len(s) > short_max:
        return f"short_too_long ({len(s)} chars)"
    if len(lo) > long_max:
        return f"long_too_long ({len(lo)} chars)"
    return None


# ---------------------------------------------------------------------------
# Re-use render + parse logic from 02_caption.py
# ---------------------------------------------------------------------------

def render_svg(svg: str, size: int = RENDER_SIZE) -> str:
    png_bytes = cairosvg.svg2png(
        bytestring=svg.encode(),
        output_width=size,
        output_height=size,
        background_color="white",
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(png_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name


def parse_captions(raw: str, icon_name: str) -> tuple[str, str]:
    import re as _re
    # Qwen3-VL emits <think>…</think> before the answer; strip it before parsing.
    raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
    short, long_ = "", ""
    for line in raw.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("SHORT:"):
            short = line[6:].strip()
        elif line.upper().startswith("LONG:"):
            long_ = line[5:].strip()
    short = short.strip('"\'')
    long_ = long_.strip('"\'')
    if not short:
        short = icon_name.replace("-", " ").replace("_", " ")
    if not long_:
        long_ = short
    return short, long_


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--merged",    type=Path, default=Path("data/icons_captioned_merged.jsonl"),
                        help="Merged caption file to scan and fix")
    parser.add_argument("--fixes-out", type=Path, default=Path("data/icons_caption_fixes.jsonl"),
                        help="Append re-captioned records here (resumable checkpoint)")
    parser.add_argument("--output",    type=Path, default=Path("data/icons_captioned_merged.jsonl"),
                        help="Write corrected merged file here (default: overwrite --merged)")
    parser.add_argument("--model",     default=VISION_MODEL)
    parser.add_argument("--short-max", type=int, default=SHORT_MAX)
    parser.add_argument("--long-max",  type=int, default=LONG_MAX)
    parser.add_argument("--recaption", action="store_true",
                        help="Actually load the VLM and re-caption (omit for a dry run)")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Re-caption at most N icons (for testing)")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help="Max tokens per generation (increase for Qwen3-VL thinking mode)")
    parser.add_argument("--render-size", type=int, default=RENDER_SIZE)
    parser.add_argument("--force",      action="store_true",
                        help="Ignore existing --fixes-out checkpoint and re-caption all bad icons")
    args = parser.parse_args()

    if not args.merged.exists():
        sys.exit(f"ERROR: {args.merged} not found. Run scripts/02_caption.py first.")

    # ── Load merged file ──────────────────────────────────────────────────────
    merged: list[dict] = []
    with open(args.merged) as f:
        for line in f:
            if line.strip():
                merged.append(json.loads(line))
    print(f"Loaded {len(merged):,} records from {args.merged}")

    # ── Identify bad records ──────────────────────────────────────────────────
    bad: list[tuple[dict, str]] = []  # (record, reason)
    for r in merged:
        reason = bad_reason(r, args.short_max, args.long_max)
        if reason:
            bad.append((r, reason))

    print(f"\nBad captions found: {len(bad)}")
    reason_counts: dict[str, int] = {}
    for _, reason in bad:
        key = reason.split(" (")[0].split("_too_long")[0] + ("_too_long" if "too_long" in reason else "")
        reason_counts[key] = reason_counts.get(key, 0) + 1
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:30s}: {count}")

    if not bad:
        print("Nothing to fix.")
        return

    if not args.recaption:
        print("\n(dry run — pass --recaption to actually fix them)")
        print("\nBad icon IDs:")
        for r, reason in bad:
            print(f"  {r['icon_id']:55s}  {reason}")
        return

    # ── Load already-fixed checkpoint ────────────────────────────────────────
    fixed_ids: set[str] = set()
    fixed_records: dict[str, dict] = {}
    if args.fixes_out.exists() and not args.force:
        with open(args.fixes_out) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    fixed_ids.add(rec["icon_id"])
                    fixed_records[rec["icon_id"]] = rec
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"\nResuming: {len(fixed_ids)} already fixed, skipping them.")
    elif args.force and args.fixes_out.exists():
        print("\n--force: ignoring existing checkpoint.")

    to_fix = [(r, reason) for r, reason in bad if r["icon_id"] not in fixed_ids]
    if args.limit is not None:
        to_fix = to_fix[: args.limit]

    print(f"To re-caption: {len(to_fix)}")

    if to_fix:
        # ── Load VLM ─────────────────────────────────────────────────────────
        print(f"\nLoading vision model: {args.model}")
        print("(first load may take 20–30 seconds while mlx compiles kernels)\n")

        # Patch: transformers >=5.1 tries to load Qwen2-VL's video processor
        from transformers.processing_utils import MODALITY_TO_AUTOPROCESSOR_MAPPING
        MODALITY_TO_AUTOPROCESSOR_MAPPING._MAPPING_NAMES.pop("video_processor", None)

        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        model, processor = load(args.model)
        config = load_config(args.model)

        stats = {"written": 0, "render_errors": 0, "still_bad": 0}
        args.fixes_out.parent.mkdir(parents=True, exist_ok=True)

        write_mode = "w" if args.force else "a"
        with open(args.fixes_out, write_mode) as out:
            for icon, old_reason in tqdm(to_fix, desc="Re-captioning", unit="icon"):
                tmp_path = None
                try:
                    tmp_path = render_svg(icon["svg"], size=args.render_size)

                    user_msg = (
                        f"This SVG icon is named \"{icon['icon_name']}\" "
                        f"from the \"{icon['collection_name']}\" icon set.\n\n{CAPTION_TASK}"
                    )
                    messages = [{"role": "user", "content": user_msg}]
                    prompt = apply_chat_template(processor, config, messages, num_images=1)

                    result = generate(
                        model, processor, prompt,
                        image=tmp_path,
                        max_tokens=args.max_tokens,
                        temp=0.0,
                        verbose=False,
                    )
                    short, long_ = parse_captions(result.text, icon["icon_name"])

                    new_reason = bad_reason(
                        {"caption_short": short, "caption_long": long_, "icon_name": icon["icon_name"]},
                        args.short_max, args.long_max,
                    )
                    if new_reason:
                        stats["still_bad"] += 1
                        tqdm.write(f"  WARN still bad [{icon['icon_id']}]: {new_reason}")

                    record = {**icon, "caption_short": short, "caption_long": long_}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    fixed_records[icon["icon_id"]] = record
                    stats["written"] += 1

                except Exception as e:
                    tqdm.write(f"  WARN render error [{icon['icon_id']}]: {e}")
                    stats["render_errors"] += 1
                finally:
                    if tmp_path:
                        Path(tmp_path).unlink(missing_ok=True)

        print(f"\n  Written       : {stats['written']}")
        print(f"  Render errors : {stats['render_errors']}")
        print(f"  Still bad     : {stats['still_bad']}  (kept new output anyway)")

    # ── Merge fixes back into merged file ────────────────────────────────────
    if not fixed_records:
        print("No fixes to merge.")
        return

    print(f"\nMerging {len(fixed_records)} fixes into {args.output} …")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.output, "w") as out:
        for r in merged:
            rec = fixed_records.get(r["icon_id"], r)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written:,} records → {args.output}")
    print(f"\nNext step: uv run scripts/03_filter.py")


if __name__ == "__main__":
    main()
