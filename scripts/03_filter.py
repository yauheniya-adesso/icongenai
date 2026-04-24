"""
Step 3 — Filter and normalise SVG icons for model training.

Reads  data/icons_filtered.jsonl  (275,912 license-filtered icons)
Writes data/icons_training.jsonl  (~216k training-ready icons)

Filtering pipeline (mirrors notebooks/01_explore_filtered.ipynb §§7–8):
  1. Keep monochrome + single-color; exclude multicolor
  2. Exclude animated icons (<animate>, <set>, @keyframes, animation:)
  3. Convert single-color explicit fills → currentColor
  4. Inject fill="currentColor" on <svg> root for icons with no explicit fill/stroke
  5. Exclude icons with > 5 paths
  6. Exclude SVG strings longer than the 99th-percentile length
  7. Normalise SVG: strip boilerplate/metadata, reduce float precision, collapse whitespace
  8. Normalise viewBox to 24×24 via <g transform="matrix(…)">

Output record fields:
  icon_id, icon_name, svg (normalised), path_count, color_class, svg_len

This script is step 3 in the pipeline — run it after 01_collect.py, then
merge its output with icons_captioned.jsonl before running 03_prepare.py
(to be renumbered 04_prepare.py once captioning is complete).

Usage:
    uv run scripts/03_filter.py
    uv run scripts/03_filter.py --input data/icons_filtered.jsonl \\
                                --output data/icons_training.jsonl
    uv run scripts/03_filter.py --max-paths 3 --svg-pct 98
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


# ── Color classification ──────────────────────────────────────────────────────

_ATTR_COLOR = re.compile(
    r'(?:fill|stroke|stop-color|color)\s*=\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_STYLE_COLOR = re.compile(
    r'(?:fill|stroke|stop-color|color)\s*:\s*([^;"\'\s>]+)',
    re.IGNORECASE,
)
_SKIP_COLORS = frozenset({"currentcolor", "none", "inherit", "transparent", ""})


def _classify(svg: str) -> str:
    vals = _ATTR_COLOR.findall(svg) + _STYLE_COLOR.findall(svg)
    colors = {v.strip().lower() for v in vals if v.strip().lower() not in _SKIP_COLORS}
    if not colors:
        return "monochrome"
    return "single_color" if len(colors) == 1 else "multicolor"


# ── Animation detection ───────────────────────────────────────────────────────

_ANIM_RE = re.compile(r'<animate|<set[\s>/]|@keyframes|animation\s*:', re.IGNORECASE)


def _is_animated(svg: str) -> bool:
    return bool(_ANIM_RE.search(svg))


# ── Color normalisation ───────────────────────────────────────────────────────

def _to_currentcolor(svg: str) -> str:
    """Replace explicit fill/stroke colors with currentColor (skips none/inherit/transparent)."""
    svg = re.sub(
        r'((?:fill|stroke|stop-color|color)\s*=\s*["\'])([^"\']+)(["\'])',
        lambda m: m.group(1) + "currentColor" + m.group(3)
            if m.group(2).strip().lower() not in _SKIP_COLORS else m.group(0),
        svg, flags=re.IGNORECASE,
    )
    svg = re.sub(
        r'((?:fill|stroke|stop-color|color)\s*:\s*)([^;"\'\s>]+)',
        lambda m: m.group(1) + "currentColor"
            if m.group(2).strip().lower() not in _SKIP_COLORS else m.group(0),
        svg, flags=re.IGNORECASE,
    )
    return svg


def _ensure_currentcolor(svg: str) -> str:
    """Inject fill="currentColor" on <svg> root for icons that rely on SVG's implicit black fill."""
    if re.search(r'currentColor', svg, re.IGNORECASE):
        return svg
    return re.sub(r'(<svg\b)', r'\1 fill="currentColor"', svg, count=1, flags=re.IGNORECASE)


# ── SVG normalisation ─────────────────────────────────────────────────────────

def _strip_boilerplate(svg: str) -> str:
    svg = re.sub(r'<\?xml[^?]*\?>', '', svg)
    svg = re.sub(r'<!DOCTYPE[^>]*>', '', svg)
    svg = re.sub(r'<!--.*?-->', '', svg, flags=re.DOTALL)
    return svg


def _strip_metadata(svg: str) -> str:
    for tag in ('title', 'desc', 'metadata'):
        svg = re.sub(fr'<{tag}[^>]*>.*?</{tag}>', '', svg, flags=re.DOTALL | re.IGNORECASE)
        svg = re.sub(fr'<{tag}\s*/>', '', svg, flags=re.IGNORECASE)
    return svg


def _reduce_precision(svg: str, dp: int = 2) -> str:
    return re.sub(r'-?\d+\.\d{3,}', lambda m: f"{float(m.group()):.{dp}f}", svg)


def _collapse_whitespace(svg: str) -> str:
    return re.sub(r'\s+', ' ', svg).strip()


def normalize_svg(svg: str) -> str:
    return _collapse_whitespace(_reduce_precision(_strip_metadata(_strip_boilerplate(svg))))


# ── ViewBox normalisation ─────────────────────────────────────────────────────

_VB_RE = re.compile(r'viewBox=["\']([^"\']+)["\']', re.IGNORECASE)
TARGET_VB = 24


def _get_viewbox(svg: str):
    m = _VB_RE.search(svg)
    if not m:
        return None
    parts = m.group(1).strip().split()
    return tuple(float(p) for p in parts) if len(parts) == 4 else None


def normalize_viewbox(svg: str, target: int = TARGET_VB) -> str:
    vb = _get_viewbox(svg)
    if vb is None:
        return svg
    vx, vy, vw, vh = vb
    if vw == target and vh == target and vx == 0 and vy == 0:
        return svg
    s  = target / max(vw, vh)
    ox = (target - vw * s) / 2 - vx * s
    oy = (target - vh * s) / 2 - vy * s
    transform = f"matrix({s:.6g} 0 0 {s:.6g} {ox:.4g} {oy:.4g})"
    svg = re.sub(
        r'viewBox=["\'][^"\']*["\']',
        f'viewBox="0 0 {target} {target}"', svg, flags=re.IGNORECASE,
    )
    m = re.search(r'(<svg[^>]*>)(.*)(</svg>)', svg, re.DOTALL | re.IGNORECASE)
    if m:
        svg = m.group(1) + f'<g transform="{transform}">' + m.group(2) + '</g>' + m.group(3)
    return svg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input",     type=Path,  default=Path("data/icons_filtered.jsonl"))
    parser.add_argument("--output",    type=Path,  default=Path("data/icons_training.jsonl"))
    parser.add_argument("--max-paths", type=int,   default=5)
    parser.add_argument("--svg-pct",   type=float, default=99.0,
                        help="Upper percentile cap for SVG string length (default: 99)")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"ERROR: {args.input} not found. Run scripts/01_collect.py first.")

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading {args.input} …", flush=True)
    records = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    total = len(records)
    print(f"  Loaded {total:,} icons")

    # ── Step 1: Color classification ──────────────────────────────────────────
    print("\nStep 1  Color classification …", flush=True)
    for r in records:
        if "color_class" not in r:
            r["color_class"] = _classify(r["svg"])

    n_mono   = sum(1 for r in records if r["color_class"] == "monochrome")
    n_single = sum(1 for r in records if r["color_class"] == "single_color")
    n_multi  = sum(1 for r in records if r["color_class"] == "multicolor")
    print(f"  monochrome   : {n_mono:>7,}  ({n_mono/total*100:.1f}%)")
    print(f"  single_color : {n_single:>7,}  ({n_single/total*100:.1f}%)")
    print(f"  multicolor   : {n_multi:>7,}  ({n_multi/total*100:.1f}%)  → excluded")

    # ── Step 2: Animation detection ───────────────────────────────────────────
    print("\nStep 2  Animation detection …", flush=True)
    for r in records:
        if "is_animated" not in r:
            r["is_animated"] = _is_animated(r["svg"])
    n_anim = sum(1 for r in records if r["is_animated"])
    print(f"  animated: {n_anim:,}  ({n_anim/total*100:.2f}%)  → excluded")

    # ── Step 3: Filter — color + animation ────────────────────────────────────
    print("\nStep 3  Keeping monochrome + single-color, non-animated …", flush=True)
    candidates = [
        r for r in records
        if r["color_class"] in ("monochrome", "single_color") and not r["is_animated"]
    ]
    print(f"  {total:,} → {len(candidates):,}  (removed {total - len(candidates):,})")

    # ── Step 4: Recolor single-color → currentColor + inject fill on root ─────
    print("\nStep 4  Recoloring → currentColor + injecting fill on root …", flush=True)
    result = []
    n_recolored = 0
    n_injected  = 0
    for r in candidates:
        svg = r["svg"]
        if r["color_class"] == "single_color":
            svg = _to_currentcolor(svg)
            n_recolored += 1
        orig_len = len(svg)
        svg = _ensure_currentcolor(svg)
        if len(svg) != orig_len:
            n_injected += 1
        result.append({**r, "svg": svg, "color_class": "monochrome"})
    print(f"  Recolored single-color fills: {n_recolored:,}")
    print(f"  Injected fill=currentColor on root: {n_injected:,}")

    # ── Step 5: Path count ≤ max-paths ────────────────────────────────────────
    print(f"\nStep 5  Keeping path_count ≤ {args.max_paths} …", flush=True)
    before = len(result)
    result = [r for r in result if r["path_count"] <= args.max_paths]
    print(f"  {before:,} → {len(result):,}  (removed {before - len(result):,})")

    # ── Step 6: SVG length cap at Nth percentile ──────────────────────────────
    print(f"\nStep 6  SVG length cap at {args.svg_pct}th percentile …", flush=True)
    svg_lengths = np.array([len(r["svg"]) for r in result])
    svg_cap = int(np.percentile(svg_lengths, args.svg_pct))
    print(f"  {args.svg_pct}th percentile = {svg_cap:,} chars")
    before = len(result)
    result = [r for r in result if len(r["svg"]) <= svg_cap]
    print(f"  {before:,} → {len(result):,}  (removed {before - len(result):,})")

    # ── Step 7: SVG normalisation ─────────────────────────────────────────────
    print("\nStep 7  Normalising SVG (boilerplate, metadata, precision, whitespace) …", flush=True)
    result = [{**r, "svg": normalize_svg(r["svg"])} for r in result]
    lens = [len(r["svg"]) for r in result]
    print(f"  mean {sum(lens)/len(lens):.0f} chars  median {sorted(lens)[len(lens)//2]:,} chars  "
          f"max {max(lens):,} chars")

    # ── Step 8: ViewBox normalisation to 24×24 ────────────────────────────────
    print(f"\nStep 8  Normalising viewBox to {TARGET_VB}×{TARGET_VB} …", flush=True)
    needs_scale = sum(
        1 for r in result
        if _get_viewbox(r["svg"]) != (0.0, 0.0, float(TARGET_VB), float(TARGET_VB))
    )
    print(f"  Icons needing rescaling: {needs_scale:,}")
    result = [{**r, "svg": normalize_viewbox(r["svg"])} for r in result]

    # ── Write output ──────────────────────────────────────────────────────────
    print(f"\nWriting {len(result):,} icons to {args.output} …", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in result:
            out = {
                "icon_id":    r["icon_id"],
                "icon_name":  r["icon_id"].split(":", 1)[-1],
                "svg":        r["svg"],
                "path_count": r["path_count"],
                "color_class": r["color_class"],
                "svg_len":    len(r["svg"]),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Input   : {total:,} icons  ({args.input})")
    print(f"  Output  : {len(result):,} icons  ({args.output})  ({len(result)/total*100:.1f}% retained)")
    print(f"\n  Path-count breakdown:")
    for n in range(args.max_paths + 1):
        n_icons = sum(1 for r in result if r["path_count"] == n)
        if n_icons:
            print(f"    {n} path{'s' if n != 1 else ''}: {n_icons:>7,}  ({n_icons/len(result)*100:.1f}%)")
    print(f"\nNext: merge {args.output} with data/icons_captioned.jsonl (once captioning completes),")
    print(f"      then run  uv run scripts/04_prepare.py  (rename 03_prepare.py → 04_prepare.py)")


if __name__ == "__main__":
    main()
