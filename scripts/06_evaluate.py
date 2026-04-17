"""
Step 6 — Evaluate generated SVGs on five metrics.

Reads results/generated.jsonl (from 05_generate.py) and computes:

  VR    — SVG Validity Rate        fraction that parse as well-formed XML
                                    with a root <svg> element
  RSR   — Rendering Success Rate   fraction of valid SVGs that render via
                                    CairoSVG without error
  MPC   — Mean Path Count          average <path> elements in valid SVGs
                                    (lower → more economical structure)
  CLIP  — CLIP cosine similarity   prompt text ↔ rendered-PNG embedding
                                    via ViT-B/32  [requires: open-clip-torch torch]
  FID   — Fréchet Inception Dist.  generated vs. reference renders
                                    [requires: cleanfid torch]

Basic metrics (VR, RSR, MPC) need only cairosvg (already in deps).
CLIP and FID require extra packages; install with:
    pip install torch open-clip-torch cleanfid

Usage:
    uv run scripts/06_evaluate.py
    uv run scripts/06_evaluate.py --input results/generated.jsonl
    uv run scripts/06_evaluate.py --skip-clip --skip-fid   # fast / no PyTorch
    uv run scripts/06_evaluate.py --device mps             # Apple GPU for CLIP/FID
"""

import argparse
import io
import json
import re
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

from tqdm import tqdm

SVG_NS = "http://www.w3.org/2000/svg"


# ─── SVG helpers ─────────────────────────────────────────────────────────────

def extract_svg(text: str) -> str:
    """Extract the first <svg …>…</svg> block (strips markdown fences etc.)."""
    m = re.search(r"(<svg[\s\S]*?</svg>)", text, re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def is_valid_xml(svg: str) -> bool:
    """True if svg parses as well-formed XML whose root tag contains 'svg'."""
    try:
        root = ET.fromstring(svg)
        return "svg" in root.tag.lower()
    except ET.ParseError:
        return False


def can_render(svg: str) -> bool:
    """True if CairoSVG renders the SVG without raising an exception."""
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg.encode(), output_width=64, output_height=64)
        return True
    except Exception:
        return False


def count_paths(svg: str) -> int:
    """Count <path> elements (handles both bare and namespace-prefixed tags)."""
    try:
        root = ET.fromstring(svg)
        return len(root.findall(".//path")) + len(root.findall(f".//{{{SVG_NS}}}path"))
    except Exception:
        return 0


def render_png(svg: str, size: int = 224) -> bytes | None:
    """Render SVG to PNG bytes at the given square size; returns None on error."""
    try:
        import cairosvg
        return cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size)
    except Exception:
        return None


# ─── CLIP score ──────────────────────────────────────────────────────────────

def compute_clip_scores(records: list[dict], device: str = "cpu") -> list[float | None]:
    """
    Return per-record CLIP cosine similarities.
    Returns a list of None values (silently) when open-clip-torch is missing.
    """
    try:
        import open_clip
        import torch
        from PIL import Image
    except ImportError:
        print("  [CLIP] open-clip-torch not found — skipping.")
        print("         Install: pip install torch open-clip-torch")
        return [None] * len(records)

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    scores: list[float | None] = []
    with torch.no_grad():
        for rec in tqdm(records, desc="  CLIP "):
            svg = extract_svg(rec["generated_svg"])
            png = render_png(svg, size=224)
            if png is None:
                scores.append(None)
                continue

            img = (
                preprocess(Image.open(io.BytesIO(png)).convert("RGB"))
                .unsqueeze(0)
                .to(device)
            )
            text = tokenizer([rec["prompt"]]).to(device)

            img_feat  = model.encode_image(img)
            text_feat = model.encode_text(text)
            img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            scores.append(float((img_feat * text_feat).sum()))

    return scores


# ─── FID ─────────────────────────────────────────────────────────────────────

def compute_fid(records: list[dict], tmp_root: Path, device: str = "cpu") -> float | None:
    """
    Render generated and reference SVGs to PNG, then compute FID.
    Returns None silently when cleanfid / torch are not installed.
    """
    try:
        from cleanfid import fid as cleanfid
    except ImportError:
        print("  [FID]  cleanfid not found — skipping.")
        print("         Install: pip install torch cleanfid")
        return None

    gen_dir = tmp_root / "gen"
    ref_dir = tmp_root / "ref"
    gen_dir.mkdir(parents=True)
    ref_dir.mkdir(parents=True)

    for i, rec in enumerate(tqdm(records, desc="  FID render")):
        for svg, out_dir in [
            (extract_svg(rec["generated_svg"]), gen_dir),
            (rec["reference_svg"],              ref_dir),
        ]:
            png = render_png(svg, size=299)  # Inception v3 input size
            if png:
                (out_dir / f"{i:06d}.png").write_bytes(png)

    return float(cleanfid.compute_fid(str(gen_dir), str(ref_dir), device=device))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input",     type=Path, default=Path("results/generated.jsonl"))
    parser.add_argument("--skip-clip", action="store_true", help="Skip CLIP score")
    parser.add_argument("--skip-fid",  action="store_true", help="Skip FID")
    parser.add_argument("--device",    default="cpu",
                        help="PyTorch device for CLIP/FID: cpu | mps | cuda")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"ERROR: {args.input} not found. Run scripts/05_generate.py first.")

    with open(args.input) as f:
        records = [json.loads(line) for line in f if line.strip()]
    n = len(records)
    print(f"Loaded {n:,} generated SVGs from {args.input}\n")

    # ── Extract SVG from raw generated text (handles markdown fences etc.) ───
    svgs = [extract_svg(r["generated_svg"]) for r in records]

    # ── 1. VR — XML validity ─────────────────────────────────────────────────
    print("Computing VR …")
    valid_mask = [is_valid_xml(s) for s in tqdm(svgs, desc="  VR  ")]
    n_valid = sum(valid_mask)
    vr = n_valid / n * 100

    # ── 2. RSR — CairoSVG rendering ──────────────────────────────────────────
    print("Computing RSR …")
    render_mask = [
        can_render(s) if v else False
        for s, v in tqdm(zip(svgs, valid_mask), total=n, desc="  RSR ")
    ]
    n_render = sum(render_mask)
    rsr = n_render / n * 100

    # ── 3. MPC — mean path count (over valid SVGs only) ──────────────────────
    path_counts = [count_paths(s) if v else 0 for s, v in zip(svgs, valid_mask)]
    mpc = sum(path_counts) / max(n_valid, 1)

    # ── 4. CLIP score ────────────────────────────────────────────────────────
    if not args.skip_clip:
        print("Computing CLIP score …")
    clip_scores = [] if args.skip_clip else compute_clip_scores(records, device=args.device)
    valid_clips = [s for s in clip_scores if s is not None]
    clip_mean = sum(valid_clips) / len(valid_clips) if valid_clips else None

    # ── 5. FID ───────────────────────────────────────────────────────────────
    fid_score = None
    if not args.skip_fid:
        print("Computing FID …")
        with tempfile.TemporaryDirectory() as tmp:
            fid_score = compute_fid(records, Path(tmp), device=args.device)

    # ── Report ───────────────────────────────────────────────────────────────
    sep = "=" * 52
    print(f"\n{sep}")
    print("  EVALUATION RESULTS")
    print(sep)
    print(f"  N evaluated                     : {n:,}")
    print(f"  SVG Validity Rate  (VR)    ↑    : {vr:.1f}%  ({n_valid}/{n})")
    print(f"  Rendering Success  (RSR)   ↑    : {rsr:.1f}%  ({n_render}/{n})")
    print(f"  Mean Path Count    (MPC)   ↓    : {mpc:.2f}")
    if clip_mean is not None:
        print(f"  CLIP Score                 ↑    : {clip_mean:.4f}  (n={len(valid_clips)})")
    else:
        print(f"  CLIP Score                      : skipped")
    if fid_score is not None:
        print(f"  FID                        ↓    : {fid_score:.2f}")
    else:
        print(f"  FID                             : skipped")
    print(sep)

    # ── Save JSON for copy-pasting into the paper tables ─────────────────────
    result = {
        "n":       n,
        "vr_pct":  round(vr, 2),
        "rsr_pct": round(rsr, 2),
        "mpc":     round(mpc, 3),
        "clip":    round(clip_mean, 4) if clip_mean is not None else None,
        "fid":     round(fid_score, 2) if fid_score is not None else None,
    }
    out_path = args.input.with_suffix("").with_suffix(".metrics.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
