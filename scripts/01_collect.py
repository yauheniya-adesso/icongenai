"""
Step 1 — Collect and filter the Iconify dataset.

Reads all icon sets from icon-sets/json/, filters by license,
skips aliases and hidden icons, assembles full SVG strings,
and writes one JSON record per icon to data/icons_filtered.jsonl.

Usage:
    uv run scripts/01_collect.py
    uv run scripts/01_collect.py --icon-sets-dir /path/to/icon-sets

License policy (matches research/research.md §7):
  INCLUDE  — MIT, Apache-2.0, OFL-1.1, CC0-1.0, ISC, BSD-3-Clause,
             Unlicense, MPL-2.0, CC-BY-4.0, CC-BY-3.0, CC-BY-NC-4.0
  EXCLUDE  — CC-BY-SA-* (ShareAlike), CC-BY-NC-SA-*, GPL-*
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# License policy
# ---------------------------------------------------------------------------

ALLOWED_LICENSES = {
    "MIT",
    "Apache-2.0",
    "OFL-1.1",          # SIL Open Font License — permissive
    "CC0-1.0",          # Public domain
    "ISC",              # Functionally identical to MIT
    "BSD-3-Clause",
    "Unlicense",        # Public domain equivalent
    "MPL-2.0",          # Mozilla Public License — permissive for use
    "CC-BY-4.0",
    "CC-BY-3.0",
    "CC-BY-NC-4.0",     # Non-commercial — ok for research
}

EXCLUDED_LICENSES = {
    "CC-BY-SA-4.0",     # ShareAlike — would require model weights under same license
    "CC-BY-SA-3.0",
    "CC-BY-NC-SA-4.0",  # ShareAlike + non-commercial
    "GPL-2.0-only",
    "GPL-2.0-or-later",
    "GPL-3.0",
    "GPL-3.0-or-later",
}


def license_decision(spdx: str) -> str:
    """Returns 'allow', 'exclude', or 'unknown'."""
    if spdx in ALLOWED_LICENSES:
        return "allow"
    if spdx in EXCLUDED_LICENSES:
        return "exclude"
    return "unknown"


# ---------------------------------------------------------------------------
# SVG assembly
# ---------------------------------------------------------------------------

def assemble_svg(body: str, width: int, height: int) -> str:
    """Wrap an Iconify icon body in a minimal, valid SVG document."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">'
        f'{body}'
        f'</svg>'
    )


def count_paths(body: str) -> int:
    """Fast path count — avoids a full XML parse."""
    return body.count("<path")


def is_multicolor(body: str) -> bool:
    """True if the icon body contains hardcoded color values (not currentColor)."""
    body_lower = body.lower()
    # Exclude currentColor references, look for actual hex/rgb/named colors
    no_current = body_lower.replace("currentcolor", "")
    return (
        "fill=" in no_current
        and any(
            token in no_current
            for token in ["#", "rgb(", "rgba(", 'fill="white"', 'fill="black"']
        )
    )


# ---------------------------------------------------------------------------
# Main collection logic
# ---------------------------------------------------------------------------

def collect(icon_sets_dir: Path, output_path: Path) -> None:
    collections_file = icon_sets_dir / "collections.json"
    json_dir = icon_sets_dir / "json"

    if not collections_file.exists():
        sys.exit(f"ERROR: collections.json not found at {collections_file}")
    if not json_dir.is_dir():
        sys.exit(f"ERROR: json/ directory not found at {json_dir}")

    with open(collections_file) as f:
        collections: dict = json.load(f)

    # --- Partition collections by license ---
    allowed_collections: list[tuple[str, dict]] = []
    excluded_collections: list[tuple[str, str]] = []   # (prefix, spdx)
    unknown_collections: list[tuple[str, str]] = []

    for prefix, meta in collections.items():
        spdx = meta.get("license", {}).get("spdx", "UNKNOWN")
        decision = license_decision(spdx)
        if decision == "allow":
            allowed_collections.append((prefix, meta))
        elif decision == "exclude":
            excluded_collections.append((prefix, spdx))
        else:
            unknown_collections.append((prefix, spdx))

    # Print pre-run summary
    print("\n=== License filter summary ===")
    print(f"  Allowed collections : {len(allowed_collections)}")
    print(f"  Excluded collections: {len(excluded_collections)}")
    for prefix, spdx in excluded_collections:
        total = collections[prefix].get("total", "?")
        print(f"    - {prefix:<30} {spdx}  ({total} icons)")
    if unknown_collections:
        print(f"  Unknown licenses    : {len(unknown_collections)}  (also excluded, review manually)")
        for prefix, spdx in unknown_collections:
            total = collections[prefix].get("total", "?")
            print(f"    ? {prefix:<30} {spdx}  ({total} icons)")
    print()

    # --- Process each allowed collection ---
    stats = {
        "total_written": 0,
        "skipped_alias": 0,
        "skipped_hidden": 0,
        "skipped_missing_body": 0,
        "missing_json_files": [],
        "by_license": {},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out:
        for prefix, col_meta in tqdm(allowed_collections, desc="Collections", unit="set"):
            json_file = json_dir / f"{prefix}.json"
            if not json_file.exists():
                stats["missing_json_files"].append(prefix)
                continue

            with open(json_file) as f:
                icon_set: dict = json.load(f)

            # Collection-level defaults
            default_width = icon_set.get("width", 24)
            default_height = icon_set.get("height", 24)
            collection_name = col_meta.get("name", prefix)
            license_spdx = col_meta.get("license", {}).get("spdx", "")
            license_url = col_meta.get("license", {}).get("url", "")
            author = col_meta.get("author", {}).get("name", "")

            # Track per-license counts
            stats["by_license"].setdefault(license_spdx, 0)

            icons: dict = icon_set.get("icons", {})
            aliases: set = set(icon_set.get("aliases", {}).keys())

            for icon_name, icon_data in icons.items():
                # Skip aliases (they duplicate a parent icon)
                if icon_name in aliases:
                    stats["skipped_alias"] += 1
                    continue

                # Skip hidden icons
                if icon_data.get("hidden", False):
                    stats["skipped_hidden"] += 1
                    continue

                body = icon_data.get("body", "")
                if not body:
                    stats["skipped_missing_body"] += 1
                    continue

                width = icon_data.get("width", default_width)
                height = icon_data.get("height", default_height)
                svg = assemble_svg(body, width, height)

                record = {
                    "icon_id": f"{prefix}:{icon_name}",
                    "collection_prefix": prefix,
                    "collection_name": collection_name,
                    "license_spdx": license_spdx,
                    "license_url": license_url,
                    "author": author,
                    "icon_name": icon_name,
                    "svg": svg,
                    "width": width,
                    "height": height,
                    "path_count": count_paths(body),
                    "is_multicolor": is_multicolor(body),
                }

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["total_written"] += 1
                stats["by_license"][license_spdx] += 1

    # --- Final report ---
    print(f"\n=== Collection complete ===")
    print(f"  Output file      : {output_path}")
    print(f"  Icons written    : {stats['total_written']:,}")
    print(f"  Skipped (alias)  : {stats['skipped_alias']:,}")
    print(f"  Skipped (hidden) : {stats['skipped_hidden']:,}")
    print(f"  Skipped (no body): {stats['skipped_missing_body']:,}")
    if stats["missing_json_files"]:
        print(f"  Missing JSON     : {stats['missing_json_files']}")
    print("\n  Icons by license:")
    for spdx, count in sorted(stats["by_license"].items(), key=lambda x: -x[1]):
        print(f"    {spdx:<25} {count:>7,}")

    # Save stats as a companion JSON for reference
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved to   : {stats_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--icon-sets-dir",
        type=Path,
        default=Path(__file__).parent.parent / "icon-sets",
        help="Path to the cloned iconify/icon-sets repo (default: ../icon-sets)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "icons_filtered.jsonl",
        help="Output JSONL file path (default: ../data/icons_filtered.jsonl)",
    )
    args = parser.parse_args()

    print(f"Icon-sets dir : {args.icon_sets_dir}")
    print(f"Output path   : {args.output}")
    collect(args.icon_sets_dir, args.output)


if __name__ == "__main__":
    main()
