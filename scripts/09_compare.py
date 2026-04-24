"""
Step 9 — Aggregate per-model metrics into a comparison table.

Each baseline should first be evaluated with 06_evaluate.py, producing a
.metrics.json file. This script reads those files and prints a formatted
table suitable for copy-paste into the paper.

Usage:
    uv run scripts/07_compare.py \\
        "Qwen3.5-9B (zero-shot):results/qwen_zeroshot.metrics.json" \\
        "GPT-5.4 (zero-shot):results/gpt_zeroshot.metrics.json" \\
        "StarVector-8B:results/starvector.metrics.json" \\
        "OmniSVG:results/omnisvg.metrics.json" \\
        "IconGenAI (ours):results/generated.metrics.json"

    # LaTeX output:
    uv run scripts/07_compare.py --latex <same args>
"""

import argparse
import json
from pathlib import Path

METRICS = [
    ("vr",   "VR (%)",   True,  lambda v: f"{v*100:.1f}"),
    ("rsr",  "RSR (%)",  True,  lambda v: f"{v*100:.1f}"),
    ("clip", "CLIP ↑",   True,  lambda v: f"{v:.3f}"),
    ("mpc",  "MPC ↓",    False, lambda v: f"{v:.1f}"),
    ("fid",  "FID ↓",    False, lambda v: f"{v:.1f}"),
]


def load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def best_indices(rows: list[dict]) -> dict[str, int]:
    """Return {metric_key: row_index_of_best_value}."""
    best = {}
    for key, _, higher_is_better, _ in METRICS:
        values = [(i, r[key]) for i, r in enumerate(rows) if key in r and r[key] is not None]
        if not values:
            continue
        best[key] = max(values, key=lambda x: x[1] if higher_is_better else -x[1])[0]
    return best


def print_table(models: list[tuple[str, dict]]) -> None:
    rows = [data for _, data in models]
    best = best_indices(rows)

    col_w = max(len(name) for name, _ in models) + 2
    headers = ["Model"] + [label for _, label, _, _ in METRICS]
    widths = [col_w] + [max(len(h), 8) for h in headers[1:]]

    sep = "  ".join("-" * w for w in widths)
    header = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header)
    print(sep)

    for i, (name, data) in enumerate(models):
        cells = [name.ljust(widths[0])]
        for j, (key, _, _, fmt) in enumerate(METRICS):
            val = data.get(key)
            cell = fmt(val) if val is not None else "—"
            if best.get(key) == i:
                cell = f"{cell}"
            cells.append(cell.ljust(widths[j + 1]))
        print("  ".join(cells))


def print_latex(models: list[tuple[str, dict]]) -> None:
    rows = [data for _, data in models]
    best = best_indices(rows)

    print(r"\begin{tabular}{lrccccc}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Params} & \textbf{VR (\%)} & "
          r"\textbf{RSR (\%)} & \textbf{CLIP $\uparrow$} & "
          r"\textbf{MPC $\downarrow$} & \textbf{FID $\downarrow$} \\")
    print(r"\midrule")

    for i, (name, data) in enumerate(models):
        cells = [name, data.get("params", r"\todo{?}")]
        for j, (key, _, _, fmt) in enumerate(METRICS):
            val = data.get(key)
            cell = fmt(val) if val is not None else r"\todo{}"
            if best.get(key) == i and val is not None:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        print(" & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "entries", nargs="+",
        metavar="NAME:PATH",
        help="model label and path to its .metrics.json, separated by ':'",
    )
    parser.add_argument("--latex", action="store_true", help="Output LaTeX tabular")
    args = parser.parse_args()

    models = []
    for entry in args.entries:
        name, _, path = entry.partition(":")
        if not path:
            parser.error(f"Expected NAME:PATH, got: {entry!r}")
        data = load(path)
        if not data:
            print(f"Warning: {path} not found or empty — showing placeholder row")
        models.append((name.strip(), data))

    if args.latex:
        print_latex(models)
    else:
        print_table(models)


if __name__ == "__main__":
    main()
