"""
Step 5 — Batch-generate SVG icons from test-set prompts.

Reads data/test.jsonl (ChatML format from 03_prepare.py), generates one SVG
per prompt using the fine-tuned model + LoRA adapter, and writes results to
results/generated.jsonl for downstream evaluation by 06_evaluate.py.

Supports resume: if the output file already exists, already-processed rows
are skipped so you can safely restart a partial run.

Usage:
    uv run scripts/05_generate.py --adapter models/icongenai-lora
    uv run scripts/05_generate.py --adapter models/icongenai-lora --n 200
    uv run scripts/05_generate.py --no-adapter          # zero-shot baseline
    uv run scripts/05_generate.py --adapter models/icongenai-lora \\
        --output results/generated_zeroshot.jsonl       # custom output path
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are an SVG icon generator. "
    "Given a description, output clean, valid SVG code for a single icon. "
    "Use currentColor for fill and stroke so the icon inherits its parent's color. "
    "Output only the SVG element — no explanation, no markdown fences."
)

DEFAULT_MODEL   = "mlx-community/Qwen3.5-9B-MLX-4bit"
DEFAULT_TEST    = Path("data/test.jsonl")
DEFAULT_OUTPUT  = Path("results/generated.jsonl")
MAX_TOKENS      = 1024


def load_test_records(path: Path) -> list[dict]:
    """Load test.jsonl; return list of {prompt, reference_svg}."""
    records = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            messages = rec["messages"]
            user_msg = next(m["content"] for m in messages if m["role"] == "user")
            asst_msg = next(m["content"] for m in messages if m["role"] == "assistant")
            records.append({"prompt": user_msg, "reference_svg": asst_msg})
    return records


def count_done(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--adapter",    default=None,
                        help="Path to LoRA adapter directory.")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Zero-shot run — do not load any adapter.")
    parser.add_argument("--test",       type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output",     type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n",          type=int,  default=None,
                        help="Limit to first N prompts (quick sanity check).")
    parser.add_argument("--max-tokens", type=int,  default=MAX_TOKENS)
    parser.add_argument("--temp",       type=float, default=0.0,
                        help="Sampling temperature (0.0 = greedy).")
    args = parser.parse_args()

    adapter_path = None if args.no_adapter else args.adapter

    if not args.test.exists():
        sys.exit(f"ERROR: {args.test} not found. Run scripts/03_prepare.py first.")

    if adapter_path and not Path(adapter_path).exists():
        sys.exit(f"ERROR: adapter path not found: {adapter_path}")

    # --- Load model (deferred import so errors above are fast) ---------------
    print(f"Loading model : {args.model}")
    if adapter_path:
        print(f"       adapter: {adapter_path}")
    else:
        print("       adapter: none (zero-shot)")

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(args.model, adapter_path=adapter_path)
    print("Model loaded.\n")

    # --- Load test records ---------------------------------------------------
    records = load_test_records(args.test)
    if args.n:
        records = records[: args.n]
    print(f"Prompts total : {len(records):,}")

    # --- Resume support ------------------------------------------------------
    done = count_done(args.output)
    if done:
        print(f"Resuming      : {done:,} already written, skipping.")
    if done >= len(records):
        print("All prompts already processed.")
        return

    # --- Generate ------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "a") as out_f:
        for i, rec in enumerate(tqdm(records, desc="Generating")):
            if i < done:
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": rec["prompt"]},
            ]
            prompt_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            generated = generate(
                model,
                tokenizer,
                prompt=prompt_str,
                max_tokens=args.max_tokens,
                sampler=make_sampler(temp=args.temp),
                verbose=False,
            )

            out_f.write(
                json.dumps(
                    {
                        "prompt":        rec["prompt"],
                        "generated_svg": generated.strip(),
                        "reference_svg": rec["reference_svg"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out_f.flush()

    print(f"\nDone. Results → {args.output}")
    print("Next step: uv run scripts/06_evaluate.py")


if __name__ == "__main__":
    main()
