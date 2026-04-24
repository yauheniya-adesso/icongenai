"""
04_merge.py — Join captions from icons_captioned_merged.jsonl into icons_training.jsonl.

Input:
  data/icons_captioned_merged.jsonl  (275,912 icons, has caption_short / caption_long)
  data/icons_training.jsonl          (216,209 icons, no captions)

Output:
  data/icons_training_captioned.jsonl  — all training rows; captioned ones get
  caption_short / caption_long added, uncaptioned rows are kept as-is (left join).

Join key: icon_id
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
CAPTIONS_PATH = ROOT / "data" / "icons_captioned_merged.jsonl"
TRAINING_PATH = ROOT / "data" / "icons_training.jsonl"
OUTPUT_PATH   = ROOT / "data" / "icons_training_captioned.jsonl"


def main() -> None:
    print("Loading captions …")
    captions: dict[str, dict] = {}
    with CAPTIONS_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            icon_id = rec.get("icon_id")
            if icon_id and ("caption_short" in rec or "caption_long" in rec):
                captions[icon_id] = {
                    k: rec[k] for k in ("caption_short", "caption_long") if k in rec
                }
    print(f"  {len(captions):,} icons with captions")

    matched = no_caption = 0
    with TRAINING_PATH.open() as fin, OUTPUT_PATH.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            cap = captions.get(rec.get("icon_id", ""))
            if cap is not None:
                rec.update(cap)
                matched += 1
            else:
                no_caption += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = matched + no_caption
    print(f"With captions    : {matched:,} / {total:,}  ({matched/total*100:.1f}%)")
    print(f"Without captions : {no_caption:,} (kept, icon_name used as fallback)")
    print(f"Output           : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
