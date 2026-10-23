"""Resume ids to render for build_real_train's rows: one split, optionally capped per posting.

  real_train_render_ids.py SPLIT OUT_FILE [--per-posting K] [--seed 0]

Reads data/built/real_train/rows.jsonl (build_real_train.py build) and writes the candidate card ids of SPLIT's rows,
one per line, in row order, for `render_cards.py resumes estimate|run OUT_FILE`. --per-posting K keeps at most K rows a
posting with build_real_train.cap_rows (round-robin over variant kinds); 0 keeps every row. Prints the rows, unique
resume contexts and bucket mix of the pick. Nothing here calls Adaption.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_real_train import OUT, cap_rows, read_jsonl  # noqa: E402


def pick(rows: list[dict], split: str, k: int, seed: int = 0) -> list[dict]:
    return cap_rows([r for r in rows if r["split"] == split], k, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=("train", "dev"))
    ap.add_argument("out")
    ap.add_argument("--per-posting", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = pick(read_jsonl(OUT / "rows.jsonl"), a.split, a.per_posting, a.seed)
    ids = list(dict.fromkeys(r["cd_id"] for r in rows))
    Path(a.out).write_text("\n".join(ids) + "\n")
    mix = Counter(r["bucket"] for r in rows)
    print(json.dumps({"split": a.split, "per_posting": a.per_posting, "rows": len(rows), "ids": len(ids),
                      "unique_contexts": len({r["ctx"] for r in rows}), "postings": len({r["rq_id"] for r in rows}),
                      "buckets": dict(mix), "bucket_share": {b: round(n / len(rows), 3) for b, n in mix.items()}}))


if __name__ == "__main__":
    main()
