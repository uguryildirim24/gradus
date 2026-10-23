"""Run leakage baselines (and optionally swap invariance) over a rows table.

uv run python scripts/leakage_report.py --rows data/built/rows.parquet \
    --splits data/built/splits.csv [--model pkg.mod:func] [--out data/built/leakage_report.md]

rows: columns resume, posting, label, optional id and profile. splits: columns id, split.
If --splits is omitted, rows must carry a 'split' column.
--model names a callable model(resume, posting) -> bucket for the swap test,
run on dev rows.
"""
import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from matcher import leakage as L  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--splits")
    ap.add_argument("--train-name", default="train")
    ap.add_argument("--dev-name", default="dev")
    ap.add_argument("--model")
    ap.add_argument("--out", default="data/built/leakage_report.md")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    df = L.load_table(a.rows)
    if a.splits:
        splits = L.load_splits(a.splits)
    elif "split" in df.columns:
        splits = dict(zip(df[L.ID], df["split"].astype(str)))
    else:
        ap.error("need --splits or a 'split' column in rows")
    train, dev = L.split_rows(df, splits, a.train_name, a.dev_name)
    if not len(train) or not len(dev):
        ap.error(f"empty split: train={len(train)} dev={len(dev)}")

    md = L.report_markdown(L.run_baselines(train, dev, a.seed), len(train), len(dev), dev[L.LABEL])
    if a.model:
        mod, fn = a.model.split(":")
        model = getattr(importlib.import_module(mod), fn)
        md += "\n" + L.swap_markdown(L.swap_invariance(model, list(zip(dev[L.RESUME], dev[L.POSTING]))))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
