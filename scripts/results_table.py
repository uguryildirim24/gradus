"""One results table from evaluator output files, with the rules baseline on the same rows. numpy only, no model.

  results_table.py [TAG=RESULT.json ...] [--out FILE.md] [--json FILE.json] [--draws 1000] [--paired "A vs B" ...]

Each RESULT.json is a local_eval.py --out file (colab_eval.py names them <tag>_<set>.json, e.g. 4b_test.json,
base_dev_cross.json); the tag defaults to the file stem's part before the set name. Results are grouped by the
set they scored (the result's "dev" file name): test (the real-posting test), dev_cross, dev. With no result
files it prints the rules baseline alone.

Rows. Every set is re-joined to its local file by pair_id, so the decidable mark comes from the current file,
not from the result: on the real test only rows data/built/real_test/test.jsonl marks decidable are scored (a
result scored before a row was marked is still filtered), on dev and dev_cross the evaluator's own visible flag
(missing facts) is used. The rows of a set are those every entry in it scored; a result that misses rows is
reported, and the table is on the common rows.

Per set: accuracy and macro-F1 (95% interval resampling postings), Keep recall and unsafe Keep (gold Skip or
Explore predicted Keep), recall per gold bucket (Keep / Explore / Skip), F1 per label, accuracy per variant_kind
(dev_cross: per candidate_kind) with enrollment_graduated and enrollment_level as named slices. On the real test
the rules baseline (rules.label on Adaption's extracted cards, baselines.py extract-rules) and the majority class
are entries on the same rows.

Paired differences (--paired "A vs B", repeatable, names as in the table): A minus B on accuracy and macro-F1, with a
95% interval from resampling postings once per draw for both systems, so the two are scored on the same rows each
draw. Written per set where both names are entries, e.g. --paired "4b-b vs rules on extracted cards".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from matcher.evalstats import LABELS, WORD, base_of, cluster_ci, confusion, macro_f1_of, prf  # noqa: E402

SETS = {"test": ROOT / "data/built/real_test/test.jsonl", "dev_cross": ROOT / "data/built/training/dev_cross.jsonl",
        "dev": ROOT / "data/built/training/dev.jsonl"}
ORDER = ("test", "dev_cross", "dev")
NAMES = {"test": "Real postings (test.jsonl)", "dev_cross": "Re-paired dev (dev_cross.jsonl)",
         "dev": "Synthetic dev (dev.jsonl)"}
SLICES = ("enrollment_graduated", "enrollment_level")


def read_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def set_of(result: dict, path: Path) -> str:
    """The set a result scored: its dev file's stem (test, dev, dev_cross), else the end of the file name."""
    stem = Path(str(result.get("dev", ""))).stem
    if stem in SETS:
        return stem
    for s in sorted(SETS, key=len, reverse=True):
        if path.stem.endswith("_" + s):
            return s
    raise SystemExit(f"{path}: cannot tell which set it scored (dev={result.get('dev')!r})")


def tag_of(path: Path, s: str) -> str:
    return path.stem.removesuffix("_" + s) or path.stem


def kind_of(row: dict) -> str:
    return row.get("variant_kind") or row.get("candidate_kind") or "-"


def source_rows(s: str, path: Path | None = None) -> dict[str, dict]:
    """pair_id -> the set's own row (bucket, kinds, decidable), prompt dropped."""
    return {r["pair_id"]: {k: v for k, v in r.items() if k != "prompt"} for r in read_jsonl(path or SETS[s])}


def decidable(s: str, src: dict, res_row: dict | None) -> bool:
    if s == "test":
        return bool(src.get("decidable", True))
    if "decidable" in src:
        return bool(src["decidable"])
    return bool(res_row.get("visible", True)) if res_row else True


def evaluate(rows: list[dict], pred: dict[str, str], draws: int) -> dict:
    gold = np.array([LABELS.index(WORD[r["bucket"]]) for r in rows])
    p = np.array([LABELS.index(pred[r["pair_id"]]) for r in rows])
    groups = [base_of(r) for r in rows]
    acc = cluster_ci(groups, lambda i: float((gold[i] == p[i]).mean()), draws)
    f1 = cluster_ci(groups, lambda i: macro_f1_of(gold[i], p[i]), draws)
    conf = confusion(gold, p)
    out = {"n": len(rows), "postings": len(set(groups)),
           "accuracy": [round(x, 4) for x in acc], "macro_f1": [round(x, 4) for x in f1],
           "unsafe_keep": int(((gold != 0) & (p == 0)).sum()),
           "recall_by_bucket": {LABELS[g]: round(float((p[gold == g] == g).mean()), 4) if (gold == g).any() else None
                                for g in range(3)},
           "n_by_bucket": {LABELS[g]: int((gold == g).sum()) for g in range(3)},
           "f1_by_label": {w: round(v["f1"], 4) for w, v in prf(conf).items()},
           "confusion": {LABELS[g]: {LABELS[q]: int(conf[g, q]) for q in range(3)} for g in range(3)},
           "pred_counts": {w: int((p == i).sum()) for i, w in enumerate(LABELS)}}
    kinds: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        kinds.setdefault(kind_of(r), []).append(i)
    out["by_kind"] = {k: {"n": len(ix), "accuracy": round(float((gold[ix] == p[ix]).mean()), 4)}
                      for k, ix in sorted(kinds.items())}
    out["slices"] = {k: out["by_kind"].get(k, {"n": 0, "accuracy": None}) for k in SLICES}
    return out


def paired(rows: list[dict], pa: dict[str, str], pb: dict[str, str], draws: int) -> dict:
    """A minus B on the same rows: point and 95% interval, postings resampled once per draw for both."""
    gold = np.array([LABELS.index(WORD[r["bucket"]]) for r in rows])
    a = np.array([LABELS.index(pa[r["pair_id"]]) for r in rows])
    b = np.array([LABELS.index(pb[r["pair_id"]]) for r in rows])
    groups = [base_of(r) for r in rows]
    acc = cluster_ci(groups, lambda i: float((gold[i] == a[i]).mean() - (gold[i] == b[i]).mean()), draws)
    f1 = cluster_ci(groups, lambda i: macro_f1_of(gold[i], a[i]) - macro_f1_of(gold[i], b[i]), draws)
    return {"accuracy": [round(x, 4) for x in acc], "macro_f1": [round(x, 4) for x in f1]}


def rules_entries(rows: list[dict]) -> dict[str, dict[str, str]]:
    """Rules on Adaption's extracted cards and the majority class, for the real-test rows (by pair_id)."""
    import baselines as bl
    want = {r["pair_id"] for r in rows}
    full = [r for r in bl.read_jsonl(bl.ROWS) if r["pair_id"] in want]
    pred, info = bl.extract_rules_preds(full)
    if len(pred) < len(want):
        print(f"rules baseline: {len(want) - len(pred)} rows without a valid extracted card ({info})")
    major = max(LABELS, key=lambda w: sum(WORD[r["bucket"]] == w for r in rows))
    return {"rules on extracted cards": pred, f"majority ({major})": {r["pair_id"]: major for r in rows}}


def build(results: list[tuple[str, Path]], draws: int = 1000, sources: dict[str, Path] | None = None,
          pairs: list[tuple[str, str]] = ()) -> dict:
    by_set: dict[str, dict[str, dict]] = {}
    for tag, path in results:
        d = json.loads(path.read_text())
        s = set_of(d, path)
        name = tag or tag_of(path, s)
        by_set.setdefault(s, {})[name] = {"path": str(path), "dtype": d.get("dtype"), "model": d.get("model"),
                                          "rows": {r["pair_id"]: r for r in d["rows"]}}
    if not results:
        by_set["test"] = {}
    out: dict = {}
    for s in [x for x in ORDER if x in by_set]:
        src = source_rows(s, (sources or {}).get(s))
        entries = by_set[s]
        notes = []
        for name, e in entries.items():
            extra = set(e["rows"]) - set(src)
            if extra:
                raise SystemExit(f"{e['path']}: {len(extra)} pair ids are not in {SETS[s].name}, e.g. {sorted(extra)[:3]}")
            short = len([p for p in src if decidable(s, src[p], e["rows"].get(p))]) - \
                len([p for p in e["rows"] if decidable(s, src[p], e["rows"][p])])
            if short:
                notes.append(f"{name} is missing {short} decidable rows")
        ids = [p for p in src if all(p in e["rows"] for e in entries.values())]
        any_row = lambda p: next((e["rows"][p] for e in entries.values()), None)  # noqa: E731
        held = [p for p in ids if decidable(s, src[p], any_row(p))]
        rows = [src[p] for p in held]
        preds = {name: {p: e["rows"][p]["pred"] for p in held} for name, e in entries.items()}
        if s == "test":
            preds.update(rules_entries(rows))
            common = set.intersection(*(set(v) for v in preds.values()))
            rows = [r for r in rows if r["pair_id"] in common]
        out[s] = {"rows": len(rows), "postings": len({base_of(r) for r in rows}),
                  "not_decidable_left_out": len(ids) - len(held), "notes": notes,
                  "dtype": {n: e["dtype"] for n, e in entries.items()},
                  "entries": {n: evaluate(rows, p, draws) for n, p in preds.items()},
                  "paired": {f"{x} vs {y}": paired(rows, preds[x], preds[y], draws)
                             for x, y in pairs if x in preds and y in preds}}
    return out


def fmt(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def markdown(res: dict) -> str:
    lines: list[str] = []
    for s, d in res.items():
        names = list(d["entries"])
        lines += [f"## {NAMES[s]}", "",
                  f"{d['rows']} rows over {d['postings']} postings; {d['not_decidable_left_out']} not decidable left out. "
                  "Intervals resample postings."]
        if any(d["dtype"].values()):
            lines.append("Model dtype: " + ", ".join(f"{n} {t}" for n, t in d["dtype"].items()) + ".")
        lines += [f"Note: {n}." for n in d["notes"]]
        lines += ["", "| System | Accuracy | 95% CI | Macro-F1 | 95% CI | Unsafe Keep | Predicted K/E/S |",
                  "|---|---|---|---|---|---|---|"]
        for n in names:
            e = d["entries"][n]
            pc = e["pred_counts"]
            lines.append(f"| {n} | {e['accuracy'][0]:.3f} | {e['accuracy'][1]:.3f}-{e['accuracy'][2]:.3f} | "
                         f"{e['macro_f1'][0]:.3f} | {e['macro_f1'][1]:.3f}-{e['macro_f1'][2]:.3f} | {e['unsafe_keep']} | "
                         f"{pc['Keep']}/{pc['Explore']}/{pc['Skip']} |")
        first = d["entries"][names[0]]
        lines += ["", "Recall per gold bucket (F1 per label in brackets):", "",
                  "| System | " + " | ".join(f"{w} (n={first['n_by_bucket'][w]})" for w in LABELS) + " |",
                  "|---|" + "---|" * len(LABELS)]
        for n in names:
            e = d["entries"][n]
            lines.append(f"| {n} | " + " | ".join(f"{fmt(e['recall_by_bucket'][w])} ({fmt(e['f1_by_label'][w])})"
                                                   for w in LABELS) + " |")
        kinds = list(first["by_kind"])
        lines += ["", "Accuracy per variant kind" + (" (candidate_kind)" if s == "dev_cross" else "") +
                  "; the enrollment slices first:", "",
                  "| Kind | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        for k in [*[x for x in SLICES if x in kinds], *[x for x in kinds if x not in SLICES]]:
            label = f"**{k}**" if k in SLICES else k
            lines.append(f"| {label} | {first['by_kind'][k]['n']} | " +
                         " | ".join(fmt(d["entries"][n]["by_kind"][k]["accuracy"]) for n in names) + " |")
        if d.get("paired"):
            lines += ["", "Paired differences (A minus B, same resampled postings each draw):", "",
                      "| A vs B | Accuracy | 95% CI | Macro-F1 | 95% CI |", "|---|---|---|---|---|"]
            for n, v in d["paired"].items():
                ac, f1 = v["accuracy"], v["macro_f1"]
                lines.append(f"| {n} | {ac[0]:+.3f} | {ac[1]:+.3f} to {ac[2]:+.3f} | "
                             f"{f1[0]:+.3f} | {f1[1]:+.3f} to {f1[2]:+.3f} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="*", help="TAG=RESULT.json or RESULT.json")
    ap.add_argument("--out", help="write the markdown table here")
    ap.add_argument("--json", help="write the numbers here")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--paired", action="append", default=[], metavar="A vs B",
                    help="A minus B on accuracy and macro-F1, paired over resampled postings (repeatable)")
    a = ap.parse_args()
    pairs = []
    for spec in a.paired:
        x, sep, y = spec.partition(" vs ")
        if not sep or not x.strip() or not y.strip():
            raise SystemExit(f"--paired wants \"A vs B\", got {spec!r}")
        pairs.append((x.strip(), y.strip()))
    results = []
    for spec in a.results:
        tag, sep, path = spec.partition("=")
        results.append((tag, Path(path)) if sep else ("", Path(spec)))
    res = build(results, a.draws, pairs=pairs)
    for x, y in pairs:
        if not any(f"{x} vs {y}" in d["paired"] for d in res.values()):
            print(f"--paired {x} vs {y}: no set has both as entries (names: "
                  + "; ".join(sorted({n for d in res.values() for n in d["entries"]})) + ")", file=sys.stderr)
    text = markdown(res)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
