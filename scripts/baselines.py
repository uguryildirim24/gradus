"""Baselines for the real-posting test, scored on the same rows as the model.

  baselines.py majority [--rows ROWS] [--out FILE]
      Always Keep, always Explore, always Skip, and the majority class of the rows.
  baselines.py extract-rules [--rows ROWS] [--cards CARDS] [--out FILE]
      "Why not just extract with Adaption and apply your rules?" Each row's gold label comes from
      the hand-corrected card; this baseline labels the same candidate with rules.label() on the
      card Adaption's extractor produced (data/built/cards/extracted_real_cards_v2.jsonl, the v2 blueprint
      cards after `extract_postings.py merge`, the cards the pairs use). Reports
      accuracy, macro-F1 with a posting-clustered 95% interval, the confusion matrix and the unsafe
      rows (gold Skip, predicted Keep).
  baselines.py table [--rows ROWS] [--model NAME=RESULT.json ...] [--out FILE.md]
      One table: the baselines above plus any model results (an evaluator --out file, base model
      zero-shot included), joined to the rows by pair_id, on the rows every entry scored.

The base-model zero-shot row is the base Qwen3.5 checkpoint run through the same evaluator on the
same rows (no weights are loaded here); pass its result file to `table --model`.
No frontier-model baseline (out of scope for the deadline).
Only rows test.jsonl holds are scored (rows.jsonl also keeps the ones with no rendered resume). Rows
test.jsonl marks not decidable (the rendered prompt does not show a fact the label rests on) are
left out, as build_real_test.py score and local_eval.py leave them out; --include-undecidable keeps
them. Each run prints how many were left out. Every result also gives the numbers without the
added Keep kinds and on the resume-dependent kinds alone (evalstats.side_headlines).
All of this is numpy and the rules: no model runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from matcher.evalstats import LABELS, WORD, cluster_ci, confusion, macro_f1_of, side_headlines  # noqa: E402
from matcher.rules import label  # noqa: E402
from matcher.schema import CandidateCard, RequirementCard  # noqa: E402

ROWS = ROOT / "data/built/real_test/rows.jsonl"
TEST = ROOT / "data/built/real_test/test.jsonl"
CARDS = ROOT / "data/built/cards/extracted_real_cards_v2.jsonl"
BASELINES = ROOT / "data/built/real_test/baselines"


def read_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def undecidable_ids(test: Path = TEST) -> set[str]:
    """Pair ids test.jsonl marks not decidable (empty before assemble has run)."""
    return {r["pair_id"] for r in read_jsonl(test) if not r.get("decidable", True)} if test.exists() else set()


def load_rows(path: Path, include_undecidable: bool = False, test: Path = TEST) -> list[dict]:
    """The rows test.jsonl holds (rows.jsonl also has the ones with no rendered resume, which the model never
    sees), without those test.jsonl (or the rows themselves) mark not decidable."""
    rows = read_jsonl(path)
    if test.exists():
        held = {r["pair_id"] for r in read_jsonl(test)}
        n_out = sum(r["pair_id"] not in held for r in rows)
        rows = [r for r in rows if r["pair_id"] in held]
        if n_out:
            print(f"{n_out} rows not in {test.name} (no rendered resume) left out")
    bad = undecidable_ids(test)
    is_bad = [r["pair_id"] in bad or r.get("decidable", True) is False for r in rows]
    n_bad = sum(is_bad)
    if include_undecidable:
        print(f"{len(rows)} rows scored, {n_bad} of them not decidable (kept: --include-undecidable)")
        return rows
    print(f"{len(rows) - n_bad} rows scored, {n_bad} not decidable left out")
    return [r for r, x in zip(rows, is_bad) if not x]


def evaluate(rows: list[dict], pred: dict[str, str], draws: int = 1000) -> dict:
    """Accuracy and macro-F1 (posting-clustered 95% CI), confusion and unsafe rows over the rows with a prediction."""
    sub = [r for r in rows if r["pair_id"] in pred]
    gold = np.array([LABELS.index(WORD[r["bucket"]]) for r in sub])
    p = np.array([LABELS.index(pred[r["pair_id"]]) for r in sub])
    groups = [r["base_rq_id"] for r in sub]
    f1 = cluster_ci(groups, lambda i: macro_f1_of(gold[i], p[i]), draws)
    acc = cluster_ci(groups, lambda i: float((gold[i] == p[i]).mean()), draws)
    conf = confusion(gold, p)
    return {
        "n": len(sub), "postings": len(set(groups)),
        "accuracy": round(acc[0], 4), "accuracy_ci95": [round(acc[1], 4), round(acc[2], 4)],
        "macro_f1": round(f1[0], 4), "macro_f1_ci95": [round(f1[1], 4), round(f1[2], 4)],
        "confusion": {LABELS[g]: {LABELS[q]: int(conf[g, q]) for q in range(3)} for g in range(3)},
        "unsafe_keep": int(((gold != 0) & (p == 0)).sum()),
        "unsafe_keep_rate": round(float(((gold != 0) & (p == 0)).sum() / max((gold != 0).sum(), 1)), 4),
        "side_headlines": side_headlines([{"gold": WORD[r["bucket"]], "pred": pred[r["pair_id"]],
                                           "variant_kind": r.get("variant_kind"), "base_rq_id": r["base_rq_id"]}
                                          for r in sub], draws),
    }


def print_eval(name: str, res: dict) -> None:
    print(f"{name}: n={res['n']} ({res['postings']} postings) accuracy {res['accuracy']:.3f} "
          f"[{res['accuracy_ci95'][0]:.3f}, {res['accuracy_ci95'][1]:.3f}]; macro-F1 {res['macro_f1']:.3f} "
          f"[{res['macro_f1_ci95'][0]:.3f}, {res['macro_f1_ci95'][1]:.3f}]; unsafe Keep {res['unsafe_keep']} "
          f"({res['unsafe_keep_rate']:.1%} of non-Keep rows)")
    for k, d in res.get("side_headlines", {}).items():
        if d:
            print(f"  {k.replace('_', ' ')}: n={d['n']} accuracy {d['accuracy']:.3f}; macro-F1 {d['macro_f1']['value']:.3f} "
                  f"[{d['macro_f1']['ci95'][0]:.3f}, {d['macro_f1']['ci95'][1]:.3f}]")


def save(name: str, rows: list[dict], pred: dict[str, str], res: dict, out: str | None) -> None:
    path = Path(out) if out else BASELINES / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"baseline": name, "summary": res,
                                "rows": [{"pair_id": r["pair_id"], "gold": WORD[r["bucket"]], "pred": pred[r["pair_id"]],
                                          "variant_kind": r["variant_kind"], "base_rq_id": r["base_rq_id"]}
                                         for r in rows if r["pair_id"] in pred]}, indent=1))


# ---------------------------------------------------------------- baselines

def constant_preds(rows: list[dict]) -> dict[str, dict[str, str]]:
    counts = Counter(WORD[r["bucket"]] for r in rows)
    major = counts.most_common(1)[0][0]
    out = {f"always_{w.lower()}": {r["pair_id"]: w for r in rows} for w in LABELS}
    out["majority"] = {r["pair_id"]: major for r in rows}
    return out


def extract_rules_preds(rows: list[dict], cards_path: Path = CARDS) -> tuple[dict[str, str], dict]:
    """Rules label on Adaption's extracted card, for every row whose posting has a valid extracted card."""
    raw = {r["id"]: r["card"] for r in read_jsonl(cards_path)}
    cache: dict[str, RequirementCard | None] = {}
    pred: dict[str, str] = {}
    missing = bad = 0
    for r in rows:
        rid = r["rq_id"]
        if rid not in cache:
            try:
                cache[rid] = RequirementCard.from_dict(raw[rid]) if rid in raw else None
            except (ValueError, TypeError, KeyError):
                cache[rid] = None
                bad += 1
            missing += rid not in raw
        card = cache[rid]
        if card is None:
            continue
        bucket = label(CandidateCard.from_dict(r["candidate"]), card)[0]
        pred[r["pair_id"]] = WORD[bucket]
    return pred, {"rows_scored": len(pred), "rows_without_extracted_card": len(rows) - len(pred),
                  "postings_missing": missing, "postings_invalid": bad}


def model_preds(path: Path) -> dict[str, str]:
    d = json.loads(path.read_text())
    return {r["pair_id"]: r["pred"] for r in d["rows"]}


# ---------------------------------------------------------------- commands

def cmd_majority(a: argparse.Namespace) -> None:
    rows = load_rows(Path(a.rows), a.include_undecidable)
    for name, pred in constant_preds(rows).items():
        res = evaluate(rows, pred, a.draws)
        print_eval(name, res)
        save(name, rows, pred, res, a.out if name == "majority" else None)


def cmd_extract_rules(a: argparse.Namespace) -> None:
    rows = load_rows(Path(a.rows), a.include_undecidable)
    pred, info = extract_rules_preds(rows, Path(a.cards))
    res = evaluate(rows, pred, a.draws) | info
    print_eval("extract+rules", res)
    print("confusion (rows gold, columns predicted):")
    print("         " + " ".join(f"{w:>8}" for w in LABELS))
    for g in LABELS:
        print(f"{g:>8} " + " ".join(f"{res['confusion'][g][q]:>8}" for q in LABELS))
    print(info)
    save("extract_rules", rows, pred, res, a.out)


def cmd_table(a: argparse.Namespace) -> None:
    rows = load_rows(Path(a.rows), a.include_undecidable)
    entries: dict[str, dict[str, str]] = {}
    entries.update(constant_preds(rows))
    entries["extract+rules (Adaption cards)"] = extract_rules_preds(rows, Path(a.cards))[0]
    for spec in a.model or []:
        name, _, path = spec.partition("=")
        entries[name] = model_preds(Path(path))
    common = set.intersection(*(set(p) for p in entries.values()))
    sub = [r for r in rows if r["pair_id"] in common]
    lines = [f"Real postings: {len(sub)} rows from {len({r['base_rq_id'] for r in sub})} postings; "
             "intervals resample postings.", "",
             "| System | Accuracy | Macro-F1 | 95% CI | Macro-F1 without added Keep kinds | Macro-F1 resume kinds "
             "| Unsafe Keep |", "|---|---|---|---|---|---|---|"]

    def side(res: dict, k: str) -> str:
        d = res["side_headlines"].get(k)
        return f"{d['macro_f1']['value']:.3f}" if d else "-"

    for name, pred in entries.items():
        res = evaluate(sub, pred, a.draws)
        lines.append(f"| {name} | {res['accuracy']:.3f} | {res['macro_f1']:.3f} | "
                     f"{res['macro_f1_ci95'][0]:.3f} to {res['macro_f1_ci95'][1]:.3f} | "
                     f"{side(res, 'without_added_keep_kinds')} | {side(res, 'resume_kinds')} | {res['unsafe_keep']} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("majority", "extract-rules", "table"):
        p = sub.add_parser(name)
        p.add_argument("--rows", default=str(ROWS))
        p.add_argument("--cards", default=str(CARDS))
        p.add_argument("--out")
        p.add_argument("--draws", type=int, default=1000)
        p.add_argument("--include-undecidable", action="store_true", help="keep rows test.jsonl marks not decidable")
        if name == "table":
            p.add_argument("--model", action="append", help="NAME=RESULT.json from the evaluator")
    a = ap.parse_args()
    {"majority": cmd_majority, "extract-rules": cmd_extract_rules, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
