"""Every system's real-test score with and without the postings test_restrictions flags (e589370's rule, applied to the
real test before any model output was read). A disclosed secondary number: the filter is fixed and nothing is
re-selected on it. numpy only, no model.

  test_screen.py --pred TAG=RESULT.json [--pred ...] [--hybrid TAG ...] [--paired 'A - B' ...]
                 [--test FILE] [--table FILE] [--draws 1000] [--out DIR]

  --pred      a local_eval / colab_eval output on test.jsonl (rows with pair_id and pred), any number.
  --hybrid    a --pred tag to also score as TAG+H and TAG+H' (hybrid_rules.hybrid on the rules' verdict on
              Adaption's extracted card, hybrid_rules.rule_verdict; H is pre-registered, H' is test-informed).
  --paired    'A - B' on the same rows, posting-resampled interval (hybrid_rules.paired); A and B are system names
              as the table prints them. Defaults: c+H - rules, c+H' - rules, e+H' - c+H' (each only when both
              systems are there).
  --table     the restriction table module (default scripts/test_restrictions.py): its test_decisions() applies
              clean_real_checked.dev_v2_decisions with TEST_RESTRICTIONS and TEST_READ_CLEAR.
  --out       writes DIR/RESULTS.md and DIR/test_screen.json (default data/models/main_4b/test_screen); a
              DIR/NOTES.md, if there, is appended to RESULTS.md on every run.

Rows: the decidable rows of the test file (the evaluators and results_table leave the rest out) that every system
scored; "without" drops every row of a posting test_decisions() removes. Metrics are results_table.evaluate's
(accuracy and macro-F1 with a 95% interval resampling postings by base_rq_id, results_table's draws and seed). The
rules baseline and always-Skip are results_table.rules_entries (rules.label on Adaption's extracted card; the
majority class of the decidable rows, which is Skip). Adding a run later is one more --pred.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import hybrid_rules as hr  # noqa: E402
import results_table as rt  # noqa: E402
from matcher.evalstats import LABELS, WORD, base_of  # noqa: E402
from matcher.schema import CandidateCard  # noqa: E402

TEST = ROOT / "data/built/real_test/test.jsonl"
BUILDER_ROWS = ROOT / "data/built/real_test/rows.jsonl"
TABLE = ROOT / "scripts/test_restrictions.py"
OUT = ROOT / "data/models/main_4b/test_screen"
RULES = "rules"
SKIP_ALL = "always Skip"
DEFAULT_PAIRED = ("c+H - rules", "c+H' - rules", "e+H' - c+H'")
HYBRIDS = {"H": hr.VARIANTS["H"], "H'": hr.VARIANTS["H_prime"]}


def load_table(path: Path):
    spec = importlib.util.spec_from_file_location("restriction_table", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def system_preds(rows: list[dict], preds: dict[str, Path], hybrid_tags: list[str]) -> tuple[dict, list[str]]:
    """name -> {pair_id: label} for every system, and notes (rows a system misses)."""
    out: dict[str, dict[str, str]] = {}
    notes = []
    want = {r["pair_id"] for r in rows}
    for tag, path in preds.items():
        p = hr.model_preds(path)
        if (miss := want - set(p)):
            notes.append(f"{tag} ({path}) has no prediction for {len(miss)} decidable rows")
        out[tag] = {k: v for k, v in p.items() if k in want}
    if hybrid_tags:
        unknown = set(hybrid_tags) - set(preds)
        if unknown:
            raise SystemExit(f"--hybrid {sorted(unknown)}: no --pred with that tag")
        cards = hr.load_cards(hr.CARDS)
        cand = {r["pair_id"]: r["candidate"] for r in hr.read_jsonl(BUILDER_ROWS) if r["pair_id"] in want}
        verdict = {r["pair_id"]: hr.rule_verdict(CandidateCard.from_dict(cand[r["pair_id"]]), cards[r["rq_id"]])
                   for r in rows if r["rq_id"] in cards and r["pair_id"] in cand}
        if len(verdict) < len(want):
            notes.append(f"hybrids: {len(want) - len(verdict)} decidable rows have no extracted-card verdict")
        for tag in hybrid_tags:
            for name, (cats, salary) in HYBRIDS.items():
                out[f"{tag}+{name}"] = {k: hr.hybrid(m, verdict[k], cats, salary)
                                        for k, m in out[tag].items() if k in verdict}
    rules = rt.rules_entries(rows)
    rule_name = next(k for k in rules if k.startswith("rules"))
    major = next(k for k in rules if k.startswith("majority"))
    if major != "majority (Skip)":
        raise SystemExit(f"the majority class of the decidable rows is not Skip ({major})")
    out[RULES] = rules[rule_name]
    out[SKIP_ALL] = rules[major]
    if hybrid_tags:  # the hybrids read the same rule (rules.label on the extracted card) as the baseline
        same = sum(verdict[k]["rule"] == v for k, v in out[RULES].items() if k in verdict)
        if same != len(out[RULES]):
            notes.append(f"hybrid verdicts and the rules baseline disagree on {len(out[RULES]) - same} rows")
    return out, notes


def score(rows: list[dict], sys_preds: dict[str, dict[str, str]], paired: list[tuple[str, str]],
          draws: int) -> dict:
    gold = [WORD[r["bucket"]] for r in rows]
    groups = [base_of(r) for r in rows]
    ent = {n: rt.evaluate(rows, p, draws) for n, p in sys_preds.items()}
    col = {n: [p[r["pair_id"]] for r in rows] for n, p in sys_preds.items()}
    diffs = {f"{a} - {b}": hr.paired(gold, col[a], col[b], groups, draws) for a, b in paired}
    return {"rows": len(rows), "postings": len(set(groups)), "labels": dict(Counter(gold)), "entries": ent,
            "paired": diffs}


def flagged_table(dec: dict[str, dict], all_rows: list[dict], decidable: set[str]) -> list[dict]:
    import assemble_real_train as art
    raw = art.brt_raw()
    out = []
    for i, d in sorted(dec.items(), key=lambda kv: (not kv[1]["remove"], kv[1]["kind"] or "~", kv[0])):
        if d["kind"] is None:
            continue
        mine = [r for r in all_rows if r["rq_id"] == i]
        out.append({"posting_id": i, "company": raw[i].get("company", ""), "title": raw[i].get("title", ""),
                    **{k: d[k] for k in ("kind", "quote", "limit", "remove", "past_limit_rows")},
                    "rows": dict(Counter(WORD[r["bucket"]] for r in mine)),
                    "decidable_rows": dict(Counter(WORD[r["bucket"]] for r in mine if r["pair_id"] in decidable))})
    return out


def run(a: argparse.Namespace) -> dict:
    table = load_table(Path(a.table))
    all_rows = [{k: v for k, v in r.items() if k != "prompt"} for r in hr.read_jsonl(Path(a.test))]
    cands = {r["pair_id"]: r["candidate"] for r in hr.read_jsonl(BUILDER_ROWS)}
    dec = table.test_decisions(all_rows, cands)
    gone = {i for i, d in dec.items() if d["remove"]}
    rows = [r for r in all_rows if r.get("decidable", True)]
    preds = {}
    for spec in a.pred:
        tag, sep, path = spec.partition("=")
        if not sep:
            raise SystemExit(f"--pred {spec}: write TAG=RESULT.json")
        preds[tag] = Path(path)
    sys_preds, notes = system_preds(rows, preds, a.hybrid)
    common = set.intersection(*(set(p) for p in sys_preds.values()))
    if len(common) < len(rows):
        notes.append(f"{len(rows) - len(common)} decidable rows are left out: some system has no label for them")
    rows = [r for r in rows if r["pair_id"] in common]
    pairs = []
    for spec in (a.paired or [p for p in DEFAULT_PAIRED if all(x.strip() in sys_preds for x in p.split(" - "))]):
        x, sep, y = spec.partition(" - ")
        if not sep or x.strip() not in sys_preds or y.strip() not in sys_preds:
            raise SystemExit(f"--paired {spec!r}: write 'A - B' with A and B among {sorted(sys_preds)}")
        pairs.append((x.strip(), y.strip()))
    kept = [r for r in rows if r["rq_id"] not in gone]
    dropped = [r for r in rows if r["rq_id"] in gone]
    decidable_ids = {r["pair_id"] for r in all_rows if r.get("decidable", True)}
    out = {"test": str(a.test), "table": str(a.table), "preds": {k: str(v) for k, v in preds.items()},
           "hybrids_of": a.hybrid, "draws": a.draws, "notes": notes,
           "postings_read": len(dec), "postings_flagged": sum(d["kind"] is not None for d in dec.values()),
           "postings_removed": len(gone), "removed_by_kind": dict(Counter(dec[i]["kind"] for i in gone)),
           "rows_removed": {"all_rows": len([r for r in all_rows if r["rq_id"] in gone]),
                            "all_rows_by_label": dict(Counter(WORD[r["bucket"]] for r in all_rows if r["rq_id"] in gone)),
                            "decidable_scored": len(dropped),
                            "decidable_scored_by_label": dict(Counter(WORD[r["bucket"]] for r in dropped))},
           "flagged": flagged_table(dec, all_rows, decidable_ids),
           "with": score(rows, sys_preds, pairs, a.draws),
           "without": score(kept, sys_preds, pairs, a.draws)}
    return out


def ci(v: list[float]) -> str:
    return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def markdown(res: dict) -> str:
    w, wo = res["with"], res["without"]
    lab = lambda d: ", ".join(f"{k} {d.get(k, 0)}" for k in LABELS)  # noqa: E731
    rr = res["rows_removed"]
    lines = ["# Real test with and without the restricted postings", "",
             "e589370's restriction rule (clean_real_checked.py dev-v2) applied to the real test by "
             "scripts/test_restrictions.py, decided from the posting text and the builder's cards before any model "
             "output was read. A fixed filter for a disclosed secondary number; nothing is selected on it.", "",
             f"Postings read: {res['postings_read']}; with a restriction entry: {res['postings_flagged']}; removed: "
             f"{res['postings_removed']} ({', '.join(f'{k} {v}' for k, v in sorted(res['removed_by_kind'].items()))}). "
             f"Rows removed: {rr['all_rows']} of the file ({lab(rr['all_rows_by_label'])}); "
             f"{rr['decidable_scored']} of the scored decidable rows ({lab(rr['decidable_scored_by_label'])}).", "",
             f"With: {w['rows']} rows over {w['postings']} postings ({lab(w['labels'])}). Without: {wo['rows']} rows "
             f"over {wo['postings']} postings ({lab(wo['labels'])}). Intervals: 95%, resampling postings "
             f"({res['draws']} draws). H is pre-registered; H' is test-informed.", ""]
    lines += [f"Note: {n}." for n in res["notes"]]
    lines += ["", "## Scores", "",
              "| System | Accuracy, all | Accuracy, without | Macro-F1, all | Macro-F1, without |", "|---|---|---|---|---|"]
    for n in w["entries"]:
        a, b = w["entries"][n], wo["entries"][n]
        lines.append(f"| {n} | {ci(a['accuracy'])} | {ci(b['accuracy'])} | {ci(a['macro_f1'])} | {ci(b['macro_f1'])} |")
    if w["paired"]:
        lines += ["", "## Paired differences (same rows, same posting draws)", "",
                  "| Difference | Accuracy, all | Accuracy, without | Macro-F1, all | Macro-F1, without | "
                  "Fixed/broken, all | Fixed/broken, without |", "|---|---|---|---|---|---|---|"]
        for k in w["paired"]:
            a, b = w["paired"][k], wo["paired"][k]
            lines.append(f"| {k} | {ci(a['accuracy_diff'])} | {ci(b['accuracy_diff'])} | {ci(a['macro_f1_diff'])} | "
                         f"{ci(b['macro_f1_diff'])} | {a['rows_fixed']}/{a['rows_broken']} | "
                         f"{b['rows_fixed']}/{b['rows_broken']} |")
    lines += ["", "## Recall per gold label, all / without", "",
              "| System | " + " | ".join(LABELS) + " |", "|---|" + "---|" * len(LABELS)]
    for n in w["entries"]:
        a, b = w["entries"][n]["recall_by_bucket"], wo["entries"][n]["recall_by_bucket"]
        lines.append(f"| {n} | " + " | ".join(f"{rt.fmt(a[x])} / {rt.fmt(b[x])}" for x in LABELS) + " |")
    lines += ["", "## Flagged postings", "",
              "Removed: every row goes. A degree_ceiling or exp_cap entry removes its posting only when a non-Skip "
              "row's candidate is past the limit (none here is). Rows: all rows in test.jsonl per gold label "
              "(decidable rows in brackets).", "",
              "| Posting | Employer, title | Kind | Quote | Removed | Keep | Explore | Skip |",
              "|---|---|---|---|---|---|---|---|"]
    for f in res["flagged"]:
        kind = f["kind"] + (f" ({f['limit']})" if f["limit"] is not None else "")
        cells = " | ".join(f"{f['rows'].get(x, 0)} ({f['decidable_rows'].get(x, 0)})" for x in LABELS)
        quote = f["quote"].replace("|", "/")
        lines.append(f"| {f['posting_id']} | {f['company']}, {f['title']} | {kind} | \"{quote}\" | "
                     f"{'yes' if f['remove'] else 'no'} | {cells} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", action="append", default=[], help="TAG=RESULT.json")
    ap.add_argument("--hybrid", action="append", default=[], help="a --pred tag to score with H and H' too")
    ap.add_argument("--paired", action="append", default=[], help="'A - B', system names as printed")
    ap.add_argument("--test", default=str(TEST))
    ap.add_argument("--table", default=str(TABLE))
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    res = run(a)
    text = markdown(res)
    d = Path(a.out)
    if (d / "NOTES.md").exists():  # hand-written notes kept across reruns
        text += "\n" + (d / "NOTES.md").read_text()
    print(text)
    d.mkdir(parents=True, exist_ok=True)
    (d / "RESULTS.md").write_text(text)
    (d / "test_screen.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {d / 'RESULTS.md'} and {d / 'test_screen.json'}")


if __name__ == "__main__":
    main()
