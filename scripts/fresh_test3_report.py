"""Write the fresh test 3 report from the one scoring run of fit_eval_score.py fresh3 (nothing is rescored here).

  fresh_test3_report.py errors [--n N]   print h+H' errors on fresh test 3 (gold, prediction, H' verdict, gold reason,
                                         profile, title) to read by hand before writing the notes
  fresh_test3_report.py write --notes N  write data/models/v4_eval/FRESH_TEST3.md: tables from eval/fresh_test3_result.json,
                                         the notes file N (verdict reading, errors, sessions) at the end

Reads eval/fresh_test3_result.json and the sealed fresh test 3 files. Loads no model.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

FRESH3 = ROOT / "data/built/fresh_test3"
RESULT = ROOT / "eval/fresh_test3_result.json"
REPORT = ROOT / "data/models/v4_eval/FRESH_TEST3.md"
HEAD = ("all", "eligibility_pass", "eligibility_pass_design")
SPLIT = ("posting_in_dev_fit", "posting_not_in_dev_fit", "posting_in_dev_fit_all", "posting_not_in_dev_fit_all")


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()]


def fmt(x: list[float]) -> str:
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"


def reasons() -> dict[str, str]:
    """The gold reason per row: the adjudicator's on adjudicated rows, else labeler A's (A and B agreed)."""
    out = {}
    for p in sorted((FRESH3 / "labels/A").glob("*.jsonl")):
        out |= {r["row_id"]: r["reason"] for r in jsonl(p)}
    for p in sorted((FRESH3 / "adjudicate").glob("*.gold.jsonl")):
        out |= {r["row_id"]: r["reason"] for r in jsonl(p)}
    return out


def chosen_preds(res: dict) -> tuple[list[dict], dict[str, list[str]]]:
    """Every system's label per row, rebuilt the way fit_eval_score builds them, from the score files the result names."""
    import fit_eval_score as fes
    rows = fes.load_fresh(FRESH3, strict_design=False)
    ids = [r["row_id"] for r in rows]
    models = {t: fes.read_scores(ROOT / v["path"], ids) for t, v in res["score_files"].items()}
    return rows, fes.systems(rows, models)


def cmd_errors(a: argparse.Namespace) -> None:
    res = json.loads(RESULT.read_text())
    me = res["chosen"]
    rows, P = chosen_preds(res)
    src = {r["row_id"]: r for r in jsonl(FRESH3 / "rows.jsonl")}
    why = reasons()
    conf = Counter((r["gold"], p) for r, p in zip(rows, P[me]))
    print("confusion (gold, pred)", dict(conf))
    errs = [(r, p) for r, p in zip(rows, P[me]) if p != r["gold"]]
    print(len(errs), "errors;", sum(r["eligibility_pass"] for r, _ in errs), "on the eligibility-pass slice")
    for r, p in errs[: a.n]:
        s = src[r["row_id"]]
        v = r["verdict"]
        print(f"\n-- {r['row_id']} | gold {r['gold']} | {me} {p} | raw {P[me.split('+')[0] + ' raw'][rows.index(r)]} | "
              f"rules_full {r['rules_full']} | elig {r['eligibility_pass']} | {s['title']} @ {s['employer']}")
        print("   gold reason:", why.get(r["row_id"]))
        print("   A/B:", r["label_A"], r["label_B"], "| verdict:", v["rule"], v.get("hard_cats"), "| salary_soft", v["salary_soft"])
        pr = s["profile"]
        print("   profile:", {k: pr[k] for k in ("degree_level", "degree_field", "years", "skills", "licenses",
                                                "current_location", "remote_preference", "work_auth", "salary_floor",
                                                "employment_types_wanted")})


def system_table(sl: dict) -> list[str]:
    L = ["| system | accuracy [95%] | macro-F1 [95%] | Keep recall | Keep pred | unsafe Keep |", "|---|---|---|---|---|---|"]
    for k, v in sl["systems"].items():
        kr = v["per_label"]["Keep"]["recall"]
        L.append(f"| {k} | {fmt(v['accuracy'])} | {fmt(v['macro_f1'])} | {'' if kr is None else f'{kr:.3f}'} | "
                 f"{v['per_label']['Keep']['pred']} | {v['unsafe_keep']} |")
    c = sl.get("labeler_ceiling")
    if c:
        L.append(f"| labeler ceiling (A vs B) | {c['raw_agreement']:.3f} | kappa {c['kappa']:.3f} | | | |")
    return L


def diff_table(sl: dict, me: str) -> list[str]:
    L = ["| comparison | accuracy diff | macro-F1 diff | macro-F1 low (exact) | rows fixed / broken |", "|---|---|---|---|---|"]
    for k, d in sl["paired"].items():
        if k.startswith(me + " - ") and " raw" not in k and "title_match" not in k:
            L.append(f"| {k} | {fmt(d['accuracy_diff'])} | {fmt(d['macro_f1_diff'])} | {d['macro_f1_diff_low_exact']:.4f} | "
                     f"{d['rows_fixed']} / {d['rows_broken']} |")
    return L


def cmd_write(a: argparse.Namespace) -> None:
    res = json.loads(RESULT.read_text())
    me, p, S = res["chosen"], res["pass"], res["slices"]
    ev = res["eligibility_vs_sealed_design"]
    sens = res["sealed_design_slice (sensitivity, not for pass)"]["against"]
    L = ["# Fresh test 3: the one scoring run (amendments 3 and 3b)", "",
         f"Sealed fresh test 3: {S['all']['n']} rows, {S['all']['postings']} postings, 46 employers; gold from two blind "
         "labelers plus blind adjudication (eval/fresh_test3_gold_manifest.json). Scored once by `fit_eval_score.py "
         f"fresh3`; chosen system **{me}** from eval/selection_test3.json (the author, 2026-10-21: score now with h, decide "
         "cycle 3 after). Intervals: two-way bootstrap (postings, then candidates), "
         f"{res['draws']} draws, seed {res['seed']}, 95% percentile.", "",
         "## Verdict (amendment 2's bar on fresh test 3)", "",
         f"**{'PASSED' if p['passed'] else 'NOT PASSED'}**: " + "; ".join(
             f"{me} - {b} macro-F1 {fmt(v['macro_f1_diff'])}, lower bound {v['lower_bound_exact']:.4f} -> "
             f"{'beaten' if v['beaten'] else 'not beaten'}" for b, v in p["against"].items()) + ".", ""]
    L += [f"- {me} - {b}: accuracy diff (reported only) {fmt(v['accuracy_diff (reported, not for pass)'])}"
          for b, v in p["against"].items()]
    L += ["", f"Eligibility-pass slice recomputed with the rules in effect: {ev['recomputed_pass_rows']} rows (sealed design "
          f"{ev['sealed_design_pass_rows']}; moved out {ev['rows_moved_out']} rows on {ev['postings_moved_out']} postings "
          f"by {ev['moved_out_by']}, their gold {ev['gold_of_moved_out_rows']}; moved in {ev['rows_moved_in']}).", "",
          "Sensitivity, sealed-design slice (same systems, its own draws, not for pass): " + "; ".join(
              f"{me} - {b} {fmt(v['macro_f1_diff'])}, low {v['lower_bound_exact']:.4f} -> "
              f"{'would be beaten' if v['would_be_beaten'] else 'would not be beaten'}" for b, v in sens.items()) + ".", ""]
    for name in HEAD:
        sl = S[name]
        L += [f"## {name}: n={sl['n']}, postings={sl['postings']}, gold {sl['gold']}", ""] + system_table(sl) + [""]
        L += diff_table(sl, me) + [""]
    L += ["## posting_in_dev_fit split (cycle 1's dev_fit selected h on these postings)", "",
          f"| slice | n | postings | gold | {me} macro-F1 | rules_full | base+H' | {me} - rules_full | {me} - base+H' | {me} acc |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name in SPLIT:
        sl = S.get(name)
        if not sl:
            continue
        Y, D, bh = sl["systems"], sl["paired"], "base+H'"
        L.append(f"| {name} | {sl['n']} | {sl['postings']} | {sl['gold']} | {fmt(Y[me]['macro_f1'])} | "
                 f"{fmt(Y['rules_full']['macro_f1'])} | {fmt(Y[bh]['macro_f1'])} | "
                 f"{fmt(D[me + ' - rules_full']['macro_f1_diff'])} | {fmt(D[me + ' - ' + bh]['macro_f1_diff'])} | "
                 f"{fmt(Y[me]['accuracy'])} |")
    L += ["", "## Slices for the chosen system (reported, not for pass)", "",
          f"| slice | n | gold | {me} macro-F1 | rules_full | base+H' | {me} - rules_full | {me} - base+H' | labeler agreement |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, sl in S.items():
        if sl is None:
            L.append(f"| {name} | 0 | | | | | | | |")
            continue
        Y, D, c = sl["systems"], sl["paired"], sl.get("labeler_ceiling") or {}
        bh = "base+H'"
        L.append(f"| {name} | {sl['n']} | {sl['gold']} | {fmt(Y[me]['macro_f1'])} | {fmt(Y['rules_full']['macro_f1'])} | "
                 f"{fmt(Y[bh]['macro_f1'])} | {fmt(D[me + ' - rules_full']['macro_f1_diff'])} | "
                 f"{fmt(D[me + ' - ' + bh]['macro_f1_diff'])} | {c.get('raw_agreement')} / k {c.get('kappa')} |")
    L += ["", "## Notes", "", Path(a.notes).read_text().strip(), "",
          "Files: " + "; ".join(f"{t} {v['path']} sha256 {v['sha256']}" for t, v in res["score_files"].items())
          + f"; selection {res['selection_file']['path']} sha256 {res['selection_file']['sha256']}.", ""]
    REPORT.write_text("\n".join(L))
    print(REPORT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("errors")
    e.add_argument("--n", type=int, default=400)
    e.set_defaults(fn=cmd_errors)
    w = sub.add_parser("write")
    w.add_argument("--notes", required=True)
    w.set_defaults(fn=cmd_write)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
