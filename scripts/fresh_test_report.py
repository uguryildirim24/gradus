"""Write the v4 fresh-test report from the one scoring run of fit_eval_score.py fresh (nothing is rescored here).

  fresh_test_report.py rows            write data/models/v4_eval/fresh_test/{TAG}.jsonl, the per-row label scores, from
                                       each session's {TAG}_run/{TAG}_fresh_test.json
  fresh_test_report.py errors          print the chosen system's per-row errors for the two error modes expected
                                       before scoring (read them by hand before writing the notes)
  fresh_test_report.py write --notes N write data/models/v4_eval/FRESH_TEST.md (tables from fresh_test.json, the notes
                                       file N in the middle) and eval/fresh_test_result_v4.json (headline, verdict,
                                       sha256 of every per-row and result file)

Reads data/models/v4_eval/fresh_test/fresh_test.json (fit_eval_score.py fresh --out data/models/v4_eval/fresh_test).
Loads no model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

OUT = ROOT / "data/models/v4_eval/fresh_test"
FRESH = ROOT / "data/built/fresh_test"
RESULT = ROOT / "eval/fresh_test_result_v4.json"
REPORT = ROOT / "data/models/v4_eval/FRESH_TEST.md"
TAGS = ("h", "d3", "base")
SUCH_AS = re.compile(r"\b(such as|e\.g\.|for example|including)\b", re.I)


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cmd_rows(_: argparse.Namespace) -> None:
    for t in TAGS:
        d = json.loads((OUT / f"{t}_run/{t}_fresh_test.json").read_text())
        with (OUT / f"{t}.jsonl").open("w") as f:
            for r in d["rows"]:
                f.write(json.dumps({"row_id": r["pair_id"], "pair_id": r["pair_id"], "gold": r["gold"],
                                    "pred": r["pred"], "scores": r["scores"]}) + "\n")
        print(t, len(d["rows"]), "rows", d["dtype"], d["timing"])


def reasons() -> dict[str, str]:
    """The gold reason per row: the adjudicator's on adjudicated rows, else labeler A's (A and B agreed)."""
    out = {}
    for p in sorted((FRESH / "labels/A").glob("*.jsonl")):
        out |= {r["row_id"]: r["reason"] for r in jsonl(p)}
    for p in sorted((FRESH / "adjudicate").glob("*.gold.jsonl")):
        out |= {r["row_id"]: r["reason"] for r in jsonl(p)}
    return out


def chosen_preds(res: dict) -> tuple[dict[str, str], dict[str, dict]]:
    """The chosen system's label per row, rebuilt the way fit_eval_score builds it (H' on the model's label)."""
    import fit_eval_score as fes
    rows = fes.load_fresh()
    tag = res["chosen"].split("+")[0]
    lab = fes.read_scores(OUT / f"{tag}_run/{tag}_fresh_test.json", [r["row_id"] for r in rows])
    P = fes.systems(rows, {tag: lab})
    return {r["row_id"]: p for r, p in zip(rows, P[res["chosen"]])}, {r["row_id"]: r for r in rows}


def cmd_errors(_: argparse.Namespace) -> None:
    res = json.loads((OUT / "fresh_test.json").read_text())
    pred, meta = chosen_preds(res)
    rows = {r["row_id"]: r for r in jsonl(FRESH / "rows.jsonl")}
    why = reasons()
    conf = Counter((meta[i]["gold"], p) for i, p in pred.items())
    print("confusion gold->pred", dict(conf))
    print("\n=== mode 1: gold Keep, predicted Skip ===")
    for i, p in pred.items():
        if meta[i]["gold"] == "Keep" and p == "Skip":
            r = rows[i]
            hits = [m.start() for m in SUCH_AS.finditer(r["posting_text"])]
            print(f"\n-- {i} | {r['title']} @ {r['employer']} | elig_pass {meta[i]['eligibility_pass']} | "
                  f"such-as hits {len(hits)}\n gold reason: {why.get(i)}")
            for h in hits[:4]:
                print("  posting:", r["posting_text"][max(0, h - 160):h + 200].replace("\n", " "))
    print("\n=== mode 2: gold Skip, predicted Explore ===")
    m2 = [i for i, p in pred.items() if meta[i]["gold"] == "Skip" and p == "Explore"]
    print(len(m2), "rows;", sum(meta[i]["eligibility_pass"] for i in m2), "on the eligibility-pass slice")
    for i in m2:
        print(f"-- {i} | {rows[i]['title']} | {why.get(i)}")


def fmt(x: list[float]) -> str:
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"


def system_table(sl: dict) -> list[str]:
    L = ["| system | accuracy [95%] | macro-F1 [95%] | Keep recall | unsafe Keep |", "|---|---|---|---|---|"]
    for k, v in sl["systems"].items():
        kr = v["per_label"]["Keep"]["recall"]
        L.append(f"| {k} | {fmt(v['accuracy'])} | {fmt(v['macro_f1'])} | "
                 f"{'' if kr is None else f'{kr:.3f}'} | {v['unsafe_keep']} |")
    c = sl.get("labeler_ceiling")
    if c:
        L.append(f"| labeler ceiling (A vs B) | {c['raw_agreement']:.3f} | kappa {c['kappa']:.3f} | | |")
    return L


def headline(res: dict) -> dict:
    me = res["chosen"]
    out = {}
    for s in ("all", "eligibility_pass"):
        sl = res["slices"][s]
        out[s] = {"n": sl["n"], "postings": sl["postings"], "gold": sl["gold"],
                  "systems": {k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"],
                                  "keep_recall": v["per_label"]["Keep"]["recall"], "unsafe_keep": v["unsafe_keep"]}
                              for k, v in sl["systems"].items()},
                  "labeler_ceiling": sl.get("labeler_ceiling"),
                  "paired": {k: {kk: vv for kk, vv in sl["paired"][k].items()}
                             for k in (f"{me} - rules_skills_free", f"{me} - rules_full", f"{me} - d3+H'")
                             if k in sl["paired"]}}
    return out


def cmd_write(a: argparse.Namespace) -> None:
    res_path = OUT / "fresh_test.json"
    res = json.loads(res_path.read_text())
    me, p = res["chosen"], res["pass"]
    notes = Path(a.notes).read_text().strip()
    L = ["# v4 fresh test: the one scoring run", "",
         "Sealed fresh test: 592 rows, 150 postings from 33 employers used nowhere else; gold from two blind labelers "
         "(Opus, Sonnet) with blind adjudication. Scored once, as eval/protocol.json and eval/protocol_amendment_1.json "
         f"say; chosen system {me} (dev_fit pick, eval/selection_v4.json). Intervals: two-way bootstrap (postings, then "
         f"candidates), {res['draws']} draws, seed {res['seed']}, 95% percentile.", "",
         "## Verdict", "",
         f"{me} minus skills-free rules, macro-F1, eligibility-pass slice: {fmt(p['macro_f1_diff'])} "
         f"(lower bound {p['lower_bound_exact']:.4f}) -> **{'PASS' if p['passed'] else 'NOT PASSED'}**.", ""]
    for s, title in (("all", "All rows"), ("eligibility_pass", "Eligibility-pass slice (the pass slice)")):
        sl = res["slices"][s]
        L += [f"## {title}: n={sl['n']}, postings={sl['postings']}, gold {sl['gold']}", "", *system_table(sl), ""]
        L += ["Paired differences (same rows, same draws):", "", "| comparison | accuracy diff | macro-F1 diff | rows fixed / broken |",
              "|---|---|---|---|"]
        for k in (f"{me} - rules_skills_free", f"{me} - rules_full", f"{me} - d3+H'", f"{me} - base+H'"):
            d = sl["paired"].get(k)
            if d:
                L.append(f"| {k} | {fmt(d['accuracy_diff'])} | {fmt(d['macro_f1_diff'])} | {d['rows_fixed']} / {d['rows_broken']} |")
        L.append("")
    L += ["## Slices (reported, not for pass)", "",
          f"| slice | n | gold K/E/S | {me} macro-F1 | rules_skills_free macro-F1 | diff | {me} acc | labeler agreement |",
          "|---|---|---|---|---|---|---|---|"]
    for name, sl in res["slices"].items():
        if sl is None:
            L.append(f"| {name} | 0 | | | | | | |")
            continue
        d = sl["paired"][f"{me} - rules_skills_free"]["macro_f1_diff"]
        g = sl["gold"]
        c = sl.get("labeler_ceiling") or {}
        L.append(f"| {name} | {sl['n']} | {g.get('Keep', 0)}/{g.get('Explore', 0)}/{g.get('Skip', 0)} | "
                 f"{fmt(sl['systems'][me]['macro_f1'])} | {fmt(sl['systems']['rules_skills_free']['macro_f1'])} | "
                 f"{fmt(d)} | {sl['systems'][me]['accuracy'][0]:.3f} | {c.get('raw_agreement', '')} |")
    L += ["", notes, ""]
    REPORT.write_text("\n".join(L))
    files = {str(f.relative_to(ROOT)): sha256(f) for f in
             [*(OUT / f"{t}.jsonl" for t in TAGS), *(OUT / f"{t}_run/{t}_fresh_test.json" for t in TAGS), res_path]}
    out = {"what": "v4 sealed fresh test, scored once (eval/protocol.json, eval/protocol_amendment_1.json): headline "
                   "numbers and the pass verdict. Full per-slice result: data/models/v4_eval/fresh_test/fresh_test.json "
                   "(gitignored; the amendment's scoring.outputs names data/models/main_4b/fit_eval/, this run wrote next to the "
                   "per-row files instead, nothing else differs); report data/models/v4_eval/FRESH_TEST.md; Colab runs by "
                   "scripts/fresh_test_colab.sh (colab_eval.py --base-model Qwen/Qwen3.5-4B --dtype bfloat16 --batch 1 "
                   "--max-min 70, one model per session).",
           "chosen": me, "selection": "eval/selection_v4.json",
           "pass": p, "verdict": "PASS" if p["passed"] else "NOT PASSED",
           "headline": headline(res), "intervals": "[point, 2.5th, 97.5th]; two-way bootstrap, "
           f"{res['draws']} draws, seed {res['seed']}", "sha256 (gitignored files)": files,
           "colab_sessions": json.loads(Path(a.sessions).read_text()) if a.sessions else None}
    RESULT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", REPORT, RESULT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rows").set_defaults(fn=cmd_rows)
    sub.add_parser("errors").set_defaults(fn=cmd_errors)
    w = sub.add_parser("write")
    w.add_argument("--notes", required=True)
    w.add_argument("--sessions", help="JSON file: Colab session name -> what it did")
    w.set_defaults(fn=cmd_write)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
