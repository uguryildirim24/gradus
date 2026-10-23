"""Cycle-3 diagnostic (eval/protocol_amendment_4.json, diagnostics list: "h2n on fresh tests 2 and 3"): h2n on fresh
tests 1-3, raw and + H', beside h + H', rules_full and base + H'. Fresh tests 1-3 are practice sets under amendment 4;
their recorded results (eval/fresh_test*_result*.json) are read by nothing here and never written.

  diag_cycle3_h2n.py infer   h2n on fresh tests 2 and 3 on one Colab L4 session (colab_eval.py, the watchdog, bf16,
                             batch 1, h's prompt build, the same first-token float32 label scores as every earlier run);
                             a set whose result file exists is skipped. Run under `caffeinate -is`.
                             -> data/models/diag_cycle3/h2n/h2n_fresh_test{2,3}.json
  diag_cycle3_h2n.py score   every system on each test and pooled, all rows and the eligibility-pass slice, with
                             fit_eval_score.py's own loaders, systems and two-way bootstrap (DRAWS, SEED); paired
                             differences of h2n+H' and h2n raw against h+H', rules_full and base+H'
                             -> eval/diag_cycle3_h2n.json

Before anything runs or is scored: the h2n archive is the one eval/selection_cycle2.json records, each prompt file is
the one h, d3 and base were scored with (data/built/fit/eval_inputs/report*.json, eval/fit_eval_hashes.json), h2n's
fresh test 1 score file is the one the selection recorded, and each test's data passes the checks its scoring command
ran (fit_eval_score.check_hashes / check_fresh2 / check_fresh3). Rows load with the rules now in effect (load_fresh;
strict design check on tests 1 and 2, as their commands, none on test 3, as amendment 3). Loads no model on the Mac.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]

import fit_eval_score as F  # noqa: E402

OUT = ROOT / "data/models/diag_cycle3/h2n"
RESULT = ROOT / "eval/diag_cycle3_h2n.json"
SELECTION = ROOT / "eval/selection_cycle2.json"
ARCHIVE = ROOT / "data/models/main_4b/h2n_0a978d55-5a40-4de4-86c9-e2558c5a3607.tar.gz"
V4 = ROOT / "data/models/v4_eval"
INPUTS = F.INPUTS / "h"  # h2n trains on train_fit.csv prompts unchanged: h's build (cycle2_colab.sh FMT)
INPUT_PINS = {"fresh_test": (F.HASHES, "sha256", "data/built/fit/eval_inputs/h/fresh_test.jsonl"),
              "fresh_test2": (F.INPUTS / "report_fresh_test2.json", "files", "h/fresh_test2.jsonl"),
              "fresh_test3": (F.INPUTS / "report_fresh_test3.json", "files", "h/fresh_test3.jsonl")}
SETS = {"fresh_test": F.FRESH, "fresh_test2": F.FRESH2, "fresh_test3": F.FRESH3}
REFS = ("h+H'", "rules_full", "base+H'")
MINE = ("h2n+H'", "h2n raw")
SHOWN = ("h2n raw", "h2n+H'", "h raw", "h+H'", "d3+H'", "base+H'", "rules_full")


def pinned_input(s: str) -> str:
    f, key, name = INPUT_PINS[s]
    d = json.loads(f.read_text())
    return d[key][name]


def check_inputs() -> None:
    if F.sha256(ARCHIVE) != json.loads(SELECTION.read_text())["archives"]["h2n"]["sha256"]:
        raise SystemExit(f"{ARCHIVE.name} is not the h2n archive {SELECTION.name} records")
    for s in SETS:
        p = INPUTS / f"{s}.jsonl"
        if F.sha256(p) != pinned_input(s):
            raise SystemExit(f"{p.relative_to(ROOT)} is not the prompt file h was scored with")


def score_path(s: str, tag: str) -> Path:
    if tag == "h2n":
        return V4 / "fresh_test/h2n_run/h2n_fresh_test.json" if s == "fresh_test" else OUT / f"h2n_{s}.json"
    return V4 / s / f"{tag}_run/{tag}_{s}.json"


def cmd_infer(a: argparse.Namespace) -> None:
    check_inputs()
    todo = [s for s in ("fresh_test2", "fresh_test3") if not score_path(s, "h2n").exists()]
    if not todo:
        print("h2n already scored on fresh tests 2 and 3")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "eval_h2n.log"
    if log.exists():
        n = 1
        while (OUT / f"eval_h2n_try{n}.log").exists():
            n += 1
        log.rename(OUT / f"eval_h2n_try{n}.log")
    cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/colab_eval.py"), "matcher-h2n-diag3",
           "--base-model", "Qwen/Qwen3.5-4B", "--model", f"h2n={ARCHIVE.relative_to(ROOT)}"]
    for s in todo:
        cmd += ["--dev", str((INPUTS / f"{s}.jsonl").relative_to(ROOT))]
    cmd += ["--out", str(OUT.relative_to(ROOT)), "--dtype", "bfloat16", "--batch", "1", "--max-min", "70"]
    with log.open("w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"h2n {todo} exit {rc}; missing now: {[s for s in todo if not score_path(s, 'h2n').exists()]}")


def check_set(s: str) -> list[dict]:
    if s == "fresh_test":
        F.check_hashes(["data/built/fresh_test/rows.jsonl", "data/built/fresh_test/cards.jsonl",
                        "data/built/fresh_test/gold.jsonl"])
        return F.load_fresh(F.FRESH)
    if s == "fresh_test2":
        F.check_fresh2()
        return F.load_fresh(F.FRESH2)
    F.check_fresh3()
    return F.load_fresh(F.FRESH3, strict_design=False)


def keep_view(m: dict) -> dict:
    k = m["per_label"]["Keep"]
    return {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "keep_recall": k["recall"],
            "keep_precision": k["precision"], "keep_pred": k["pred"], "keep_gold": k["gold"],
            "unsafe_keep": m["unsafe_keep"]}


def score_block(rows: list[dict], P: dict[str, list[str]]) -> dict:
    diffs = [(a, b) for a in MINE for b in REFS]
    out = {}
    for name, mask in (("all", [True] * len(rows)), ("eligibility_pass", [r["eligibility_pass"] for r in rows])):
        r = F.score_slice(rows, {k: P[k] for k in SHOWN}, mask, diffs)
        out[name] = {"n": r["n"], "postings": r["postings"], "gold": r["gold"],
                     "systems": {k: keep_view(v) for k, v in r["systems"].items()},
                     "per_label": {k: r["systems"][k]["per_label"] for k in MINE},
                     "paired": {k: {"macro_f1_diff": v["macro_f1_diff"], "low_exact": v["macro_f1_diff_low_exact"],
                                    "accuracy_diff": v["accuracy_diff"], "rows_fixed": v["rows_fixed"],
                                    "rows_broken": v["rows_broken"]} for k, v in r["paired"].items()},
                     "confusion_gold_to_pred": {k: {f"{g}->{q}": n for (g, q), n in sorted(Counter(
                         (rows[i]["gold"], P[k][i]) for i, m in enumerate(mask) if m).items())}
                         for k in ("h2n+H'", "h+H'")},
                     "labeler_ceiling": r.get("labeler_ceiling")}
    return out


def cmd_score(a: argparse.Namespace) -> None:
    check_inputs()
    sel = json.loads(SELECTION.read_text())
    if F.sha256(score_path("fresh_test", "h2n")) != sel["score_files"]["h2n"]["sha256"]:
        raise SystemExit("h2n's fresh test 1 score file is not the one eval/selection_cycle2.json records")
    res = {"what": "cycle-3 diagnostic, eval/protocol_amendment_4.json: h2n on fresh tests 1-3 (practice sets), raw "
                   "and + H', beside h+H', rules_full and base+H'; no recorded fresh*_result file is read or written",
           "amendment": "eval/protocol_amendment_4.json", "draws": F.DRAWS, "seed": F.SEED,
           "inference": {"where": "Colab L4 under the watchdog (colab_eval.py), one adapter per session",
                         "dtype": "bfloat16", "batch": 1, "prompt_build": "eval_inputs/h (h's format)",
                         "label_read": "first answer token, float32 label scores, fit_eval_score.read_scores",
                         "archive": {"file": ARCHIVE.name, "sha256": F.sha256(ARCHIVE)}},
           "rows_loaded": "fit_eval_score.load_fresh with the rules in effect; strict design check on tests 1 and 2, "
                          "none on test 3 (amendment 3)",
           "pooled": "rows of tests 1-3 concatenated (no posting shared across tests); two-way bootstrap over all "
                     "postings, unstratified",
           "sets": {}, "score_files": {}, "input_files": {}}
    all_rows, all_P = [], {k: [] for k in SHOWN}
    for s in SETS:
        rows = check_set(s)
        ids = [r["row_id"] for r in rows]
        paths = {t: score_path(s, t) for t in ("h2n", "h", "d3", "base")}
        models = {t: F.read_scores(p, ids) for t, p in paths.items()}
        P = F.systems(rows, models)
        res["sets"][s] = score_block(rows, P)
        res["score_files"][s] = {t: {"path": str(p.relative_to(ROOT)), "sha256": F.sha256(p)} for t, p in paths.items()}
        res["input_files"][s] = {"path": str((INPUTS / f"{s}.jsonl").relative_to(ROOT)), "sha256": pinned_input(s)}
        all_rows += rows
        for k in SHOWN:
            all_P[k] += P[k]
    res["sets"]["pooled"] = score_block(all_rows, all_P)
    RESULT.write_text(json.dumps(res, indent=1) + "\n")
    print(table(res))


def table(res: dict) -> str:
    lines = []
    for sl in ("all", "eligibility_pass"):
        lines.append(f"\n[{sl}]  macro-F1 [95%]  | acc | Keep recall | unsafe Keep")
        for s, b in res["sets"].items():
            x = b[sl]
            lines.append(f"{s} (n={x['n']}, postings={x['postings']}, gold Keep={x['gold'].get('Keep', 0)})")
            for k, m in x["systems"].items():
                f = m["macro_f1"]
                lines.append(f"  {k:9s} {f[0]:.4f} [{f[1]:.4f}, {f[2]:.4f}]  {m['accuracy'][0]:.4f}  "
                             f"{m['keep_recall'] if m['keep_recall'] is not None else '-':<7}  {m['unsafe_keep']}")
            for k, d in x["paired"].items():
                f = d["macro_f1_diff"]
                lines.append(f"    {k:22s} dF1 {f[0]:+.4f} [{f[1]:+.4f}, {f[2]:+.4f}]  dacc {d['accuracy_diff'][0]:+.4f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("infer").set_defaults(fn=cmd_infer)
    sub.add_parser("score").set_defaults(fn=cmd_score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
