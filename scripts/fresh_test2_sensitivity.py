"""Fresh test 2 sensitivities, from the stored per-row score files; rescoring no model and changing no verdict.

  1. The headline comparisons (h+H' minus rules_full, h+H' minus base+H'; macro-F1 and accuracy, two-way bootstrap,
     pinned DRAWS and SEED) on the sealed-design eligibility-pass slice (rows.jsonl design.eligibility_pass, 214 rows)
     instead of the recomputed slice amendment 2 scores (164 rows).
  2. The 164-row headline at seeds 1, 2, 3, 4, 7 (DRAWS draws) and at 10,000 draws with the pinned seed.

The pinned verdict is eval/fresh_test2_result.json; this writes eval/fresh_test2_sensitivity.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import fit_eval_score as F  # noqa: E402

RUNS = ROOT / "data/models/v4_eval/fresh_test2"
OUT = ROOT / "eval/fresh_test2_sensitivity.json"
ME = "h+H'"
AGAINST = ("rules_full", "base+H'")
SEEDS = (1, 2, 3, 4, 7)
BIG_DRAWS = 10_000


def headline(rows, P, mask, draws, seed) -> dict:
    s = F.score_slice(rows, P, mask, [(ME, b) for b in AGAINST], draws, seed)
    out = {"n": s["n"], "postings": s["postings"], "draws": draws, "seed": seed}
    for b in AGAINST:
        d = s["paired"][f"{ME} - {b}"]
        out[f"{ME} - {b}"] = {"macro_f1_diff": d["macro_f1_diff"], "macro_f1_lower_bound_exact": d["macro_f1_diff_low_exact"],
                             "accuracy_diff": d["accuracy_diff"], "lower_bound_above_0": bool(d["macro_f1_diff_low_exact"] > 0)}
    return out


def main() -> None:
    F.check_fresh2()
    rows = F.load_fresh(F.FRESH2)
    ids = [r["row_id"] for r in rows]
    models = {t: F.read_scores(RUNS / f"{t}_run/{t}_fresh_test2.json", ids) for t in ("h", "d3", "base")}
    P = F.systems(rows, models)
    recomputed = [r["eligibility_pass"] for r in rows]
    design = [r["eligibility_pass_design"] for r in rows]

    pinned = headline(rows, P, recomputed, F.DRAWS, F.SEED)
    pinned_file = json.loads(F.FRESH2_RESULT.read_text())["pass"]["against"]
    for b in AGAINST:  # the pinned run must reproduce exactly before any sensitivity is trusted
        if pinned[f"{ME} - {b}"]["macro_f1_lower_bound_exact"] != pinned_file[b]["lower_bound_exact"]:
            raise SystemExit(f"{b}: recomputed pinned lower bound differs from eval/fresh_test2_result.json")

    res = {
        "set": "fresh_test2",
        "status": "sensitivities only; the pinned verdict in eval/fresh_test2_result.json (NOT PASSED: h+H' minus "
                  "rules_full macro-F1 lower bound -0.0018 on the 164-row recomputed eligibility-pass slice, 2000 draws, "
                  "seed 20261021) stands. Nothing here rescored a model or changes that verdict.",
        "source": "stored per-row label scores data/models/v4_eval/fresh_test2/{h,d3,base}_run/*_fresh_test2.json, "
                  "data/built/fresh_test2/{gold,rows,cards}.jsonl (hashes checked by fit_eval_score.check_fresh2), "
                  "functions of scripts/fit_eval_score.py; script scripts/fresh_test2_sensitivity.py",
        "pinned_reproduced": pinned,
        "sealed_design_slice": {
            "slice": "rows.jsonl design.eligibility_pass (the sealed design flag, before the cycle-2 work-mode fix)",
            "note": "the 50 extra rows are the ones the cycle-2 work-mode fix makes a hard Skip; H' and rules_full both "
                    "Skip them, base+H' too, so they shift the comparison by how often gold is Skip there (32 of 50)",
            **headline(rows, P, design, F.DRAWS, F.SEED)},
        "reading": "the h+H' minus rules_full macro-F1 lower bound sits at about 0 on fresh test 2: the pinned seed "
                   "gives -0.0018, seed 4 gives -0.0009, seeds 1, 2, 3, 7 and 10,000 draws give +0.004 to +0.009, and "
                   "the sealed-design 214-row slice gives +0.0013. The point estimate (+0.152) does not move with seed. "
                   "The result is a near-miss on Monte Carlo noise; amendment 2 pinned seed and draws before scoring, "
                   "so the verdict stays NOT PASSED. h+H' minus base+H' clears 0 by about 0.18 or more everywhere.",
        "recomputed_slice_seeds_and_draws": [headline(rows, P, recomputed, F.DRAWS, s) for s in SEEDS]
                                            + [headline(rows, P, recomputed, BIG_DRAWS, F.SEED)],
    }
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
