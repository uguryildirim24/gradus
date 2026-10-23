"""Refit B's decider cut-offs on fresh tests 1-4 pooled (amendment 6, candidates.B "refit-B").

  refit_cutoffs_t1to4.py [--out eval/cutoffs_B_t1to4.json]

Rule: amendment 5b's cut-off fit, unchanged, run on 1,400 (tests 1-3) + 1,139 (test 4) = 2,539 pooled rows of
B + H''_without_work_mode, decider readout (s = P(Keep) + 0.5 P(Explore), softmax temperature 1 over the stored float32
first-token scores). Grid = np.quantile(s over the pooled rows, linspace(0, 1, 201)); every pair t_lo <= t_hi of grid
values; highest pooled macro-F1 after H'' among pairs with exact recall >= 0.60 Keep, 0.25 Explore, 0.25 Skip; ties on
macro-F1 rounded to 6 decimals: higher Keep recall, lower t_lo, lower t_hi. The fit itself is
fit_select_cycle3.fit_cutoffs, called as is.

Reads only stored files: data/models/c3_eval/selection_rows.jsonl (tests 1-3) and
data/models/c3_eval/test4/test4_rows.jsonl (test 4), each checked against the sha256 that the committed
selection_cycle3.json / fresh_test4_result.json record. Loads no model, calls nothing, and does not read fresh test 5.
Checks before fitting: tests 1-3 refit with the tests 1-3 rows alone must return the shipped cut-offs, and the shipped
cut-offs must reproduce every stored B decider label on all four tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import fit_select_cycle3 as S  # noqa: E402
from matcher.evalstats import LABELS  # noqa: E402

VARIANT = "H''_without_work_mode"
SYSTEM = f"B+{VARIANT}|decider"
SEL = ROOT / "eval/selection_cycle3.json"
T4R = ROOT / "eval/fresh_test4_result.json"
ROWS123 = ROOT / "data/models/c3_eval/selection_rows.jsonl"
ROWS4 = ROOT / "data/models/c3_eval/test4/test4_rows.jsonl"
AMENDS = ("eval/protocol_amendment_5b.json", "eval/protocol_amendment_6.json")
SELF = "scripts/refit_cutoffs_t1to4.py"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return str(Path(p).relative_to(ROOT))


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def metrics(g: np.ndarray, p: np.ndarray) -> dict:
    f1, rec = S.class_stats(g, p[None, :])
    pk, gk = p == 0, g == 0
    tp = int((pk & gk).sum())
    return {"macro_f1": float(f1[0]), "keep_recall": float(rec[0, 0]), "explore_recall": float(rec[0, 1]),
            "skip_recall": float(rec[0, 2]), "keep_found": tp, "keep_gold": int(gk.sum()),
            "keep_predicted": int(pk.sum()), "keep_precision": float(tp / pk.sum()) if pk.sum() else None,
            "true_keeps": tp, "wrong_keeps": int((pk & ~gk).sum()), "accuracy": float((g == p).mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "eval/cutoffs_B_t1to4.json"))
    out = Path(ap.parse_args().out)

    sel, t4 = json.loads(SEL.read_text()), json.loads(T4R.read_text())
    assert sha(ROWS123) == sel["per_row_file"]["sha256"], "selection_rows.jsonl is not the file selection_cycle3 records"
    assert sha(ROWS4) == t4["per_row_file"]["sha256"], "test4_rows.jsonl is not the file fresh_test4_result records"
    old = sel["decider_cut_offs"][f"B+{VARIANT}"]
    old_lo, old_hi = old["t_lo"], old["t_hi"]
    cats = S.VARIANTS[VARIANT]

    rows = load(ROWS123) + load(ROWS4)
    for r in rows:
        r["test"] = r.get("test", "fresh_test4")
    tests = ("fresh_test", "fresh_test2", "fresh_test3", "fresh_test4")
    Sraw = np.array([r["scores"]["B"] for r in rows], dtype=np.float64)
    s = S.decider_score(Sraw)
    assert np.allclose(s, [r["s"]["B"] for r in rows], rtol=0, atol=1e-12), "stored s differs from recomputed s"
    g = S.F.as_idx([r["gold"] for r in rows])
    hard, sal = S.override_masks(rows, cats)  # rows without a card hold rule None: no override
    elig = np.array([bool(r["eligibility_pass"]) for r in rows])
    tid = np.array([tests.index(r["test"]) for r in rows])

    # check 1: the shipped cut-offs reproduce every stored B decider label
    stored = S.F.as_idx([r["systems"][SYSTEM] for r in rows])
    shipped_lab = S.apply_h(S.decider_idx(s, old_lo, old_hi), hard, sal)
    assert (shipped_lab == stored).all(), "shipped cut-offs do not reproduce the stored B labels"
    # check 2: the fit, run on tests 1-3 alone, returns the shipped cut-offs
    m123 = tid < 3
    chk = S.fit_cutoffs(s[m123], g[m123], hard[m123], sal[m123])
    assert chk["t_lo"] == old_lo and chk["t_hi"] == old_hi, f"tests 1-3 refit {chk['t_lo']}, {chk['t_hi']} != shipped"

    fit = S.fit_cutoffs(s, g, hard, sal)
    assert fit["floors_met"] == "0.60/0.25/0.25", fit["floors_met"]
    new_lab = S.apply_h(S.decider_idx(s, fit["t_lo"], fit["t_hi"]), hard, sal)

    def block(mask: np.ndarray) -> dict:
        return {"n": int(mask.sum()), "gold": dict(Counter(LABELS[i] for i in g[mask])),
                "old_cutoffs": metrics(g[mask], shipped_lab[mask]), "new_cutoffs": metrics(g[mask], new_lab[mask])}

    pooled = {"all_rows_tests_1to4_pooled (the fit set)": block(np.ones(len(rows), bool)),
              "eligibility_pass_slice_tests_1to4_pooled": block(elig),
              **{f"{t}_all_rows": block(tid == k) for k, t in enumerate(tests)}}

    inputs = {rel(p): sha(p) for p in (ROWS123, ROWS4, SEL, T4R, ROOT / "scripts/fit_select_cycle3.py",
                                       *(ROOT / a for a in AMENDS))}
    for k, v in {**{f"{t}": x for t, x in sel["score_files"].items() if t.startswith("B/")},
                 "B/fresh_test4": t4["score_files"]["B"]}.items():
        p = ROOT / v["path"]
        assert sha(p) == v["sha256"], f"{v['path']} differs from its recorded sha256"
        inputs[v["path"]] = v["sha256"]
    res = {
        "what": "B's decider cut-offs refit on fresh tests 1-4 pooled: amendment 6's refit-B comparator for test 5",
        "rule": "amendment 5b's cut-off fit unchanged (see scripts/refit_cutoffs_t1to4.py docstring); amendment 6 adds "
                "only 'dev (tests 1-4 pooled) by amendment 5b's grid and rule'. Fit on all rows of tests 1-4 "
                "(the 5b rule fits on all pooled rows, not the eligibility-pass slice).",
        "system": SYSTEM,
        "script": SELF,
        "rows": {"pooled": len(rows), **{t: int((tid == k).sum()) for k, t in enumerate(tests)},
                 "gold_pooled": dict(Counter(r["gold"] for r in rows))},
        "old_cutoffs": {"t_lo": old_lo, "t_hi": old_hi, "source": "eval/selection_cycle3.json (fit on tests 1-3)"},
        "new_cutoffs": {"t_lo": fit["t_lo"], "t_hi": fit["t_hi"], "quantile_lo": fit["quantile_lo"],
                        "quantile_hi": fit["quantile_hi"], "grid_index_lo": fit["grid_index_lo"],
                        "grid_index_hi": fit["grid_index_hi"], "floors_met": fit["floors_met"],
                        "fit_macro_f1": fit["macro_f1"], "fit_recall": fit["recall"],
                        "pairs_tried": fit["pairs_tried"], "pairs_clearing_floors": fit["pairs_clearing_floors"]},
        "checks_passed": ["selection_rows.jsonl and test4_rows.jsonl match the sha256 recorded by selection_cycle3.json "
                          "and fresh_test4_result.json",
                          "stored s equals s recomputed from the stored B scores (atol 1e-12)",
                          "shipped cut-offs reproduce every stored B decider label on all 2,539 rows",
                          "the same fit on tests 1-3 alone returns the shipped cut-offs (0.27310 / 0.58904)"],
        "metrics": pooled,
        "inputs_sha256": inputs,
        "fresh_test5": "not read",
    }
    out.write_text(json.dumps(res, indent=1) + "\n")
    b = pooled["all_rows_tests_1to4_pooled (the fit set)"]
    print(f"old {old_lo:.5f} / {old_hi:.5f}   new {fit['t_lo']:.5f} / {fit['t_hi']:.5f}  ({fit['floors_met']})")
    for k in ("all_rows_tests_1to4_pooled (the fit set)", "eligibility_pass_slice_tests_1to4_pooled"):
        for w in ("old_cutoffs", "new_cutoffs"):
            m = pooled[k][w]
            print(f"{k[:30]:30s} {w:12s} mF1 {m['macro_f1']:.4f} rK {m['keep_recall']:.4f} ({m['keep_found']}/{m['keep_gold']}) "
                  f"pK {m['keep_precision']:.4f} wrongK {m['wrong_keeps']}")
    print("wrote", rel(out))


if __name__ == "__main__":
    main()
