"""Cycle-3 CPU diagnostics on fresh tests 1-3 (practice sets under eval/protocol_amendment_4.json), from stored scores.

  diag_cycle3_cpu.py [--out data/models/diag_cycle3/cpu] [--summary eval/diag_cycle3_cpu.json]

Reads the stored first-token label scores of h, d3 and base (the *_run/*.json score files each recorded result used)
and the three tests' rows through scripts/fit_eval_score.py (load_fresh with the rules in effect now, read_scores,
systems, metrics, paired, two_way_draws). Writes nothing under the write-once eval/fresh*_result paths and loads no
model.

  1. copy rate: model raw label == rules_full, all rows and the eligibility-pass slice, with the model x rules
     confusion; gold == rules_full beside it as the reference, and who is right where model and rules disagree.
  2. eligibility-layer ablation: raw, full H' (HP_CATS + salary cap), H' on work_auth + enrollment only, and that
     plus work_mode (the two reduced variants without the salary cap), paired against raw; a leave-one-category-out
     pass over full H' and each category's override precision beside it.
  3. prior correction: additive offsets on the Keep and Explore scores (Skip fixed at 0), grid -10..10 step 0.1,
     chosen for macro-F1 of the raw label on all rows of the fit set (ties: smallest L2 norm, then smallest Keep
     offset, then smallest Explore offset), the label read with evalstats.pick; fit on test 1 -> applied to 2 and 3,
     fit on 1+2 (pooled rows) -> applied to 3. The analytic shift log(gold share of the fit set / 1/3) is reported
     beside it.

Pooled numbers concatenate the three tests; the pooled bootstrap draws each test's postings (then rows) on its own
with seed SEED + k for test k and concatenates the index arrays, so no posting crosses tests. Intervals are 95%
percentile over DRAWS draws, shared by every system on that slice.
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

import fit_eval_score as F  # noqa: E402
from hybrid_rules import HP_CATS, hybrid  # noqa: E402
from matcher.evalstats import LABELS, macro_f1_of  # noqa: E402

MODELS = ("h", "d3", "base")
V4 = ROOT / "data/models/v4_eval"
TESTS = {
    "t1": {"dir": F.FRESH, "strict": True, "result": ROOT / "eval/fresh_test_result_v4.json",
           "scores": {m: V4 / f"fresh_test/{m}_run/{m}_fresh_test.json" for m in MODELS}},
    "t2": {"dir": F.FRESH2, "strict": True, "result": ROOT / "eval/fresh_test2_result.json",
           "scores": {m: V4 / f"fresh_test2/{m}_run/{m}_fresh_test2.json" for m in MODELS}},
    "t3": {"dir": F.FRESH3, "strict": False, "result": ROOT / "eval/fresh_test3_result.json",
           "scores": {m: V4 / f"fresh_test3/{m}_run/{m}_fresh_test3.json" for m in MODELS}},
}
OUT = ROOT / "data/models/diag_cycle3/cpu"
SUMMARY = ROOT / "eval/diag_cycle3_cpu.json"
SEED, DRAWS = F.SEED, F.DRAWS
AUTH_ENROLL = frozenset({"work_auth", "enrollment"})
AUTH_ENROLL_MODE = AUTH_ENROLL | {"work_mode"}
VARIANTS = {"raw": None, "H'": (HP_CATS, True), "auth+enroll": (AUTH_ENROLL, False),
            "auth+enroll+mode": (AUTH_ENROLL_MODE, False)}
GRID = np.round(np.arange(-10.0, 10.0001, 0.1), 1)
TRAIN_SHARE = 1 / 3  # h's train_fit.csv: 3,521 each of Keep, Explore and Skip (scripts/train_main.py)
SLICES = ("all", "eligibility_pass")


def r3(x):
    return None if x is None else round(float(x), 3)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_raw_scores(path: Path, ids: list[str], labels: dict[str, str]) -> np.ndarray:
    """rows x (Keep, Explore, Skip) float scores in `ids` order; each row's pick must equal read_scores' label."""
    d = json.loads(path.read_text())
    by = {r.get("pair_id", r.get("row_id")): [float(x) for x in r["scores"]] for r in d["rows"]}
    S = np.array([by[i] for i in ids], dtype=np.float64)
    if any(LABELS[F.pick(s)] != labels[i] for s, i in zip(S, ids)):
        raise SystemExit(f"{path}: scores and read_scores disagree")
    return S


def pick_all(S: np.ndarray) -> np.ndarray:
    """evalstats.pick on every row, vectorised: argmax with an exact tie to Skip, then Explore, then Keep."""
    rev = S[:, ::-1]  # Skip, Explore, Keep: argmax takes the first maximum
    return 2 - np.argmax(rev, axis=1)


# ---------------------------------------------------------------- loading


def load() -> dict:
    tests = {}
    for t, spec in TESTS.items():
        rows = F.load_fresh(spec["dir"], strict_design=spec["strict"])
        ids = [r["row_id"] for r in rows]
        labs = {m: F.read_scores(p, ids) for m, p in spec["scores"].items()}
        S = {m: read_raw_scores(p, ids, labs[m]) for m, p in spec["scores"].items()}
        for r in rows:
            r["posting_key"] = f"{t}:{r['posting_id']}"
        tests[t] = {"rows": rows, "labels": labs, "S": S,
                    "sha256": {m: sha256(p) for m, p in spec["scores"].items()}}
    return tests


def recorded_hashes(t: str) -> dict:
    d = json.loads(TESTS[t]["result"].read_text())
    if "score_files" in d:
        return {m: v["sha256"] for m, v in d["score_files"].items()}
    h = d.get("sha256 (gitignored files)") or {}
    out = {}
    for m in MODELS:
        k = f"data/models/v4_eval/fresh_test/{m}_run/{m}_fresh_test.json"
        if k in h:
            out[m] = h[k]
    return out


def reproduction(tests: dict) -> dict:
    """Recorded point values of h/d3/base raw and +H' against the same computed here (rules in effect now)."""
    out = {}
    for t, x in tests.items():
        d = json.loads(TESTS[t]["result"].read_text())
        rec = d["slices"] if "slices" in d else d["intervals"] if isinstance(d.get("intervals"), dict) \
            and "all" in d["intervals"] else d["headline"]
        P = F.systems(x["rows"], x["labels"])
        g = F.as_idx([r["gold"] for r in x["rows"]])
        ok = {}
        for sl in SLICES:
            if sl not in rec or "systems" not in rec[sl]:
                continue
            mask = np.array([True] * len(g)) if sl == "all" else np.array([r["eligibility_pass"] for r in x["rows"]])
            for k in ("h raw", "h+H'", "d3 raw", "d3+H'", "base raw", "base+H'", "rules_full"):
                if k not in rec[sl]["systems"]:
                    continue
                p = F.as_idx(P[k])
                now = (round(float((g[mask] == p[mask]).mean()), 4), round(macro_f1_of(g[mask], p[mask]), 4))
                was = (rec[sl]["systems"][k]["accuracy"][0], rec[sl]["systems"][k]["macro_f1"][0])
                ok[f"{sl} | {k}"] = {"recorded_acc_f1": list(was), "now_acc_f1": list(now), "match": now == tuple(was),
                                     "n_recorded": rec[sl].get("n"), "n_now": int(mask.sum())}
        rh = recorded_hashes(t)
        out[t] = {"score_file_sha256_match": {m: rh.get(m) == x["sha256"][m] if m in rh else None for m in MODELS},
                  "values": ok}
    return out


# ---------------------------------------------------------------- pooled bootstrap


def pooled_draws(parts: list[list[dict]], seeds: list[int], draws: int = DRAWS) -> list[np.ndarray]:
    """Per-test two-way draws (postings, then rows) concatenated with each test's row offset."""
    per = [F.two_way_draws([r["posting_id"] for r in rows], draws, s) for rows, s in zip(parts, seeds)]
    offs = np.cumsum([0] + [len(rows) for rows in parts[:-1]])
    return [np.concatenate([p[i] + o for p, o in zip(per, offs)]) for i in range(draws)]


def keep_recall(g: np.ndarray, p: np.ndarray) -> float | None:
    k = g == 0
    return float((p[k] == 0).mean()) if k.any() else None


def stat_ci(f, g, p, draws) -> list:
    pt = f(g, p)
    vals = [v for v in (f(g[i], p[i]) for i in draws) if v is not None]
    return [r3(pt), r3(np.percentile(vals, 2.5)), r3(np.percentile(vals, 97.5))]


def summary_block(g: np.ndarray, p: np.ndarray, draws: list | None) -> dict:
    acc = lambda a, b: float((a == b).mean())  # noqa: E731
    if draws is None:
        return {"macro_f1": r3(macro_f1_of(g, p)), "accuracy": r3(acc(g, p)), "keep_recall": r3(keep_recall(g, p)),
                "pred_mix": mix(p)}
    return {"macro_f1": stat_ci(macro_f1_of, g, p, draws), "accuracy": stat_ci(acc, g, p, draws),
            "keep_recall": stat_ci(keep_recall, g, p, draws), "pred_mix": mix(p)}


def paired_diff(g, a, b, draws) -> dict:
    d_f1, d_kr = [], []
    for i in draws:
        g_ = g[i]
        d_f1.append(macro_f1_of(g_, a[i]) - macro_f1_of(g_, b[i]))
        ka, kb = keep_recall(g_, a[i]), keep_recall(g_, b[i])
        if ka is not None:
            d_kr.append(ka - kb)
    pt_f1 = macro_f1_of(g, a) - macro_f1_of(g, b)
    pt_kr = keep_recall(g, a) - keep_recall(g, b)
    return {"macro_f1_diff": [r3(pt_f1), r3(np.percentile(d_f1, 2.5)), r3(np.percentile(d_f1, 97.5))],
            "keep_recall_diff": [r3(pt_kr), r3(np.percentile(d_kr, 2.5)), r3(np.percentile(d_kr, 97.5))],
            "rows_fixed": int(((a == g) & (b != g)).sum()), "rows_broken": int(((a != g) & (b == g)).sum())}


def mix(p: np.ndarray) -> dict:
    n = len(p)
    return {LABELS[w]: r3((p == w).mean()) for w in range(3)} | {"n": n}


# ---------------------------------------------------------------- sets


def sets(tests: dict) -> dict:
    """name -> (rows, {model: scores}, draws): each test (own two-way draws, SEED) and the pooled set."""
    out = {}
    for t, x in tests.items():
        out[t] = (x["rows"], x["S"], F.two_way_draws([r["posting_id"] for r in x["rows"]], DRAWS, SEED))
    parts = [tests[t]["rows"] for t in TESTS]
    rows = sum(parts, [])
    S = {m: np.concatenate([tests[t]["S"][m] for t in TESTS]) for m in MODELS}
    out["pooled"] = (rows, S, pooled_draws(parts, [SEED + k for k in range(len(parts))]))
    return out


def subset(rows, S, draws, mask):
    """Rows, scores and draws restricted to mask (draw indices kept only where they land in the slice)."""
    idx = np.flatnonzero(mask)
    pos = -np.ones(len(rows), dtype=int)
    pos[idx] = np.arange(len(idx))
    sub_d = [pos[i][pos[i] >= 0] for i in draws] if draws is not None else None
    return [rows[i] for i in idx], {m: s[idx] for m, s in S.items()}, sub_d


# ---------------------------------------------------------------- 1. copy rate


def confusion(a: np.ndarray, b: np.ndarray) -> dict:
    return {LABELS[i]: {LABELS[j]: int(((a == i) & (b == j)).sum()) for j in range(3)} for i in range(3)}


def copy_rate(all_sets: dict) -> dict:
    out = {}
    for name, (rows, S, draws) in all_sets.items():
        out[name] = {}
        for sl in SLICES:
            mask = np.ones(len(rows), bool) if sl == "all" else np.array([r["eligibility_pass"] for r in rows])
            rr, SS, dd = subset(rows, S, draws, mask)
            if name != "pooled":
                dd = None
            g = F.as_idx([r["gold"] for r in rr])
            rf = F.as_idx([r["rules_full"] for r in rr])
            eq = lambda a, b: float((a == b).mean())  # noqa: E731
            blk = {"n": len(rr), "gold_mix": mix(g), "rules_full_mix": mix(rf),
                   "gold_eq_rules_full": stat_ci(eq, g, rf, dd) if dd else r3(eq(g, rf))}
            for m in MODELS:
                p = pick_all(SS[m])
                dis = p != rf
                blk[m] = {"copy_rate": stat_ci(eq, p, rf, dd) if dd else r3(eq(p, rf)),
                          "kappa_vs_rules_full": F.kappa([LABELS[i] for i in p], [LABELS[i] for i in rf]),
                          "confusion_model_rows_x_rules_cols": confusion(p, rf),
                          "disagreements": int(dis.sum()),
                          "on_disagreements_model_right": int(((p == g) & dis).sum()),
                          "on_disagreements_rules_right": int(((rf == g) & dis).sum()),
                          "on_agreements_accuracy": r3((p[~dis] == g[~dis]).mean()) if (~dis).any() else None}
            out[name][sl] = blk
    return out


# ---------------------------------------------------------------- 2. ablation


def apply_variant(raw: np.ndarray, rows: list[dict], v) -> np.ndarray:
    if v is None:
        return raw
    cats, cap = v
    return F.as_idx([hybrid(LABELS[m], r["verdict"], cats, cap) for m, r in zip(raw, rows)])


def ablation(all_sets: dict) -> dict:
    out = {}
    cats_all = sorted(HP_CATS)
    for name, (rows, S, draws) in all_sets.items():
        g = F.as_idx([r["gold"] for r in rows])
        dd = draws if name == "pooled" else None
        blk = {"n": len(rows)}
        for m in MODELS:
            raw = pick_all(S[m])
            P = {k: apply_variant(raw, rows, v) for k, v in VARIANTS.items()}
            blk[m] = {k: summary_block(g, p, dd) for k, p in P.items()}
            if name == "pooled":
                blk[m]["paired_vs_raw"] = {k: paired_diff(g, P[k], raw, draws) for k in VARIANTS if k != "raw"}
                loo = {}
                for c in cats_all:
                    p = apply_variant(raw, rows, (HP_CATS - {c}, True))
                    loo[f"H' without {c}"] = {"macro_f1_minus_full_H'": r3(macro_f1_of(g, p) - macro_f1_of(g, P["H'"])),
                                              "keep_recall_minus_full_H'": r3(keep_recall(g, p) - keep_recall(g, P["H'"]))}
                p = apply_variant(raw, rows, (HP_CATS, False))
                loo["H' without salary cap"] = {"macro_f1_minus_full_H'": r3(macro_f1_of(g, p) - macro_f1_of(g, P["H'"])),
                                                "keep_recall_minus_full_H'": r3(keep_recall(g, p) - keep_recall(g, P["H'"]))}
                blk[m]["leave_one_out"] = loo
        if name == "pooled":
            blk["override_precision"] = override_precision(rows, {m: pick_all(S[m]) for m in MODELS})
        out[name] = blk
    return out


def override_precision(rows: list[dict], raws: dict) -> dict:
    """Per H' category: rows where the rules Skip with that hard category, their gold mix, and how many of them each
    raw model already called Skip (the override only changes the rest)."""
    out = {}
    for c in sorted(HP_CATS):
        idx = [i for i, r in enumerate(rows) if r["verdict"]["rule"] == "Skip" and c in r["verdict"]["hard_cats"]]
        only = [i for i in idx if set(rows[i]["verdict"]["hard_cats"]) & HP_CATS == {c}]
        gold = Counter(rows[i]["gold"] for i in idx)
        out[c] = {"rows": len(idx), "rows_this_category_alone": len(only), "gold": dict(gold),
                  "gold_skip_share": r3(gold["Skip"] / len(idx)) if idx else None,
                  "changed_by_override": {m: int(sum(raws[m][i] != 2 for i in idx)) for m in raws},
                  "changed_and_gold_skip": {m: int(sum(raws[m][i] != 2 and rows[i]["gold"] == "Skip" for i in idx))
                                            for m in raws}}
    sal = [i for i, r in enumerate(rows) if r["verdict"]["salary_soft"]]
    out["salary_cap"] = {"rows": len(sal), "gold": dict(Counter(rows[i]["gold"] for i in sal)),
                         "model_keep_capped": {m: int(sum(raws[m][i] == 0 for i in sal)) for m in raws},
                         "capped_and_gold_keep": {m: int(sum(raws[m][i] == 0 and rows[i]["gold"] == "Keep" for i in sal))
                                                  for m in raws}}
    return out


# ---------------------------------------------------------------- 3. prior correction


def grid_fit(S: np.ndarray, g: np.ndarray) -> dict:
    """Best (Keep, Explore) offsets on GRID for macro-F1 of the raw pick; ties by smallest L2 norm, then Keep, then
    Explore offset."""
    F_ = np.zeros((len(GRID), len(GRID)))
    gk, ge, gs = g == 0, g == 1, g == 2
    for a, bk in enumerate(GRID):
        # rows x len(GRID) Explore offsets
        k = S[:, 0][:, None] + bk
        e = S[:, 1][:, None] + GRID[None, :]
        s = np.broadcast_to(S[:, 2][:, None], k.shape)
        pred = np.where((s >= e) & (s >= k), 2, np.where(e >= k, 1, 0))  # ties: Skip, then Explore, then Keep
        f1 = np.zeros(len(GRID))
        for w, gm in ((0, gk), (1, ge), (2, gs)):
            pw = pred == w
            tp = (pw & gm[:, None]).sum(0)
            fp = (pw & ~gm[:, None]).sum(0)
            fn = gm.sum() - tp
            f1 += np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
        F_[a] = f1 / 3
    mx = F_.max()
    cand = np.argwhere(np.isclose(F_, mx, rtol=0, atol=1e-12))
    key = sorted(cand.tolist(), key=lambda ij: (GRID[ij[0]] ** 2 + GRID[ij[1]] ** 2, GRID[ij[0]], GRID[ij[1]]))
    i, j = key[0]
    return {"offset_keep": float(GRID[i]), "offset_explore": float(GRID[j]), "fit_macro_f1": r3(mx),
            "fit_macro_f1_at_zero": r3(F_[np.flatnonzero(GRID == 0)[0], np.flatnonzero(GRID == 0)[0]]),
            "plateau_grid_points": len(cand),
            "plateau_keep_range": [float(GRID[cand[:, 0].min()]), float(GRID[cand[:, 0].max()])],
            "plateau_explore_range": [float(GRID[cand[:, 1].min()]), float(GRID[cand[:, 1].max()])]}


def shift(S: np.ndarray, ok: float, oe: float) -> np.ndarray:
    return S + np.array([ok, oe, 0.0])


def analytic_offsets(g: np.ndarray) -> tuple[float, float]:
    """log(gold share / train share) for each label, relative to Skip (a Bayes prior shift on the 3-way softmax)."""
    sh = np.array([(g == w).mean() for w in range(3)])
    lo = np.log(sh / TRAIN_SHARE)
    return float(lo[0] - lo[2]), float(lo[1] - lo[2])


def prior_correction(tests: dict) -> dict:
    plans = {"fit_t1": (["t1"], ["t2", "t3"]), "fit_t1+t2": (["t1", "t2"], ["t3"])}
    out = {"grid": [float(GRID[0]), float(GRID[-1]), 0.1], "objective": "macro-F1 of the raw label, all rows of the fit set",
           "tie_break": "smallest L2 norm of (Keep, Explore) offsets, then smallest Keep, then smallest Explore",
           "train_share_assumed": "1/3 each (h's train_fit.csv); d3 and base reported with the same analytic shift for "
                                  "comparison only"}
    for plan, (fit, apply) in plans.items():
        frows = sum((tests[t]["rows"] for t in fit), [])
        fg = F.as_idx([r["gold"] for r in frows])
        arows = sum((tests[t]["rows"] for t in apply), [])
        ag = F.as_idx([r["gold"] for r in arows])
        adraws = pooled_draws([tests[t]["rows"] for t in apply], [SEED + list(TESTS).index(t) for t in apply])
        blk = {"fit_on": fit, "apply_to": apply}
        for m in MODELS:
            fS = np.concatenate([tests[t]["S"][m] for t in fit])
            gf = grid_fit(fS, fg)
            ak, ae = analytic_offsets(fg)
            res = {"grid_fit": gf, "analytic_offsets": {"keep": r3(ak), "explore": r3(ae)}}
            for label, (ok, oe) in (("grid", (gf["offset_keep"], gf["offset_explore"])), ("analytic", (ak, ae))):
                per = {}
                for t in apply:
                    rows = tests[t]["rows"]
                    g = F.as_idx([r["gold"] for r in rows])
                    raw = pick_all(tests[t]["S"][m])
                    cor = pick_all(shift(tests[t]["S"][m], ok, oe))
                    per[t] = {"before": summary_block(g, raw, None), "after": summary_block(g, cor, None),
                              "after_plus_H'": summary_block(g, apply_variant(cor, rows, VARIANTS["H'"]), None),
                              "before_plus_H'": summary_block(g, apply_variant(raw, rows, VARIANTS["H'"]), None)}
                aS = np.concatenate([tests[t]["S"][m] for t in apply])
                raw, cor = pick_all(aS), pick_all(shift(aS, ok, oe))
                pool = {"before": summary_block(ag, raw, adraws), "after": summary_block(ag, cor, adraws),
                        "after_minus_before": paired_diff(ag, cor, raw, adraws)}
                rawH = apply_variant(raw, arows, VARIANTS["H'"])
                corH = apply_variant(cor, arows, VARIANTS["H'"])
                pool["plus_H'"] = {"before": summary_block(ag, rawH, adraws), "after": summary_block(ag, corH, adraws),
                                   "after_minus_before": paired_diff(ag, corH, rawH, adraws)}
                if len(apply) > 1:
                    per["pooled_apply"] = pool
                else:
                    per[apply[0]]["with_intervals"] = pool
                res[label] = per
            blk[m] = res
        out[plan] = blk
    return out


# ---------------------------------------------------------------- text tables


def ci(x) -> str:
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]" if isinstance(x, list) else ("  -  " if x is None else f"{x:.3f}")


def auc_keep(S: np.ndarray, g: np.ndarray) -> float:
    mg = S[:, 0] - np.maximum(S[:, 1], S[:, 2])
    pos, neg = mg[g == 0], mg[g != 0]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def keep_separability(all_sets: dict) -> dict:
    """How well each model's Keep margin (Keep score minus the larger of the other two) ranks gold Keeps above the
    rest: ROC AUC (ties count half), per test and pooled, all rows and the eligibility-pass slice; pooled with 95%
    intervals and the paired h - base difference on the same draws. An offset only moves the threshold on this margin."""
    out = {}
    for name, (rows, S, draws) in all_sets.items():
        out[name] = {}
        for sl in SLICES:
            mask = np.ones(len(rows), bool) if sl == "all" else np.array([r["eligibility_pass"] for r in rows])
            rr, SS, dd = subset(rows, S, draws, mask)
            g = F.as_idx([r["gold"] for r in rr])
            blk = {"gold_keep": int((g == 0).sum())}
            if name != "pooled":
                blk |= {m: r3(auc_keep(SS[m], g)) for m in MODELS}
            else:
                vals = {m: [] for m in MODELS}
                for i in dd:
                    if (g[i] == 0).any() and (g[i] != 0).any():
                        for m in MODELS:
                            vals[m].append(auc_keep(SS[m][i], g[i]))
                pt = {m: auc_keep(SS[m], g) for m in MODELS}
                blk |= {m: [r3(pt[m]), r3(np.percentile(vals[m], 2.5)), r3(np.percentile(vals[m], 97.5))] for m in MODELS}
                d = np.array(vals["h"]) - np.array(vals["base"])
                blk["h_minus_base"] = [r3(pt["h"] - pt["base"]), r3(np.percentile(d, 2.5)), r3(np.percentile(d, 97.5))]
            out[name][sl] = blk
    return out


def tables(res: dict) -> str:
    PH, BPH, APH = "plus_H'", "before_plus_H'", "after_plus_H'"
    L = []
    cr = res["1_copy_rate"]
    L.append("TABLE 1  copy rate: raw label == rules_full (pooled: 95% interval)")
    L.append(f"{'set':8} {'slice':16} {'n':>4}  {'gold==rules':>24}  " + "  ".join(f"{m:>24}" for m in MODELS))
    for name in ("t1", "t2", "t3", "pooled"):
        for sl in SLICES:
            b = cr[name][sl]
            L.append(f"{name:8} {sl:16} {b['n']:>4}  {ci(b['gold_eq_rules_full']):>24}  "
                     + "  ".join(f"{ci(b[m]['copy_rate']):>24}" for m in MODELS))
    L.append("")
    for sl in SLICES:
        b = cr["pooled"][sl]
        for m in MODELS:
            c = b[m]["confusion_model_rows_x_rules_cols"]
            L.append(f"pooled {sl}, {m} rows x rules_full cols (K/E/S): "
                     + " | ".join(f"{k}: " + "/".join(str(c[k][j]) for j in LABELS) for k in LABELS)
                     + f"   kappa {b[m]['kappa_vs_rules_full']:.3f}; disagree {b[m]['disagreements']}: model right "
                       f"{b[m]['on_disagreements_model_right']}, rules right {b[m]['on_disagreements_rules_right']}")
    L.append("")
    ab = res["2_ablation"]
    L.append("TABLE 2  eligibility-layer ablation (all rows; pooled with 95% intervals)")
    for m in MODELS:
        for name in ("t1", "t2", "t3", "pooled"):
            for k in VARIANTS:
                b = ab[name][m][k]
                L.append(f"{m:5} {name:7} {k:17} F1 {ci(b['macro_f1']):>24}  acc {ci(b['accuracy']):>24}  "
                         f"KeepR {ci(b['keep_recall']):>24}  mix K/E/S {b['pred_mix']['Keep']:.3f}/"
                         f"{b['pred_mix']['Explore']:.3f}/{b['pred_mix']['Skip']:.3f}")
        for k, d in ab["pooled"][m]["paired_vs_raw"].items():
            L.append(f"{m:5} pooled  {k} - raw: F1 {ci(d['macro_f1_diff'])}  KeepR {ci(d['keep_recall_diff'])}  "
                     f"fixed {d['rows_fixed']} broken {d['rows_broken']}")
        L.append("")
    pc = res["3_prior_correction"]
    L.append("TABLE 3  prior correction (grid offsets fit on one set, applied unchanged)")
    for plan in ("fit_t1", "fit_t1+t2"):
        for m in MODELS:
            b = pc[plan][m]
            gf = b["grid_fit"]
            L.append(f"{plan} {m}: offsets Keep {gf['offset_keep']:+.1f} Explore {gf['offset_explore']:+.1f} "
                     f"(fit F1 {gf['fit_macro_f1_at_zero']:.3f} -> {gf['fit_macro_f1']:.3f}, plateau "
                     f"{gf['plateau_grid_points']} pts); analytic Keep {b['analytic_offsets']['keep']:+.3f} "
                     f"Explore {b['analytic_offsets']['explore']:+.3f}")
            for lab in ("grid", "analytic"):
                for t, v in b[lab].items():
                    if t == "pooled_apply":
                        d = v["after_minus_before"]
                        L.append(f"   {lab:8} pooled{'+'.join(pc[plan]['apply_to'])}: F1 {ci(v['before']['macro_f1'])} -> "
                                 f"{ci(v['after']['macro_f1'])}  dF1 {ci(d['macro_f1_diff'])}  KeepR "
                                 f"{ci(v['before']['keep_recall'])} -> {ci(v['after']['keep_recall'])}  mix K/E/S "
                                 f"{fmtmix(v['before']['pred_mix'])} -> {fmtmix(v['after']['pred_mix'])}")
                        dh = v["plus_H'"]["after_minus_before"]
                        L.append(f"   {lab:8}   with H': F1 {ci(v[PH]['before']['macro_f1'])} -> "
                                 f"{ci(v[PH]['after']['macro_f1'])}  dF1 {ci(dh['macro_f1_diff'])}")
                        continue
                    L.append(f"   {lab:8} {t}: F1 {v['before']['macro_f1']:.3f} -> {v['after']['macro_f1']:.3f}  KeepR "
                             f"{v['before']['keep_recall']:.3f} -> {v['after']['keep_recall']:.3f}  mix K/E/S "
                             f"{fmtmix(v['before']['pred_mix'])} -> {fmtmix(v['after']['pred_mix'])}  (+H' F1 "
                             f"{v[BPH]['macro_f1']:.3f} -> {v[APH]['macro_f1']:.3f})")
                    if "with_intervals" in v:
                        w = v["with_intervals"]
                        L.append(f"      intervals: F1 {ci(w['before']['macro_f1'])} -> {ci(w['after']['macro_f1'])}  dF1 "
                                 f"{ci(w['after_minus_before']['macro_f1_diff'])}  dKeepR "
                                 f"{ci(w['after_minus_before']['keep_recall_diff'])}; +H' dF1 "
                                 f"{ci(w[PH]['after_minus_before']['macro_f1_diff'])}")
        L.append("")
    L.append("Keep-margin ROC AUC (gold Keep vs rest)")
    for n, d in res["3b_keep_margin_auc"].items():
        for sl, b in d.items():
            L.append(f"  {n:7} {sl:16} Keeps {b['gold_keep']:>3}  " + "  ".join(f"{m} {ci(b[m])}" for m in MODELS)
                     + (f"  h-base {ci(b['h_minus_base'])}" if "h_minus_base" in b else ""))
    return "\n".join(L)


def fmtmix(mx: dict) -> str:
    return f"{mx['Keep']:.3f}/{mx['Explore']:.3f}/{mx['Skip']:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--summary", type=Path, default=SUMMARY)
    a = ap.parse_args()
    for p in (a.out, a.summary):
        if p.resolve().parent == (ROOT / "eval").resolve() and p.name.startswith("fresh"):
            raise SystemExit(f"{p}: write-once fresh result path")
    tests = load()
    all_sets = sets(tests)
    res = {"what": "cycle-3 CPU diagnostics on fresh tests 1-3 (practice sets, eval/protocol_amendment_4.json): copy "
                   "rate vs full rules, eligibility-layer ablation, prior correction; from stored first-token scores",
           "script": "scripts/diag_cycle3_cpu.py", "draws": DRAWS, "seed": SEED,
           "pooled_bootstrap": "each test's postings then rows drawn on their own (seed SEED+k for test k), index "
                               "arrays concatenated",
           "rules": "load_fresh with the rules in effect now (t1, t2 strict design check; t3 not strict)",
           "variants": {"raw": "model label", "H'": "HP_CATS " + ",".join(sorted(HP_CATS)) + " + salary cap",
                        "auth+enroll": "work_auth, enrollment; no salary cap",
                        "auth+enroll+mode": "work_auth, enrollment, work_mode; no salary cap"},
           "sets": {t: {"n": len(x["rows"]), "postings": len({r["posting_id"] for r in x["rows"]}),
                        "eligibility_pass": sum(r["eligibility_pass"] for r in x["rows"]),
                        "gold": dict(Counter(r["gold"] for r in x["rows"])),
                        "score_files": {m: {"path": str(TESTS[t]["scores"][m].relative_to(ROOT)), "sha256": x["sha256"][m]}
                                        for m in MODELS}}
                    for t, x in tests.items()},
           "reproduction": reproduction(tests),
           "1_copy_rate": copy_rate(all_sets),
           "2_ablation": ablation(all_sets),
           "3_prior_correction": prior_correction(tests),
           "3b_keep_margin_auc": keep_separability(all_sets)}
    res["reproduction"]["note"] = ("t1: raw labels on all rows reproduce the recorded values; H', rules_full and the "
                                   "eligibility-pass slice (344 recorded, 271 now) differ because fix cycle 2 changed the "
                                   "rules after test 1 was scored (load_fresh with the rules in effect now, as asked). t2 "
                                   "and t3 reproduce exactly. t2's result file records no score-file hashes.")
    text = tables(res)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "diag_cycle3_cpu.json").write_text(json.dumps(res, indent=1) + "\n")
    (a.out / "tables.txt").write_text(text + "\n")
    per_row = []
    for t, x in tests.items():
        for i, r in enumerate(x["rows"]):
            per_row.append({"test": t, "row_id": r["row_id"], "posting_id": r["posting_id"], "gold": r["gold"],
                            "rules_full": r["rules_full"], "eligibility_pass": r["eligibility_pass"],
                            "hard_cats": r["verdict"]["hard_cats"], "salary_soft": r["verdict"]["salary_soft"],
                            **{m: x["labels"][m][r["row_id"]] for m in MODELS},
                            **{f"{m}_scores": [round(float(v), 4) for v in x["S"][m][i]] for m in MODELS}})
    with (a.out / "rows.jsonl").open("w") as f:
        for r in per_row:
            f.write(json.dumps(r) + "\n")
    a.summary.write_text(json.dumps(res, indent=1) + "\n")
    print(text)


if __name__ == "__main__":
    main()
