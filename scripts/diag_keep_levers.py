"""Keep levers for the cycle-3 pick (run B + H'' without work_mode, decider), from stored files only.

  diag_keep_levers.py [--summary eval/diag_keep_levers.json] [--rows SCRATCH/numbers_rows.jsonl] [--power-sims 300] [--power-inner 1000]

Question: how can the pick find more of the real Keeps (true Keeps: pick Keep, gold Keep) while keeping the wrong
Keeps (pick Keep, gold Explore or Skip; the result files' unsafe Keeps) down? Reads the per-row score files
data/models/c3_eval/selection_rows.jsonl (fresh tests 1-3) and data/models/c3_eval/test4/test4_rows.jsonl (fresh test
4, used up), gold.jsonl (both labelers, adjudicator side) and rows.jsonl (design fields) under data/built/fresh_test*.
Loads no model and calls nothing. The summary holds aggregates only; the row table goes to --rows (outside the repo).

Sections: 1 where the missed Keeps go; 2 what the wrong Keeps share against the true Keeps; 3 the falling Keep rate
(build mix or labels); 4 levers, each as (true Keeps, wrong Keeps) with a paired change against the pick; 5 power for
a fresh test 5.

Fitting: nothing is fitted on test 4. A fitted lever is fitted on tests 1-3 pooled and applied once to test 4; its
tests 1-3 numbers are leave-one-test-out (fit on two, apply to the third, pool the three held-out label sets). Two fit
rules only: count match (a threshold that trades Keeps is set so the number of final Keep calls on the fit rows equals
the pick's there; the Skip cut-off likewise matches the pick's Skip count where one is refit), and net (a lever that
can only remove or only add Keeps maximises true Keeps minus wrong Keeps on the fit rows; ties go to the smaller
change). The pick's own cut-offs were fitted on tests 1-3 (in-sample there).

Bootstrap: fit_eval_score.two_way_draws (postings, then rows within postings, 2,000 draws, seed 20261021) per slice;
the tests 1-3 pooled slice draws each test on its own with seed SEED + k (k = 1, 2, 3) and concatenates, as
diag_cycle3_cpu does. Paired changes use the same draws for the lever and the pick; intervals are 95% percentile.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fit_eval_score import DRAWS, SEED, kappa, two_way_draws  # noqa: E402
from hybrid_rules import hybrid  # noqa: E402
from matcher.evalstats import LABELS, softmax  # noqa: E402

SEL_ROWS = ROOT / "data/models/c3_eval/selection_rows.jsonl"
T4_ROWS = ROOT / "data/models/c3_eval/test4/test4_rows.jsonl"
BUILT = ROOT / "data/built"
TESTS = ("fresh_test", "fresh_test2", "fresh_test3", "fresh_test4")
T123 = TESTS[:3]
PICK = "B+H''_without_work_mode|decider"
T_HI = 0.5890379172624224
T_LO = 0.2731045004103801
HPP = frozenset({"degree_field", "degree_level", "enrollment", "location", "work_auth"})
SOFT = ("years", "license", "skills", "employment_type", "work_mode")
K, E, S = 0, 1, 2
SCRATCH = Path(os.environ.get("KEEP_LEVERS_SCRATCH", ROOT / "data/scratch/keep_levers"))


def jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def r4(x):
    return None if x is None else round(float(x), 4)


# ---------------------------------------------------------------- the row table


def load() -> dict:
    raw = [dict(r, test=r["test"]) for r in jsonl(SEL_ROWS)] + [dict(r, test="fresh_test4") for r in jsonl(T4_ROWS)]
    gold, design, field = {}, {}, {}
    for t in TESTS:
        for g in jsonl(BUILT / t / "gold.jsonl"):
            gold[g["row_id"]] = g
        for x in jsonl(BUILT / t / "rows.jsonl"):
            design[x["row_id"]], field[x["row_id"]] = x["design"], x.get("field")
    for x in jsonl(BUILT / "fresh_test4b" / "rows.jsonl"):
        design.setdefault(x["row_id"], x["design"])
        field.setdefault(x["row_id"], x.get("field"))
    miss_g = [r["row_id"] for r in raw if r["row_id"] not in gold]
    miss_d = [r["row_id"] for r in raw if r["row_id"] not in design]
    assert not miss_g and not miss_d, (len(miss_g), len(miss_d))
    n = len(raw)
    T = {}
    T["test"] = np.array([r["test"] for r in raw])
    T["posting"] = np.array([r["posting_id"] for r in raw])
    T["gold"] = np.array([LABELS.index(r["gold"]) for r in raw])
    for r in raw:
        assert gold[r["row_id"]]["gold"] == r["gold"]
    T["A"] = np.array([LABELS.index(gold[r["row_id"]]["label_A"]) for r in raw])
    T["Bl"] = np.array([LABELS.index(gold[r["row_id"]]["label_B"]) for r in raw])
    T["adj"] = np.array([str(gold[r["row_id"]].get("adjudicator_side")) for r in raw])
    T["agreed"] = T["A"] == T["Bl"]
    T["pass"] = np.array([bool(r["eligibility_pass"]) for r in raw])
    T["rule"] = np.array([str(r["verdict"]["rule"]) for r in raw])
    T["cats"] = [frozenset(r["verdict"]["hard_cats"]) for r in raw]
    T["sal"] = np.array([bool(r["verdict"]["salary_soft"]) for r in raw])
    T["hskip"] = np.array([r["verdict"]["rule"] == "Skip" and bool(set(r["verdict"]["hard_cats"]) & HPP) for r in raw])
    for m in ("B", "h", "base", "A"):
        z = np.array([r["scores"].get(m, [np.nan] * 3) for r in raw], dtype=np.float64)
        T[f"z_{m}"] = z
        P = np.full((n, 3), np.nan)
        ok = ~np.isnan(z).any(1)
        P[ok] = softmax(z[ok])
        T[f"P_{m}"] = P
        T[f"s_{m}"] = P[:, K] + 0.5 * P[:, E]
        T[f"pk_{m}"] = P[:, K]
    zb = T["z_B"]
    T["margin_B"] = zb[:, K] - np.maximum(zb[:, E], zb[:, S])
    T["argmax_B"] = np.argmax(T["P_B"], 1)
    T["pre"] = decide(T["s_B"], T_HI, T["s_B"], T_LO)
    T["pick"] = layer(T, T["pre"])
    stored = np.array([LABELS.index(r["systems"][PICK]) for r in raw])
    assert (stored == T["pick"]).all(), "readout does not reproduce the stored pick"
    for r, p, f in zip(raw, T["pre"], T["pick"]):  # the repo's own H'' agrees with the vectorised one
        assert hybrid(LABELS[p], r["verdict"], HPP, True) == LABELS[f]
    d = [design[r["row_id"]] for r in raw]
    T["archetype"] = np.array([str(x.get("archetype")) for x in d])
    T["target_pass"] = np.array([str(x.get("target_pass")) for x in d])
    T["posting_length"] = np.array([str(x.get("posting_length")) for x in d])
    T["teaser"] = np.array([str(x.get("teaser")) for x in d])
    T["source"] = np.array([str(x.get("source")) for x in d])
    T["field"] = np.array([str(field[r["row_id"]]) for r in raw])
    T["kind"] = np.array([str(r.get("kind")) for r in raw])
    T["pool_field"] = np.array([str(r.get("pool_field")) for r in raw])
    T["n"] = n
    T["row_id"] = [r["row_id"] for r in raw]
    return T


def decide(xkeep: np.ndarray, thi: float, sskip: np.ndarray, tlo: float) -> np.ndarray:
    """Keep when xkeep >= thi, else Skip when sskip < tlo, else Explore (before the eligibility layer)."""
    return np.where(xkeep >= thi, K, np.where(sskip < tlo, S, E))


def layer(T: dict, pre: np.ndarray) -> np.ndarray:
    """H'' without work_mode: Skip on a rules Skip in HPP; a Keep with salary_soft becomes Explore."""
    return np.where(T["hskip"], S, np.where(T["sal"] & (pre == K), E, pre))


def tkwk(lab: np.ndarray, ref: np.ndarray, m: np.ndarray | None = None) -> tuple[int, int]:
    m = np.ones(len(lab), bool) if m is None else m
    k = (lab == K) & m
    return int((k & (ref == K)).sum()), int((k & (ref != K)).sum())


# ---------------------------------------------------------------- slices and draws


class Slices:
    """Named row masks with their bootstrap count matrices (draws x slice rows)."""

    def __init__(self, T: dict):
        t = T["test"]
        is4 = t == "fresh_test4"
        is123 = ~is4
        self.T = T
        self.masks = {"t4_pass": is4 & T["pass"], "t4_all": is4, "t123_pass": is123 & T["pass"], "t123_all": is123}
        self._mat = {}

    def mat(self, name: str) -> np.ndarray:
        if name not in self._mat:
            T, m = self.T, self.masks[name]
            idx = np.flatnonzero(m)
            if name.startswith("t4"):
                dr = two_way_draws(list(T["posting"][idx]), DRAWS, SEED)
            else:
                parts = []
                for k, t in enumerate(T123, 1):
                    sub = np.flatnonzero(T["test"][idx] == t)
                    parts.append((sub, two_way_draws(list(T["posting"][idx][sub]), DRAWS, SEED + k)))
                dr = [np.concatenate([sub[d[i]] for sub, d in parts]) for i in range(DRAWS)]
            M = np.zeros((DRAWS, len(idx)), dtype=np.int32)
            for i, d in enumerate(dr):
                M[i] = np.bincount(d, minlength=len(idx))
            self._mat[name] = M
        return self._mat[name]


def ci(vals: np.ndarray, point: float) -> list:
    return [r4(point), r4(np.percentile(vals, 2.5)), r4(np.percentile(vals, 97.5))]


def paired(sl: Slices, lever: np.ndarray, base: np.ndarray, ref: np.ndarray | None = None, names=None) -> dict:
    """Per slice: the lever's (true, wrong) Keeps, the pick's, and the paired change with its 95% interval."""
    T = sl.T
    ref = T["gold"] if ref is None else ref
    out = {}
    for name in names or ("t4_pass", "t4_all", "t123_all"):
        m = sl.masks[name]
        M = sl.mat(name)
        lv, bs, rf = lever[m], base[m], ref[m]
        dtk = ((lv == K) & (rf == K)).astype(int) - ((bs == K) & (rf == K)).astype(int)
        dwk = ((lv == K) & (rf != K)).astype(int) - ((bs == K) & (rf != K)).astype(int)
        a, b = M @ dtk, M @ dwk
        tk, wk = tkwk(lv, rf)
        tk0, wk0 = tkwk(bs, rf)
        out[name] = {"true_keep": tk, "wrong_keep": wk, "keep_calls": int((lv == K).sum()),
                     "pick_true_keep": tk0, "pick_wrong_keep": wk0,
                     "d_true_keep": ci(a, tk - tk0), "d_wrong_keep": ci(b, wk - wk0),
                     "d_true_crosses_zero": bool(np.percentile(a, 2.5) <= 0 <= np.percentile(a, 97.5)),
                     "d_wrong_crosses_zero": bool(np.percentile(b, 2.5) <= 0 <= np.percentile(b, 97.5)),
                     "rows_changed": int((lv != bs).sum())}
    return out


# ---------------------------------------------------------------- fitting helpers


def folds(T: dict):
    """(fit mask, apply mask, fold name): all of 1-3 -> test 4, then leave-one-test-out within 1-3."""
    t = T["test"]
    yield np.isin(t, T123), t == "fresh_test4", "fit_t123_apply_t4"
    for h in T123:
        yield np.isin(t, T123) & (t != h), t == h, f"loto_{h}"


def fitted(T: dict, fit_fn, apply_fn) -> tuple[np.ndarray, dict]:
    """Full label vector: each test's rows labelled with params fitted on its fold's fit rows."""
    lab = np.full(T["n"], -1)
    params = {}
    for fm, am, name in folds(T):
        p = fit_fn(fm)
        params[name] = p
        lab[am] = apply_fn(p)[am]
    assert (lab >= 0).all()
    return lab, params


def match_hi(x: np.ndarray, elig: np.ndarray, target: int) -> float:
    """Threshold giving exactly `target` rows of x >= thr among elig (midpoint between neighbours)."""
    v = np.sort(x[elig])[::-1]
    if target <= 0:
        return float(v[0] + 1e-9)
    if target >= len(v):
        return float(v[-1] - 1e-9)
    return float((v[target - 1] + v[target]) / 2)


def match_lo(x: np.ndarray, elig: np.ndarray, target: int) -> float:
    v = np.sort(x[elig])
    if target <= 0:
        return float(v[0] - 1e-9)
    return float((v[target - 1] + v[target]) / 2)


def count_match_keep(T: dict, x: np.ndarray, fm: np.ndarray) -> float:
    elig = fm & ~T["hskip"] & ~T["sal"]
    return match_hi(x, elig, int(((T["pick"] == K) & fm).sum()))


def count_match_skip(T: dict, s: np.ndarray, fm: np.ndarray) -> float:
    elig = fm & ~T["hskip"]
    return match_lo(s, elig, int(((T["pick"] == S) & elig).sum()))


def net(lab: np.ndarray, ref: np.ndarray, m: np.ndarray) -> int:
    tk, wk = tkwk(lab, ref, m)
    return tk - wk


# ---------------------------------------------------------------- section 1


def vote(T, i):
    return f"{LABELS[T['A'][i]]}/{LABELS[T['Bl'][i]]}"


S_BINS = [T_LO, 0.35, 0.40, 0.45, 0.50, 0.55, T_HI]
PK_BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0001]
SK_BINS = [T_HI, 0.65, 0.70, 0.80, 0.90, 1.0001]


def binname(x, edges):
    for a, b in zip(edges[:-1], edges[1:]):
        if a <= x < b:
            return f"[{a:.3f},{b:.3f})"
    return "out"


def scope_masks(T):
    t = T["test"]
    out = {x: t == x for x in TESTS}
    out["t123_all"] = np.isin(t, T123)
    out["t123_pass"] = out["t123_all"] & T["pass"]
    out["t4_all"] = t == "fresh_test4"
    out["t4_pass"] = out["t4_all"] & T["pass"]
    return out


def miss_bucket(T: dict) -> np.ndarray:
    b = np.full(T["n"], "", dtype=object)
    miss = (T["gold"] == K) & (T["pick"] != K)
    b[miss & T["hskip"]] = "H''_skip"
    b[miss & ~T["hskip"] & T["sal"] & (T["pre"] == K)] = "salary_cap"
    b[miss & ~T["hskip"] & (T["pre"] == E)] = "explore_band"
    b[miss & ~T["hskip"] & (T["pre"] == S)] = "low_score_skip"
    assert (b[miss] != "").all()
    return b


def section1(T: dict, bucket: np.ndarray) -> dict:
    out = {}
    for sc, m in scope_masks(T).items():
        gk = m & (T["gold"] == K)
        miss = gk & (T["pick"] != K)
        idx = np.flatnonzero(miss)
        d = {"gold_keep": int(gk.sum()), "found": int((gk & (T["pick"] == K)).sum()), "missed": int(miss.sum()),
             "buckets": dict(Counter(bucket[idx]))}
        cross_v, cross_a = defaultdict(Counter), defaultdict(Counter)
        for i in idx:
            cross_v[bucket[i]][vote(T, i)] += 1
            cross_a[bucket[i]][T["adj"][i]] += 1
        d["bucket_x_votes_A/B"] = {k: dict(v) for k, v in cross_v.items()}
        d["bucket_x_adjudicator_side"] = {k: dict(v) for k, v in cross_a.items()}
        hs = [i for i in idx if bucket[i] == "H''_skip"]
        d["H''_skip_categories"] = dict(Counter(c for i in hs for c in sorted(T["cats"][i] & HPP)))
        d["H''_skip_pre_label"] = dict(Counter(LABELS[T["pre"][i]] for i in hs))
        eb = [i for i in idx if bucket[i] == "explore_band"]
        d["explore_band_s_bins"] = dict(Counter(binname(T["s_B"][i], S_BINS) for i in eb))
        d["explore_band_pkeep_bins"] = dict(Counter(binname(T["pk_B"][i], PK_BINS) for i in eb))
        d["explore_band_argmax"] = dict(Counter(LABELS[T["argmax_B"][i]] for i in eb))
        d["explore_band_s_quartiles"] = [r4(q) for q in np.percentile(T["s_B"][eb], [0, 25, 50, 75, 100])] if eb else None
        ls = [i for i in idx if bucket[i] == "low_score_skip"]
        d["low_score_skip_s"] = sorted(r4(T["s_B"][i]) for i in ls)
        out[sc] = d
    return out


# ---------------------------------------------------------------- section 2


def section2(T: dict) -> dict:
    out = {}
    feats = {
        "gold": lambda i: LABELS[T["gold"][i]],
        "votes_A/B": lambda i: vote(T, i),
        "adjudicator_side": lambda i: T["adj"][i],
        "s_bin": lambda i: binname(T["s_B"][i], SK_BINS),
        "pkeep_bin": lambda i: binname(T["pk_B"][i], PK_BINS),
        "rule": lambda i: T["rule"][i],
        "salary_soft": lambda i: str(T["sal"][i]),
        "archetype": lambda i: T["archetype"][i],
        "target_pass": lambda i: T["target_pass"][i],
        "posting_length": lambda i: T["posting_length"][i],
        "teaser": lambda i: T["teaser"][i],
        "kind (test 4 only)": lambda i: T["kind"][i],
        "pool_field (test 4 only)": lambda i: T["pool_field"][i],
    }
    for c in SOFT:
        feats[f"hard_cats has {c}"] = (lambda c: lambda i: str(c in T["cats"][i]))(c)
    sm = scope_masks(T)
    for sc in ("t4_pass", "t4_all", "t123_all", "t123_pass"):
        m = sm[sc]
        tk = np.flatnonzero(m & (T["pick"] == K) & (T["gold"] == K))
        wk = np.flatnonzero(m & (T["pick"] == K) & (T["gold"] != K))
        d = {"true_keep": len(tk), "wrong_keep": len(wk)}
        for f, fn in feats.items():
            if "test 4 only" in f and not sc.startswith("t4"):
                continue
            a, b = Counter(fn(i) for i in tk), Counter(fn(i) for i in wk)
            d[f] = {k: {"true": a.get(k, 0), "wrong": b.get(k, 0)} for k in sorted(set(a) | set(b))}
        d["s_median"] = {"true": r4(np.median(T["s_B"][tk])), "wrong": r4(np.median(T["s_B"][wk]))}
        d["pkeep_median"] = {"true": r4(np.median(T["pk_B"][tk])), "wrong": r4(np.median(T["pk_B"][wk]))}
        out[sc] = d
    return out


# ---------------------------------------------------------------- section 3


def section3(T: dict) -> dict:
    out = {"note": "Keep rate = gold Keep share. Shift-share against a reference test, per cell definition: mix effect "
                   "= sum over cells (w_t - w_ref) * r_ref, label effect = sum w_t * (r_t - r_ref); they add up to the "
                   "change. A cell absent from the reference uses r_ref = 0. 'Label effect' means the same design cell "
                   "has a lower Keep rate; it cannot tell stricter labels from harder postings inside a cell. field is "
                   "rows.jsonl's posting pool field; source is the posting's ATS."}
    sm = scope_masks(T)
    celldefs = {"archetype x target_pass": np.array([f"{a}|{b}" for a, b in zip(T["archetype"], T["target_pass"])]),
                "field": T["field"], "archetype x field": np.array([f"{a}|{b}" for a, b in zip(T["archetype"], T["field"])]),
                "source": T["source"]}
    for sl in ("all", "pass"):
        d = {}
        for t in TESTS + ("t123_all",):
            m = sm[t] & (T["pass"] if sl == "pass" else True)
            row = {"rows": int(m.sum()), "keep_rate": r4((T["gold"][m] == K).mean()),
                   "label_A_keep_rate": r4((T["A"][m] == K).mean()), "label_B_keep_rate": r4((T["Bl"][m] == K).mean())}
            for fname, arr in (("archetype", T["archetype"]), ("target_pass", T["target_pass"]), ("field", T["field"]),
                               ("archetype x target_pass", celldefs["archetype x target_pass"])):
                row[fname] = {k: {"rows": int((m & (arr == k)).sum()),
                                  "share": r4((m & (arr == k)).sum() / m.sum()),
                                  "keep_rate": r4((T["gold"][m & (arr == k)] == K).mean()) if (m & (arr == k)).any() else None}
                              for k in sorted(set(arr[m]))}
            d[t] = row
        shifts = {}
        for cname, cell in celldefs.items():
            shifts[cname] = {}
            for ref in ("fresh_test", "t123_all"):
                for t in TESTS[1:] if ref == "fresh_test" else ("fresh_test4",):
                    mr = sm[ref] & (T["pass"] if sl == "pass" else True)
                    mt = sm[t] & (T["pass"] if sl == "pass" else True)
                    mix = lab = 0.0
                    for c in sorted(set(cell[mr]) | set(cell[mt])):
                        wr = (mr & (cell == c)).sum() / mr.sum()
                        wt = (mt & (cell == c)).sum() / mt.sum()
                        rr = (T["gold"][mr & (cell == c)] == K).mean() if (mr & (cell == c)).any() else 0.0
                        rt = (T["gold"][mt & (cell == c)] == K).mean() if (mt & (cell == c)).any() else 0.0
                        mix += (wt - wr) * rr
                        lab += wt * (rt - rr)
                    tot = (T["gold"][mt] == K).mean() - (T["gold"][mr] == K).mean()
                    shifts[cname][f"{t}_vs_{ref}"] = {"change": r4(tot), "mix_effect": r4(mix), "label_effect": r4(lab)}
        d["shift_share"] = shifts
        out[sl] = d
    return out


# ---------------------------------------------------------------- section 4 levers


def sweep(T: dict, x: np.ndarray, m: np.ndarray, elig: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep = elig & x >= thr, swept over every threshold: (thr, true Keeps, wrong Keeps) per cut."""
    elig = (~T["hskip"] & ~T["sal"]) if elig is None else elig
    idx = np.flatnonzero(m & elig)
    o = idx[np.argsort(-x[idx], kind="stable")]
    g = T["gold"][o] == K
    return x[o], np.cumsum(g), np.cumsum(~g)


def matched(tk: np.ndarray, wk: np.ndarray, tk0: int, wk0: int) -> dict:
    ok = wk <= wk0
    a = int(tk[ok].max()) if ok.any() else 0
    ok2 = tk >= tk0
    b = int(wk[ok2].min()) if ok2.any() else None
    return {"true_keep_at_pick_wrong": a, "wrong_keep_at_pick_true": b}


def curve(T, x, m, grid):
    out = []
    elig = ~T["hskip"] & ~T["sal"]
    for g in grid:
        k = m & elig & (x >= g)
        out.append([r4(g), int((k & (T["gold"] == K)).sum()), int((k & (T["gold"] != K)).sum())])
    return out


def section4(T: dict, sl: Slices) -> tuple[dict, dict]:
    res, labs = {}, {}
    pick, gold = T["pick"], T["gold"]
    sm = scope_masks(T)
    elig = ~T["hskip"] & ~T["sal"]
    is4 = T["test"] == "fresh_test4"

    # (a) t_hi curve per test, and the matched points on test 4 (curve readouts, no intervals)
    grid = np.round(np.arange(0.30, 0.901, 0.01), 2)
    a = {"grid_t_hi": "[t_hi, true Keeps, wrong Keeps]; t_lo fixed; Keep needs s >= t_hi and passes H''"}
    for sc in TESTS + ("t123_all", "t4_pass", "t123_pass"):
        a[sc] = curve(T, T["s_B"], sm[sc], grid)
    a["pick_at_t_hi"] = {sc: list(tkwk(pick, gold, sm[sc])) for sc in sm}
    for thi in (0.50, 0.55, 0.65, 0.70):
        lv = layer(T, decide(T["s_B"], thi, T["s_B"], T_LO))
        a[f"t_hi_{thi:.2f}_paired"] = paired(sl, lv, pick)
    res["4a_t_hi_curve"] = a

    # base-rate pieces
    pi_old = float((gold[sm["t123_pass"]] == K).mean())
    pi_t3 = float((gold[sm["fresh_test3"] & T["pass"]] == K).mean())

    def prior_shift(P, pi_new):
        w = np.array([pi_new / pi_old, (1 - pi_new) / (1 - pi_old), (1 - pi_new) / (1 - pi_old)])
        Q = P * w
        return Q / Q.sum(1, keepdims=True)

    scores = {"B_s": T["s_B"], "B_pkeep": T["pk_B"], "B_margin": T["margin_B"],
              "B+h_s": (T["s_B"] + T["s_h"]) / 2, "B+h_pkeep": (T["pk_B"] + T["pk_h"]) / 2,
              "B+base_s": (T["s_B"] + T["s_base"]) / 2, "B+base_pkeep": (T["pk_B"] + T["pk_base"]) / 2}

    # matched points on test 4 for each single-score lever
    mp = {"note": "Curve readouts on test 4: the best true Keeps with at most the pick's wrong Keeps, and the fewest "
                  "wrong Keeps with at least the pick's true Keeps. Chosen on the test-4 point estimate, so no "
                  "interval applies; nothing here is a fit."}
    for sc in ("t4_pass", "t4_all"):
        tk0, wk0 = tkwk(pick, gold, sm[sc])
        d = {"pick": [tk0, wk0]}
        for nm, x in scores.items():
            _, tk, wk = sweep(T, x, sm[sc])
            d[nm] = matched(tk, wk, tk0, wk0)
        for nm in ("h", "base"):  # veto: s_B >= t and s_x >= bar, best over a 2-D grid
            best_a, best_b = 0, None
            for t in np.arange(0.30, 0.90, 0.01):
                _, tk, wk = sweep(T, T[f"s_{nm}"], sm[sc], elig & (T["s_B"] >= t))
                if len(tk):
                    q = matched(tk, wk, tk0, wk0)
                    best_a = max(best_a, q["true_keep_at_pick_wrong"])
                    if q["wrong_keep_at_pick_true"] is not None:
                        best_b = q["wrong_keep_at_pick_true"] if best_b is None else min(best_b, q["wrong_keep_at_pick_true"])
            d[f"veto_{nm}_2d"] = {"true_keep_at_pick_wrong": best_a, "wrong_keep_at_pick_true": best_b}
        prs = []
        for pn in np.arange(0.02, 0.301, 0.005):
            P = prior_shift(T["P_B"], pn)
            lv = layer(T, decide(P[:, K] + 0.5 * P[:, E], T_HI, P[:, K] + 0.5 * P[:, E], T_LO))
            prs.append(tkwk(lv, gold, sm[sc]))
        prs = np.array(prs)
        d["base_rate_correction_over_pi"] = matched(prs[:, 0], prs[:, 1], tk0, wk0)
        mp[sc] = d
    res["4a_matched_points"] = mp

    # (b) P(Keep) >= tau and the Keep logit margin, count-matched on 1-3
    b = {"fit_rule": "count match: tau so final Keep calls on the fit tests equal the pick's; Skip as now (s < t_lo)",
         "margin_definition": "z_Keep - max(z_Explore, z_Skip) on the raw first-token scores (= log P(Keep) - log max P(other))"}
    for nm, x in (("pkeep", T["pk_B"]), ("margin", T["margin_B"])):
        lv, p = fitted(T, lambda fm, x=x: count_match_keep(T, x, fm),
                       lambda thr, x=x: layer(T, decide(x, thr, T["s_B"], T_LO)))
        labs[f"b_{nm}"] = lv
        b[nm] = {"thresholds": {k: r4(v) for k, v in p.items()}, "paired": paired(sl, lv, pick)}
    # reference: B's s itself count-matched leave-one-test-out (the handicap any refit lever carries on 1-3)
    lv, p = fitted(T, lambda fm: count_match_keep(T, T["s_B"], fm), lambda thr: layer(T, decide(T["s_B"], thr, T["s_B"], T_LO)))
    labs["b_s_countmatch"] = lv
    b["s_countmatch_reference"] = {"thresholds": {k: r4(v) for k, v in p.items()}, "paired": paired(sl, lv, pick)}
    br = {"pi_old_t123_pass": r4(pi_old), "pi_test3_pass": r4(pi_t3),
          "pi_test4_pass_oracle": r4((gold[sm["t4_pass"]] == K).mean()),
          "how": "P(Keep) times pi_new/pi_old, the other two times (1-pi_new)/(1-pi_old), renormalised; s recomputed; "
                 "committed cut-offs kept. pi_new = test 3's pass rate is the non-oracle choice; the 7% row uses "
                 "test 4's own gold and is a what-if.", "grid": {}}
    for pn in (0.05, 0.07, 0.09, 0.11, 0.13, round(pi_t3, 4)):
        P = prior_shift(T["P_B"], pn)
        s2 = P[:, K] + 0.5 * P[:, E]
        lv = layer(T, decide(s2, T_HI, s2, T_LO))
        br["grid"][f"{pn:.4f}"] = paired(sl, lv, pick, names=("t4_pass", "t4_all"))
        if pn == round(pi_t3, 4):
            labs["b_baserate_t3"] = lv
    b["base_rate_correction"] = br
    res["4b_readout"] = b

    # (c) eligibility layer: what each H'' category kills and stops; then soft demotions
    c = {"per_category": {}}
    for sc in TESTS + ("t123_all", "t4_pass"):
        m = sm[sc]
        d = {}
        for cat in sorted(HPP):
            hit = m & (T["rule"] == "Skip") & np.array([cat in x for x in T["cats"]])
            only = hit & np.array([len(x & HPP) == 1 for x in T["cats"]])
            d[cat] = {"rows": int(hit.sum()), "gold_keep": int((hit & (gold == K)).sum()),
                      "true_keep_killed": int((hit & (T["pre"] == K) & (gold == K)).sum()),
                      "wrong_keep_stopped": int((hit & (T["pre"] == K) & (gold != K)).sum()),
                      "sole_category_true_killed": int((only & (T["pre"] == K) & (gold == K)).sum()),
                      "sole_category_wrong_stopped": int((only & (T["pre"] == K) & (gold != K)).sum())}
        cap = m & ~T["hskip"] & T["sal"] & (T["pre"] == K)
        d["salary_cap"] = {"true_keep_killed": int((cap & (gold == K)).sum()), "wrong_keep_stopped": int((cap & (gold != K)).sum())}
        c["per_category"][sc] = d
    nol = T["pre"].copy()  # the readout with no eligibility layer, against the pick
    c["no_eligibility_layer_paired"] = paired(sl, nol, pick)

    def demote(cats):
        hit = (T["rule"] == "Skip") & np.array([bool(x & set(cats)) for x in T["cats"]])
        return np.where((pick == K) & hit, E, pick)

    c["soft_demotion_each"] = {}
    for cat in SOFT:
        c["soft_demotion_each"][cat] = paired(sl, demote([cat]), pick)
    subsets = [x for r in range(0, len(SOFT) + 1) for x in itertools.combinations(SOFT, r)]

    def fit_combo(fm):
        best = max(subsets, key=lambda s: (net(demote(s), gold, fm), -len(s)))
        return list(best)

    lv, p = fitted(T, fit_combo, demote)
    labs["c_demote_combo"] = lv
    c["soft_demotion_combo"] = {"fit_rule": "net (true minus wrong Keeps) over all 32 subsets on the fit tests; ties to fewer categories",
                                "chosen": p, "paired": paired(sl, lv, pick)}
    res["4c_eligibility"] = c

    # (d) averaging probabilities, cut-offs refit by count match
    d = {"fit_rule": "P averaged per class; s = P(Keep) + 0.5 P(Explore); t_hi count-matched to the pick's Keep calls "
                     "and t_lo to the pick's Skip calls (outside H'') on the fit tests"}
    auc = {}
    for nm, other in (("B+h", "h"), ("B+base", "base")):
        Pa = (T["P_B"] + T[f"P_{other}"]) / 2
        sa = Pa[:, K] + 0.5 * Pa[:, E]
        lv, p = fitted(T, lambda fm, sa=sa: (count_match_keep(T, sa, fm), count_match_skip(T, sa, fm)),
                       lambda q, sa=sa: layer(T, decide(sa, q[0], sa, q[1])))
        labs[f"d_{nm}"] = lv
        d[nm] = {"cut_offs": {k: [r4(v[0]), r4(v[1])] for k, v in p.items()}, "paired": paired(sl, lv, pick)}
        pa = Pa[:, K]  # Keep read from the averaged P(Keep), Skip from the averaged s
        lv, p = fitted(T, lambda fm, sa=sa, pa=pa: (count_match_keep(T, pa, fm), count_match_skip(T, sa, fm)),
                       lambda q, sa=sa, pa=pa: layer(T, decide(pa, q[0], sa, q[1])))
        labs[f"d_{nm}_pkeep"] = lv
        d[f"{nm}_pkeep"] = {"cut_offs": {k: [r4(v[0]), r4(v[1])] for k, v in p.items()}, "paired": paired(sl, lv, pick)}
    # A+B on 1-3 only, leave-one-test-out, against B refit the same way
    t123 = np.isin(T["test"], T123)
    Pab = (T["P_B"] + T["P_A"]) / 2
    sab = Pab[:, K] + 0.5 * Pab[:, E]
    out_ab, out_b = np.full(T["n"], -1), np.full(T["n"], -1)
    cuts = {}
    for h in T123:
        fm, am = t123 & (T["test"] != h), T["test"] == h
        q = (count_match_keep(T, sab, fm), count_match_skip(T, sab, fm))
        qb = (count_match_keep(T, T["s_B"], fm), count_match_skip(T, T["s_B"], fm))
        out_ab[am] = layer(T, decide(sab, q[0], sab, q[1]))[am]
        out_b[am] = layer(T, decide(T["s_B"], qb[0], T["s_B"], qb[1]))[am]
        cuts[f"loto_{h}"] = {"A+B": [r4(q[0]), r4(q[1])], "B": [r4(qb[0]), r4(qb[1])]}
    out_ab[~t123] = pick[~t123]
    out_b[~t123] = pick[~t123]
    d["A+B_t123_loto"] = {"cut_offs": cuts,
                          "vs_pick": paired(sl, out_ab, pick, names=("t123_all", "t123_pass")),
                          "vs_B_refit_same_way": paired(sl, out_ab, out_b, names=("t123_all", "t123_pass"))}
    # Keep-vs-rest AUC on rows H'' leaves alone
    ascores = dict(scores, h_s=T["s_h"], h_pkeep=T["pk_h"], base_s=T["s_base"], base_pkeep=T["pk_base"],
                   A_s=T["s_A"], A_pkeep=T["pk_A"], **{"A+B_s": sab, "A+B_pkeep": Pab[:, K]})
    for sc in TESTS + ("t123_all", "t4_pass"):
        m = sm[sc] & ~T["hskip"]
        y = gold[m] == K
        auc[sc] = {nm: (r4(roc_auc_score(y, x[m])) if not np.isnan(x[m]).any() else None) for nm, x in ascores.items()}
    d["keep_auc_outside_H''"] = auc
    res["4d_averaging"] = d

    # (e) veto: Keep only if the pick says Keep and h (or base) clears a bar; and t_hi refit jointly with the veto (net picks it; here it tightens)
    e = {"fit_rule": "net on the fit tests; bar grid = every 2nd percentile of the other model's s over all rows, "
                     "plus no veto; ties to the lower bar"}
    for nm in ("h", "base"):
        x = T[f"s_{nm}"]
        bars = np.concatenate([[-np.inf], np.percentile(x, np.arange(2, 100, 2))])

        def fit_bar(fm, x=x, bars=bars):
            return float(max(bars, key=lambda bb: (net(np.where((pick == K) & (x < bb), E, pick), gold, fm), -bb)))

        lv, p = fitted(T, fit_bar, lambda bb, x=x: np.where((pick == K) & (x < bb), E, pick))
        labs[f"e_veto_{nm}"] = lv
        e[f"veto_{nm}"] = {"bar": {k: r4(v) if np.isfinite(v) else "none" for k, v in p.items()}, "paired": paired(sl, lv, pick)}
        tgrid = np.round(np.arange(0.35, 0.801, 0.01), 2)

        def fit_two(fm, x=x, bars=bars, tgrid=tgrid):
            best, key = None, None
            for t in tgrid:
                pre = decide(T["s_B"], t, T["s_B"], T_LO)
                for bb in bars:
                    lab = layer(T, np.where((pre == K) & (x < bb), E, pre))
                    k = (net(lab, gold, fm), -abs(t - T_HI), -bb)
                    if key is None or k > key:
                        best, key = (float(t), float(bb)), k
            return best

        def ap_two(q, x=x):
            pre = decide(T["s_B"], q[0], T["s_B"], T_LO)
            return layer(T, np.where((pre == K) & (x < q[1]), E, pre))

        lv, p = fitted(T, fit_two, ap_two)
        labs[f"e_refit_t_hi_veto_{nm}"] = lv
        e[f"refit_t_hi_plus_veto_{nm}"] = {"t_hi_and_bar": {k: [r4(v[0]), r4(v[1]) if np.isfinite(v[1]) else "none"] for k, v in p.items()},
                                          "paired": paired(sl, lv, pick)}
    res["4e_veto"] = e

    # (f) second-pass Keep check by a rubric reader, simulated with the labelers
    f = {"circularity": "gold equals label_A and label_B wherever they agree, and the adjudicator picks one of them "
                        "elsewhere, so a reader that is label_A or label_B scored against gold is partly scored against "
                        "itself: those rows are an upper bound only. The honest estimate is cross-labeler: reader B "
                        "scored against label_A as the reference, and reader A against label_B.",
         "band_fit_rule": "net against the reference on the fit tests over lower edges t_lo..t_hi step 0.01; ties to the "
                          "higher edge (fewer reads)"}
    readers = {"upper_A_vs_gold": (T["A"], gold), "upper_B_vs_gold": (T["Bl"], gold),
               "cross_B_vs_refA": (T["Bl"], T["A"]), "cross_A_vs_refB": (T["A"], T["Bl"])}
    band_ok = (T["pre"] == E) & (pick == E) & ~T["sal"]
    edges = np.round(np.concatenate([np.arange(0.28, T_HI, 0.01), [T_HI]]), 4)
    for nm, (rd, rf) in readers.items():
        chk = np.where((pick == K) & (rd != K), E, pick)

        def band(edge, rd=rd, base=pick):
            return np.where(band_ok & (T["s_B"] >= edge) & (rd == K), K, base)

        lvb, p = fitted(T, lambda fm, rf=rf, band=band: float(max(edges, key=lambda ed: (net(band(ed), rf, fm), ed))), band)
        both = np.where((lvb == K) & (pick != K), K, chk)
        g = {"check_keeps": paired(sl, chk, pick, rf), "reread_band": {"edge": {k: r4(v) for k, v in p.items()},
                                                                       "paired": paired(sl, lvb, pick, rf)},
             "both": paired(sl, both, pick, rf)}
        # what the check reads, and A-vs-B agreement on exactly those rows
        reads = {}
        for sc in ("t4_pass", "t4_all", "t123_all"):
            m = sl.masks[sc]
            ek = np.full(T["n"], np.nan)
            for fm, am, fn in folds(T):
                ek[am] = p[fn]
            rk = m & (pick == K)
            rb = m & band_ok & (T["s_B"] >= ek)
            reads[sc] = {}
            for rn, rm in (("check_keeps", rk), ("reread_band", rb)):
                aa, bb = T["A"][rm], T["Bl"][rm]
                reads[sc][rn] = {"rows_read": int(rm.sum()), "per_1000_rows": r4(1000 * rm.sum() / m.sum()),
                                 "A_B_exact_agree": r4((aa == bb).mean()) if rm.any() else None,
                                 "A_B_keep_vs_not_agree": r4(((aa == K) == (bb == K)).mean()) if rm.any() else None,
                                 "kappa_3class": kappa([LABELS[x] for x in aa], [LABELS[x] for x in bb]) if rm.any() else None}
        g["reads"] = reads
        f[nm] = g
        if nm.startswith("cross"):
            labs[f"f_check_{nm}"] = chk
            labs[f"f_band_{nm}"] = lvb
    res["4f_second_pass"] = f

    # (g) agreement subset
    g = {}
    for sc in ("t4_pass", "t4_all", "t123_all", "t123_pass"):
        m = sm[sc]
        g[sc] = {}
        for an, am in (("agreed", T["agreed"]), ("split", ~T["agreed"])):
            mm = m & am
            gk = mm & (gold == K)
            g[sc][an] = {"rows": int(mm.sum()), "gold_keep": int(gk.sum()), "found": int((gk & (pick == K)).sum()),
                         "missed": int((gk & (pick != K)).sum()), "wrong_keep": int((mm & (pick == K) & (gold != K)).sum()),
                         "wrong_keep_gold_explore": int((mm & (pick == K) & (gold == E)).sum()),
                         "wrong_keep_gold_skip": int((mm & (pick == K) & (gold == S)).sum()),
                         "keep_recall": r4((gk & (pick == K)).sum() / gk.sum()) if gk.any() else None}
        k = m & (pick == K)
        g[sc]["pick_keeps_either_labeler_keep_but_gold_not"] = int((k & (gold != K) & ((T["A"] == K) | (T["Bl"] == K))).sum())
        g[sc]["pick_keeps_neither_labeler_keep"] = int((k & (T["A"] != K) & (T["Bl"] != K)).sum())
    res["4g_agreement_subset"] = g
    return res, labs


# ---------------------------------------------------------------- section 5 power


def section5(T: dict, sl: Slices, sims: int, inner: int) -> dict:
    """A synthetic lever on resampled postings: how many gold Keeps a test 5 needs so both paired intervals clear 0."""
    rng = np.random.default_rng(SEED + 5)
    gold, pick, pre = T["gold"], T["pick"], T["pre"]
    tk = (pick == K) & (gold == K)
    wk = (pick == K) & (gold != K)
    miss_band = (gold == K) & (pick != K) & ~T["hskip"] & (pre == E)
    band_nk = (gold != K) & (pick == E) & (pre == E) & ~T["hskip"]
    nk = int((gold == K).sum())
    rates = {"gold_keep": nk, "true_keep": int(tk.sum()), "wrong_keep": int(wk.sum()),
             "missed_in_explore_band": int(miss_band.sum()), "gold_not_keep_in_explore_band": int(band_nk.sum())}
    # per 43 gold Keeps: gross flips (gains, true-Keep losses, wrong-Keep drops, wrong-Keep adds)
    settings = {"pure": (4, 0, 8, 0), "churn_2x": (6, 2, 10, 2)}
    cells = np.array([f"{a}|{b}" for a, b in zip(T["archetype"], T["target_pass"])])
    t123 = np.isin(T["test"], T123)
    rate_cell = {c: (gold[t123 & (cells == c)] == K).mean() for c in set(cells[t123])}
    rich_cells = sorted(str(c) for c, r in rate_cell.items() if r >= 0.15)
    mixes = {"test4_mix": T["test"] == "fresh_test4", "pooled_t1_4": np.ones(T["n"], bool),
             "keep_enriched": np.isin(cells, rich_cells)}
    out = {"how": f"Pool = tests 1-4 rows (postings never cross tests). A synthetic lever flips rows at random: gains "
                  f"among gold Keeps the pick leaves in the Explore band, losses among the pick's true Keeps, drops "
                  f"among its wrong Keeps, adds among gold non-Keeps in the Explore band, at the per-gold-Keep rates "
                  f"below. A simulated test 5 draws postings with replacement from the mix until it holds N gold Keeps; "
                  f"its paired change gets a two-way bootstrap ({inner} draws; rows within a drawn posting resampled "
                  f"multinomially); it clears when the true-Keep interval is above 0 and the wrong-Keep interval below "
                  f"0. {sims} simulated tests per point. Keep-enriched mix = rows from design cells (archetype x "
                  f"target_pass) whose tests 1-3 gold Keep rate is at least 15%: {rich_cells}.",
           "pool_counts": rates, "settings_per_43_gold_keeps": {k: dict(zip(("gain", "true_loss", "wrong_drop", "wrong_add"), v)) for k, v in settings.items()}}
    post = T["posting"]
    types = np.stack([miss_band, tk, wk, band_nk], 1)  # gain, loss, drop, add candidates (disjoint)
    assert (types.sum(1) <= 1).all()
    for mix_name, mm in mixes.items():
        pids, inv = np.unique(post[mm], return_inverse=True)
        cnt = np.zeros((len(pids), 4), dtype=np.int64)
        np.add.at(cnt, inv, types[mm].astype(np.int64))
        nrow = np.bincount(inv, minlength=len(pids))
        gk_per = np.bincount(inv, weights=(gold[mm] == K), minlength=len(pids)).astype(int)
        out[f"{mix_name}_gold_keep_per_1000_rows"] = r4(1000 * gk_per.sum() / nrow.sum())
        mk = gk_per.sum()
        for sname, (ga, tl, wd, wa) in settings.items():
            probs = np.array([ga, tl, wd, wa]) / 43 * mk / np.maximum(cnt.sum(0), 1)
            assert (probs <= 1).all(), (mix_name, sname, probs)
            res = {"flip_probs": [r4(x) for x in probs], "by_n_gold_keep": {}}
            for N in (43, 86, 129, 172, 258, 344, 430):
                clear_t = clear_w = clear_b = 0
                rows_used = []
                for _ in range(sims):
                    j = rng.integers(0, len(pids), max(64, int(3 * N * len(pids) / max(mk, 1))))
                    cs = np.cumsum(gk_per[j])
                    while cs[-1] < N:
                        j = np.concatenate([j, rng.integers(0, len(pids), len(j))])
                        cs = np.cumsum(gk_per[j])
                    j = j[:int(np.searchsorted(cs, N)) + 1]
                    flips = rng.binomial(cnt[j], probs)  # postings x 4
                    comp = np.concatenate([flips, (nrow[j] - flips.sum(1))[:, None]], 1)
                    rows_used.append(int(nrow[j].sum()))
                    ntot = nrow[j]
                    pv = comp / ntot[:, None]
                    pk = rng.integers(0, len(j), (inner, len(j)))
                    tot = rng.multinomial(ntot[pk], pv[pk]).sum(1)  # inner x 5
                    dT = tot[:, 0] - tot[:, 1]
                    dW = tot[:, 3] - tot[:, 2]
                    ct = np.percentile(dT, 2.5) > 0
                    cw = np.percentile(dW, 97.5) < 0
                    clear_t += ct
                    clear_w += cw
                    clear_b += ct and cw
                res["by_n_gold_keep"][str(N)] = {"rows_median": int(np.median(rows_used)),
                                                 "power_true_keep": r4(clear_t / sims), "power_wrong_keep": r4(clear_w / sims),
                                                 "power_both": r4(clear_b / sims)}
            need = [int(n) for n, v in res["by_n_gold_keep"].items() if v["power_both"] >= 0.8]
            res["smallest_n_gold_keep_power_0.8"] = min(need) if need else f"> {max(int(n) for n in res['by_n_gold_keep'])}"
            out[f"{mix_name}|{sname}"] = res
    return out


# ---------------------------------------------------------------- main


def first_pass(T: dict, bucket: np.ndarray) -> dict:
    sm = scope_masks(T)
    out = {}
    for sc in ("t4_all", "t4_pass", "t123_all", "t123_pass") + TESTS:
        m = sm[sc]
        gk = m & (T["gold"] == K)
        tk, wk = tkwk(T["pick"], T["gold"], m)
        out[sc] = {"rows": int(m.sum()), "gold_keep": int(gk.sum()), "true_keep": tk, "wrong_keep": wk,
                   "keep_calls": tk + wk, "wrong_gold_explore": int((m & (T["pick"] == K) & (T["gold"] == E)).sum()),
                   "wrong_gold_skip": int((m & (T["pick"] == K) & (T["gold"] == S)).sum()),
                   "miss_buckets": dict(Counter(bucket[gk & (T["pick"] != K)])),
                   "pass_slice_keep_rate": r4((T["gold"][m & T["pass"]] == K).mean())}
    return out


def write_rows(T: dict, bucket: np.ndarray, labs: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i in range(T["n"]):
            r = {"row_id": T["row_id"][i], "test": T["test"][i], "posting_id": T["posting"][i],
                 "gold": LABELS[T["gold"][i]], "label_A": LABELS[T["A"][i]], "label_B": LABELS[T["Bl"][i]],
                 "adjudicator_side": T["adj"][i], "agreed": bool(T["agreed"][i]), "eligibility_pass": bool(T["pass"][i]),
                 "rule": T["rule"][i], "hard_cats": sorted(T["cats"][i]), "salary_soft": bool(T["sal"][i]),
                 "hpp_skip": bool(T["hskip"][i]), "s_B": r4(T["s_B"][i]), "pkeep_B": r4(T["pk_B"][i]),
                 "margin_B": r4(T["margin_B"][i]), "argmax_B": LABELS[T["argmax_B"][i]],
                 "s_h": r4(T["s_h"][i]), "s_base": r4(T["s_base"][i]),
                 "s_A": None if np.isnan(T["s_A"][i]) else r4(T["s_A"][i]),
                 "pre_layer": LABELS[T["pre"][i]], "pick": LABELS[T["pick"][i]],
                 "keep_outcome": ("true_keep" if T["pick"][i] == K and T["gold"][i] == K else
                                  "wrong_keep" if T["pick"][i] == K else "missed_keep" if T["gold"][i] == K else ""),
                 "miss_bucket": bucket[i], "archetype": T["archetype"][i], "target_pass": T["target_pass"][i],
                 "posting_length": T["posting_length"][i], "teaser": T["teaser"][i],
                 "kind": T["kind"][i], "pool_field": T["pool_field"][i],
                 "levers": {k: LABELS[v[i]] for k, v in labs.items()}}
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default=str(ROOT / "eval/diag_keep_levers.json"))
    ap.add_argument("--rows", default=str(SCRATCH / "numbers_rows.jsonl"))
    ap.add_argument("--power-sims", type=int, default=300)
    ap.add_argument("--power-inner", type=int, default=1000)
    a = ap.parse_args()
    T = load()
    bucket = miss_bucket(T)
    write_rows(T, bucket, {}, Path(a.rows))  # early, so the join table exists even if a later step fails
    sl = Slices(T)
    res = {"what": "Keep levers for the cycle-3 pick (B + H'' without work_mode, decider): where the missed Keeps go, "
                   "what the wrong Keeps share, why the Keep rate falls, what each lever does to (true Keeps, wrong "
                   "Keeps) with a paired 95% two-way bootstrap interval, and the test-5 size a lever needs. Stored "
                   "files only; aggregates only.",
           "script": "scripts/diag_keep_levers.py", "draws": DRAWS, "seed": SEED,
           "pick": {"system": PICK, "t_hi": T_HI, "t_lo": T_LO, "H''_cats": sorted(HPP), "salary_cap": True},
           "slices": "t4_pass = fresh test 4 eligibility-pass rows (the pass bar's slice), t4_all = all test-4 rows, "
                     "t123_all = fresh tests 1-3 pooled (leave-one-test-out for fitted levers; the pick is in-sample "
                     "there), t123_pass its pass slice",
           "fit_rules": {"count_match": "a threshold that trades Keeps is set so final Keep calls on the fit tests "
                                        "equal the pick's there; a refit t_lo matches the pick's Skip calls outside H''",
                         "net": "levers that only remove or only add Keeps maximise true minus wrong Keeps on the "
                                "fit tests; ties to the smaller change",
                         "where": "fit on tests 1-3 pooled -> test 4; tests 1-3 numbers are leave-one-test-out"},
           "intervals": "d_true_keep and d_wrong_keep are [point, 2.5%, 97.5%] of lever minus pick on shared draws; "
                        "*_crosses_zero says whether the interval holds 0"}
    res["0_reproduction"] = first_pass(T, bucket)
    res["1_missed_keeps"] = section1(T, bucket)
    res["2_wrong_vs_true_keeps"] = section2(T)
    res["3_keep_rate"] = section3(T)
    r4_, labs = section4(T, sl)
    res.update(r4_)
    write_rows(T, bucket, labs, Path(a.rows))
    res["5_power"] = section5(T, sl, a.power_sims, a.power_inner)
    with open(a.summary, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    print("wrote", a.summary, "and", a.rows)


if __name__ == "__main__":
    main()
