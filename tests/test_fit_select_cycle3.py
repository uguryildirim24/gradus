"""fit_select_cycle3: H'' variants, the decider readout, the cut-off grid search and the pick (amendments 5, 5b) on
synthetic numbers (no data/ file is read)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import fit_select_cycle3 as S  # noqa: E402
from hybrid_rules import hybrid  # noqa: E402
from matcher.evalstats import LABELS, macro_f1_of, pick  # noqa: E402

A5 = json.loads((ROOT / "eval/protocol_amendment_5.json").read_text())
A5B = json.loads((ROOT / "eval/protocol_amendment_5b.json").read_text())


def test_variants_are_amendment_5s():
    v = A5["eligibility_layer"]["variants"]
    for name in S.VARIANTS:
        listed = v[name].split(":", 1)[1].split("+")[0]
        assert S.VARIANTS[name] == {x.strip() for x in listed.split(",")}
        assert "salary cap" in v[name]
    assert list(S.VARIANTS) == ["H''_without_work_mode", "H''_with_work_mode"]  # tie order


def test_floors_margin_and_order_are_the_amendments():
    assert S.FLOORS == {"Keep": A5B["selection"]["keep_recall_floor"], "Explore": A5["selection"]["recall_floor"],
                        "Skip": A5["selection"]["recall_floor"]}
    assert S.MARGIN == A5["selection"]["tie_margin"]
    assert S.CANDIDATES == ("h", "A", "B") and S.READOUTS == ("argmax", "decider")
    assert len(S.GRID_LEVELS) == 201 and S.GRID_LEVELS[1] == pytest.approx(0.005)


def test_decider_score_is_pkeep_plus_half_pexplore():
    sc = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0], [-5.0, 3.0, 1.0]])
    e = np.exp(sc)
    p = e / e.sum(1, keepdims=True)
    assert np.allclose(S.decider_score(sc), p[:, 0] + 0.5 * p[:, 1])


def test_argmax_idx_is_evalstats_pick_with_ties():
    rng = np.random.default_rng(0)
    sc = np.round(rng.normal(size=(500, 3)), 0)  # many exact ties
    assert list(S.argmax_idx(sc)) == [pick(list(x)) for x in sc]


def rand_rows(n, seed=1):
    rng = np.random.default_rng(seed)
    cats = ["work_auth", "enrollment", "degree_level", "degree_field", "employment_type", "location", "work_mode"]
    rows = []
    for _ in range(n):
        rule = ["Keep", "Explore", "Skip"][rng.integers(3)]
        hc = sorted(set(rng.choice(cats, rng.integers(1, 3)))) if rule == "Skip" else []
        rows.append({"verdict": {"rule": rule, "hard_cats": hc, "salary_soft": rule == "Explore" and bool(rng.integers(2))}})
    return rows


def test_vectorised_override_is_hybrid():
    rows = rand_rows(600)
    lab = np.random.default_rng(2).integers(0, 3, 600)
    for cats in S.VARIANTS.values():
        hard, sal = S.override_masks(rows, cats)
        want = [hybrid(LABELS[x], r["verdict"], cats, True) for x, r in zip(lab, rows)]
        assert [LABELS[x] for x in S.apply_h(lab, hard, sal)] == want
        assert [LABELS[x] for x in S.apply_h_rows(lab, rows, cats)] == want


def test_class_stats_is_macro_f1_of():
    rng = np.random.default_rng(3)
    g = rng.integers(0, 3, 300)
    P = rng.integers(0, 3, (50, 300))
    P[0] = 2  # one class never predicted
    f1, rec = S.class_stats(g, P)
    for k in range(50):
        assert f1[k] == pytest.approx(macro_f1_of(g, P[k]), abs=1e-12)
        for w in range(3):
            assert rec[k, w] == pytest.approx(((g == w) & (P[k] == w)).sum() / (g == w).sum())


def test_decider_idx_and_equal_cutoffs_give_no_explore():
    s = np.array([0.1, 0.3, 0.5, 0.7])
    assert list(S.decider_idx(s, 0.3, 0.7)) == [2, 1, 1, 0]
    assert 1 not in S.decider_idx(s, 0.5, 0.5)


def test_fit_cutoffs_separates_a_clean_three_way_score():
    # gold Skip on low s, Explore in the middle, Keep high: a perfect pair exists
    s = np.concatenate([np.linspace(0.0, 0.2, 40), np.linspace(0.4, 0.6, 30), np.linspace(0.8, 1.0, 30)])
    g = np.array([2] * 40 + [1] * 30 + [0] * 30)
    no = np.zeros(len(s), bool)
    c = S.fit_cutoffs(s, g, no, no)
    assert c["macro_f1"] == pytest.approx(1.0) and c["floors_met"] == "0.60/0.25/0.25"
    p = S.decider_idx(s, c["t_lo"], c["t_hi"])
    assert (p == g).all()
    # the lowest perfect t_lo is the first grid value above every Skip s, the lowest t_hi the first above every Explore
    grid = np.quantile(s, S.GRID_LEVELS)
    assert c["t_lo"] == grid[grid > 0.2].min() and c["t_hi"] == grid[grid > 0.6].min()
    assert c["pairs_tried"] == 201 * 202 // 2


def test_fit_cutoffs_override_and_fallback_flag():
    s = np.concatenate([np.linspace(0.0, 0.2, 40), np.linspace(0.4, 0.6, 30), np.linspace(0.8, 1.0, 30)])
    g = np.array([2] * 40 + [1] * 30 + [0] * 30)
    hard = np.zeros(len(s), bool)
    hard[78:] = True  # 22 of 30 gold Keeps forced to Skip: Keep recall reaches 8/30 at most, under 0.60, over 0.25
    c = S.fit_cutoffs(s, g, hard, np.zeros(len(s), bool))
    assert c["floors_met"] == "fallback_0.25" and c["recall"]["Keep"] == pytest.approx(8 / 30)
    hard[:] = False
    hard[72:] = True  # 28 of 30 forced: no pair reaches 0.25 Keep recall either
    c = S.fit_cutoffs(s, g, hard, np.zeros(len(s), bool))
    assert c["t_lo"] is None and c["floors_met"] is None


def ex(f1, k=0.7, e=0.3, s=0.8):
    return {"macro_f1": f1, "recall": {"Keep": k, "Explore": e, "Skip": s}, "accuracy": 0.5}


def test_choose_margin_and_tie_order():
    w, wo = "H''_with_work_mode", "H''_without_work_mode"
    sys_ = {S.system_name("B", wo, "argmax"): ex(0.600),
            S.system_name("A", w, "argmax"): ex(0.604),
            S.system_name("h", w, "decider"): ex(0.596),   # within 0.005 of 0.604? 0.008 away: not tied
            S.system_name("A", wo, "decider"): ex(0.599),  # 0.005 away exactly: tied
            S.system_name("h", wo, "argmax"): ex(0.700, k=0.59)}  # best but under the Keep floor
    c = S.choose(sys_)
    assert not c["no_system_cleared_floors"]
    assert c["tied_within_margin"] == [S.system_name("A", wo, "decider"), S.system_name("B", wo, "argmax"),
                                       S.system_name("A", w, "argmax")]
    assert c["pick"] == S.system_name("A", wo, "decider")


def test_choose_argmax_before_decider_and_fallback():
    wo = "H''_without_work_mode"
    c = S.choose({S.system_name("h", wo, "decider"): ex(0.6), S.system_name("h", wo, "argmax"): ex(0.6)})
    assert c["pick"] == S.system_name("h", wo, "argmax")
    c = S.choose({S.system_name("h", wo, "argmax"): ex(0.7, k=0.3), S.system_name("B", wo, "argmax"): ex(0.8, k=0.5)})
    assert c["no_system_cleared_floors"] and c["pick"] == S.system_name("B", wo, "argmax")


def test_build_fits_every_model_and_variant_and_keeps_base_out_of_the_pick():
    rng = np.random.default_rng(5)
    n = 300
    rows = [r | {"gold": LABELS[rng.integers(3)]} for r in rand_rows(n, 7)]
    g = np.array([LABELS.index(r["gold"]) for r in rows])
    sc = {m: rng.normal(size=(n, 3)) + 2.5 * np.eye(3)[g] * (1 if m != "base" else 3) for m in ("h", "A", "B", "base")}
    B = S.build(rows, sc)
    assert set(B["cutoffs"]) == {f"{m}+{v}" for m in sc for v in S.VARIANTS}
    assert len([k for k in B["labels"] if not k.startswith("base")]) == 12
    assert not B["choice"]["pick"].startswith("base")
    for k, p in B["labels"].items():
        assert B["exact"][k]["macro_f1"] == pytest.approx(macro_f1_of(g, p), abs=1e-12)
        m, v, r = S.parse(k)
        if r == "decider":
            c = B["cutoffs"][f"{m}+{v}"]
            assert c["macro_f1"] == pytest.approx(B["exact"][k]["macro_f1"], abs=1e-12)
