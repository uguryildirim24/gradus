import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hybrid_rules as hr  # noqa: E402
from matcher import RequirementCard, full_match  # noqa: E402

CARD = RequirementCard(
    source_posting_id="p1", title="Lab Technician", field="biology", industry="biotech",
    degree_level="bachelor", degree_fields=("biochemistry", "biology"), min_years=0,
    must_skills=(("pcr",), ("cell culture",)), nice_skills=(), license=None, locations=("Boston",),
    work_mode="onsite", sponsorship="will_not_sponsor", salary_min=50_000, salary_max=70_000,
    employment_type="full_time")


def test_rule_verdict_names_the_hard_categories_and_splits_degree_level_from_field():
    base = full_match(CARD)
    assert hr.rule_verdict(base, CARD)["rule"] == "Keep"
    v = hr.rule_verdict(replace(base, work_auth="needs_h1b_now"), CARD)
    assert v["rule"] == "Skip" and v["hard_cats"] == ["work_auth"]
    v = hr.rule_verdict(replace(base, degree_level="associate"), CARD)
    assert v["hard_cats"] == ["degree_level"]
    v = hr.rule_verdict(replace(base, degree_field="philosophy"), CARD)
    assert v["hard_cats"] == ["degree_field"]
    v = hr.rule_verdict(replace(base, skills=()), CARD)
    assert v["rule"] == "Skip" and v["hard_cats"] == ["skills"]


def test_work_mode_skip_is_its_own_category_and_h_prime_overrides_it():
    base = full_match(CARD)
    v = hr.rule_verdict(replace(base, remote_preference="remote_only"), CARD)
    assert v["rule"] == "Skip" and v["hard_cats"] == ["work_mode"]
    assert hr.hybrid("Keep", v, hr.HP_CATS, True) == "Skip"
    assert hr.hybrid("Explore", v) == "Explore"  # H (pre-registered) is unchanged: work mode is not its category
    away = replace(base, remote_preference="remote_only", current_location="Denver, CO", willing_locations=())
    assert hr.rule_verdict(away, CARD)["hard_cats"] == ["location"]  # a place fail stays location


def test_work_mode_on_a_thin_card_leaves_the_model_alone():
    v = hr.rule_verdict(replace(full_match(CARD), remote_preference="hybrid_or_remote"), replace(CARD, thin_posting=True))
    assert v["rule"] == "Explore" and v["hard_cats"] == []
    assert hr.hybrid("Keep", v, hr.HP_CATS, True) == "Keep"


def test_salary_floor_is_soft_and_flagged():
    v = hr.rule_verdict(replace(full_match(CARD), salary_floor=90_000), CARD)
    assert v["rule"] == "Explore" and v["salary_soft"] and v["hard_cats"] == []


def test_hybrid_overrides_only_on_its_categories():
    auth = {"rule": "Skip", "hard_cats": ["work_auth"], "salary_soft": False}
    skill = {"rule": "Skip", "hard_cats": ["skills"], "salary_soft": False}
    field = {"rule": "Skip", "hard_cats": ["degree_field", "skills"], "salary_soft": False}
    pay = {"rule": "Explore", "hard_cats": [], "salary_soft": True}
    assert hr.hybrid("Keep", auth) == "Skip"
    assert hr.hybrid("Keep", skill) == "Keep"  # a skill Skip is the model's call
    assert hr.hybrid("Explore", field) == "Explore"
    assert hr.hybrid("Explore", field, hr.HP_CATS) == "Skip"
    assert hr.hybrid("Keep", pay) == "Keep"
    assert hr.hybrid("Keep", pay, hr.HP_CATS, salary_cap=True) == "Explore"
    assert hr.hybrid("Skip", pay, hr.HP_CATS, salary_cap=True) == "Skip"  # a cap, never a raise


def test_paired_counts_fixed_and_broken_rows():
    gold = ["Keep", "Skip", "Skip", "Explore"]
    a = ["Keep", "Skip", "Skip", "Keep"]
    b = ["Keep", "Explore", "Skip", "Explore"]
    p = hr.paired(gold, a, b, ["x", "x", "y", "y"], draws=50)
    assert p["rows_fixed"] == 1 and p["rows_broken"] == 1 and p["accuracy_diff"][0] == 0.0


def test_override_report_counts_agreement_with_gold():
    rows = [{"pair_id": "a", "gold": "Skip", "model": "Explore",
             "verdict": {"rule": "Skip", "hard_cats": ["enrollment"], "salary_soft": False}},
            {"pair_id": "b", "gold": "Keep", "model": "Keep",
             "verdict": {"rule": "Skip", "hard_cats": ["work_auth"], "salary_soft": False}},
            {"pair_id": "c", "gold": "Keep", "model": "Keep",
             "verdict": {"rule": "Skip", "hard_cats": ["skills"], "salary_soft": False}}]
    hr.attach(rows, {"a": "Explore", "b": "Keep", "c": "Keep"})
    o = hr.override_report(rows, "H")
    assert o["fired"] == 2 and o["gold_agrees"] == 1 and o["changed_model_label"] == 2
    assert [w["pair_id"] for w in o["wrong"]] == ["b"]


def test_macro_f1_present_averages_only_labels_seen():
    gold = ["Skip", "Skip", "Keep"]
    assert hr.macro_f1_present(gold, ["Skip", "Skip", "Skip"]) == round((0.8 + 0.0) / 2, 4)
    assert hr.macro_f1_present(gold, gold) == 1.0


def test_v2_score_keep_recall_by_ask_gap_and_pairs_against_ref():
    rows = [{"gold": g, "ask_gap": n, "base_rq_id": p} for g, n, p in
            [("Keep", 6, "x"), ("Keep", 1, "x"), ("Skip", 0, "y"), ("Keep", 9, "y")]]
    cols = {"c|raw": ["Skip", "Keep", "Skip", "Skip"], "c|H": ["Skip", "Keep", "Skip", "Skip"],
            "c|H_prime": ["Skip", "Keep", "Skip", "Skip"],
            "e|raw": ["Keep", "Keep", "Keep", "Keep"], "e|H": ["Keep", "Keep", "Skip", "Keep"],
            "e|H_prime": ["Keep", "Keep", "Skip", "Keep"]}
    out = hr.v2_score(rows, cols, "c", draws=50)
    s = out["scores"]
    assert s["c|raw"]["keep_recall"][0] == round(1 / 3, 4)
    assert s["c|raw"]["keep_recall_gap_over_4"][0] == 0.0 and s["e|raw"]["keep_recall_gap_over_4"][0] == 1.0
    assert s["e|raw"]["unsafe_keep"] == 1 and s["e|H"]["accuracy"][0] == 1.0
    p = out["paired"]
    assert p["e|raw - c|raw"]["keep_recall_gap_over_4"][0] == 1.0 and p["e|raw - c|raw"]["rows_fixed"] == 2
    assert p["e|H - e|raw"]["rows_fixed"] == 1 and "c|raw - c|raw" not in p
    assert set(p) >= {"c|H_prime - c|H", "e|H_prime - c|H_prime"}


def test_drop_postings_leaves_out_whole_postings_and_counts_them():
    rows = [{"base_rq_id": p, "pair_id": i} for i, p in enumerate("xxyz")]
    kept, n = hr.drop_postings(rows, ["x"])
    assert [r["pair_id"] for r in kept] == [2, 3] and n == {"x": 2}
    import pytest
    with pytest.raises(SystemExit):
        hr.drop_postings(rows, ["w"])
