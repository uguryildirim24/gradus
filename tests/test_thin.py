"""The thin-posting rule: a posting that says too little to judge the fit is Explore, hard eligibility fails still win,
and every card without the flag labels (and serializes) exactly as before."""
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hybrid_rules as hr  # noqa: E402
from matcher import CandidateCard, RequirementCard, full_match  # noqa: E402
from matcher.blueprint import OUTPUT_KEYS, PIPELINE_KEYS  # noqa: E402
from matcher.rules import (OTHERWISE_OK, THIN_CHARS, THIN_REASON, format_completion, is_thin, label,  # noqa: E402
                           mark_thin, reasons_by_check, says_too_little)
from matcher.visibility import missing_facts  # noqa: E402

TEASER = RequirementCard(source_posting_id="v1", title="Account Manager", field="Sales", industry="healthcare",
                         locations=("Boston, MA",), work_mode="onsite", employment_type="full_time")
SHORT = "Join our growing team as an Account Manager in Boston. Apply today!"
LONG = "x" * THIN_CHARS


def test_says_too_little_needs_an_empty_card():
    assert says_too_little(TEASER)
    assert says_too_little(replace(TEASER, must_skills=(("Communication",),), nice_skills=("Teamwork",)))  # soft only
    assert not says_too_little(replace(TEASER, must_skills=(("Salesforce",),)))
    assert not says_too_little(replace(TEASER, nice_skills=("Salesforce",)))
    assert not says_too_little(replace(TEASER, degree_level="bachelor"))
    assert not says_too_little(replace(TEASER, degree_fields=("Biology",)))
    assert not says_too_little(replace(TEASER, min_years=2))
    assert not says_too_little(replace(TEASER, license="RN license"))
    # eligibility facts do not make a posting judgeable for fit
    assert says_too_little(replace(TEASER, citizens_only=True, sponsorship="will_not_sponsor", enrolled_in="bachelor"))


def test_is_thin_needs_short_text():
    assert is_thin(TEASER, SHORT)
    assert is_thin(TEASER, "")
    assert not is_thin(TEASER, LONG)
    assert not is_thin(replace(TEASER, min_years=1), SHORT)
    assert mark_thin(TEASER, SHORT).thin_posting
    assert not mark_thin(TEASER, LONG).thin_posting
    assert not mark_thin(replace(TEASER, thin_posting=True), LONG).thin_posting  # recomputed, not kept


def test_thin_match_is_explore_with_its_reason():
    card = mark_thin(TEASER, SHORT)
    b, reasons, _ = label(full_match(card), card)
    assert b == "Explore"
    assert reasons == [THIN_REASON]
    assert OTHERWISE_OK not in reasons
    assert format_completion(b, reasons).startswith("Explore — the posting says too little to judge the fit")
    # the same candidate on the same card without the flag is a Keep
    assert label(full_match(TEASER), TEASER)[0] == "For you"


def test_thin_keeps_other_soft_reasons_first():
    card = mark_thin(replace(TEASER, employment_type="full_time"), SHORT)
    c = replace(full_match(card), work_auth="f1_opt", degree_level="bachelor", degree_field="Biology",
                graduation_year=2026)
    b, reasons, _ = label(c, card)
    assert b == "Explore"
    assert reasons[-1] == THIN_REASON and len(reasons) == 2
    assert "OPT" in reasons[0]


@pytest.mark.parametrize("edit, card_edit", [
    (dict(current_location="Denver, CO"), {}),
    (dict(employment_types_wanted=("part_time",)), {}),
    (dict(work_auth="needs_h1b_now", degree_level="bachelor", degree_field="Biology", graduation_year=2024),
     dict(sponsorship="will_not_sponsor")),
    (dict(work_auth="f1_opt", degree_level="bachelor", degree_field="Biology", graduation_year=2026),
     dict(citizens_only=True)),
])
def test_hard_fails_still_win(edit, card_edit):
    card = mark_thin(replace(TEASER, **card_edit), SHORT)
    assert card.thin_posting
    b, reasons, _ = label(replace(full_match(card), **edit), card)
    assert b == "Excluded"
    assert THIN_REASON not in reasons


def test_thin_reason_belongs_to_its_own_check_and_states_no_hidden_fact():
    card = mark_thin(TEASER, SHORT)
    c = full_match(card)
    assert reasons_by_check(c, card)[THIN_REASON] == "thin"
    assert missing_facts(c, card, [THIN_REASON], "Candidate", "Lives in: Boston, MA", SHORT) == []


def test_hybrid_is_unchanged_by_the_flag():
    """H' reads only Skip categories and the salary reason, so the flag can never change what it outputs."""
    card = mark_thin(replace(TEASER, salary_max=50_000), SHORT)
    base = full_match(card)
    for cand in (base, replace(base, salary_floor=90_000), replace(base, current_location="Denver, CO"),
                 replace(base, employment_types_wanted=("internship",))):
        thin, plain = hr.rule_verdict(cand, card), hr.rule_verdict(cand, replace(card, thin_posting=False))
        assert thin["hard_cats"] == plain["hard_cats"]
        assert thin["salary_soft"] == plain["salary_soft"]
        for model in ("Keep", "Explore", "Skip"):
            for cats, cap in hr.VARIANTS.values():
                assert hr.hybrid(model, thin, cats, cap) == hr.hybrid(model, plain, cats, cap)


def test_card_serializes_as_before_without_the_flag():
    d = TEASER.to_dict()
    assert "thin_posting" not in d
    assert RequirementCard.from_dict(d) == TEASER
    t = replace(TEASER, thin_posting=True).to_dict()
    assert t["thin_posting"] is True
    assert RequirementCard.from_dict(json.loads(json.dumps(t))).thin_posting


def test_flag_is_a_pipeline_key_never_extracted():
    assert "thin_posting" in PIPELINE_KEYS and "thin_posting" not in OUTPUT_KEYS
    assert set(OUTPUT_KEYS) | set(PIPELINE_KEYS) == {f.name for f in fields(RequirementCard)}


MANIFEST = ROOT / "data/built/pairs/manifest.jsonl"


@pytest.mark.skipif(not MANIFEST.exists(), reason="data/ is gitignored; the pair manifest lives in the worktree only")
def test_every_stored_synthetic_label_is_unchanged():
    """The 10,500 rule labels the pair builder stored before the thin rule existed come out the same, except the pairs
    whose candidate's work arrangement rules the role's work mode out: since fix cycle 2 (eval/fix_cycle2.json) those
    are Excluded, on that reason (before, their soft one) or with it as one more hard reason."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import render_cards as rc
    from matcher.rules import WORK_MODE_PREFIX, mode_ruled_out

    reqs, cands = rc.load_cards("postings"), rc.load_cards("resumes")
    n = moved = 0
    for line in MANIFEST.open():
        m = json.loads(line)
        if m["rq_id"] not in reqs or m["cd_id"] not in cands:
            continue
        c, r = CandidateCard.from_dict(cands[m["cd_id"]]), RequirementCard.from_dict(reqs[m["rq_id"]])
        b, reasons, _ = label(c, r)
        n += 1
        mode = [x for x in reasons if x.startswith(WORK_MODE_PREFIX)]
        if mode:  # a place fail comes first and writes no work-mode reason: such a pair is unchanged
            assert mode_ruled_out(c, r) and b == "Excluded" and len(mode) == 1, m["pair_id"]
            if m["bucket"] == "Excluded":  # already a Skip: the work mode is one more hard reason
                assert format_completion(b, [x for x in reasons if x not in mode]) == m["completion"], m["pair_id"]
            else:
                assert mode[0] in m["completion"], m["pair_id"]
                moved += 1
            continue
        assert b == m["bucket"], m["pair_id"]
        assert format_completion(b, reasons) == m["completion"], m["pair_id"]
    assert n > 9000 and moved < n // 10
