import inspect
import random
import re
from dataclasses import replace

import pytest

from matcher import (
    CandidateCard,
    RequirementCard,
    format_completion,
    full_match,
    label,
    make_variants,
    parse_completion,
)
from matcher import rules
from matcher.schema import WORK_AUTH, skill_meets
from matcher.variants import industry_clash, prefs_fit, unique_skills, years_fit

CARD = RequirementCard(
    source_posting_id="p1",
    title="Lab Technician",
    field="biology",
    industry="biotech",
    degree_level="bachelor",
    degree_fields=("biochemistry", "biology"),
    min_years=2,
    must_skills=(("pcr",), ("cell culture",)),
    nice_skills=("python",),
    license=None,
    locations=("Boston",),
    work_mode="onsite",
    sponsorship="unstated",
    salary_min=50_000,
    salary_max=70_000,
    employment_type="full_time",
)


def bucket(c, r=CARD):
    return label(c, r)[0]


@pytest.fixture
def base():
    return full_match(CARD)


def test_full_match_is_for_you(base):
    b, reasons, flags = label(base, CARD)
    assert b == "For you" and reasons and flags == []


def test_label_is_deterministic(base):
    assert label(base, CARD) == label(base, CARD)


# ---- hard filters -------------------------------------------------------

def test_degree_too_low_excludes(base):
    assert bucket(replace(base, degree_level="associate")) == "Excluded"


def test_higher_degree_ok(base):
    assert bucket(replace(base, degree_level="master")) == "For you"


def test_wrong_degree_field_excludes(base):
    assert bucket(replace(base, degree_field="classics")) == "Excluded"


def test_degree_field_case_insensitive(base):
    assert bucket(replace(base, degree_field="  Biochemistry ")) == "For you"


def test_missing_must_skill_excludes(base):
    assert bucket(replace(base, skills=("pcr", "python"))) == "Excluded"


def test_must_skill_case_insensitive(base):
    assert bucket(replace(base, skills=("PCR", "Cell  Culture", "python"))) == "For you"


def test_skill_light_normalization(base):
    assert bucket(replace(base, skills=("pcr", "cell-culture.", "Python"))) == "For you"


ANY_OF = replace(CARD, must_skills=(("pcr",), ("C", "C++", "Java")))


@pytest.mark.parametrize("held", ["C", "c++", "JAVA"])
def test_any_of_group_met_by_any_alternative(held):
    c = replace(full_match(ANY_OF), skills=("pcr", held, "python"))
    assert bucket(c, ANY_OF) == "For you"


def test_any_of_group_unmet_names_all_alternatives():
    c = replace(full_match(ANY_OF), skills=("pcr", "python", "rust"))
    b, reasons, _ = label(c, ANY_OF)
    assert b == "Excluded" and "needs one of: C, C++, Java" in reasons


def test_single_and_group_misses_both_reported():
    c = replace(full_match(ANY_OF), skills=("python",))
    _, reasons, _ = label(c, ANY_OF)
    assert reasons[:2] == ["missing required skill: pcr", "needs one of: C, C++, Java"]


def test_flat_must_skills_rejected():
    with pytest.raises(ValueError):
        replace(CARD, must_skills=("pcr", "cell culture"))
    with pytest.raises(ValueError):
        replace(CARD, must_skills=((),))


def test_license_required():
    card = replace(CARD, license="RN")
    c = replace(full_match(card), licenses=())
    assert bucket(c, card) == "Excluded"
    assert bucket(replace(c, licenses=("rn",)), card) == "For you"


def test_wrong_employment_type_excludes(base):
    assert bucket(replace(base, employment_types_wanted=("internship",))) == "Excluded"


def test_any_employment_type_when_empty(base):
    assert bucket(replace(base, employment_types_wanted=())) == "For you"


def test_location_impossible_excludes(base):
    c = replace(base, current_location="Denver", willing_locations=("Austin",))
    assert bucket(c) == "Excluded"


def test_willing_location_ok(base):
    c = replace(base, current_location="Denver", willing_locations=("boston",))
    assert bucket(c) == "For you"


def test_remote_job_ignores_location(base):
    card = replace(CARD, work_mode="remote")
    assert bucket(replace(base, current_location="Denver"), card) == "For you"


@pytest.mark.parametrize("card_loc,cand_loc", [
    ("NYC, NY", "New York, NY"),
    ("New York City", "New York, NY"),
    ("New York, New York", "New York, NY"),
    ("SF, CA", "San Francisco, CA"),
    ("DC, DC", "Washington, DC"),
    ("Washington, D.C.", "Washington, DC"),
    ("Saint Louis, MO", "St. Louis, MO"),
])
def test_city_aliases_match(base, card_loc, cand_loc):
    card = replace(CARD, locations=(card_loc,))
    assert bucket(replace(base, current_location=cand_loc), card) == "For you"


def test_same_city_other_state_does_not_match(base):
    card = replace(CARD, locations=("Portland, OR",))
    assert bucket(replace(base, current_location="Portland, ME"), card) == "Excluded"


def test_onsite_without_listed_locations_is_unconstrained(base):
    card = replace(CARD, locations=())
    assert bucket(replace(base, current_location="Denver"), card) == "For you"


# ---- soft gaps ----------------------------------------------------------

def test_one_year_short_explores(base):
    assert bucket(replace(base, years=1)) == "Explore"


def test_two_years_short_excludes(base):
    assert bucket(replace(base, years=0)) == "Excluded"


def test_exact_years_ok(base):
    assert bucket(replace(base, years=2)) == "For you"


def test_overqualified_explores(base):
    b, reasons, _ = label(replace(base, years=2 + rules.OVERQUALIFIED_YEARS), CARD)
    assert b == "Explore" and any("overqualified" in r for r in reasons)
    assert bucket(replace(base, years=2 + rules.OVERQUALIFIED_YEARS - 1)) == "For you"


def test_missing_nice_skill_explores(base):
    # CARD lists one nice-to-have: missing it alone is no stretch; missing two of two is
    assert bucket(replace(base, skills=("pcr", "cell culture"))) == "For you"
    card = replace(CARD, nice_skills=("python", "r"))
    assert bucket(replace(base, skills=("pcr", "cell culture")), card) == "Explore"


def test_salary_below_floor_explores(base):
    assert bucket(replace(base, salary_floor=90_000)) == "Explore"


def test_unknown_salary_never_penalised(base):
    card = replace(CARD, salary_min=None, salary_max=None)
    assert bucket(replace(base, salary_floor=500_000), card) == "For you"


def test_industry_wanted_mismatch_explores(base):
    assert bucket(replace(base, industries_wanted=("finance",))) == "Explore"


def test_industry_avoided_explores(base):
    assert bucket(replace(base, industries_avoided=("Biotech",))) == "Explore"


def test_role_avoided_explores(base):
    assert bucket(replace(base, roles_avoided=("technician",))) == "Explore"


@pytest.mark.parametrize("pref,mode,reason", [
    ("remote_only", "onsite", "the role is onsite; you said remote only"),
    ("remote_only", "hybrid", "the role is hybrid; you said remote only"),
    ("hybrid_or_remote", "onsite", "the role is onsite; you prefer hybrid or remote"),
])
def test_work_mode_the_profile_rules_out_is_excluded(base, pref, mode, reason):
    """The rubric: a work mode the profile rules out is Skip, whatever else fits."""
    b, reasons, _ = label(replace(base, remote_preference=pref), replace(CARD, work_mode=mode))
    assert b == "Excluded" and reasons == [reason]
    assert rules.reasons_by_check(replace(base, remote_preference=pref), replace(CARD, work_mode=mode))[reason] == "location"


@pytest.mark.parametrize("pref,mode", [("hybrid_or_remote", "hybrid"), ("hybrid_or_remote", "remote"),
                                       ("remote_only", "remote"), ("any", "onsite")])
def test_work_mode_the_profile_allows_is_not_a_fail(base, pref, mode):
    card = replace(CARD, work_mode=mode, locations=() if mode == "remote" else CARD.locations)
    assert bucket(replace(base, remote_preference=pref), card) == "For you"


def test_work_mode_on_a_thin_posting_stays_a_stretch(base):
    """A teaser that states nothing about the role cannot rule a mode out: its onsite is only the card default."""
    thin = replace(CARD, thin_posting=True)
    b, reasons, _ = label(replace(base, remote_preference="remote_only"), thin)
    assert b == "Explore" and reasons[0] == "the role is onsite; you said remote only"


def test_place_fail_comes_before_work_mode(base):
    c = replace(base, remote_preference="remote_only", current_location="Denver, CO", willing_locations=())
    b, reasons, _ = label(c, CARD)
    assert b == "Excluded" and len(reasons) == 1 and reasons[0].startswith("based in")


def test_hybrid_pref_vs_hybrid_job_ok(base):
    card = replace(CARD, work_mode="hybrid")
    assert bucket(replace(base, remote_preference="hybrid_or_remote"), card) == "For you"


def test_hard_beats_soft(base):
    c = replace(base, skills=("pcr",), salary_floor=900_000)
    b, reasons, _ = label(c, CARD)
    assert b == "Excluded" and all("pay" not in r for r in reasons)


# ---- work authorization: every case ------------------------------------

def auth(status, stance, employment="full_time", **card_kw):
    card = replace(CARD, sponsorship=stance, employment_type=employment, **card_kw)
    cand = replace(full_match(card), work_auth=status)
    return label(cand, card)


STANCES = ["sponsors", "will_not_sponsor", "will_not_sponsor_or_opt", "unstated"]


@pytest.mark.parametrize("stance", STANCES)
@pytest.mark.parametrize("employment", ["internship", "full_time", "part_time", "contract"])
def test_citizen_or_pr_never_affected(stance, employment):
    assert auth("citizen_or_pr", stance, employment)[0] == "For you"


def test_h1b_now_vs_will_not_sponsor_excludes():
    b, reasons, flags = auth("needs_h1b_now", "will_not_sponsor")
    assert b == "Excluded" and flags
    assert any("sponsor" in r for r in reasons)


def test_h1b_now_vs_sponsors_ok():
    assert auth("needs_h1b_now", "sponsors")[0] == "For you"


def test_h1b_now_vs_unstated_never_excluded():
    b, _, flags = auth("needs_h1b_now", "unstated")
    assert b == "Explore" and "work_auth:sponsorship_unstated" in flags


def test_h1b_now_excluded_even_for_internship_with_no_sponsor():
    assert auth("needs_h1b_now", "will_not_sponsor", "internship")[0] == "Excluded"


@pytest.mark.parametrize("stance", STANCES)
def test_cpt_internship_eligible_whatever_the_stance(stance):
    assert auth("f1_cpt", stance, "internship")[0] == "For you"


def test_cpt_internship_not_excluded_by_no_sponsorship():
    b, _, _ = auth("f1_cpt", "will_not_sponsor", "internship")
    assert b != "Excluded"


@pytest.mark.parametrize("employment", ["full_time", "part_time", "contract"])
def test_cpt_outside_internships_is_explore_with_flag(employment):
    """CPT covers internships and co-ops (profile.WORK_AUTH_TEXT), whatever the hours."""
    for stance in ("unstated", "will_not_sponsor"):
        b, reasons, flags = auth("f1_cpt", stance, employment)
        assert b == "Explore" and "work_auth:cpt_not_internship" in flags
        assert "CPT only covers internships and co-ops" in reasons


@pytest.mark.parametrize("status", ["f1_opt", "stem_opt"])
def test_opt_explores_with_flag_when_not_sponsoring(status):
    for stance in ("will_not_sponsor", "unstated"):
        b, _, flags = auth(status, stance)
        assert b == "Explore" and f"work_auth:{status}_check_sponsorship" in flags


@pytest.mark.parametrize("status", ["f1_opt", "stem_opt"])
def test_opt_for_you_when_card_sponsors(status):
    b, _, flags = auth(status, "sponsors")
    assert b == "For you" and flags == []


@pytest.mark.parametrize("status", ["f1_opt", "stem_opt"])
def test_opt_never_excluded_by_no_sponsorship(status):
    assert auth(status, "will_not_sponsor", "internship")[0] == "Explore"


def test_stem_opt_everify_noted():
    _, reasons, _ = auth("stem_opt", "unstated", e_verify=True)
    assert any("E-Verify" in r for r in reasons)


def test_other_visa_explores_with_flag():
    for stance in STANCES:
        b, _, flags = auth("other_visa", stance)
        assert b == "Explore" and "work_auth:other_visa_check" in flags


@pytest.mark.parametrize("status", [s for s in WORK_AUTH if s != "citizen_or_pr"])
def test_citizens_only_excludes_every_non_citizen(status):
    for stance in STANCES:
        assert auth(status, stance, citizens_only=True)[0] == "Excluded"


def test_citizens_only_allows_citizen_or_pr():
    assert auth("citizen_or_pr", "unstated", citizens_only=True)[0] == "For you"


def test_citizens_only_reason_names_permanent_residents():
    _, reasons, flags = auth("f1_cpt", "unstated", "internship", citizens_only=True)
    assert "work_auth:citizens_only" in flags
    assert any("permanent residents" in r for r in reasons)


@pytest.mark.parametrize("status", ["f1_opt", "stem_opt"])
@pytest.mark.parametrize("employment", ["internship", "full_time"])
def test_no_opt_stance_excludes_opt(status, employment):
    b, reasons, flags = auth(status, "will_not_sponsor_or_opt", employment)
    assert b == "Excluded" and "work_auth:no_opt" in flags
    assert any("OPT" in r for r in reasons)


def test_no_opt_stance_excludes_h1b_now():
    b, _, flags = auth("needs_h1b_now", "will_not_sponsor_or_opt")
    assert b == "Excluded" and "work_auth:needs_sponsorship_no_sponsor" in flags


def test_no_opt_stance_keeps_cpt_internship_eligible():
    assert auth("f1_cpt", "will_not_sponsor_or_opt", "internship")[0] == "For you"


def test_no_opt_stance_cpt_full_time_still_soft():
    b, _, flags = auth("f1_cpt", "will_not_sponsor_or_opt", "full_time")
    assert b == "Explore" and "work_auth:cpt_not_internship" in flags


# ---- enrollment ("for students currently pursuing ...") ---------------

PHD_INTERN = replace(CARD, degree_level="bachelor", enrolled_in="doctoral", min_years=0,
                     employment_type="internship")
GRAD_ONLY = replace(PHD_INTERN, enrolled_in="master")  # master's or PhD students only


def test_phd_intern_admits_current_phd_student():
    c = replace(full_match(PHD_INTERN), degree_level="doctoral", graduation_year=2029)
    b, reasons, _ = label(c, PHD_INTERN)
    assert b == "For you" and any("currently pursuing" in r for r in reasons)


def test_phd_intern_excludes_finished_phd():
    c = replace(full_match(PHD_INTERN), degree_level="doctoral", graduation_year=2024)
    b, reasons, _ = label(c, PHD_INTERN)
    assert b == "Excluded" and any("graduated" in r for r in reasons)


def test_phd_intern_excludes_masters_student():
    c = replace(full_match(PHD_INTERN), degree_level="master", graduation_year=2028)
    assert bucket(c, PHD_INTERN) == "Excluded"


@pytest.mark.parametrize("level,grad,want", [
    ("master", 2027, "For you"),      # current master's student
    ("doctoral", 2030, "For you"),    # current PhD student
    ("bachelor", 2027, "Excluded"),   # undergraduate
    ("master", 2025, "Excluded"),     # holds a master's, no longer enrolled
    ("doctoral", 2026, "Excluded"),   # graduating this year counts as done
])
def test_graduate_students_only(level, grad, want):
    c = replace(full_match(GRAD_ONLY), degree_level=level, graduation_year=grad)
    assert bucket(c, GRAD_ONLY) == want


def test_enrollment_without_graduation_year_excludes():
    c = replace(full_match(GRAD_ONLY), graduation_year=None)
    assert bucket(c, GRAD_ONLY) == "Excluded"


def test_undergrad_enrollment_admits_any_current_student():
    card = replace(CARD, degree_level="none", degree_fields=(), enrolled_in="bachelor",
                   min_years=0, employment_type="internship")
    for level in ("bachelor", "master", "doctoral"):
        c = replace(full_match(card), degree_level=level, graduation_year=2028)
        assert bucket(c, card) == "For you", level
    assert bucket(replace(full_match(card), degree_level="associate", graduation_year=2028), card) == "Excluded"


def test_current_year_is_a_constant():
    assert rules.CURRENT_YEAR == 2026


# ---- invariance ---------------------------------------------------------

COUNTRIES = ["USA", "India", "Nigeria", "Germany", "China", "Brazil", "", "Atlantis"]
LANGUAGES = ["en", "tr", "zh", "es", "ar", ""]


def test_country_and_language_never_change_label():
    for v in make_variants(CARD):
        want = label(v.candidate, v.card)
        for country in COUNTRIES:
            for lang in LANGUAGES:
                swapped = replace(v.candidate, education_country=country, language=lang)
                assert label(swapped, v.card) == want, (v.kind, country, lang)


def test_rules_source_never_mentions_nuisance_fields():
    src = inspect.getsource(rules)
    for word in ("education_country", "language", "name"):
        assert f".{word}" not in src


def test_candidate_card_has_no_name_field():
    assert "name" not in CandidateCard.__dataclass_fields__


# ---- variant generator --------------------------------------------------

def _groups(rng: random.Random, names: list[str], n: int) -> tuple[tuple[str, ...], ...]:
    """n must groups from names; some get an any-of alternative from ALT_POOL."""
    out = []
    for i, name in enumerate(names[:n]):
        out.append((name, ALT_POOL[i]) if rng.random() < 0.4 else (name,))
    return tuple(out)


ALT_POOL = ("r", "google sheets", "matlab")


def random_card(rng: random.Random, i: int) -> RequirementCard:
    skills = ["sql", "excel", "pcr", "welding", "python", "nursing care", "tax law", "cad"]
    rng.shuffle(skills)
    mode = rng.choice(["onsite", "hybrid", "remote"])
    return RequirementCard(
        source_posting_id=f"r{i}",
        title=rng.choice(["Data Analyst", "Staff Nurse", "Machinist", "Teaching Assistant"]),
        industry=rng.choice(["", "health", "tech", "education"]),
        seniority=rng.choice(["entry", "mid", "senior"]),
        degree_level=rng.choice(["none", "associate", "bachelor", "master", "doctoral"]),
        enrolled_in=rng.choice(["none", "none", "associate", "bachelor", "master", "doctoral"]),
        degree_fields=rng.choice([(), ("nursing",), ("statistics", "math")]),
        min_years=rng.choice([0, 1, 2, 3, 5]),
        must_skills=_groups(rng, skills[:3], rng.randint(0, 3)),
        nice_skills=tuple(skills[3 : 3 + rng.randint(0, 2)]),
        license=rng.choice([None, None, "RN", "CPA"]),
        locations=() if mode == "remote" and rng.random() < 0.5 else ("Boston", "Chicago")[: rng.randint(1, 2)],
        work_mode=mode,
        sponsorship=rng.choice(["sponsors", "will_not_sponsor", "will_not_sponsor_or_opt", "unstated"]),
        citizens_only=rng.random() < 0.15,
        e_verify=rng.random() < 0.3,
        salary_min=rng.choice([None, 40_000]),
        salary_max=rng.choice([None, 90_000]),
        employment_type=rng.choice(["internship", "full_time", "part_time", "contract"]),
    )


def test_variants_match_expected_over_many_random_cards():
    rng = random.Random(7)
    kinds = set()
    for i in range(400):
        card = random_card(rng, i)
        for v in make_variants(card, seed=i):
            got = label(v.candidate, v.card)[0]
            assert got == v.expected, (v.kind, v.expected, got, card, v.candidate)
            assert not rules.status_problems(v.candidate), (v.kind, v.candidate)
            assert prefs_fit(v.candidate), (v.kind, v.candidate)
            assert years_fit(v.candidate), (v.kind, v.candidate)
            assert unique_skills(v.candidate.skills) == v.candidate.skills, (v.kind, v.candidate.skills)
            assert not industry_clash(v.candidate), (v.kind, v.candidate.industries_wanted, v.candidate.industries_avoided)
            kinds.add(v.kind)
    assert {
        "full_match", "degree_level", "years_one_short", "years_far_short", "overqualified",
        "must_skill", "nice_skill", "license", "location", "employment_type", "salary",
        "pref_industry_wanted", "pref_industry_avoided", "pref_remote_only", "pref_role_avoided",
        "work_auth_needs_h1b", "work_auth_cpt_internship", "work_auth_f1_opt", "work_auth_stem_opt",
        "work_auth_opt_sponsors", "work_auth_stem_opt_sponsors", "work_auth_h1b_sponsors",
        "work_auth_citizens_only", "degree_field", "degree_field_adjacent", "degree_still_enrolled",
        "work_auth_no_opt", "work_auth_cpt_internship_no_opt", "must_skill_alternative",
        "enrollment_graduated", "enrollment_level", "location_willing", "location_same_metro",
        "location_anywhere", "pref_industry_avoided_other", "pref_role_avoided_other", "work_auth_other_visa",
    } <= kinds


def test_no_variant_wants_internships_and_avoids_intern():
    """The role-avoided break never draws "intern" for someone after an internship, and the
    copy base never has both; the break still fires when the title has another word."""
    card = replace(CARD, title="Lab Technician Intern", employment_type="internship", min_years=0)
    kinds = set()
    for seed in range(40):
        for v in make_variants(card, seed):
            assert prefs_fit(v.candidate), (v.kind, v.candidate.roles_avoided, v.candidate.employment_types_wanted)
            if v.kind == "pref_role_avoided":
                kinds.add(v.candidate.roles_avoided)
    assert ("technician",) in kinds and ("intern",) in kinds  # intern only for those with no internship wish
    assert not prefs_fit(replace(full_match(card), employment_types_wanted=("internship",), roles_avoided=("Intern",)))


def test_no_cpt_student_avoids_intern():
    """CPT covers internships and co-ops only, so a CPT student never lists "intern" as a role to avoid:
    not from the sampler, not from the role-avoided break, not after a status swap (the integration
    review found 23 manifest rows like pr_a25a1d19d0ed)."""
    from matcher.variants import with_status

    card = replace(CARD, title="Lab Technician Intern", employment_type="internship", min_years=0,
                   citizens_only=False, sponsorship="unstated")
    assert not prefs_fit(replace(full_match(card), work_auth="f1_cpt", roles_avoided=("intern",)))
    for seed in range(40):
        for v in make_variants(card, seed):
            if v.candidate.work_auth == "f1_cpt":
                assert "intern" not in v.candidate.roles_avoided, (v.kind, seed)
        base = replace(full_match(card), roles_avoided=("intern",))
        out = with_status(base, "f1_cpt", card, random.Random(seed))
        assert out is None or "intern" not in out.roles_avoided


def test_still_enrolled_break_keeps_a_cpt_students_years_in_reach():
    """A doctoral CPT student with six years of work made a bachelor's student (degree_still_enrolled) keeps
    no more years than a bachelor's student on CPT can have (seed 2 built one with six)."""
    card = replace(CARD, degree_level="bachelor", enrolled_in="none", min_years=1, employment_type="internship",
                   sponsorship="unstated", citizens_only=False)
    base = replace(full_match(card), work_auth="f1_cpt", degree_level="doctoral", graduation_year=2029, years=6)
    assert label(base, card)[0] == "For you" and years_fit(base)
    vs = [v for v in make_variants(card, 0, base=base, base_kind="repaired") if v.kind == "degree_still_enrolled"]
    assert vs and all(years_fit(v.candidate) and v.candidate.years >= card.min_years for v in vs)
    assert all(label(v.candidate, v.card)[0] == v.expected for v in vs)


def test_every_bucket_has_several_edit_kinds():
    rng = random.Random(11)
    by_bucket = {"For you": set(), "Explore": set(), "Excluded": set()}
    for i in range(200):
        for v in make_variants(random_card(rng, i), seed=i):
            by_bucket[v.expected].add(v.kind)
    assert len(by_bucket["Excluded"]) >= 8
    assert len(by_bucket["Explore"]) >= 8
    assert len(by_bucket["For you"]) >= 8


# Fields a variant may change together because they are one fact: a status comes with the
# education it implies, a move clears the relocation list, a degree drop to none clears its year.
ONE_FACT = {
    "degree_level": {"degree_level", "graduation_year", "degree_field", "years"},
    "degree_still_enrolled": {"degree_level", "graduation_year"},
    "degree_enrolled_higher": {"degree_level", "graduation_year"},
    "enrollment_graduated": {"graduation_year", "work_auth"},
    "overqualified": {"years", "graduation_year"},
    "years_one_short": {"years", "graduation_year"},  # years and graduation year move together
    "years_far_short": {"years", "graduation_year"},
    "enrollment_level": {"degree_level", "graduation_year", "years"},
    "location": {"current_location", "willing_locations"},
    "location_willing": {"current_location", "willing_locations"},
    "location_anywhere": {"current_location", "willing_locations"},
    "location_same_metro": {"current_location", "willing_locations"},
    "pref_industry_avoided": {"industries_avoided", "industries_wanted"},
}
STATUS_FIELDS = {"work_auth", "degree_level", "degree_field", "graduation_year", "years"}


@pytest.mark.parametrize("card", [CARD, ANY_OF, PHD_INTERN, GRAD_ONLY])
def test_each_variant_changes_one_thing(card):
    vs = make_variants(card)
    base = vs[0].candidate
    assert vs[0].kind == "full_match" and label(base, card)[0] == "For you"
    for v in vs[1:]:
        diff = {k for k, val in v.candidate.to_dict().items() if val != base.to_dict()[k]}
        cdiff = [k for k, val in v.card.to_dict().items() if val != card.to_dict()[k]]
        allowed = STATUS_FIELDS if v.kind.startswith("work_auth") else ONE_FACT.get(v.kind, set())
        assert len(diff) == 1 or (diff and diff <= allowed), (v.kind, diff)
        assert len(cdiff) <= 2, (v.kind, cdiff)


def test_missing_skill_variant_removes_whole_group():
    for v in make_variants(ANY_OF, seed=1):
        if v.kind == "must_skill":
            held = {s.lower() for s in v.candidate.skills}
            assert any(not held & {x.lower() for x in g} for g in ANY_OF.must_skills)


@pytest.mark.parametrize("card", [CARD, ANY_OF])
def test_skill_swap_keeps_skill_count(card):
    pool = ("western blot", "elisa", "flow cytometry", "microscopy", "hplc")
    vs = make_variants(card, extra_pool=pool)
    base = vs[0].candidate
    for v in vs:
        if v.kind in ("must_skill", "must_skill_alternative", "nice_skill"):
            assert len(v.candidate.skills) == len(base.skills), v.kind


def test_variants_deterministic():
    assert make_variants(CARD, seed=3) == make_variants(CARD, seed=3)


def test_variants_on_a_repaired_base_match_expected():
    from matcher.variants import repair

    rng = random.Random(3)
    for i in range(300):
        card = random_card(rng, i)
        cand = CandidateCard(field="x", degree_level=rng.choice(["none", "bachelor", "master"]),
                             degree_field="Biology", graduation_year=rng.choice([2015, 2022, 2028]),
                             years=rng.randint(0, 9), skills=("welding", "excel"), current_location="Denver, CO",
                             willing_locations=rng.choice([(), ("Austin, TX",)]),
                             remote_preference=rng.choice(["any", "remote_only", "hybrid_or_remote"]),
                             work_auth=rng.choice(["citizen_or_pr", "other_visa"]),
                             salary_floor=rng.choice([None, 150_000]), employment_types_wanted=("contract",),
                             industries_wanted=("finance",), industries_avoided=("healthcare",),
                             roles_avoided=("nurse",))
        if cand.degree_level == "none":
            cand = replace(cand, graduation_year=None, degree_field="")
        fixed = repair(cand, card, random.Random(i))
        if fixed is None:
            continue
        base, expected, _ = fixed
        assert label(base, card)[0] == expected, (card, base)
        assert not rules.status_problems(base)
        if expected != "For you":
            continue
        assert {"welding", "excel"} <= set(base.skills)  # their own skills stay
        for v in make_variants(card, seed=i, base=base, base_kind="repaired"):
            assert label(v.candidate, v.card)[0] == v.expected, (v.kind, card, v.candidate)


# ---- reason wording ------------------------------------------------------

DOUBLED = re.compile(r"\b(\w+)\s+\1\b", re.I)


def test_no_reason_has_a_doubled_word():
    """Every license name from O*NET (most already end in "license" or "certification")
    and multi-city locations with state codes ("Portland, OR or ...")."""
    from matcher.onet import LICENSES

    rng = random.Random(5)
    cards = [replace(random_card(rng, i), license=lic, employment_type="full_time")
             for i, lic in enumerate(sorted(set(LICENSES.values())))]
    cards += [replace(random_card(rng, 100 + i), work_mode="onsite",
                      locations=("Portland, OR", "Columbus, OH", "Boston")[: 1 + i % 3]) for i in range(30)]
    cards += [random_card(rng, 200 + i) for i in range(200)]
    seen = 0
    for card in cards:
        for v in make_variants(card, seed=1):
            for reason in label(v.candidate, v.card)[1]:
                assert not DOUBLED.search(reason), reason
                seen += 1
    assert seen > 2000


@pytest.mark.parametrize("name,text", [
    ("RN license", "RN license"), ("CNA certification", "CNA certification"),
    ("Bar admission", "Bar admission"), ("Series 65", "Series 65 license"), ("CDL Class A", "CDL Class A license"),
])
def test_credential_wording(name, text):
    assert rules.credential(name) == text


def test_join_places():
    assert rules.join_places(("Boston", "Denver")) == "Boston or Denver"
    assert rules.join_places(("Portland, OR", "Columbus, OH")) == "Portland, OR, or Columbus, OH"


# ---- completion formatter ----------------------------------------------

@pytest.mark.parametrize("bucket,word", [("For you", "Keep"), ("Explore", "Explore"), ("Excluded", "Skip")])
def test_completion_starts_with_label_word(bucket, word):
    text = format_completion(bucket, ["meets the degree", "remote matches your preference"])
    assert text.split()[0] == word
    assert "\n" not in text
    assert parse_completion(text) == bucket


def test_completion_shape():
    assert format_completion("For you", ["a", "b."]) == "Keep — a; b."


def test_completion_caps_reasons():
    text = format_completion("Explore", [f"r{i}" for i in range(9)])
    assert text.count(";") == rules.MAX_REASONS - 1


def test_label_words_start_differently():
    assert len({w[0] for w in rules.LABEL_WORDS.values()}) == 3


def test_parse_rejects_unknown_word():
    with pytest.raises(ValueError):
        parse_completion("Maybe - dunno")


def test_end_to_end_completion(base):
    b, reasons, _ = label(replace(base, skills=("pcr",)), CARD)
    text = format_completion(b, reasons)
    assert text.startswith("Skip — missing required skill: cell culture")


# ---- schema -------------------------------------------------------------

def test_roundtrip_json_dicts(base):
    import json

    assert CandidateCard.from_dict(json.loads(json.dumps(base.to_dict()))) == base
    assert RequirementCard.from_dict(json.loads(json.dumps(CARD.to_dict()))) == CARD


def test_invalid_enum_rejected():
    with pytest.raises(ValueError):
        CandidateCard(work_auth="tourist")
    with pytest.raises(ValueError):
        replace(CARD, sponsorship="maybe")


# ---- v3 rule changes ------------------------------------------------------

def test_finishing_the_required_degree_is_a_stretch(base):
    c = replace(base, graduation_year=2027)  # bachelor's student, the card needs a finished bachelor's
    b, reasons, _ = label(c, CARD)
    assert b == "Explore" and reasons[0] == "needs a finished bachelor's; you finish your bachelor's in 2027"
    assert bucket(replace(c, degree_level="master"), CARD) == "For you"  # pursuing a master's, holds the bachelor's
    assert bucket(replace(base, graduation_year=2026), CARD) == "For you"  # 2026 counts as finished


def test_student_role_that_names_its_level_admits_students_at_it():
    card = replace(CARD, degree_level="doctoral", enrolled_in="doctoral", min_years=0, employment_type="internship",
                   degree_fields=())
    c = replace(full_match(card), degree_level="doctoral", graduation_year=2029)
    assert bucket(c, card) == "For you"


def test_degree_field_checked_on_enrolled_degree():
    card = replace(CARD, degree_level="none", enrolled_in="bachelor", degree_fields=("Electrical Engineering",
                   "Computer Engineering"), min_years=0, employment_type="internship")
    c = replace(full_match(card), degree_level="bachelor", graduation_year=2027)
    assert bucket(replace(c, degree_field="Biology"), card) == "Excluded"
    assert bucket(replace(c, degree_field="Physics"), card) == "Explore"  # same family: a stretch
    assert bucket(replace(c, degree_field="EE"), card) == "For you"  # alias


def test_related_field_wording_accepts_the_family_and_vague_only_accepts_all(base):
    card = replace(CARD, degree_fields=("Computer Science", "related field"))
    assert bucket(replace(base, degree_field="Computer Engineering"), card) == "For you"
    assert bucket(replace(base, degree_field="Biology"), card) == "Excluded"
    vague = replace(CARD, degree_fields=("related field", "equivalent"))
    assert bucket(replace(base, degree_field="Theology"), vague) == "For you"


def test_soft_skills_never_count(base):
    card = replace(CARD, must_skills=(*CARD.must_skills, ("Communication",), ("Problem-solving",)),
                   nice_skills=("Teamwork", "python"))
    b, reasons, _ = label(base, card)
    # no "2 of 2" against a posting that lists four required skills
    assert b == "For you" and "all required skills met" in reasons
    assert not any(re.search(r"\d of \d", x) for x in reasons)
    assert rules.required_groups(card) == CARD.must_skills


def test_problem_solving_spellings_unify():
    from matcher.schema import skill_canon

    assert skill_canon("Problem-solving") == skill_canon("Problem Solving") == skill_canon("problem solving")
    assert skill_canon("MS Excel") == skill_canon("Microsoft Excel")
    assert skill_canon("ML") == skill_canon("Machine learning")
    # O*NET's own names are for onet.display_name only: no resume writes them
    assert skill_canon("Medicine and Dentistry") != skill_canon("Clinical medicine")
    from matcher.onet import display_name

    assert display_name("Structured query language SQL") == "SQL"
    assert display_name("Medicine and Dentistry") == "Clinical medicine"


def test_skill_synonyms_meet_a_requirement(base):
    card = replace(CARD, must_skills=(("SQL",), ("Microsoft Excel",)), nice_skills=())
    c = replace(base, skills=("SQL programming", "Excel"))
    assert bucket(c, card) == "For you"


@pytest.mark.parametrize("n_nice,missing,want", [
    (1, 1, "For you"), (2, 1, "For you"), (2, 2, "Explore"), (3, 1, "For you"), (3, 2, "Explore"),
    (6, 2, "For you"), (6, 3, "Explore"), (10, 3, "Explore"),
])
def test_nice_to_have_soft_only_when_many_missing(base, n_nice, missing, want):
    nice = tuple(f"tool {i}" for i in range(n_nice))
    card = replace(CARD, nice_skills=nice)
    c = replace(base, skills=(*base.skills, *nice[missing:]))
    b, reasons, _ = label(c, card)
    assert b == want
    if want == "Explore":
        assert reasons[0] == "missing nice-to-have skill: " + ", ".join(nice[:missing])


def test_industry_outside_the_list_is_checked_by_its_own_name(base):
    card = replace(CARD, industry="Aerospace & Defense")
    b, reasons, _ = label(replace(base, industries_wanted=("healthcare",)), card)
    assert b == "Explore" and reasons[0] == "Aerospace & Defense is not one you listed as wanted"
    assert bucket(replace(base, industries_wanted=("aerospace & defense", "healthcare")), card) == "For you"
    assert bucket(replace(base, industries_avoided=("aerospace & defense",)), card) == "Explore"
    assert bucket(base, card) == "For you"  # no wanted list: any industry
    assert bucket(replace(base, industries_wanted=("healthcare",)), replace(CARD, industry="")) == "For you"
    syn = replace(CARD, industry="Financial Services")
    assert bucket(replace(base, industries_wanted=("finance",)), syn) == "For you"
    assert bucket(replace(base, industries_wanted=("healthcare",)), syn) == "Explore"
    assert bucket(replace(base, industries_avoided=("finance",)), syn) == "Explore"


def test_overqualified_on_full_roles_and_for_non_students_on_internships(base):
    card = replace(CARD, min_years=0, seniority="entry")
    assert bucket(replace(base, years=6), card) == "Explore"
    assert bucket(replace(base, years=5), card) == "For you"
    assert bucket(replace(base, years=12), replace(card, seniority="senior")) == "For you"
    _, reasons, _ = label(replace(base, years=8), card)
    assert "you may be overqualified (8 years; no minimum asked)" in reasons
    intern = replace(card, employment_type="internship")
    assert bucket(replace(base, years=3), intern) == "For you"
    b, reasons, _ = label(replace(base, years=12), intern)
    assert b == "Explore" and reasons[0] == "you may be overqualified for an internship (12 years; no minimum asked)"
    assert bucket(replace(base, years=4), replace(intern, min_years=1)) == "For you"
    assert bucket(replace(base, years=5), replace(intern, min_years=1)) == "Explore"
    # students with years of work are normal on internships and student roles
    student = replace(base, degree_level="master", graduation_year=2027, years=6)
    assert bucket(student, intern) == "For you"
    assert bucket(student, replace(intern, enrolled_in="bachelor")) == "For you"


def test_one_year_minimum_is_not_soft(base):
    card = replace(CARD, min_years=1)
    assert bucket(replace(base, years=0), card) == "Excluded"


def test_reason_wording_fixes(base):
    _, reasons, _ = label(replace(base, employment_types_wanted=("full_time",)), replace(CARD, employment_type="internship"))
    assert reasons[0] == "it is an internship role and you want full-time"
    _, reasons, _ = label(replace(base, years=1), CARD)
    assert reasons == ["1 year short of the 2-year minimum"]
    _, reasons, _ = label(replace(base, salary_floor=90_000), CARD)
    assert reasons[-1] == "otherwise meets the requirements"


def test_avoided_role_is_a_whole_word(base):
    card = replace(CARD, title="General Internal Medicine Physician")
    assert bucket(replace(base, roles_avoided=("intern",)), card) == "For you"
    assert bucket(replace(base, roles_avoided=("intern",)), replace(CARD, title="Lab Intern")) == "Explore"


@pytest.mark.parametrize("word,title", [
    ("technician", "Medical and Clinical Laboratory Technologists and Technicians"),
    ("operator", "Cutting and Slicing Machine Setters, Operators, and Tenders"),
    ("analyst", "Fraud Examiners, Investigators and Analysts"),
    ("teacher", "Secondary School Teachers, Except Special and Career/Technical Education"),
    ("secretary", "Executive Secretaries and Executive Administrative Assistants"),  # -y -> -ies (no ROLE_W word ends in -y; the rule still takes it)
    ("coach", "Coaches and Scouts"),  # -es
])
def test_avoided_role_matches_the_plural_title(base, word, title):
    card = replace(CARD, title=title)
    assert rules.role_hits(replace(base, roles_avoided=(word,)), card) == [word]
    _, reasons, _ = label(replace(base, roles_avoided=(word,)), card)
    assert f"title includes {word}, a role you said you avoid" in reasons


@pytest.mark.parametrize("word,title", [
    ("intern", "General Internal Medicine Physician"),
    ("intern", "International Trade Analyst"),
    ("intern", "Software Engineering Internship"),
    ("aide", "Computer-Aided Design Drafter"),
    ("sales", "Aftersales Service Advisor"),
])
def test_avoided_role_never_matches_a_prefix(base, word, title):
    assert rules.role_hits(replace(base, roles_avoided=(word,)), replace(CARD, title=title)) == []


@pytest.mark.parametrize("card_loc,cand_loc,want", [
    ("Boston, MA", "Newton, MA", "For you"),
    ("Cambridge, MA", "Waltham, MA", "For you"),
    ("Palo Alto, CA", "San Jose, CA", "For you"),
    ("South San Francisco, CA", "San Francisco, CA", "For you"),
    ("California", "San Diego, CA", "For you"),
    ("Massachusetts", "Denver, CO", "Excluded"),
    ("Boston, MA", "Providence, RI", "Excluded"),
    ("Toronto, ON", "Toronto", "For you"),
    ("Boston, MA", "Seattle, WA", "Excluded"),
])
def test_location_compares_metros(base, card_loc, cand_loc, want):
    card = replace(CARD, locations=(card_loc,))
    assert bucket(replace(base, current_location=cand_loc, willing_locations=()), card) == want


def test_anywhere_in_the_us(base):
    c = replace(base, current_location="Denver, CO", willing_locations=("anywhere in the US",))
    assert bucket(c, replace(CARD, locations=("Boston, MA",))) == "For you"
    assert bucket(c, replace(CARD, locations=("London, UK",))) == "Excluded"


@pytest.mark.parametrize("changes,problem", [
    (dict(work_auth="f1_cpt", graduation_year=2024), "CPT needs current enrollment"),
    (dict(work_auth="f1_opt", graduation_year=2028), "OPT is the year after graduating"),
    (dict(work_auth="stem_opt", graduation_year=2024, degree_field="Graphic Design"), "STEM OPT needs a STEM degree field"),
    (dict(work_auth="needs_h1b_now", graduation_year=2028), "needs H-1B now but still enrolled"),
    (dict(degree_level="none", graduation_year=2020, degree_field=""), "no degree but a graduation year or field"),
    (dict(work_auth="stem_opt", graduation_year=2024, years=0), "STEM OPT follows a year of OPT work"),
    (dict(work_auth="f1_cpt", graduation_year=2030), "CPT comes after a year of study"),
])
def test_status_problems(changes, problem):
    c = CandidateCard(degree_level="bachelor", degree_field="Biology", graduation_year=2025, years=1)
    assert rules.status_problems(c) == []
    assert problem in rules.status_problems(replace(c, **changes))
    assert rules.status_problems(replace(c, work_auth="stem_opt", graduation_year=2024)) == []
    assert rules.status_problems(replace(c, work_auth="f1_cpt", graduation_year=2029)) == []
    assert rules.status_problems(replace(c, work_auth="f1_cpt", degree_level="master", graduation_year=2028)) == []


def test_no_stem_opt_swap_onto_a_card_without_stem_fields():
    """STEM OPT work relates to the STEM degree: a Marketing graduate is not turned into a
    Computer Science graduate on STEM OPT to fill a Concierge posting."""
    from matcher.variants import make_variants, with_status

    concierge = replace(CARD, title="Concierge", field="Hospitality and personal care", industry="hospitality",
                        degree_level="bachelor", degree_fields=(), must_skills=(("Customer service",),),
                        nice_skills=(), sponsorship="sponsors")
    grad = CandidateCard(degree_level="bachelor", degree_field="Marketing", graduation_year=2022, years=3,
                         skills=("Customer service",), current_location="Boston, MA")
    for seed in range(20):
        assert with_status(grad, "stem_opt", concierge, random.Random(seed)) is None
        assert all(v.candidate.work_auth != "stem_opt" for v in make_variants(concierge, seed))
    lab = replace(CARD, sponsorship="sponsors")
    got = with_status(replace(grad, degree_field="Biology"), "stem_opt", lab, random.Random(0))
    assert got is not None and got.degree_field == "Biology"


def test_location_reason_names_the_place_that_matches(base):
    card = replace(CARD, locations=("Chicago, IL", "Overland Park, KS"))
    c = replace(base, current_location="Kansas City, KS", willing_locations=())
    b, reasons, _ = label(c, card)
    assert b == "For you" and "location works (Overland Park, KS)" in reasons
    mover = replace(base, current_location="Denver, CO", willing_locations=("Boston, MA", "Chicago, IL"))
    assert "location works (Chicago, IL)" in label(mover, card)[1]
    home_first = replace(base, current_location="Evanston, IL", willing_locations=("Overland Park, KS",))
    assert "location works (Chicago, IL)" in label(home_first, card)[1]


@pytest.mark.parametrize("card_loc,cand_loc,want", [
    ("Washington State", "Seattle, WA", "For you"),
    ("Washington State", "Washington, DC", "Excluded"),
    ("Washington State", "Silver Spring, MD", "Excluded"),
    ("New York State", "Brooklyn, NY", "For you"),
    ("New York State", "Jersey City, NJ", "Excluded"),
    ("New York, NY", "Jersey City, NJ", "For you"),
])
def test_state_only_places(base, card_loc, cand_loc, want):
    card = replace(CARD, locations=(card_loc,))
    assert bucket(replace(base, current_location=cand_loc, willing_locations=()), card) == want


def test_home_for_a_state_only_posting_is_a_city():
    from matcher.metros import parse_place
    from matcher.variants import _home

    for loc in ("Washington State", "New York State", "Wyoming", "United States", "Bay Area"):
        card = replace(CARD, locations=(loc,))
        for seed in range(20):
            home = _home(card, random.Random(seed))
            assert parse_place(home).kind == "city", (loc, home)
        assert parse_place(_home(card, None)).kind == "city"


def test_generic_skills_are_met_by_concrete_ones(base):
    base = replace(base, years=1)  # a junior: four years on an internship is overqualified
    swe = replace(CARD, title="Software Engineer Intern", must_skills=(("Programming",),), nice_skills=(),
                  degree_fields=(), employment_type="internship", min_years=0)
    coder = replace(base, skills=("Python", "Git"))
    assert bucket(coder, swe) == "For you"
    lab = replace(base, skills=("PCR", "Cell culture", "Western blot"))
    b, reasons, _ = label(lab, swe)
    assert b == "Excluded" and reasons[0] == "missing required skill: Programming"
    assert bucket(replace(base, skills=("SQL",)), swe) == "Excluded"  # a query language is not programming
    for need, have in (("Coding", "C++"), ("Debugging", "Rust"), ("Mathematics", "Linear algebra"),
                       ("Math", "Calculus"), ("Research", "UX research"), ("Research skills", "Laboratory research")):
        assert bucket(replace(base, skills=(have,)), replace(swe, must_skills=((need,),))) == "For you", need


def test_soft_list_covers_what_postings_add_and_drops_concrete_skills():
    from matcher.schema import is_soft_skill

    for x in ("Positive Attitude", "Troubleshooting", "Independent work", "Team collaboration",
              "Technical communication", "Analytical ability", "English fluency", "Communication"):
        assert is_soft_skill(x), x
    for x in ("Programming", "Coding", "Debugging", "Research", "Mathematics", "Math", "Python"):
        assert not is_soft_skill(x), x


def test_variants_on_a_programming_card_match_the_rule():
    card = replace(CARD, title="Software Engineer Intern", must_skills=(("Programming",), ("Git",)),
                   nice_skills=("Python", "Docker", "Linux"), degree_fields=(), employment_type="internship",
                   min_years=0, seniority="entry")
    pool = ("Python", "Java", "Agile", "Jira", "Unit testing", "Figma")
    for seed in range(10):
        for v in make_variants(card, seed, extra_pool=pool):
            assert bucket(v.candidate, v.card) == v.expected, (v.kind, v.candidate.skills)
            if v.kind == "must_skill" and "Programming" in v.note:
                assert not any(s in ("Python", "Java") for s in v.candidate.skills)


def test_synonym_variant_writes_real_wordings_only():
    from matcher.schema import ONET_NAMES, skill_key

    onet = {skill_key(a) for alts in ONET_NAMES.values() for a in alts}
    card = replace(CARD, must_skills=(("Clinical medicine",), ("Microsoft Excel",), ("Customer service",)),
                   nice_skills=(), degree_fields=())
    seen = set()
    for seed in range(40):
        for v in make_variants(card, seed):
            if v.kind == "must_skill_synonym":
                seen.add(v.note)
                assert not any(skill_key(s) in onet for s in v.candidate.skills), v.note
    assert seen and all("Clinical medicine" not in n for n in seen)


@pytest.mark.parametrize("field,posting,want", [
    ("Geology", ("Biochemistry", "Biology", "Neuroscience", "related field"), "none"),
    ("Political Science", ("Biology", "Psychology", "related field"), "none"),
    ("Film", ("Elementary Education", "English", "related field"), "none"),
    ("Graphic Design", ("Business Administration", "Economics", "Management", "Marketing", "related field"), "none"),
    ("Machine Learning", ("Accounting", "Economics", "Finance", "Management", "related field"), "none"),
    ("Fine Arts", ("Communications", "Management", "related field"), "none"),
    ("English", ("Computer Science", "Data Science", "Statistics", "Mathematics"), "none"),
    ("Hospitality Management", ("Anthropology", "Economics", "Psychology", "Statistics"), "none"),
    ("Pharmacy", ("Social Work", "Sociology"), "none"),
    ("Biology", ("Nursing", "Health Sciences", "equivalent"), "adjacent"),
    ("Computer Engineering", ("Computer Science", "related field"), "exact"),
    ("Biochemistry", ("Biology", "related field"), "exact"),
    ("Chemistry", ("Biology", "related field"), "adjacent"),  # a neighbour, not a related field
    ("Biochemistry", ("Biology", "equivalent"), "adjacent"),  # "or equivalent" widens nothing
    ("Theology", ("related field", "equivalent"), "exact"),  # only vague wording: any field
])
def test_related_fields_are_tight_and_equivalent_widens_nothing(field, posting, want):
    from matcher.degrees import field_match

    assert field_match(field, posting) == want


def test_years_and_graduation_year_agree_in_every_variant():
    from matcher.variants import years_fit

    rng = random.Random(11)
    for i in range(300):
        card = random_card(rng, i)
        for v in make_variants(card, seed=i):
            assert years_fit(v.candidate), (v.kind, v.candidate.years, v.candidate.graduation_year,
                                            v.candidate.work_auth)


def test_reason_polish(base):
    card = replace(CARD, min_years=1)
    assert "1 year of experience covers the 1-year minimum" in label(replace(base, years=1), card)[1]
    _, reasons, _ = label(replace(base, salary_floor=90_000), CARD)
    assert reasons[0] == "pay tops out at $70,000, under your $90,000 floor"
    assert "meets the bachelor's degree requirement" in label(base, CARD)[1]
    _, reasons, _ = label(replace(base, degree_level="associate"), CARD)
    assert reasons[0] == "needs a bachelor's degree or higher; you have an associate degree"
    role = replace(CARD, degree_level="none", enrolled_in="bachelor", degree_fields=(), min_years=0,
                   employment_type="internship")
    assoc = replace(base, degree_level="associate", graduation_year=2027)
    assert label(assoc, role)[1][0] == ("for students currently pursuing a bachelor's degree or higher; "
                                       "you are pursuing an associate degree")
    worker = replace(base, degree_level="none", graduation_year=None, degree_field="")
    assert label(worker, role)[1][0].endswith("; you are not a student")


def test_doctoral_student_is_not_credited_with_a_masters(base):
    card = replace(CARD, degree_level="master", degree_fields=())
    phd = replace(base, degree_level="doctoral", graduation_year=2029)
    b, reasons, _ = label(phd, card)
    assert b == "Explore" and reasons[0] == "needs a finished master's; you are pursuing a doctorate, expected 2029"
    assert bucket(replace(phd, graduation_year=2024), card) == "For you"  # a finished doctorate holds it
    assert bucket(replace(base, degree_level="master", graduation_year=2028), replace(CARD, degree_fields=())) == \
        "For you"  # a master's student holds the bachelor's


def test_bachelors_student_is_not_credited_with_an_associate_degree(base):
    card = replace(CARD, degree_level="associate", degree_fields=(), min_years=0)
    student = replace(base, degree_level="bachelor", graduation_year=2028, years=0)
    assert rules.held_level(student) == "none"
    b, reasons, _ = label(student, card)
    assert b == "Explore" and reasons[0] == ("needs a finished associate degree; you are pursuing a bachelor's, "
                                             "expected 2028")
    assert bucket(replace(student, graduation_year=2025), card) == "For you"  # a finished bachelor's holds it
    for seed in range(10):
        vs = make_variants(card, seed)
        assert "degree_enrolled_higher" not in {v.kind for v in vs}
        for v in vs:
            assert bucket(v.candidate, v.card) == v.expected, (v.kind, v.candidate)
        for v in make_variants(replace(card, employment_type="internship"), seed):
            assert bucket(v.candidate, v.card) == v.expected, (v.kind, v.candidate)


def test_degree_field_shows_its_own_skill(base):
    chem = replace(CARD, must_skills=(("Chemistry",),), nice_skills=("Biology", "Statistics", "Physics"),
                   degree_fields=())
    grad = replace(base, degree_field="Chemistry", skills=("PCR",))
    b, reasons, _ = label(grad, chem)
    assert b == "Explore" and reasons[0] == "missing nice-to-have skill: Biology, Statistics, Physics"
    b, reasons, _ = label(replace(grad, degree_field="History"), chem)
    assert b == "Excluded" and reasons == ["missing required skill: Chemistry"]
    # Biochemistry is Biology and Chemistry coursework; a field outside FIELD_SKILLS shows only itself
    assert rules.nice_gap(replace(grad, degree_field="Biochem"), chem) == ["Statistics", "Physics"]
    assert rules.nice_gap(replace(grad, degree_field="Data Science"), chem) == ["Biology", "Physics"]
    assert rules.nice_gap(replace(grad, degree_field="Medicine"), chem) == ["Statistics", "Physics"]
    # no degree, no degree skills
    assert bucket(replace(grad, degree_level="none", degree_field="", graduation_year=None),
                  replace(chem, degree_level="none")) == "Excluded"


def test_concrete_skills_show_sql_oop_and_machine_learning(base):
    card = replace(CARD, must_skills=(("SQL",), ("Object-oriented programming",), ("Machine learning",)),
                   nice_skills=(), degree_fields=())
    c = replace(base, skills=("MySQL", "Java", "PyTorch"))
    assert bucket(c, card) == "For you"
    assert bucket(replace(c, skills=("Microsoft SQL Server", "C#", "Deep learning")), card) == "For you"
    assert bucket(replace(c, skills=("Excel", "Java", "PyTorch")), card) == "Excluded"
    assert bucket(replace(c, skills=("MySQL", "C", "PyTorch")), card) == "Excluded"  # C is not object-oriented
    assert bucket(replace(c, skills=("MySQL", "Java", "Statistics")), card) == "Excluded"


def test_skill_breaks_skip_skills_the_degree_shows():
    card = replace(CARD, must_skills=(("Chemistry",), ("PCR",)), nice_skills=("Biology", "Physics", "Statistics"),
                   degree_fields=("Biochemistry",))
    for seed in range(20):
        for v in make_variants(card, seed):
            assert bucket(v.candidate, v.card) == v.expected, (v.kind, v.candidate)
            if v.kind == "must_skill":
                assert v.note == "no PCR", v.note


def test_alternative_swap_keeps_skills_other_requirements_lean_on():
    card = replace(CARD, must_skills=(("Deep learning", "REST APIs"), ("Machine learning",)),
                   nice_skills=("Machine learning", "Statistics"), degree_fields=())
    for seed in range(30):
        for v in make_variants(card, seed):
            assert bucket(v.candidate, v.card) == v.expected, (v.kind, v.candidate.skills)


def test_degree_field_reason_quotes_related_field_wording(base):
    card = replace(CARD, degree_fields=("Marketing", "Accounting", "related field"))
    b, reasons, _ = label(replace(base, degree_field="Economics"), card)
    assert b == "Explore"
    assert reasons[0] == "degree field should be Marketing or Accounting (or a related field); yours, Economics, is close"
    b, reasons, _ = label(replace(base, degree_field="History"), card)
    assert b == "Excluded" and reasons == ["degree field should be Marketing or Accounting (or a related field)"]
    b, reasons, _ = label(replace(base, degree_field="History"), replace(card, degree_fields=("Marketing", "equivalent")))
    assert reasons == ["degree field should be Marketing"]


def test_entry_certificates_stop_at_a_bachelors():
    from matcher.onet import license_fits

    for lic in ("Pharmacy technician certification", "CNA certification", "LPN license", "EMT certification",
                "Security guard license", "Phlebotomy certification"):
        assert license_fits(lic, "none", "")
        assert license_fits(lic, "bachelor", "Biology"), lic  # pre-pharmacy and pre-med students hold these
        assert not license_fits(lic, "master", "Biology"), lic
        assert not license_fits(lic, "doctoral", "Pharmacy"), lic
    assert license_fits("Veterinary technician credential", "bachelor", "Veterinary Technology")
    assert not license_fits("Veterinary technician credential", "master", "Veterinary Technology")
    # a doctoral student is credited with a bachelor's, so a PhD student may still hold one
    phd = CandidateCard(degree_level="doctoral", degree_field="Biology", graduation_year=2029,
                        licenses=("Pharmacy technician certification",))
    assert rules.license_problems(phd) == []
    assert rules.license_problems(replace(phd, graduation_year=2020)) != []


def test_f1_cpt_lab_student():
    """An F-1 biology undergraduate on CPT with lab skills."""
    me = CandidateCard(degree_level="bachelor", degree_field="Biology", graduation_year=2027, years=1,
                       skills=("PCR", "Chemistry", "Cell culture", "Western blot", "Microscopy", "Pipetting"),
                       current_location="Boston, MA", work_auth="f1_cpt", employment_types_wanted=("internship",))
    intern = RequirementCard(source_posting_id="x", title="Research Intern", employment_type="internship",
                             seniority="entry", degree_fields=("Biology", "Chemistry", "related field"),
                             enrolled_in="bachelor", must_skills=(("PCR",), ("Chemistry",)),
                             nice_skills=("Python", "Biology"), locations=("Cambridge, MA",), sponsorship="unstated")
    b, reasons, _ = label(me, intern)
    assert b == "For you" and "CPT covers this internship" in reasons
    coder = replace(intern, title="Software Engineer Intern", must_skills=(("Programming",),), nice_skills=(),
                    degree_fields=())
    b, reasons, _ = label(me, coder)
    assert b == "Excluded" and reasons == ["missing required skill: Programming"]
    full = replace(intern, employment_type="full_time", enrolled_in="none", degree_level="bachelor")
    assert label(me, full)[0] == "Excluded"  # wants internships only
    b, reasons, _ = label(replace(me, employment_types_wanted=()), full)
    assert b == "Explore" and "CPT only covers internships and co-ops" in reasons
    assert label(me, replace(intern, citizens_only=True))[0] == "Excluded"


@pytest.mark.parametrize("posting,wanted,want", [
    ("Healthcare Technology", ("technology",), "For you"),
    ("Healthcare Technology", ("healthcare",), "For you"),
    ("Healthcare Technology", ("retail",), "Explore"),
    ("AI Robotics", ("technology",), "For you"),
    ("Defense & Aerospace", ("aerospace & defense",), "For you"),
    ("Defense & Aerospace", ("technology",), "Explore"),
    ("Health Insurance", ("finance",), "For you"),
    ("Religious Organizations", ("nonprofit",), "For you"),
    ("Infrastructure", ("construction", "utilities"), "Explore"),  # no word names a list item
    ("Infrastructure", ("infrastructure",), "For you"),
    ("Law Enforcement", ("legal",), "Explore"),
    ("Security and Safety", ("technology",), "Explore"),
])
def test_free_text_industries_are_read_by_their_words(base, posting, wanted, want):
    card = replace(CARD, industry=posting)
    assert bucket(replace(base, industries_wanted=wanted), card) == want


def test_free_text_industry_avoided_by_its_words(base):
    card = replace(CARD, industry="Healthcare Technology")
    b, reasons, _ = label(replace(base, industries_avoided=("healthcare",)), card)
    assert b == "Explore" and reasons[0] == "Healthcare Technology is an industry you said you avoid"
    for seed in range(10):
        for v in make_variants(card, seed):
            assert bucket(v.candidate, v.card) == v.expected, (v.kind, v.candidate.industries_wanted)


def test_degree_fields_show_their_coursework(base):
    """Skills every program in a field teaches count as held (degrees.FIELD_SKILLS)."""
    def gap(field, *skills):
        card = replace(CARD, must_skills=tuple((x,) for x in skills), nice_skills=(), degree_fields=())
        return [x for x in label(replace(base, degree_field=field, skills=("PCR",)), card)[1] if "skill" in x]

    assert gap("Software Engineering", "Algorithms", "Data structures", "Engineering principles") == \
        ["all required skills met"]
    assert gap("CS", "Computer systems", "Algorithms") == ["all required skills met"]
    assert gap("Engineering Technology", "Engineering principles", "Mechanical systems") == ["all required skills met"]
    assert gap("Electrical Engineering", "Engineering principles") == ["all required skills met"]
    assert gap("Social Work", "Counseling") == ["all required skills met"]
    assert gap("Management", "Human resources", "Business management") == ["all required skills met"]
    assert gap("Marketing", "Sales and marketing") == ["all required skills met"]
    assert gap("Graphic Design", "Fine arts", "Design") == ["all required skills met"]
    assert gap("Nursing", "Clinical medicine") == ["all required skills met"]
    assert gap("Statistics", "Data analysis") == ["all required skills met"]
    assert gap("Data Science", "Data analysis", "Computer systems", "Statistics") == ["all required skills met"]
    assert gap("Finance", "Financial modeling", "Accounting") == ["all required skills met"]
    # only what every program teaches: an Economics degree is not a management degree
    assert gap("Economics", "Business management") == ["missing required skill: Business management"]
    assert gap("Nursing", "Biology") == ["missing required skill: Biology"]


def test_office_apps_show_microsoft_office_and_cad_shows_design(base):
    card = replace(CARD, must_skills=(("Microsoft Office",),), nice_skills=("Design", "AutoCAD", "Tableau"),
                   degree_fields=())
    c = replace(base, degree_field="History", skills=("Microsoft Excel", "AutoCAD Civil 3D"))
    assert label(c, card)[0] == "For you"  # Excel shows the suite; Civil 3D shows AutoCAD and Design
    b, reasons, _ = label(replace(c, skills=("Google Sheets", "PCB design")), card)
    assert b == "Excluded" and reasons == ["missing required skill: Microsoft Office"]
    for seed in range(10):
        for v in make_variants(card, seed):
            assert bucket(v.candidate, v.card) == v.expected, (v.kind, v.candidate.skills)


@pytest.mark.parametrize("required,held,want", [
    # a suite on the resume shows its core apps, and each app shows the suite
    ("Microsoft Excel", "Microsoft Office", True), ("Microsoft Word", "MS Office", True),
    ("Microsoft PowerPoint", "Office 365", True), ("Microsoft Outlook", "Microsoft Office", True),
    ("Microsoft Office", "Excel", True), ("Microsoft Access", "Microsoft Office", False),
    ("Microsoft Word", "Microsoft Excel", False),  # not transitive
    ("Photoshop", "Adobe Creative Cloud", True), ("InDesign", "Adobe Creative Suite", True),
    ("Adobe Creative Cloud", "Adobe Illustrator", True), ("Illustrator", "Photoshop", False),
    ("CAD", "AutoCAD", True), ("CAD", "SolidWorks", True), ("CAD", "Revit", True), ("AutoCAD", "CAD", False),
    ("Database software", "MySQL", True), ("Database software", "Microsoft Access", True),
    ("Database software", "SQL", True), ("SQL Server", "Database software", False),
    ("Learning management system LMS", "Blackboard Learn", True), ("Course management system", "Schoology", True),
    ("Learning management system LMS", "Course management system", True), ("Blackboard Learn", "Schoology", False),
    ("Accounting", "QuickBooks", True), ("Accounting", "Sage 50 Accounting", True),
])
def test_generic_and_specific_skills_show_each_other(required, held, want):
    assert skill_meets(required, held) is want


@pytest.mark.parametrize("must,nice", [
    (("Microsoft Excel",), ("Microsoft Word", "Microsoft Outlook", "Tableau")),
    (("CAD",), ("Photoshop", "InDesign", "Tableau")),
    (("Database software",), ("Learning management system LMS", "Accounting", "Tableau")),
])
def test_suite_on_the_resume_meets_its_apps(base, must, nice):
    """A candidate who lists the suite (or the CAD program) holds what it shows; every variant built
    for such a card still gets the bucket it was built for (a must-skill break drops the suite too)."""
    card = replace(CARD, must_skills=(must,), nice_skills=nice, degree_fields=())
    c = replace(base, degree_field="History",
                skills=("Microsoft Office", "AutoCAD", "Adobe Creative Cloud", "MySQL", "Blackboard Learn",
                        "QuickBooks"))
    b, reasons, _ = label(c, card)
    assert b == "For you" and not any("missing" in x for x in reasons)
    for seed in range(10):
        for v in make_variants(card, seed, base=c, base_kind="repaired", extra_pool=("Python", "Statistics")):
            assert bucket(v.candidate, v.card) == v.expected, (v.kind, v.candidate.skills)
        for v in make_variants(card, seed):
            assert bucket(v.candidate, v.card) == v.expected, (v.kind, v.candidate.skills)


@pytest.mark.parametrize("field,posting,want", [
    # science, technology, engineering and math named together mean STEM
    ("Computer Science", ("Science", "Technology", "Engineering", "Math", "Physics"), "exact"),
    ("Biochemistry", ("Technology", "Engineering", "Mathematics"), "exact"),
    ("Finance", ("Science", "Technology", "Engineering", "Math"), "none"),
    ("Chemistry", ("Engineering", "Mathematics"), "none"),  # two of them name only themselves
    ("CS", ("Technology",), "exact"),  # a bare Technology is the computing group
    ("Psychology", ("Technology",), "none"),
    # a posting's Business is the business group; a candidate's Business degree is Business Administration
    ("Finance", ("Business",), "exact"),
    ("Accounting", ("Business", "Business Management"), "exact"),
    ("Business", ("Business Administration",), "exact"),
    # joint entries split into fields the tables know
    ("Electrical Engineering", ("Related Engineering/Technical Field",), "exact"),
    ("Statistics", ("Math/Statistics",), "exact"),
    ("Machine Learning", ("ML/AI",), "exact"),
    ("Mechanical Engineering", ("Electrical or Mechanical Engineering",), "exact"),
    ("Data Science", ("Data/Analytics",), "exact"),
    ("Computer Science", ("Mathematics", "Engineering", "UI/UX", "Technical backgrounds"), "exact"),
    ("Strength and Conditioning", ("Strength and Conditioning",), "exact"),  # unknown names stay whole
    # real-posting names
    ("Data Science", ("Business Analytics",), "exact"),
    ("Computer Science", ("Analytics", "Financial Engineering", "Quantitative Disciplines"), "exact"),
    ("Mechanical Engineering", ("Robotics",), "adjacent"),
    ("Mechanical Engineering", ("Robotics", "related field"), "exact"),
    ("Computer Science", ("Computer Vision",), "adjacent"),
    ("Electrical Engineering", ("Electronics Engineering",), "exact"),
    ("Information Systems", ("Information Science",), "exact"),
])
def test_posting_degree_wording(field, posting, want):
    from matcher.degrees import field_match
    assert field_match(field, posting) == want


def test_joint_entry_reason_quotes_the_posting(base):
    card = replace(CARD, degree_fields=("Math/Statistics",))
    b, reasons, _ = label(replace(base, degree_field="History"), card)
    assert b == "Excluded" and reasons == ["degree field should be Math/Statistics"]


def test_no_variant_repeats_its_base_or_another_variant():
    from matcher.variants import years_fit
    cards = [CARD, replace(CARD, work_mode="remote", locations=()), replace(CARD, employment_type="internship",
                                                                          sponsorship="will_not_sponsor_or_opt")]
    for card in cards:
        for seed in range(15):
            vs = make_variants(card, seed)
            seen = [(v.candidate, v.card) for v in vs]
            assert len(seen) == len(set(seen)) or all(v.kind in ("work_auth_cpt_internship",
                                                                 "work_auth_cpt_internship_no_opt") for v in vs
                                                      if seen.count((v.candidate, v.card)) > 1), card.title
            base = vs[0]
            assert all((v.candidate, v.card) != (base.candidate, base.card) for v in vs[1:])
            for v in vs:
                assert years_fit(v.candidate), (v.kind, v.candidate)


def test_opt_bachelors_years_follow_graduation():
    from matcher.variants import years_fit
    opt = CandidateCard(degree_level="bachelor", degree_field="Biology", graduation_year=2026, years=2,
                        work_auth="f1_opt")
    assert not years_fit(opt) and years_fit(replace(opt, years=1))
    assert years_fit(replace(opt, degree_level="master", years=4))  # master's students often worked first
    assert years_fit(replace(opt, work_auth="stem_opt", graduation_year=2024, years=3))
    assert not years_fit(replace(opt, work_auth="stem_opt", graduation_year=2024, years=4))


def test_an_emr_product_shows_emr(base):
    card = replace(CARD, must_skills=(("Electronic medical record EMR",),), nice_skills=(), degree_fields=())
    assert bucket(replace(base, skills=("Patient electronic medical record EMR",)), card) == "For you"
    assert bucket(replace(base, skills=("GE Healthcare Centricity EMR",)), card) == "For you"
    assert bucket(replace(base, skills=("Epic",)), card) == "For you"
    assert bucket(replace(base, skills=("Microsoft Excel",)), card) == "Excluded"


def test_degree_break_on_an_opt_masters_keeps_years_that_fit():
    from matcher.variants import years_fit
    card = replace(CARD, degree_level="master", degree_fields=(), min_years=2, sponsorship="sponsors")
    base = replace(full_match(card), degree_field="Chemistry", graduation_year=2024, years=5, work_auth="stem_opt")
    assert label(base, card)[0] == "For you"
    for seed in range(5):
        for v in make_variants(card, seed, base=base, base_kind="repaired"):
            assert years_fit(v.candidate), (v.kind, v.candidate)
            assert bucket(v.candidate, v.card) == v.expected, (v.kind, v.candidate)


def test_related_ok_softens_a_field_mismatch_like_a_related_field_entry():
    from dataclasses import replace as _replace

    from matcher.degrees import card_fields
    from matcher.rules import label as _label
    from matcher.variants import full_match as _full

    strict = RequirementCard(source_posting_id="r1", title="Lab Analyst", degree_level="bachelor",
                             degree_fields=("Biology",), employment_type="full_time")
    loose = _replace(strict, related_ok=True)
    listed = _replace(strict, degree_fields=("Biology", "related field"))
    assert card_fields(strict) == ("Biology",) and card_fields(loose) == ("Biology", "related field")
    assert card_fields(listed) == ("Biology", "related field")  # no second entry
    c = _replace(_full(strict), degree_field="Biochemistry")
    assert _label(c, strict)[0] != "For you"  # a near field is a stretch on the strict card
    assert _label(c, loose) == _label(c, listed)  # the flag reads as the entry
    assert _label(c, loose)[0] == "For you"
