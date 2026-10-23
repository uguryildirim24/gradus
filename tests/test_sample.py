"""Card samplers: determinism, enum validity, and the intended mix."""

from __future__ import annotations

from collections import Counter

import pytest

from matcher.onet import ABSTRACT_SKILLS, DEFAULT_DIR, GENERIC_SOFTWARE, load_occupations
from matcher.metros import parse_place
from matcher.rules import status_problems
from matcher.sample import (
    CONCEPTS,
    FIELDS,
    SOFT_MUST,
    sample_candidate_cards,
    sample_requirement_cards,
)
from matcher.schema import DEGREE_LEVELS, CandidateCard, RequirementCard, norm


@pytest.fixture(scope="module")
def occs():
    if not DEFAULT_DIR.exists():
        pytest.skip("needs O*NET under data/raw/open/onet (gitignored)")
    return load_occupations()


@pytest.fixture(scope="module")
def reqs(occs):
    return sample_requirement_cards(occs, 600, seed=7)


@pytest.fixture(scope="module")
def cands(occs):
    return sample_candidate_cards(occs, 600, seed=8)


def test_onet_loads_broad_set(occs):
    assert len(occs) > 800
    assert {o.degree_level for o in occs} >= {"none", "bachelor", "master", "doctoral"}


def test_requirements_deterministic(occs, reqs):
    again = sample_requirement_cards(occs, 600, seed=7)
    assert [(i, c.to_dict()) for i, c in again] == [(i, c.to_dict()) for i, c in reqs]
    other = sample_requirement_cards(occs, 600, seed=9)
    assert [c.to_dict() for _, c in other] != [c.to_dict() for _, c in reqs]


def test_candidates_deterministic(occs, cands):
    again = sample_candidate_cards(occs, 600, seed=8)
    assert [(i, c.to_dict(), s) for i, c, s in again] == [(i, c.to_dict(), s) for i, c, s in cands]


def test_requirement_ids_and_enums(occs, reqs):
    by_soc = {o.soc: o for o in occs}
    ids = [i for i, _ in reqs]
    assert len(set(ids)) == len(ids) == 600
    assert all(i.startswith("rq_") for i in ids)
    for _, c in reqs:
        RequirementCard.from_dict(c.to_dict())  # re-validates enums
        assert c.title and c.field in FIELDS
        alts = [x for g in c.must_skills for x in g]
        occ = by_soc[c.onet_soc]
        # 0-1 must groups happen on purpose (student postings that ask for little); never more than 7
        assert len(c.must_skills) <= 8
        assert all(isinstance(g, tuple) and g for g in c.must_skills)
        assert not set(map(norm, alts)) & set(map(norm, c.nice_skills))
        if c.enrolled_in != "none":
            assert c.employment_type == "internship"
            assert DEGREE_LEVELS.index(c.degree_level) < DEGREE_LEVELS.index(c.enrolled_in)
        if c.work_mode != "remote":
            assert c.locations and all(parse_place(m).metro or parse_place(m).kind == "state" for m in c.locations)
        if c.salary_min is not None:
            assert c.salary_max >= c.salary_min
        if c.employment_type == "internship":
            assert c.min_years == 0 and c.seniority == "entry"


def test_must_skills_are_concrete(occs, reqs):
    """Abstract O*NET skills (Critical Thinking, English Language, ...) and generic software are
    never must-haves. A must-have is the occupation's knowledge, its software or a field concept,
    or one of the soft skills real postings list (which the rule ignores)."""
    by_soc = {o.soc: o for o in occs}
    soft = 0
    for _, c in reqs:
        occ = by_soc[c.onet_soc]
        for g in c.must_skills:
            for x in g:
                assert x not in ABSTRACT_SKILLS and x not in GENERIC_SOFTWARE, (c.title, x)
                assert x in occ.core_knowledge or x in occ.tools or x in CONCEPTS.get(c.field, ()) \
                    or x in SOFT_MUST, (c.title, x)
                soft += x in SOFT_MUST
    assert 0.1 * len(reqs) < soft < 0.3 * len(reqs)


def test_tech_postings_use_plain_tech_names(reqs):
    names = Counter(x for _, c in reqs if c.field == "IT and data" for g in c.must_skills for x in g)
    assert {"Python", "SQL"} <= set(names)
    assert not any("Structured query language" in x or x.endswith(" software") for x in names)


def test_core_knowledge_is_occupational(occs):
    by_title = {o.title: o for o in occs}
    assert "Food production" in by_title["Cooks, Restaurant"].core_knowledge
    assert by_title["Architects, Except Landscape and Naval"].core_knowledge[:2] == ["Construction methods", "Design"]
    for o in occs:
        assert not set(o.core_knowledge) & ABSTRACT_SKILLS


def test_requirement_mix(reqs):
    cards = [c for _, c in reqs]
    assert {c.sponsorship for c in cards} == {"sponsors", "will_not_sponsor", "will_not_sponsor_or_opt", "unstated"}
    assert {c.enrolled_in for c in cards} >= {"none", "bachelor", "master", "doctoral"}
    assert any(len(g) > 1 for c in cards for g in c.must_skills)
    n = len(cards)
    assert sum(c.employment_type == "internship" for c in cards) > 0.25 * n
    assert sum(c.enrolled_in != "none" for c in cards) > 0.18 * n
    assert sum(c.citizens_only for c in cards) > 0.04 * n
    assert sum(len(c.must_skills) <= 1 for c in cards) > 0.05 * n
    assert any(parse_place(x).kind == "state" for c in cards for x in c.locations)
    assert any(parse_place(x).metro and x != parse_place(x).city for c in cards for x in c.locations)
    assert {c.work_mode for c in cards} == {"onsite", "hybrid", "remote"}
    assert any(c.license for c in cards)
    assert len({c.field for c in cards}) == len(FIELDS)


def test_candidate_ids_and_enums(cands):
    ids = [i for i, _, _ in cands]
    assert len(set(ids)) == len(ids) == 600
    assert all(i.startswith("cd_") for i in ids)
    for _, c, _ in cands:
        CandidateCard.from_dict(c.to_dict())
        assert 0 <= c.years <= 20 and c.skills and c.education_country and c.language
        assert (c.degree_level == "none") == (c.degree_field == "")
        if c.work_auth == "stem_opt":
            assert c.field in {"Lab and life sciences", "IT and data", "Engineering", "Agriculture and environment"}
        assert not status_problems(c), (c, status_problems(c))
        if c.work_auth == "f1_cpt":  # CPT covers internships only: no CPT student avoids "intern"
            assert "intern" not in {norm(x) for x in c.roles_avoided}, c


def test_candidate_mix(cands):
    cards = [c for _, c, _ in cands]
    intl = sum(c.work_auth != "citizen_or_pr" for c in cards) / len(cards)
    assert 0.28 < intl < 0.42
    assert {c.work_auth for c in cards} >= {"f1_cpt", "f1_opt", "stem_opt", "needs_h1b_now", "other_visa"}
    assert len(Counter(c.education_country for c in cards)) >= 30
    assert len({c.language for c in cards}) >= 20
    assert len({c.field for c in cards}) == len(FIELDS)


def test_singular_titles():
    from matcher.sample import singular

    assert singular("Budget Analysts") == "Budget Analyst"
    assert singular("Nursing Assistants") == "Nursing Assistant"
    assert singular("Secretaries") == "Secretary"
    assert singular("Teachers, Postsecondary") == "Teacher, Postsecondary"
    assert singular("Dietitians and Nutritionists") == "Dietitians and Nutritionists"
    assert singular("Physics") == "Physics"
    assert singular("Nurse Midwives") == "Nurse Midwife"
    assert singular("Detectives") == "Detective"


def test_license_holders_have_the_education_it_follows_from(cands, reqs):
    from matcher.degrees import canon_field
    from matcher.onet import LICENSE_EDUCATION, license_fits
    from matcher.rules import license_problems

    held = [c for _, c, _ in cands if c.licenses]
    assert held and all(not license_problems(c) for c in held)
    for c in held:
        if "Medical license" in c.licenses:
            assert c.degree_level == "doctoral" and c.degree_field == "Medicine"
    for _, r in reqs:
        need = LICENSE_EDUCATION.get(r.license) if r.license else None
        if need and need.fields:
            assert {canon_field(f) for f in r.degree_fields} <= {canon_field(f) for f in need.fields}, r
    assert not license_fits("Medical license", "doctoral", "Psychology")
    assert not license_fits("Journeyman electrician license", "bachelor", "Nursing")
    assert license_fits("Journeyman electrician license", "none", "")
    assert not license_fits("Cosmetology license", "bachelor", "Cosmetology")
    assert license_fits("CNA certification", "bachelor", "Biology")


def test_repair_never_adds_a_license_the_degree_cannot_lead_to():
    import random

    from matcher.variants import repair

    card = RequirementCard(source_posting_id="x", title="Physician", field="Clinical healthcare",
                           degree_level="doctoral", license="Medical license", min_years=0)
    psych = CandidateCard(degree_level="doctoral", degree_field="Psychology", graduation_year=2018, years=6)
    assert repair(psych, card, random.Random(0)) is None
    md = CandidateCard(degree_level="doctoral", degree_field="Medicine", graduation_year=2018, years=6)
    fixed = repair(md, card, random.Random(0))
    assert fixed is not None and "Medical license" in fixed[0].licenses


def test_relocation_list_never_names_the_home_metro():
    import random
    from matcher.metros import ANYWHERE_US, places_match
    from matcher.sample import sample_willing
    rng = random.Random(3)
    for home in ("Newton, MA", "Boston, MA", "Palo Alto, CA", "Brooklyn, NY", "Austin, TX"):
        for _ in range(300):
            assert not any(w != ANYWHERE_US and places_match(w, home) for w in sample_willing(rng, home)), home


def test_opt_bachelors_graduates_worked_a_year_at_most_before_graduating(cands):
    for _, c, _ in cands:
        if c.work_auth in ("f1_opt", "stem_opt") and c.degree_level == "bachelor":
            assert c.years <= 2026 - c.graduation_year + 1, c


def test_a_license_the_degree_gives_has_no_work_before_the_graduation(cands, reqs):
    """cd_v_00823 (r4 reader): a residency in June 2015, the MD in 2016. The resume dates every year of work as a job,
    so the holder of a license their degree gives has no more years than years since that graduation."""
    import random

    from matcher.variants import license_needs_degree, make_variants, pre_degree_years, repair, years_fit

    def ok(c):
        if c.graduation_year is None or c.graduation_year > 2026 or not license_needs_degree(c.licenses, c.degree_level):
            return True
        return c.years <= 2026 - c.graduation_year

    assert all(ok(c) for _, c, _ in cands)
    seen = 0
    for _, r in reqs:
        if r.license:
            for v in make_variants(r, seed=0):
                assert ok(v.candidate) and years_fit(v.candidate), (v.kind, v.candidate)
                seen += license_needs_degree(v.candidate.licenses, v.candidate.degree_level)
    assert seen > 50
    md = CandidateCard(degree_level="doctoral", degree_field="Medicine", graduation_year=2016, years=11,
                       licenses=("Medical license",))
    assert pre_degree_years(md) == 0 and not years_fit(md)
    assert years_fit(CandidateCard(degree_level="doctoral", degree_field="Medicine", graduation_year=2016, years=11))
    # repair gives a real candidate the posting's license and moves the graduation to fit their years
    card = RequirementCard(source_posting_id="x", title="Physician", field="Clinical healthcare",
                           degree_level="doctoral", license="Medical license", min_years=0)
    for seed in range(20):
        fixed = repair(CandidateCard(degree_level="doctoral", degree_field="Medicine", graduation_year=2018, years=12),
                       card, random.Random(seed))
        assert fixed is not None and ok(fixed[0]) and "Medical license" in fixed[0].licenses
