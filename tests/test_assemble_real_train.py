"""The real-posting screen: each screen drops the row the label read found wrong and keeps the near misses that
first fooled it; the synthetic rows are refused when their md5 changed."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import assemble_real_train as art  # noqa: E402
from matcher.schema import CandidateCard  # noqa: E402

BASE = {"field": "Software Engineer", "degree_level": "bachelor", "degree_field": "Computer Science",
        "graduation_year": 2028, "years": 0, "skills": ["Python"], "licenses": [], "current_location": "New York, NY",
        "willing_locations": [], "remote_preference": "any", "work_auth": "citizen_or_pr", "salary_floor": None,
        "employment_types_wanted": [], "industries_wanted": [], "industries_avoided": [], "roles_avoided": [],
        "education_country": "", "language": ""}


def cand(**kw) -> CandidateCard:
    return CandidateCard.from_dict(BASE | kw)


# ---------------------------------------------------------------- class standing


@pytest.mark.parametrize("year,text,drops", [
    (2030, "Qualified candidates will be pursuing a bachelor's degree with Junior standing at the time of application.", True),
    (2030, "This full-time, 12-week internship is open to Junior-status college students only.", True),
    (2028, "Junior or Senior status working towards a four-year college degree is required.", False),
    (2029, "A Sophomore, Junior or Senior standing in school with requisite educational experience.", False),
    (2029, "- Rising junior or senior working towards Bachelor's Degree", False),
    (2030, "- Rising junior or senior working towards Bachelor's Degree", True),
    (2029, "- An expected graduation date in 2028 or 2029 (rising Junior or Senior)", False),  # a year: grad_window's
    (2029, "Senior standing preferred.", False),
])
def test_class_standing(year, text, drops):
    assert bool(art.screen_class_standing(cand(graduation_year=year), text)) is drops


def test_class_standing_only_for_enrolled_bachelors():
    text = "Junior standing at the time of application."
    assert art.screen_class_standing(cand(graduation_year=2030, degree_level="master"), text) is None
    assert art.screen_class_standing(cand(graduation_year=2024), text) is None


# ---------------------------------------------------------------- graduation window


@pytest.mark.parametrize("year,text,drops", [
    (2028, "- Must be continuing academic studies through at least Fall 2027 and not graduate before Fall 2027.", False),
    (2030, "- Must not graduate before September 2026.", False),
    (2030, "Graduation date of December 2025 or 2026", True),
    (2030, "Expected degree completion within a year of the internship", True),
    (2027, "Expected degree completion within a year of the internship", False),
    (2027, "Graduating December 2027 or later", False),
    (2026, "Graduating December 2027 or later", True),
    (2030, "We hire graduating students for our Summer 2027 program.", False),
])
def test_grad_window(year, text, drops):
    assert bool(art.screen_grad_window(cand(graduation_year=year), text)) is drops


# ---------------------------------------------------------------- current students only


@pytest.mark.parametrize("text,drops", [
    ("- Currently pursuing a BS, MS, or PhD in Robotics, Computer Science, or a related field;", True),
    ("About Our Part Time Student Worker Program", True),
    ("- Cannot be enrolled in university classes during the internship period", False),
    ("Currently enrolled in a university or educational institution, or recently completed a qualification", False),
    ("Currently pursuing or recently completed a degree in Computer Science.", False),
    ("- Currently pursuing a degree in Computer Science or a related field (or equivalent experience)", False),
])
def test_student_only(text, drops):
    assert bool(art.screen_student_only(cand(graduation_year=2025), text)) is drops


def test_student_only_skips_enrolled_candidates():
    assert art.screen_student_only(cand(graduation_year=2028), "Must be currently enrolled.") is None


# ---------------------------------------------------------------- work authorization


@pytest.mark.parametrize("auth,text,drops", [
    ("f1_cpt", "- Must have the ability to work in the United States for an unlimited amount of time without sponsorship.", True),
    ("f1_opt", "Students on CPT or OPT do not qualify for this role.", True),
    ("f1_cpt", "Pay depends on factors including without limitation the qualifications of the applicant.", False),
    ("f1_cpt", "You must have the ability to work without a need for current or future visa sponsorship.", False),
    ("citizen_or_pr", "- Must have the ability to work in the United States for an unlimited amount of time.", False),
])
def test_permanent_auth(auth, text, drops):
    """Unlimited or permanent authorization is the rule's citizens_only; "now or in the future" alone is its
    will_not_sponsor, which keeps CPT on an internship."""
    assert bool(art.screen_permanent_auth(cand(work_auth=auth), text)) is drops


def test_no_sponsor_only_for_h1b_now():
    text = "FNB will not provide sponsorship for employment-based visas for this position."
    assert art.screen_no_sponsor(cand(work_auth="needs_h1b_now"), text)
    assert art.screen_no_sponsor(cand(work_auth="other_visa"), text) is None
    assert art.screen_no_sponsor(cand(work_auth="needs_h1b_now"), "Sponsorship for future FTE roles is not guaranteed.") is None


# ---------------------------------------------------------------- school, prior internship, location


UT = ("- Must be a current undergraduate or graduate student at The University of Texas at Austin enrolled in at least "
      "12 class hours each semester (NCAA Requirement)")


def test_school_only():
    assert art.screen_school_only("Columbia University, New York, NY", UT)
    assert art.screen_school_only("The University of Texas at Austin, Austin, TX", UT) is None
    assert art.screen_school_only("NYU", "Must be a student enrolled at an accredited university.") is None


def test_prior_internship():
    text = "- Proven software development experience through at least one prior internship or co-op."
    assert art.screen_prior_internship("Education\nProjects", text)
    assert art.screen_prior_internship("Software Engineering Intern, Acme, Summer 2025", text) is None
    assert art.screen_prior_internship("Projects", "Prior internship experience preferred.") is None


@pytest.mark.parametrize("locs,where,willing,drops", [
    (["Ohio"], "New York, NY", [], True),
    (["NYC"], "New York, NY", [], False),
    (["Remote in USA"], "New York, NY", [], False),
    (["Ohio"], "New York, NY", ["anywhere in the US"], False),
    (["South SF"], "South San Francisco, CA", [], False),
    (["Lake St Louis, MO"], "Lake Saint Louis, MO", [], False),
])
def test_location(locs, where, willing, drops):
    c = cand(current_location=where, willing_locations=willing)
    assert bool(art.screen_location(c, {"locations": locs})) is drops


# ---------------------------------------------------------------- soft skills, degree field, industry, Skip rows


def test_soft_nice_skill():
    assert art.screen_soft_nice("nice_skill", "Explore — missing nice-to-have skill: Big-picture understanding, "
                                              "Ethical standards; otherwise meets the requirements.")
    assert art.screen_soft_nice("nice_skill", "Explore — missing nice-to-have skill: 8D Problem Solving, Six Sigma; "
                                              "otherwise meets the requirements.") is None
    assert art.screen_soft_nice("full_match", "Explore — missing nice-to-have skill: teamwork;") is None


def test_broad_degree_field():
    c = cand(degree_field="Electrical Engineering")
    text = "Currently pursuing a Bachelor's or Master's degree in Computer Science or a technical discipline is required."
    assert art.screen_broad_degree(c, "degree_field_adjacent", text)
    assert art.screen_broad_degree(c, "degree_field", text) is None
    assert art.screen_broad_degree(c, "degree_field_adjacent", "0-2 years of experience in VLSI or related technical "
                                                               "areas.") is None


def test_industry_conflict():
    c = cand(industries_avoided=["healthcare"])
    raw = {"title": "AI Engineer Intern", "company": "Direct Supply",
           "description": "Build apps that transform healthcare delivery. The future of healthcare technology."}
    assert art.screen_industry(c, raw, "pref_industry_avoided_other")
    assert art.screen_industry(c, raw, "pref_industry_avoided") is None  # that row's label already says so
    assert art.screen_industry(cand(industries_avoided=["research"]), {"description": "research and research"},
                               "full_match") is None


def test_skip_rows_pass_every_screen():
    row = {"bucket": "Excluded", "prompt": "", "variant_kind": "location", "completion": "Skip — x."}
    assert art.screen_row(row, cand(), {"locations": ["Ohio"], "description": UT}) is None


# ---------------------------------------------------------------- assembly


def test_read_synth_refuses_changed_rows(tmp_path):
    p = tmp_path / "smoke.csv"
    p.write_text("prompt,completion\na,Keep — b.\n")
    with pytest.raises(SystemExit):
        art.read_synth(p, want="0" * 32)
    assert art.read_synth(p, want=art.md5(p)) == [{"prompt": "a", "completion": "Keep — b."}]


def test_mix_keeps_every_row_once():
    synth = [{"prompt": f"s{i}", "completion": "Keep — x."} for i in range(5)]
    real = [{"prompt": f"r{i}", "completion": "Skip — y."} for i in range(3)]
    rows = art.mix(synth, real)
    assert sorted(rows) == sorted([(r["prompt"], r["completion"]) for r in synth + real])
    assert rows == art.mix(synth, real)  # seeded


def test_near_pairs_finds_copies_only():
    a = {"x": " ".join(f"w{i}" for i in range(50)), "y": " ".join(f"v{i}" for i in range(50))}
    b = {"z": " ".join(f"w{i}" for i in range(48))}
    hits = art.near_pairs(a, b)
    assert [(h["a"], h["b"]) for h in hits] == [("x", "z")]
