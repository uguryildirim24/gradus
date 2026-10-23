"""The real-posting test builder, the metrics, the baselines and the resume redaction."""

import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from matcher import evalstats as ev
from matcher.rules import label
from matcher.schema import CandidateCard, RequirementCard

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import baselines  # noqa: E402
import build_real_test as brt  # noqa: E402
import ingest_resumes as ing  # noqa: E402

CARD = RequirementCard(
    source_posting_id="ats_t1", title="Data Intern", field="Data analysis", industry="fintech",
    enrolled_in="bachelor", must_skills=(("Python",), ("SQL", "R")), nice_skills=("Tableau",),
    locations=("Boston, MA",), work_mode="hybrid", sponsorship="unstated", salary_min=60_000, salary_max=70_000,
    employment_type="internship",
)
POSTING = (
    "Title: Data Intern\nCompany: Acme\nLocations: Boston, MA\n\nAcme is a fintech company. This internship is for "
    "students currently pursuing a bachelor's degree, expected graduation in 2027 or 2028. The role is hybrid in "
    "Boston, MA. Required: Python and SQL or R. Preferred: Tableau. Pay: $60,000 - $70,000 per year."
)


def in_window(card: RequirementCard) -> CandidateCard:
    """The full match with its graduation year inside POSTING's window (2027 or 2028): since v3/rules the
    raw full match of a bachelor's student graduates in 2030, which the builder's base_candidate fits."""
    c = brt.full_match(card)
    return replace(c, graduation_year=2028) if c.graduation_year and c.graduation_year > 2028 else c


# ---------------------------------------------------------------- build_real_test

def test_graduation_window_takes_the_latest_future_year():
    assert brt.graduation_window("Expected graduation in 2027 or 2028.") == 2028
    assert brt.graduation_window("Summer 2027 internship") is None
    assert brt.graduation_window("graduated in 2020") is None


def test_candidate_is_fitted_to_the_posting():
    raw, fitted, notes = brt.base_candidate(CARD, POSTING)
    assert raw.graduation_year == 2030 and fitted.graduation_year == 2028  # the posting's window
    assert any("posting window" in n for n in notes)
    assert fitted.years == 0 and raw.years == 2  # internship cards get the minimum, not min + 2
    assert fitted.industries_wanted == ("fintech",) and fitted.salary_floor == 60_000
    # an industry or pay the posting never states is not set as a preference
    bare = POSTING.replace("fintech", "software").replace("$60,000 - $70,000", "competitive")
    _, fitted2, notes2 = brt.base_candidate(CARD, bare)
    assert fitted2.industries_wanted == () and fitted2.salary_floor is None and len(notes2) > len(notes)


def test_rows_never_edit_the_posting_card_and_labels_come_from_label():
    kept, dropped = brt.build_rows({"ats_t1": CARD}, {"ats_t1": POSTING})
    assert kept, dropped
    kinds = {r["variant_kind"] for r in kept}
    assert {"full_match", "auth_f1_cpt", "auth_stem_opt", "enrollment_graduated", "location", "must_skill"} <= kinds
    for r in kept:
        cand = CandidateCard.from_dict(r["candidate"])
        assert label(cand, CARD)[0] == r["bucket"]
        assert r["base_rq_id"] == "ats_t1" and r["pair_id"].startswith("pr_")
    assert len({r["pair_id"] for r in kept}) == len(kept)
    full = next(r for r in kept if r["variant_kind"] == "full_match")
    assert full["bucket"] == "For you" and full["candidate"]["years"] == 0
    # the work-auth crossing: CPT keeps the student's resume facts, a different profile
    cpt = next(r for r in kept if r["variant_kind"] == "auth_f1_cpt")
    assert cpt["candidate"]["work_auth"] == "f1_cpt" and cpt["bucket"] == "For you"  # CPT covers an internship
    assert cpt["candidate"]["graduation_year"] == full["candidate"]["graduation_year"]
    # OPT, STEM OPT and H-1B now need a graduate: on this student posting they are enrollment Skips (the OPT one
    # is the same person as work_auth_opt_student_role, so it is kept once)
    for r in kept:
        c = CandidateCard.from_dict(r["candidate"])
        assert not brt.person_problems(c), (r["variant_kind"], brt.person_problems(c))
        if c.work_auth in ("f1_opt", "stem_opt", "needs_h1b_now"):
            assert c.graduation_year <= 2026 and r["bucket"] == "Excluded", r["variant_kind"]
    stem = next(r for r in kept if r["variant_kind"] == "auth_stem_opt")["candidate"]
    assert stem["years"] >= 1 and 2023 <= stem["graduation_year"] <= 2025


def test_a_label_that_hinges_on_an_unstated_fact_is_dropped():
    text = POSTING.replace("hybrid in Boston, MA", "flexible")  # the location fact is gone from the body
    kept, dropped = brt.build_rows({"ats_t1": CARD}, {"ats_t1": text.replace("Locations: Boston, MA", "Locations: ")})
    assert any(d["drop"] == "posting_does_not_state" and "location" in d["categories"] for d in dropped)
    assert not any(r["variant_kind"] == "relocate" for r in kept)


def test_excluded_rows_need_only_one_visible_hard_reason():
    cand = CandidateCard(degree_level="bachelor", graduation_year=2020, years=0, current_location="Boston, MA",
                         skills=("Python", "SQL"), employment_types_wanted=("internship",))
    bucket, reasons, _ = label(cand, CARD)
    assert bucket == "Excluded"
    assert brt.posting_misses(cand, CARD, reasons, "", POSTING, hard=True) == []


def test_report_counts_buckets_and_kinds():
    kept, dropped = brt.build_rows({"ats_t1": CARD}, {"ats_t1": POSTING})
    rep = brt.report_of(kept, dropped, 1)
    assert rep["n_rows"] == len(kept) and sum(rep["bucket_counts"].values()) == len(kept)
    assert rep["per_kind"]["full_match"]["For you"] == 1


def test_score_rows_clusters_by_posting():
    kept, _ = brt.build_rows({"ats_t1": CARD}, {"ats_t1": POSTING})
    perfect = {r["pair_id"]: ev.WORD[r["bucket"]] for r in kept}
    res = brt.score_rows(kept, perfect, draws=50)
    assert res["headline"]["macro_f1"] == 1.0 and res["headline"]["postings"] == 1
    assert all(v["accuracy"] == 1.0 for v in res["per_kind"].values()) and res["core"]["accuracy"] == 1.0


def test_real_test_set_when_built():
    rows = Path(brt.OUT / "rows.jsonl")
    if not rows.exists():
        pytest.skip("run build_real_test.py build first")
    data = brt.read_jsonl(rows)
    assert len(data) >= 600
    # For you was 15% until the strict-citizenship guard dropped the Keeps a "citizen or permanent
    # resident" profile gets on citizens-only postings (12.7% at the rebuild); the way back to 15% is a
    # citizen-only work-authorization state in the rules and profile branch, not more added Keep kinds.
    floor = {"For you": 0.12, "Explore": 0.15, "Excluded": 0.15}
    for b in brt.BUCKETS:
        assert sum(r["bucket"] == b for r in data) / len(data) >= floor[b], b


def test_undergraduate_is_not_a_graduation_word():
    assert brt.graduation_window("Enrolled full-time in an undergraduate or graduate degree in Fall 2027.") is None
    assert brt.graduation_window("Graduating in 2027.") == 2027


def test_far_city_is_never_in_the_posting_metro():
    assert brt.same_metro("Minneapolis, MN", "St Paul, MN") and brt.same_metro("Newton, MA", "Cambridge, MA")
    assert not brt.same_metro("Boston, MA", "Denver, CO")
    card = replace(CARD, locations=("Hartford, CT", "St Paul, MN"))
    text = POSTING.replace("Boston, MA", "Hartford, CT")
    kept, _ = brt.build_rows({"ats_t1": card}, {"ats_t1": text})
    for r in kept:
        if r["variant_kind"] in ("location", "relocate"):
            assert not brt.same_metro(r["candidate"]["current_location"], "St Paul, MN")
            assert not brt.same_metro(r["candidate"]["current_location"], "Hartford, CT")


def test_wanted_job_type_is_one_the_posting_does_not_name():
    text = POSTING + " Part-time hours are possible during the semester; full-time in the summer."
    kept, dropped = brt.build_rows({"ats_t1": CARD}, {"ats_t1": text})
    rows = [r for r in kept + dropped if r["variant_kind"] == "employment_type"]
    assert rows and all(tuple(r["candidate"]["employment_types_wanted"]) == ("contract",) for r in rows)


def test_soft_contained_and_example_only_skills_are_not_skip_reasons():
    soft = replace(CARD, must_skills=(("Communication",),))
    # since v3/rules the rule itself never enforces a soft group, so there is no Skip to guard
    assert label(replace(in_window(soft), skills=("Linux",)), soft)[0] != "Excluded"
    assert brt.guard("must_skill", replace(in_window(soft), skills=("Linux",)), soft, "Excluded",
                     POSTING + " Communication") is None
    # a generic word the rule does enforce is still no Skip reason here
    stats = replace(CARD, must_skills=(("Statistics",),))
    assert brt.guard("must_skill", replace(in_window(stats), skills=("Linux",)), stats, "Excluded",
                     POSTING + " Statistics") == "missing skill is a soft or generic word"
    coding = replace(CARD, must_skills=(("Coding",),))
    why = brt.guard("must_skill", replace(in_window(coding), skills=("Coding Agents",)), coding, "Excluded", POSTING)
    assert why and "contains" in why
    java = replace(CARD, must_skills=(("Java",),))
    shown = POSTING + " Interested in languages such as Java."
    assert "example" in brt.guard("must_skill", replace(in_window(java), skills=("Linux",)), java, "Excluded", shown)
    assert brt.guard("must_skill", replace(in_window(java), skills=("Linux",)), java, "Excluded",
                     POSTING + " Java is required.") is None


def test_future_sponsorship_wording_drops_temporary_status_keeps_and_explores():
    card = replace(CARD, sponsorship="will_not_sponsor")
    bar = POSTING + " Individuals with temporary visas or who need sponsorship now or in the future are not eligible."
    kept, dropped = brt.build_rows({"ats_t1": card}, {"ats_t1": bar})
    temp = [r for r in kept if r["candidate"]["work_auth"] in brt.TEMPORARY]
    assert temp and all(r["bucket"] == "Excluded" for r in temp)
    assert any(d["drop"] == "text_contradicts_label" and d["candidate"]["work_auth"] == "f1_cpt" for d in dropped)
    welcome = bar + " F-1 students are eligible if they are using CPT."
    kept, _ = brt.build_rows({"ats_t1": card}, {"ats_t1": welcome})
    assert any(r["candidate"]["work_auth"] == "f1_cpt" and r["bucket"] == "For you" for r in kept)


def test_obtain_a_degree_by_names_the_graduation_window():
    text = "Scheduled to obtain a Bachelor's degree in a Technology field by Summer 2028 or Summer 2029"
    assert brt.graduation_window(text) == 2029
    # a date for finishing coursework or the internship itself is not a graduation date
    assert brt.graduation_window("Ability to complete a 12-week in-person internship in summer 2027.") is None
    assert brt.graduation_window("1st year of PhD coursework must be completed by June 2027") is None


def test_a_student_posting_gets_a_student_even_when_the_card_does_not_ask():
    card = replace(CARD, enrolled_in="none")
    text = POSTING.replace("expected graduation in 2027 or 2028", "with time left in school")
    text = text.replace("students currently pursuing", "undergraduate students pursuing")
    _, fitted, notes = brt.base_candidate(card, text)
    assert fitted.graduation_year == brt.CURRENT_YEAR + 2 and any("current students" in n for n in notes)
    raw, fitted, notes = brt.base_candidate(card, text + " Recent graduates are welcome too.")
    assert raw.graduation_year == 2023 and raw.years == 2  # recent graduates fit: the candidate stays a graduate,
    # but with the internship's 0 years a 2023 graduate does not fit (variants.years_fit), so a recent one
    assert fitted.years == 0 and 2024 <= fitted.graduation_year <= brt.CURRENT_YEAR and brt.years_fit(fitted)
    assert any("years of work" in n for n in notes)
    # every kept non-Skip row carries a graduation year inside a window the text states
    kept, _ = brt.build_rows({"ats_t1": CARD}, {"ats_t1": POSTING})
    assert all(2027 <= r["candidate"]["graduation_year"] <= 2028 for r in kept if r["bucket"] != "Excluded")
    late = replace(in_window(CARD), graduation_year=2030)
    assert "window" in brt.guard("full_match", late, CARD, "For you", POSTING)


def test_required_gpa_drops_keeps_and_explores_but_not_a_preferred_one():
    req = POSTING + "\n\nRequirements:\n\n-\n\nMinimum 3.0 GPA\n"
    assert brt.required_gpa(req) == "Minimum 3.0 GPA"
    assert brt.required_gpa(POSTING + "\n\nPreferred Qualifications:\n\n-\n\nGPA of 3.5 or above\n") is None
    assert brt.required_gpa(POSTING + "\n- Cumulative GPA of 3.0 or greater preferred.") is None
    kept, dropped = brt.build_rows({"ats_t1": CARD}, {"ats_t1": req})
    assert kept and all(r["bucket"] == "Excluded" for r in kept)
    assert any(d.get("detail") == "posting requires a minimum GPA the card cannot hold" for d in dropped)


def test_a_missing_nice_skill_on_an_any_of_line_the_candidate_meets_is_dropped():
    card = replace(CARD, nice_skills=("Tableau", "Power BI"))
    cand = replace(in_window(card), skills=("Python", "SQL", "Power BI"))
    anyof = POSTING.replace("Preferred: Tableau.", "Preferred: Tableau or Power BI.")
    assert "one choice" in brt.guard("nice_skill", cand, card, "Explore", anyof)
    both = POSTING.replace("Preferred: Tableau.", "Preferred: Tableau. Also preferred: Power BI.")
    assert brt.guard("nice_skill", cand, card, "Explore", both) is None
    # a list introduced as "one of the following" counts too
    listed = POSTING + "\nPreferred: research in one of the following areas:\n-\nTableau\n-\nPower BI\n"
    assert "one choice" in brt.guard("nice_skill", cand, card, "Explore", listed.replace("Preferred: Tableau.", ""))


def test_slash_pairs_generic_words_and_interest_are_not_skip_reasons():
    card = replace(CARD, must_skills=(("Python",), ("C",)), nice_skills=())
    cand = replace(in_window(card), skills=("Python", "C++"))
    shown = POSTING + " Required: Experience with C/C++ and scripting language(s)."
    assert "slash" in brt.guard("must_skill", cand, card, "Excluded", shown)
    trouble = replace(CARD, must_skills=(("Microsoft Office",),))
    assert "soft" in brt.guard("must_skill", replace(in_window(trouble), skills=("Linux",)), trouble, "Excluded",
                               POSTING + " Proficiency with Microsoft Office.")
    cloud = replace(CARD, must_skills=(("Cloud services",),))
    shown = ("Title: Intern - Cloud Services\n\nOur Cloud Services team builds things.\n"
             "- Experience or strong interest in hardware-software integration, cloud services, and AI.")
    assert "example" in brt.guard("must_skill", replace(in_window(cloud), skills=("Linux",)), cloud, "Excluded",
                                  shown)


def test_a_held_example_shows_the_missing_skill_only_for_the_phrase_it_follows():
    held = ("PyTorch", "pandas", "NumPy", "Data analysis")
    assert brt.held_example("Experience with deep learning frameworks such as PyTorch", "Deep Learning", held)
    assert brt.held_example("Experience with cheminformatics tools (e.g., RDKit).", "Cheminformatics", ("RDKit",))
    # the examples belong to "the data stack", and "data analysis" is a second item, not an example
    assert not brt.held_example("Strong Python and the data stack (pandas, NumPy, scikit-learn).", "Python", held)
    assert not brt.held_example("Experience with AI-enabled tools, including Claude AI Agents, and data analysis "
                                "methodologies.", "AI-enabled tools", held)
    # a conjunctive parenthetical is not a list of examples
    assert not brt.paren_example("Required: Proficiency in Python and its ML ecosystem (NumPy and pandas).", "pandas")


def test_part_time_hours_without_the_word_count_as_naming_part_time():
    for hours in ("up to 20 hours/week during the fall and spring semesters",
                  "a minimum of 20 hours for spring and fall interns", "flexibility regarding hours"):
        assert re.search(brt.TYPE_PHRASE["part_time"], hours, re.I), hours
    assert not re.search(brt.TYPE_PHRASE["part_time"], "40 hours per week, Monday-Friday", re.I)
    kept, dropped = brt.build_rows({"ats_t1": CARD}, {"ats_t1": POSTING + " Up to 20 hours/week during the semesters."})
    rows = [r for r in kept + dropped if r["variant_kind"] == "employment_type"]
    assert rows and all("part_time" not in r["candidate"]["employment_types_wanted"] for r in rows)


def test_assemble_marks_rows_the_rendered_resume_does_not_show():
    kept, _ = brt.build_rows({"ats_t1": CARD}, {"ats_t1": POSTING})
    rows = [r for r in kept if r["variant_kind"] in ("full_match", "must_skill")]
    skill = next(r for r in rows if r["variant_kind"] == "must_skill")
    gone = skill["note"].removeprefix("no ").split(" / ")[0]
    good = {r["cd_id"]: "Candidate\nB.S. student, expected graduation 2028\nSkills: " + ", ".join(r["candidate"]["skills"])
            for r in rows}
    out, missing, _ = brt.assemble_rows(rows, {"ats_t1": CARD}, {"ats_t1": POSTING}, lambda r: good[r["cd_id"]])
    assert not missing and all(o["decidable"] and o["missing"] == [] for o in out)
    bad = dict(good)
    full = next(r for r in rows if r["variant_kind"] == "full_match")
    bad[full["cd_id"]] = good[full["cd_id"]].replace("2028", "soon")  # no graduation year
    bad[skill["cd_id"]] = good[skill["cd_id"]] + ", " + gone  # names the skill the candidate lacks
    out, _, cats = brt.assemble_rows(rows, {"ats_t1": CARD}, {"ats_t1": POSTING}, lambda r: bad[r["cd_id"]])
    by = {o["pair_id"]: o for o in out}
    assert not by[full["pair_id"]]["decidable"] and not by[skill["pair_id"]]["decidable"]
    assert cats["enrollment"] >= 1 and cats["skill_contradiction"] >= 1



def test_a_month_bounded_graduation_window_takes_a_year_safe_at_every_edge():
    assert brt.month_safe_years("Must be graduating in 2028.") is None  # a bare year is no month bound
    assert min(brt.month_safe_years("Graduation date must occur on or after December 2027.")) == 2028
    assert 2027 not in brt.month_safe_years("Must be enrolled and not graduate before August 2027.")
    assert max(brt.month_safe_years("Actively pursuing a degree and graduating by June 2028")) == 2027
    assert brt.month_safe_years("Expected graduation date: Fall 2027 through Spring 2029") == {2028}
    assert not brt.month_safe_years("Must be graduating between December 2027 - May 2028")
    assert brt.month_safe_years("Expected graduation date of Fall 2025 or later.") is None  # past bound only
    text = POSTING.replace("expected graduation in 2027 or 2028", "graduation date on or after December 2027")
    _, fitted, notes = brt.base_candidate(CARD, text)
    assert fitted.graduation_year == 2028 and any("month window" in n for n in notes)
    kept, dropped = brt.build_rows({"ats_t1": CARD}, {"ats_t1": text})
    assert all(r["candidate"]["graduation_year"] >= 2028 for r in kept if r["bucket"] != "Excluded")
    # a window no whole year fits leaves only Skips
    narrow = POSTING.replace("expected graduation in 2027 or 2028", "graduating between December 2027 and May 2028")
    kept, dropped = brt.build_rows({"ats_t1": CARD}, {"ats_t1": narrow})
    assert kept and all(r["bucket"] == "Excluded" for r in kept)
    assert any("month edges" in str(d.get("detail")) for d in dropped)


def test_assemble_checks_a_graduation_month_against_a_month_window():
    shown = "Graduation date must occur on or after December 2027."
    assert brt.resume_month_ok("EDUCATION\nB.S., Expected May 2028", shown, 2028) is None
    assert brt.resume_month_ok("EDUCATION\nB.S., Expected 2028", shown, 2028) is None  # every month of 2028 fits
    assert "outside" in brt.resume_month_ok("EDUCATION\nB.S., Expected May 2027", shown, 2027)
    assert "no graduation month" in brt.resume_month_ok("EDUCATION\nB.S., Expected 2027", shown, 2027)
    assert brt.resume_month_ok("EDUCATION\nB.S., Expected December 2027", shown, 2027) is None
    assert brt.resume_month_ok("anything", "Graduating in 2028.", 2028) is None


def test_expected_counts_as_a_graduation_word_only_next_to_graduation():
    assert brt.graduation_window("Expected in-office schedule is 3 days a week, summer of 2027.") is None
    assert brt.graduation_window("Expected graduation in 2028.") == 2028
    assert brt.graduation_window("Following the completion of the internship in 2027.") is None


def test_avoided_industry_is_not_also_wanted():
    kept, _ = brt.build_rows({"ats_t1": CARD}, {"ats_t1": POSTING})
    rows = [r for r in kept if r["variant_kind"] == "pref_industry_avoided"]
    assert rows and all("fintech" not in r["candidate"]["industries_wanted"] for r in rows)
    assert all(r["bucket"] == "Explore" for r in rows)


def test_a_held_skill_that_implies_the_missing_one_is_not_a_skip_reason():
    card = replace(CARD, must_skills=(("Python",),), nice_skills=())
    cand = replace(in_window(card), skills=("pandas", "NumPy", "scikit-learn"))
    shown = POSTING + " Strong Python and the data stack (pandas, NumPy, scikit-learn)."
    assert "implies" in brt.guard("must_skill", cand, card, "Excluded", shown)
    excel = replace(CARD, must_skills=(("Power Query",),), nice_skills=())
    assert "implies" in brt.guard("must_skill", replace(in_window(excel), skills=("Excel", "PivotTables")), excel,
                                  "Excluded", POSTING + " Excel, including PivotTables, Power Query, formulas.")
    java = replace(CARD, must_skills=(("Java",),), nice_skills=())
    assert brt.guard("must_skill", replace(in_window(java), skills=("pandas",)), java, "Excluded",
                     POSTING + " Java is required.") is None


def test_or_equivalent_degree_and_forty_hours_and_student_workers():
    assert brt.or_equivalent_degree("Pursuing a BS or BA in Computer Science or equivalent experience", "bachelor")
    assert not brt.or_equivalent_degree("Pursuing a BS in Computer Science.", "bachelor")
    for hours in ("40 hours/week over the summer", "up to 40 hours per week with a minimum of 20 hours"):
        assert re.search(brt.TYPE_PHRASE["full_time"], hours, re.I), hours
    assert not re.search(brt.TYPE_PHRASE["full_time"], "up to 20 hours per week", re.I)
    cpt = replace(in_window(CARD), work_auth="f1_cpt")
    assert "student-worker" in brt.guard("auth_f1_cpt", cpt, CARD, "Explore", POSTING + " A contract student worker program.")
    assert brt.guard("auth_f1_cpt", cpt, CARD, "For you", POSTING) is None
    conoco = POSTING + (" Must be legally authorized to work in the United States on a full-time basis for anyone other"
                        " than current employer and/or education institution.")
    assert "anyone other than" in brt.guard("auth_f1_cpt", cpt, CARD, "For you", conoco)
    assert "anyone other than" in brt.guard("work_auth_cpt_internship", cpt, CARD, "Explore", conoco)
    assert brt.guard("auth_f1_cpt", replace(cpt, work_auth="citizen_or_pr"), CARD, "For you", conoco) is None

# ---------------------------------------------------------------- metrics

def test_exact_ties_go_to_the_safer_label():
    assert ev.pick([1.0, 1.0, 0.5]) == 1  # Keep vs Explore tie -> Explore
    assert ev.pick([2.0, 2.0, 2.0]) == 2  # three-way tie -> Skip
    assert ev.pick([3.0, 1.0, 3.0]) == 2  # Keep vs Skip tie -> Skip
    assert ev.pick([0.0, 1.0, 0.5]) == 1


def test_temperature_fit_recovers_the_calibration():
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 4, (4000, 3))
    true_t = 2.5
    p = ev.softmax(scores, true_t)
    gold = np.array([rng.choice(3, p=row) for row in p])
    assert ev.fit_temperature(scores, gold) == pytest.approx(true_t, rel=0.12)
    fitted = ev.fit_temperature(scores, gold)
    assert ev.ece(ev.softmax(scores, fitted), gold) < ev.ece(ev.softmax(scores, 1.0), gold)


def test_ece_and_brier_extremes():
    gold = np.array([0, 1, 2, 0])
    sure_wrong = np.eye(3)[[1, 2, 0, 1]]
    assert ev.ece(sure_wrong, gold) == pytest.approx(1.0)
    assert ev.brier(sure_wrong, gold) == pytest.approx(2.0)
    sure_right = np.eye(3)[gold]
    assert ev.ece(sure_right, gold) == 0.0 and ev.brier(sure_right, gold) == 0.0
    uniform = np.full((4, 3), 1 / 3)
    assert ev.brier(uniform, gold) == pytest.approx(2 / 3)


def test_confusion_and_per_class():
    gold = np.array([0, 0, 1, 2, 2, 2])
    pred = np.array([0, 1, 1, 2, 2, 0])
    conf = ev.confusion(gold, pred)
    assert conf.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 2]]
    pc = ev.prf(conf)
    assert pc["Keep"]["precision"] == 0.5 and pc["Skip"]["recall"] == round(2 / 3, 4)


def test_macro_f1_of_words_and_indices():
    assert ev.macro_f1_of(["Keep", "Explore", "Skip"], ["Keep", "Explore", "Skip"]) == 1.0
    assert ev.macro_f1_of([0, 1, 2], [0, 0, 0]) == pytest.approx(0.5 / 3)


def test_cluster_interval_is_wider_when_rows_share_a_posting():
    rng = np.random.default_rng(1)
    rate = rng.random(10)
    hit = rng.random(300) < np.repeat(rate, 30)  # each posting has its own hit rate
    many = [f"p{i}" for i in range(300)]
    few = [f"p{i // 30}" for i in range(300)]
    fn = lambda idx: float(hit[idx].mean())  # noqa: E731
    _, lo1, hi1 = ev.cluster_ci(many, fn, 400)
    _, lo2, hi2 = ev.cluster_ci(few, fn, 400)
    assert (hi2 - lo2) > (hi1 - lo1)


def _rows(n=60, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        g = ev.LABELS[i % 3]
        s = rng.normal(0, 1, 3)
        s[ev.LABELS.index(g)] += 2.0
        out.append({"pair_id": f"pr{i}", "rq_id": f"rq_{i // 4}~k", "gold": g, "scores": [float(x) for x in s],
                    "pred_old": ev.LABELS[int(np.argmax(s))],
                    "completion": "Skip — limited to US citizens or permanent residents." if g == "Skip" and i % 2 == 0 else ""})
    return out


def test_analyze_reports_every_metric():
    res = ev.analyze(_rows(), draws=100)
    for key in ("accuracy", "macro_f1", "ece15", "brier"):
        assert res[key]["ci95"][0] <= res[key]["value"] <= res[key]["ci95"][1] + 1e-9
    assert res["postings"] == 15 and res["temperature_source"].startswith("fitted on these rows")
    assert set(res["confusion"]) == set(ev.LABELS) and set(res["per_class"]) == set(ev.LABELS)
    assert res["hard_filter"]["gold_skip_on_work_auth_hard_rule"] == 10
    assert res["agreement_with_old_argmax"] == 1.0
    assert len(res["reliability"]) >= 1
    ev.report(res)  # prints without error


def test_hard_filter_violation_counts_gold_skip_predicted_keep():
    rows = _rows(30)
    for r in rows:
        r["flags"] = ["work_auth:citizens_only"] if r["gold"] == "Skip" else []
    for r in rows:
        if r["gold"] == "Skip":
            r["scores"] = [5.0, 0.0, 0.0]  # every Skip predicted Keep
    res = ev.analyze(rows, 1.0, "given", draws=50)
    assert res["hard_filter"]["violation_rate"]["value"] == 1.0
    assert res["hard_filter"]["predicted_keep"] == res["hard_filter"]["gold_skip_on_work_auth_hard_rule"] == 10


# ---------------------------------------------------------------- baselines

def test_constant_baselines_and_majority():
    rows = [{"pair_id": f"p{i}", "bucket": b, "base_rq_id": f"r{i}", "variant_kind": "k"}
            for i, b in enumerate(["Explore", "Explore", "Excluded", "For you"])]
    preds = baselines.constant_preds(rows)
    assert set(preds["majority"].values()) == {"Explore"}
    res = baselines.evaluate(rows, preds["always_explore"], 50)
    assert res["accuracy"] == 0.5 and res["unsafe_keep"] == 0
    assert baselines.evaluate(rows, preds["always_keep"], 50)["unsafe_keep"] == 3


def test_extract_rules_uses_the_extracted_card_not_the_gold_one(tmp_path):
    kept, _ = brt.build_rows({"ats_t1": CARD}, {"ats_t1": POSTING})
    wrong = replace(CARD, locations=("Denver, CO",))  # the extractor got the city wrong
    cards = tmp_path / "cards.jsonl"
    cards.write_text(json.dumps({"id": "ats_t1", "card": wrong.to_dict()}) + "\n")
    pred, info = baselines.extract_rules_preds(kept, cards)
    assert info["rows_without_extracted_card"] == 0
    full = next(r for r in kept if r["variant_kind"] == "full_match")
    assert pred[full["pair_id"]] == "Skip" and full["bucket"] == "For you"


# ---------------------------------------------------------------- resumes

def test_redact_strips_contact_data_and_the_name_line():
    raw = ("Jane Q. Doe\n123 Main Street, Apt 4\nBoston, MA 02118\n(617) 555-0143 | jane.doe@example.com | "
           "linkedin.com/in/janedoe\n\nSUMMARY\nAnalyst with 5 years. Portfolio: https://jane.dev/work\n")
    text, counts = ing.redact(raw)
    assert text.startswith("Candidate\n") and "Jane" not in text
    for bad in ("@", "555", "linkedin", "https", "Main Street", "02118"):
        assert bad not in text, bad
    assert "Boston, MA" in text and "SUMMARY" in text
    assert counts["email"] == 1 and counts["phone"] == 1 and counts["name_line_removed"] == 1


def test_redact_keeps_a_job_title_first_line():
    text, counts = ing.redact("ACCOUNTANT\nSummary\nCareful accountant.")
    assert "ACCOUNTANT" in text and counts["name_line_removed"] == 0
    text, counts = ing.redact("HR ADMINISTRATOR/MARKETING ASSOCIATE\nSummary\nPeople person.")
    assert "HR ADMINISTRATOR/MARKETING ASSOCIATE" in text and counts["name_line_removed"] == 0
    text, counts = ing.redact("SENIOR ACCOUNTANT\nSummary\nCareful accountant.")
    assert "SENIOR ACCOUNTANT" in text
    text, counts = ing.redact("Jane Q. Doe\nSummary\nCareful accountant.")
    assert "Jane" not in text and counts["name_line_removed"] == 1
    text, counts = ing.redact("Summary\nCareful accountant, 9 years.")
    assert text == "Candidate\nSummary\nCareful accountant, 9 years." and counts["name_line_removed"] == 0


def test_prompt_ready_normalizes_and_caps():
    t = ing.prompt_ready("Candidate\n\n\n\n• built   things\x00\n● led teams\n" + "x line\n" * 2000, max_chars=300)
    assert "\n\n\n" not in t and "•" not in t and "\x00" not in t and len(t) <= 300
    assert t.startswith("Candidate\n\n* built things")


def test_pick_files_spreads_over_categories():
    files = [f"data/data/{c}/{i}.pdf" for c in ("A", "B", "C") for i in range(10)]
    got = ing.pick_files(files, 6, 0)
    assert len(got) == 6 and {f.split("/")[-2] for f in got} == {"A", "B", "C"}


def test_pdf_text_roundtrip(tmp_path):
    pytest.importorskip("pypdf")
    stream = b"BT /F1 12 Tf 72 720 Td (Jane Doe jane@example.com) Tj ET"
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    pdf, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(pdf))
        pdf += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offs)
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    p = tmp_path / "r.pdf"
    p.write_bytes(pdf)
    text, counts = ing.redact(ing.pdf_text(p))
    assert "jane@example.com" not in text and counts["email"] == 1


# ---------------------------------------------------------------- fixes from the v3 review

CITIZENS = replace(CARD, citizens_only=True)


def test_strict_citizenship_drops_keeps_and_explores_for_the_citizen_or_pr_profile():
    strict = POSTING + "\nU.S. Citizen, U.S. Person, or Immigration Status Requirements:\nU.S. citizenship is required, as only U.S. citizens are authorized."
    assert brt.strict_citizen(strict)  # the heading naming "U.S. Person" does not count
    full = in_window(CITIZENS)
    assert "citizenship" in brt.guard("full_match", full, CITIZENS, "For you", strict)
    assert brt.guard("location", replace(full, current_location="Denver, CO"), CITIZENS, "Excluded", strict) is None
    for ok in (" Must be a U.S. citizen or lawful permanent resident.", " Must be a U.S. Person (green card holders count)."):
        assert brt.guard("full_match", full, CITIZENS, "For you", POSTING + " US Citizenship required." + ok) is None
    for text in ("Candidates must be US Citizens", "restricted to U.S. Citizens, and applicants with dual citizenship",
                 "- Ability to obtain a Secret clearance", "US CITIZENSHIP REQUIRED and ability to obtain a clearance",
                 "must be a U.S. citizen to apply."):
        assert brt.strict_citizen(POSTING + "\n" + text), text
    assert not brt.strict_citizen(POSTING + "\nUS Citizenship Required:\nNo\n")  # a form field, not an ask
    # a card without citizens_only is left to the other guards
    assert brt.guard("full_match", in_window(CARD), CARD, "For you", strict) is None
    kept, dropped = brt.build_rows({"ats_t1": CITIZENS}, {"ats_t1": strict})
    assert kept and all(r["bucket"] == "Excluded" for r in kept)
    assert any("citizenship" in str(d.get("detail")) for d in dropped)


def test_a_citizens_only_skip_on_an_export_license_route_is_left_out():
    itar = (POSTING + "\nTo conform to U.S. Government export regulations, applicant must be a U.S. citizen, lawful "
            "permanent resident of the U.S., protected individual as defined by 8 U.S.C. 1324b(a)(3), or be eligible to "
            "obtain the required authorizations from the U.S. Department of State.")
    clearance = POSTING + "\nU.S. citizenship is required. Must be eligible to obtain a Secret clearance."
    assert brt.EXPORT_ROUTE.search(itar) and not brt.EXPORT_ROUTE.search(clearance)
    assert brt.EXPORT_ROUTE.search("or otherwise eligible for a federally issued export control license.")
    kept, dropped = brt.build_rows({"ats_t1": CITIZENS}, {"ats_t1": itar})
    assert not any(brt.CITIZEN_REASON in x for r in kept for x in r["reasons"])
    assert any("export license" in str(d.get("detail")) for d in dropped)
    kept2, _ = brt.build_rows({"ats_t1": CITIZENS}, {"ats_t1": clearance})
    assert any(brt.CITIZEN_REASON in x for r in kept2 for x in r["reasons"])  # a clearance is no route: the Skips stay


def test_a_local_student_posting_has_no_relocate_keep():
    assert brt.LOCAL_ONLY.search("We are seeking a local student who can start in June 2027.")
    kept, _ = brt.build_rows({"ats_t1": CARD}, {"ats_t1": POSTING + " We are seeking a local student."})
    assert not any(r["variant_kind"] == "relocate" for r in kept)


def test_ml_work_implies_python_and_a_shared_parenthetical_is_no_skip():
    # pr_1d1f86df31cb: "Proficiency in Python and experience with machine learning, computer vision"
    card = replace(CARD, must_skills=(("Python",),), nice_skills=())
    cand = replace(in_window(card), skills=("Machine learning", "Image Processing", "Generative AI", "Git"))
    shown = POSTING + " Proficiency in Python and experience with machine learning, computer vision, and AI."
    assert "implies" in brt.guard("must_skill", cand, card, "Excluded", shown)
    assert brt.implied_by_held("Python", ("Git", "GitHub")) is None  # version control is not Python work
    # pr_4d44b28afb6d: "Proficiency in Python and its ML ecosystem (NumPy and pandas)", candidate has NumPy
    pandas = replace(CARD, must_skills=(("Python",), ("pandas",)), nice_skills=())
    cand = replace(in_window(pandas), skills=("Python", "NumPy"))
    shown = POSTING + " Proficiency in Python and its ML ecosystem (NumPy and pandas)."
    assert "parenthetical" in brt.guard("must_skill", cand, pandas, "Excluded", shown)
    # pandas alone in the brackets, no held neighbour: the Skip stands
    assert brt.guard("must_skill", cand, pandas, "Excluded", POSTING + " Experience with data tools (pandas) is required.") is None


def test_a_plain_opt_skip_needs_the_text_to_name_opt():
    card = replace(CARD, sponsorship="will_not_sponsor_or_opt")
    opt = replace(in_window(card), work_auth="f1_opt")
    stem_only = POSTING + " The company does not sponsor/support H-1B petitions, TN, or Forms I-983/STEM OPT."
    assert not brt.names_plain_opt(stem_only)
    assert "plain OPT" in brt.guard("work_auth_no_opt", opt, card, "Excluded", stem_only)
    stem = replace(opt, work_auth="stem_opt")
    assert brt.guard("work_auth_stem_opt", stem, card, "Excluded", stem_only) is None
    named = POSTING + " We cannot hire candidates on OPT or STEM OPT."
    assert brt.names_plain_opt(named) and brt.guard("work_auth_no_opt", opt, card, "Excluded", named) is None
    assert not brt.names_plain_opt("You can opt in to updates.")  # the verb is not OPT


def test_wanted_industry_draw_avoids_the_postings_own_industry_family():
    assert brt.industry_related("manufacturing", "Semiconductor")
    assert brt.industry_related("government", "Aerospace and Defense")
    assert brt.industry_related("government", "Nuclear Energy")
    assert brt.industry_related("biotech and pharma", "Research and Development")
    assert brt.industry_related("technology", "Defense Technology")
    assert not brt.industry_related("hospitality", "Semiconductor")
    card = replace(CARD, industry="Semiconductor")
    cand = replace(in_window(card), industries_wanted=("manufacturing",))
    out = brt.refit("pref_industry_wanted", cand, card, POSTING, brt.random.Random(0))
    assert out.industries_wanted and not brt.industry_related(out.industries_wanted[0], "Semiconductor")


def test_a_work_model_section_cut_off_drops_location_rows():
    cut = POSTING + "\n\nWork Model\n\nOur culture prioritizes collaboration, whether in person or virtually. [truncated]"
    assert brt.mode_hidden_by_cut(cut)
    assert not brt.mode_hidden_by_cut(POSTING + "\n\nWork Model\n\nThis role is on-site in Boston. [truncated]")
    assert not brt.mode_hidden_by_cut(POSTING + "\n\nWork Model\n\nCollaboration matters.")  # not cut
    # GDIT: a "Work Location:" field and later headings, the mode stated above ("Telecommuting Options: Hybrid")
    gdit = POSTING + "\n\nTelecommuting Options:\nHybrid\n\nWork Location:\nUSA LA Bossier City\n\nTotal Rewards:\nBenefits [truncated]"
    assert not brt.mode_hidden_by_cut(gdit)
    assert not brt.mode_hidden_by_cut(POSTING + "\n\nWork Model\n\nCollaboration.\n\nBenefits:\nMedical [truncated]")
    away = replace(in_window(CARD), current_location="Denver, CO")
    assert "work-model" in brt.guard("location", away, CARD, "Excluded", cut)


def test_posting_text_unescapes_html_entities():
    p = {"title": "Intern", "company": "RTX", "locations": ["Tucson, AZ"], "description": "Line one&#xa;&#xa;R&amp;D"}
    assert brt.posting_text(p).endswith("Line one\n\nR&D")


def test_report_gives_the_for_you_share_without_added_kinds():
    kept, dropped = brt.build_rows({"ats_t1": CARD}, {"ats_t1": POSTING})
    rep = brt.report_of(kept, dropped, 1, {"ats_t1": CARD})
    core = [r for r in kept if r["variant_kind"] not in ev.ADDED_KEEP_KINDS]
    assert rep["for_you_share_without_added_keep_kinds"] == round(
        sum(r["bucket"] == "For you" for r in core) / len(core), 3)
    assert rep["posting_employment_types"] == {"internship": 1}
    assert rep["resume_kind_rows"] == sum(r["variant_kind"] in ev.RESUME_KINDS for r in kept)


def test_score_rows_gives_side_headlines():
    kept, _ = brt.build_rows({"ats_t1": CARD}, {"ats_t1": POSTING})
    # right on every row but the added Keep kinds, which are all called Explore
    preds = {r["pair_id"]: ("Explore" if r["variant_kind"] in ev.ADDED_KEEP_KINDS else ev.WORD[r["bucket"]]) for r in kept}
    res = brt.score_rows(kept, preds, draws=50)
    assert res["headline"]["macro_f1"] < 1.0
    side = res["side_headlines"]
    assert side["without_added_keep_kinds"]["macro_f1"]["value"] == 1.0
    assert side["resume_kinds"]["n"] == sum(r["variant_kind"] in ev.RESUME_KINDS for r in kept)


def test_analyze_leaves_undecidable_rows_out_unless_asked():
    rows = _rows(30)
    for i, r in enumerate(rows[:5]):
        r["visible"] = False
        r["scores"] = [0.0, 0.0, 9.0] if r["gold"] != "Skip" else [9.0, 0.0, 0.0]  # wrong on purpose
    for r in rows[5:]:
        r["visible"] = True
    res = ev.analyze(rows, 1.0, "given", draws=50)
    assert res["n"] == 25 and res["rows"] == {"in_file": 30, "scored": 25, "not_decidable": 5,
                                              "not_decidable_left_out": True}
    assert res["all_rows"]["n"] == 30 and res["all_rows"]["accuracy"] < res["accuracy"]["value"]
    both = ev.analyze(rows, 1.0, "given", draws=50, include_undecidable=True)
    assert both["n"] == 30 and both["accuracy"]["value"] == res["all_rows"]["accuracy"]
    ev.report(res)


def test_baselines_leave_undecidable_rows_out(tmp_path):
    kept, _ = brt.build_rows({"ats_t1": CARD}, {"ats_t1": POSTING})
    rows, test = tmp_path / "rows.jsonl", tmp_path / "test.jsonl"
    brt.write_jsonl(rows, kept)
    brt.write_jsonl(test, [{"pair_id": r["pair_id"], "decidable": i != 0} for i, r in enumerate(kept)])
    assert len(baselines.load_rows(rows, test=test)) == len(kept) - 1
    assert len(baselines.load_rows(rows, True, test=test)) == len(kept)
    assert len(baselines.load_rows(rows, test=tmp_path / "none.jsonl")) == len(kept)
    # a row with no rendered resume is in rows.jsonl, not test.jsonl: the model never sees it
    brt.write_jsonl(test, [{"pair_id": r["pair_id"], "decidable": True} for r in kept[1:]])
    assert [r["pair_id"] for r in baselines.load_rows(rows, test=test)] == [r["pair_id"] for r in kept[1:]]
    res = baselines.evaluate(kept, baselines.constant_preds(kept)["always_skip"], 50)
    assert set(res["side_headlines"]) == {"without_added_keep_kinds", "resume_kinds"}


def test_ece_breaks_ties_like_pick():
    probs = np.array([[0.5, 0.0, 0.5]])  # Keep and Skip tie: pick() says Skip
    assert ev.top(probs).tolist() == [2]
    assert ev.ece(probs, np.array([2])) == pytest.approx(0.5)  # a hit at confidence 0.5
    assert ev.ece(probs, np.array([0])) == pytest.approx(0.5)  # a miss at confidence 0.5


def test_batch_diff_flags_a_score_that_moves():
    import local_eval
    a = [{"pair_id": "p1", "scores": [1.0, 0.0, 0.0], "pred": "Keep"}, {"pair_id": "p2", "scores": [0.0, 2.0, 0.0], "pred": "Explore"}]
    assert local_eval.batch_diff(a, [dict(r) for r in a])["ok"]
    b = [dict(a[0], scores=[1.0005, 0.0, 0.0]), dict(a[1], scores=[0.0, 0.0, 3.0], pred="Skip")]
    res = local_eval.batch_diff(a, b)
    assert not res["ok"] and res["pred_flips"] == ["p2"] and res["over_tol"] == 1


def test_decide_run_picks_batch_and_whether_the_jobs_fit():
    import local_eval
    rows = [{"pair_id": f"p{i}", "scores": [1.0, 0.0, 0.0], "pred": "Keep", "forward_s": s}
            for i, s in enumerate((9.0, 1.0, 1.0, 1.2, 0.9))]  # the first row carries the warm-up
    b1 = {"rows": rows}
    b4 = {"rows": [dict(r, forward_s=0.5) for r in rows]}
    res = local_eval.decide_run(b1, b4, forwards=1000, jobs=2, seconds_left=1000, load_s=100)
    assert res["batch"] == 4 and res["seconds_per_row"] == 0.5 and res["need_s"] == 700 and res["fits"]
    # batch 4 moves a score: batch 1 and its rate (median 1.0 s, not the 9 s warm-up)
    moved = {"rows": [dict(r, scores=[0.0, 0.0, 3.0], pred="Skip") for r in b4["rows"]]}
    res = local_eval.decide_run(b1, moved, 1000, 2, 1000, 100)
    assert res["batch"] == 1 and res["seconds_per_row"] == 1.0 and not res["fits"]
    # out of memory at batch 4, or at batch 1
    assert local_eval.decide_run(b1, None, 10, 1, 1000)["batch"] == 1
    assert not local_eval.decide_run(None, None, 10, 1, 1000)["fits"]


def test_an_associate_student_skip_needs_the_text_to_name_a_bachelor_degree():
    assoc = replace(in_window(CARD), degree_level="associate")
    assert label(assoc, CARD)[0] == "Excluded"
    # a0f4b708, bae50bf6, Northrop fe413853 (its only "Bachelor's" is a pay tier): an associate degree is
    # an undergraduate degree
    for text in ("Enrolled in an accredited full-time undergraduate or graduate degree program required.",
                 "- Pursuing an undergraduate degree in Physics.\n- Currently at the junior or senior undergraduate level.",
                 "- Be a student pursuing an undergraduate or graduate degree.\nBachelor’s: $26.75 to $33.50 per hour"):
        why = brt.guard("enrollment_level", assoc, CARD, "Excluded", "Title: Intern\n" + text)
        assert why and "bachelor" in why, text
    for text in ("Pursuing a bachelor's degree in Physics.", "Pursuing a B.S. in Physics.",
                 "Students pursuing a four-year undergraduate or advanced degree."):
        assert brt.guard("enrollment_level", assoc, CARD, "Excluded", "Title: Intern\n" + text) is None, text
    # "needs a bachelor degree or higher; you have an associate" on a master posting that never says bachelor
    master = replace(CARD, enrolled_in="master", degree_level="bachelor")
    why = brt.guard("degree_level", assoc, master, "Excluded", "Title: Intern\nCurrently pursuing a M.S. or Ph.D.")
    assert why and "bachelor" in why
    assert brt.guard("degree_level", assoc, master, "Excluded",
                     "Title: Intern\nPursuing a Master's degree; a Bachelor's degree completed.") is None
    # "MS Word" is not a master degree
    assert brt.unnamed_levels("Pursuing a degree. Skills: MS Word.", ["for students currently pursuing a master's "
                                                                      "degree or higher; you are not a student"]) == ["master"]
    # a graduated candidate is not pursuing anything: that Skip does not hinge on the level word
    assert brt.unnamed_levels("Pursuing an undergraduate degree.",
                              ["for students currently pursuing a bachelor's degree or higher; you have graduated"]) == []


def test_graduated_skip_beside_an_or_later_line_the_candidate_meets_is_dropped():
    grad = replace(in_window(CARD), graduation_year=2025)
    shown = "Title: Intern\nCurrently pursuing a bachelor's degree. Expected graduation date of Fall {} or later."
    assert "or later" in brt.guard("enrollment_graduated", grad, CARD, "Excluded", shown.format(2025))
    assert brt.guard("enrollment_graduated", grad, CARD, "Excluded", shown.format(2027)) is None


def test_a_sponsorship_skip_needs_a_sponsorship_refusal_not_a_sponsored_benefit():
    card = replace(CARD, sponsorship="will_not_sponsor")
    h1b = replace(in_window(card), work_auth="needs_h1b_now")
    assert label(h1b, card)[0] == "Excluded"
    mckesson = POSTING + "\n- No company-sponsored housing, relocation, student visa or Green Card assistance available"
    assert "sponsor" in brt.guard("work_auth_needs_h1b", h1b, card, "Excluded", mckesson)
    assert brt.guard("work_auth_needs_h1b", h1b, card, "Excluded",
                     POSTING + "\nWe are unable to sponsor an employment visa for this role.") is None


def test_header_only_location_skip_on_a_hybrid_card_is_dropped():
    far = replace(in_window(CARD), current_location="Denver, CO")
    assert label(far, CARD)[0] == "Excluded"
    header_only = "Title: Data Intern\nCompany: Acme\nLocations: Boston, MA\n\nPursuing a bachelor's degree."
    assert "hybrid card" in brt.guard("location", far, CARD, "Excluded", header_only)
    assert brt.guard("location", far, CARD, "Excluded", POSTING) is None
    assert brt.guard("location", far, replace(CARD, work_mode="onsite"), "Excluded", header_only) is None


def test_graduated_skip_beside_or_equivalent_experience_needs_a_line_that_shuts_out_graduates():
    grad = replace(in_window(CARD), graduation_year=2025)
    # Qumulo c2cbd149: "or equivalent experience" and no graduation window or already-graduated line
    qumulo = ("Title: Intern\n- Pursuing a BS or BA in Computer Science or Computer Engineering or equivalent experience, "
              "available for a full-time internship during Winter, Spring, or Summer 2027")
    assert "shuts out graduates" in brt.guard("enrollment_graduated", grad, CARD, "Excluded", qumulo)
    # Astranis 25f578d9: an explicit line for those who have graduated
    astranis = ("Title: Intern\nInternships are designed for students who are currently enrolled at a four-year university.\n"
                "If you have already graduated from a four-year university, please apply to be an Associate Engineer.\n"
                "- Currently pursuing a B.S. in electrical or computer engineering (or equivalent technical degree)")
    assert brt.guard("enrollment_graduated", grad, CARD, "Excluded", astranis) is None
    # DoorDash 66fcf084: a future graduation window the 2025 graduate misses
    doordash = ("Title: Intern\n- You are pursuing a B.S. or M.S. Degree in Computer Science or equivalent, and are "
                "graduating between Fall 2027 to Summer 2028.")
    assert brt.guard("enrollment_graduated", grad, CARD, "Excluded", doordash) is None


def test_degree_field_skip_on_a_related_discipline_line_the_skills_name_is_dropped():
    card = replace(CARD, degree_level="doctoral", enrolled_in="none", degree_fields=("Machine Learning", "Computer Science"))
    base = in_window(card)
    theology = replace(base, degree_field="theology", skills=(*base.skills, "Machine Learning"))
    assert label(theology, card)[0] == "Excluded"
    line = "Title: Scientist\n- Ph.D. in Machine Learning, Computer Science, or a related discipline."
    assert "related field" in brt.guard("degree_field", theology, card, "Excluded", line)
    assert brt.guard("degree_field", replace(theology, skills=base.skills), card, "Excluded", line) is None
    assert brt.guard("degree_field", theology, card, "Excluded", line.replace(", or a related discipline", "")) is None


def test_must_skill_named_only_under_an_ideal_candidate_heading_is_dropped():
    card = replace(CARD, must_skills=(("Python",), ("Linux", "UNIX")))
    cand = replace(in_window(card), skills=("Python", "SQL"))
    assert label(cand, card)[0] == "Excluded"
    lines = "\n- Experience with Linux or UNIX-like operating systems.\n- Experience with Python."
    assert "ideal-candidate" in brt.guard("must_skill", cand, card, "Excluded", "Title: Intern\nThe ideal candidate has:" + lines)
    assert brt.guard("must_skill", cand, card, "Excluded", "Title: Intern\nRequirements:" + lines) is None


def test_circuit_design_implies_bench_equipment():
    held = ("Analog, digital and power circuit design", "Microcontrollers")
    assert brt.implied_by_held("Test and Measurement Equipment", held) == held[0]
    assert brt.implied_by_held("Test and Measurement Equipment", ("Python",)) is None


def test_location_skip_whose_text_says_only_remote_is_dropped():
    onsite = replace(CARD, work_mode="onsite")
    far = replace(in_window(onsite), current_location="Denver, CO")
    head = "Title: Intern\nCompany: Acme\nLocations: Boston, MA\n\n"
    # Rundoo 21ae5c99: the only mode words are the remote interview steps
    assert "remote" in brt.guard("location", far, onsite, "Excluded", head + "- Coding Interview (remote): with an engineer.")
    assert brt.guard("location", far, onsite, "Excluded", head + "Coding interview (remote). The role is on-site.") is None
    # Hy-Vee 0a09d266: "located at the West Des Moines corporate office" backs the Skip
    assert brt.guard("location", far, onsite, "Excluded", head + "The internship is located at the corporate office.") is None
    assert brt.guard("location", far, onsite, "Excluded", head + "Lidar remote sensing research.") is None


def test_overqualified_zero_asked_beside_a_years_range_is_dropped():
    over = replace(in_window(CARD), years=8)
    marsh = POSTING + " Ideally you will have 2-3 years’ work experience prior to obtaining your master's degree."
    assert "range of years" in brt.guard("overqualified", over, CARD, "Explore", marsh)
    assert brt.guard("overqualified", over, CARD, "Explore", POSTING + " A company with 60 years of experience.") is None


def test_every_variant_kind_is_known():
    """Each kind variants.py can make is in the builder's VARIANT_KINDS, so the coverage report, the core
    lists and the added-kind headline see it (the v3/rules merge added about twenty kinds)."""
    import matcher.variants as variants

    src = Path(variants.__file__).read_text()
    made = set(re.findall(r'\badd\("([a-z_]+)"', src)) | set(re.findall(r'\bstatus\("([a-z_]+)"', src))
    made |= {f"work_auth_citizens_only_{s}" for s in ("f1_cpt", "f1_opt", "stem_opt", "needs_h1b_now")}
    made |= {"full_match", "years_one_short", "years_far_short"}
    assert made <= set(brt.VARIANT_KINDS), sorted(made - set(brt.VARIANT_KINDS))
    for card in (CARD, replace(CARD, enrolled_in="none", degree_level="bachelor", min_years=3, employment_type="full_time",
                               seniority="entry", sponsorship="sponsors"),
                 replace(CARD, citizens_only=True, work_mode="remote")):
        for v in brt.make_variants(card, 0, base=brt.full_match(card)):
            assert v.kind in brt.VARIANT_KINDS, v.kind


K2_DEFINITION = ("As defined in the ITAR, “U.S. Persons” include U.S. citizens, lawful permanent residents (i.e., Green "
                 "Card holders), and certain protected individuals (e.g., refugees/asylees, American Samoans).")
K2_ASK = ("The person hired for this role will have access to information and items controlled by U.S. export control "
          "regulations (ITAR). The person hired for this role must therefore either be a “U.S. person” as defined by "
          "22 C.F.R. § 120.15 or otherwise eligible for a federally issued export control license.")


def test_a_citizenship_skip_needs_a_shown_line_that_asks_for_the_status():
    ask = brt.citizen_requirement_shown
    # K2 Space 6fe74a2d: the definition alone asks nothing; the "must therefore be a U.S. person" line does
    assert ask(f"Export Compliance\n{K2_DEFINITION}\n{K2_ASK}") and not ask(f"Export Compliance\n{K2_DEFINITION}")
    # General Motors 0231a390, Astranis e6fe2e61, RTX 27282ffa, Booz Allen c0d0a7a8, Marsh 73d1f84a
    assert ask("The position is subject to export control restrictions and requires the successful candidate to be a "
               "U.S. Person (U.S. citizen, U.S. permanent resident, asylee or refugee).")
    assert ask("U.S. Citizenship, Lawful Permanent Residency, or Refugee/Asylee Status Required")
    assert ask("U.S. Citizen, U.S. Person, or Immigration Status Requirements:\nU.S. citizenship is required, as only U.S. "
               "citizens are authorized to access information under this program/contract.")
    assert ask("You Have:\nAbility to obtain a Secret clearance")
    assert ask("- You must have legal authorization to work in the U.S. on a permanent basis without the need for "
               "sponsorship now or in the future")
    # Federal Reserve aaeb44bf: a bare bullet under a requirements heading
    assert ask("Requirements:\n- U.S. Citizen or Permanent Residency") and not ask("About us:\n- U.S. Citizen or Permanent Residency")
    # not asks: an EEO line, USCIS by name, an empty clearance field
    assert not ask("All employment decisions are made without regard to sex, race, national origin, citizenship status.")
    assert not ask("E-Verify is operated by the U.S. Citizenship and Immigration Services and is required by law.")
    assert not ask("CLEARANCE REQUIRED FOR START: No")


def test_an_onsite_explore_needs_the_text_to_state_the_mode_not_a_perk():
    shown = brt.onsite_mode_shown
    # Axos a1902097, Trimble bee2e809, Rundoo 21ae5c99, Cboe dd3566b3: a perk, an errand or an event is onsite
    assert not shown("- Access to Onsite Gym and Walking Distance from UTC for San Diego Interns")
    assert not shown("- Travel Requirement: Up to 10% (for on-site field visits with farmers and dealers)")
    assert not shown("- Meals: Free daily in-office lunch (~$1,000/3 months)\n- Our office is in Redwood City, CA")
    assert not shown("- Daily complimentary in-office lunch\n- Monthly in office networking events and happy hours")
    assert not shown("- Daily lunch catered onsite, complimentary barista-made drinks")
    # the mode stated
    assert shown("This role requires that you be on-site at our HQ in San Mateo, CA.")
    assert shown("The work model for the role is onsite in New Berlin, Wisconsin.")
    assert shown("This in-person role is based in Clarksburg, MD.")
    assert shown("- Available to work full-time onsite at the HQ\n- Onsite fitness center to keep you fresh")


def test_a_citizenship_skip_is_unreadable_when_the_ask_is_not_shown():
    card = replace(CARD, citizens_only=True)
    cand = replace(in_window(card), work_auth="f1_opt")
    bucket, reasons, _ = label(cand, card)
    assert bucket == "Excluded" and any(brt.CITIZEN_REASON in r for r in reasons)
    full = f"{POSTING}\nExport Compliance\n{K2_DEFINITION}\n{K2_ASK}"
    cut = f"{POSTING}\nExport Compliance\n{K2_DEFINITION}"
    assert not [m for m in brt.posting_misses(cand, card, reasons, "", full, hard=True) if m[0] == "work_auth"]
    assert ("work_auth", brt.CITIZEN_MISS) in brt.posting_misses(cand, card, reasons, "", cut, hard=True)


# ---------------------------------------------------------------- class-level wording (integration review)

def _class_posting(line: str, title: str = "Software Engineering Intern - Summer 2027") -> str:
    return (f"Title: {title}\nCompany: Acme\nLocations: Boston, MA\n\nRequired:\n- Pursuing a bachelor's degree in "
            f"Computer Science.\n- {line}\n- Python and SQL.\nThe role is hybrid in Boston, MA.")


CLASS_CARD = replace(CARD, title="Software Engineering Intern", salary_min=None, salary_max=None, industry="")


@pytest.mark.parametrize("line,years,grads_ok", [
    # CACI ats_a82e8f66: a rising junior or senior after the Summer 2027 internship graduates 2029 or 2028
    ("Must be a Rising Junior/Senior in a bachelor's program or a graduate student", {2028, 2029}, True),
    # AMA ats_bae50bf6: a junior or senior now (fall 2026) graduates 2028 or 2027
    ("Currently at the junior or senior undergraduate level.", {2027, 2028}, False),
    # ats_7b0bc5f5
    ("Currently enrolled in a Bachelor's degree as a Junior or Senior student, or recent graduate", {2027, 2028}, False),
    # Voloridge ats_d429d89b: three years done now, the stricter of its two readings
    ("Completion of at least three years of an undergraduate or graduate degree program in Math", {2027}, True),
    # ats_792ba5ee: sophomore year done by the summer
    ("Sophomore year must be complete by the start of the summer internship program for undergraduate candidates.",
     {2027, 2028, 2029}, True),
])
def test_class_level_wording_gives_a_graduation_window(line, years, grads_ok):
    assert brt.class_window(_class_posting(line)) == (years, grads_ok)


def test_class_words_outside_a_requirement_name_no_window():
    assert brt.class_window(_class_posting("Present recommendations to senior management")) is None
    assert brt.class_window(_class_posting("Software Engineering Intern/Junior/Senior: $34.00/hour")) is None
    assert brt.class_window(_class_posting("Completion of junior year preferred")) is None


@pytest.mark.parametrize("line,year", [
    ("Must be a Rising Junior/Senior in a bachelor's program or a graduate student", 2029),
    ("Currently at the junior or senior undergraduate level.", 2028),
])
def test_class_level_fits_the_full_match_and_drops_what_falls_outside(line, year):
    """The CACI and AMA rows kept 2030 graduates (rising sophomores, freshmen now) as Keep and Explore."""
    text = _class_posting(line)
    raw, fitted, notes = brt.base_candidate(CLASS_CARD, text)
    assert raw.graduation_year == 2030 and fitted.graduation_year == year
    assert any("class level" in n for n in notes)
    kept, dropped = brt.build_rows({"ats_t1": CLASS_CARD}, {"ats_t1": text})
    for r in kept:
        c = r["candidate"]
        if r["bucket"] != "Excluded" and c["degree_level"] == "bachelor" and c["graduation_year"] > 2026:
            assert c["graduation_year"] in brt.class_window(text)[0], (r["variant_kind"], c["graduation_year"])
    assert brt.guard("auth_f1_cpt", replace(fitted, graduation_year=2030, work_auth="f1_cpt"), CLASS_CARD,
                     "For you", brt.shown_posting(text))


def test_years_done_wording_drops_a_sophomore_cpt_keep():
    """Voloridge ats_d429d89b kept a CPT student graduating 2029 as Keep against 'at least three years'."""
    text = _class_posting("Completion of at least three years of an undergraduate or graduate degree program in Math")
    card = replace(CLASS_CARD, enrolled_in="none")
    cpt = replace(brt.full_match(card), work_auth="f1_cpt", graduation_year=2029, years=0)
    assert label(cpt, card)[0] == "For you"
    assert "class level" in brt.guard("auth_f1_cpt", cpt, card, "For you", brt.shown_posting(text))
    assert brt.guard("auth_f1_cpt", replace(cpt, graduation_year=2027), card, "For you", brt.shown_posting(text)) is None


def test_degree_deadline_drops_the_still_enrolled_explore():
    """Merck ats_61dda24d: 'Receive a Ph.D. no later than spring 2027' takes no doctoral student finishing in 2031,
    and the rule's 'still enrolled' Explore cannot read the deadline at all."""
    text = ("Title: Postdoctoral Fellow\nCompany: Acme\nLocations: Boston, MA\n\nRequired:\n- Must currently hold a PhD OR "
            "Receive a Ph.D. no later than spring 2027\n- Python.")
    assert brt.degree_deadline(text) == 2027
    card = replace(CARD, title="Postdoctoral Fellow", degree_level="doctoral", enrolled_in="none",
                   employment_type="full_time", salary_min=None, salary_max=None, industry="")
    late = replace(brt.full_match(card), graduation_year=2031)
    assert "2027" in brt.guard("degree_still_enrolled", late, card, "Explore", brt.shown_posting(text))
    assert "2027" in brt.guard("must_skill_alternative", late, card, "For you", brt.shown_posting(text))


def test_salary_explore_needs_no_higher_figure_in_the_text():
    """Rundoo ats_21ae5c99: 'pay tops out at $98,000' beside '$98k base; $111k OTE' with a $110,000 floor."""
    card = replace(CARD, salary_min=98_000, salary_max=98_000)
    cand = replace(in_window(card), salary_floor=110_000)
    text = POSTING.replace("Pay: $60,000 - $70,000 per year.", "Base Salary: $98k base; $111k OTE")
    assert brt.money_figures(text) == [98_000, 111_000]
    assert "pay figure" in brt.guard("salary", cand, card, "Explore", brt.shown_posting(text))
    assert brt.guard("salary", cand, card, "Explore", brt.shown_posting(text.replace("$111k OTE", "bonus"))) is None


def test_real_test_rows_fit_the_class_level_when_built():
    rows = Path(brt.OUT / "rows.jsonl")
    if not rows.exists():
        pytest.skip("run build_real_test.py build first")
    posts, cards = brt.load_postings(), brt.load_test_cards()
    windows = {k: brt.class_window(brt.shown_posting(posts[k])) for k in cards}
    for r in brt.read_jsonl(rows):
        c = CandidateCard.from_dict(r["candidate"])
        if r["bucket"] != "Excluded":
            assert brt.class_problem(c, brt.shown_posting(posts[r["base_rq_id"]])) is None, r["pair_id"]
            assert r["variant_kind"] != "degree_still_enrolled" or brt.degree_deadline(
                brt.shown_posting(posts[r["base_rq_id"]])) is None, r["pair_id"]
    assert sum(w is not None for w in windows.values()) >= 5


def test_junior_and_senior_staff_name_no_class_level():
    assert brt.class_window(_class_posting("Pair with junior and senior engineers on code reviews.")) is None
    assert brt.class_window(_class_posting("Learn from junior/senior team members")) is None
