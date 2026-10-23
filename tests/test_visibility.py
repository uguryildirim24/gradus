"""Candidate profile rendering and the fact-visibility checker."""

from dataclasses import replace

from matcher import full_match, label, make_variants
from matcher.profile import render_profile
from matcher.schema import CandidateCard, RequirementCard
from matcher.visibility import has_money, has_phrase, has_years, missing_facts

CARD = RequirementCard(
    source_posting_id="p1", title="Lab Technician", field="Lab and life sciences", industry="biotech and pharma",
    degree_level="bachelor", degree_fields=("Biology", "Chemistry"), min_years=2,
    must_skills=(("Biology",), ("Microsoft Excel", "Google Sheets")), nice_skills=("Critical Thinking",),
    license="Phlebotomy certification", locations=("Boston, MA",), work_mode="onsite", sponsorship="will_not_sponsor",
    salary_min=50_000, salary_max=70_000, employment_type="full_time",
)
POSTING = (
    "Lab Technician\nBrightway Labs is a biotech and pharma company in Boston, MA. This is a full-time, onsite role.\n"
    "Requirements: bachelor's degree in Biology or Chemistry; at least 2 years of experience; knowledge of Biology; "
    "Microsoft Excel or Google Sheets; an active Phlebotomy certification.\nNice to have: Critical Thinking.\n"
    "Pay: $50,000 - $70,000 per year. We are unable to sponsor employment visas now or in the future."
)
RESUME = (
    "Candidate\nBoston, MA\nLab technician with 4 years of experience.\nSkills: Biology, Microsoft Excel, "
    "Critical Thinking\nLicenses: Phlebotomy certification\nB.S. in Biology, 2022"
)


def test_profile_has_every_preference_and_no_nuisance_fields():
    c = CandidateCard(
        field="x", work_auth="stem_opt", current_location="Boston, MA", willing_locations=("Denver, CO", "Austin, TX"),
        remote_preference="hybrid_or_remote", salary_floor=65_000, employment_types_wanted=("full_time", "contract"),
        industries_wanted=("healthcare",), industries_avoided=("retail",), roles_avoided=("sales",),
        education_country="Turkey", language="Turkish",
    )
    p = render_profile(c)
    for fact in ("STEM OPT", "Boston, MA", "Denver, CO; Austin, TX", "hybrid or remote only", "full-time, contract",
                 "$65,000", "healthcare", "retail", "sales"):
        assert fact in p, fact
    assert "Turkey" not in p and "Turkish" not in p
    blank = render_profile(CandidateCard())
    assert "Job types wanted: any" in blank and "Minimum salary: not set" in blank and "Roles to avoid: none" in blank


def test_text_probes():
    assert has_phrase("Proficient in microsoft-excel.", "Microsoft Excel")
    assert has_phrase("C++ and C#", "C++") and not has_phrase("Biology", "Bio")
    assert has_phrase("Current CNA (Certified Nursing Assistant) certification", "CNA certification")
    assert has_years("at least three (3) years of", 3) and has_years("5+ years", 5) and not has_years("GPA 3.8", 3)
    assert has_money("$50,000 - $70,000", 70_000) and has_money("$70K", 70_000) and not has_money("$75,000", 70_000)


def test_full_match_is_fully_visible():
    c = replace(full_match(CARD), years=4)
    b, reasons, _ = label(c, CARD)
    assert b == "For you"
    assert missing_facts(c, CARD, reasons, RESUME, render_profile(c), POSTING) == []


def test_missing_industry_and_stance_are_reported():
    c = replace(full_match(CARD), industries_wanted=("healthcare",), work_auth="needs_h1b_now")
    _, reasons, _ = label(c, CARD)
    posting = POSTING.replace("a biotech and pharma company", "a company").replace(
        "We are unable to sponsor employment visas now or in the future.", "")
    cats = {k for k, _ in missing_facts(c, CARD, reasons, RESUME, render_profile(c), posting)}
    assert cats == {"work_auth"}  # Excluded: only the hard reason counts
    _, reasons, _ = label(replace(c, work_auth="citizen_or_pr"), CARD)
    cats = {k for k, _ in missing_facts(replace(c, work_auth="citizen_or_pr"), CARD, reasons, RESUME,
                                        render_profile(c), posting)}
    assert cats == {"industry"}


def test_skill_named_in_resume_but_missing_is_a_contradiction():
    # a Chemistry graduate: a Biology degree would itself show the Biology skill
    c = replace(full_match(CARD), degree_field="Chemistry", skills=("Microsoft Excel", "Critical Thinking"))
    b, reasons, _ = label(c, CARD)
    assert b == "Excluded"
    cats = {k for k, _ in missing_facts(c, CARD, reasons, RESUME, render_profile(c), POSTING)}
    assert cats == {"skill_contradiction"}  # RESUME still lists Biology


def test_every_variant_reason_has_an_owner():
    for v in make_variants(CARD):
        _, reasons, _ = label(v.candidate, v.card)
        miss = missing_facts(v.candidate, v.card, reasons, RESUME, render_profile(v.candidate), POSTING)
        assert not [m for m in miss if m[0] == "unknown_reason"], (v.kind, miss)


def test_unrestricted_authorization_is_a_stance_contradiction():
    c = replace(full_match(CARD), work_auth="f1_opt")
    b, reasons, _ = label(c, CARD)
    assert b == "Explore"
    posting = POSTING + " Applicants must already possess unrestricted work authorization."
    cats = {k for k, _ in missing_facts(c, CARD, reasons, RESUME, render_profile(c), posting)}
    assert cats == {"work_auth"}


def test_profile_states_degree_graduation_and_years():
    from matcher.profile import degree_line

    c = CandidateCard(degree_level="bachelor", degree_field="Computer Science", graduation_year=2027, years=1)
    p = render_profile(c)
    assert "Currently pursuing: Bachelor's degree in Computer Science, expected 2027" in p
    assert "Highest degree: Bachelor's degree in Computer Science, graduated 2026" in \
        render_profile(replace(c, graduation_year=2026))
    assert "Years of work experience: 1" in p
    assert degree_line(replace(c, graduation_year=2026)) == "Bachelor's degree in Computer Science, graduated 2026"
    assert degree_line(CandidateCard(degree_level="none")) == "no college degree"
    assert degree_line(CandidateCard(degree_level="doctoral", degree_field="Biology", graduation_year=2020)) == \
        "Doctorate in Biology, graduated 2020"


STUDENT_CARD = RequirementCard(
    source_posting_id="p2", title="Research Intern", field="Lab and life sciences", industry="biotech and pharma",
    degree_level="none", enrolled_in="bachelor", degree_fields=("Biology", "Biochemistry", "related field"),
    must_skills=(("Laboratory techniques",), ("Python", "R")), nice_skills=("Microsoft Excel", "PCR", "Statistics"),
    locations=("Boston, MA",), work_mode="onsite", sponsorship="unstated", salary_min=40_000, salary_max=50_000,
    employment_type="internship",
)
OFFICE_CARD = RequirementCard(
    source_posting_id="p3", title="Office Coordinator", field="Office and administrative", industry="Real Estate",
    degree_level="associate", degree_fields=("Business", "Management"), min_years=3,
    must_skills=(("Microsoft Office",), ("Programming",), ("Communication",)),
    nice_skills=("SQL", "Human resources", "Design"), locations=("California",), work_mode="hybrid",
    sponsorship="sponsors", e_verify=True, salary_min=55_000, salary_max=65_000, employment_type="full_time",
)


def test_synthetic_full_rendering_leaves_nothing_missing():
    """Texts that state every card fact (tests/synthetic_text.py): no crash and nothing reported, for
    every variant of a degree-free student role (KeyError 'none' before), a bachelor's role and an
    associate role whose skills are met through Office apps, a language and the degree field."""
    from synthetic_text import posting_text, resume_text

    for card in (STUDENT_CARD, CARD, OFFICE_CARD):
        variants = make_variants(card)
        assert len(variants) > 10, card.title
        for v in variants:
            _, reasons, _ = label(v.candidate, v.card)
            miss = missing_facts(v.candidate, v.card, reasons, resume_text(v.candidate), render_profile(v.candidate),
                                 posting_text(v.card))
            assert miss == [], (card.title, v.kind, reasons, miss)


def test_student_role_degree_reasons_are_checked():
    from synthetic_text import posting_text, resume_text

    c = full_match(STUDENT_CARD)
    b, reasons, _ = label(c, STUDENT_CARD)
    assert b == "For you" and any(x.startswith("studying ") for x in reasons)
    resume = resume_text(c).replace(c.degree_field, "Undeclared")
    cats = missing_facts(c, STUDENT_CARD, reasons, resume, render_profile(c), posting_text(STUDENT_CARD))
    assert ("degree", f"resume does not show the degree field {c.degree_field}") in cats


def test_industry_probe_reads_and_for_ampersand_and_a_singular_last_word():
    from matcher.visibility import has_industry, industry_forms

    assert has_industry("Join a leading hospital and health care provider.", "Hospital & Health Care")
    assert has_industry("We are a semiconductor company.", "Semiconductors")
    assert has_industry("A regional utility company.", "utilities")
    assert has_industry("A biotech & pharma firm.", "biotech and pharma")
    assert has_industry("A medical devices maker.", "Medical Devices")
    assert not has_industry("A hospital company.", "Hospital & Health Care")
    assert not has_industry("We build software.", "Semiconductors")
    assert industry_forms("Glass") == {"Glass"}  # a final 'ss' is not a plural


def test_enrollment_in_a_degree_program_says_the_role_is_for_students():
    # rq_01285: "Enrollment in a bachelor's or higher degree program" under the minimum qualifications
    from synthetic_text import posting_text, resume_text

    c = full_match(STUDENT_CARD)
    _, reasons, _ = label(c, STUDENT_CARD)
    old = "Open to students currently enrolled in a bachelor's program."
    assert old in posting_text(STUDENT_CARD)
    gap = ("enrollment", "posting does not say it is for current students")
    for new, missing in (("Requirements: enrollment in a bachelor's or higher degree program in any academic field.", False),
                         ("Enrollment in our benefits program starts on day one.", True),
                         ("Enrollment in our new graduate program is open.", True)):
        ps = posting_text(STUDENT_CARD).replace(old, new)
        cats = missing_facts(c, STUDENT_CARD, reasons, resume_text(c), render_profile(c), ps)
        assert (gap in cats) is missing, new


def test_citizens_only_card_without_a_stance_may_say_it_does_not_sponsor():
    # rq_01334~citizens (r11, r13 check renders): citizens-only already means no sponsorship
    from matcher.visibility import stance_conflict
    no = "Only U.S. citizens or permanent residents may apply. We do not sponsor visas for this role."
    assert stance_conflict({"sponsorship": "unstated", "citizens_only": True}, no) is None
    assert stance_conflict({"sponsorship": "unstated", "citizens_only": False}, no) is not None
    assert stance_conflict({"sponsorship": "sponsors", "citizens_only": True}, no) is not None
    # an OPT refusal is still the will_not_sponsor_or_opt stance's alone
    assert stance_conflict({"sponsorship": "unstated", "citizens_only": True}, "We cannot accept OPT.") is not None


def test_citizens_only_labels_do_not_read_the_sponsorship_stance():
    """Every citizens-only card labels the same whatever its stance, so 'no sponsorship' is implied."""
    import dataclasses
    import random
    from matcher.rules import label
    from matcher.schema import WORK_AUTH, CandidateCard, RequirementCard
    rng = random.Random(0)
    q = RequirementCard(source_posting_id="x:1", title="Analyst", must_skills=(("SQL",),), citizens_only=True, sponsorship="unstated")
    for auth in WORK_AUTH:
        c = CandidateCard(skills=("SQL",), work_auth=auth, graduation_year=rng.choice((2020, 2028)))
        for s in ("will_not_sponsor", "will_not_sponsor_or_opt", "sponsors"):
            assert label(c, q) == label(c, dataclasses.replace(q, sponsorship=s)), (auth, s)


def test_a_degree_field_reason_needs_every_listed_field_in_the_posting():
    """pr_4023ff03fb20 (RTX): the Skip quoted six fields; the posting said only 'an applicable discipline' and had
    'Software Engineering' in its title, and the old any-field probe passed. Every listed field must be named, its
    'or a related field' tail aside, also in a list that shares the head word ('Electrical or Computer Engineering')."""
    from matcher.visibility import field_core, field_shown

    card = replace(CARD, degree_fields=("Computer Science", "Software Engineering", "Mathematics"))
    c = replace(full_match(card), degree_field="Business Administration")
    b, reasons, _ = label(c, card)
    assert b == "Excluded" and any(r.startswith("degree field should be") for r in reasons)
    resume = RESUME.replace("Biology, 2022", "Business Administration, 2022")
    title_only = POSTING.replace("Lab Technician\n", "Software Engineering Intern\n").replace(
        "in Biology or Chemistry", "in an applicable discipline")
    miss = missing_facts(c, card, reasons, resume, render_profile(c), title_only)
    assert any(k == "degree" and "Computer Science" in d and "Mathematics" in d for k, d in miss), miss
    named = title_only.replace("an applicable discipline", "Computer Science, Software Engineering or Mathematics")
    assert not [m for m in missing_facts(c, card, reasons, resume, render_profile(c), named) if m[0] == "degree"]
    for entry, core in (("Biology (or a related field)", "Biology"), ("Chemistry or a closely related discipline", "Chemistry"),
                        ("Physics, or equivalent", "Physics"), ("Computer Science", "Computer Science")):
        assert field_core(entry) == core
    for text in ("Pursuing a B.S. in electrical or computer engineering (or equivalent technical degree)",
                 "a degree in Chemical or Biomedical Engineering", "Electrical/Computer Engineering majors"):
        assert field_shown(text, "Electrical Engineering" if "lectrical" in text else "Chemical Engineering"), text
    assert not field_shown("in the field of Data, Analytics, & Automation", "Computer Science")
    assert not field_shown("an applicable discipline. Electrical safety is key.", "Electrical Engineering")


def test_an_enrollment_reason_needs_the_profile_education_line():
    from synthetic_text import posting_text, resume_text

    c = full_match(STUDENT_CARD)
    b, reasons, _ = label(c, STUDENT_CARD)
    enrol = [r for r in reasons if r.startswith("currently pursuing")]
    assert enrol
    pf = render_profile(c)
    assert not missing_facts(c, STUDENT_CARD, enrol, resume_text(c), pf, posting_text(STUDENT_CARD))
    cut = "\n".join(x for x in pf.splitlines() if not x.startswith(("Currently pursuing", "Highest degree")))
    assert ("enrollment", "profile lacks the education line") in missing_facts(c, STUDENT_CARD, enrol, resume_text(c), cut,
                                                                                posting_text(STUDENT_CARD))
