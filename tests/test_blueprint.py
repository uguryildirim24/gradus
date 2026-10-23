import json
import re
from dataclasses import fields
from pathlib import Path

import pytest

from matcher.blueprint import (
    BLUEPRINT,
    BLUEPRINT_V2,
    BLUEPRINT_V3,
    BLUEPRINT_VERSION,
    LOCATIONS_WORK_MODE_V3,
    MUST_SKILLS_V3,
    NICE_SKILLS_V3,
    UNIVERSAL,
    OUTPUT_KEYS,
    PIPELINE_KEYS,
    clean_card,
    clean_location,
    clean_skills,
    is_soft,
    is_vague_field,
    merge_anyof,
    parse_card,
    phrase_groups,
    posting_meta,
    training_ok,
    degree_segs,
    duty_only,
    pref_spans,
    _comma_or_scopes,
    _head_kind,
    repair_enrollment,
    repair_sponsorship,
    repair_vague,
    repair_work_auth,
    repair_work_mode,
    sections,
    snap_industry,
    where,
)
from matcher.sample import INDUSTRIES
from matcher.schema import (
    DEGREE_LEVELS,
    EMPLOYMENT_TYPES,
    SENIORITIES,
    SPONSORSHIP,
    WORK_MODES,
    RequirementCard,
)


def test_output_keys_are_the_card_fields():
    names = [f.name for f in fields(RequirementCard)]
    assert set(OUTPUT_KEYS) | set(PIPELINE_KEYS) == set(names)
    assert not set(OUTPUT_KEYS) & set(PIPELINE_KEYS)


def test_blueprint_defines_every_key_and_no_pipeline_key():
    for key in OUTPUT_KEYS:
        assert re.search(rf"\b{key} (\(|and salary_max \()", BLUEPRINT), key
    for key in PIPELINE_KEYS:
        assert key not in BLUEPRINT


@pytest.mark.parametrize("values", [DEGREE_LEVELS, SENIORITIES, SPONSORSHIP, WORK_MODES, EMPLOYMENT_TYPES])
def test_blueprint_lists_every_enum_value(values):
    for v in values:
        assert f'"{v}"' in BLUEPRINT, v


def test_live_blueprint_is_v2_and_unchanged():
    """The v3 draft was not adopted (eval/fix_cycle2.json): every pipeline still sends the v2 text that made b2b2ed8e."""
    import hashlib
    assert BLUEPRINT_VERSION == "v2" and "blueprint v2" in BLUEPRINT and BLUEPRINT_V2 is BLUEPRINT
    assert hashlib.sha1((BLUEPRINT + UNIVERSAL).encode()).hexdigest()[:8] == "b2b2ed8e"
    assert hashlib.sha1((BLUEPRINT_V3 + UNIVERSAL).encode()).hexdigest()[:8] == "d46a0614"  # the draft measured


def test_v3_draft_changes_only_the_skill_and_work_mode_lines():
    old, new = BLUEPRINT_V2.split("\n"), BLUEPRINT_V3.split("\n")
    assert len(old) == len(new)
    changed = [n.split(" (", 1)[0] for o, n in zip(old, new) if o != n]
    assert changed == ["Requirement card blueprint v3. Output exactly one JSON object and nothing else", "must_skills",
                       "nice_skills", "locations"]


def test_v3_draft_must_skills_are_only_what_the_posting_states_as_required():
    must = MUST_SKILLS_V3
    for phrase in ("states as REQUIRED", "duties and responsibilities", '"you will ..."', "even when it names a tool",
                   "a required area of experience", "certifications", "conditions of the job",
                   "a list of examples is ONE group of alternatives", "such as", "e.g.", "including"):
        assert phrase in must, phrase
    assert "List every concrete" not in must  # v2's line that turned every named subject into its own group
    assert "preferred" in NICE_SKILLS_V3 and "duties are not nice_skills" in NICE_SKILLS_V3
    assert "remote is one of the listed locations" in LOCATIONS_WORK_MODE_V3


def test_v3_instruction_names_no_posting_or_employer():
    """The draft is general: no employer of fresh test 1 (the set it was designed on) is named."""
    manifest = Path(__file__).resolve().parents[1] / "eval/fresh_test_manifest.json"
    employers = json.loads(manifest.read_text())["employers"]
    names = employers if isinstance(employers, list) else list(employers)
    text = BLUEPRINT_V3.casefold()
    assert names and not [e for e in names if len(e) > 3 and re.search(rf"\b{re.escape(e.casefold())}\b", text)]


EXAMPLE = {
    "title": "ML Research Intern", "field": "Machine learning", "seniority": "entry",
    "industry": "technology", "degree_level": "bachelor", "enrolled_in": "doctoral",
    "degree_fields": ["Computer Science"], "related_ok": False, "min_years": 0,
    "must_skills": [["Python"], ["PyTorch", "TensorFlow"]], "nice_skills": ["CUDA"],
    "license": None, "locations": ["New York, NY"], "work_mode": "hybrid",
    "sponsorship": "will_not_sponsor_or_opt", "citizens_only": False, "e_verify": False,
    "salary_min": 124800, "salary_max": 124800, "employment_type": "internship",
}


def test_parse_card_builds_a_valid_card():
    card = parse_card("ats_x", json.dumps(EXAMPLE))
    assert card.source_posting_id == "ats_x" and card.onet_soc is None
    assert card.must_skills == (("Python",), ("PyTorch", "TensorFlow"))
    assert set(EXAMPLE) == set(OUTPUT_KEYS)


def test_parse_card_rejects_missing_or_extra_keys():
    with pytest.raises(ValueError):
        parse_card("x", json.dumps({k: v for k, v in EXAMPLE.items() if k != "enrolled_in"}))
    with pytest.raises(ValueError):
        parse_card("x", json.dumps({**EXAMPLE, "visa": "maybe"}))


def test_parse_card_rejects_flat_must_skills():
    with pytest.raises(ValueError):
        parse_card("x", json.dumps({**EXAMPLE, "must_skills": ["Python", "SQL"]}))


def test_blueprint_lists_the_closed_industries():
    for ind in INDUSTRIES:
        assert f'"{ind}"' in BLUEPRINT, ind


def test_blueprint_covers_the_v2_fixes():
    for phrase in ("slash pair", "now or in the future", "related_ok", "nice_skills, never must_skills"):
        assert phrase in BLUEPRINT, phrase


def test_parse_card_allows_related_ok_to_be_left_out():
    data = {k: v for k, v in EXAMPLE.items() if k != "related_ok"}
    assert parse_card("x", json.dumps(data)).related_ok is False


def test_soft_skills_move_to_nice_and_slash_pairs_become_groups():
    must, nice = clean_skills(
        [["C/C++"], ["Problem-solving"], ["Strong communication skills"], ["CI/CD"], ["Python", "Programming"],
         ["Linux / Unix"], ["Communication and teamwork"], ["SQL"], ["sql"]],
        ["Git"],
    )
    assert must == [["C", "C++"], ["CI/CD"], ["Python"], ["Linux", "Unix"], ["SQL"]]
    assert nice[0] == "Git"
    assert {"Problem-solving", "Strong communication skills", "Programming"} <= set(nice)


@pytest.mark.parametrize("name,soft", [("Communication", True), ("Problem solving", True), ("problem-solving", True),
                                         ("Analytical skills", True), ("Teamwork", True), ("Multitasking", True),
                                         ("Research", True), ("Debugging", True), ("Python", False),
                                         ("Data Analysis", False), ("Verilog", False)])
def test_is_soft(name, soft):
    assert is_soft(name) is soft


def test_vague_degree_fields_are_dropped_and_set_related_ok():
    d = clean_card({"degree_fields": ["Computer Science", "related field", "STEM", "Related technical field",
                                      "Quantitative Field", "Equivalent"], "related_ok": False})
    # one "related field" entry stays: the rules read the flag from degree_fields (degrees.parse_fields)
    assert d["degree_fields"] == ["Computer Science", "related field"] and d["related_ok"] is True
    assert clean_card(d) == d  # cleaning a clean card changes nothing
    d = clean_card({"degree_fields": ["Engineering"], "related_ok": False})
    assert d["degree_fields"] == ["Engineering"] and d["related_ok"] is False
    assert clean_card({"degree_fields": [], "related_ok": True})["degree_fields"] == []


@pytest.mark.parametrize("raw,want", [("NYC, NY", "New York, NY"), ("DC, DC", "Washington, DC"),
                                      ("Washington, D.C.", "Washington, DC"), ("SF, CA", "San Francisco, CA"),
                                      ("Boston, Massachusetts", "Boston, MA"), ("Austin, TX (Hybrid)", "Austin, TX"),
                                      ("Remote, USA", None), ("California", None), ("Paris, France", "Paris, France")])
def test_clean_location(raw, want):
    assert clean_location(raw) == want


def test_industry_is_snapped_to_the_closed_list():
    assert snap_industry("Technology") == ("technology", True)
    assert snap_industry("Semiconductors")[0] == "manufacturing"
    assert snap_industry("Financial Services")[0] == "finance"
    assert snap_industry("Biotechnology")[0] == "biotech and pharma"
    assert snap_industry("zzz") == ("technology", False)
    assert all(snap_industry(i)[0] == i for i in INDUSTRIES)


def test_sponsorship_wording_from_the_posting():
    no = "Applicants must be able to work in the U.S. without sponsorship now or in the future."
    assert repair_sponsorship("unstated", no) == "will_not_sponsor"
    assert repair_sponsorship("unstated", "Candidates will not require sponsorship for employment.") == "will_not_sponsor"
    assert repair_sponsorship("unstated", "We do not sponsor visas, including OPT or STEM OPT.") == "will_not_sponsor_or_opt"
    assert repair_sponsorship("will_not_sponsor", "US citizens only, ITAR controlled.") == "unstated"
    assert repair_sponsorship("unstated", "We will sponsor H-1B visas. Some roles are without sponsorship.") == "unstated"


def test_license_not_held_and_hourly_salary_are_repaired():
    d = clean_card({"license": "Secret U.S. Security Clearance", "salary_min": 60, "salary_max": 60})
    assert d["license"] is None and d["salary_min"] == d["salary_max"] == 124800
    assert clean_card({"license": "RN license"})["license"] == "RN license"
    assert clean_card({"license": "State gaming license (if applicable)"})["license"] is None


def test_non_us_postings_get_country_and_no_salary():
    assert posting_meta(["Toronto, ON"], "x")["country"] == "CA"
    assert posting_meta(["London, UK"], "x")["country"] == "UK"
    assert posting_meta(["Paris, France"], "x") ["us"] is False
    assert posting_meta(["Boston, MA", "Remote"], "x")["us"] is True
    assert posting_meta(["Boston, MA", "Toronto, ON"], "x")["mixed"] is True
    fr = "Nous recherchons un stagiaire avec une exp\u00e9rience dans notre \u00e9quipe. Le poste est pour les \u00e9tudiants. " * 6
    assert posting_meta(["Montreal, QC"], fr)["language"] == "fr"
    d = clean_card({"salary_min": 90000, "salary_max": 90000, "locations": ["Toronto, ON"]}, "x", ["Toronto, ON"])
    assert d["salary_min"] is None and d["salary_max"] is None


def test_posting_meta_remote_abroad_and_canadian_ca():
    assert posting_meta(["Remote - EMEA"], "x")["us"] is False
    assert posting_meta(["Remote"], "x")["us"] is True
    assert posting_meta(["Toronto, CA"], "x")["country"] == "CA"
    assert posting_meta(["San Francisco, CA"], "x")["country"] == "US"


def _enroll(text, **card):
    d = {"enrolled_in": "bachelor", "degree_level": "none", **card}
    repair_enrollment(d, text)
    return d


def test_enrollment_kept_when_the_posting_says_graduating_or_expected():
    # no level named: any program counts ("associate")
    assert _enroll("Must be planning on graduating in 2028. This should be your final internship.")["enrolled_in"] == "associate"
    assert _enroll("Expected Bachelor's Degree in an engineering field")["enrolled_in"] == "bachelor"
    assert _enroll("Scheduled to obtain a Bachelor's degree by Winter 2028")["enrolled_in"] == "bachelor"


def test_enrollment_dropped_when_no_student_is_mentioned_or_recent_graduates_count():
    assert _enroll("Software engineering intern. A top-notch programmer with a love for technology.")["enrolled_in"] == "none"
    assert _enroll("Currently pursuing a degree in CS, or a recent graduate.")["enrolled_in"] == "none"
    assert _enroll("Pursuing or recently completed a Master's in CS.", enrolled_in="master")["enrolled_in"] == "none"


def test_enrollment_ignores_recent_graduates_in_visa_boilerplate():
    t = ("Currently enrolled in a Bachelors degree. Students or recent graduates in the United States on an F1 visa "
         "are not eligible for this role.")
    assert _enroll(t)["enrolled_in"] == "bachelor"


def test_graduate_students_hold_a_bachelors():
    d = _enroll("Currently enrolled in a PhD program.", enrolled_in="doctoral", degree_level="master")
    assert d["degree_level"] == "bachelor"
    d = _enroll("Currently enrolled in a PhD program.", enrolled_in="doctoral", degree_level="none")
    assert d["degree_level"] == "bachelor"
    d = _enroll("PhD students who already hold a master's degree.", enrolled_in="doctoral", degree_level="master")
    assert d["degree_level"] == "master"
    d = _enroll("Pursuing or recently completed a Master's.", enrolled_in="master", degree_level="none")
    assert d["degree_level"] == "bachelor"  # set before the recent-graduate reset


def test_enrollment_none_when_graduates_or_equivalent_experience_also_qualify():
    for t in ["- Pursuing or obtained a Bachelor\u2019s or Master\u2019s degree in a STEM related discipline.",
              "Qualifications include working toward, or recently acquired, a degree in Computer Science.",
              "- Currently pursuing a bachelor's degree or higher in electrical engineering or related field or can "
              "demonstrate equivalent experience",
              "- Pursuing a Bachelor's degree in Business or related field (or equivalent experience)."]:
        assert _enroll(t)["enrolled_in"] == "none", t


def test_enrollment_kept_when_equivalent_is_about_the_degree_or_a_later_requirement():
    t = "- Currently enrolled in a Computer Science or equivalent degree program"
    assert _enroll(t)["enrolled_in"] == "associate"
    t = ("Must be currently enrolled in an undergraduate or graduate program and have completed at least one course "
         "or equivalent experience with programming in Java.")
    assert _enroll(t)["enrolled_in"] == "bachelor"


def test_enrollment_not_invented_from_co_op_or_company_blurbs():
    risepoint = ("Analytics Engineering Co-op. Our suite of products and services supports the full student journey and "
                 "each university's long-term goals. This co-op is a real seat in that work. This role will serve as a "
                 "liaison between Student Success Center (SSC) Analytics and our Business Technology team.")
    assert _enroll(risepoint)["enrolled_in"] == "none"
    rundoo = ("WHY CO-OPS ARE CORE TO HOW WE BUILD OUR TEAM. At Rundoo, hiring co-op students who come back full-time "
              "after graduation is core to how we build our team. Students who pair rigorous academics with hands-on "
              "co-op experience become builders.")
    assert _enroll(rundoo)["enrolled_in"] == "none"
    thales = "Earn a salary while studying for your qualification. You will study towards a Graduate Apprenticeship."
    assert _enroll(thales)["enrolled_in"] == "none"


def test_enrollment_kept_on_real_study_requirements():
    clio = "- You've completed your second year coursework and are a 3rd or 4th year Computer Science/Engineering student;"
    assert _enroll(clio)["enrolled_in"] == "bachelor"
    caci = "- Completion of at least one year towards a Bachelor's degree in Computer Science, Computer Engineering"
    assert _enroll(caci)["enrolled_in"] == "bachelor"
    for t in ["Bose Corporation is seeking students to participate in our Co-op Program.",
              "- Must be graduating with an Master\u2019s degree in a technical subject",
              "- Qualified candidate must be in progress of obtaining a BS degree in computer science",
              "- Bachelor\u2019s degree expected between Fall 2027 and Spring 2028",
              "Are you a PhD / MS student working in NLP?",
              "- Studying Computer Science or Computer Engineering"]:
        assert _enroll(t)["enrolled_in"] != "none", t


def test_enrollment_with_no_level_named_is_associate():
    assert _enroll("- Currently pursuing a degree in Software Engineering or a related technical field.")["enrolled_in"] == "associate"
    assert _enroll("We can only move forward with those in their penultimate year. Those working towards a computer "
                   "science degree are welcome.")["enrolled_in"] == "associate"
    # "masters of a diverse set of commodities" names no level
    assert _enroll("Our traders are the masters of a diverse set of commodities. Graduation date of December 2027/Spring "
                   "2028.")["enrolled_in"] == "associate"
    assert _enroll("Currently pursuing a Bachelor's degree in CS.")["enrolled_in"] == "bachelor"
    assert _enroll("Currently pursuing a BSEE.")["enrolled_in"] == "bachelor"
    assert _enroll("Currently enrolled in a master's program.", enrolled_in="master")["enrolled_in"] == "master"


def _auth(text, **card):
    d = {"sponsorship": "unstated", "citizens_only": False, **card}
    repair_work_auth(d, text)
    return d


def test_sponsored_benefits_are_not_visa_sponsorship():
    t = "Sponsored One Medical memberships for employees. Company-sponsored events. U.S. persons only (ITAR)."
    assert _auth(t, sponsorship="will_not_sponsor_or_opt")["sponsorship"] == "unstated"
    assert _auth("Sponsorship: Yes", sponsorship="sponsors")["sponsorship"] == "sponsors"
    assert _auth("Visa status may be considered when the internship can be employed legally.",
                 sponsorship="sponsors")["sponsorship"] == "unstated"


def test_permanent_authorization_and_cpt_refusal_mean_citizens_only():
    assert _auth("Permanently authorized to work in the U.S., must not require sponsorship.")["citizens_only"] is True
    assert _auth("F1 students (including those with CPT or OPT) are not eligible for this role.")["citizens_only"] is True
    # Motorola: CPT accepted for MS students, so not citizens-only
    assert _auth("Must be a US citizen, permanent resident or an MS student (F1 on CPT accepted).")["citizens_only"] is False


@pytest.mark.parametrize("text", [
    "- Must be authorized to work in the U.S. on a permanent basis. We do not offer any type of sponsorship.",  # Invesco
    "- You must have legal authorization to work in the U.S.\non a permanent basis without the need for sponsorship",  # Marsh
    "- Permanent US work authorization without the need for sponsorship now, or in the future",
    "- Applicants must be authorized to work permanently for any employer in the U.S.",
    "Please apply only if you have ongoing, permanent authorization to work in the United States.",  # Viget
    "Please note that this role requires permanent authorization to work in the U.S.",
    "- Candidates must have permanent U.S. work authorization; we do not sponsor employment visas.",
])
def test_permanent_authorization_wordings_mean_citizens_only(text):
    assert _auth(text)["citizens_only"] is True


@pytest.mark.parametrize("text", [
    "Must be a U.S. citizen, permanent resident or possess other unrestricted US work authorization.",  # Ingredion
    "Must be a US citizen, permanent resident or be an MS student with work authorization (F1 on CPT accepted).",
    "Applicants must be authorized to work in the U.S. Interns whose permanent address is 50 miles away get housing.",
    "Citizens, lawful permanent residents, or protected individuals as defined by 8 U.S.C. 1324b(a)(3).",
])
def test_permanent_resident_wording_alone_is_not_citizens_only(text):
    assert _auth(text)["citizens_only"] is False


def test_stem_spelled_out_is_no_field_and_experience_phrases_are_no_skill():
    d = {"degree_fields": ["Physics", "Optical", "Science", "Technology", "Engineering", "Mathematics"],
         "must_skills": [["building a nontrivial agent"], ["customer-facing experience"], ["Python"]]}
    repair_vague(d)
    assert d["degree_fields"] == ["Physics", "Optical"] and d["related_ok"] is True
    assert d["must_skills"] == [["Python"]]
    d = {"degree_fields": ["Engineering"], "must_skills": []}
    repair_vague(d)
    assert d["degree_fields"] == ["Engineering"]


POSTING = """About the role
We use Go and Kubernetes every day.
Basic Qualifications:
- Currently pursuing a Bachelor's degree in Computer Science
- Proficiency in Python
Strong SQL skills are required. Experience with Tableau is a plus.
Preferred Qualifications:
- Experience with Rust
- Physics degree"""


def test_sections_tag_preferred_headings_and_plus_sentences():
    kinds = {k for k, _ in sections(POSTING)}
    assert {"req", "pref"} <= kinds
    segs = sections(POSTING)
    assert where(["Rust"], segs) == "pref"
    assert where(["Python"], segs) == "keep"
    assert where(["Tableau"], segs) == "pref"  # "... is a plus" in the same paragraph as a required line
    assert where(["SQL"], segs) == "keep"
    assert where(["Go"], segs) == "keep"  # intro mention: duties are not used to drop skills


def test_degree_fields_only_count_degree_sentences():
    segs = degree_segs(sections(POSTING))
    assert where(["Computer Science"], segs) == "keep"
    assert where(["Physics"], segs) == "pref"
    d = clean_card({"degree_fields": ["Computer Science", "Physics"], "must_skills": [["Python"], ["Rust"]],
                    "nice_skills": [], "related_ok": True}, POSTING)
    assert d["degree_fields"] == ["Computer Science", "related field"] and d["must_skills"] == [["Python"]]
    assert "Rust" in d["nice_skills"]


def test_clean_card_without_text_still_cleans_the_card_itself():
    d = clean_card({"degree_fields": ["Science", "Technology", "Engineering", "Math"], "must_skills": [["C/C++"]],
                    "nice_skills": []})
    assert d["degree_fields"] == [] and d["related_ok"] is True and d["must_skills"] == [["C", "C++"]]


def test_english_postings_with_experience_and_stage_are_english():
    en = ("Experience with Python is required. You will gain experience at an early stage startup. "
          "Prior experience, experience with SQL, and stage-gate experience are a plus. ") * 12
    assert posting_meta(["Boston, MA"], en)["language"] == "en"
    assert posting_meta(["Boston, MA"], "Experience des clients. Une stage. " * 20)["language"] == "en"


def test_ats_shorthands_and_bare_states_are_us():
    for loc in (["NYC"], ["SF"], ["South SF"], ["LA"], ["California"], ["Iowa"], ["Washington"], ["Albuquerque, New Mexico"]):
        m = posting_meta(loc, "x")
        assert m["us"] is True and m["country"] == "US", loc
    assert posting_meta(["Mexico City, Mexico"], "x")["country"] == "other"
    assert posting_meta(["NYC", "London, UK"], "x")["mixed"] is True
    # ATS locations that say nothing usable: the card's cleaned locations decide
    assert posting_meta(["Hybrid"], "x", ["Austin, TX"])["us"] is True
    assert posting_meta(["Hybrid"], "x", ["Toronto, ON"])["country"] == "CA"


def test_dotted_names_and_conjunctive_slash_pairs_survive_cleaning():
    must, nice = clean_skills([[".NET", "C#"], ["Node.js"], ["HTML/CSS"], ["ONNX/Runtime"], ["UI/UX"], ["HTML/CSS", "React"]],
                              [".NET Core."])
    assert must == [[".NET", "C#"], ["Node.js"], ["HTML"], ["CSS"], ["ONNX Runtime"], ["UI/UX"], ["HTML", "CSS", "React"]]
    assert nice == [".NET Core"]


def test_quantitative_finance_is_a_field_quantitative_field_is_not():
    assert not is_vague_field("Quantitative Finance")
    assert is_vague_field("quantitative field") and is_vague_field("Quantitative") and is_vague_field("STEM")
    assert not is_vague_field("Stem Cell Biology")


SKYDIO = ("What makes you a good fit:\n\n - Solid computer science fundamentals.\n\n"
          " - Proficiency with C/C++, Python, Golang, and/or Javascript/Typescript.\n\n"
          " - Familiarity with REST-ful APIs, Linux operating system configuration, relational databases (e.g. Postgres, MySQL).")
INTEL = ("Minimum Qualifications:\n\n- Minimum of 6 months of experience in one or more of the following:\n\n"
         "- Semiconductor materials, fabrication, and device physics\n\n"
         "- Electrical characterization of Semiconductor Devices (transistor, diode, etc.)\n\n"
         "- Familiarity with statistical methods and techniques used for data analysis.\n\n"
         "Preferred Qualifications\n\n- Advanced Transistor Device Structures and Device Physics.")


def test_anyof_lists_the_model_split_become_one_group():
    d = clean_card({"must_skills": [["C/C++"], ["Python"], ["Golang"], ["Javascript/Typescript"], ["REST-ful APIs"], ["Linux"],
                                    ["Postgres", "MySQL"]], "nice_skills": []}, SKYDIO)
    assert d["must_skills"] == [["C", "C++", "Python", "Golang", "Javascript", "Typescript"], ["REST-ful APIs"], ["Linux"],
                                ["Postgres", "MySQL"]]
    d = clean_card({"must_skills": [["Semiconductor materials", "fabrication", "device physics"],
                                    ["Electrical characterization of Semiconductor Devices"], ["statistical methods"]],
                    "nice_skills": []}, INTEL)
    assert d["must_skills"] == [["Semiconductor materials", "fabrication", "device physics",
                                 "Electrical characterization of Semiconductor Devices", "statistical methods"]]


def test_anyof_merge_stays_inside_the_choice():
    t = "Requirements:\n- Operating Systems and tools: Linux and/or Android, Claude Code and OpenAI API"
    assert merge_anyof([["Linux"], ["Android"], ["Claude Code"], ["OpenAI API"]], t) == \
        [["Linux", "Android"], ["Claude Code"], ["OpenAI API"]]
    t = "- Knowledge of SQL; data analytics skills; Python and/or Java programming skills"
    assert merge_anyof([["SQL"], ["Python"], ["Java"]], t) == [["SQL"], ["Python", "Java"]]
    t = "- experience with large data sets and with statistical programming (Python and/or R)"
    assert merge_anyof([["large data sets"], ["Python"], ["R"]], t) == [["large data sets"], ["Python", "R"]]
    t = ("- Strong programming skills in one or more of the following:\n- C++\n- Java\n"
         "- Understanding of object-oriented programming principles.\n- Familiarity with tools such as Git.")
    assert merge_anyof([["C++"], ["Java"], ["Object-oriented programming"], ["Git"]], t) == \
        [["C++", "Java"], ["Object-oriented programming"], ["Git"]]
    assert merge_anyof([["Python"], ["SQL"]], "- Python and SQL are required.") == [["Python"], ["SQL"]]


def test_phrase_groups_flag_and_training_ok():
    assert phrase_groups([["Python"], ["data processing concepts"], ["Ability to learn fast"], ["Large-scale agent systems"],
                          ["writing software", "Java"]]) == [["data processing concepts"], ["Ability to learn fast"],
                                                             ["Large-scale agent systems"]]
    # category words alone: a resume names the instance (PostgreSQL, Copilot), not the category
    assert phrase_groups([["databases"], ["AI tools"], ["AI-assisted tools"], ["testing"], ["SQL"], ["cloud platforms"],
                          ["unit testing"], ["databases", "PostgreSQL"]]) == [["databases"], ["AI tools"],
                                                                              ["AI-assisted tools"], ["testing"]]
    assert training_ok(None) and training_ok({"us": True, "language": "en", "must_phrases": []})
    assert not training_ok({"us": False, "language": "en", "must_phrases": []})
    assert not training_ok({"us": True, "language": "fr", "must_phrases": []})
    assert not training_ok({"us": True, "language": "en", "must_phrases": [["data processing concepts"]]})


LEIDOS = """The division is seeking Software Engineer Interns.
Primary Responsibilities:
Tasks will include system architecture and requirements engineering. The candidate will also support
regression testing and coding of the program software, including Object Oriented Programming Concepts, XML,
JavaScript, C, C++, Java and Python on Redhat Linux.
Basic Qualifications:
- Candidates must be enrolled in a degree program in a relevant discipline, for example, computer science.
- Exposure to system engineering, software engineering, artificial intelligence, or machine learning.
- Candidates must be US Citizens"""

HYVEE = """Job Description:
What you'll do:
- Contribute to the design and development of data pipelines.
- Support research projects with data retrieval and analysis using Microsoft SQL Server.
Qualifications:
- Knowledge of data development technologies
- Strong critical thinking and problem solving skills"""


def test_duty_only_groups_leave_must_and_nice_skills():
    # a tool named only in the duties is neither required nor a nice-to-have the applicant brings
    must = [["C"], ["C++"], ["Java"], ["Python"], ["JavaScript"], ["XML"], ["Redhat"]]
    d = clean_card({"must_skills": must, "nice_skills": []}, LEIDOS)
    assert d["must_skills"] == []
    assert not {"C++", "Java", "Python", "Redhat"} & set(d["nice_skills"])
    d = clean_card({"must_skills": [["Microsoft SQL Server"]], "nice_skills": []}, HYVEE)
    assert d["must_skills"] == [] and "Microsoft SQL Server" not in d["nice_skills"]


def test_duty_only_keeps_skills_the_requirements_name_or_without_a_requirements_section():
    t = HYVEE + "\n- Experience with SQL Server and Python"
    assert not duty_only(["Python"], sections(t))
    segs = sections(POSTING)
    assert not duty_only(["Python"], segs)
    assert not duty_only(["Kubernetes"], sections("About the role\nWe use Kubernetes every day."))  # no requirements
    assert not duty_only(["Kubernetes"], segs)  # an untitled intro line is no duty heading
    leidos = sections(LEIDOS)
    assert duty_only(["C"], leidos) and not duty_only(["c"], leidos)  # "C", "R", "Go": exact case
    assert not duty_only(["R"], leidos)  # not found anywhere counts as required


KLA = """Minimum Qualifications:
- Requires less than 1 year of related experience with a Bachelor's degree.
Preferred Qualifications:
- Programming in C++, Java, C#, JavaScript or TypeScript."""


def test_short_names_under_preferred_only_move_to_nice_skills():
    d = clean_card({"must_skills": [["C++", "Java", "C#", "JavaScript", "TypeScript"]], "nice_skills": []}, KLA)
    assert d["must_skills"] == [] and "C#" in d["nice_skills"]
    # "C" inside "C++" or "R" inside "R&D" is no mention; a name found nowhere stays required
    segs = sections("Requirements:\n- Experience in R&D.\nPreferred:\n- C++ and R.")
    assert where(["R"], segs) == "pref" and where(["C"], segs) == "keep"
    assert where(["C#"], sections("Requirements:\n- C# and .NET.\nPreferred:\n- C#")) == "keep"


def test_enrollment_cue_needs_study_wording_not_any_candidate_for():
    figma = ("Experience solving technical problems through school, work, or personal projects. Don't meet every "
             "requirement? You may be just the right candidate for this or other roles.")
    assert _enroll(figma, enrolled_in="associate")["enrolled_in"] == "none"
    assert _enroll("Ideal candidate for this role enjoys a team environment.", enrolled_in="associate")["enrolled_in"] == "none"
    assert _enroll("- Must be a candidate for a Bachelor's degree in CS.")["enrolled_in"] == "bachelor"
    applied = "- Current PhD (or MSc) studies in machine learning, computer vision and graphics, robotics"
    assert _enroll(applied, enrolled_in="master")["enrolled_in"] == "master"
    axon = "- Pursuing graduation in a Computer Science, Engineering, or a related field at a community college or bootcamp"
    assert _enroll(axon, enrolled_in="associate")["enrolled_in"] == "associate"


def test_enrollment_junior_status_names_a_level():
    t = "This full-time, 12-week internship is open to Junior-status college students only."
    assert _enroll(t)["enrolled_in"] == "bachelor"
    assert _enroll("Must have junior status or higher at an accredited university.")["enrolled_in"] == "bachelor"


def test_enrollment_none_when_graduates_within_a_window_qualify():
    shark = ("- Currently enrolled in a bachelor's program in Engineering, or has graduated from such a program within "
             "the past year.")
    assert _enroll(shark)["enrolled_in"] == "none"
    assert _enroll("- Must be enrolled in a degree program or graduate within the past 6 months.",
                   enrolled_in="associate")["enrolled_in"] == "none"
    kyndryl = "Currently enrolled in a university or educational institution, or recently completed a qualification"
    assert _enroll(kyndryl)["enrolled_in"] == "none"


def test_recently_completed_project_or_degree_equivalent_is_no_way_around_study():
    viking = ("- Currently pursuing a Master's degree or a PhD program (3rd year and above) in Data Science.\n"
              "In 1-2 pages, the supplement should describe a quantitative research project that you recently completed.")
    assert _enroll(viking, enrolled_in="master", degree_level="bachelor")["enrolled_in"] == "master"
    blue = ("- Currently enrolled in a graduate or MBA program with at least one semester remaining.\n"
            "- Bachelor's degree in engineering, computer science, physics, or other technical field, or equivalent "
            "technical work experience.")
    assert _enroll(blue, enrolled_in="master", degree_level="bachelor")["enrolled_in"] == "master"


ELEVEN = ["communication", "problem solving", "teamwork", "collaboration", "analytical skills", "attention to detail",
          "time management", "research", "programming", "coding", "debugging"]


def test_copied_soft_skill_list_keeps_only_what_the_posting_names():
    t = ("Requirements:\n- Python and SQL.\n- Strong communication skills and a problem-solving mindset.\n"
         "Preferred:\n- Experience with AWS.")
    d = clean_card({"must_skills": [["Python"], ["SQL"]], "nice_skills": ["AWS", *ELEVEN]}, t)
    assert d["nice_skills"] == ["AWS", "communication", "problem solving"]
    # stems count: "collaborative" names collaboration; concrete skills stay even when the posting renames them
    d = clean_card({"must_skills": [], "nice_skills": ["collaboration", "Amazon Web Services"]}, "A collaborative team.")
    assert d["nice_skills"] == ["collaboration", "Amazon Web Services"]
    # without the posting text nothing is dropped
    assert clean_card({"must_skills": [], "nice_skills": ELEVEN})["nice_skills"] == ELEVEN


# ------------------------------------------------------------ v3 review: clause-local "preferred", headings


def _spans(sent):
    return [sent[a:b].strip() for a, b in pref_spans(sent)]


def test_preferred_cue_covers_its_own_clause_only():
    # DIRECTV 4a01e4d3, Q2 4b9d78b3, Astera 8dd69eeb, DraftKings 4e50989c, Arch 483a6e7d
    assert _spans("Must have Python and SQL (experience with Databricks is a plus)") == ["experience with Databricks is a plus"]
    assert _spans("Coursework or project experience with Python (R or Java a plus)") == ["R or Java a plus"]
    assert _spans("Proficiency in C (preferred) or C++, including development of firmware.") == ["Proficiency in C (preferred)"]
    assert _spans("A strong grasp of statistics and hypothesis testing, with exposure to model-building "
                  "considered a plus.") == ["with exposure to model-building considered a plus"]
    assert _spans("Proficiency in SQL required and Python/Spark preferred") == ["and Python/Spark preferred"]
    assert _spans("Strong programming ability in Python (C++, Rust, or another language is a plus).") == [
        "C++, Rust, or another language is a plus"]  # Anthelion 5715d5fe: a list tail inside parentheses
    # a cue at the end of a list covers the list; a leading cue covers the sentence
    assert _spans("Experience with Docker, Kubernetes, and AWS is a plus") == ["Experience with Docker, Kubernetes, and AWS is a plus"]
    assert _spans("Experience with large datasets preferred.") == ["Experience with large datasets preferred"]
    assert _spans("Preferred: Rust, Go") == ["Preferred: Rust, Go"]


def test_preferred_clause_keeps_the_required_items_required():
    t = ("Requirements:\n- Must have Python and SQL (experience with Databricks is a plus)\n"
         "- Proficient in Python and SQL.Solid understanding of modern AI/LLM is a plus\n"  # Tencent 560fa8e4
         "- Exposure to Object-Oriented Programming (C# or LabVIEW experience a plus).\n"  # AeroVironment fa72707f
         "- Experience with electrical CAD software (ideally KiCAD)\n")  # Persona 63e35d52
    must = [["Python"], ["SQL"], ["Databricks"], ["Object-Oriented Programming"], ["C#", "LabVIEW"],
            ["KiCAD", "electrical CAD software"]]
    d = clean_card({"must_skills": must, "nice_skills": []}, t)
    assert d["must_skills"] == [["Python"], ["SQL"], ["Object-Oriented Programming"], ["KiCAD", "electrical CAD software"]]
    assert {"Databricks", "C#", "LabVIEW"} <= set(d["nice_skills"])


@pytest.mark.parametrize("line,kind", [
    ("You'll Come With", "req"),  # Burlington 8de05538
    ("What You'll Have", "req"),
    ("What We Look For", "req"),  # Affirm b7368293
    ("What You Already Have", "req"),  # Dynamic Catholic fe9adce5
    ("What experience you need", "req"),  # Equifax 3e77a1a2
    ("What Life Fitness / Hammer Strength is looking for in our Embedded Software Engineering Intern:", "req"),  # 2913f51c
    ("REQUIRED FOR THE ROLE", "req"),  # Perchwell 70276fbb
    ("Student qualities we\u2019re seeking for this Co-Op assignment:", "req"),  # Flex 1ce187ae
    ("Role Expectations", "req"),
    ("Essential experience and job requirements", "req"),  # BP 27d35509
    ("Role:", "duty"),  # Astranis 78ea3e37
    ("Job Summary:", "duty"),
    ("Key Accountabilities", "duty"),
    ("What could set you apart", "pref"),  # Equifax
    ("About the Role", "about"),
    ("Title: Software Engineer Intern - Machine Learning", None),  # 'learn' in a title is no duty heading
    ("Machine Learning Engineering", None),
    ("Experience with Python required", None),  # an item, not a heading
    ("Desirable Candidates", "req"),  # Point72: its only list; not read as preferred
])
def test_requirement_and_duty_headings(line, kind):
    assert _head_kind(line) == kind


def test_no_requirements_heading_keeps_duty_named_skills():
    # the WIP rule dropped names found only under duty headings when no requirements heading was found;
    # with an unread requirements heading that lost real requirements, so such postings keep them
    t = "What You'll Do\n- Build services in Python\nWhat we kinda want\n- Python and SQL"
    assert not duty_only(["Python"], sections(t))
    assert not duty_only(["Python"], sections("Areas of Focus\n- Python scripting"))
    t = "Role:\n- RTL Development for FPGA targeted applications\nRequirements:\n- Proficiency with Verilog"
    assert duty_only(["RTL Development"], sections(t)) and not duty_only(["Verilog"], sections(t))


def test_comma_or_lists_offer_a_choice():
    assert _comma_or_scopes("Strong training in econometrics, empirical finance, statistics, or quantitative economics") == [
        "econometrics, empirical finance, statistics, or quantitative economics"]
    assert _comma_or_scopes("APIs such as Crossref, Semantic Scholar, or SSRN.") == ["Crossref, Semantic Scholar, or SSRN"]
    assert _comma_or_scopes("Python, SQL, and Spark, or equivalent") == []  # 'and' in the list: unclear
    t = "Requirements:\n- Experience with C/C++, Python, or Rust"
    assert merge_anyof([["C", "C++"], ["Python"], ["Rust"]], t) == [["C", "C++", "Python", "Rust"]]
    # a list in a duty line says nothing about which skills the applicant may choose between
    t = "Responsibilities:\n- Perform validation using Python, C++, or Rust\nRequirements:\n- Python\n- C++"
    assert merge_anyof([["Python"], ["C++"]], t) == [["Python"], ["C++"]]


def test_nice_skills_keep_the_preferred_list_only():
    t = ("Responsibilities:\n- Write dashboards in Tableau\n- Help maintain documentation, update timelines\n"
         "Requirements:\n- Python\n- You will receive training in SQL\n- Interest in robotics\n"
         "Our tech stack is a blend of Go and Rust.\n"
         "Preferred:\n- " + "\n- ".join(f"Tool{k}" for k in range(10)) + "\n- Docker")
    nice = ["Tableau", "updating timelines", "SQL", "robotics", "Go", "fast-paced environment",
            *(f"Tool{k}" for k in range(10)), "Docker", "communication"]
    d = clean_card({"must_skills": [["Python"]], "nice_skills": nice}, t + "\nStrong communication")
    assert d["must_skills"] == [["Python"]]
    kept = [a for a in d["nice_skills"] if a != "communication"]
    assert kept == [f"Tool{k}" for k in range(8)]  # duty, learn, interest, stack and phrase items out; at most 8
    assert "communication" in d["nice_skills"]  # soft skills sit outside the cap (the rules never count them)


def _mode(mode, text, locs=None):
    d = {"work_mode": mode}
    repair_work_mode(d, text, locs)
    return d["work_mode"]


def test_work_mode_remote_needs_a_role_specific_cue():
    booz = ("- Remote: If this position is listed as remote, there may still be occasions when you are required "
            "to work in person.\n- Hybrid: If this position is listed as hybrid, you will be expected to work from a "
            "Booz Allen facility.")
    assert _mode("remote", booz, ["McLean, VA"]) == "onsite"  # Booz Allen 8c3babb7
    assert _mode("remote", "Join our team.", ["United States"]) == "onsite"  # SingleStore e24e39d5
    assert _mode("remote", "Cohere is remote-friendly, but we also have offices in Toronto.", ["SF"]) == "onsite"
    assert _mode("remote", "Join our team.", ["Remote in USA"]) == "remote"
    assert _mode("remote", "Telecommuting Options: | Remote", ["Falls Church, VA"]) == "remote"  # GDIT db3236dd
    assert _mode("remote", "Affirm is proud to be a remote-first company.", ["SF"]) == "remote"
    assert _mode("remote", "This internship is fully remote.", ["Austin, TX"]) == "remote"


def test_work_mode_hybrid_needs_the_word_or_a_schedule():
    assert _mode("hybrid", "Join our Seattle team.", ["Seattle, WA"]) == "onsite"
    assert _mode("hybrid", "This is a hybrid role.") == "hybrid"
    assert _mode("hybrid", "Must be able to be in office 3+ days a week.") == "hybrid"  # Mastercard 12b0c5bd
    assert _mode("hybrid", "in-office collaboration Monday-Wednesday in our Palo Alto office") == "hybrid"  # Pebl
    assert _mode("hybrid", "based in our Sunnyvale, CA office or remotely in the US") == "hybrid"  # Azul: remote option
    assert _mode("onsite", "This is a hybrid role.") == "onsite"  # onsite is never changed


def test_enrollment_master_students_count_when_a_phd_posting_names_them():
    assert _enroll("- Pursuing PhD or Master's Degree in related field", enrolled_in="doctoral",
                   degree_level="master")["enrolled_in"] == "master"  # Symbotic 9513d076
    lseg = ("To be eligible, you must be in your penultimate year of graduate study. You should have a predicted "
            "PhD or master’s degree in Math.")
    d = _enroll(lseg, enrolled_in="doctoral", degree_level="master")
    assert d["enrolled_in"] == "master" and d["degree_level"] == "bachelor"  # LSEG a4a4b9ce
    autodesk = ("- Currently pursuing a PhD in Computer Science.\n- $72,800 and $118,500 (Masters/MBA students)\n"
                "- $118,560 and $162,240 (PhD students)")
    assert _enroll(autodesk, enrolled_in="doctoral")["enrolled_in"] == "doctoral"  # a pay table is no cue
    assert _enroll("- Currently pursuing a PhD. Master's degree preferred.", enrolled_in="doctoral")["enrolled_in"] == "doctoral"
    # a master's student already holds a bachelor's (LSEG 9bd3bc29 came back master/master)
    assert _enroll("- Pursuing a master's degree in Finance", enrolled_in="master", degree_level="master")["degree_level"] == "bachelor"


def test_enrollment_new_grad_only_as_an_alternative():
    chevron_like = ("Our interns and new graduates work on real projects.\n- Enrolled and currently pursuing a BS/MS "
                    "in Computer Science")
    assert _enroll(chevron_like)["enrolled_in"] == "bachelor"  # Salesforce 3cb45823
    assert _enroll("New grads who join full-time enter the Horizons Program.\n- Currently enrolled in a bachelor's "
                   "degree program")["enrolled_in"] == "bachelor"
    assert _enroll("Title: Aerospace Intern New Grad\n- Currently enrolled in a Bachelor's degree program",
                   )["enrolled_in"] == "bachelor"  # Nidec 3f6c628b
    assert _enroll("- Students currently pursuing a bachelor's degree or new grads are welcome")["enrolled_in"] == "none"
    assert _enroll("- Recent PhD graduate OR PhD candidate in Photonics", enrolled_in="doctoral")["enrolled_in"] == "none"


def test_enrollment_undergraduate_studies_is_a_bachelors():
    assert _enroll("- Currently pursuing undergraduate studies in Supply Chain", enrolled_in="associate")["enrolled_in"] == "bachelor"


def test_citizens_only_us_person_citizenship_and_clearance():
    for t in ["This job requires a U.S. Person.",  # RTX fd622461
              "To meet U.S. export control compliance requirements, a “U.S. Person” as defined by 22 C.F.R. "
              "§120.62 is required.",  # Boeing e2123de3
              "- Ability to obtain and maintain a Secret security clearance.",  # CACI 8c3ef60f
              "- Active US Security clearance, or eligibility and willingness to obtain a US Security clearance.",
              "- Applicant must be a (i) U.S. citizen or national, (ii) U.S. lawful permanent resident",  # SpaceX
              "This position requires access to information subject to the International Traffic in Arms Regulations.",
              "US Citizenship Required:\nYes"]:
        assert _auth(t)["citizens_only"] is True, t
    for t in ["This position does not require a Security Clearance.", "US Citizenship Required:\nNo",
              "Some positions may require a security clearance.",
              "Must be a U.S. Citizen, an alien lawfully admitted for permanent residence, or an alien authorized to work.",
              "Must have unconditional right to work in the UK and be eligible for UK Security Clearance."]:
        assert _auth(t)["citizens_only"] is False, t


def test_citizens_only_false_when_f1_students_are_accepted():
    motorola = ("- Must be a US Citizen, permanent resident or be an MS student with work authorization (F1 Visa on CPT "
                "accepted only for masters-level students).")
    assert _auth(motorola, citizens_only=True)["citizens_only"] is False  # Motorola dffa3f5f
    assert _auth("International students on F-1 visas are not eligible. Must be a US citizen.",
                 citizens_only=True)["citizens_only"] is True


def test_must_and_lists_are_separate_requirements():
    t = "Requirements:\n- Experience in a direct software engineering utilizing C, C++, and Python\n"  # J&J 635bf0d1
    assert clean_card({"must_skills": [["C", "C++", "Python"]], "nice_skills": []}, t)["must_skills"] == [["C"], ["C++"], ["Python"]]
    t = "Requirements:\n- Experience with .NET technologies (C#, Visual Basic) and SQL databases\n"  # Nissan: examples
    assert clean_card({"must_skills": [["C#", "Visual Basic"]], "nice_skills": []}, t)["must_skills"] == [["C#", "Visual Basic"]]
    t = "Requirements:\n- Proficiency in Java or Python, and SQL\n"
    assert clean_card({"must_skills": [["Java", "Python"]], "nice_skills": []}, t)["must_skills"] == [["Java", "Python"]]


def test_must_slash_pairs_tech_stack_examples_and_subfeatures():
    t = "Required\n- Required: Python, SQL, ETL/Data Processing, APIs & File Integration, Git/GitHub, Microsoft Excel"
    d = clean_card({"must_skills": [["Python"], ["Git"], ["GitHub"], ["ETL"], ["Data Processing"]], "nice_skills": []}, t)
    assert d["must_skills"] == [["Python"], ["Git", "GitHub"], ["ETL", "Data Processing"]]  # Marmon b31bb072
    t = ("Requirements:\n- Experience with Python, SQL and Power BI Desktop, including foundational DAX\n"
         "- Troubleshoot DAX/SQL")  # Baird: SQL is named on its own, so no slash merge
    assert clean_card({"must_skills": [["SQL"], ["DAX"]], "nice_skills": []}, t)["must_skills"] == [["SQL"], ["DAX"]]
    t = "Our tech stack is a blend of Python and C++ that runs in the cloud."  # Xantium baf0289f
    assert clean_card({"must_skills": [["Python"], ["C++"]], "nice_skills": []}, t)["must_skills"] == []
    t = "Requirements:\n- Proficient in Python and SQL, scaled into tools within the broader tech stack"
    assert clean_card({"must_skills": [["Python"]], "nice_skills": []}, t)["must_skills"] == [["Python"]]
    t = "Requirements:\n- Experience with robotics frameworks such as ROS2 and physics simulators such as MuJoCo, Isaac Sim"
    assert clean_card({"must_skills": [["ROS2"]], "nice_skills": []}, t)["must_skills"] == []  # NVIDIA 876c3f6f
    t = "Requirements:\n- Tools like SPICE and Cadence"
    assert clean_card({"must_skills": [["SPICE"]], "nice_skills": []}, t)["must_skills"] == [["SPICE"]]
    t = ("Requirements:\n- Experience with Microsoft Excel including work with pivot tables, XLOOKUP, basic formulas\n"
         "- Understanding of data modeling concepts, including fact and dimension tables")  # Baird 8c416546
    must = [["Microsoft Excel"], ["pivot tables"], ["XLOOKUP"], ["basic formulas"], ["data modeling"], ["fact tables"]]
    assert clean_card({"must_skills": must, "nice_skills": []}, t)["must_skills"] == [["Microsoft Excel"], ["data modeling"]]
    t = "Requirements:\n- Strong Python skills, including pandas and NumPy"
    assert clean_card({"must_skills": [["Python"], ["pandas"]], "nice_skills": []}, t)["must_skills"] == [["Python"], ["pandas"]]


def test_many_required_groups_are_kept_out_of_training():
    assert not training_ok({"us": True, "language": "en", "must_phrases": [], "must_many": True})


def test_degree_fields_under_a_preferred_majors_list_are_not_required():
    t = "Requirements:\n- Pursuing a bachelor's degree\nDesired Majors\nComputer Science, Data Science, Business Analytics"
    d = clean_card({"degree_fields": ["Computer Science", "Data Science"], "related_ok": False}, t)
    assert d["degree_fields"] == []  # OCC cddeab16
    t = "Requirements:\n- Excellent academic background (hard sciences preferred - physics, chemistry or math)"
    assert clean_card({"degree_fields": ["physics", "math"], "related_ok": False}, t)["degree_fields"] == []  # Virtu
    t = "PREFERRED SKILLS AND EXPERIENCE\n- Excel\nEDUCATION\n- Pursuing a degree in Data Science or Statistics"
    assert clean_card({"degree_fields": ["Data Science"], "related_ok": False}, t)["degree_fields"] == ["Data Science"]


def test_no_visa_assistance_is_no_sponsorship():
    t = "- No company-sponsored housing, relocation, student visa or Green Card assistance available"
    assert clean_card({"sponsorship": "unstated"}, t)["sponsorship"] == "will_not_sponsor"  # CoverMyMeds 90d4be1a


def test_locations_and_pay_from_the_posting():
    d = clean_card({"locations": [], "work_mode": "onsite"}, "x", ["SF", "California"])
    assert d["locations"] == ["San Francisco, CA"]  # a state alone is no city
    d = clean_card({"locations": ["Overland Park"], "work_mode": "onsite"}, "x", ["Overland Park, KS"])
    assert d["locations"] == ["Overland Park, KS"]
    assert clean_card({"locations": [], "work_mode": "remote"}, "This role is fully remote.", ["SF"])["locations"] == []
    d = clean_card({"salary_min": 96051, "salary_max": 96051}, "Pay: $46.18/hour")
    assert d["salary_min"] == d["salary_max"] == 96054


def test_forward_preferred_cues_and_hourly_ranges():
    t = "Requirements:\n- Experience with programming or scripting, preferably Python\n"  # Leidos e3e1417f
    assert clean_card({"must_skills": [["Python"]], "nice_skills": []}, t)["must_skills"] == []
    t = "Requirements:\n- Experience with CAD software, ideally SolidWorks or Creo.\n- Python"
    assert clean_card({"must_skills": [["SolidWorks", "Creo"], ["Python"]], "nice_skills": []}, t)["must_skills"] == [["Python"]]
    t = "Students studying Electrical Engineering or Physics are encouraged to apply."  # Penn State 69430fc1
    assert clean_card({"degree_fields": ["Electrical Engineering", "Physics"], "related_ok": False}, t)["degree_fields"] == []
    d = clean_card({"salary_min": 40704, "salary_max": 55008}, "The likely hourly rate is between $19.55 - $26.45.")
    assert (d["salary_min"], d["salary_max"]) == (40664, 55016)  # GDIT 03a8913b
    d = clean_card({"salary_min": 90000, "salary_max": 90000}, "Salary: $90,000 a year ($43.26 per hour)")
    assert d["salary_min"] == 90000  # an annual figure the posting states stays


@pytest.mark.parametrize("text,value,want", [
    # Amgen 5012230e (and 7 more Amgen cards): authorization only; future sponsorship "not guaranteed"
    ("Sponsorship\n\nCandidates must be authorized to work in the U.S. for the duration of this program. "
     "Sponsorship for future FTE roles is not guaranteed.", "will_not_sponsor_or_opt", "unstated"),
    # Penn State 9b3a298c / 8b147aea: research sponsors, not visas
    ("expert knowledge for our Navy, Army, and other U.S. government sponsors.", "will_not_sponsor_or_opt", "unstated"),
    ("expert knowledge for our Navy, Army, and other U.S. government sponsors.", "will_not_sponsor", "unstated"),
    # Cloudflare adc756c8: export control, not a visa
    ("any offer may be conditioned on your authorization to receive technology controlled under these U.S. export "
     "laws without sponsorship for an export license.", "will_not_sponsor_or_opt", "unstated"),
    # Assurant d3562842: a refusal that names no OPT
    ("Note: You must be work authorized in the United States on a full-time basis without the need for employer "
     "sponsorship now or in the future.", "will_not_sponsor_or_opt", "will_not_sponsor"),
    # American Family 26a963c1: bare refusal of sponsorship still counts, without OPT
    ("- Sponsorship will not be considered for this position unless specified in the posting", "will_not_sponsor_or_opt",
     "will_not_sponsor"),
    # OPT named in the sentence after the refusal (Ankura 26c5d37a), or inside it (Newrez df045001)
    ("Interns are not eligible for U.S. immigration sponsorship, work visa or any other temporary work authorization (i.e. "
     "OPT, CPT).", "will_not_sponsor", "will_not_sponsor_or_opt"),
    ("Applicants who require sponsorship now or in the future are not eligible, this includes F-1 students on OPT or STEM OPT.",
     "will_not_sponsor", "will_not_sponsor_or_opt"),
    # "opt out" is no OPT
    ("We cannot sponsor visas for this role. You may opt out of AI screening.", "will_not_sponsor_or_opt", "will_not_sponsor"),
    # ATS labels with the value on the next line (Merck 61dda24d, Flex 1ce187ae); a benefit bullet (Fab2 0859ff2b)
    ("Relocation:\nDomestic\n\nVISA Sponsorship:\nYes\n\nTravel Requirements:\n10%", "sponsors", "sponsors"),
    ("Is Sponsorship Available?\nNo\n\nFlex is an Equal Opportunity Employer", "unstated", "will_not_sponsor"),
    ("Benefits:\n - Paid Time Off\n - Visa Sponsorship\n - Medical, Dental, and Vision insurance", "sponsors", "sponsors"),
    ("Sponsorship is available for this position and other select roles.", "sponsors", "sponsors"),  # SingleStore
])
def test_sponsorship_needs_visa_sponsorship_sentences(text, value, want):
    assert repair_sponsorship(value, text) == want


def test_nested_skill_lists_are_flattened():
    must, nice = clean_skills([["Swift", ["Kotlin", "C++"]], ["iOS", "Android"]], [["collaboration"], ["Kotlin"], ["Rust"]])
    assert must == [["Swift", "Kotlin", "C++"], ["iOS", "Android"]]  # Argmax db2ae7c8
    assert nice == ["collaboration", "Rust"]  # Medline e296a9f7: no "['collaboration']"
    card = {k: None for k in OUTPUT_KEYS}
    with pytest.raises(ValueError, match="string"):
        parse_card("x", json.dumps({**card, "must_skills": [["Swift", {"name": "Kotlin"}]], "nice_skills": []}))


@pytest.mark.parametrize("text,enrolled,degree,want", [
    # no study asked: the master's or PhD the model read as study is the degree required
    ("The ideal candidate will possess most of the following:\n- PhD degree in computer science or a related field",
     "doctoral", "none", ("none", "doctoral")),  # Seven Research e1fc4e73
    ("Requirements:\n- Master’s degree in Electrical/Electronics/Computer Engineering with 0-5 years of experience.",
     "master", "bachelor", ("none", "master")),  # Ambarella c9f13018
    ("Qualifications:\n- PhD or master’s in computer science or electrical engineering or related field.",
     "doctoral", "master", ("none", "master")),  # Ambarella c4841e3b
    ("- PhD / MSC preferred. BSc with high gpa, multiple majors, notable projects is also accepted",
     "doctoral", "none", ("none", "none")),  # Wurl adb0bde1: no held degree read from a preferred line
    ("The ideal candidate is a PhD / MSc in Statistics.\n- PhD / MSC preferred. BSc with high gpa is also accepted",
     "doctoral", "none", ("none", "bachelor")),
    # study cues
    ("- Education: Current Master's or Ph.D. candidate in Robotics", "master", "bachelor", ("master", "bachelor")),  # Bot Auto
    ("PhD student who already holds a master's degree", "doctoral", "master", ("doctoral", "master")),
    ("Pursuing an advanced degree Phd preferred in Computer Science", "doctoral", "none", ("master", "bachelor")),  # Klaviyo
    ("- Pursuing an advanced degree (PhD strongly preferred) in Computer Science", "doctoral", "master",
     ("master", "bachelor")),  # Ancestry 6d124577
    ("Virtu's internship program offers graduate students an opportunity.\n\nREQUIREMENTS\n\n- Advanced degree "
     "(preferably PhD) in Science, Math, Engineering", "doctoral", "master", ("master", "bachelor")),  # Virtu 462f2077
    ("What you need:\n- Current PhD student in statistics\n\nWe’d love if you had:\n- An advanced degree in a "
     "technical field", "doctoral", "none", ("doctoral", "bachelor")),  # 9e38ba0d: a wish, not the study asked
    # study the posting only prefers
    ("What You'll Need\n- Master’s in Computer Science, Engineering, or a related major; PhD candidates preferred",
     "doctoral", "master", ("none", "master")),  # AspenTech c671ab74
    ("- Education: Bachelor of Arts or Science in Economics, or related field. Master’s degree candidates preferred.",
     "bachelor", "bachelor", ("none", "bachelor")),  # SouthState fcc5cf43
    ("- Currently pursuing an advanced degree (Master's or PhD preferred) in Computer Science", "master", "bachelor",
     ("master", "bachelor")),
    # graduate students in a blurb do not lower a PhD requirement
    ("We seek an exceptional graduate student intern.\nRequired Education\n- Must be pursuing a PhD (enrolled student).",
     "doctoral", "master", ("doctoral", "bachelor")),  # Genentech 2e55c5cc
])
def test_enrollment_and_the_degree_held(text, enrolled, degree, want):
    d = _enroll(text, enrolled_in=enrolled, degree_level=degree)
    assert (d["enrolled_in"], d["degree_level"]) == want


@pytest.mark.parametrize("text,must,want", [
    ("Requirements:\n- Experience with high level software languages. Useful examples include C, Microsoft VBA, "
     "National Instruments LabVIEW, and Matlab.", [["C"], ["Microsoft VBA"], ["National Instruments LabVIEW"], ["Matlab"]],
     [["C", "Microsoft VBA", "National Instruments LabVIEW", "Matlab"]]),  # ADI 13cc39f8
    ("Requirements:\n- Strong interest in writing automation scripts in Python, and familiarity with test environment "
     "tools such as Git, Confluence, and Jira.", [["Python"], ["Git"], ["Confluence"], ["Jira"]],
     [["Python"], ["Git", "Confluence", "Jira"]]),  # Marvell 05682b26
    ("Additional Required Qualifications\n- Proficiency in Python for data analysis (e.g., pandas, NumPy, Matplotlib).",
     [["Python"], ["pandas"], ["NumPy"], ["Matplotlib"]], [["Python"]]),  # National Lab 8ac76c64: examples of Python
    ("Requirements:\n- Computer skills: knowledge of Microsoft Office Suite (Outlook, Word, Excel, Teams, PowerPoint)",
     [["Microsoft Office Suite"], ["Outlook"], ["Word"], ["Excel"], ["Teams"], ["PowerPoint"]], [["Microsoft Office Suite"]]),
    # a name the requirements also ask for on its own stays its own group
    ("Requirements:\n- Proficiency in Git.\n- Familiarity with tools such as Git, Confluence, and Jira.",
     [["Git"], ["Confluence"], ["Jira"]], [["Git"], ["Confluence", "Jira"]]),
    # "a typical day may include" is no choice: an and-list there is still split
    ("Requirements:\n- Experience with VHDL and Verilog.", [["VHDL", "Verilog"]], [["VHDL"], ["Verilog"]]),
])
def test_example_lists_are_one_requirement(text, must, want):
    assert clean_card({"must_skills": must, "nice_skills": []}, text)["must_skills"] == want


def test_example_lists_need_a_category_and_abbreviations_join_their_long_form():
    t = "Requirements:\n- Demonstrated full-stack development experience (React, Node.js, Python)"  # fresh 1e9bfb18
    assert clean_card({"must_skills": [["React"], ["Node.js"], ["Python"]], "nice_skills": []}, t)["must_skills"] == \
        [["React"], ["Node.js"], ["Python"]]
    t = "Requirements:\n- Experience with direct preference optimization (DPO).\n- Arduino (C++) projects."
    must = [["direct preference optimization"], ["DPO"], ["Arduino"], ["C++"]]
    assert clean_card({"must_skills": must, "nice_skills": []}, t)["must_skills"] == \
        [["direct preference optimization", "DPO"], ["Arduino"], ["C++"]]


def test_preferred_years_are_no_minimum_and_a_city_takes_the_state_the_text_gives():
    t = ("Basic Qualifications:\n- 18 years or older\n- Currently enrolled in a Master's program\n\n"
         "Preferred Qualifications:\n- 1 or more years relevant work experience")  # Amgen 27460c05
    assert clean_card({"min_years": 1}, t)["min_years"] == 0
    t = "Requirements:\n- 2+ years of experience with Python\n\nPreferred:\n- 4+ years of experience"
    assert clean_card({"min_years": 4}, t)["min_years"] == 2
    assert clean_card({"min_years": 2}, t)["min_years"] == 2
    t = "You will be required to work in the office at the Fort Myers, Florida location."  # Watts 9ad5d044
    d = clean_card({"locations": ["Fort Myers"], "work_mode": "onsite"}, t, ["Fort Worth, TX"])
    assert d["locations"] == ["Fort Myers, FL"]
    d = clean_card({"locations": ["Fort Myers"], "work_mode": "onsite"}, "Work in Fort Myers, or remotely.", ["Fort Worth, TX"])
    assert d["locations"] == ["Fort Myers"]  # "or" is no state


@pytest.mark.parametrize("text", [
    "Qualifications:\n- Bachelor’s degree in Computer Science, Machine Learning, Systems, or a related field",  # Unity 10418cfa
    "Qualifications:\n\nRequired:\n- Bachelor’s Degree in Computer Science; Software Engineering or related area",  # CACI 820f340e
    "Required\n- Bachelor's degree in Business, Finance, Economics, or a related quantitative field, preferably a May 2028 grad",  # 0421e841
    "Requirements:\n- BS, MS or PhD in Electrical Engineering, Computer Science, Computer Engineering (obtained)",  # Mill a9414ec5
    "Requirements:\n- PhD, master’s, or bachelor’s degree in computer science or a related field",  # Seven Research bd7fc029
])
def test_a_bachelors_the_model_read_as_study_is_the_degree_held(text):
    d = _enroll(text)
    assert (d["enrolled_in"], d["degree_level"]) == ("none", "bachelor")


@pytest.mark.parametrize("text", [
    "Required\n- Hands-on experience with PCBs\n\nNice to have\n- Bachelor's degree or higher in Electrical Engineering",  # WindBorne 88ed3204
    "Bachelor’s or higher degree related to the internship area should be under pursuit or have been obtained",  # ONEOK 7edf2e03
    "· This position requires either equivalent practical experience or a Bachelor’s degree in a related field",  # Affirm 55d83f42
    "- Completion of at least two years of an undergraduate or graduate degree program in a quantitative field",  # Voloridge e08774ee
])
def test_no_held_bachelors_from_preferred_study_equivalent_or_years_of_study(text):
    d = _enroll(text)
    assert (d["enrolled_in"], d["degree_level"]) == ("none", "none")


@pytest.mark.parametrize("text", [
    "- Working on a BS degree in a computer related field (e.g. CS, MIS)",  # Genuine Parts 1334781b
    "- Bachelor’s degree in computer science, Engineering and equivalent program (3rd year and later)",  # Danaher 0c0c50f3
    "- Are persuing a Bachelors/Masters in Electrical Engineering, Mechatronics Engineering",  # Human Computer Lab 93005bff
])
def test_student_wording_working_on_third_year_and_persuing(text):
    assert _enroll(text)["enrolled_in"] == "bachelor"


def test_student_cue_lists_and_seeking_a_degree():
    from matcher.blueprint import student_cue
    assert student_cue("- Current undergraduate, master's, or PhD students; postdoctoral researchers")  # Rainmaker a37b51de
    assert student_cue("- Seeking a Bachelor’s degree in Computer Science")  # Assured Guaranty 125bd988
    # Rainmaker also admits recent graduates and postdocs: no study is required
    t = ("- Current undergraduate, master's, or PhD students; postdoctoral researchers; recent graduates; and other "
         "early-career researchers are all eligible")
    assert _enroll(t)["enrolled_in"] == "none"


def test_or_lists_without_the_oxford_comma_offer_a_choice():
    t = "Requirements:\n- Candidate must have demonstrated 1 year programming experience in C/C++, Java or Python."  # Intel 2d77d594
    assert merge_anyof([["C", "C++"], ["Java"], ["Python"]], t) == [["C", "C++", "Java", "Python"]]
    t = ("Requirements:\n- Coursework or practical experience in linear algebra, three-dimensional geometry, kinematics, "
         "dynamics, controls or motion planning.")  # Valstad e1ae2c85
    assert len(merge_anyof([["linear algebra"], ["three-dimensional geometry"], ["kinematics"], ["dynamics"],
                            ["controls"], ["motion planning"]], t)) == 1
    t = "Requirements:\n· 1 years of research, project, or professional experience with LLMs or Agents."  # Brightstar d96a5b80
    assert merge_anyof([["LLMs"], ["Agents"]], t) == [["LLMs", "Agents"]]


def test_or_lists_the_model_read_locally_or_that_mix_lists_stay_apart():
    # the model grouped the two items around "or": "HTML, CSS and (JavaScript or TypeScript)"
    t = "Requirements:\n- Working knowledge of HTML, CSS, JavaScript or TypeScript, and a modern web framework."
    assert merge_anyof([["HTML"], ["CSS"], ["JavaScript", "TypeScript"]], t) == [["HTML"], ["CSS"], ["JavaScript", "TypeScript"]]
    # "machine learning" inside "efficient machine learning" is no alternative to foundation models
    t = ("Requirements:\n- Strong knowledge of Machine Learning\n"
         "- Prior research experience with training foundation models or efficient machine learning")  # Together 464ccc3f
    assert merge_anyof([["Machine Learning"], ["foundation models"]], t) == [["Machine Learning"], ["foundation models"]]
    t = ("Requirements:\n- Experience with programming languages and data tools such as Python, Matlab, C++, C or Java as well "
         "as SQL, Teradata, Tableau, Gephi or Airflow.")  # Boeing 6aef6415
    must = [["Python", "Matlab", "C++", "C", "Java"], ["SQL", "Teradata", "Tableau", "Gephi", "Airflow"]]
    assert merge_anyof(must, t) == must
    t = "Requirements:\n- Proficiency in Python (NumPy, Pandas, SciPy, Matplotlib or similar) or R or C++"  # DV cca94bc1
    must = [["Python"], ["NumPy", "Pandas", "SciPy", "Matplotlib"], ["R", "C++"]]
    assert merge_anyof(must, t) == must


def test_and_or_between_two_example_lists_is_one_choice():
    t = ("Requirements:\n- Familiarity with ML frameworks (e.g., PyTorch, TensorFlow) and/or distributed systems "
         "(e.g., Ray, Spark)")  # Unity 10418cfa
    assert merge_anyof([["Python"], ["PyTorch", "TensorFlow"], ["Ray", "Spark"]], t) == \
        [["Python"], ["PyTorch", "TensorFlow", "Ray", "Spark"]]
    t = "Requirements:\n- Object-Oriented Design and Programming using C/C++ and/or C# Unit testing (GoogleTest, NUnit or similar)"
    must = [["C", "C++", "C#"], ["GoogleTest", "NUnit"]]
    assert merge_anyof(must, t) == must  # RTX 3633ac91: no example list before "and/or"


def test_an_or_list_item_left_optional_takes_its_siblings_along():
    t = ("Qualifications:\n- Strong foundation in algorithms and data structures, with proficiency in C++, Python, or another "
         "object-oriented programming language.\n- Experience with algorithms, software development, or systems "
         "engineering fundamentals")  # GM 037d751e
    d = clean_card({"must_skills": [["C++", "Python"], ["algorithms"], ["software development"], ["systems engineering"]],
                    "nice_skills": []}, t)
    assert d["must_skills"] == [["C++", "Python"]] and "systems engineering" in d["nice_skills"]
    t = ("Requirements:\n- Understanding of digital logic, RTL, simulation, testbenches, interfaces, or operating-system "
         "concepts.")  # Astera 37bb29c7: the model kept simulation optional
    d = clean_card({"must_skills": [["RTL"]], "nice_skills": ["simulation"]}, t)
    assert d["must_skills"] == [] and "RTL" in d["nice_skills"]
    # GE Vernova a42bb64b: generative AI is also asked for outside the or-list, so it stays
    t = ("Requirements:\n- Familiarity with machine learning concepts and generative AI technologies.\n"
         "- Experience with large language models, RAG, or generative AI.")
    d = clean_card({"must_skills": [["generative AI"]], "nice_skills": ["large language models", "RAG"]}, t)
    assert d["must_skills"] == [["generative AI"]]


def test_visa_support_wording_is_a_no_sponsorship_statement():
    t = ("Therefore, this temporary internship is open only to individuals who are eligible for employment in the United "
         "States and who would not require visa support including H-1B, CPT/OPT/STEM, etc., now or in the future.")  # Cigna
    assert repair_sponsorship("unstated", t) == "will_not_sponsor_or_opt"
    assert repair_sponsorship("unstated", "- Must not require visa support to work in the USA") == "will_not_sponsor"  # Autostore


def test_a_city_the_model_made_up_for_a_state_only_ats_location_is_dropped():
    d = clean_card({"locations": ["Atlanta, GA"], "work_mode": "onsite"}, "Build things.", ["Georgia"])  # Veeam 8b79a805
    assert d["locations"] == []
    d = clean_card({"locations": ["Washington, DC", "Mountain View, CA"], "work_mode": "onsite"},
                   "Work in our offices.", ["Washington", "Mountain View, CA"])  # Unity 10418cfa
    assert d["locations"] == ["Mountain View, CA"]
    d = clean_card({"locations": ["Atlanta, GA"], "work_mode": "onsite"}, "You will sit in our Atlanta office.", ["Georgia"])
    assert d["locations"] == ["Atlanta, GA"]  # the body names the city


def test_pay_under_the_floor_needs_a_stated_period():
    d = clean_card({"salary_min": 10000, "salary_max": 10000}, "The pay range is\n$10,000—$10,000 USD")  # Kodiak
    assert d["salary_min"] is None and d["salary_max"] is None
    d = clean_card({"salary_min": 18000, "salary_max": 18000}, "$1,500 monthly stipend")  # Arkansas e4f4a19a
    assert d["salary_min"] == 18000
    d = clean_card({"salary_min": 17160, "salary_max": None}, "$8.25/hour based on experience")
    assert d["salary_min"] == 17160


def test_plain_list_is_no_choice_when_its_items_are_asked_for_together():
    t = ("Requirements:\n- Completion of three years of a degree program in Math, Statistics, Physics or equivalent field\n"
         "- Demonstrated aptitude in Math and Statistics")  # Voloridge d429d89b
    assert merge_anyof([["Python"], ["Math"], ["Statistics"]], t) == [["Python"], ["Math"], ["Statistics"]]


def test_an_or_list_that_runs_into_a_second_list_finds_no_optional_sibling():
    t = ("Requirements:\n- Experience with Verilog, SystemVerilog, or VHDL through academic, research, or project work")
    d = clean_card({"must_skills": [["Verilog", "SystemVerilog", "VHDL"]], "nice_skills": ["research"]}, t)  # Micron 5a07cf07
    assert d["must_skills"] == [["Verilog", "SystemVerilog", "VHDL"]]


def test_or_sibling_check_reads_a_slash_compound_as_both_names():
    t = ("Basic Qualifications:\n- 3+ months of software development, hardware engineering, or chip design/development "
         "experience")  # SpaceX 92effc23
    d = clean_card({"must_skills": [["hardware engineering", "chip design"], ["chip development"]], "nice_skills": ["software development"]}, t)
    assert d["must_skills"] == []


OSHKOSH = ("Certain positions with Oshkosh Corporation require access to controlled goods and technologies subject to the "
           "International Traffic in Arms Regulations or the Export Administration Regulations. Applicants for these "
           "positions may need to be \"U.S. Persons,\" as defined in these regulations. Generally, a \"U.S. Person\" is a "
           "U.S. citizen, lawful permanent resident, or an individual who has been admitted as a refugee or granted asylum.")
NLR_BADGE = ("Badging Requirement\nAll employees must also be able to obtain and maintain a federal Personal Identity "
             "Verification (PIV) card as required by Homeland Security Presidential Directive 12 (HSPD-12), which includes "
             "a favorable background investigation.")


@pytest.mark.parametrize("text", [
    OSHKOSH,  # Oshkosh 521c0e75 and its 11 sister postings: "certain positions ... may need to be"
    "Must be a U.S. Citizen or National of the U.S., an alien lawfully admitted for permanent residence, or an alien "
    "authorized to work in the U.S. for this employer.",  # UPS 2c43392f
    "Applicants selected for this position may be required to obtain and maintain a government security clearance",  # Draper be5ac22c
    NLR_BADGE,  # National Laboratory of the Rockies ff8232ae (its sister 3f47069f is False)
    "We build connected devices for public safety. You will design and test circuit boards.",  # Axon ae5e81fe: no rule at all
    "All employment decisions are made without regard to race, national origin or citizenship status. "
    "No security clearance is required for this role.",
])
def test_a_model_citizens_only_without_a_stated_rule_is_undone(text):
    assert _auth(text, citizens_only=True)["citizens_only"] is False


@pytest.mark.parametrize("text", [
    # Moderna 16e7a042: "may involve" in one sentence, a hard rule in the next
    "This position may involve access to technology that is subject to U.S. export control laws. Due to the nature of "
    "the work, only individuals who qualify as U.S. persons (citizens, permanent residents, asylees, or refugees) are "
    "eligible for this position.",
    "Requirements:\n- U.S. Citizen or Permanent Residency",  # Federal Reserve aaeb44bf
    "- U.S. permanent residency required",  # Terranova 168be0d0
    "US Citizenship and ability to get a US Security Clearance",  # Meridian ce5ce805
    # The Nuclear Company e53bf9a1: the "does not provide visa sponsorship" clause is no negation of the rule
    "U.S. Person status (U.S. citizen or lawful permanent resident) is required, and TNC does not provide visa "
    "sponsorship for these roles.",
    "Because our employees are provided access to export-controlled items, our policy is to only hire “U.S. "
    "persons” who are permitted to have access to our technology without an export license.",  # Varda 892a9689
])
def test_a_model_citizens_only_with_a_stated_rule_stays(text):
    assert _auth(text, citizens_only=True)["citizens_only"] is True


@pytest.mark.parametrize("text", [
    # Apex ad59a004 and Hermeus 768502d1: the ITAR wording that also admits anyone who can obtain an export
    # authorization reads as citizens_only, as for SpaceX, Hadrian and K2 (the convention the pairs use)
    "To be eligible for this role, applicants must either (1) qualify as a U.S. Person, meaning a U.S. citizen or "
    "national, a lawful permanent resident, or (2) be able to obtain the required export authorizations from the U.S. "
    "Department of State.",
    "The person hired will have access to information and items subject to U.S. export controls, and therefore, must "
    "either be a “U.S. person” as defined by 22 C.F.R. § 120.62 or otherwise eligible for deemed export licensing.",
])
def test_itar_or_export_authorization_wording_is_citizens_only_both_ways(text):
    assert _auth(text)["citizens_only"] is True
    assert _auth(text, citizens_only=True)["citizens_only"] is True


def test_anyof_header_with_a_noun_collects_the_bullets_below():
    # Astranis 32ef9b27: "one or more of the following areas:" heads the list; before, every area was required
    t = ("Requirements:\n- Solid EE foundation in one or more of the following areas:\n\n- analog design (op amps, "
         "transistors, RLC filters)\n\n- digital design (FPGAs, SOCs, SERDES, DDR, LVDS)\n\n- mixed signal design (ADCs, "
         "DACs, power integrity)\n\n- RF design (Lineups, s-parameters, impedance matching)\n\nBonus\n- Experience with Python")
    got = merge_anyof([["analog design", "op amps"], ["digital design", "FPGAs"], ["mixed signal design", "ADCs"],
                       ["RF design", "s-parameters"], ["Python"]], t)
    assert got == [["analog design", "op amps", "digital design", "FPGAs", "mixed signal design", "ADCs", "RF design",
                    "s-parameters"], ["Python"]]
    # Acron 52d30fe4: "Analog or digital circuit design" names both groups' names; the next requirement ends the list
    t = ("- Foundational knowledge in one or more of the following areas:\n\n- Analog or digital circuit design\n\n"
         "- Electronic hardware development\n\n- Embedded systems or computer architecture\n\n"
         "- Basic understanding of the engineering development lifecycle")
    assert merge_anyof([["Analog circuit design", "Digital circuit design"], ["Electronic hardware development"],
                        ["Embedded systems", "Computer architecture"], ["Engineering development lifecycle"]], t) == \
        [["Analog circuit design", "Digital circuit design", "Electronic hardware development", "Embedded systems",
          "Computer architecture"], ["Engineering development lifecycle"]]


def test_or_bullets_are_alternatives_to_the_bullet_above():
    # Relay c4d18418
    t = ("What You Will Bring\n- Strong proficiency in C, Java, or Kotlin.\n\n- Or experience in embedded systems "
         "programming, e.g., NRF or ST-based products running Zephyr, ThreadX, or Linux.\n\n- Or solid experience with "
         "scripting languages such as Bash, Python, or similar.\n\n- Familiarity with IP networking concepts is preferred.")
    assert merge_anyof([["C", "Java", "Kotlin"], ["Zephyr", "ThreadX", "Linux"], ["Bash", "Python"]], t) == \
        [["C", "Java", "Kotlin", "Zephyr", "ThreadX", "Linux", "Bash", "Python"]]


def test_and_or_between_two_example_lists_is_one_choice():
    # Intel 8d699dd6
    t = "- Graphics APIs such as OpenGL, Vulkan, or DirectX, and/or shading languages including GLSL and HLSL."
    assert merge_anyof([["shading", "OpenGL", "Vulkan", "DirectX"], ["GLSL", "HLSL"]], t) == \
        [["shading", "OpenGL", "Vulkan", "DirectX", "GLSL", "HLSL"]]
    # still inside the choice only: "simulation tools such as FineSim and/or HSPICE" starts after "such as"
    assert merge_anyof([["SPICE"], ["FineSim"], ["HSPICE"]], "- SPICE; simulation tools such as FineSim and/or HSPICE") == \
        [["SPICE"], ["FineSim", "HSPICE"]]


def test_one_or_more_areas_of_a_list_with_a_parenthesis():
    # NVIDIA d58843db: the "and" and commas inside the parenthesis belong to its item
    t = ("- Depth in one or more areas of high-performance circuit design (PLLs and clocking, SerDes, power "
         "delivery/regulation), advanced packaging technologies, or thermal analysis/design.")
    assert merge_anyof([["PLLs", "clocking", "SerDes", "power regulation"], ["advanced packaging technologies"],
                        ["thermal analysis", "thermal design"]], t) == \
        [["PLLs", "clocking", "SerDes", "power regulation", "advanced packaging technologies", "thermal analysis",
          "thermal design"]]
    # a far-apart word pair is no name: "agentic, tool-using, or multi-turn conversational LLM systems"
    t = "- Familiarity with large language models, including agentic, tool-using, or multi-turn conversational LLM systems"
    assert merge_anyof([["Agentic systems"], ["Python"]], t + "\n- Python and/or Bash") == [["Agentic systems"], ["Python"]]


@pytest.mark.parametrize("raw,want", [("director", "lead"), ("Director", "lead"), ("junior", "entry"), ("Senior", "senior"),
                                      ("whatever", "mid")])
def test_seniority_outside_the_levels_is_snapped(raw, want):
    assert parse_card("x", json.dumps({**EXAMPLE, "seniority": raw})).seniority == want
