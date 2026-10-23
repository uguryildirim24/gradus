import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import render_cards as rc  # noqa: E402

from matcher import textclean as tc  # noqa: E402
from matcher.onet import DEFAULT_DIR as ONET_DIR  # noqa: E402

# These gates read the O*NET software names (data/raw/open/onet, gitignored); without them they skip.
needs_onet = pytest.mark.skipif(not ONET_DIR.exists(), reason="needs O*NET under data/raw/open/onet")

CAND = dict(field="Lab and life sciences", degree_level="bachelor", degree_field="Biology", graduation_year=2026, years=0,
            skills=["Chemistry"], licenses=[], current_location="Boston, MA", education_country="United States", language="English")
POST = dict(source_posting_id="x:1", title="Lab Intern", industry="healthcare", seniority="entry", degree_level="none", enrolled_in="bachelor",
            degree_fields=[], min_years=0, must_skills=[["Microsoft Word", "Microsoft Outlook"], ["Chemistry"]], nice_skills=["Biology"],
            license=None, locations=["Boston, MA"], work_mode="onsite", sponsorship="unstated", citizens_only=False, e_verify=False,
            salary_min=41600, salary_max=52000, employment_type="internship")
# the rest of POST's facts, for test texts that must pass the presence gates (posting_missing_facts)
FACTS = " This healthcare internship is onsite in Boston, MA and pays $41,600 to $52,000 a year. Biology is a plus."
STUDENT = " Open to students currently enrolled in a bachelor's program."


def test_finished_degree_is_stated_as_finished():
    ctx = rc.context_of("resumes", CAND)
    assert "Finished" in ctx["education_status"] and "never 'expected'" in ctx["education_status"]
    assert ctx["years_of_experience"] == 0 and "graduation_year" not in ctx
    future = rc.context_of("resumes", CAND | {"graduation_year": 2028})
    assert "Still enrolled" in future["education_status"] and "expected graduation in 2028" in future["education_status"]


def test_degree_none_has_no_degree_or_year():
    ctx = rc.context_of("resumes", CAND | {"degree_level": "none", "degree_field": "", "graduation_year": 2026})
    assert "degree" not in ctx and "degree_field" not in ctx and "No college degree" in ctx["education_status"]


def test_any_of_group_is_written_as_alternatives():
    ctx = rc.context_of("postings", POST)
    assert ctx["required_skills"] == [{"any_one_of": ["Microsoft Word", "Microsoft Outlook"]}, "knowledge of chemistry"]
    assert "Required skill" not in json.dumps(ctx)


def test_unset_posting_fields_are_absent():
    ctx = rc.context_of("postings", POST)
    for k in ("min_years", "minimum_years_experience", "required_license", "work_authorization", "sponsorship", "citizens_only",
              "e_verify", "degree_level", "minimum_degree", "fields_of_study"):
        assert k not in ctx
    assert ctx["must_be_student_in"] == "a bachelor's or higher degree program"
    assert "none" not in json.dumps(ctx).lower().replace("nonexistent", "")


def test_posting_context_is_bare_facts_with_correct_articles():
    ctx = rc.context_of("postings", POST | {"degree_level": "associate", "enrolled_in": "none", "degree_fields": ["Biology"],
                                            "min_years": 3, "license": "RN license", "employment_type": "full_time"})
    assert ctx["minimum_degree"] == "associate degree" and ctx["fields_of_study"] == ["Biology"]
    assert ctx["minimum_years_experience"].startswith("at least 3 years of ") and ctx["required_license"] == "RN license"
    text = json.dumps(ctx)
    for stock in ("already hold", "must be one of", "currently enrolled in and pursuing", "a associate", "a master degree"):
        assert stock not in text
    assert rc.context_of("postings", POST | {"enrolled_in": "master"})["must_be_student_in"] == "a master's or doctoral program"


def test_company_name_is_given_and_varies():
    names = {rc.company_name(POST | {"source_posting_id": f"x:{i}"}) for i in range(200)}
    assert len({n.split()[0] for n in names}) >= 30
    assert rc.context_of("postings", POST)["company"] == rc.company_name(POST)


def test_work_auth_wordings_keep_their_scope():
    for stance, texts in rc.WORK_AUTH_TEXT.items():
        for t in texts:
            assert not rc.SCOPE_WORDS.search(t)
            assert ("OPT" in t) == (stance == "will_not_sponsor_or_opt"), t
    for t in rc.CITIZENS_TEXT + rc.E_VERIFY_TEXT:
        assert not rc.SCOPE_WORDS.search(t)


def test_work_auth_wordings_read_as_their_own_stance():
    """Every wording means its stance, and only that stance, to visibility (the training check), the render gates and
    the extraction's repair rules: a refusal never reads as an offer, an offer never as a refusal, OPT only in the
    OPT stance, citizenship only in the citizens wordings, no scope word anywhere."""
    from matcher import blueprint as bp
    from matcher import visibility as vis
    every = [t for ts in (*rc.WORK_AUTH_TEXT.values(), rc.CITIZENS_TEXT, rc.E_VERIFY_TEXT) for t in ts]
    assert len(every) == len(set(every))
    for stance, texts in rc.WORK_AUTH_TEXT.items():
        assert len(texts) >= 5, stance
        for t in texts:
            assert re.search(r"sponsor", t, re.I) and tc.is_work_auth(t) and rc.AUTH_TOPIC.search(t), t
            assert not (vis.CITIZENS.search(t) or vis.PERMANENT.search(t) or rc.SCOPE_WORDS.search(t)), t
            assert not re.search(r"citizen|permanent resid|green card|e-?verify", t, re.I), t
            card = {"sponsorship": stance, "citizens_only": False}
            assert vis.stance_conflict(card, t) is None, t
            if stance == "sponsors":
                assert rc.SPONSOR_OFFERED.search(t) or bp._YES_SPONSOR.search(t), t
                assert not (vis.NO_SPONSOR.search(t) or vis.NO_OPT.search(t) or bp._NO_SPONSOR.search(t)), t
                assert bp.repair_sponsorship("sponsors", t) == "sponsors", t
                assert vis.stance_conflict({"sponsorship": "will_not_sponsor"}, t) is None, t
            else:
                assert vis.NO_SPONSOR.search(t) and bp._NO_SPONSOR.search(t), t
                assert not (rc.SPONSOR_OFFERED.search(t) or bp._YES_SPONSOR.search(t)), t
                assert vis.stance_conflict({"sponsorship": "sponsors"}, t), t  # a sponsoring card could not carry it
                assert bp.repair_sponsorship("unstated", t) == stance, t
                assert bool(vis.NO_OPT.search(t)) == (stance == "will_not_sponsor_or_opt"), t
            d = {"sponsorship": stance, "citizens_only": False}
            bp.repair_work_auth(d, t)
            assert d == {"sponsorship": stance, "citizens_only": False}, t
            # the render gates read each wording as the card's own scope: no time scope, no student status, nothing
            # the posting gates flag on a card of this stance (r4: 6 of 39 added 'now or in the future')
            full = POST | {"sponsorship": stance, "enrolled_in": "none", "employment_type": "full_time"}
            assert not (rc.sponsor_time_scope(full, t) or rc.student_auth_refused(full, t) or rc.NEG_AUTH.search(t)), t
            assert not re.search(r"\bfuture\b|\bnow\b|I-983|student|\bCPT\b", t, re.I), t
            assert [x for x in rc.problems("postings", full, t) if not x.startswith("missing_") and x not in (
                "no_company_line", "company_not_given", "required_skill_not_stated_required", "any_of_as_all_of")] == [], t
    for t in rc.CITIZENS_TEXT:
        assert re.search(r"citizen|permanent resident|green card", t, re.I) and tc.is_work_auth(t), t
        assert not (re.search(r"sponsor|\bOPT\b|e-?verify", t, re.I) or vis.PERMANENT.search(t)), t
        d = {"sponsorship": "unstated", "citizens_only": True}
        bp.repair_work_auth(d, t)
        assert d["citizens_only"] is True and bp.repair_sponsorship("unstated", t) == "unstated", t
    for t in rc.E_VERIFY_TEXT:
        assert re.search(r"e-?verify", t, re.I), t
        assert not re.search(r"sponsor|\bOPT\b|citizen|permanent resid|green card|authori", t, re.I), t
        d = {"sponsorship": "unstated", "citizens_only": False}
        bp.repair_work_auth(d, t)
        assert d["citizens_only"] is False and vis.stance_conflict(d, t) is None, t


def test_work_auth_wordings_spread_over_the_check_batch():
    # with three wordings a stance one wording reached 21% of the check batch; picks are by card hash
    cards = [POST | {"source_posting_id": f"x:{i}", "sponsorship": "will_not_sponsor"} for i in range(600)]
    got = Counter(rc.context_of("postings", c)["work_authorization"][0] for c in cards)
    assert set(got) == set(rc.WORK_AUTH_TEXT["will_not_sponsor"]) and max(got.values()) / 600 < 0.25


def test_internship_terms_have_not_started():
    assert "Fall 2026" not in rc.TERMS and all(int(t.split()[1]) >= 2027 for t in rc.TERMS)


def test_internship_pay_is_hourly_with_the_yearly_figure():
    assert rc.context_of("postings", POST)["pay"] == "$20.00 to $25.00 per hour (about $41,600 to $52,000 a year)"
    assert rc.context_of("postings", POST | {"employment_type": "full_time"})["pay"] == "$41,600 to $52,000 a year"
    assert "pay" not in rc.context_of("postings", POST | {"salary_min": None, "salary_max": None})


def test_fake_contact_share_and_redaction_round_trip():
    cards = [CAND | {"years": i, "skills": [f"s{i}"]} for i in range(1000)]
    with_c = [c for c in cards if rc.fake_contact(c)]
    assert 0.10 < len(with_c) / 1000 < 0.20
    c = rc.fake_contact(with_c[0])
    c = rc.fake_contact(with_c[0])
    block = "\n".join([c["name"], *(c.get("address", "").split("\n")), f"{c['email']} | {c['phone']}", "Boston, MA", "", "SUMMARY", "Chemist."])
    out = tc.redact(block)
    assert out.startswith("Candidate") and c["name"] not in out and c["email"] not in out and not tc.has_contact(out)
    assert rc.context_of("resumes", with_c[0])["contact"] == c
    one = [rc.fake_contact(x) for x in with_c if rc.fake_contact(x).get("layout") == "one line"]
    assert 0.3 < len(one) / len(with_c) < 0.7
    for c in one[:20]:
        line = " | ".join([c["name"], "Boston, MA", c["phone"], c["email"], *([c["linkedin"]] if "linkedin" in c else [])])
        out = tc.redact(line + "\n\nSUMMARY\nChemist.")
        assert out.startswith("Candidate\nBoston, MA\n") and c["name"] not in out and not tc.has_contact(out), out


def test_stale_expected_check():
    assert rc.stale_expected("B.S. Biology, Expected May 2026")
    assert rc.stale_expected("Biology, in progress, 2024 - 2026")
    assert not rc.stale_expected("B.S. Biology, May 2025\nB.S. Nursing, Expected May 2028")
    assert not rc.stale_expected("Pursuing a career in research.")


def test_alternatives_and_negation_checks():
    good = "You must have experience with Microsoft Word or Microsoft Outlook, plus Chemistry."
    bad = "Proficiency in Microsoft Word and Microsoft Outlook is mandatory."
    assert rc.posting_problems(POST, f"Lab Intern\n{rc.company_name(POST)}\n\n" + good + FACTS + STUDENT) == []
    assert "any_of_as_all_of" in rc.posting_problems(POST, "Lab Intern\nAcme Labs\n\n" + bad)
    p = rc.posting_problems(POST | {"salary_min": None, "salary_max": None, "enrolled_in": "none"},
                            "Lab Intern\nAcme Labs\n\n" + good + " No degree required. Pay is commensurate with experience. We do not sponsor visas.")
    assert {"neg_degree", "neg_pay", "work_auth_topic_unset"} <= set(p)
    assert "markdown_after_clean" not in rc.posting_problems(POST, "**Lab Intern**\nAcme Labs\n\n" + good)


def test_batch_tag_changes_with_the_contexts():
    assert rc.batch_tag(["a", "b"], ["x", "y"]) == rc.batch_tag(["a", "b"], ["x", "y"])
    assert rc.batch_tag(["a", "b"], ["x", "y"]) != rc.batch_tag(["a", "b"], ["x", "z"])


def test_job_dates_span_the_card_years():
    for y in range(0, 21):
        card = CAND | {"years": y, "skills": [f"s{y}"]}
        jobs = rc.job_dates(card)
        assert len(jobs) == (0 if y == 0 else 1 if y <= 2 else 2 if y <= 6 else 3)
        if y:
            assert jobs[0].endswith("to Present")
            text = "Candidate\nBoston, MA\n\nExperience\n" + "\n".join(f"Job {i}\nAcme\n{j.replace(' to ', ' – ')}"
                                                                         for i, j in enumerate(jobs))
            assert 0 <= rc.work_span_years(text) - y < 1, (y, jobs)
            assert not rc.years_mismatch(card, text)
    # the first start month varies by card (it was always October), and the whole years stay the card's
    starts = {rc.job_dates(CAND | {"years": 2, "skills": [f"s{i}"]})[-1].split(" to ")[0].split()[0] for i in range(200)}
    assert len(starts) == rc.EXTRA_MONTHS
    for i in range(300):
        card = CAND | {"years": 1 + i % 12, "skills": [f"t{i}"]}
        text = "Candidate\n\nExperience\n" + "\n".join(f"Job\n{j.replace(' to ', ' – ')}" for j in rc.job_dates(card))
        assert not rc.years_mismatch(card, text) and int(rc.work_span_years(text)) == card["years"]
    # years is a floor: 1.5 years of dates fail a years=2 card, 3.1 fail it too
    two = CAND | {"years": 2}
    assert rc.years_mismatch(two, "Candidate\n\nExperience\nJob\nApril 2025 – Present\n")
    assert rc.years_mismatch(two, "Candidate\n\nExperience\nJob\nAugust 2023 – Present\n")
    assert not rc.years_mismatch(two, "Candidate\n\nExperience\nJob\nMarch 2024 – Present\n")
    assert "jobs" not in rc.context_of("resumes", CAND)
    jobs = rc.context_of("resumes", CAND | {"years": 3})["jobs"]
    assert [j["dates"] for j in jobs] == rc.job_dates(CAND | {"years": 3})


def test_jobs_carry_titles_and_the_current_city():
    # r4 reader: 'Registered Nurse' with no license (cd_v_00286), the Present job in Riyadh for a Portland card
    # (cd_v_01354), a resident physician a year before the MD (cd_v_00823)
    card = CAND | {"field": "Nursing", "years": 8, "current_location": "Portland, OR", "graduation_year": 2015}
    jobs = rc.context_of("resumes", card)["jobs"]
    assert jobs[0]["employer_city"] == "Portland, OR" and all("employer_city" not in j for j in jobs[1:])
    assert all(j["title"] in rc.FIELD_TITLES["Nursing"] for j in jobs)
    rn = card | {"licenses": ["RN license"], "degree_field": "Nursing"}
    assert all(j["title"] in rc.LICENSE_TITLES["RN license"] for j in rc.jobs_of(rn))
    # a degree-given license: a job that starts before the graduation gets a plain title, and job_dates starts the
    # first job no earlier than July of the graduation year when the years allow it
    md = CAND | {"field": "Clinical healthcare", "degree_level": "doctoral", "degree_field": "Medicine", "graduation_year": 2016,
                 "years": 11, "licenses": ["Medical license"]}
    jobs = rc.jobs_of(md)
    assert jobs[-1]["dates"].split(" to ")[0].endswith("2015") and jobs[-1]["title"] in rc.FIELD_TITLES["Clinical healthcare"]
    assert jobs[0]["title"] in rc.LICENSE_TITLES["Medical license"]
    for i in range(60):
        ok = md | {"years": 10, "skills": [f"s{i}"]}
        first = rc.job_dates(ok)[-1].split(" to ")[0]
        assert rc.MONTHS.index(first.split()[0]) >= 6 and first.endswith("2016"), first
        assert all(j["title"] in rc.LICENSE_TITLES["Medical license"] for j in rc.jobs_of(ok))
    # the summary names the card's own skills
    assert set(rc.context_of("resumes", card)["summary_skills"]) <= set(card["skills"])


def test_job_titles_are_safe():
    """No plain title is a licensed title or names a skill; every licensed title is covered by its license."""
    from matcher.onet import LICENSE_EDUCATION
    bare = CAND | {"skills": [], "licenses": [], "degree_level": "bachelor", "degree_field": ""}
    real = [c for i, c in rc.load_cards("resumes").items() if c.get("field") not in rc.FIELD_TITLES]
    for card in [bare | {"field": f} for f in rc.FIELD_TITLES] + real:
        for t in rc._plain_titles(card):
            text = f"Candidate\nBoston, MA\n\nExperience\n{t}\nAcme, Boston, MA\nMay 2024 to Present\n- Did the work.\n"
            assert rc.protected_title(t) == [] and rc.title_without_license(card | {"licenses": []}, text) == [], t
            assert rc.lacked_skill_named(card, text) == [], (card["field"], t)
    assert set(LICENSE_EDUCATION) <= set(rc.LICENSE_TITLES)
    for lic, ts in rc.LICENSE_TITLES.items():
        for t in ts:
            text = f"Candidate\nBoston, MA\n\nExperience\n{t}\nAcme, Boston, MA\nMay 2024 to Present\n- Did the work.\n"
            assert rc.title_without_license(bare | {"licenses": [lic]}, text) == [], (lic, t)


RES = ("Candidate\nWashington, DC\n\nSummary\nNurse with two years in hospital care.\n\nEducation\nMaster of Science in Health "
       "Sciences\nGeorge Washington University, Washington, DC\nAugust 2026\n\nSkills\nMicrosoft Word\n\nProfessional Experience\n"
       "Registered Nurse\nCapitol Hill Medical Center\nJune 2024 – Present\n- Kept records in Microsoft Excel.\n\nStaff Nurse\n"
       "Potomac Valley Hospital\nJune 2022 – May 2024\n")


def test_years_gate_catches_dates_that_do_not_add_up():
    card = CAND | {"years": 2, "skills": ["Microsoft Word"], "current_location": "Washington, DC", "degree_field": "Health Sciences"}
    assert round(rc.work_span_years(RES), 2) == 4.33
    assert rc.years_mismatch(card, RES)
    assert not rc.years_mismatch(card | {"years": 4}, RES)
    # 0 years: a dated job anywhere but education or volunteering fails, a dated volunteer role does not
    zero = CAND | {"years": 0}
    assert rc.years_mismatch(zero, "Candidate\n\nAcademic Projects and Research\nGraduate Research Assistant\nKyoto University\n"
                                   "September 2024 – Present\n")
    assert not rc.years_mismatch(zero, "Candidate\n\nVolunteer Experience\nTutor\nJanuary 2024 – May 2024\n\n"
                                       "Education\nB.S.\nSeptember 2022 – May 2026\n")
    # a job title that contains a section word is not a heading
    t = "Candidate\n\nProfessional Experience\nCertified Nursing Assistant\nJune 2024 – Present\nSenior Training Coordinator\nJune 2022 – May 2024\n"
    assert round(rc.work_span_years(t), 2) == 4.33


@needs_onet
def test_resume_fact_gates():
    card = CAND | {"years": 4, "skills": ["Microsoft Word"], "current_location": "Washington, DC", "degree_field": "Health Sciences",
                   "degree_level": "master", "graduation_year": 2026}
    assert rc.extra_tools(card, RES) == ["Excel"] or rc.extra_tools(card, RES) == ["Microsoft Excel"]
    assert rc.extra_tools(card | {"skills": ["Microsoft Word", "Microsoft Excel"]}, RES) == []
    assert rc.extra_tools(card, "Proficient in Microsoft Office suite.") and \
        not rc.extra_tools(card | {"skills": ["Microsoft Office software"]}, "Proficient in Microsoft Office suite.")
    assert not rc.extra_tools(card | {"skills": ["Python"]}, "Wrote Python scripting tools.")
    assert not rc.city_missing(card, RES) and rc.city_missing(card | {"current_location": "Nashville, TN"}, RES)
    blank = card | {"degree_field": "", "field": "Social sciences"}
    assert rc.major_named(blank, RES) == ["Health"]
    assert rc.major_named(blank, "Education\nMaster of Arts, Social Sciences\nEmory University") 
    assert rc.major_named(blank, "Education\nMSN and BSN\nEmory University")
    assert not rc.major_named(blank, "Education\nMaster of Science\nOregon State University, Corvallis, OR\nMay 2024")
    assert not rc.major_named(card, RES)


def test_posting_gates_for_scope_duties_students_and_travel():
    card = POST | {"sponsorship": "will_not_sponsor", "enrolled_in": "none"}
    head = f"Lab Intern\n{rc.company_name(card)}\n\n"
    good = head + ("Requirements\n- Microsoft Word or Microsoft Outlook\n- Chemistry\n\n"
                   "We cannot sponsor employment visas for this role." + FACTS)
    assert rc.posting_problems(card, good) == []
    scoped = good + " Only candidates who already possess unrestricted work authorization should apply."
    assert "scope_word" in rc.posting_problems(card, scoped)
    duty = head + "You will apply your knowledge of Chemistry daily. You must use Microsoft Word or Microsoft Outlook."
    assert rc.required_as_duty_only(card, duty) == [["Chemistry"]]
    assert "student_topic_unset" in rc.posting_problems(card, good + " The ideal candidate is a student or recent graduate.")
    assert "travel_added" in rc.posting_problems(card, good + " You may travel between sites.")
    assert "travel_added" not in rc.posting_problems(card | {"title": "Travel Agent"}, good + " You will plan travel.")
    cit = card | {"citizens_only": True}
    assert "scope_word" not in rc.posting_problems(cit, good + " Applicants must hold permanent work authorization as citizens or permanent residents.")
    assert "company_not_given" in rc.posting_problems(card, "Lab Intern\nApex Labs\n\n" + good[len(head):])


def test_echo_rates_gate():
    r = rc.echo_rates("postings", ["Applicants must already hold a degree. Required skill: X", "Plain text here for a job."])
    assert r["stock_phrase_rates"]["already hold"] == 0.5 and r["stock_phrase_rates"][rc.REQUIRED_SKILL_LABEL] == 0.5
    assert not r["stock_phrase_gate"]
    assert rc.echo_rates("postings", ["Required Skills:\n- X"] + ["Plain text."] * 9)["stock_phrase_rates"][rc.REQUIRED_SKILL_LABEL] == 0
    assert rc.echo_rates("resumes", ["Nurse with five years in hospital care."] * 3)["stock_phrase_gate"]


def test_pick_ids_covers_every_stratum():
    cards = {f"rq_{i:05d}": POST | {"source_posting_id": f"x:{i}", "citizens_only": i % 7 == 0, "e_verify": i % 5 == 0,
                                    "sponsorship": ("sponsors", "will_not_sponsor", "will_not_sponsor_or_opt")[i % 3],
                                    "degree_fields": ["Biology"] if i % 4 == 0 else [],
                                    "license": "RN license" if i % 6 == 0 else None,
                                    "work_mode": ("onsite", "hybrid", "remote")[i % 3 if i % 2 else 0]}
             for i in range(300)}
    ids = rc.pick_ids("postings", cards, 30)
    assert len(ids) == len(set(ids)) >= 30 and ids == rc.pick_ids("postings", cards, 30)
    for name, test in rc.POSTING_STRATA.items():
        assert sum(bool(test(cards[i])) for i in ids) >= 3, name


@pytest.mark.skipif(not (rc.CARDS / "candidate_cards.jsonl").exists() or not (rc.CARDS / "requirement_cards.jsonl").exists(),
                    reason="needs data/")
def test_every_stratum_has_a_card_in_the_loaded_cards():
    # a stratum no card can fill keeps the sign-off from ever being written (blank_degree_field did)
    for kind, strata in (("resumes", rc.RESUME_STRATA), ("postings", rc.POSTING_STRATA)):
        cards = rc.load_cards(kind)
        if kind == "postings":
            cards = {i: c for i, c in cards.items() if i.startswith("rq_")}  # what pick_ids draws from
        empty = [n for n, test in strata.items() if not any(test(c) for c in cards.values())]
        assert not empty, (kind, empty)


def test_gates_are_enforced_at_download_and_at_read(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "OUT", tmp_path)
    card = POST | {"sponsorship": "will_not_sponsor", "enrolled_in": "none"}
    head = f"Lab Intern\n{rc.company_name(card)}\n\n"
    good = head + ("Requirements\n- Microsoft Word or Microsoft Outlook\n- Chemistry\n\n"
                   "We cannot sponsor employment visas for this role." + FACTS)
    bad = good + " Only candidates who already possess unrestricted work authorization should apply."
    cards = {"rq_good": card, "rq_bad": card | {"title": "Lab Assistant"}}
    assert rc.fatal("postings", card, good) == [] and "scope_word" in rc.fatal("postings", card, bad)
    assert rc.fatal("postings", card, "**x**") != [] and "markdown_after_clean" in rc.NONFATAL

    # download: the failing text goes to quarantine, not into the store
    sent = {i: rc.ctx_str("postings", c) for i, c in cards.items()}
    store: dict = {}
    rows = [{"row_id": "rq_good", "enhanced_completion": good}, {"row_id": "rq_bad", "enhanced_completion": bad}]
    new, quarantined = rc.gate_download("postings", cards, rows, sent, store, "ds1")
    assert [r["id"] for r in new] == ["rq_good"] and [r["id"] for r in quarantined] == ["rq_bad"]
    assert "scope_word" in quarantined[0]["tags"] and rc.ctx_hash(sent["rq_bad"]) not in store
    with rc.quarantine_path("postings").open("w") as f:
        f.write(json.dumps(quarantined[0]) + "\n")
    assert rc.tries("postings")[rc.ctx_hash(sent["rq_bad"])] == 1

    # read: a failing text already in the store (rendered before a gate existed) is left out and listed
    with (tmp_path / "postings.jsonl").open("w") as f:
        for i, t in (("rq_good", good), ("rq_bad", bad)):
            f.write(json.dumps({"id": i, "ctx": rc.ctx_hash(sent[i]), "text": t}) + "\n")
    assert set(rc.rendered_texts("postings", cards)) == {"rq_good"}
    dropped = rc.read_jsonl(tmp_path / "dropped_postings.jsonl")
    assert [r["id"] for r in dropped] == ["rq_bad"] and "scope_word" in dropped[0]["tags"]
    # a call on a subset keeps the drop rows of the other ids
    assert set(rc.rendered_texts("postings", {"rq_good": card})) == {"rq_good"}
    assert [r["id"] for r in rc.read_jsonl(tmp_path / "dropped_postings.jsonl")] == ["rq_bad"]
    assert set(rc.rendered_texts("postings", cards, clean=False)) == {"rq_good", "rq_bad"}
    rc.assert_clean("postings", cards, {"rq_good": good})
    try:
        rc.assert_clean("postings", cards, {"rq_bad": bad})
        raise AssertionError("assert_clean let a failing text through")
    except ValueError as e:
        assert "rq_bad" in str(e)


def test_redo_is_a_new_batch():
    ctx = rc.ctx_str("postings", POST)
    assert rc.batch_tag(["a"], [f"{ctx}#0"]) != rc.batch_tag(["a"], [f"{ctx}#1"])


def test_noun_needs_is_not_a_requirement_word():
    card = POST | {"must_skills": [["Biology"]], "enrolled_in": "none"}
    head = f"Lab Intern\n{rc.company_name(card)}\n\n"
    duty = head + "You will apply your background in Biology to understand the specific needs of different species."
    assert rc.required_as_duty_only(card, duty) == [["Biology"]]
    soft = head + "The ideal applicant would have experience with Biology."
    assert rc.required_as_duty_only(card, soft) == [["Biology"]]
    for ok in ("You need to know Biology.", "Candidates need a background in Biology.", "Biology is required.",
               "Requirements for this role include Biology.", "You will need solid Biology."):
        assert rc.required_as_duty_only(card, head + ok) == [], ok
    assert rc.required_as_duty_only(card, head + "Meet customer requirements using Biology.") == [["Biology"]]


def test_requiring_before_an_ability_states_a_requirement():
    # rq_01180 (r11 check render): the Zoom sentence was correct, but 'requiring' was no requirement word
    card = POST | {"must_skills": [["Zoom"]], "enrolled_in": "none"}
    head = f"Lab Intern\n{rc.company_name(card)}\n\n"
    zoom = "You will conduct telehealth sessions using Zoom, requiring comfort and competence with this digital platform."
    assert rc.required_as_duty_only(card, head + zoom) == []
    assert rc.required_as_duty_only(card, head + "Maintain records, requiring proficiency with Zoom.") == []
    # a task's need is still a duty
    assert rc.required_as_duty_only(card, head + "Resolve tickets requiring escalation over Zoom.") == [["Zoom"]]


def test_enrollment_in_a_degree_program_is_the_student_requirement():
    # rq_01285 (r11 check render): "Enrollment in a bachelor's or higher degree program" under the minimum qualifications
    head = f"Lab Intern\n{rc.company_name(POST)}\n\n"
    base = head + "You must have experience with Microsoft Word or Microsoft Outlook, plus Chemistry." + FACTS
    enroll = "\nMinimum qualifications\n- Enrollment in a bachelor's or higher degree program in any academic field\n"
    assert rc.posting_missing_facts(POST, base + enroll) == []
    for other in ("Enrollment in our benefits program starts on day one.", "Enrollment in our new graduate program is open.",
                  "Enrollment in our master plan is required.", "Open enrollment for health benefits runs in May."):
        assert rc.posting_missing_facts(POST, base + " " + other) == ["missing_student_requirement"], other


def test_reviewing_your_credentials_is_no_credential_requirement():
    # rq_01223~citizens (r11 check render): the closing line, in a sentence with 'necessary'
    doc = FT | {"degree_level": "doctoral"}
    close = ("If you hold the requisite doctorate and bring the necessary skills in media and technology to the table, "
             "we look forward to reviewing your credentials.")
    assert rc.license_added(doc, close) == []
    assert rc.license_added(doc, "The necessary credentials include a teaching license.") != []
    assert rc.license_added(doc, "We look forward to reviewing your credentials, including your CPR certification, "
                                 "which is necessary.") == ["CPR"]


def test_a_paragraph_that_leads_into_a_list_heads_it():
    # rq_00693: the list's lead-in is the last sentence of a long paragraph ending in ':'
    card = POST | {"must_skills": [["Biology"]], "enrolled_in": "none"}
    head = f"Lab Intern\n{rc.company_name(card)}\n\n"
    lead = ("Technical competencies are essential for success in this role. There are specific skills you need to possess to "
            "perform the duties effectively. The required skills for this position include:\n- Biology\n")
    assert rc.required_as_duty_only(card, head + lead) == []
    one = "In addition to the degree above, the following technical skills are required for success in this position:\n- Biology\n"
    assert rc.required_as_duty_only(card, head + one) == []
    # the lead-in heads its list only: a duty sentence after the list is not stated as required
    after = lead.replace("- Biology", "- Chemistry") + "You will use Biology in the lab every day.\n"
    assert rc.required_as_duty_only(card, head + after) == [["Biology"]]
    # a lead-in to duties or to preferred skills still leaves the list item unrequired
    for lead in ("You will join a busy team that runs assays for many clients every single day. Your daily responsibilities "
                 "will include:\n- Biology\n",
                 "We also value some extra knowledge that helps our team grow over time. While not mandatory, these are a plus:"
                 "\n- Biology\n"):
        assert rc.required_as_duty_only(card, head + lead) == [["Biology"]], lead


def test_contact_layout_key_is_an_artifact():
    assert rc.ARTIFACTS.search('Candidate\nlayout: one line') and rc.ARTIFACTS.search('{"layout": "one line"}')
    assert not rc.ARTIFACTS.search("Designed page layouts for the newsletter.")


def test_training_postings_puts_the_header_and_bank_on_gated_texts(monkeypatch):
    bank = ["We are an equal opportunity employer and welcome applications from individuals with disabilities."]
    monkeypatch.setattr(rc.textclean, "training_bank", lambda root: bank)
    body = ("Lab Intern\nVarrow Labs\n\nWe run assays. You must know Microsoft Word or Microsoft Outlook, and Chemistry."
            + FACTS + STUDENT)
    monkeypatch.setattr(rc, "rendered_texts", lambda kind, cards: {"rq_1": body})
    out = rc.training_postings({"rq_1": POST, "rq_2": POST})
    assert list(out) == ["rq_1"]  # a card without gated text has no posting
    assert out["rq_1"].startswith("Title: Lab Intern\nCompany: Varrow Labs\nLocations: ")
    assert out["rq_1"] == tc.training_posting(POST, body, "rq_1", bank)
    # a posting whose card fact does not survive the prompt budget is dropped
    monkeypatch.setattr(rc, "POSTING_BUDGET", 200)
    assert rc.posting_missing_facts(POST, tc.fit_posting(tc.strip_markdown(out["rq_1"]), 200))
    assert rc.training_postings({"rq_1": POST}) == {}
    monkeypatch.undo()
    assert rc.POSTING_BUDGET == 6000


def test_posting_presence_gates_name_each_missing_fact():
    head = f"Lab Intern\n{rc.company_name(POST)}\n\n"
    good = head + "You must have experience with Microsoft Word or Microsoft Outlook, plus Chemistry." + FACTS + STUDENT
    assert rc.posting_missing_facts(POST, good) == []
    # an any-of group needs every alternative named (the context lists them all)
    assert rc.posting_missing_facts(POST, good.replace(" or Microsoft Outlook", "")) == ["missing_required_skill"]
    cases = {
        "missing_preferred_skill": (POST, good.replace(" Biology is a plus.", "")),
        "missing_pay": (POST, good.replace("$41,600 to ", "")),
        "missing_location": (POST | {"locations": ["Boston, MA", "Austin, TX"]}, good),
        "missing_work_mode": (POST, good.replace("is onsite in", "is in")),
        "missing_job_type": (POST | {"employment_type": "part_time"}, good),
        "missing_industry": (POST, good.replace("healthcare ", "")),
        "missing_student_requirement": (POST, good.replace(STUDENT, "")),
        "missing_license": (POST | {"license": "Registered Nurse (RN)"}, good),
        "missing_years": (POST | {"min_years": 3}, good),
        "missing_degree_level": (POST | {"degree_level": "master"}, good),
        "missing_field_of_study": (POST | {"degree_fields": ["Chemistry", "Biochemistry"]}, good),
        "missing_sponsorship": (POST | {"sponsorship": "sponsors"}, good),
        "missing_opt": (POST | {"sponsorship": "will_not_sponsor_or_opt"}, good + " We cannot sponsor visas."),
        "missing_citizens_only": (POST | {"citizens_only": True}, good),
        "missing_e_verify": (POST | {"e_verify": True}, good),
        "sponsorship_added": (POST, good + " Visa sponsorship is available."),
    }
    for tag, (card, text) in cases.items():
        assert tag in rc.posting_missing_facts(card, text), tag
        assert tag in rc.fatal("postings", card, text), tag
    assert rc.posting_missing_facts(POST | {"min_years": 3}, good + " You need three (3) years of lab work.") == []
    assert rc.posting_missing_facts(POST | {"locations": [], "work_mode": "remote"}, good.replace("onsite in Boston, MA", "remote")) == []


def test_resume_presence_gates_name_each_missing_fact():
    card = CAND | {"years": 2, "skills": ["Microsoft Word"], "current_location": "Washington, DC",
                   "degree_level": "master", "degree_field": "Health Sciences", "graduation_year": 2026}
    good = RES.replace("June 2022 – May 2024", "June 2022 – May 2024").replace("Nurse with two years", "Nurse with 2 years")
    assert rc.resume_missing_facts(card, good) == []
    cases = {
        "missing_skill": (card | {"skills": ["Microsoft Word", "Phlebotomy"]}, good),
        "missing_license": (card | {"licenses": ["Registered Nurse (RN)"]}, good),
        "missing_degree_level": (card | {"degree_level": "doctoral"}, good),
        "missing_degree_field": (card | {"degree_field": "Public Health"}, good),
        "missing_graduation_year": (card | {"graduation_year": 2025}, good),
        "missing_expected_marker": (card | {"graduation_year": 2027}, good.replace("August 2026", "May 2027")),
        "missing_years": (card | {"years": 3}, good),
        "years_overstated": (card, good.replace("2 years", "nearly 3 years")),
    }
    for tag, (c, text) in cases.items():
        assert tag in rc.resume_missing_facts(c, text), tag
    assert "missing_expected_marker" not in rc.resume_missing_facts(card | {"graduation_year": 2027},
                                                                    good.replace("August 2026", "Expected May 2027"))
    # a '4-year degree' is not a years claim
    assert rc.stated_years("Earned a 4-year degree; 2 years in care.") == [2]
    assert rc.stated_years("over five years of work, 3+ years with Excel") == [5, 3]


def test_fake_address_left_after_redact_is_caught():
    for i in range(400):
        card = CAND | {"skills": [f"x{i}"], "current_location": "Springfield"}
        c = rc.fake_contact(card)
        if c and c.get("address"):
            break
    street = c["address"].split(",")[0]
    # a writer that moved the street below the header: redact keeps it (no city/state after it), the gate does not
    text = f"{c['name']}\nSpringfield\n\nSUMMARY\nBased near {street}.\n\nSKILLS\nx{i}"
    assert street in tc.redact(text)
    assert "redact_leaves_contact" in rc.resume_problems(card, text)


def test_full_render_waits_for_a_signed_off_check_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "OUT", tmp_path)
    monkeypatch.setattr(rc, "CHECK_IDS", {"postings": tmp_path / "check_posting_ids.txt",
                                          "resumes": tmp_path / "check_resume_ids.txt"})
    cards = {"rq_a": POST, "rq_b": POST | {"title": "Lab Assistant"}}
    assert "no check batch" in rc.require_signoff("postings", cards, ["rq_a", "rq_b"])
    rc.CHECK_IDS["postings"].write_text("rq_a\n")
    assert rc.require_signoff("postings", cards, ["rq_a"]) is None  # the check batch itself may run
    assert "no sign-off" in rc.require_signoff("postings", cards, ["rq_a", "rq_b"])
    # nothing rendered: signoff refuses
    nums, why = rc.signoff_problems("postings", cards, ["rq_a"])
    assert why and nums["tries"] == 0
    # one kept text, two quarantined tries: 33% of first tries pass, so still no sign-off
    h = rc.ctx_hash(rc.ctx_str("postings", POST))
    w = rc.writer_fingerprint("postings")
    (tmp_path / "postings.jsonl").write_text(json.dumps({"id": "rq_a", "ctx": h, "text": "x", "writer": w}) + "\n")
    with rc.quarantine_path("postings").open("w") as f:
        for _ in range(2):
            f.write(json.dumps({"id": "rq_a", "ctx": h, "tags": ["missing_pay"], "writer": w}) + "\n")
    ft = rc.first_tries("postings", cards, ["rq_a"])
    assert ft["kept"] == 1 and ft["quarantined"] == 2 and ft["stored_pass_rate"] == 0.333
    # re-gated under the current gates, the kept text 'x' fails too (it states none of the card's facts)
    assert ft["tries"] == 3 and ft["passing_tries"] == 0 and ft["pass_rate"] == 0.0 and ft["kept_passing"] == 0
    assert ft["tag_pass_rate"]["missing_pay"] == 0.0
    assert any("first tries" in w for w in rc.signoff_problems("postings", cards, ["rq_a"])[1])
    # a sign-off counts only for the blueprint and contexts it was made for
    so = {"fingerprint": rc.check_fingerprint("postings", cards, ["rq_a"])}
    rc.signoff_path("postings").write_text(json.dumps(so))
    assert rc.require_signoff("postings", cards, ["rq_a", "rq_b"]) is None
    monkeypatch.setitem(rc.KINDS, "postings", rc.KINDS["postings"] | {"blueprint": rc.POSTING_BLUEPRINT + " Be brief."})
    assert "another blueprint" in rc.require_signoff("postings", cards, ["rq_a", "rq_b"])
    # --redo for the new blueprint: the old text and the old failed tries move aside, so the next render's first tries are
    # the new blueprint's alone (the old ones would hold the pass rate at 1 / (1 + 2) whatever the new texts do)
    other = rc.ctx_hash(rc.ctx_str("postings", cards["rq_b"]))
    with rc.quarantine_path("postings").open("a") as f:
        f.write(json.dumps({"id": "rq_b", "ctx": other, "tags": ["missing_pay"]}) + "\n")
    rc.redo_contexts("postings", cards, ["rq_a"])
    ft = rc.first_tries("postings", cards, ["rq_a"])
    assert ft["kept"] == 0 and ft["quarantined"] == 0 and rc.tries("postings")[h] == 0
    assert len(rc.read_jsonl(tmp_path / "replaced_postings.jsonl")) == 1
    assert len(rc.read_jsonl(tmp_path / "replaced_quarantine_postings.jsonl")) == 2
    assert rc.tries("postings")[other] == 1  # ids outside the redo keep their tries


def test_batch_tag_changes_with_the_writer():
    ctx = rc.ctx_str("postings", POST)
    assert rc.batch_tag(["a"], [f"{ctx}#0"], rc.writer_fingerprint("postings")) != rc.batch_tag(["a"], [f"{ctx}#0"], "old")


def test_negation_and_work_auth_gates_on_an_unset_card():
    card = POST | {"enrolled_in": "none"}
    head = f"Lab Intern\n{rc.company_name(card)}\n\n"
    good = head + "Requirements\n- Microsoft Word or Microsoft Outlook\n- Chemistry\n\nYou will opt in to safety training." + FACTS
    assert rc.posting_problems(card, good) == []
    cases = {
        "neg_experience": ("Prior experience is not mandated.", "Previous experience is not expected for this role.",
                           "Experience is not a requirement.", "While prior professional experience is not mandated, care is essential.",
                           "You do not need prior experience.", "No experience required."),
        "neg_degree": ("A degree is not a requirement.", "You do not need a degree.", "Degrees are not mandatory."),
        "neg_license": ("No certification is needed.", "A license is not required.", "You don't need a certification."),
        "neg_work_auth": ("We welcome applicants of all immigration statuses.",
                          "Applicants are considered regardless of their current immigration status."),
        "work_auth_topic_unset": ("International students are welcome to apply.", "Candidates on F-1 visas are welcome.",
                                  "We hire candidates on any visas.", "Applicants must be legally permitted to work in the US.",
                                  "Employment authorization is checked on day one.", "Foreign nationals may apply."),
    }
    for tag, sents in cases.items():
        for x in sents:
            assert tag in rc.posting_problems(card, good + " " + x), (tag, x)


def test_enrolled_graduate_student_lists_the_finished_bachelors():
    ms = CAND | {"degree_level": "master", "graduation_year": 2028, "degree_field": "Biology"}
    st = rc.context_of("resumes", ms)["education_status"]
    assert "Also list a finished bachelor's degree in Biology with a past graduation date in 2026" in st
    phd = CAND | {"degree_level": "doctoral", "graduation_year": 2029, "degree_field": ""}
    assert "bachelor's degree (by level only, no subject) with a past graduation date in 2024" in \
        rc.context_of("resumes", phd)["education_status"]
    assert rc.prior_bachelor_year(CAND | {"graduation_year": 2028}) is None  # a bachelor's student
    assert rc.prior_bachelor_year(ms | {"graduation_year": 2025}) is None  # finished
    only_ms = "Education\nM.S. Biology, Expected May 2028\nBoston University"
    assert "missing_prior_bachelor" in rc.resume_missing_facts(ms, only_ms)
    both = only_ms + "\nB.S. Biology, May 2026\nUniversity of Iowa"
    assert "missing_prior_bachelor" not in rc.resume_missing_facts(ms, both)


def test_finished_degree_written_as_unfinished_is_stale():
    for line in ("B.S. Biology, Currently completing degree, May 2026", "Final-year student, graduating May 2026",
                 "B.S. Biology - Senior, class of 2026", "Degree to be conferred May 2026"):
        assert rc.stale_expected(line), line
    assert not rc.stale_expected("B.S. Biology, Class of 2025\nGraduated May 2025")
    assert not rc.stale_expected("M.S. Biology, graduating May 2028")


def test_extra_license_gate():
    vocab = ["RN license", "CNA certification", "Pharmacy technician certification"]
    card = CAND | {"licenses": ["CNA certification"]}
    ok = "Licenses\n- CNA certification (Certified Nursing Assistant)\n\nSKILLS\nChemistry"
    assert rc.extra_licenses(card, ok, vocab) == []
    assert "CPR" in rc.extra_licenses(card, ok.replace("\n\nSKILLS", "\n- CPR certified\n\nSKILLS"), vocab)
    assert rc.extra_licenses(CAND, "Certifications\n- Laboratory Safety Certification", vocab) == ["Laboratory Safety Certification"]
    assert rc.extra_licenses(CAND, "Worked beside the RN license holders on the ward.", vocab) == ["RN license"]
    assert rc.extra_licenses(CAND, "EXPERIENCE\nLab aide\n- Ran PCR assays", vocab) == []


def test_resume_signoff_counts_every_fatal_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "OUT", tmp_path)
    cards = {"cd_a": CAND}
    h = rc.ctx_hash(rc.ctx_str("resumes", CAND))
    w = rc.writer_fingerprint("resumes")
    (tmp_path / "resumes.jsonl").write_text(json.dumps({"id": "cd_a", "ctx": h, "text": "x", "writer": w}) + "\n")
    rc.quarantine_path("resumes").write_text(json.dumps({"id": "cd_a", "ctx": h, "tags": ["extra_tool"], "writer": w}) + "\n")
    assert any("first tries pass every gate" in w for w in rc.signoff_problems("resumes", cards, ["cd_a"])[1])


def test_read_by_is_a_persons_name():
    assert rc.is_reader_name("Jordan") and rc.is_reader_name("Jordan Lee")
    for bad in (None, "", "claude", "Claude Opus", "fix agent", "auto", "ci", "x" * 60, "123"):
        assert not rc.is_reader_name(bad), bad


FT = POST | {"enrolled_in": "none", "employment_type": "full_time"}
FT_GOOD = (f"Lab Intern\n{rc.company_name(FT)}\n\nRequirements\n- Microsoft Word or Microsoft Outlook\n- Chemistry\n\n"
           "You will opt in to safety training. This healthcare job is full-time and onsite in Boston, MA and pays "
           "$41,600 to $52,000 a year. Biology is a plus.")


@needs_onet
def test_posting_absence_gates_catch_facts_the_card_does_not_have():
    assert rc.posting_problems(FT, FT_GOOD) == []
    cases = {
        "years_added": ("Two years of experience required.", "Candidates should have 3+ years in the field.",
                        "Must have 5 years of managerial experience."),
        "license_added": ("CPR certification is required.", "Candidates must hold a PMP certification.",
                          "Must have a valid driver's license.", "Security clearance is required.", "Must have a TS/SCI clearance."),
        "pay_added": ("Salary range $50,000 - $60,000.", "Starting at $35 per hour."),
        "mode_conflict": ("This is a remote position.", "This is a hybrid role."),
        "type_conflict": ("This is a part-time position.", "This is an internship."),
        "extra_required_skill": ("Must know SQL.", "Experience with Salesforce is required."),
        "location_added": ("This role is based in Chicago, IL.", "The team sits in Austin, Texas."),
    }
    for tag, sents in cases.items():
        for x in sents:
            assert tag in rc.posting_problems(FT, FT_GOOD + " " + x), (tag, x)


def test_posting_absence_gates_leave_the_card_facts_and_look_alikes_alone():
    clean = ("Remote work is not available for this role.", "We do not offer remote or hybrid options.",
             "You will plan hybrid meetings and events.", "Open to full-time students in the evening program.",
             "Top performers may convert to full-time employment.", "Teach five and six-year-olds.",
             "Track metrics in Microsoft Word, OR manage schedules.", "Manage a $2 million budget.",
             "Pay is $20.00 per hour.", "Our district serves Boston, Massachusetts.", "Seniors and senior citizens visit daily.",
             "Prior internship experience is welcome.")
    for x in clean:
        assert rc.posting_problems(FT, FT_GOOD + " " + x) == [], x
    hourly = POST | {"enrolled_in": "none"}  # internship: the context gives $20.00 to $25.00 per hour
    assert rc.pay_added(hourly, "Pay is $20.00 to $25.00 per hour (about $41,600 to $52,000 a year).") == []
    assert rc.years_added(FT | {"min_years": 3}, "At least 3 years of experience.") == []
    assert rc.years_added(FT | {"min_years": 3}, "3 to 5 years of experience.") == [5]
    lic = FT | {"license": "RN license"}
    assert rc.license_added(lic, "Requirements: an active RN license and copies of your licensure.") == []
    assert rc.license_added(lic, "Requirements: an active RN license and BLS certification.") == ["BLS"]
    assert rc.mode_conflict(FT | {"work_mode": "hybrid"}, "Hybrid: remote two days, on-site three days.") == []
    assert rc.mode_conflict(FT | {"work_mode": "hybrid"}, "This role is fully remote.") == ["fully"]


def test_negation_and_work_auth_gates_cover_optional_and_residence_wordings():
    cases = {
        "neg_degree": ("Formal education is optional.", "Degree optional.", "No formal qualifications needed.",
                       "No specific college degree is required.", "We value skills over formal credentials."),
        "neg_license": ("Licensing is not necessary.", "No specific professional license is required to apply."),
        "work_auth_topic_unset": ("You do not need to be a citizen.", "Applicants must reside in the US.",
                                  "Open to non-US residents.", "Candidates from any country may apply.",
                                  "Applications from outside the United States will not be considered."),
    }
    for tag, sents in cases.items():
        for x in sents:
            assert tag in rc.posting_problems(FT, FT_GOOD + " " + x), (tag, x)
    for x in ("We serve the citizens of Dallas County.", "Over two years of experience in the lab is a plus."):
        assert not {"work_auth_topic_unset", "neg_experience"} & set(rc.posting_problems(FT, FT_GOOD + " " + x)), x


def test_extra_license_reads_credential_entries_only():
    vocab = ["RN license"]
    lic = "Certifications\n- Laboratory Safety Certification\n"
    for x in ("GPA: 3.9", "Relocated to Chicago, IL", "No licenses listed."):
        assert rc.extra_licenses(CAND, lic + x, vocab) == ["Laboratory Safety Certification"], x
    assert rc.extra_licenses(CAND, "EXPERIENCE\nCPR Visual Estimator | 2024 - Present\n- Wrote estimates", vocab) == []
    assert rc.extra_licenses(CAND, "Certifications\n- BLS (American Heart Association)", vocab) == ["BLS", "BLS (American Heart Association)"]
    assert rc.extra_licenses(CAND, "Skills\nChemistry; CPR certified", vocab) == ["CPR"]


def test_prior_bachelor_year_must_sit_with_the_bachelors():
    ms = CAND | {"degree_level": "master", "graduation_year": 2027, "degree_field": "Biology"}
    by = rc.prior_bachelor_year(ms)
    far = (f"Education\nM.S. Biology, Expected 2027\nB.S. Biology, {by - 1}\nState University\n\nExperience\n"
           + "Lab aide at a large research hospital in the city, running assays and keeping records. " * 2 + f"\nTech | {by} - Present")
    assert "missing_prior_bachelor" in rc.resume_missing_facts(ms, far)
    assert "missing_prior_bachelor" not in rc.resume_missing_facts(ms, far.replace(f"B.S. Biology, {by - 1}", f"B.S. Biology, {by}"))


def test_posting_absence_gates_leave_the_pilot_look_alikes_alone():
    # the 11 pilot postings that failed only on a new absence gate (review of 9b7211e): none is a real slip
    intern, part = POST | {"employment_type": "internship"}, POST | {"employment_type": "part_time"}
    pay_lines = (
        "The pay rate is $31.25 to $35.58 per hour, which is about $65,000 to $74,000 a year when prorated for full-time equivalent work.",
        "This translates to an approximate annual salary of $46,000 to $56,000 based on a full-time schedule.",
        "This hourly rate translates to approximately $40,000 to $47,000 a year based on full-time hours.",
        "The pay is $27.40 to $37.02 per hour, which is about $57,000 to $77,000 a year based on full-time equivalency.",
        "Based on a standard full-time equivalent schedule, this hourly range translates to about $53,000 a year.",
        "This experience is intended to prepare you for a future career as a full-time technologist.",
    )
    for card in (intern, part):
        for x in pay_lines:
            assert rc.type_conflict(card, x) == [], (card["employment_type"], x)
    assert rc.type_conflict(part, "Part-time team members are treated with the same respect as our full-time staff.") == []
    assert rc.type_conflict(FT, "The annual pay is $23,000 to $26,000, prorated based on part-time hours.") == []
    # the job's own type still counts, with a dollar figure in the sentence or not
    for x in ("This role requires full-time onsite presence to support mentorship.", "This is a full-time position.",
              "You will work full-time, 40 hours a week."):
        assert rc.type_conflict(intern, x) == ["full_time"], x
    # on an internship card 'fixed-term' is the internship's term (rq_01081); 'a full-time / part-time internship' is a job
    # type a candidate card can want, which the card does not give (rq_00433, r6 sample read)
    for x in ("As this is a fixed-term internship, it offers structure.", "It is a fixed-term summer internship."):
        assert rc.type_conflict(intern, x) == [], x
    for x, kind in (("This is a full-time internship position.", "full_time"), ("This full-time internship pays $20 an hour.",
                    "full_time"), ("We offer a part-time intern role.", "part_time")):
        assert rc.type_conflict(intern, x) == [kind], x
    # the same words on a card of another type, and a fixed-term or contract job that is not the internship, still count
    assert rc.type_conflict(FT, "This is a full-time internship position.") == ["internship"]
    assert rc.type_conflict(part, "This full-time internship pays $20 an hour.") == ["internship", "full_time"]
    for x in ("This is a 6-month contract internship.", "This is a contract role."):
        assert rc.type_conflict(intern, x) == ["contract"], x
    assert rc.mode_conflict(FT, "You will be working physically at our offices rather than remotely or in a hybrid arrangement.") == []
    assert rc.mode_conflict(FT, "This is a hybrid role.") == ["hybrid"]
    vocab = ["RN license"]
    for x in ("In addition to these educational credentials, candidates must possess at least 1 year of experience.",
              "Requirements: we provide the necessary software licenses and hardware support."):
        assert rc.license_added(FT, x, vocab) == [], x
    # a bare 'credentials' the applicant must hold is a license requirement (test_bare_credentials_name_no_credential)
    assert rc.license_added(FT, "Candidates must hold relevant credentials.", vocab) != []
    assert rc.license_added(FT, "Candidates must hold relevant CPR credentials.", vocab) != []
    tools = ["Microsoft Excel", "Excel"]
    assert rc.extra_required_skills(FT, "Requirements: the training will help you excel in creating reports.", tools) == []
    assert "Excel" in rc.extra_required_skills(FT, "Requirements: proficiency in excel and Microsoft Word.", tools)
    assert "Excel" in rc.extra_required_skills(FT, "Must be able to use Excel at an advanced level.", tools)
    x = "The officer ensures the well-being of both citizens and local wildlife."
    assert "work_auth_topic_unset" not in rc.posting_problems(FT, FT_GOOD + " " + x)
    for x in ("Join us in shaping the next generation of thinkers and global citizens.",  # rq_00475, a teaching posting
              "We prepare students to become informed citizens.", "Our courses raise engaged and responsible citizens."):
        assert "work_auth_topic_unset" not in rc.posting_problems(FT, FT_GOOD + " " + x), x
    for x in ("Applicants must be citizens.", "You do not need to be a citizen.", "This role is open to U.S. citizens.",
              "Applicants must be United States citizens."):
        assert "work_auth_topic_unset" in rc.posting_problems(FT, FT_GOOD + " " + x), x


def test_posting_gates_cover_the_remaining_low_holes():
    cases = {
        "experience_added": ("Prior experience in a similar setting is required.", "Candidates must have previous retail experience."),
        "pay_added": ("Pay: 19 an hour.", "The hiring range is 45k to 55k per year."),
        "mode_conflict": ("You will work from our office three days a week.", "This is a work-from-anywhere job."),
        "student_topic_unset": ("Applicants must be currently enrolled in a graduate program.", "Must be currently enrolled in a college program.",
                                "A PhD is required.", "A graduate degree is a plus.", "You have a high school diploma or GED."),
        "type_conflict": ("This is a seasonal position.",),
        "neg_experience": ("Prior experience is optional.",),
    }
    for tag, sents in cases.items():
        for x in sents:
            assert tag in rc.posting_problems(FT, FT_GOOD + " " + x), (tag, x)
    clean = ("We offer a 401k match.", "Join our annual 5k run.", "The lab processes 20 samples per hour.",
             "You will work over licensed practitioners.", "Experience with Microsoft Word is optional but helpful.",
             "Experience with Chemistry is required.")
    assert rc.experience_added(FT, "Prior experience is not required.") == []
    for x in clean:
        assert rc.posting_problems(FT, FT_GOOD + " " + x) == [], x
    assert rc.experience_added(FT | {"min_years": 2}, "Prior experience in a similar setting is required.") == []
    assert rc.mode_conflict(FT, "You will work in the office five days a week.") == []


def test_related_ok_stays_out_of_the_posting_context():
    """related_ok is a card-side flag (v3/extract): the posting says "or a related field" through the
    "related field" entry clean_card appends to degree_fields, never through the flag itself."""
    from matcher.schema import RequirementCard

    card = RequirementCard(source_posting_id="x", title="Analyst", degree_level="bachelor",
                           degree_fields=("Biology", "related field"), related_ok=True).to_dict()
    ctx = rc.ctx_str("postings", card)
    assert "related_ok" not in ctx and "related field" in ctx
    assert rc.ctx_str("postings", card) == rc.ctx_str("postings", {**card, "related_ok": False})


def test_retired_work_auth_wordings_still_count_as_stock_phrases():
    old = "We note that visa sponsorship is not available for this position."
    rates = rc.echo_rates("postings", [old, "Some other posting."])["stock_phrase_rates"]
    assert rates[r"visa\ sponsorship\ is\ not\ available\ for\ this\ position"] == 0.5
    assert not set(rc.RETIRED_WORK_AUTH_TEXT) & {t for ts in rc.WORK_AUTH_TEXT.values() for t in ts}


def test_degree_field_added_only_on_a_card_without_fields():
    """A card without degree_fields accepts any field (degrees.field_match), so a posting may not name a subject
    for the degree or say the field matters. Sentences from the check render (rq_00437, rq_01285, rq_01081, rq_01478)."""
    bach = FT | {"degree_level": "bachelor"}
    good = FT_GOOD + " A bachelor's degree is required."
    assert rc.posting_problems(bach, good) == []
    added = ("Your degree program must be in one of the following fields: Sociology, Philosophy, or Counseling.",
             "Your field of study must align with the technical demands of the role.",
             "Your degree must be in the field of Geography.",
             "Ensure that your resume clearly highlights your bachelor's degree in Geography.",
             "You must hold at least a bachelor's degree in one of these same fields: engineering or business.",
             "Applicants must hold a bachelor's degree in a related field.", "We welcome students majoring in Biology.")
    for x in added:
        assert "degree_field_added" in rc.fatal("postings", bach, good + " " + x), x
    for x in ("You will need a degree in order to advance.", "You will gain experience in the field of public health.",
              "Your degree may be in any field of study, but you must know Chemistry.", "We hire graduates in May 2027."):
        assert "degree_field_added" not in rc.posting_problems(bach, good + " " + x), x
    fields = bach | {"degree_fields": ["Geography"]}
    assert "degree_field_added" not in rc.posting_problems(fields, good + " Your degree must be in the field of Geography.")


def test_degree_field_added_catches_other_wordings_of_a_subject():
    """Wordings review2 of the last fix round found the gate missing, among them rq_00357~unstated's sentence, and
    sentences that name no subject and must pass."""
    bach = FT | {"degree_level": "bachelor"}
    good = FT_GOOD + " A bachelor's degree is required."
    added = ("Your course of study must be in actuarial science, mathematics, statistics, or a related quantitative field.",
             "Your course of study must be in Biology.", "Your program of study should be in Biology.",
             "Your studies must align with the technical demands of the role.", "You must be enrolled in a Biology program.",
             "You must be enrolled in a biology program.", "Students studying computer science are welcome.",
             "Open to Biology majors.", "Open to biology majors.", "We want a Biology major.",
             "A bachelor's degree in engineering is required.", "A degree in sales and marketing.",
             "A Bachelor of Science in Nursing is required.", "A Master's in Finance is required.",
             "You are currently studying accounting.", "It implies a professional doctorate in podiatry.")
    for x in added:
        assert "degree_field_added" in rc.fatal("postings", bach, good + " " + x), x
    for x in ("The field of study is open.", "You must be enrolled in a degree program.",
              "You must be enrolled in an accredited program.", "A degree in any field is welcome.",
              "A high degree of accuracy is expected in the field of banking.",
              "You will study the codes of conduct in the area of safety.", "Students majoring in any discipline are welcome.",
              "A bachelor's degree in any field of study is required.", "All majors are welcome.",
              "With major hubs in Miami, we grow fast.", "Our office is in Boston, MA in person.",
              "A bachelor's degree in hand by June is required."):
        assert "degree_field_added" not in rc.posting_problems(bach, good + " " + x), x


def test_degree_field_added_reads_a_comma_and_study_before_the_field_of():
    """The gate before 2698c25 caught a comma between the degree word and 'in the field of', and 'study in the
    field of'; narrowing that branch lost both, and the any-field strip hid 'any program of study in the field of'."""
    bach = FT | {"degree_level": "bachelor"}
    good = FT_GOOD + " A bachelor's degree is required."
    added = ("A bachelor's degree, preferably in the field of nursing.", "Your degree, ideally in the area of engineering.",
             "A degree, preferably in one of the following areas: Biology.",
             "Your studies, ideally in the discipline of finance.", "A major, preferably in the subject of chemistry.",
             "Candidates study in the field of nursing.", "Any program of study in the field of nursing.")
    for x in added:
        assert "degree_field_added" in rc.fatal("postings", bach, good + " " + x), x
    for x in ("A high degree of accuracy, and care, in the field of banking.",
              "You will study the codes of conduct in the area of safety.", "Students majoring in any discipline are welcome."):
        assert "degree_field_added" not in rc.posting_problems(bach, good + " " + x), x


def test_missing_industry_takes_and_for_ampersand_and_a_singular_last_word():
    hc = FT | {"industry": "Hospital & Health Care"}
    good = FT_GOOD.replace("This healthcare job", "This job")
    assert "missing_industry" not in rc.posting_problems(hc, good + " We are a hospital and health care network.")
    assert "missing_industry" in rc.posting_problems(hc, good + " We are a hospital network.")
    semi = FT | {"industry": "Semiconductors"}
    assert "missing_industry" not in rc.posting_problems(semi, good + " We are a semiconductor company.")


def test_internship_without_enrollment_says_who_can_apply_and_a_completed_degree():
    """An internship with no student requirement gets who_can_apply, and its degree under completed_degree (the
    writer made 6 of 12 such check internships a student role)."""
    open_deg = rc.context_of("postings", POST | {"enrolled_in": "none", "degree_level": "bachelor"})
    assert open_deg["completed_degree"] == "bachelor's degree" and "minimum_degree" not in open_deg
    assert open_deg["who_can_apply"] in rc.OPEN_TO_TEXT and "must_be_student_in" not in open_deg
    open_none = rc.context_of("postings", POST | {"enrolled_in": "none"})
    assert "who_can_apply" in open_none and "completed_degree" not in open_none and "minimum_degree" not in open_none
    student = rc.context_of("postings", POST | {"degree_level": "bachelor", "enrolled_in": "master"})
    assert student["minimum_degree"] == "bachelor's degree" and "who_can_apply" not in student
    job = rc.context_of("postings", FT | {"degree_level": "bachelor"})
    assert job["minimum_degree"] == "bachelor's degree" and "who_can_apply" not in job
    for w in rc.OPEN_TO_TEXT:  # no wording reads as a school, student, degree or work-authorization statement
        for rx in (rc.STUDENT_TOPIC, rc.GRADUATE_TOPIC, rc.NEG_DEGREE, rc.ENROLLED_MARK, rc.FIELD_SAID, rc.NEG_ANY,
                   rc.NEG_EXP, rc.NEG_LICENSE, rc.AUTH_TOPIC, rc.SCOPE_WORDS, rc.NEG_AUTH, rc.STALE_WORD):
            assert not rx.search(w), (w, rx.pattern[:40])
        sent = f"This internship is open to {w}."
        assert rc.posting_problems(POST | {"enrolled_in": "none"}, FT_GOOD.replace("full-time", "internship") + " " + sent) == []
    picks = {rc.context_of("postings", POST | {"enrolled_in": "none", "source_posting_id": f"x:{i}"})["who_can_apply"]
             for i in range(200)}
    assert picks == set(rc.OPEN_TO_TEXT) and len(rc.OPEN_TO_TEXT) >= 8
    assert not set(rc.RETIRED_OPEN_TO_TEXT) & set(rc.OPEN_TO_TEXT)
    for w in rc.OPEN_TO_TEXT:  # no wording is a requirement of its own to the requirement gates
        assert not re.search(r"\b(?:must|minimum|qualifications?|necessary|essential|students?|enrol\w*|degrees?|"
                             r"school|graduat\w*|authori[sz]\w*|sponsor\w*)\b", w, re.I), w
    # no wording opens the door to everyone (rq_00105: 'no restrictions based on current student status')
    for w in rc.OPEN_TO_TEXT:
        assert not re.search(r"\b(?:any|anyone|everyone|all|regardless|no restrictions?)\b", w, re.I), w


def test_retired_who_can_apply_wording_still_counts_as_a_stock_phrase():
    """'candidates who meet this listing's requirements' was in 15.4% of the r5 check postings; no context carries it
    now, but the writer learned it, so it stays counted."""
    rates = rc.echo_rates("postings", ["Only candidates who meet this listing’s requirements should apply.", "Apply now."])
    assert rates["stock_phrase_rates"][re.escape("candidates who meet this listing's requirements")] == 0.5


def test_completed_degree_wording_counts_as_a_stock_phrase():
    """'must have already completed a master's degree' came from the old minimum_degree line (18 of 38 check postings)."""
    rates = rc.echo_rates("postings", ["Applicants must have already completed a master's degree.",
                                       "You need a completed bachelor's degree.", "A bachelor's degree is required."])["stock_phrase_rates"]
    assert rates["already completed"] == round(1 / 3, 3)
    assert rates[r"a completed (?:associate|bachelor|master|doctoral)"] == round(1 / 3, 3)
    for cue in ("already have finished", "already completed", "completed bachelor"):
        assert cue not in rc.POSTING_BLUEPRINT, cue


def test_echo_rates_read_a_typographic_apostrophe_as_a_plain_one():
    rates = rc.echo_rates("postings", ["Open to candidates who meet this listing’s requirements.",
                                       "Open to candidates who meet this listing's requirements."])["stock_phrase_rates"]
    assert rates[re.escape("candidates who meet this listing's requirements")] == 1.0


def test_a_degree_without_fields_says_any_field():
    """A degree or a student requirement with no field on the card gets degree_field, an any-field wording picked by
    card hash (with nothing to say the writer invented a subject, 5 of 38 check postings). The rules read the card's
    empty degree_fields; degrees reads every wording as no field; no posting gate fires on it."""
    from matcher import degrees
    bach = FT | {"degree_level": "bachelor"}
    assert rc.context_of("postings", bach)["degree_field"] in rc.ANY_FIELD_TEXT
    assert "degree_field" in rc.context_of("postings", POST)  # student requirement, no field
    assert "degree_field" not in rc.context_of("postings", FT)  # no degree
    assert "degree_field" not in rc.context_of("postings", bach | {"degree_fields": ["Biology"]})
    picks = {rc.context_of("postings", bach | {"source_posting_id": f"x:{i}"})["degree_field"] for i in range(60)}
    assert picks == set(rc.ANY_FIELD_TEXT)
    good = FT_GOOD + " A bachelor's degree is required."
    student = POST | {"employment_type": "internship"}
    student_good = FT_GOOD.replace("full-time", "internship") + STUDENT
    for w in rc.ANY_FIELD_TEXT:
        assert not degrees.constrains((w,)) and degrees.field_match("Biology", (w,)) == "exact", w
        for rx in (rc.STUDENT_TOPIC, rc.GRADUATE_TOPIC, rc.NEG_DEGREE, rc.ENROLLED_MARK, rc.FIELD_SAID, rc.NEG_ANY):
            assert not rx.search(w), (w, rx.pattern[:40])
        for sent in (f"A bachelor's degree in {w} is required.", f"Your bachelor's degree may be in {w}.",
                     f"We accept a bachelor's degree in {w} of study."):
            assert rc.posting_problems(bach, good + " " + sent) == [], sent
        sent = f"Open to students currently enrolled in a bachelor's program in {w}."
        assert rc.posting_problems(student, student_good + " " + sent) == [], sent
    rates = rc.echo_rates("postings", ["A degree in any field.", "A degree in any discipline.", "Nothing."])["stock_phrase_rates"]
    assert rates["any\\ field"] == round(1 / 3, 3) and rates["any\\ discipline"] == round(1 / 3, 3)


def test_subject_like_knowledge_area_reads_as_knowledge():
    """A required O*NET knowledge area that is also a degree subject is 'knowledge of <it>' in the context (the writer
    made Sociology, Philosophy and Counseling the degree's fields, rq_00437); other skills keep their name. The card
    keeps the name, and the gates find it in the wording."""
    from matcher.visibility import has_phrase
    card = FT | {"must_skills": [["Sociology"], ["Counseling", "Python"], ["Law and regulations"]]}
    assert rc.context_of("postings", card)["required_skills"] == [
        "knowledge of sociology", {"any_one_of": ["knowledge of counseling", "Python"]}, "Law and regulations"]
    assert card["must_skills"][0] == ["Sociology"]
    text = FT_GOOD.replace("- Microsoft Word or Microsoft Outlook\n- Chemistry",
                           "- Knowledge of sociology\n- Knowledge of counseling or Python\n- Law and regulations")
    assert rc.posting_problems(card, text) == []
    assert has_phrase(text, "Sociology") and has_phrase(text, "Counseling")
    for name in rc.SUBJECT_KNOWLEDGE:
        assert has_phrase(rc.skill_name(name), name), name


def test_open_internship_has_no_term_and_degree_subjects():
    """An internship with no student requirement: no program_term, and its fields keyed as the finished degree's
    subjects (next to a term and fields_of_study the writer made it a student role, 6 of 12 such check postings)."""
    open_card = POST | {"enrolled_in": "none", "degree_level": "bachelor", "degree_fields": ["Biology"]}
    terms = [rc.context_of("postings", open_card | {"source_posting_id": f"x:{i}"}) for i in range(30)]
    assert not any("program_term" in c for c in terms)
    assert all(c["degree_subjects"] == ["Biology"] and "fields_of_study" not in c for c in terms)
    student = [rc.context_of("postings", POST | {"degree_fields": ["Biology"], "source_posting_id": f"x:{i}"})
               for i in range(30)]
    assert any("program_term" in c for c in student) and all(c["fields_of_study"] == ["Biology"] for c in student)
    job = rc.context_of("postings", FT | {"degree_level": "bachelor", "degree_fields": ["Biology"]})
    assert job["fields_of_study"] == ["Biology"]
    for key in ("degree_subjects", "degree_field"):
        assert f"- {key}" in rc.POSTING_BLUEPRINT
    text = (FT_GOOD.replace("full-time", "internship") + " A bachelor's degree in Biology is required.")
    assert rc.posting_problems(open_card, text) == []
    assert "year_unset" in rc.posting_problems(open_card, text + " The internship runs in Fall 2027.")


def test_bare_credentials_name_no_credential():
    # rq_00868, rq_01152~sponsors (r13 check render): the applicant's record, not a license or certification
    for ok in ("If you meet these requirements, we invite you to submit your credentials for review.",
               "Applicants must possess specific educational and professional credentials.",
               "We look forward to reviewing your credentials, which are required.",
               "Required: please send your credentials for verification with your application.",
               "Candidates must provide copies of their credentials.",
               # the degree itself (rq_01223~citizens, r4 replay)
               "If you possess the required doctoral credential and the necessary skills, we welcome your interest."):
        assert rc.license_added(FT, ok) == [], ok
    # a named credential, or a license or certification noun, still counts
    assert rc.license_added(FT, "A valid PMP credential is required.") == ["PMP"]
    assert rc.license_added(FT, "Candidates must hold a CompTIA credential.") == ["CompTIA"]
    for bad in ("A state license is required.", "Applicants must hold the required certification.",
                "Must hold relevant credentials and a nursing license.",
                "The applicant must hold the necessary clinical credentials to practice as a Nurse Midwife.",
                # review medium (r4): credentials tied to a profession or field are a license, on a card with none
                "Applicants must hold valid professional credentials in nursing.",
                "Required: current professional credentials for this field.",
                "Must hold the necessary clinical credentials.",
                "Candidates must hold relevant credentials."):
        assert rc.license_added(FT, bad) == [bad[:80]], bad
    assert "license_added" in rc.posting_problems(FT, FT_GOOD + " Applicants must hold the required certification.")


def test_the_employers_e_verify_enrollment_is_no_student_statement():
    # rq_01186~sponsors (r13 check render): 'enrolled in the E-Verify program' read as a student requirement
    import re as _re
    from matcher import visibility
    ev = FT | {"e_verify": True, "sponsorship": "sponsors"}
    for x in ("Grisham Systems is enrolled in the E-Verify program.", "The company is enrolled in E-Verify.",
              "We are an E-Verify enrolled employer.", "The company is currently enrolled in E-Verify."):
        assert not rc.STUDENT_TOPIC.search(x) and not _re.search(visibility.ENROLLED, x, _re.I), x
        assert "student_topic_unset" not in rc.posting_problems(ev, FT_GOOD + " " + x), x
        # on a student card, E-Verify alone does not state the student requirement
        assert "missing_student_requirement" in rc.posting_missing_facts(POST, FT_GOOD + " " + x), x
    for x in ("Applicants must be enrolled in a bachelor's degree program.", "Candidates must be currently enrolled.",
              "Students enrolled in the nursing program may apply."):
        assert rc.STUDENT_TOPIC.search(x) and _re.search(visibility.ENROLLED, x, _re.I), x
        assert "student_topic_unset" in rc.posting_problems(ev, FT_GOOD + " " + x), x


def test_enrollment_in_a_degree_program_is_a_student_topic():
    # review2 of fa9f115: the label side read 'Enrollment in a ... degree program' as a student role, the
    # student-topic gate on a card without a student requirement did not
    bach = FT | {"degree_level": "bachelor"}
    good = FT_GOOD + " A bachelor's degree is required."
    for x in ("Enrollment in a bachelor's degree program is required.", "Current enrollment in a doctoral degree program is mandatory."):
        assert "student_topic_unset" in rc.posting_problems(bach, good + " " + x), x
    for x in ("Enrollment in our benefits program starts on day one.", "Open enrollment for health benefits runs in May."):
        assert "student_topic_unset" not in rc.posting_problems(bach, good + " " + x), x


NURSE = CAND | {"field": "Nursing", "degree_field": "Nursing", "graduation_year": 2014, "years": 2, "licenses": [],
                "current_location": "Tampa, FL"}
NURSE_JOB = ("Candidate\nTampa, FL\n\nProfessional Experience\nRegistered Nurse\nBayfront Community Hospital, Tampa, FL\n"
             "September 2024 to Present\n- Provide direct patient care.\n")


def test_a_licensed_job_title_needs_its_license_on_the_card():
    # cd_v_00286 (r13 check read): 'Registered Nurse' with licenses []; cd_v_00940: 'Elementary Teacher' with no teaching license
    assert rc.title_without_license(NURSE, NURSE_JOB) == ["Registered Nurse"]
    assert "title_without_license" in rc.resume_problems(NURSE, NURSE_JOB)
    assert rc.title_without_license(NURSE | {"licenses": ["RN license"]}, NURSE_JOB) == []
    teach = "Experience\nElementary Teacher\nMaple Grove Primary School, Deerfield, IL\nMarch 2025 to Present\n"
    assert rc.title_without_license(CAND, teach) == ["Elementary Teacher"]
    assert rc.title_without_license(CAND | {"licenses": ["State teaching license"]}, teach) == []
    # cd_v_00511: an instructor with a teaching license, and titles no license stands behind
    for ok in ("Mathematics Instructor | Metro Atlanta High School | Atlanta, GA", "Teaching Assistant", "Nursing Assistant",
               "Physician Assistant", "Pharmacy Technician", "Dental Assistant", "Information Security Officer",
               "Student Nurse", "Teacher's Aide", "Substitute Teacher", "Physical Therapist Assistant",
               "- Worked alongside Registered Nurses and Physicians on the ward."):
        assert rc.title_without_license(CAND, f"Experience\n{ok}\nMay 2020 to Present\n") == [], ok
    for bad, lic in (("Staff RN", "RN license"), ("Licensed Practical Nurse", "LPN license"), ("Pharmacist", "Pharmacist license"),
                     ("Certified Public Accountant", "CPA license"), ("Associate Attorney", "Bar admission"),
                     ("Attending Physician, Community Health Partners, Reston, VA", "Medical license")):
        text = f"Experience\n{bad}\nMay 2020 to Present\n"
        assert rc.title_without_license(CAND, text) != [], bad
        assert rc.title_without_license(CAND | {"licenses": [lic]}, text) == [], bad
    # a license section is extra_licenses' to read
    assert rc.title_without_license(CAND, "Licenses\n- Registered Nurse\n") == []


def test_a_licensed_title_dated_before_its_degree_is_caught():
    # cd_v_00823 (r13 check read): a resident physician in June 2015, an MD in 2016
    md = CAND | {"degree_level": "doctoral", "degree_field": "Medicine", "graduation_year": 2016, "licenses": ["Medical license"]}
    text = ("PROFESSIONAL EXPERIENCE\n\nAttending Physician\nCommunity Health Partners, Reston, VA\nNovember 2021 to Present\n"
            "- Deliver care.\n\nResident Physician\nUniversity Medical Center, Arlington, VA\nJune 2015 to July 2018\n- Rotated.\n")
    assert rc.licensed_title_too_early(md, text) == ["Resident Physician"]
    assert "licensed_title_too_early" in rc.resume_problems(md, text)
    assert rc.licensed_title_too_early(md, text.replace("June 2015", "June 2016")) == []
    assert rc.licensed_title_too_early(md, text.replace("Resident Physician", "Medical Scribe")) == []
    # a job before a bachelor's, or a title whose license needs no graduate degree, is fine
    assert rc.licensed_title_too_early(md | {"degree_level": "bachelor"}, text) == []
    rn = NURSE | {"degree_level": "master", "graduation_year": 2024, "licenses": ["RN license"]}
    assert rc.licensed_title_too_early(rn, NURSE_JOB.replace("September 2024", "September 2020")) == []


def test_the_present_job_sits_in_the_current_metro():
    # cd_v_01354 (Portland, OR; Present job in Riyadh) and cd_rt_673458ad45 (Hampton, VA; Present job in Arlington, VA)
    pdx = CAND | {"current_location": "Portland, OR"}
    job = "Experience\nLaboratory Assistant\n{place}\nMay 2025 to Present\n- Ran assays.\n"
    assert rc.present_job_elsewhere(pdx, job.format(place="Al-Riyadh Research Institute, Riyadh, Saudi Arabia")) == [
        "Riyadh, Saudi Arabia"]
    assert "present_job_elsewhere" in rc.resume_problems(pdx, job.format(place="Lab Co, Riyadh, Saudi Arabia"))
    for ok in ("Cascade Labs, Portland, OR", "Cascade Labs, Beaverton, OR", "Cascade Labs", "Cascade Labs | Remote"):
        assert rc.present_job_elsewhere(pdx, job.format(place=ok)) == [], ok
    earlier = job.format(place="Cascade Labs, Portland, OR") + "\nLab Aide\nNorth Labs, Riyadh, Saudi Arabia\nMay 2023 to April 2025\n"
    assert rc.present_job_elsewhere(pdx, earlier) == []
    hampton = CAND | {"current_location": "Hampton, VA"}
    sensing = "Professional Experience\nRemote Sensing Analyst\nAerospace Data Solutions, {p}\nJune 2025 to Present\n"
    assert rc.present_job_elsewhere(hampton, sensing.format(p="Arlington, VA")) == ["Arlington, VA"]
    assert rc.present_job_elsewhere(hampton, sensing.format(p="Hampton, VA")) == []
    # 'Capital Insight Group, Washington, DC' reads as Washington, DC (cd_00704)
    dc = CAND | {"current_location": "Washington, DC"}
    assert rc.present_job_elsewhere(dc, "Professional Experience\nSenior Research Associate, Capital Insight Group, "
                                        "Washington, DC\nJanuary 2024 to Present\n") == []


@needs_onet
def test_a_skill_the_rule_says_the_candidate_lacks_is_not_named():
    # cd_00704, cd_v_00091 (r13 check read): 'data analysis' on cards without Data analysis
    vocab = ["Data analysis", "Relational databases", "Biology", "Design", "Statistics"]
    card = CAND | {"skills": ["Survey design", "Microsoft Excel"], "degree_field": "Anthropology"}
    text = "Summary\nCombines academic rigor with practical skills in data analysis and clinical counseling.\n"
    assert rc.lacked_skill_named(card, text, vocab) == ["Data analysis"]
    assert rc.lacked_skill_named(card | {"skills": ["Data analysis"]}, text, vocab) == []
    assert rc.lacked_skill_named(card, "- Extract large datasets from relational databases.", vocab) == ["Relational databases"]
    # the degree field shows Biology (cd_v_00678's 'Cellular Biology' coursework on a Biomedical Science degree)
    assert rc.lacked_skill_named(card | {"degree_field": "Biomedical Science"}, "Coursework: Cellular Biology", vocab) == []
    assert rc.lacked_skill_named(card, "Coursework: Cellular Biology", vocab) == ["Biology"]
    # an everyday word counts only as a list entry; a held longer skill is no mention of the shorter one
    for ok in ("- Design and distribute surveys.", "- Applied survey design to 3 studies.", "- Tracked patient history."):
        assert rc.lacked_skill_named(card, ok, vocab) == [], ok
    assert rc.lacked_skill_named(card, "Skills: Design, Microsoft Excel", vocab) == ["Design"]
    assert "lacked_skill_named" in rc.resume_problems(card, text)


def test_a_refusal_stance_takes_no_time_scope():
    # rq_01180 (r13 check read): 'does not require employer sponsorship now or in the future' on a will_not_sponsor card
    wns = FT | {"sponsorship": "will_not_sponsor"}
    good = FT_GOOD + " The company does not sponsor employment visas for this role."
    assert "sponsor_time_scope" not in rc.posting_problems(wns, good)
    for bad in ("Candidates must have existing work authorization that does not require employer sponsorship now or in the future.",
                "We will not sponsor visas at any time.", "Applicants must not need now or future visa sponsorship."):
        assert "sponsor_time_scope" in rc.posting_problems(wns, good + " " + bad), bad
    for ok in ("Candidates must already have the right to work in the United States.",
               "The company does not sponsor visas, and this will not change for this role."):
        assert "sponsor_time_scope" not in rc.posting_problems(wns, good + " " + ok), ok
    future = "We do not sponsor visas now or in the future."
    assert rc.sponsor_time_scope(FT | {"sponsorship": "sponsors"}, "We offer sponsorship to those who need it now or in the future.") == []
    assert rc.sponsor_time_scope(FT | {"sponsorship": "sponsors"}, "We welcome people who may require visa support now or in the "
                                                                    "future.") == []
    # r6 sample read: a citizens-only card is no exemption (rq_00003~citizens), and a work-status word other than
    # 'sponsor' carries the scope too (rq_00319 'visa support', rq_00608 'current or future visa sponsorship')
    cit = wns | {"citizens_only": True}
    assert rc.sponsor_time_scope(cit, future) != []
    assert rc.sponsor_time_scope(cit, "If you need visa sponsorship now or in the future, please do not submit an application.") != []
    for bad in ("We are unable to consider candidates who would need visa support now or in the future for this specific role.",
                "We cannot proceed with applications from individuals who require current or future visa sponsorship.",
                "Candidates must be legally authorized to work without any need for future or current visa support."):
        assert rc.sponsor_time_scope(wns, bad) != [], bad
    # 'at this time' on any stance: the stance may change (14 citizens-only texts)
    cit_un = FT | {"citizens_only": True}
    assert rc.sponsor_time_scope(cit_un, "We are unable to sponsor visas or consider other work authorization statuses at this "
                                         "time.") != []
    assert rc.sponsor_time_scope(wns, "The company is currently unable to sponsor work visas.") != []
    for ok in ("Applicants must be currently authorized to work in the United States.",
               "We look forward to your future with our team.", "At this time, we are reviewing applications weekly."):
        assert rc.sponsor_time_scope(wns, ok) == [], ok
    assert rc.sponsor_time_scope(FT, "We are unable to sponsor visas at this time.") == []  # no stance: work_auth_topic_unset's


def test_a_refusal_names_no_student_status_the_stance_does_not():
    # rq_00693~no_opt (r13 check read): 'student-based work authorization programs' also refuses CPT
    opt = FT | {"sponsorship": "will_not_sponsor_or_opt"}
    for bad in ("Applicants must already possess the legal right to work in the United States without the need for visa "
                "sponsorship or student-based work authorization programs.", "We cannot accept CPT.",
                "We are unable to hire international students.", "We do not hire F-1 students.",
                "Candidates must not need student work authorization support."):
        assert rc.student_auth_refused(opt, bad) != [], bad
    for ok in ("The employer cannot support F-1 OPT or STEM OPT employment, including I-983 training plans.",
               "We cannot hire F-1 students on OPT.", "The company does not sponsor employment visas.",
               "Students on F-1 visas are welcome to apply."):
        assert rc.student_auth_refused(opt, ok) == [], ok
    assert "student_auth_refused" in rc.posting_problems(opt, FT_GOOD + " We cannot accept CPT.")


def test_a_license_takes_no_place_the_card_does_not_name():
    # rq_00817 (r13 check read): 'RN license' written as authorization to practice in Florida
    rn = FT | {"license": "RN license", "locations": ["Tampa, FL"]}
    for bad in ("A current RN license is mandatory, and you must be authorized to practice in Florida.",
                "You must hold an RN license in the state of Florida.", "A Florida RN license is required.",
                "A multistate RN license under the Nurse Licensure Compact is required."):
        assert rc.license_scope_added(rn, bad) != [], bad
    assert rc.license_scope_added(rn, "A current RN license is required.") == []
    teach = FT | {"license": "State teaching license", "locations": ["Austin, TX"]}
    for ok in ("A State teaching license is required.", "A valid Texas teaching license is required."):
        assert rc.license_scope_added(teach, ok) == [], ok
    assert rc.license_scope_added(teach, "A California teaching license is required.") != []
    assert "license_scope_added" in rc.posting_problems(rn, FT_GOOD + " You must be licensed to practice in Florida.")


def test_a_required_skill_takes_no_level_the_card_does_not_give():
    # rq_01297, rq_01384 (r13 check read)
    for bad in ("Show advanced ability in using Microsoft Word.", "Advanced skills in Microsoft Word are required.",
                "You must have expert knowledge of Chemistry.", "Requirements: extensive Microsoft Outlook experience."):
        assert "skill_level_added" in rc.posting_problems(FT, FT_GOOD + " " + bad), bad
    for ok in ("Proficiency in Microsoft Word is required.", "Strong Microsoft Word skills are a must.",
               "You must have a working knowledge of Chemistry."):
        assert "skill_level_added" not in rc.posting_problems(FT, FT_GOOD + " " + ok), ok
    duty = (f"Lab Intern\n{rc.company_name(FT)}\n\nWhat you will do\n- Build advanced reports in Microsoft Word.\n")
    assert rc.skill_level_added(FT, duty) == []


def test_years_of_experience_take_no_domain_the_card_does_not_give():
    # rq_01297 (r13 check read): 'seven years of professional experience in an educational or related field'
    y7 = FT | {"min_years": 7}
    for bad in ("Possess at least seven years of professional experience in an educational or related field.",
                "At least 7 years of experience in a clinical setting is required.",
                "You need 7 years of experience as a classroom teacher."):
        assert rc.years_domain_added(y7, bad) != [], bad
    for ok in ("At least 7 years of professional experience is required.", "7 years of relevant experience.",
               "At least 7 years of experience in a related field.", "7 years of experience in a relevant capacity.",
               "7 years of experience with Microsoft Word.", "At least 7 years of experience in this field."):
        assert rc.years_domain_added(y7, ok) == [], ok
    assert rc.years_domain_added(FT, "Possess seven years of experience in an educational field.") == []  # years_added's
    assert "years_domain_added" in rc.posting_problems(y7, FT_GOOD.replace("Requirements", "Requirements\n- 7 years of "
                                                                            "experience in an educational field"))


@needs_onet
def test_a_skill_asked_of_the_applicant_is_on_the_card():
    # rq_01362 (a skill), rq_00475 (a soft skill, reported only)
    data = "Our team values curiosity and a strong foundation in data analysis."
    assert rc.extra_plain_skills(FT, data) == (["Data analysis"], [])
    assert "extra_plain_skill" in rc.fatal("postings", FT, FT_GOOD + " " + data)
    assert rc.extra_plain_skills(FT | {"nice_skills": ["Data analysis"]}, data) == ([], [])
    soft = "The ideal candidate will demonstrate strong communication skills."
    assert rc.extra_plain_skills(FT, soft) == ([], ["communication skills"])
    assert "soft_skill_added" in rc.posting_problems(FT, FT_GOOD + " " + soft)
    assert "soft_skill_added" not in rc.fatal("postings", FT, FT_GOOD + " " + soft)
    for ok in ("Proficiency in Microsoft Word for data analysis tasks is required.",
               "Our human resources department reviews every application.",
               "Join our public safety mission; candidates must know Chemistry.",
               "You will automate data analysis tasks with our team."):
        assert rc.extra_plain_skills(FT, ok) == ([], []), ok


def test_signoff_regates_every_recorded_try(tmp_path, monkeypatch):
    """Item 6 of the r13 fix pass: the pass rate re-scores every recorded try under the current gates (kept,
    quarantined, and replaced rows the current writer rendered) instead of the tags stored at render time."""
    monkeypatch.setattr(rc, "OUT", tmp_path)
    monkeypatch.setattr(rc, "STATE", tmp_path / "state")
    cards = {"rq_a": FT}
    h, w = rc.ctx_hash(rc.ctx_str("postings", FT)), rc.writer_fingerprint("postings")
    bad = FT_GOOD + " Must have a valid driver's license."
    rows = {"postings.jsonl": [{"id": "rq_a", "ctx": h, "text": bad, "writer": w}],  # kept at render time, fails a newer gate
            "quarantine_postings.jsonl": [{"id": "rq_a", "ctx": h, "tags": ["license_added"], "text": FT_GOOD,
                                           "writer": w}],  # passes now
            "replaced_postings.jsonl": [{"id": "rq_a", "ctx": h, "text": FT_GOOD, "writer": w},
                                        {"id": "rq_a", "ctx": h, "text": FT_GOOD, "writer": "old-blueprint"},
                                        {"id": "rq_a", "ctx": h, "text": FT_GOOD}],  # no stamp, no batch: unknown
            "replaced_quarantine_postings.jsonl": [{"id": "rq_a", "ctx": h, "tags": ["x"], "text": bad, "writer": w}]}
    for name, rs in rows.items():
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rs))
    ft = rc.first_tries("postings", cards, ["rq_a"])
    assert ft["tries"] == 4 and ft["passing_tries"] == 2 and ft["pass_rate"] == 0.5
    assert ft["stored_pass_rate"] == 0.5 and ft["kept_passing"] == 0
    assert ft["other_writer_rows"] == 1 and ft["unattributed_rows"] == 1 and ft["unattributed_failing"] == 0
    assert ft["passing_by_source"] == {"quarantine_postings": 1, "replaced_postings": 1}
    assert ft["tag_pass_rate"] == {"license_added": 0.5}
    # a row from a first render or a --redo batch of the current writer counts without a stamp
    (tmp_path / "state").mkdir()
    ids = ["rq_a"]
    tag = rc.batch_tag(ids, [f"{rc.ctx_str('postings', FT)}#0"], w)
    import csv
    with open(tmp_path / "b.csv", "w", newline="") as f:
        csv.writer(f).writerows([["row_id", "card"], ["rq_a", rc.ctx_str("postings", FT)]])
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    (tmp_path / "state" / f"render_postings_{tag}.json").write_text(json.dumps({"ids": ids, "csv": "b.csv"}))
    ft = rc.first_tries("postings", cards, ["rq_a"])
    assert ft["tries"] == 5 and ft["unattributed_rows"] == 0 and ft["other_writer_rows"] == 1


def test_signoff_rules(tmp_path, monkeypatch):
    """SIGNOFF_RULES: a kept text that passes for every check context within 3 tries, of this writer; every stratum has
    one; first tries (no retry feedback) pass at SIGNOFF_MIN_PASS or more; stock phrases at or under 15%; rows that
    cannot be tied to a writer are reported, and block when they fail or sit on a context with no counted try."""
    monkeypatch.setattr(rc, "OUT", tmp_path)
    monkeypatch.setattr(rc, "STATE", tmp_path / "state")
    monkeypatch.setattr(rc, "POSTING_STRATA", {"all": lambda c: True})
    w = rc.writer_fingerprint("postings")
    cards = {f"rq_{i}": FT | {"title": f"Lab Intern {i}"} for i in range(10)}
    good = {i: FT_GOOD.replace("Lab Intern", c["title"]).replace(rc.company_name(FT), rc.company_name(c))
            for i, c in cards.items()}
    assert all(rc.fatal("postings", cards[i], good[i]) == [] for i in cards)
    h = {i: rc.ctx_hash(rc.ctx_str("postings", c)) for i, c in cards.items()}
    bad = " Must have a valid driver's license."

    def write(store, quar):
        (tmp_path / "postings.jsonl").write_text("".join(json.dumps(r) + "\n" for r in store))
        rc.quarantine_path("postings").write_text("".join(json.dumps(r) + "\n" for r in quar))

    ids = sorted(cards)
    # 6 of 10 first tries pass; the 4 that failed passed on a retry with feedback: signed off (but for the read)
    store = [{"id": i, "ctx": h[i], "text": good[i], "writer": w, "feedback": int(n >= 6)} for n, i in enumerate(ids)]
    quar = [{"id": i, "ctx": h[i], "tags": ["license_added"], "text": good[i] + bad, "writer": w, "feedback": 0} for i in ids[6:]]
    write(store, quar)
    nums, why = rc.signoff_problems("postings", cards, ids)
    assert why == [] and nums["pass_rate"] == 0.6 and nums["retry_pass_rate"] == 1.0 and nums["tries_per_context"] == 1.4
    # 5 of 10: under the cost floor
    quar.append({"id": ids[5], "ctx": h[ids[5]], "tags": ["license_added"], "text": good[ids[5]] + bad, "writer": w,
                 "feedback": 0})
    store[5]["feedback"] = 1
    write(store, quar)
    assert any("first tries pass" in x and "(< 55%)" in x for x in rc.signoff_problems("postings", cards, ids)[1])
    quar.pop(); store[5]["feedback"] = 0
    # a context kept on a 4th try, a kept text that fails the current gates, a kept text of another writer
    write(store, quar + [{"id": ids[9], "ctx": h[ids[9]], "tags": ["x"], "writer": w, "feedback": 1}] * 3)
    assert any("more than 3 tries" in x for x in rc.signoff_problems("postings", cards, ids)[1])
    store2 = [dict(r) for r in store]
    store2[0]["text"] += bad
    store2[1]["writer"] = "old"
    write(store2, quar)
    why = rc.signoff_problems("postings", cards, ids)[1]
    assert any("no kept text that passes" in x and "rq_0" in x for x in why)
    assert any("kept check texts fail" in x for x in why) and any("another writer" in x and "rq_1" in x for x in why)
    # rows no record ties to a writer: counted in the numbers; a failing one blocks
    write(store, quar + [{"id": ids[0], "ctx": h[ids[0]], "text": good[ids[0]]}])
    nums, why = rc.signoff_problems("postings", cards, ids)
    assert why == [] and nums["unattributed_rows"] == 1
    write(store, quar + [{"id": ids[0], "ctx": h[ids[0]], "text": good[ids[0]] + bad}])
    nums, why = rc.signoff_problems("postings", cards, ids)
    assert nums["unattributed_failing"] == 1 and any("cannot be tied to a writer" in x for x in why)
    # a stratum with no kept text that passes
    monkeypatch.setattr(rc, "POSTING_STRATA", {"all": lambda c: True, "none": lambda c: c["title"] == "Lab Intern 0"})
    write(store2, quar)
    assert any("strata" in x and "none" in x for x in rc.signoff_problems("postings", cards, ids)[1])
    assert rc.SIGNOFF_MIN_PASS == 0.55 and "55%" in rc.SIGNOFF_RULES


def test_gate_download_stamps_the_writer():
    sent = {"rq_a": rc.ctx_str("postings", FT)}
    new, bad = rc.gate_download("postings", {"rq_a": FT}, [{"row_id": "rq_a", "enhanced_completion": FT_GOOD}], sent, {}, "d1")
    assert new and new[0]["writer"] == rc.writer_fingerprint("postings") and new[0]["dataset_id"] == "d1"


def test_a_batch_sends_each_context_once():
    # adapt drops exact-duplicate rows silently (a1ab39c6b8: 593 rows, 6 duplicate contexts, 587 back)
    cards = {"rq_a": FT, "rq_a~f1": FT, "rq_b": FT | {"title": "Lab Assistant"}}
    assert rc.unique_contexts("postings", cards, ["rq_a", "rq_a~f1", "rq_b"]) == ["rq_a", "rq_b"]
    assert sorted(rc.unique_contexts("postings", cards, ["rq_a~f1", "rq_b", "rq_a"])) == ["rq_a~f1", "rq_b"]  # the first id


def test_every_posting_context_gives_a_duty_opener():
    """'In this role, you will' was in 31% of the r13 check postings: the context names an opener (picked by the whole
    card), each opener counts as a stock phrase, and none reads as a requirement, profile or student wording."""
    assert rc.context_of("postings", FT)["duties_opener"] in rc.DUTY_OPENERS
    picks = {rc.context_of("postings", FT | {"source_posting_id": f"x:{i}"})["duties_opener"] for i in range(400)}
    assert picks == set(rc.DUTY_OPENERS) and len(rc.DUTY_OPENERS) == 20
    # the variants of one base posting do not all share one opener
    variants = {rc.context_of("postings", FT | {"sponsorship": s, "citizens_only": c})["duties_opener"]
                for s in ("unstated", "sponsors", "will_not_sponsor") for c in (False, True)}
    assert len(variants) > 1
    for o in rc.DUTY_OPENERS:
        for rx in (rc.REQ_MARK, rc.PREF_MARK, rc.PROFILE_MARK, rc.REQ_HEADING, rc.STUDENT_TOPIC, rc.GRADUATE_TOPIC,
                   rc.AUTH_TOPIC, rc.FIELD_SAID, rc.NEG_ANY, rc.STALE_WORD, rc.TRAVEL):
            assert not rx.search(o), (o, rx.pattern[:40])
        assert rc.posting_problems(FT, FT_GOOD + f"\n\n{o} keeping records in Microsoft Word.") == [], o
        rates = rc.echo_rates("postings", [f"{o} keep records.", "Nothing here."])["stock_phrase_rates"]
        assert max(r for p, r in rates.items() if p != "in this role you will") == 0.5, o
    assert rc.echo_rates("postings", ["In this role, you will keep records."])["stock_phrase_rates"]["in this role you will"] == 1.0
    assert "duties_opener" in rc.POSTING_BLUEPRINT


@needs_onet
def test_what_the_job_gives_is_no_skill_asked():
    # replay of the r13 fix pass: benefit sentences and tool kinds read as asks
    for ok in ("We are seeking a motivated intern to join our team and gain hands-on experience in the public safety sector.",
               "This internship is designed to provide hands-on experience in general human resources functions.",
               "Requirements: proficiency with data analysis tools, specifically Microsoft Word.",
               "Submit a cover letter detailing your experience with office administration."):
        assert rc.extra_plain_skills(FT, ok)[0] == [], ok
    scada = FT | {"must_skills": [["Supervisory control and data acquisition SCADA"]]}
    assert rc.extra_plain_skills(scada, "- Proficiency with Supervisory control and data acquisition (SCADA) software.")[0] == []
    assert rc.extra_plain_skills(FT, "Requirements: hands-on experience in public safety.")[0] == ["Public safety"]


def test_successful_applicants_will_have_states_a_requirement():
    # rq_00475 (r4 check render): the applicant profile of the must-have paragraph
    card = POST | {"must_skills": [["History"]], "enrolled_in": "none"}
    head = f"Lab Intern\n{rc.company_name(card)}\n\n"
    for ok in ("Successful applicants will demonstrate a solid knowledge of history.",
               "The ideal candidate will have a background in history.", "Successful candidates will possess history skills."):
        assert rc.required_as_duty_only(card, head + ok) == [], ok
    # true positives: the same sentence under a preferred heading, a 'would', a duty sentence
    assert rc.required_as_duty_only(card, head + "Nice to have\nSuccessful applicants will have history.") == [["History"]]
    assert rc.required_as_duty_only(card, head + "The ideal candidate would have history.") == [["History"]]
    assert rc.required_as_duty_only(card, head + "You will teach history to adults.") == [["History"]]


def test_a_short_heading_after_a_must_sentence_heads_its_list():
    # rq_00437 (r4 check render): 'Successful applicants will have:' right after '... must possess ...'
    card = POST | {"must_skills": [["Sociology"], ["Philosophy"]], "enrolled_in": "none"}
    head = f"Lab Intern\n{rc.company_name(card)}\n\n"
    must = "To be effective, all candidates must possess specific foundational knowledge.\n"
    for h in ("Successful applicants will have:", "In particular:", "Specifically:"):
        assert rc.required_as_duty_only(card, head + must + h + "\n- Sociology\n- Philosophy\n") == [], h
    # true positives: a duties or preferred heading after the must sentence, a heading after a plain sentence, and a
    # paragraph after the list
    for h in ("Day to day:", "Your responsibilities:", "Here is what the job involves:", "Nice to have:", "Preferred:"):
        assert rc.required_as_duty_only(card, head + must + h + "\n- Sociology\n- Philosophy\n") == [["Sociology"], ["Philosophy"]], h
    plain = "Our team is small and friendly.\nIn particular:\n- Sociology\n- Philosophy\n"
    assert rc.required_as_duty_only(card, head + plain) == [["Sociology"], ["Philosophy"]]
    after = must + "In particular:\n- Sociology\nYou will discuss Philosophy with clients.\n"
    assert rc.required_as_duty_only(card, head + after) == [["Philosophy"]]
    # the same lead-in at the end of the must paragraph itself (the rq_00437 text)
    one = must.strip() + " Successful applicants will have:\n- Sociology\n- Philosophy\n"
    assert rc.required_as_duty_only(card, head + one) == []
    one = must.strip() + " In particular:\n- Sociology\n- Philosophy\n"
    assert rc.required_as_duty_only(card, head + one) == []
    one = must.strip() + " Day to day you will:\n- Sociology\n- Philosophy\n"
    assert rc.required_as_duty_only(card, head + one) == [["Sociology"], ["Philosophy"]]


def test_fixed_term_on_an_internship_card_is_its_term():
    # rq_01362 (r4 check render): 'join our development team for a fixed term' on an internship card
    intern = POST | {"employment_type": "internship"}
    for x in ("You will join our development team for a fixed term.", "This is a fixed-term position.",
              "The fixed term runs twelve weeks."):
        assert rc.type_conflict(intern, x) == [], x
    # true positives: a full-time or part-time card saying fixed-term, and a contract wording on the internship card
    for card in (FT, POST | {"employment_type": "part_time"}):
        assert "contract" in rc.type_conflict(card, "This is a fixed-term position."), card["employment_type"]
        assert "contract" in rc.type_conflict(card, "You will join our team for a fixed term."), card["employment_type"]
    assert rc.type_conflict(intern, "This is a contract role.") == ["contract"]


def test_a_retry_carries_the_failed_sentence_and_the_card_wording():
    # r4 check render: 6 of 39 won't-sponsor postings added 'now or in the future'; a retry sent the same context again
    card = FT | {"sponsorship": "will_not_sponsor", "min_years": 2}
    ctx = rc.context_of("postings", card)
    bad = (FT_GOOD.replace("full-time", "full-time") + " We cannot sponsor visas now or in the future."
           " At least 2 years of experience in hospital administration is required.")
    tags = rc.fatal("postings", card, bad)
    assert "sponsor_time_scope" in tags and "years_domain_added" in tags
    sent = rc.retry_context("postings", card, bad)
    d = json.loads(sent)
    lines = d[rc.FEEDBACK_KEY]
    # the card facts are unchanged, the key comes last, and the text is stored under the card's own context
    assert {k: v for k, v in d.items() if k != rc.FEEDBACK_KEY} == ctx and list(d)[-1] == rc.FEEDBACK_KEY
    assert rc.base_ctx(sent) == rc.ctx_str("postings", card)
    time = next(x for x in lines if "now or in the future" in x)
    assert '"We cannot sponsor visas now or in the future' in time and ctx["work_authorization"][0] in time
    years = next(x for x in lines if "hospital administration" in x)
    assert f"minimum_years_experience: {ctx['minimum_years_experience']}" in years
    # one line per fatal tag at most, each with a quote or the missing fact
    assert len(lines) <= len(tags) and all(x.startswith("Your last version ") for x in lines)
    # a missing fact is named
    gone = rc.retry_context("postings", card, bad.replace("Chemistry", "chemicals"))
    assert any("left out Chemistry" in x for x in json.loads(gone)[rc.FEEDBACK_KEY])
    # a first try, and a try whose text is gone, still send the card's context (stored tags as the finding)
    assert rc.base_ctx(rc.ctx_str("postings", card)) == rc.ctx_str("postings", card)
    notext = json.loads(rc.retry_context("postings", card, None, ["sponsor_time_scope"]))
    assert ctx["work_authorization"][0] in notext[rc.FEEDBACK_KEY][0]


def test_a_retry_is_stored_under_the_cards_context_and_counted(tmp_path, monkeypatch):
    card = FT | {"sponsorship": "will_not_sponsor"}
    cards = {"rq_x": card}
    base = rc.ctx_str("postings", card)
    sent = {"rq_x": rc.retry_context("postings", card, FT_GOOD + " We cannot sponsor visas now or in the future.")}
    good = FT_GOOD + " " + rc.context_of("postings", card)["work_authorization"][0]
    rows = [{"row_id": "rq_x", "enhanced_completion": good}]
    new, bad = rc.gate_download("postings", cards, rows, sent, {}, "ds")
    assert bad == [] and new[0]["ctx"] == rc.ctx_hash(base) and new[0]["feedback"] >= 1
    rows = [{"row_id": "rq_x", "enhanced_completion": FT_GOOD + " We cannot sponsor visas now or in the future."}]
    new, bad = rc.gate_download("postings", cards, rows, {"rq_x": base}, {}, "ds")
    assert new == [] and bad[0]["ctx"] == rc.ctx_hash(base) and bad[0]["feedback"] == 0
    # the retry's quoted sentence is a stock phrase of its own when copied
    assert any(re.search(p, "the card says") for p in rc.ECHO["postings"])
    assert "- fix_from_last_try" in rc.POSTING_BLUEPRINT and "- fix_from_last_try" in rc.RESUME_BLUEPRINT


def test_every_fatal_tag_has_a_finding():
    card = FT | {"sponsorship": "will_not_sponsor"}
    text = FT_GOOD + " We cannot sponsor visas now or in the future. Travel to Denver, CO is required. Pay is $99,000."
    tags = rc.fatal("postings", card, text)
    fs = rc.findings("postings", card, text)
    assert [f["tag"] for f in fs] == list(dict.fromkeys(tags))
    assert all(f.get("quote") or f.get("missing") or f.get("named") for f in fs), fs
    assert all(not f["card_says"].startswith("the check") for f in fs), fs


def test_minimum_years_say_any_kind_of_work():
    # r4: 'one year of relevant experience in a clinical or support setting' (rq_01180) narrowed the card's any-work years
    card = FT | {"min_years": 3}
    ctx = rc.context_of("postings", card)
    assert ctx["minimum_years_experience"] in [t.format(n="3 years") for t in rc.YEARS_TEXT]
    assert rc.context_of("postings", FT | {"min_years": 1})["minimum_years_experience"].startswith("at least 1 year of")
    shares = Counter(rc.context_of("postings", FT | {"min_years": 2, "title": f"T{i}"})["minimum_years_experience"]
                     for i in range(600))
    assert len(shares) == len(rc.YEARS_TEXT) and max(shares.values()) / 600 < 0.25
    for t in rc.YEARS_TEXT:
        x = "Requirements: " + t.format(n="3 years") + "."
        assert rc.has_years(x, 3) and rc.years_domain_added(card, x) == [] and rc.years_added(card, x) == [], x
        low = " ".join(x.lower().replace(",", "").split())
        assert any(re.search(p, low) for p in rc.ECHO["postings"]), x
        assert not any(re.search(re.escape(f), low) for f in rc.ANY_FIELD_TEXT), x
    # the writer's own any-work words pass; a named domain still fails
    for ok in ("At least 3 years of experience in any industry.", "3 years of work experience in any role is required.",
               "You need at least 3 years of experience in any kind of job."):
        assert rc.years_domain_added(card, ok) == [], ok
    for bad in ("At least 3 years of experience in any clinical setting.", "3 years of experience in hospital administration.",
                "At least 3 years of experience in a clinical or support setting."):
        assert rc.years_domain_added(card, bad) != [], bad


def test_a_resume_retry_quotes_the_job_entry_and_the_card():
    # cd_v_01354 (r4 reader): the Present job in Riyadh for a Portland card; the school is in Riyadh too
    card = CAND | {"years": 1, "current_location": "Portland, OR", "skills": ["Chemistry"]}
    text = ("Candidate\nPortland, OR\n\nSummary\nChemistry graduate with 1 year of lab work.\n\nEducation\nBachelor of Science "
            "in Biology\nKing Saud University, Riyadh, Saudi Arabia\nMay 2026\n\nExperience\nLaboratory Assistant\n"
            "Al-Riyadh Research Institute, Riyadh, Saudi Arabia\nMay 2025 to Present\n- Ran Chemistry assays.\n\nSkills\nChemistry\n")
    assert "present_job_elsewhere" in rc.fatal("resumes", card, text)
    lines = json.loads(rc.retry_context("resumes", card, text))[rc.FEEDBACK_KEY]
    job = next(x for x in lines if "Present" in x and "Portland, OR" in x)
    assert "Al-Riyadh Research Institute" in job and "King Saud" not in job
    assert rc.base_ctx(rc.retry_context("resumes", card, text)) == rc.ctx_str("resumes", card)


def test_run_sends_a_retry_with_feedback_and_a_first_try_without(tmp_path, monkeypatch):
    """main(): the batch csv holds the card's context for a first try and the context plus fix_from_last_try for a
    context with a failed try; nothing reaches the API here (the fake client stops at the upload)."""
    import csv

    class Stop(Exception):
        pass

    class Fake:
        class datasets:
            @staticmethod
            def upload_file(*a, **k):
                raise Stop

    cards = {"rq_a": FT | {"sponsorship": "will_not_sponsor"}, "rq_b": FT | {"title": "Lab Assistant"}}
    monkeypatch.setattr(rc, "OUT", tmp_path)
    monkeypatch.setattr(rc, "STATE", tmp_path / "state")
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    monkeypatch.setattr(rc, "CHECK_IDS", {"postings": tmp_path / "ids.txt", "resumes": tmp_path / "r.txt"})
    monkeypatch.setattr(rc, "load_cards", lambda kind: cards)
    monkeypatch.setattr(rc, "client", lambda: Fake())
    (tmp_path / "state").mkdir()
    (tmp_path / "ids.txt").write_text("rq_a\nrq_b\n")
    ha = rc.ctx_hash(rc.ctx_str("postings", cards["rq_a"]))
    bad = FT_GOOD + " We cannot sponsor visas now or in the future."
    rc.quarantine_path("postings").write_text(json.dumps({"id": "rq_a", "ctx": ha, "tags": ["sponsor_time_scope"],
                                                          "text": bad, "writer": rc.writer_fingerprint("postings")}) + "\n")
    monkeypatch.setattr(sys, "argv", ["render_cards.py", "postings", "run", str(tmp_path / "ids.txt")])
    with pytest.raises(Stop):
        rc.main()
    rows = {r["row_id"]: r["card"] for p in (tmp_path / "batches").glob("*.csv") for r in csv.DictReader(open(p))}
    assert set(rows) == {"rq_a", "rq_b"}
    assert rows["rq_b"] == rc.ctx_str("postings", cards["rq_b"])
    a = json.loads(rows["rq_a"])
    assert rc.base_ctx(rows["rq_a"]) == rc.ctx_str("postings", cards["rq_a"]) and "now or in the future" in a[rc.FEEDBACK_KEY][0]


def test_enrolled_program_names_the_degree_field():
    """The rule reads an enrolled student's degree_field as the program being studied (rules._check_degree: 'studying
    Criminal Justice'): cd_v_01236 wrote an MBA for a Criminal Justice master's and put the subject on the bachelor's only."""
    card = CAND | {"degree_level": "master", "degree_field": "Criminal Justice", "graduation_year": 2027, "years": 0}
    bach = "\n\nBachelor of Science in Criminal Justice, May 2025\nTemple University, Philadelphia, PA\n"
    mba = "Candidate\nBoston, MA\n\nEDUCATION\nMaster of Business Administration, Expected May 2027\nDrexel University, Philadelphia, PA" + bach
    assert rc.program_field_missing(card, mba) and "program_field_missing" in rc.resume_problems(card, mba)
    split = "Candidate\nBoston, MA\n\nEducation\nMaster of Arts in Education\nExpected May 2027\nDrexel University" + bach
    assert rc.program_field_missing(card, split)
    lines = json.loads(rc.retry_context("resumes", card, mba))[rc.FEEDBACK_KEY]
    assert any("Master of Business Administration" in x and "Criminal Justice" in x for x in lines)
    # near misses: the subject on the in-progress degree, on its own line, or as the degree's short name
    ok = "Candidate\nBoston, MA\n\nEDUCATION\nMaster of Science in Criminal Justice, Expected May 2027\nDrexel University" + bach
    assert rc.program_field_missing(card, ok) == []
    ok2 = "Candidate\nBoston, MA\n\nEducation\nMaster of Science\nCriminal Justice\nExpected May 2027\nDrexel University" + bach
    assert rc.program_field_missing(card, ok2) == []
    nurse = card | {"degree_field": "Nursing"}
    assert rc.program_field_missing(nurse, "Education\nMSN, Expected May 2027\nDrexel University\n") == []
    # the context names the subject for the degree in progress too, not only for the finished bachelor's
    st = rc.context_of("resumes", card)["education_status"]
    assert st.startswith("Still enrolled in the master's degree in Criminal Justice;") and "bachelor's degree in Criminal Justice" in st
    assert "master's degree;" in rc.context_of("resumes", card | {"degree_field": ""})["education_status"]
    assert "Still enrolled in the bachelor's degree;" in rc.context_of("resumes", card | {"degree_level": "bachelor"})["education_status"]
    # a finished degree, or a card with no degree_field, is not this check's
    assert rc.program_field_missing(card | {"graduation_year": 2025}, mba) == []
    assert rc.program_field_missing(card | {"degree_field": ""}, mba) == []
    # a summary sentence is no education entry
    summ = "Candidate\nBoston, MA\n\nSummary\nPursuing a master's degree, expected 2027, while working in operations.\n\nEducation\n" \
           "Master of Science in Criminal Justice, Expected May 2027\nDrexel University" + bach
    assert rc.program_field_missing(card, summ) == []


def test_certification_the_card_lacks_is_caught_anywhere():
    """cd_v_00840 (licenses empty) turned its software skill 'CPR Visual Estimator' into 'Certified in CPR Visual
    Estimator techniques'; cd_00966 wrote 'Secondary Education Certification'. Someone else's license, a record kept for
    others and the card's own license are no claim."""
    card = CAND | {"skills": ["CPR Visual Estimator", "Java"], "licenses": []}
    for bad in ("Certified in CPR Visual Estimator techniques for emergency response scenarios.",
                "Secondary Education Certification", "Holds a Google Data Analytics certificate.",
                "Certifications: CPR Visual Estimator"):
        assert rc.credential_claimed(card, "Candidate\n" + bad + "\n"), bad
        assert "certification_added" in rc.resume_problems(card, "Candidate\nBoston, MA\n\n" + bad + "\n"), bad
    for ok in ("Skills: CPR Visual Estimator, Java", "- Assisted licensed pharmacists in dispensing medications.",
               "- Maintained detailed training records and certification logs in Java.",
               "- Verified visitor credentials at the front desk.", "No licenses or certifications.",
               "- Completed certification courses in new design tools."):
        assert rc.credential_claimed(card, "Candidate\n" + ok + "\n") == [], ok
    cna = CAND | {"licenses": ["CNA certification"]}
    for ok in ("Certified Nursing Assistant | Mercy Health | Boston, MA", "Licenses: CNA certification",
               "Holds an active CNA certification."):
        assert rc.credential_claimed(cna, ok) == [], ok
    assert rc.credential_claimed(cna, "Certified in BLS and CNA certification") == []  # names the card's own
    assert rc.credential_claimed(cna, "Certified in Phlebotomy") == ["Certified in Phlebotomy"]


def test_skill_under_a_credentials_heading_is_a_credential():
    card = CAND | {"skills": ["CPR Visual Estimator"], "licenses": []}
    assert rc.extra_licenses(card, "Candidate\n\nCertifications\nCPR Visual Estimator\n", vocab=[])
    assert rc.extra_licenses(card, "Candidate\n\nSkills & Certifications\nCPR Visual Estimator\n", vocab=[]) == []


def test_wont_sponsor_scope_widening_is_caught():
    """will_not_sponsor stops needs_h1b_now only; OPT and STEM OPT get a soft check, and STEM OPT needs the employer's
    E-Verify and I-983 training plan (rules._check_work_auth). r5 check postings widened it: rq_00868, rq_01180,
    rq_00957, rq_00853."""
    wns = FT | {"sponsorship": "will_not_sponsor"}
    stance = " The employer does not sponsor employment visas such as H-1B, so applicants must be able to take the job without visa sponsorship."
    base = FT_GOOD + stance
    assert rc.posting_problems(wns, base) == []
    for bad in ("This policy applies to all candidates regardless of their current location or status.",
                "We are unable to assist with immigration processes or provide sponsorship for work authorization.",
                "We are unable to assist with work authorization processes, so candidates should ensure they have the "
                "necessary rights to work independently before applying.",
                "If you meet the qualifications and are authorized to work immediately, we would welcome your application.",
                "We cannot support I-983 training plans.", "All hires must be eligible to work independently."):
        assert "auth_scope_added" in rc.posting_problems(wns, base + " " + bad), bad
    for ok in ("Applicants must already have the legal right to work in the United States.",
               "We consider all applicants without regard to race, color, religion, sex, national origin, disability or "
               "veteran status.", "Qualified applicants are considered regardless of race, age or veteran status.",
               "We cannot assist with visa procurement.", "Our HR team will help you with onboarding paperwork."):
        assert "auth_scope_added" not in rc.posting_problems(wns, base + " " + ok), ok
    # the or_opt stance refuses STEM OPT itself, I-983 included; a citizens-only card refuses every non-citizen
    opt = FT | {"sponsorship": "will_not_sponsor_or_opt"}
    assert rc.auth_scope_added(opt, "We do not hire on OPT or STEM OPT, including I-983 training plans.") == []
    assert rc.auth_scope_added(opt, "We are unable to assist with immigration processes.")
    assert rc.auth_scope_added(wns | {"citizens_only": True}, "We are unable to assist with immigration processes.") == []
    lines = json.loads(rc.retry_context("postings", wns, base + " This policy applies to all candidates regardless of their "
                                                               "current location or status."))[rc.FEEDBACK_KEY]
    assert any("regardless of their current location or status" in x and "anyone's status" in x for x in lines)


MULTI = FT | {"locations": ["Dublin, OH", "Kissimmee, FL"], "work_mode": "hybrid"}
MULTI_GOOD = FT_GOOD.replace("onsite in Boston, MA", "hybrid, based in Dublin, OH or Kissimmee, FL")


def test_multi_city_posting_does_not_split_the_person_between_cities():
    """The rule matches a candidate on any one location (rules.matched_location): rq_00646, rq_00099~sponsors, rq_00853
    put the person in both cities."""
    assert rc.posting_problems(MULTI, MULTI_GOOD) == []
    for bad in ("This position is based in a hybrid work model, splitting time between our offices in Dublin, OH and Kissimmee, FL.",
                "You will split your time between our locations in Dublin and Kissimmee.",
                "You will rotate between our offices as needed.", "You will support teams at both locations.",
                "This role operates on a hybrid basis between our Dublin, OH, and Kissimmee, FL studios."):
        assert "locations_split" in rc.posting_problems(MULTI, MULTI_GOOD + " " + bad), bad
    for ok in ("You will split your time between working onsite at one of these offices and working remotely.",
               "You will split your time between our office and home.",
               "You must live near or be willing to commute to one of our offices: Dublin or Kissimmee.",
               "Our sites in Dublin, OH, and Kissimmee, FL bridge the gap between research and practice.",
               "We offer a hybrid arrangement, splitting time between our on-site locations and remote duties."):
        assert "locations_split" not in rc.posting_problems(MULTI, MULTI_GOOD + " " + ok), ok
    one = MULTI | {"locations": ["Dublin, OH"]}  # one city: a split is office and home, not this gate's
    assert rc.locations_split(one, "You will split your time between our offices in Dublin, OH and remote work.") == []


def test_recent_graduate_rule_is_never_the_cards():
    """No card limits how recently a degree was finished (rq_01399, a finished bachelor's)."""
    bach = FT | {"degree_level": "bachelor"}
    good = FT_GOOD + " A bachelor's degree is required."
    for bad in ("The position is designed for individuals who have recently completed their undergraduate education.",
                "This is an excellent opportunity for a recent graduate.", "We welcome new graduates.",
                "Ideal for someone who just graduated."):
        assert "recent_grad_added" in rc.posting_problems(bach, good + " " + bad), bad
    for ok in ("You will help with recently completed infrastructure projects.", "We recently earned a national award.",
               "Our team recently finished a major software rollout."):
        assert "recent_grad_added" not in rc.posting_problems(bach, good + " " + ok), ok
    # a student card's posting gets no second door for recent graduates either
    assert rc.recent_grad_added(POST, "Open to students or recent graduates.")


def test_non_latin_script_is_fatal_in_resumes_and_postings():
    """'visitor接待' in 8 kept resumes (cd_00370): a word of another script in an English text. Accented Latin letters,
    dashes, quotes, bullets and letterlike symbols pass; a character the card itself holds is the card's."""
    for kind, card, text in (("resumes", CAND, "Candidate\nBoston, MA\n\n- Handled scheduling and visitor接待."),
                             ("resumes", CAND, "Candidate\n- Kept classroom秩序 daily."),
                             ("postings", FT, FT_GOOD + " Работа в офисе."),
                             ("postings", FT, FT_GOOD + " Apply today，we reply fast.")):
        assert rc.non_latin_script(card, text), text
        assert "non_latin_script" in rc.fatal(kind, card, text), text
    assert rc.non_latin_script(CAND, "visitor接待 and by接待 visitors") == ["visitor接待", "by接待"]
    for ok in ("Studied in Bogotá and São Paulo; wrote a résumé – fast — “well” • done ™ ℹ 5 µg.", FT_GOOD):
        assert rc.non_latin_script(CAND, ok) == [], ok
    assert rc.non_latin_script(CAND | {"skills": ["TNF-α assays"]}, "Ran TNF-α assays.") == []
    assert "non_latin_script" not in rc.posting_problems(FT, FT_GOOD)


def test_finished_education_is_never_the_cards():
    """'for individuals who have completed their formal education' (rq_00980, 12 kept postings) beside a CPT student the
    rule calls Keep: no card says the applicant is done with school, so the wording is fatal on every card. A
    completion that names its level is the card's degree requirement and passes."""
    bach = FT | {"degree_level": "bachelor"}
    good = FT_GOOD + " A bachelor's degree is required."
    for bad in ("This role is designed for individuals who have completed their formal education.",
                "It is aimed at individuals who have finished their academic training.",
                "We are looking for candidates who have finished their studies.",
                "This internship is for individuals who have already completed their education.",
                "It suits people who have already completed their academic groundwork.",
                "This role is open to candidates who have already graduated.",
                "Applicants must be no longer enrolled in school."):
        assert "education_done_added" in rc.posting_problems(bach, good + " " + bad), bad
        assert "education_done_added" in rc.fatal("postings", bach, good + " " + bad), bad
    for ok in ("To qualify, you must have already completed a bachelor's degree.",
               "This internship is designed for those who have already completed their undergraduate education.",
               "We welcome graduates from any discipline.", "We are looking for individuals who have graduated with a bachelor's.",
               "You will help clients who have completed their training plans.",
               "While this role is designed for those still completing their academic journey, the work matters."):
        assert "education_done_added" not in rc.posting_problems(bach, good + " " + ok), ok
    # a student card says the opposite: fatal there too
    assert rc.education_done_added(POST, "Open to people who have completed their formal education.")


def test_enrollment_said_on_a_card_without_a_student_rule():
    """rq_00633: 'We do not require current enrollment in a school program' on an open internship; rq_00105 'no
    restrictions based on student status'. The negation is a student statement too."""
    intern = FT | {"employment_type": "internship", "degree_level": "bachelor"}
    good = FT_GOOD.replace("full-time", "internship") + " A bachelor's degree is required."
    assert rc.posting_problems(intern, good) == []
    for bad in ("We do not require current enrollment in a school program.", "There are no restrictions based on student status.",
                "You do not need to be a student to apply.", "This role is open to candidates regardless of their current "
                "educational status.", "This role is not limited to students."):
        assert "student_topic_unset" in rc.posting_problems(intern, good + " " + bad), bad
    for ok in ("You will manage student enrollment records for our school.", "Enrollment in our benefits program starts on day one.",
               "Open enrollment for health benefits runs in May.", "The company is enrolled in E-Verify.",
               "You will track enrollment data for the district."):
        assert "student_topic_unset" not in rc.posting_problems(intern, good + " " + ok), ok
    lines = json.loads(rc.retry_context("postings", intern, good + " We do not require current enrollment in a school program."))
    assert any("not even that they are not required" in x for x in lines[rc.FEEDBACK_KEY])
    # a student card says it on purpose
    assert rc.student_said(POST, "Applicants must be students currently enrolled in a bachelor's program.") is None


def test_fully_remote_card_without_places_names_no_region():
    """rq_00566 'from any location within the United States', rq_01384 'within the country': location_ok is true for any
    remote card."""
    remote = FT | {"work_mode": "remote", "locations": []}
    good = FT_GOOD.replace("onsite in Boston, MA", "fully remote")
    assert rc.posting_problems(remote, good) == []
    for bad in ("You may work from any location within the United States.", "This is a fully remote role, allowing you to "
                "work from any location within the country.", "Candidates may reside anywhere within the United States.",
                "Location: Remote (Work from anywhere in the U.S.)", "You must live in the Eastern time zone to work remotely."):
        assert "location_added" in rc.posting_problems(remote, good + " " + bad), bad
    for ok in ("You can work from anywhere with a reliable internet connection.",
               "We serve clients across the country.", "You will coordinate across different time zones while working remotely.",
               "Applicants must be authorized to work within the United States."):
        assert rc.remote_region_added(remote, ok) == [], ok
    # a remote card that names a place, and a hybrid card, are not this gate's
    assert rc.remote_region_added(remote | {"locations": ["Boston, MA"]}, "You may work from anywhere within the United States.") == []
    assert rc.remote_region_added(FT, "You may work from anywhere within the United States.") == []


def test_expertise_is_a_skill_level():
    """rq_00897: 'Deep learning expertise is a strict requirement'. Proficiency and plain knowledge stay allowed."""
    card = FT | {"must_skills": [["Microsoft Word", "Microsoft Outlook"], ["Chemistry"], ["Deep learning"]]}
    base = FT_GOOD.replace("- Chemistry\n", "- Chemistry\n- Deep learning\n")
    assert "skill_level_added" not in rc.posting_problems(card, base)
    for bad in ("Deep learning expertise is a strict requirement.", "Applicants must have expertise in Deep learning.",
                "Mastery of Deep learning is required."):
        assert "skill_level_added" in rc.posting_problems(card, base + " " + bad), bad
    for ok in ("Proficiency in Deep learning is required.", "Applicants must have experience with Deep learning.",
               "Our team's expertise spans many fields."):
        assert "skill_level_added" not in rc.posting_problems(card, base + " " + ok), ok


def _fake_run(tmp_path, monkeypatch, cards, mode="run"):
    """main() on `cards` with every path in tmp_path; the fake client stops at the upload (nothing reaches the API)."""
    class Stop(Exception):
        pass

    class Fake:
        class datasets:
            @staticmethod
            def upload_file(*a, **k):
                raise Stop

    monkeypatch.setattr(rc, "OUT", tmp_path)
    monkeypatch.setattr(rc, "STATE", tmp_path / "state")
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    monkeypatch.setattr(rc, "CHECK_IDS", {"postings": tmp_path / "ids.txt", "resumes": tmp_path / "r.txt"})
    monkeypatch.setattr(rc, "load_cards", lambda kind: cards)
    monkeypatch.setattr(rc, "client", lambda: Fake())
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "ids.txt").write_text("\n".join(cards) + "\n")
    monkeypatch.setattr(sys, "argv", ["render_cards.py", "postings", mode, str(tmp_path / "ids.txt")])
    return Stop


def test_kept_text_failing_a_newer_gate_is_a_failed_try_and_is_rendered_again(tmp_path, monkeypatch, capsys):
    """A kept text rendered before a gate existed ('We welcome new graduates.' before recent_grad_added) is moved to the
    quarantine as a failed try with its re-gated tags, counted, and its context is sent again with the findings; a kept
    text that passes stays. Offline: the fake client stops at the upload."""
    import csv

    cards = {"rq_a": FT, "rq_b": FT | {"title": "Lab Assistant"}}
    Stop = _fake_run(tmp_path, monkeypatch, cards)
    wf = rc.writer_fingerprint("postings")
    ha, hb = (rc.ctx_hash(rc.ctx_str("postings", cards[x])) for x in ("rq_a", "rq_b"))
    stale = FT_GOOD + " We welcome new graduates."
    good_b = FT_GOOD.replace(f"Lab Intern\n{rc.company_name(FT)}", f"Lab Assistant\n{rc.company_name(cards['rq_b'])}", 1)
    assert rc.fatal("postings", FT, stale) == ["recent_grad_added"] and rc.fatal("postings", cards["rq_b"], good_b) == []
    rows = [{"id": "rq_a", "ctx": ha, "text": stale, "dataset_id": "d1", "writer": wf, "feedback": 0},
            {"id": "rq_b", "ctx": hb, "text": good_b, "dataset_id": "d1", "writer": wf, "feedback": 0}]
    (tmp_path / "postings.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(Stop):
        rc.main()
    assert [r["id"] for r in rc.read_jsonl(tmp_path / "postings.jsonl")] == ["rq_b"]
    q = rc.read_jsonl(rc.quarantine_path("postings"))
    assert len(q) == 1 and q[0]["id"] == "rq_a" and q[0]["tags"] == ["recent_grad_added"] and q[0]["stale_gate"]
    assert q[0]["writer"] == wf and q[0]["dataset_id"] == "d1" and q[0]["feedback"] == 0 and q[0]["text"] == stale
    assert rc.tries("postings")[ha] == 1
    sent = {r["row_id"]: r["card"] for p in (tmp_path / "batches").glob("*.csv") for r in csv.DictReader(open(p))}
    assert set(sent) == {"rq_a"}
    lines = json.loads(sent["rq_a"])[rc.FEEDBACK_KEY]
    assert any("We welcome new graduates" in x for x in lines) and rc.base_ctx(sent["rq_a"]) == rc.ctx_str("postings", FT)
    assert "1 kept texts fail a gate added after they were rendered" in capsys.readouterr().out
    # recorded as the failing first try it was: same writer, no feedback
    tr, other = rc.recorded_tries("postings", cards, ["rq_a"])
    assert [(r["source"], r["tags"], r["feedback"]) for r in tr] == [("quarantine_postings", ["recent_grad_added"], 0)]


def test_kept_text_failing_a_newer_gate_with_no_try_left_stays_without_text(tmp_path, monkeypatch, capsys):
    """Two failed tries and a kept third that a newer gate fails: the third becomes the third failed try, the context
    has no try left, nothing is sent and the id is reported."""
    cards = {"rq_a": FT}
    _fake_run(tmp_path, monkeypatch, cards)
    wf = rc.writer_fingerprint("postings")
    ha = rc.ctx_hash(rc.ctx_str("postings", FT))
    bad = FT_GOOD + " We cannot sponsor visas now or in the future."
    q = [{"id": "rq_a", "ctx": ha, "tags": ["sponsorship_added"], "text": bad, "writer": wf, "feedback": n} for n in (0, 1)]
    rc.quarantine_path("postings").write_text("".join(json.dumps(r) + "\n" for r in q))
    (tmp_path / "postings.jsonl").write_text(json.dumps({"id": "rq_a", "ctx": ha, "text": FT_GOOD + " We welcome new graduates.",
                                                         "writer": wf, "feedback": 1}) + "\n")
    rc.main()  # no upload: the fake client would raise
    out = capsys.readouterr().out
    assert rc.read_jsonl(tmp_path / "postings.jsonl") == [] and rc.tries("postings")[ha] == 3
    assert "failed the gates 3 times" in out and "had a kept text that a newer gate fails and no try left: ['rq_a']" in out
    assert not list((tmp_path / "batches").glob("*.csv")) if (tmp_path / "batches").exists() else True
    assert "rq_a" not in rc.rendered_texts("postings", cards, clean=False)


def test_watched_openers_are_reported_not_gated():
    """The r5 reader's untracked openers and closers are counted for the report; they do not block the stock-phrase gate
    this round."""
    texts = ["We are looking for a technician. We look forward to your application."] * 3 + ["Apply today."]
    e = rc.echo_rates("postings", texts)
    assert e["watched_phrase_rates"]["we are looking for"] == 0.75 and e["watched_phrase_rates"]["we look forward to"] == 0.75
    assert e["watched_over_cap"] == ["we are looking for", "we look forward to"]
    assert e["stock_phrase_gate"] and "we are looking for" not in e["stock_phrase_rates"]
    assert set(rc.ECHO_WATCH["postings"]) >= {"we are looking for", "while not mandatory", "to qualify for this",
                                              "we look forward to", "candidates who bring additional", "to succeed in this"}


# ---------------------------------------------------------------- r6 sample reads (postings 120, resumes 150)

def test_a_trait_is_never_a_requirement():
    """rq_00526, rq_01009, rq_00065~citizens: a soft trait made essential, required or what the ideal candidate is."""
    for bad in ("Safety matters on site, so a proactive approach to safety is essential.",
                "The fully remote nature of the role requires self-discipline and strong communication skills.",
                "Our ideal candidate is detail-oriented and eager to grow.", "Attention to detail is a must."):
        assert rc.trait_required(FT, bad) != [], bad
        assert "trait_required" in rc.posting_problems(FT, FT_GOOD + " " + bad), bad
        assert "trait_required" not in rc.fatal("postings", FT, FT_GOOD + " " + bad), bad  # reported, not fatal
    for ok in ("Maintaining a clean and organized workspace is essential for compliance with state regulations.",
               "You will keep an organized inventory of lab supplies.", "Attention to detail helps in this lab.",
               # read-20 false alarms: a thing's reliability, an invitation (rq_00094, rq_00579, rq_00504, rq_00400)
               "The successful candidate will work onsite to enhance system reliability.",
               "The successful candidate will support projects that drive efficiency and reliability within our network.",
               "Redfern Electric delivers essential power solutions and resilient infrastructure.",
               "If you have the required background in fine arts and are eager to learn, we would love to hear from you."):
        assert rc.trait_required(FT, ok) == [], ok
    assert rc.trait_required(FT, "The position demands reliability and the ability to work alone.") != []
    assert rc.trait_required(FT, "You must be reliable and punctual.") != []
    # the card's own soft skill: named as a preference is fine, made a must only when the card has it as one
    nice = FT | {"nice_skills": ["Attention to detail"]}
    assert rc.trait_required(nice, "The ideal candidate brings attention to detail.") == []
    assert rc.trait_required(nice, "Attention to detail is essential.") != []
    must = FT | {"must_skills": [["Attention to detail"]]}
    assert rc.trait_required(must, "Attention to detail is essential.") == []


def _report_only(tmp_path, capsys, card, text, tag):
    """`tag` is in the QA line and the report's problem counts for `text`, never in fatal(), and a retry of it carries no
    feedback for it."""
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"id": "x", "text": text}) + "\n")
    rc.qa("postings", {"x": card}, [{"id": "x", "text": text}])
    assert f"'{tag}': 1" in capsys.readouterr().out
    r = rc.report("postings", {"x": card}, ["x"], str(rows))
    assert r["problem_counts"].get(tag) == 1 and tag in r["problem_ids"]["x"]
    assert tag in rc.problems("postings", card, text) and tag not in rc.fatal("postings", card, text)
    assert tag in rc.NONFATAL
    assert tag not in [f["tag"] for f in rc.findings("postings", card, text)]


def test_a_required_trait_is_reported_not_fatal(tmp_path, capsys):
    """Decided 2026-10-19: no label counts a soft trait, and real postings almost always ask for one, so trait_required
    is report-only like soft_skill_added."""
    _report_only(tmp_path, capsys, FT, FT_GOOD + " Attention to detail is essential.", "trait_required")
    assert rc.fatal("postings", FT, FT_GOOD) == []
    assert rc.FEEDBACK_KEY not in json.loads(rc.retry_context("postings", FT, FT_GOOD + " Attention to detail is a must."))


def test_a_no_sponsorship_line_on_a_citizens_only_card_is_reported_not_fatal(tmp_path, capsys):
    """Decided 2026-10-19 (as 82efb17 for stance_conflict): citizens-only already excludes everyone who needs
    sponsorship, so 'We are unable to sponsor visas' there moves no label."""
    cit = FT | {"citizens_only": True}
    good = FT_GOOD + " This position is open only to U.S. citizens and lawful permanent residents (green card holders)."
    assert rc.fatal("postings", cit, good) == []
    _report_only(tmp_path, capsys, cit, good + " We are unable to sponsor visas.", "no_sponsor_added")
    assert rc.FEEDBACK_KEY not in json.loads(rc.retry_context("postings", cit, good + " We are unable to sponsor visas."))


def test_a_working_condition_is_never_added():
    """rq_00754 (transportation), rq_01446~no_sponsor (standing), rq_00193 (a subject foundation on an any-field degree)."""
    for bad in ("You will be working onsite in Boston, so reliable transportation to our location is necessary.",
                "The role requires standing for long periods and working efficiently.",
                "Candidates must be able to lift up to 50 pounds.", "A background check is required before the start date.",
                "Applicants must pass a drug screening."):
        assert rc.condition_added(FT, bad) != [], bad
        assert "condition_added" in rc.fatal("postings", FT, FT_GOOD + " " + bad), bad
    for ok in ("You will analyse transportation routes for our clients.", "Our team lifts each other up.",
               "You will run background research for reports."):
        assert rc.condition_added(FT, ok) == [], ok
    anyf = FT | {"degree_level": "bachelor"}
    bad = "While your degree may be in any discipline, you must possess a solid foundation in the life sciences."
    assert rc.condition_added(anyf, bad) != []
    assert rc.condition_added(anyf, "A bachelor's degree in any field is required.") == []
    # read-20 false alarms: a foundation asked without the open-field wording is the skill gates', not this one
    for ok in ("The role requires a solid foundation in technical concepts.",
               "Knowledge of biology supports your understanding of relevant scientific concepts."):
        assert rc.condition_added(anyf, ok) == [], ok
    assert rc.condition_added(anyf, "While the discipline of your doctorate is open, the role demands a robust foundation in "
                                    "healthcare sciences.") != []
    assert rc.condition_added(anyf | {"degree_fields": ["Biology"]}, bad) == []  # a field card's subject is the field gate's


def test_hours_and_pay_terms_are_the_cards():
    """rq_00586 (flexible part-time hours), rq_01221~no_opt (business hours), rq_00114~unstated (pay depends on experience)."""
    part = FT | {"employment_type": "part_time"}
    for bad in ("The employment type is part-time, suitable for those seeking flexible scheduling alongside other commitments.",):
        assert rc.hours_added(part, bad) != [], bad
    assert rc.hours_added(FT, "We offer a flexible schedule.") == []  # a full-time card's 'flexible' is no job type
    for bad in ("You will work standard business hours, Monday through Friday.", "The schedule includes weekend rotations.",
                "Occasional overtime is expected.", "This position works rotating shifts."):
        assert rc.hours_added(FT, bad) != [], bad
        assert "hours_added" in rc.fatal("postings", FT, FT_GOOD + " " + bad), bad
    for ok in ("The pay is $20 an hour, about $41,600 a year based on 40 hours per week.",
               "You will be scheduling shifts for the front desk staff.", "We host weekend events for families.",
               # rq_00962: the store's hours a customer duty is done in
               "As an onsite role, you will be present in the store to handle fresh inventory and assist customers during "
               "business hours.",
               "You will report to the lab manager."):
        assert rc.hours_added(FT, ok) == [], ok
    for bad in ("You will troubleshoot hardware issues as they arise during business hours.",  # rq_00218, the worker's hours
                "You will assist customers during business hours and work weekends.",
                "You will be present in the store during business hours."):
        assert rc.hours_added(FT, bad) != [], bad
    for bad in ("Pay depends on experience.", "The salary is commensurate with qualifications.",
                "Compensation is based on the selected candidate's qualifications.", "The role pays $50,000 plus a bonus."):
        assert rc.pay_terms_added(FT, bad) != [], bad
        assert "pay_terms_added" in rc.fatal("postings", FT, FT_GOOD + " " + bad), bad
    for ok in ("The pay is $20 an hour, about $17,000 a year depending on hours worked.", "Pay is $41,600 to $52,000 a year.",
               "That is about $16,000 to $18,000 a year depending on the exact hours worked.",  # rq_00869
               "We offer $80,000 to $96,000 a year, commensurate with the responsibilities of the position."):  # rq_01027
        assert rc.pay_terms_added(FT, ok) == [], ok
    assert rc.pay_terms_added(FT | {"salary_min": None, "salary_max": None}, "Pay depends on experience.") == []  # neg_pay's


def test_a_multi_city_posting_names_its_cities_as_alternatives():
    """rq_01245~no_opt: 'at our locations in Coral Gables, FL and Miami, FL' reads as both; rq_00414 'across both'."""
    assert not rc.locations_not_alternatives(MULTI, MULTI_GOOD)
    bad = MULTI_GOOD.replace("based in Dublin, OH or Kissimmee, FL", "based at our locations in Dublin, OH and Kissimmee, FL")
    assert rc.locations_not_alternatives(MULTI, bad)
    assert "locations_not_alternatives" in rc.fatal("postings", MULTI, bad)
    assert not rc.locations_not_alternatives(MULTI, bad + " You will work at one of our offices.")
    # the review's plural and available wordings read as one or the other (rq_00322, rq_01163, rq_00299, rq_01391)
    for ok in ("We have openings in Dublin, OH and Kissimmee, FL.",
               "Positions are available at our offices in Dublin, OH and Kissimmee, FL.",
               "The available cities for this internship are Dublin, OH and Kissimmee, FL.",
               "This internship is available in two locations: Dublin, OH and Kissimmee, FL.",
               "This role is based in Ohio, with work locations available in Dublin, OH and Kissimmee, FL."):
        assert not rc.locations_not_alternatives(MULTI, MULTI_GOOD.replace(
            "based in Dublin, OH or Kissimmee, FL", "onsite") + " " + ok), ok
    # 'based' in the lead-in, or both / across, still reads as both (rq_01245, rq_00966, rq_00499~citizens)
    for bad in ("This role is based at our offices in Dublin, OH and Kissimmee, FL.",
                "Positions based at our offices in Dublin, OH and Kissimmee, FL.",
                "Elstow College invites applications for a role based at our campuses in Dublin, OH and Kissimmee, FL.",
                "Positions are available, with duties across our offices in Dublin, OH and Kissimmee, FL.",
                "Openings are available at both our Dublin, OH and Kissimmee, FL offices.",
                "Work is performed at our locations in Dublin, OH and Kissimmee, FL."):
        text = MULTI_GOOD.replace("based in Dublin, OH or Kissimmee, FL", "onsite") + " " + bad
        assert rc.locations_not_alternatives(MULTI, text), bad
        assert "locations_not_alternatives" in rc.fatal("postings", MULTI, text), bad
    stl = MULTI | {"locations": ["St. Louis, MO", "Chicago, IL"]}
    assert not rc.locations_not_alternatives(stl, "This is an onsite position with available locations in St. Louis, MO and "
                                                  "Chicago, IL.")  # rq_00712
    pete = MULTI | {"locations": ["St. Petersburg, FL", "Huntsville, AL"]}
    assert not rc.locations_not_alternatives(pete, "The job is onsite in St. Petersburg, FL or Huntsville, AL.")
    assert not rc.locations_not_alternatives(FT, FT_GOOD)  # one city
    for bad in ("You will teach across our Dublin, OH and Kissimmee, FL campuses.", "You will support all of our locations.",
                "Expect some travel between sites."):
        assert "locations_split" in rc.posting_problems(MULTI, MULTI_GOOD + " " + bad), bad
    for ok in ("The role requires adaptability across all project phases.", "We ensure that all plants receive the care they need."):
        assert rc.locations_split(MULTI, ok) == [], ok


def test_a_citizens_only_card_gives_no_reason_and_no_sponsorship_stance():
    """rq_00130 ('due to the nature of our projects and contractual obligations', 'unable to sponsor ... at this time'),
    rq_00374~citizens ('As a county government position')."""
    cit = FT | {"citizens_only": True}
    good = FT_GOOD + " This position is open only to U.S. citizens and lawful permanent residents (green card holders)."
    assert rc.citizens_reason_added(cit, good) == [] and rc.no_sponsor_added(cit, good) == []
    for bad in ("Please note that due to the nature of our projects and contractual obligations, there are specific "
                "eligibility criteria for this role.",
                "As a county government position, there are specific eligibility criteria regarding work authorization.",
                "Eligibility for this role is restricted by federal regulations."):
        assert rc.citizens_reason_added(cit, bad) != [], bad
        assert "citizens_reason_added" in rc.fatal("postings", cit, good + " " + bad), bad
    assert rc.citizens_reason_added(FT, "Eligibility is restricted by federal regulations.") == []
    # rq_00591~citizens: the card's own contract job is no contract reason
    assert rc.citizens_reason_added(cit | {"employment_type": "contract"}, "This contract position is subject to specific work "
                                                                          "authorization rules.") == []
    assert rc.citizens_reason_added(cit | {"e_verify": True}, "We use E-Verify to confirm eligibility, as federal law "
                                                                "requires.") == []
    bad = "We are unable to sponsor other visa types or work authorizations."
    assert rc.no_sponsor_added(cit, bad) != []
    assert "no_sponsor_added" in rc.posting_problems(cit, good + " " + bad)
    assert "no_sponsor_added" not in rc.fatal("postings", cit, good + " " + bad)  # reported, not fatal
    assert rc.no_sponsor_added(cit | {"sponsorship": "will_not_sponsor"}, bad) == []  # the card's own stance
    assert rc.no_sponsor_added(FT, bad) == []


def test_hybrid_remote_days_have_no_place_of_their_own():
    """rq_00330: 'completing tasks remotely from a location of your choice'."""
    hyb = FT | {"work_mode": "hybrid"}
    assert rc.hybrid_anywhere(hyb, "On remote days you complete tasks remotely from a location of your choice.") != []
    assert rc.hybrid_anywhere(hyb, "You will work from home two days a week.") == []
    assert rc.hybrid_anywhere(FT, "You can work from anywhere.") == []  # not hybrid: mode_conflict's


def test_a_resume_names_no_higher_degree_than_the_card():
    """cd_v_00639_r2: 'Currently pursuing advanced studies in cybersecurity' on an enrolled bachelor's card."""
    bach = CAND | {"graduation_year": 2028}
    for bad in ("Currently pursuing advanced studies in cybersecurity.", "Completing advanced academic training in nutrition.",
                "Pursuing a master's degree in biology.", "Applying lab skills during graduate studies."):
        assert rc.higher_degree_implied(bach, bad) != [], bad
        assert "higher_degree_implied" in rc.resume_problems(bach, "Candidate\n" + bad), bad
    for ok in ("Coursework: Advanced Programming, Data Structures", "Recent graduate with lab experience.",
               "Undergraduate studies in biology.", "Completed advanced coursework in organic chemistry."):
        assert rc.higher_degree_implied(bach, ok) == [], ok
    master = CAND | {"degree_level": "master"}
    assert rc.higher_degree_implied(master, "Graduate studies in biology.") == []
    assert rc.higher_degree_implied(master, "Pursuing a PhD in biology.") != []
    assert rc.higher_degree_implied(CAND | {"degree_level": "doctoral"}, "Pursuing a PhD in biology.") == []


def test_a_resume_never_names_a_skill_its_pairs_say_it_lacks(monkeypatch):
    """cd_v_00270_must_skill ('public accounting' under 'missing required skill: Accounting'), cd_rt_076c07da45 (a real-test
    resume listing Computer Architecture under 'needs one of: ..., Computer architecture, ...')."""
    card = CAND | {"skills": ["Microsoft Excel", "Cost accounting"]}
    assert rc.pair_lacked_named(card, "Three years in public accounting and financial analysis.", {"Accounting"}) == ["Accounting"]
    assert rc.pair_lacked_named(card, "- Computer Architecture: memory hierarchies", {"Computer architecture"}) == \
        ["Computer architecture"]
    assert rc.pair_lacked_named(card, "Built budgets in Microsoft Excel.", {"Accounting"}) == []
    assert rc.pair_lacked_named(card, "Skills: Cost accounting", {"Accounting"}) == []  # inside a held, longer skill name
    monkeypatch.setattr(rc, "_PAIR_LACKED", {rc._card_key(card): {"Accounting"}})
    assert "pair_lacked_named" in rc.resume_problems(card, "Candidate\nBoston, MA\n\nThree years in public accounting.")
    assert "pair_lacked_named" not in rc.resume_problems(CAND, "Candidate\nBoston, MA\n\nThree years in public accounting.")
    fs = rc._resume_findings(card, "Candidate\nBoston, MA\n\nThree years in public accounting.", ["pair_lacked_named"])
    assert fs[0]["quote"] and "does not have Accounting" in fs[0]["card_says"]


def test_a_resume_without_jobs_lists_no_dated_role():
    """cd_v_00027_r0_years_far_short: 'Academic Tutor, August 2023 - Present' under Volunteer Experience, pair 'you have 0'."""
    zero = CAND | {"years": 0}
    assert rc.jobs_of(zero) == []
    for bad in ("Candidate\n\nVolunteer Experience\nAcademic Tutor\nAugust 2023 – Present\n",
                "Candidate\n\nCommunity Safety Volunteer\nRetail Associate / Green Valley Market, Brentwood, TN / June 2024 – August 2025\n",
                "Candidate\n\nProjects\nCapstone Team Lead, January 2025 – May 2025\n"):
        assert rc.no_job_dated_role(zero, bad) != [], bad
        assert "no_job_dated_role" in rc.resume_problems(zero, bad), bad
    for ok in ("Candidate\n\nEducation\nBachelor of Science in Biology\nBoston University\nSeptember 2022 – May 2026\n",
               "Candidate\n\nVolunteer Experience\nTutor at the public library, helping students read.\n",
               "Candidate\n\nProjects\nWater quality study, January 2025 – May 2025\n"):
        assert rc.no_job_dated_role(zero, ok) == [], ok
    assert rc.no_job_dated_role(CAND | {"years": 2}, "Candidate\n\nVolunteer Experience\nTutor\nAugust 2023 – Present\n") == []


def test_placeholder_school_and_contact_markers_are_artifacts():
    for bad in ("Bachelor of Science\nUniversity located in United States", "Logistics Coordinator | [Email Redacted] | [Phone Redacted]"):
        assert rc.ARTIFACTS.search(bad), bad
    for ok in ("University of Texas, Austin, TX", "Skills: Excel [advanced]"):
        assert not rc.ARTIFACTS.search(ok), ok


def test_every_new_gate_sends_a_quote_back():
    """A retry is worth its credit only when the feedback quotes what to fix."""
    cit = FT | {"citizens_only": True}
    cases = [("postings", FT, FT_GOOD + " A background check is required.", "condition_added"),
             ("postings", FT, FT_GOOD + " You will work standard business hours.", "hours_added"),
             ("postings", FT, FT_GOOD + " Pay depends on experience.", "pay_terms_added"),
             ("postings", cit, FT_GOOD + " Due to federal regulations, eligibility is limited.", "citizens_reason_added"),
             ("postings", FT | {"work_mode": "hybrid"}, FT_GOOD + " Remote days can be worked from anywhere.", "hybrid_anywhere"),
             ("postings", FT | {"degree_level": "bachelor"}, FT_GOOD + " It is for people who have completed their formal education.",
              "education_done_added"),
             ("postings", FT, FT_GOOD + " Работа в офисе.", "non_latin_script"),
             ("resumes", CAND, "Candidate\nBoston, MA\n\n- Handled scheduling and visitor接待.", "non_latin_script"),
             ("resumes", CAND | {"graduation_year": 2028}, "Candidate\nCurrently pursuing advanced studies in biology.",
              "higher_degree_implied"),
             ("resumes", CAND, "Candidate\n\nVolunteer Experience\nAcademic Tutor\nAugust 2023 – Present\n", "no_job_dated_role")]
    for kind, card, text, tag in cases:
        fs = (rc._posting_findings if kind == "postings" else rc._resume_findings)(card, text, [tag])
        assert fs[0].get("quote") and "failed" not in fs[0]["card_says"], (tag, fs)
        assert rc.FEEDBACK_KEY in json.loads(rc.retry_context(kind, card, text)), tag
    bad = MULTI_GOOD.replace("based in Dublin, OH or Kissimmee, FL", "based in Dublin, OH and Kissimmee, FL")
    fs = rc._posting_findings(MULTI, bad, ["locations_not_alternatives"])
    assert "Dublin, OH or Kissimmee, FL" in fs[0]["missing"] and "never 'in A and B'" in fs[0]["card_says"]
