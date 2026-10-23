"""clean_real_checked: the checked cards it reads, the kinds it guards a relabeled row by, the hidden-profile draws
(label-blind, consistent people, preferences only where the posting shows the fact) and the label balance."""

import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import clean_real_checked as crc  # noqa: E402
from matcher.rules import label  # noqa: E402
from matcher.sample import wanted_role_words  # noqa: E402
from matcher.schema import CandidateCard, RequirementCard  # noqa: E402
from matcher.variants import industry_clash, prefs_fit  # noqa: E402

CARD = {"source_posting_id": "ats_x", "title": "Data Science Intern", "field": "Data Scientist", "onet_soc": None,
        "seniority": "entry", "industry": "finance", "degree_level": "none", "enrolled_in": "bachelor",
        "degree_fields": [], "related_ok": False, "min_years": 0, "must_skills": [["Python"], ["SQL"]],
        "nice_skills": ["Tableau", "Spark"], "license": None, "locations": ["Chicago, IL"], "work_mode": "onsite",
        "sponsorship": "unstated", "citizens_only": False, "e_verify": False, "salary_min": 50000,
        "salary_max": 62400, "employment_type": "internship"}
CAND = {"field": "Data Scientist", "degree_level": "bachelor", "degree_field": "Statistics", "graduation_year": 2028,
        "years": 0, "skills": ["Python", "SQL", "Tableau", "Spark"], "licenses": [], "current_location": "Chicago, IL",
        "willing_locations": [], "remote_preference": "any", "work_auth": "citizen_or_pr", "salary_floor": None,
        "employment_types_wanted": [], "industries_wanted": [], "industries_avoided": [], "roles_avoided": [],
        "education_country": "", "language": ""}
POSTING = ("Title: Data Science Intern\nCompany: Acme Bank\nLocations: Chicago, IL\n\nAcme Bank is a finance company. "
           "This onsite internship in our Chicago office pays $50,000 - $62,400 a year.\nRequired: Python and SQL. "
           "Must be pursuing a bachelor's degree.\nPreferred: Tableau, Spark.")


def card(**kw) -> RequirementCard:
    return RequirementCard.from_dict(CARD | kw)


def cand(**kw) -> CandidateCard:
    return CandidateCard.from_dict(CAND | kw)


# ---------------------------------------------------------------- checked cards


def test_nice_extracted_keeps_the_extracted_list_with_the_checkers_moves():
    ext = {"must_skills": [["Excel"], ["Python"]], "nice_skills": ["SQL", "Git", "communication"]}
    chk = {"must_skills": [["Python"], ["SQL"]], "nice_skills": ["Excel", "Docker", "Kubernetes"]}
    # SQL moved to must leaves nice; Excel moved from must joins it; Docker/Kubernetes (checker-only) stay out
    assert crc.nice_extracted(chk, ext) == ["Git", "communication", "Excel"]


def test_drop_flag():
    assert crc.drop_flag([{"field": "drop", "note": "Canadian posting"}]) == "Canadian posting"
    assert crc.drop_flag(["dropped RAG from nice", {"field": "must_skills", "note": "x"}]) is None


def test_load_checked(tmp_path):
    cc = tmp_path / "cc"
    cc.mkdir()
    rows = [{"posting_id": "ats_a", "split": "train", "card": CARD | {"source_posting_id": "ats_a", "nice_skills": ["Rust"]},
             "flags": []},
            {"posting_id": "ats_b", "split": "dev", "card": CARD, "flags": [{"field": "drop", "note": "high school"}]}]
    (cc / "checked_1.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    v2 = tmp_path / "v2.jsonl"
    v2.write_text("\n".join(json.dumps({"id": i, "card": CARD}) for i in ("ats_a", "ats_b", "ats_c")))
    fresh = tmp_path / "fresh.jsonl"
    fresh.write_text(json.dumps({"id": "ats_c", "card": CARD | {"enrolled_in": "master", "nice_skills": ["Go"]}}))
    verd = tmp_path / "verd.jsonl"
    verd.write_text(json.dumps({"id": "ats_c", "wrong": {"enrolled_in": ["none", "or recent graduates"]},
                                "debatable": {"min_years": [1, "one year"]}, "note": ""}))
    out, rep = crc.load_checked("extracted", cc, v2, fresh, verd, {"ats_a": "train", "ats_b": "dev", "ats_c": "dev"})
    assert sorted(out) == ["ats_a", "ats_c"] and "ats_b" in rep["dropped"]
    assert out["ats_a"]["card"].nice_skills == ("Tableau", "Spark")  # the extracted nice list, not "Rust"
    c = out["ats_c"]["card"]
    assert (c.enrolled_in, c.min_years, c.nice_skills) == ("none", 1, ("Tableau", "Spark"))
    with pytest.raises(SystemExit):
        crc.load_checked("extracted", cc, v2, fresh, verd, {"ats_a": "dev", "ats_c": "dev"})


# ---------------------------------------------------------------- the kinds a relabeled row is guarded as


@pytest.mark.parametrize("c,kinds", [
    (dict(skills=["Python", "Tableau", "Spark"]), {"must_skill"}),
    (dict(skills=["Python", "SQL", "Tableau"]), set()),  # one nice-to-have short never decides
    (dict(current_location="Denver, CO"), {"location"}),
    (dict(current_location="Denver, CO", willing_locations=["Chicago, IL"]), {"relocate"}),
    (dict(graduation_year=2025), {"enrollment_graduated"}),
    (dict(employment_types_wanted=["full_time"]), {"employment_type"}),
    (dict(industries_avoided=["finance"]), {"pref_industry_avoided"}),
])
def test_effective_kinds(c, kinds):
    cd, rc = cand(**c), card()
    bucket, reasons, _ = label(cd, rc)
    assert crc.effective_kinds(cd, rc, bucket, reasons) == kinds


def test_effective_kinds_nice_gap():
    rc = card(nice_skills=["Tableau", "Spark", "Airflow", "dbt"])
    cd = cand(skills=["Python", "SQL"])
    bucket, reasons, _ = label(cd, rc)
    assert bucket == "Explore" and crc.effective_kinds(cd, rc, bucket, reasons) == {"nice_skill"}


@pytest.mark.parametrize("reason,check,fields", [
    ("it is an internship role and you want full-time", "employment_type", ("employment_types_wanted",)),
    ("pay tops out at $62,400, under your $80,000 floor", "preferences", ("salary_floor",)),
    ("finance is an industry you said you avoid", "preferences", ("industries_avoided",)),
    ("finance is not one you listed as wanted", "preferences", ("industries_wanted",)),
    ("title includes intern, a role you said you avoid", "preferences", ("roles_avoided",)),
    ("the role is onsite; you said remote only", "location", ("remote_preference",)),
    ("based in Chicago, IL; you are not there or willing to move", "location", ()),
])
def test_hidden_fields_of(reason, check, fields):
    assert crc.hidden_fields_of(reason, check) == fields


# ---------------------------------------------------------------- the hidden-profile draws


RESUME_SIDE = ("field", "degree_level", "degree_field", "graduation_year", "years", "skills", "licenses",
               "current_location", "work_auth", "education_country", "language")


def test_draw_hidden_touches_only_hidden_fields_and_is_seeded():
    c0 = cand(work_auth="f1_cpt")
    a = crc.draw_hidden(c0, card(), random.Random("s:1"), POSTING)
    b = crc.draw_hidden(c0, card(), random.Random("s:1"), POSTING)
    assert a == b
    for f in RESUME_SIDE:
        assert getattr(a, f) == getattr(c0, f)


def test_draw_hidden_gives_people_who_could_exist():
    rc = card()
    for k in range(400):
        c0 = cand(work_auth=("citizen_or_pr", "f1_cpt")[k % 2], current_location=("Chicago, IL", "Denver, CO")[k % 3 > 0])
        c = crc.draw_hidden(c0, rc, random.Random(f"t:{k}"), POSTING)
        assert prefs_fit(c) and not industry_clash(c)
        assert not set(c.roles_avoided) & wanted_role_words(c.employment_types_wanted, c.work_auth)
        assert list(c.employment_types_wanted) == [t for t in ("internship", "full_time", "part_time", "contract")
                                                   if t in c.employment_types_wanted]


def test_draw_hidden_label_blind_mix():
    """Over many draws a set value lands in more than one label, and the failing job-type list never names a type
    the posting text names (here 'full-time' is named, so the failing lists avoid it)."""
    rc = card()
    text = POSTING + "\nFull-time conversion possible."
    seen = {}
    for k in range(600):
        c = crc.draw_hidden(cand(current_location="Denver, CO"), rc, random.Random(f"m:{k}"), text)
        bucket = label(c, rc)[0]
        if c.employment_types_wanted and "internship" not in c.employment_types_wanted:
            assert "full_time" not in c.employment_types_wanted
        if c.willing_locations:
            seen.setdefault("willing", set()).add(bucket)
    assert seen["willing"] >= {"For you", "Excluded"}


def test_draw_hidden_leaves_unshown_facts_at_default():
    facts = {f: False for f in crc.PROBES} | {"industries_avoided": False}
    for k in range(200):
        c = crc.draw_hidden(cand(), card(), random.Random(f"f:{k}"), POSTING, facts)
        assert (c.employment_types_wanted, c.remote_preference, c.salary_floor, c.industries_wanted,
                c.industries_avoided) == ((), "any", None, (), ())


def test_shown_facts_reads_pay_from_the_text():
    resume = "Jane Doe\nChicago, IL\nB.S. Statistics, expected 2028\nSkills: Python, SQL, Tableau, Spark"
    with_pay = crc.shown_facts(cand(), card(), resume, POSTING)
    no_pay = crc.shown_facts(cand(), card(), resume, POSTING.replace(" pays $50,000 - $62,400 a year", ""))
    assert with_pay["salary_floor"] and not no_pay["salary_floor"]
    assert not crc.shown_facts(cand(), card(salary_max=None), resume, POSTING)["salary_floor"]


def test_shown_facts_does_not_depend_on_the_label():
    resume = "Jane Doe\nChicago, IL\nB.S. Statistics, graduated 2024\nSkills: Python, SQL, Tableau, Spark"
    skip = cand(graduation_year=2024)  # a graduate on a student role: Skip whatever the preferences
    assert label(skip, card())[0] == "Excluded"
    assert crc.shown_facts(skip, card(), resume, POSTING) == crc.shown_facts(
        replace(skip, graduation_year=2028), card(), resume, POSTING)


# ---------------------------------------------------------------- balance and the profile split


def test_balance_cuts_to_the_smallest_label_spread_over_postings():
    rows = [{"pair_id": f"p{i}", "rq_id": f"ats_{i % 7}", "bucket": b}
            for i, b in enumerate(["For you"] * 5 + ["Explore"] * 9 + ["Excluded"] * 20)]
    out = crc.balance(rows, seed=0)
    assert sorted(r["bucket"] for r in out).count("Excluded") == 5
    assert {r["bucket"] for r in out} == {"For you", "Explore", "Excluded"} and len(out) == 15
    assert out == crc.balance(rows, seed=0)
    assert len({r["rq_id"] for r in out if r["bucket"] == "Excluded"}) == 5  # five rows, five postings


def test_profile_lines():
    from matcher.profile import render_profile
    p = render_profile(cand(salary_floor=50000, industries_wanted=["finance"]))
    hidden, other = crc.profile_lines(p, True), crc.profile_lines(p, False)
    assert "Minimum salary: $50,000" in hidden and "Industries wanted: finance" in hidden
    assert "Work authorization" in other and "Currently pursuing" in other and "salary" not in other
    assert len(hidden.split("\n")) == len(crc.HIDDEN_LINES) == len(crc.HIDDEN)


# ---------------------------------------------------------------- checked real dev v2


def _row(pid, post, bucket):
    return {"pair_id": pid, "rq_id": post, "bucket": bucket}


def test_past_limit():
    assert crc.past_limit("degree_ceiling", "bachelor", cand(degree_level="master", graduation_year=2028))
    assert not crc.past_limit("degree_ceiling", "master", cand(degree_level="master", graduation_year=2028))
    # a graduate is not studying above the ceiling: the card's enrollment check already fails them
    assert not crc.past_limit("degree_ceiling", "bachelor", cand(degree_level="master", graduation_year=2024))
    assert crc.past_limit("exp_cap", 2, cand(years=3)) and not crc.past_limit("exp_cap", 2, cand(years=2))


def test_dev_v2_decisions_remove_whole_postings_and_check_exact_limits():
    restr = {"p_school": ("school_only", "read", "MIT only", None),
             "p_cap_hit": ("exp_cap", "read", "0-2 years", 2),
             "p_cap_skip_only": ("exp_cap", "read", "0-2 years", 2),
             "p_ceiling_ok": ("degree_ceiling", "read", "bachelor's", "bachelor")}
    files = {"builder": [_row("a", "p_school", "Excluded"), _row("b", "p_cap_hit", "For you"),
                         _row("c", "p_cap_skip_only", "Excluded"), _row("d", "p_ceiling_ok", "Explore"),
                         _row("e", "p_clear", "For you")]}
    cands = {"builder": {"a": CAND, "b": CAND | {"years": 4}, "c": CAND | {"years": 9}, "d": CAND, "e": CAND}}
    dec = crc.dev_v2_decisions(files, cands, restr, frozenset({"p_clear"}))
    # a Skip row past a cap is still right, so it does not remove its posting
    assert {i for i, d in dec.items() if d["remove"]} == {"p_school", "p_cap_hit"}
    assert dec["p_cap_hit"]["past_limit_rows"] == 1 and dec["p_clear"]["kind"] is None
    with pytest.raises(SystemExit):  # a posting nobody read stops the build
        crc.dev_v2_decisions(files, cands, restr, frozenset())


def test_restriction_table_is_consistent():
    assert not set(crc.RESTRICTIONS) & crc.READ_CLEAR
    for i, (kind, source, quote, limit) in crc.RESTRICTIONS.items():
        assert kind in crc.REMOVE_KINDS + crc.CHECK_KINDS and source in ("read", "checker", "both") and quote
        assert (limit is not None) == (kind in crc.CHECK_KINDS), i
