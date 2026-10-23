import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


score = load("score_extraction")
extract = load("extract_postings")

TRUTH = {"degree_level": "none", "enrolled_in": "bachelor", "degree_fields": ["Computer Science"], "related_ok": True,
         "min_years": 0, "must_skills": [["C", "C++"], ["Python"]], "license": None, "locations": ["New York, NY"],
         "work_mode": "onsite", "sponsorship": "unstated", "citizens_only": False, "employment_type": "internship",
         "salary_min": None, "salary_max": None}


def test_exact_card_scores_all_right():
    t = score.tally([("a", dict(TRUTH), TRUTH)])
    assert t["all_label_strict"] == 1 and all(t["per"][f]["strict"] == 1 for f in score.FIELDS)


def test_must_skills_strict_and_lenient_differ_on_group_boundaries():
    split = {**TRUTH, "must_skills": [["C"], ["C++"], ["Python"]]}
    assert score.compare("must_skills", split, TRUTH) == (False, True)
    extra = {**TRUTH, "must_skills": [["C", "C++"], ["Python"], ["Excel"]]}
    assert score.compare("must_skills", extra, TRUTH) == (False, False)
    renamed = {**TRUTH, "must_skills": [["c", "C++"], ["python"]]}
    assert score.compare("must_skills", renamed, TRUTH) == (True, True)


def test_hyphens_and_case_do_not_count_as_errors():
    a = {**TRUTH, "must_skills": [["Object-oriented programming"]]}
    b = {**TRUTH, "must_skills": [["object oriented programming"]]}
    assert score.compare("must_skills", a, b) == (True, True)


def test_degree_fields_lenient_allows_partial_overlap():
    e = {**TRUTH, "degree_fields": ["Computer Science", "Math"]}
    t = {**TRUTH, "degree_fields": ["Computer Science", "Math", "Physics"]}
    assert score.compare("degree_fields", e, t) == (False, True)


def test_wrong_field_breaks_all_label_fields():
    t = score.tally([("a", {**TRUTH, "sponsorship": "will_not_sponsor"}, TRUTH)])
    assert t["all_label_strict"] == 0 and t["per"]["sponsorship"]["strict"] == 0 and t["all_but_skills_strict"] == 0


def test_fresh_tally_counts_debatable_as_not_strict(tmp_path, capsys):
    rows = [{"id": "a"}, {"id": "b", "debatable": {"must_skills": [[], "x"]}},
            {"id": "c", "wrong": {"must_skills": [[], "x"], "sponsorship": ["unstated", "y"]}}]
    p = tmp_path / "v.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))

    class A:
        verdicts = str(p)
        json = None
        checked = None
        cards = None
        diff = None

    score.cmd_fresh(A)
    out = capsys.readouterr().out
    assert "| must_skills | 33% | 67% |" in out
    assert "1/3 = 33%" in out


@pytest.mark.skipif(not extract.POSTS.exists() or not extract.AUDIT_IDS.exists(), reason="needs data/")
def test_id_sets_partition_the_postings():
    sets = extract.id_sets(extract.load_postings())
    assert len(sets["audit"]) == 100 and len(sets["fresh"]) == 50
    assert not set(sets["audit"]) & set(sets["fresh"])
    assert not (set(sets["audit"]) | set(sets["fresh"])) & set(sets["rest"])
    assert len(sets["all"]) == len(set(sets["all"])) == 2655


def test_hand_truth_applies_corrections_and_suggestions():
    checked = {"a": {"sponsorship": "sponsors", "must_skills": [["x"]], "min_years": 0}}
    verdicts = [{"id": "a", "wrong": {"sponsorship": ["unstated", "why"]}, "debatable": {"must_skills": [[["y"]], "why"]}}]
    t = score.hand_truth(verdicts, checked)["a"]
    assert t == {"sponsorship": "unstated", "must_skills": [["y"]], "min_years": 0}


def test_load_requirements_skips_real_rows_that_fail_training_ok(tmp_path):
    import json as _json
    from matcher.pairs import load_requirements
    from matcher.schema import RequirementCard

    first = RequirementCard(source_posting_id="x", title="Data Intern", field="IT and data", onet_soc=None,
                            degree_level="none", min_years=0, work_mode="onsite", industry="technology",
                            employment_type="internship", sponsorship="unstated").to_dict()
    rows = [
        {"id": "rq_a", "card": first},
        {"id": "ats_us", "source": "extracted_real", "card": first,
         "meta": {"country": "US", "us": True, "mixed": False, "language": "en", "must_phrases": []}},
        {"id": "ats_ca", "source": "extracted_real", "card": first,
         "meta": {"country": "CA", "us": False, "mixed": False, "language": "en", "must_phrases": []}},
        {"id": "ats_fr", "source": "extracted_real", "card": first,
         "meta": {"country": "US", "us": True, "mixed": False, "language": "fr", "must_phrases": []}},
        {"id": "ats_ph", "source": "extracted_real", "card": first,
         "meta": {"country": "US", "us": True, "mixed": False, "language": "en", "must_phrases": [["data concepts"]]}},
    ]
    p = tmp_path / "cards.jsonl"
    p.write_text("".join(_json.dumps(r) + "\n" for r in rows))
    assert [r.rq_id for r in load_requirements(p)] == ["rq_a", "ats_us"]


def test_rest_us_keeps_us_english_postings_only():
    us = {"locations": ["Boston, MA"], "description": "Build data pipelines in Python."}
    ca = {"locations": ["Toronto, ON"], "description": "Build data pipelines in Python."}
    fr = {"locations": ["Montreal, QC"],
          "description": "Nous recherchons un stagiaire avec vous dans notre équipe pour les projets. " * 3}
    assert extract.us_english(us) and not extract.us_english(ca) and not extract.us_english(fr)


def _post(desc, title="Intern", locs=("Austin, TX",)):
    return {"title": title, "company": "Acme", "locations": list(locs), "description": desc}


def test_identical_postings_are_sent_once_and_share_one_card(monkeypatch):
    posts = {"ats_b": _post("Same text"), "ats_a": _post("Same text"), "ats_c": _post("Other text"),
             "ats_d": _post("Same text", locs=("Remote",))}
    tw = extract.twins(posts, list(posts))
    assert tw == {"ats_a": "ats_a", "ats_b": "ats_a", "ats_c": "ats_c", "ats_d": "ats_d"}  # locations are part of the text
    monkeypatch.setattr(extract, "have_raw", lambda name: set())
    assert extract.todo_ids("x", list(posts), posts) == ["ats_a", "ats_c", "ats_d"]
    monkeypatch.setattr(extract, "have_raw", lambda name: {"ats_b"})  # a twin came back: nothing to resend
    assert set(extract.todo_ids("x", list(posts), posts)) == {"ats_c", "ats_d"}


def test_training_ok_keeps_one_copy_of_identical_postings():
    from matcher.blueprint import training_ok
    meta = {"us": True, "language": "en", "must_phrases": []}
    assert training_ok(meta) and not training_ok({**meta, "dup_of": "ats_a"})


def test_todo_ids_skips_postings_another_set_already_has(monkeypatch):
    posts = {"ats_a": _post("A"), "ats_b": _post("B"), "ats_c": _post("C"), "ats_d": _post("A")}
    monkeypatch.setattr(extract, "have_raw", lambda name: {"ats_a", "ats_b"} if name == "rest_us" else set())
    assert extract.todo_ids("rest", list(posts), posts) == ["ats_c"]  # ats_d has ats_a's text


def test_same_description_keeps_one_row_for_training():
    from matcher.blueprint import training_ok
    posts = {"ats_a": _post("Same", title="Intern"), "ats_b": _post("Same", title="Co-op"),
             "ats_c": _post("Same", title="Intern", locs=("London",)), "ats_d": _post("Other")}
    # the company string and spacing or punctuation do not split a description ("KLA" vs "KLA Corporation")
    posts["ats_x"] = {**_post("  same\n\n"), "company": "Acme Corporation"}
    posts["ats_y"] = {**_post("Same."), "company": "Other Co"}
    rep = extract.same_description(posts, list(posts), {"ats_a": False, "ats_b": True, "ats_c": True, "ats_d": True})
    assert rep == {"ats_a": "ats_b", "ats_b": "ats_b", "ats_c": "ats_b", "ats_d": "ats_d", "ats_x": "ats_b",
                   "ats_y": "ats_b"}
    meta = {"us": True, "language": "en", "must_phrases": []}
    assert not training_ok({**meta, "same_description_as": "ats_b"})


def test_training_ok_leaves_out_the_real_test_postings():
    from matcher.blueprint import training_ok
    meta = {"us": True, "language": "en", "must_phrases": []}
    assert not training_ok({**meta, "in_real_test": True})


def test_merge_twins_ignore_company_string_and_spacing():
    posts = {"ats_a": {**_post("Build  data pipelines - in Python."), "company": "KLA"},
             "ats_b": {**_post("build data pipelines \u2013 in Python"), "company": "KLA Corporation"},
             "ats_c": {**_post("Build data pipelines in Python.", title="Co-op"), "company": "KLA"}}
    assert extract.twins(posts, list(posts), None, norm=True) == {"ats_a": "ats_a", "ats_b": "ats_a", "ats_c": "ats_c"}
    assert extract.twins(posts, list(posts), None)["ats_b"] == "ats_b"  # what adapt is sent stays exact
