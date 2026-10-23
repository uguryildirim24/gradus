"""The real-posting training builder: near copies of the test postings stay out, and the split keeps every posting
(and every employer) in one split."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_real_train as brr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Distinct filler words: a description of n of them has no 6-word window in common with another of other words.
WORDS = [f"w{i}" for i in range(5000)]


def text(start: int, n: int = 300) -> str:
    return " ".join(WORDS[start:start + n])


def post(company: str, title: str, description: str) -> dict:
    return {"company": company, "title": title, "description": description, "locations": []}


TEST = {
    "t1": post("Acme Inc", "Data Science Intern", text(0)),
    "t2": post("Globex Corporation", "Mechanical Engineering Co-op", text(1000)),
}


# ---------------------------------------------------------------- titles


def test_title_core_drops_filler_and_stems():
    assert brr.title_core("Software Engineering Intern - Summer 2027") == brr.title_core("Software Engineer Intern")
    assert brr.title_core("Summer Intern 2027") == frozenset()


def test_title_sim():
    assert brr.title_sim("Data Science Intern", "Data Science Intern - Summer 2027") == 1.0
    assert brr.title_sim("Software Engineer Intern", "Software Engineer Intern - Commerce Applications") >= brr.TITLE_SIM
    assert brr.title_sim("Software Engineer Intern", "Mechanical Engineering Intern") < brr.TITLE_SIM
    assert brr.title_sim("Intern", "Summer Intern 2027") == 1.0  # no core word on either side


# ---------------------------------------------------------------- near copies


def near(p: dict) -> dict | None:
    return brr.TestIndex(TEST).near(p)


def test_same_company_similar_title_is_a_near_copy():
    hit = near(post("ACME", "Data Science Intern - Summer 2027", text(3000)))  # another text, same job
    assert hit and hit["rule"] == "same company, similar title" and hit["test_id"] == "t1"


def test_same_company_other_job_is_kept():
    # the employer's boilerplate: a third of the test posting's text, another title
    assert near(post("Acme", "Mechanical Design Intern", text(0, 100) + " " + text(3000, 200))) is None


def test_mostly_shared_text_is_a_near_copy_under_any_title():
    hit = near(post("Acme", "Mechanical Design Intern", text(0, 280) + " " + text(3000, 20)))
    assert hit and hit["rule"] == "most of the text shared"


def test_same_employer_under_another_name_is_a_near_copy():
    # half of the shorter text is the test posting's, under another company string (a subsidiary, a renamed employer)
    hit = near(post("Initech", "Thermal Analyst Intern", text(1000, 170) + " " + text(3000, 130)))
    assert hit and hit["rule"] == "same employer under another name" and hit["test_id"] == "t2"


def test_unrelated_posting_is_kept():
    assert near(post("Umbrella", "Data Science Intern", text(3000))) is None


def test_exclude_near_splits_the_pool():
    raw = {**TEST,
           "p_copy": post("Acme", "Data Science Intern", text(3000)),
           "p_other": post("Hooli", "Software Engineer Intern", text(3500)),
           "p_sub": post("Initech", "Thermal Analyst Intern", text(1000, 170) + " " + text(4000, 130))}
    kept, excluded, _, same_co = brr.exclude_near(["p_copy", "p_other", "p_sub"], raw, ["t1", "t2"])
    assert kept == ["p_other"]
    assert {e["id"] for e in excluded} == {"p_copy", "p_sub"}
    assert same_co == 1


# ---------------------------------------------------------------- split

RAW = {
    "a1": post("Acme", "Data Intern", text(0)),
    "a2": post("Acme Inc.", "Hardware Intern", text(300)),  # same company_key as a1
    "n1": post("Nissan", "Data Intern", text(600)),
    "n2": post("Nissan Global", "Planning Intern", text(600, 200) + " " + text(900, 100)),  # another name, shared text
    "u1": post("Umbrella", "Data Intern", text(1200)),
    "h1": post("Hooli", "Data Intern", text(1500)),
    "i1": post("Initech", "Data Intern", text(1800)),
    "g1": post("Globex", "Data Intern", text(2100)),
}


def test_components_join_one_company_and_one_employer_under_two_names():
    comps = brr.components(list(RAW), RAW)
    by = {i: tuple(c) for c in comps for i in c}
    assert by["a1"] == by["a2"]
    assert by["n1"] == by["n2"]
    assert len({by[i] for i in ("u1", "h1", "i1", "g1")}) == 4


def test_split_keeps_components_whole_and_forces_checked_postings_into_dev():
    comps = brr.components(list(RAW), RAW)
    split = brr.split_components(comps, {"n2"}, dev_share=0.3, seed=0)
    assert set(split) == set(RAW)  # every posting in exactly one split
    assert set(split.values()) <= {"train", "dev"}
    assert split["n1"] == split["n2"] == "dev"
    assert split["a1"] == split["a2"]
    assert sum(v == "dev" for v in split.values()) <= round(0.3 * len(RAW)) or split["n2"] == "dev"
    assert split == brr.split_components(comps, {"n2"}, dev_share=0.3, seed=0)  # deterministic


def test_cap_rows_round_robins_over_kinds():
    rows = [{"pair_id": f"p{k}{n}", "rq_id": "q", "variant_kind": k} for k in "abc" for n in range(3)]
    capped = brr.cap_rows(rows, 4)
    assert len(capped) == 4 and {r["variant_kind"] for r in capped} == set("abc")
    assert brr.cap_rows(rows, 0) == rows


# ---------------------------------------------------------------- the written build (data/ is not in git)

OUT = ROOT / "data/built/real_train"


@pytest.mark.skipif(not (OUT / "splits.json").exists(), reason="build_real_train.py build has not been run")
def test_written_build_has_no_test_posting_and_no_posting_in_two_splits():
    split = json.loads((OUT / "splits.json").read_text())
    tests = set(brr.test_ids())
    assert not tests & set(split)
    excluded = {r["id"] for r in brr.read_jsonl(OUT / "excluded.jsonl")}
    assert not excluded & set(split)
    rows = brr.read_jsonl(OUT / "rows.jsonl")
    assert all(r["split"] == split[r["rq_id"]] for r in rows)
    assert not {r["rq_id"] for r in rows} & tests
    # no near copy of a test posting slipped into a split
    raw = brr.load_raw()
    index = brr.TestIndex({i: raw[i] for i in tests})
    assert not [i for i in split if index.near(raw[i])]
    # no employer in two splits
    comp_split: dict[str, set[str]] = {}
    for c in brr.components(list(split), raw):
        comp_split[c[0]] = {split[i] for i in c}
    assert all(len(s) == 1 for s in comp_split.values())


def test_checked_card_dicts_sources(tmp_path):
    card = {"enrolled_in": "master", "min_years": 0}
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"id": "ats_c", "card": card}) + "\n")
    verd = tmp_path / "verd.jsonl"
    verd.write_text(json.dumps({"id": "ats_c", "wrong": {"enrolled_in": ["none", "or recent graduates"]},
                                "debatable": {"min_years": [1, "one year"]}, "note": ""}) + "\n")
    clean = tmp_path / "clean.jsonl"
    clean.write_text(json.dumps({"id": "ats_d", "split": "dev", "source": "x", "card": card}) + "\n")
    paths = dict(checked=raw, verdicts=verd, clean=clean)
    assert brr.checked_card_dicts("raw", **paths) == {"ats_c": card}  # the default: as read, uncorrected
    assert brr.checked_card_dicts("hand_truth", **paths) == {"ats_c": {"enrolled_in": "none", "min_years": 1}}
    assert brr.checked_card_dicts("clean", **paths) == {"ats_d": card}
    with pytest.raises(ValueError):
        brr.checked_card_dicts("fresh", **paths)
