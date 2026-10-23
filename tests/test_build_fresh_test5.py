"""Fresh test 5's adapter: a round-2 Simplify ATS posting in the pool-posting fields the sampler reads, rendering to
exactly the text extract_postings.posting_text gives the ATS posting (amendment 6 inputs.posting)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import build_fresh_test as b  # noqa: E402
import extract_postings as ext  # noqa: E402


@pytest.fixture(autouse=True)
def _default_set():
    yield
    b.use_set("1")  # use_set rebinds module globals; leave set 1 for the other test files

ATS = {"simplify_id": "0000-test", "company": "Acme Robotics ", "title": "Machine Learning Engineer Intern",
       "locations": ["Boston, MA", "Remote in USA"], "url": "https://jobs.example.com/1", "ats": "greenhouse",
       "category": "AI/ML/Data", "terms": ["Summer 2026"],
       "description": "About us\n\nRequirements:\n- Python and PyTorch\n- Currently enrolled in a BS or MS program"}


def test_fields():
    p = b.round2_adapter(ATS, "real_test")
    assert p["key"] == "ats_0000-test"
    assert p["employer"] == ATS["company"]  # as written, so the Company line is the ATS one
    assert p["emp_key"] == "acme robotics"
    assert p["text"] == ATS["description"] and p["chars"] == len(ATS["description"])
    assert p["field"] in {"data", "software", "internship", "other", "lab_bio", "health_nursing", "sales_ops"}
    assert p["length"] and p["country"] and p["source_set"] == "real_test"


def test_posting_text_round_trips():
    p = b.round2_adapter(ATS, "real_test")
    assert b.posting_text(p) == ext.posting_text(ATS, None)
    assert b.adapter(p)["locations"] == ATS["locations"]


def test_empty_description_and_no_locations():
    d = {**ATS, "description": None, "locations": []}
    p = b.round2_adapter(d, "real_train dev")
    assert p["text"] == "" and p["chars"] == 0
    assert b.posting_text(p) == ext.posting_text({**d, "description": ""}, None)


@pytest.mark.parametrize("bad", [["Boston, MA; Cambridge, MA"], [" Boston, MA"], [""]])
def test_location_that_would_not_round_trip_is_refused(bad):
    with pytest.raises(ValueError):
        b.round2_adapter({**ATS, "locations": bad}, "real_test")


def test_candidate_builds_on_an_adapted_posting():
    p = b.round2_adapter(ATS, "real_test")
    c = b.build_candidate(p, None, 0, "strong", True, b.mentions(p["text"]))
    banned = b.posting_titles(p)
    assert all(j["title"].casefold().strip() not in banned for j in c["writer"]["jobs"])


@pytest.mark.skipif(not b.ATS_POSTINGS.exists(), reason="data/ is local to the worktree")
def test_every_round2_posting_round_trips():
    ats = {"ats_" + d["simplify_id"]: d for d in b.read_jsonl(b.ATS_POSTINGS)}
    pool, notes = b.load_round2()
    assert notes["after_drop"] == 1632 and len(notes["dropped_by_amendment6"]) == 32
    for p in pool:
        assert b.posting_text(p) == ext.posting_text(ats[p["key"]], None), p["key"]
    keys = [(p["emp_key"], p["title"]) for p in pool]
    assert len({(e, t.casefold()) for e, t in keys}) == len(keys)


def _point_at(tmp_path):
    b.use_set("5")
    b.OUT = tmp_path
    b.STATE, b.CANDS, b.RESUMES, b.TRIES = tmp_path / "adaption", tmp_path / "c.jsonl", tmp_path / "r.jsonl", tmp_path / "t.jsonl"
    b.WRITER_STAGE = tmp_path / "stage"


class _A:
    limit = 0


def test_sonnet_writer_round(tmp_path):
    """prepare -> agents save files -> absorb gates them with gate() as Adaption's completions were; a failed try
    goes out again with its feedback, a row with no saved file goes out again with no try counted."""
    _point_at(tmp_path)
    p = b.round2_adapter(ATS, "real_test")
    ms = b.mentions(p["text"])
    cands = [b.build_candidate(p, None, s, a, True, ms) for s, a in enumerate(("strong", "near", "gap"))]
    b.write_jsonl(b.CANDS, cands)
    b.render_sonnet_prepare(_A())
    st = json.loads((b.STATE / "sonnet_0.json").read_text())
    d0 = b.stage_dir("sonnet_0")
    assert st["n"] == 3 and (d0.parent / b.WRITER_PROMPT).read_text() == b.WRITER_UNIVERSAL + "\n\n" + b.WRITER_BLUEPRINT
    assert not list(d0.parent.parent.glob("*.jsonl"))  # nothing of the set's own files beside the writers
    rows = [r for c in st["chunks"] for r in b.read_jsonl(d0 / c)]
    assert len(st["chunks"]) == 3  # three candidates of one posting: one a chunk
    assert {r["row_id"] for r in rows} == {c["row_id"] for c in cands}
    assert all(r["context"] == b.writer_context(next(c for c in cands if c["row_id"] == r["row_id"])) for r in rows)
    with pytest.raises(SystemExit):  # one open batch at a time
        b.render_sonnet_prepare(_A())
    (d0 / "out" / f"{cands[0]['row_id']}.txt").write_text("too short\n")
    b.render_sonnet_absorb(_A())
    with pytest.raises(SystemExit):  # closed: nothing to absorb twice
        b.render_sonnet_absorb(_A())
    tries = b.read_jsonl(b.TRIES)
    assert len(tries) == 1 and tries[0]["problems"] and tries[0]["try"] == 1
    b.render_sonnet_prepare(_A())
    st1 = json.loads((b.STATE / "sonnet_1.json").read_text())
    rows = {r["row_id"]: r for c in st1["chunks"] for r in b.read_jsonl(b.stage_dir("sonnet_1") / c)}
    assert len(rows) == 3
    assert rows[cands[0]["row_id"]]["context"]["fix_from_last_try"] == tries[0]["problems"]
    assert "fix_from_last_try" not in rows[cands[1]["row_id"]]["context"]


def test_adaption_modes_refused_for_set5(tmp_path):
    _point_at(tmp_path)

    class A:
        mode, limit = "run", 0
    with pytest.raises(SystemExit):
        b.cmd_render(A())


def test_deal_never_puts_two_candidates_of_one_posting_in_a_chunk():
    import random
    todo = [f"r{p}_{k}" for p in range(97) for k in range(4)]
    random.Random(1).shuffle(todo)
    post = {i: i.split("_")[0] for i in todo}
    chunks = b.deal(todo, post, 24)
    assert sorted(x for c in chunks for x in c) == sorted(todo)
    assert all(len(c) <= 24 and len({post[i] for i in c}) == len(c) for c in chunks)
    small = b.deal(todo[:10], post, 24)  # a small retry batch still keeps siblings apart
    assert all(len({post[i] for i in c}) == len(c) for c in small)


def test_round2_walk(monkeypatch, tmp_path):
    """The seeded walk: sorted ids, Random(SEED).shuffle, an employer at the cap is passed over; the test is the
    walk's first N_POSTINGS."""
    import random
    _point_at(tmp_path)
    b.POSTINGS, b.CARDS = tmp_path / "postings.jsonl", tmp_path / "cards.jsonl"
    pool = [b.round2_adapter({**ATS, "simplify_id": f"{i:03d}", "company": f"Emp{i % 3}", "title": f"Intern {i}"},
                             "real_test") for i in range(40)]
    monkeypatch.setattr(b, "load_round2", lambda: (pool, {"dropped_by_amendment6": [], "same_employer_and_title_ids": [],
                                                          "not_english": []}))
    monkeypatch.setattr(b, "N_POSTINGS", 10)
    monkeypatch.setattr(b, "ATS_CARDS", tmp_path / "v2.jsonl")
    b.write_jsonl(b.ATS_CARDS, [{"id": p["key"], "card": {}} for p in pool])
    b.cmd_sample_round2()
    order = sorted(p["key"] for p in pool)
    random.Random(20261022).shuffle(order)
    emp = {p["key"]: p["emp_key"] for p in pool}
    want, n = [], {}
    for k in order:
        if n.get(emp[k], 0) < 7:
            n[emp[k]] = n.get(emp[k], 0) + 1
            want.append(k)
    walk = json.loads((tmp_path / "sample_walk.json").read_text())["walk"]
    assert walk == want and len(walk) == 21
    assert sorted(p["key"] for p in b.read_jsonl(b.POSTINGS)) == sorted(want[:10])
