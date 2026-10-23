"""build_train_mix3: the class-standing screen on the phrasings the checkers and the data review found, the
restriction table (posting-level drops, candidate fit tests, Skip rows left alone), and the synthetic byte check."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_train_mix3 as m  # noqa: E402
from matcher.schema import CandidateCard  # noqa: E402

YEARS = (2024, 2027, 2028, 2029, 2030)


def bach(y: int, **kw) -> CandidateCard:
    return CandidateCard(degree_level="bachelor", graduation_year=y, **kw)


def caught(s: str, **kw) -> set[int]:
    return {y for y in YEARS if m.screen_class_standing(bach(y, **kw), s)}


@pytest.mark.parametrize("sentence, fails", [
    ("- Current student, Junior or higher (as of Fall 2027)", {2024, 2030}),  # Xcel, the data review's row
    ("Current UT student in undergraduate programs, and has completed the first year of an undergraduate program.",
     {2030}),  # UT Austin RA
    ("- Actively enrolled in a bachelor's degree (sophomore standing or above) in data science.", {2024, 2030}),
    ("If pursuing a bachelor's degree, completion of at least sophomore year (60 or more credit hours).", {2029, 2030}),
    ("● Sophomore or Junior year standing by the start of the internship", {2027, 2028}),
    ("- Must be a first-year (class of 2030) or second year (class of 2029) undergraduate", {2027, 2028}),
    ("This program gives current first and second year undergrad students a chance.", {2024, 2027, 2028}),
    ("GovSignals is looking for third- and fourth-year undergraduate students and recent graduates.", {2029, 2030}),
    ("- Rising Junior or Senior graduating in 2027 or 2028", {2030}),
    ("3rd Year of bachelor's degree completed in Finance or computer science.", {2028, 2029, 2030}),
    ("Open only to 2nd/3rd-year students.", {2024, 2027, 2030}),
])
def test_class_standing_catches(sentence, fails):
    assert caught(sentence) == fails


@pytest.mark.parametrize("sentence", [
    "Work with senior engineers and junior developers.",
    "Are a 4th or 5th year senior Ph.D student in AI",  # a PhD year: the restriction table's job
    "What Success Looks Like in the First Year:",
    "We build first-class tools for the second level of support.",
    "Rising sophomores, juniors, and seniors are encouraged to apply.",  # encouraged: preferred
    "Junior standing preferred.",
    "Senior level engineers mentor you.",
])
def test_class_standing_leaves(sentence):
    assert caught(sentence) == set()


def test_standing_only_screens_bachelors():
    s = "Junior or higher (as of Fall 2027)"
    assert m.screen_class_standing(CandidateCard(degree_level="master", graduation_year=2030), s) is None


def test_standing_at():
    assert m.standing_at(2028, 2026) == 3  # a junior this fall
    assert m.standing_at(2030, 2027) == 2  # a sophomore next fall


def test_ref_fall():
    assert m.ref_fall("Junior or higher (as of Fall 2027)") == 2027
    assert m.ref_fall("Junior standing by Spring 2027") == 2026
    assert m.ref_fall("Rising juniors") == m.CURRENT_YEAR + 1
    assert m.ref_fall("Junior standing") == m.CURRENT_YEAR


def row(rq: str, y: int, bucket: str = "For you", **kw) -> dict:
    return {"rq_id": rq, "bucket": bucket, "candidate": bach(y, **kw).to_dict()}


def test_remove_kinds_drop_the_posting_whatever_the_rows():
    i = "ats_6f1a27e9-3515-4a43-a644-80dbddacc4bb"  # ASU students only
    assert i in m.removed_postings([row(i, 2028, "Excluded")])
    assert "ats_unlisted" not in m.removed_postings([row("ats_unlisted", 2030)])


def test_check_kinds_drop_only_past_the_limit_on_non_skip_rows():
    i = "ats_29379ac0-6351-43a9-a3a3-cd3b286670cb"  # KLA: less than 1 year of experience
    assert i not in m.removed_postings([row(i, 2028, years=0), row(i, 2020, "Excluded", years=5)])
    assert i in m.removed_postings([row(i, 2028, years=0), row(i, 2020, "Explore", years=5)])


def test_every_kind_is_the_dev_sides():
    assert {r.kind for r in m.RESTRICTED.values()} <= set(m.REMOVE_KINDS) | set(m.CHECK_KINDS)
    assert not set(m.RESTRICTED) & set(m.REVIEWED_NOT_RESTRICTED)
    import clean_real_checked as ccr
    if not hasattr(ccr, "REMOVE_KINDS"):
        pytest.skip("clean_real_checked has no dev-v2 kinds here")
    assert m.REMOVE_KINDS == ccr.REMOVE_KINDS and m.CHECK_KINDS == ccr.CHECK_KINDS


def test_known_bad_are_xcel_and_ut():
    assert len(m.KNOWN_BAD) == 3
    assert sum("Xcel" in v for v in m.KNOWN_BAD.values()) == 2


def test_quotes_are_in_the_postings():
    import assemble_real_train as art
    try:
        raw = art.brt_raw()
    except (FileNotFoundError, OSError):
        pytest.skip("raw postings not here")
    if not raw:
        pytest.skip("raw postings not here")
    assert m.quote_problems(raw) == []


def test_synth_bytes_check(tmp_path):
    syn, out = tmp_path / "s.csv", tmp_path / "o.csv"
    rows = [("a, b\n\"q\"", "Keep — x"), ("c", "Skip — y")]
    with open(syn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "completion"])
        w.writerows(rows)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "completion"])
        w.writerows([rows[1], ("real", "Explore — z"), rows[0]])
    got = m.synth_bytes_check(out, syn)
    assert got == {"synthetic_records": 2, "missing_in_output": 0, "not_byte_equal_in_smoke": 0, "output_records": 3}
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "completion"])
        w.writerows([rows[1]])
    assert m.synth_bytes_check(out, syn)["missing_in_output"] == 1
