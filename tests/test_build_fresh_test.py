"""The fresh-test / cycle-3 candidate generator: no held job title is the target posting's title, and the skills list
is the person's own (posting requirements met, not the posting's keyword list in the posting's order)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]

import build_fresh_test as b  # noqa: E402

# titles that are also kit titles, so the old copy and the kit draw can both hit them
POSTINGS = [
    ("Registered Nurse", "health_nursing"), ("Data Analyst", "data"), ("Software Engineer", "software"),
    ("Senior Scientist", "lab_bio"), ("Research Associate Intern", "internship"), ("Summer Intern", "internship"),
    ("Clinical Research Coordinator", "lab_bio"), ("Staff Accountant", "sales_ops"), ("Barista", "other"),
]
REQ = ["Python", "SQL", "Tableau"]
DUTY = ["Airflow", "Snowflake", "dbt", "Spark", "Looker"]
PREF = ["AWS", "R"]
MS = ([{"name": x, "where": "required"} for x in REQ] + [{"name": x, "where": "duty"} for x in DUTY]
      + [{"name": x, "where": "preferred"} for x in PREF]
      + [{"name": "Power BI", "where": "list", "list": 1, "pos": 0}, {"name": "Looker Studio", "where": "list", "list": 1,
                                                                       "pos": 1}])


def _cands():
    out = []
    for i, (title, field) in enumerate(POSTINGS):
        p = {"key": f"t:{i}", "title": title, "field": field, "text": "Requirements: 3+ years of experience."}
        for slot in range(24):
            arch = ("strong", "near", "gap", "off")[slot % 4]
            out.append((p, arch, b.build_candidate(p, None, slot, arch, slot % 2 == 0, MS)))
    return out


def test_no_job_title_is_the_posting_title():
    for p, _, c in _cands():
        banned = b.posting_titles(p)
        for j in c["writer"]["jobs"]:
            assert j["title"].casefold().strip() not in banned, (p["title"], j["title"])


def test_skills_are_not_the_posting_keyword_list():
    names = [m["name"] for m in MS]
    led, n = 0, 0
    for p, arch, c in _cands():
        sk = c["candidate_card"]["skills"]
        if arch == "strong":
            assert set(REQ) <= set(sk)  # still meets every required item
        assert sum(x in DUTY for x in sk) <= 1  # duties are not dumped in as skills
        posting_order = [x for x in names if x in sk]
        if len(posting_order) >= 3:
            n += 1
            led += sk[:len(posting_order)] == posting_order  # the list opens with the posting's keywords, in order
        if arch != "off":
            assert sum(x not in names for x in sk) >= 3  # the person's own field skills are on it too
    assert n and led / n < 0.1


def test_strong_order_varies():
    orders = {tuple(c["candidate_card"]["skills"][:3]) for _, arch, c in _cands() if arch == "strong"}
    assert len(orders) > 5
