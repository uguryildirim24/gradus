"""Near copies of the real test postings stay out of training."""
from matcher.blueprint import training_ok
from matcher.nearcopy import NearIndex

BODY = " ".join(f"Line {i} of the internship posting says what the analyst does on the team." for i in range(40))
TEST = {"ats_t": {"company": "Booz Allen Hamilton Inc.", "title": "University - Data Analyst Intern", "description": BODY}}


def test_same_company_and_title_is_a_near_copy():
    idx = NearIndex(TEST)
    post = {"company": "Booz Allen Hamilton", "title": "University – Data Analyst Intern", "description": "Other text."}
    assert idx.near(post) == ("ats_t", "same company and title")


def test_a_description_eighty_percent_alike_is_a_near_copy():
    idx = NearIndex(TEST)
    # another location and pay line (Booz Allen e36f52f0 has nine of these)
    post = {"company": "Booz Allen", "title": "Data Analyst Intern, Annapolis Junction", "description": BODY + " Pay: $52,900."}
    assert idx.near(post)[0] == "ats_t"
    half = BODY[: len(BODY) // 2] + " ".join(f"Other sentence {i} about something else entirely here." for i in range(20))
    assert idx.near({"company": "Booz Allen", "title": "Data Engineer Intern", "description": half}) is None


def test_a_description_shared_but_for_a_few_lines_is_a_near_copy():
    """Same company, another title, the test posting's text but for its first lines (Booz Allen's "Data Scientist
    Intern" rows were 0.86-0.88 alike to its test posting and got through at 0.9)."""
    idx = NearIndex(TEST)
    def edited(k):
        return " ".join((f"Line {i} of the internship posting says what the analyst does on the team." if i >= k else
                         f"Data scientist sentence {i} is about modelling work for a different group entirely.")
                        for i in range(40))
    hit = idx.near({"company": "Booz Allen", "title": "Data Scientist Intern", "description": edited(2)})  # 0.87 alike
    assert hit and hit[0] == "ats_t" and hit[1].startswith("description 0.8")
    assert idx.near({"company": "Booz Allen", "title": "Data Scientist Intern", "description": edited(5)}) is None  # 0.75


def test_near_names_the_most_alike_test_posting_whatever_the_order():
    """Two Booz Allen test postings share most of their text; a row gets the one it is closer to, not the first one
    a set happened to list (merge built the index from a set, so replays named different test postings)."""
    lines = [f"Line {i} of the internship posting says what the analyst does on the team." for i in range(40)]
    a = " ".join(lines)
    b = " ".join(lines[:37] + [f"Other closing line {i} about benefits and pay ranges here." for i in range(3)])
    tests = {"ats_a": {"company": "Booz Allen", "title": "Systems Engineer Intern", "description": a},
             "ats_b": {"company": "Booz Allen", "title": "Software Developer Intern", "description": b}}
    post = {"company": "Booz Allen", "title": "Data Scientist Intern", "description": b + " Pay: $52,900."}
    for order in (["ats_a", "ats_b"], ["ats_b", "ats_a"]):
        assert NearIndex({i: tests[i] for i in order}).near(post)[0] == "ats_b"


def test_training_ok_leaves_out_a_row_marked_in_real_test():
    assert not training_ok({"us": True, "language": "en", "in_real_test": True, "near_test": ["ats_t", "x"]})
