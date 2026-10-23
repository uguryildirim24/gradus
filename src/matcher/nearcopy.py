"""Near copies of the real-posting test set (the 87 hand-checked postings): nothing that trains may hold one.

A posting is a near copy of a test posting when it has the same company and title (after norm_text, with
company suffixes such as "Inc" or "Corporation" dropped), or when at least NEAR of the two descriptions'
N-word windows are shared (Jaccard). The board lists one job under several ids with another location or a
pay line (Booz Allen, Northrop Grumman, GE Aerospace); an exact-text check misses those. At 0.9, 14 training rows
0.86-0.89 alike to a test posting got through (GE Aerospace, HPE, Northrop Grumman, 11 Booz Allen "Data Scientist
Intern" rows sharing the Booz Allen test posting's text), so NEAR is 0.8: losing a few training rows costs less than
test text in training.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

NEAR = 0.8
N = 6
_SUFFIX = re.compile(r"\b(?:inc|incorporated|corp|corporation|co|company|llc|ltd|limited|plc|holdings)\b")


def norm_text(s: str) -> str:
    """Casefolded, every run of non-alphanumeric characters one space."""
    return " ".join(re.sub(r"[\W_]+", " ", s.casefold()).split())


def company_key(s: str) -> str:
    return " ".join(_SUFFIX.sub(" ", norm_text(s or "")).split())


def shingles(text: str, n: int = N) -> frozenset[str]:
    w = norm_text(text).split()
    return frozenset(" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 1)))


class NearIndex:
    """The test postings, ready for `near`."""

    def __init__(self, tests: dict[str, dict]):  # id -> raw posting with title, company, description
        ids = sorted(tests)  # the same answer whatever order the caller's ids come in
        self.keys: dict[tuple[str, str], str] = {}
        for i in ids:
            self.keys.setdefault((company_key(tests[i].get("company", "")), norm_text(tests[i].get("title", ""))), i)
        self.sh = {i: shingles(tests[i].get("description", "")) for i in ids}

    def near(self, post: dict) -> tuple[str, str] | None:
        """(test id, why) when `post` is a near copy of a test posting, else None: the same company and title,
        else the test posting whose description is most alike (the first id of a tie), when that is NEAR or more."""
        k = (company_key(post.get("company", "")), norm_text(post.get("title", "")))
        if k in self.keys:
            return self.keys[k], "same company and title"
        s = shingles(post.get("description", ""))
        best: tuple[float, str] | None = None
        for i, t in self.sh.items():
            if not s or not t or min(len(s), len(t)) < NEAR * max(len(s), len(t)):
                continue  # the Jaccard index cannot reach NEAR
            j = len(s & t) / len(s | t)
            if j >= NEAR and (best is None or j > best[0]):
                best = (j, i)
        return (best[1], f"description {best[0]:.2f} alike") if best else None


def load_index(root: Path) -> tuple[NearIndex, dict[str, dict]]:
    """(index of the test postings, every raw posting by id) from data/built/audit/test_cards.jsonl and
    data/raw/ats/postings.jsonl."""
    posts = {}
    with open(root / "data/raw/ats/postings.jsonl") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                posts["ats_" + d["simplify_id"]] = d
    with open(root / "data/built/audit/test_cards.jsonl") as f:
        test = [json.loads(line)["id"] for line in f if line.strip()]
    return NearIndex({i: posts[i] for i in test if i in posts}), posts
