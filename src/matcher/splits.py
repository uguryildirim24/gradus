"""Leak-free splits: no requirement card and no candidate lands in two splits.

Whole occupation families are set aside for the held-out-family test first. The
rest is split 80/10/10 by occupation (SOC code), not by card, so a dev or test
occupation never appears in train (pairs.assign_soc_splits). Candidates follow the
split of the occupation they were built around; candidates without one are split
by id into pools, plus a small pool reserved for the held-out-family test.
"""

from __future__ import annotations

import hashlib
import random
import re

SPLITS = ("train", "dev", "test")
HELDOUT = "test_heldout_family"
RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}
CAND_RATIOS = {"train": 0.76, "dev": 0.09, "test": 0.09, HELDOUT: 0.06}


def family_of(field: str, onet_soc: str | None) -> str:
    """Occupation family: O*NET major group (first two SOC digits), else the field's first word."""
    if onet_soc:
        m = re.match(r"\d{2}", onet_soc.strip())
        if m:
            return f"soc{m.group(0)}"
    words = re.findall(r"[a-z]+", (field or "").lower())
    return f"f_{words[0]}" if words else "f_unknown"


# Computer and mathematical (SOC 15), engineering (17), business and finance (13) and life, physical and
# social science (19) stay in training: the real-posting test set is tech, hardware, quant, product and
# research internships, and the demo persona is a CS student, so holding one of them out
# would mix a family shift into the synthetic-to-real number.
NEVER_HOLD_OUT = frozenset({"soc15", "soc17", "soc13", "soc19"})


def pick_heldout_families(families: dict[str, int], n: int = 2, seed: int = 0) -> list[str]:
    """Choose n families, preferring ones big enough to test on but not most of the data."""
    total = sum(families.values()) or 1
    families = {f: c for f, c in families.items() if f not in NEVER_HOLD_OUT}
    ok = sorted(f for f, c in families.items() if 0.03 * total <= c <= 0.25 * total)
    pool = ok if len(ok) >= n else sorted(families)
    return sorted(random.Random(f"heldout:{seed}").sample(pool, min(n, len(pool))))


def assign(ids: list[str], ratios: dict[str, float], seed: int = 0) -> dict[str, str]:
    """Deterministic shuffle-and-cut assignment of ids to split names."""
    order = sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    out: dict[str, str] = {}
    names = list(ratios)
    n = len(order)
    cut = 0.0
    start = 0
    for k, name in enumerate(names):
        cut += ratios[name]
        end = n if k == len(names) - 1 else round(n * cut)
        for i in order[start:end]:
            out[i] = name
        start = end
    return out
