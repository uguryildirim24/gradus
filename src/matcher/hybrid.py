"""The rules-override hybrid H and H' (one implementation, numpy-free so the demo server can import it).

H (pre-registered 2026-10-20): when the rules on Adaption's extracted card say Skip for a hard-filter reason
(work authorization, enrollment status or level, degree level), output Skip; otherwise the model's label.
H' (the shipped system of the v4 selection, eval/selection_v4.json) also overrides on degree field, employment type, location and work mode
Skips, and caps a model Keep at Explore when the rules flag pay under the candidate's salary floor. Work mode (fix
cycle 2, eval/fix_cycle2.json): a work mode the profile rules out is a hard rules Skip (rules._check_location) except
on a thin posting; its reasons come from the location check and are split off as their own category.

H'' without work mode (cycle 3, eval/selection_cycle3.json, the shipped system with run B): H' without the employment
type and work mode overrides (degree field, degree level, enrollment, location, work authorization, plus the salary cap);
scripts/fit_select_cycle3.py VARIANTS["H''_without_work_mode"].

scripts/hybrid_rules.py scores H and H' from stored eval outputs and re-exports everything here; demo/demo_lib.py
applies H'' to the live page.
"""
from __future__ import annotations

from .rules import LABEL_WORDS, WORK_MODE_PREFIX, label, reasons_by_check
from .schema import CandidateCard, RequirementCard

# Reason categories: the rules' check names (rules.CHECKS), with the degree check split into level and field and the
# location check into place and work mode.
H_CATS = frozenset({"work_auth", "enrollment", "degree_level"})
HP_CATS = H_CATS | {"degree_field", "employment_type", "location", "work_mode"}
SALARY_PREFIX = "pay tops out at"
HPP_CATS = HP_CATS - {"employment_type", "work_mode"}  # H'' without work mode
VARIANTS = {"H": (H_CATS, False), "H_prime": (HP_CATS, True)}


def category(check: str | None, reason: str) -> str:
    """The category of one reason, from the check that wrote it; the degree and location checks are split by their
    wording."""
    if check == "degree":
        return "degree_field" if reason.startswith("degree field") else "degree_level"
    if check == "location" and reason.startswith(WORK_MODE_PREFIX):
        return "work_mode"
    return check or "unknown"


def rule_verdict(cand: CandidateCard, card: RequirementCard) -> dict:
    """rules.label on this card: the label word, the categories of its hard reasons (empty unless Skip), whether
    a salary-floor reason is among its soft ones, its reasons, and each hard reason's category."""
    bucket, reasons, _ = label(cand, card)
    by_check = reasons_by_check(cand, card)
    reason_cats = {x: category(by_check.get(x), x) for x in reasons} if bucket == "Excluded" else {}
    salary = bucket == "Explore" and any(x.startswith(SALARY_PREFIX) for x in reasons)
    return {"rule": LABEL_WORDS[bucket], "hard_cats": sorted(set(reason_cats.values())), "salary_soft": salary,
            "reasons": reasons, "reason_cats": reason_cats}


def hybrid(model: str, verdict: dict, cats: frozenset = H_CATS, salary_cap: bool = False) -> str:
    """Skip when the rules Skip for a reason in `cats`; with salary_cap, a model Keep becomes Explore when the rules
    flag pay under the floor; otherwise the model's label."""
    if verdict["rule"] == "Skip" and set(verdict["hard_cats"]) & cats:
        return "Skip"
    if salary_cap and verdict["salary_soft"] and model == "Keep":
        return "Explore"
    return model


def overrides(model: str, verdict: dict, cats: frozenset, salary_cap: bool) -> bool:
    """Whether the override fires on this row (whatever the model said)."""
    return (verdict["rule"] == "Skip" and bool(set(verdict["hard_cats"]) & cats)) or \
        (salary_cap and verdict["salary_soft"] and model == "Keep")
