"""The shipped readout of run B's three label scores: the decider (eval/protocol_amendment_5b.json), standard library only
so the demo server and the scorers import it without numpy.

    s = P(Keep) + 0.5 P(Explore), P = softmax at temperature 1 over the three float32 first-token label scores
    s >= T_HI -> Keep, s < T_LO -> Skip, else Explore

T_LO and T_HI are the pick's cut-offs in eval/selection_cycle3.json (pick.cut_offs, fitted on fresh tests 1-3 pooled),
exact. The softmax is evalstats.softmax's arithmetic (float64, subtract the max, exp, divide by the sum, the sum taken
left to right), so s is the eval's s (fit_select_cycle3.decider_score).
"""
from __future__ import annotations

import math

LABELS = ("Keep", "Explore", "Skip")  # evalstats.LABELS
T_LO = 0.2731045004103801
T_HI = 0.5890379172624224


def softmax(scores) -> list[float]:
    """evalstats.softmax(scores, 1.0) on one row of three scores."""
    z = [float(x) for x in scores]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    total = e[0] + e[1] + e[2]
    return [x / total for x in e]


def keep_score(scores) -> float:
    """s = P(Keep) + 0.5 P(Explore) from the scores in Keep, Explore, Skip order."""
    p = softmax(scores)
    return p[0] + 0.5 * p[1]


def decide(scores, t_lo: float = T_LO, t_hi: float = T_HI) -> str:
    """The decider's label from the scores in Keep, Explore, Skip order (before the eligibility rules)."""
    s = keep_score(scores)
    return "Keep" if s >= t_hi else "Skip" if s < t_lo else "Explore"


def decide_dict(scores: dict[str, float]) -> str:
    """decide() on a {Keep, Explore, Skip} score dict, as the scorers return it (pass the unrounded float32 scores)."""
    return decide([scores[w] for w in LABELS])
