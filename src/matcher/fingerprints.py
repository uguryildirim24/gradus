"""Surface features of a training pair that must not give the label away.

Most features look at one side only (the candidate, their profile, or the posting) or at
how the pair was built. If one value of such a feature predicts one bucket in most rows, a
model can learn that value instead of reading the pair: "a blank profile means Skip".
build_rows steers its picks against this, and build_pairs fails the build when any value
predicts a bucket more than MAX_SHARE of the time.

FIT_FEATURES are different: years against the posting's minimum, the overlap of the
candidate's skills with the card, and skill count by years (which tracks years). The rule's
own checks drive them, so a skew there is the rule working ("3 years short" is never Keep).
Steering against them starved the years-short Skips (45 rows of 10,500), so they are tallied
and reported but neither steered nor gated.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .profile import TYPE_TEXT
from .rules import is_enrolled
from .schema import CandidateCard, RequirementCard, skill_canon

MAX_SHARE = 0.80  # no feature value may put more than this share of its rows in one bucket
MIN_ROWS = 50  # values seen fewer times are too rare to judge
FIT_FEATURES = frozenset({"years_minus_min", "skill_overlap", "skill_count_x_years"})  # reported only


def _bin(x: float, edges: tuple[float, ...], names: tuple[str, ...]) -> str:
    for e, n in zip(edges, names):
        if x <= e:
            return n
    return names[-1]


# Statuses the rule checks the same on every posting stance: a citizen passes everywhere, CPT
# turns on the job type, another visa is always "check". Their bucket mix should not move with
# the stance (only the citizens-only flag decides for the two visas).
STANCE_BLIND = {"citizen_or_pr": "citizen", "f1_cpt": "F-1 CPT", "other_visa": "other visa"}
STANCE_DECIDES = "stance decides"


def stance_cell(c: CandidateCard, r: RequirementCard) -> str:
    """"citizen / sponsors": a stance-blind status by the posting's stance, or STANCE_DECIDES where
    the rule reads the stance (OPT, STEM OPT, H-1B; a visa on a citizens-only posting).
    STANCE_DECIDES is steered like any value: it is mostly the visa holders' bucket mix, and leaving
    it unsteered lifted the candidate-field leakage probe over its margin (+0.105 accuracy, +0.117
    macro-F1 on seed 0); the coverage floors keep the work-authorization Skips it could starve."""
    group = STANCE_BLIND.get(c.work_auth)
    if group is None or (r.citizens_only and c.work_auth != "citizen_or_pr"):
        return STANCE_DECIDES
    return f"{group} / {'citizens only' if r.citizens_only else r.sponsorship}"


def features(c: CandidateCard, r: RequirementCard, candidate_kind: str, card_edited: bool) -> dict[str, str]:
    on_card = {skill_canon(x) for g in r.must_skills for x in g} | {skill_canon(x) for x in r.nice_skills}
    n = len(c.skills)
    overlap = sum(skill_canon(s) in on_card for s in c.skills) / n if n else 0.0
    gap = c.years - r.min_years
    floor = c.salary_floor
    prefs = (floor is not None, bool(c.employment_types_wanted), bool(c.industries_wanted), bool(c.industries_avoided),
             bool(c.roles_avoided), c.remote_preference != "any", bool(c.willing_locations))
    willing = "anywhere" if any("anywhere" in x for x in c.willing_locations) else str(len(c.willing_locations))
    types = ", ".join(TYPE_TEXT[t] for t in c.employment_types_wanted) or "any"
    return {
        # joint values: a boosted model reads preferences together, so they are steered together too
        "preferences_set": str(sum(prefs)),
        "preference_pattern": "".join("x" if x else "." for x in prefs),
        "work_auth_x_enrolled": f"{c.work_auth} / {'enrolled' if is_enrolled(c) else 'not enrolled'}",
        # the four profile fields the one-edit breaks set by design, read together; with the two below they
        # brought the candidate-field probe from +0.102 macro-F1 on seed 0 (over the 0.10 margin) to +0.081
        "salary_x_industries_x_remote_x_auth": f"{'floor' if floor is not None else 'no floor'} / "
                                               f"{len(c.industries_wanted)} wanted / {c.remote_preference} / {c.work_auth}",
        "relocation_x_enrolled_x_level": f"{willing} / {'enrolled' if is_enrolled(c) else 'not enrolled'} / {c.degree_level}",
        "industries_x_years": f"{len(c.industries_wanted)} wanted / {len(c.industries_avoided)} avoided / "
                              f"{_bin(c.years, (0, 2, 5), ('0', '1-2', '3-5', '6+'))}",
        "work_auth_x_stance": stance_cell(c, r),
        "remote_x_job_types": f"{c.remote_preference} / {types}",
        "relocation_x_remote": f"{willing} / {c.remote_preference}",
        "work_auth_x_job_types": f"{c.work_auth} / {types}",
        "enrolled_x_years": f"{'enrolled' if is_enrolled(c) else 'not enrolled'} / "
                            f"{_bin(c.years, (0, 1, 3, 6), ('0', '1', '2-3', '4-6', '7+'))}",
        "skill_count_x_years": f"{_bin(n, (5, 9, 13), ('0-5', '6-9', '10-13', '14+'))} / "
                               f"{_bin(c.years, (0, 2, 5), ('0', '1-2', '3-5', '6+'))}",
        "candidate_kind": candidate_kind,
        "skill_count": _bin(n, (3, 6, 9, 12), ("0-3", "4-6", "7-9", "10-12", "13+")),
        "skill_overlap": _bin(overlap, (0.0, 0.25, 0.5, 0.75, 0.99), ("0", "<=25%", "<=50%", "<=75%", "<100%", "100%")),
        "job_types_wanted": ", ".join(TYPE_TEXT[t] for t in c.employment_types_wanted) or "any",
        "willing_locations": "empty" if not c.willing_locations else "anywhere" if any(
            "anywhere" in x for x in c.willing_locations) else "some",
        "work_mode": r.work_mode,
        "remote_preference": c.remote_preference,
        "work_auth": c.work_auth,
        "salary_floor": "unset" if floor is None else "= salary_min" if floor == r.salary_min else "set",
        "years_minus_min": _bin(gap, (-1, 0, 1, 2, 5), ("<0", "0", "1", "2", "3-5", "6+")),
        "current_location": c.current_location or "(none)",
        "degree_level": c.degree_level,
        "enrolled": "enrolled" if is_enrolled(c) else "graduated" if c.graduation_year else "no year",
        "graduation_year": str(c.graduation_year),
        "industries_wanted": "any" if not c.industries_wanted else str(len(c.industries_wanted)),
        "industries_avoided": ", ".join(c.industries_avoided) or "none",
        "roles_avoided": ", ".join(c.roles_avoided) or "none",
        "salary_floor_level": "unset" if floor is None else _bin(floor, (40_000, 70_000, 100_000),
                                                                  ("<=40k", "<=70k", "<=100k", ">100k")),
        "licenses": str(len(c.licenses)),
        "education_country": "US" if c.education_country == "United States" else (c.education_country and "other") or "none",
        "posting_employment_type": r.employment_type,
        "posting_citizens_only": str(r.citizens_only),
        "posting_sponsorship": r.sponsorship,
        "posting_edited": str(card_edited),
        "posting_enrolled_in": r.enrolled_in,
        "posting_must_groups": str(min(len(r.must_skills), 6)),
    }


class Tally:
    """Bucket counts per (feature, value)."""

    def __init__(self) -> None:
        self.n: dict[tuple[str, str], Counter] = defaultdict(Counter)

    def add(self, feats: dict[str, str], bucket: str) -> None:
        for k, v in feats.items():
            self.n[(k, v)][bucket] += 1

    def share_after(self, feats: dict[str, str], bucket: str) -> float:
        """Sum over steered features (not FIT_FEATURES) of the squared share this bucket would have
        after one more row."""
        total = 0.0
        for k, v in feats.items():
            if k in FIT_FEATURES:
                continue
            c = self.n.get((k, v))
            tot = sum(c.values()) if c else 0
            s = ((c[bucket] if c else 0) + 1) / (tot + 3)
            total += s * s
        return total

    def table(self, min_rows: int = MIN_ROWS) -> list[dict]:
        """One row per (feature, value) with n >= min_rows: its top bucket and that bucket's share."""
        out = []
        for (k, v), c in sorted(self.n.items()):
            tot = sum(c.values())
            if tot < min_rows:
                continue
            b, m = c.most_common(1)[0]
            out.append({"feature": k, "value": v, "n": tot, "top_bucket": b, "top_share": m / tot,
                        "counts": dict(c)})
        return out


def tally_rows(rows: list[dict], cands: dict[str, CandidateCard], cards: dict[str, RequirementCard]) -> Tally:
    t = Tally()
    for r in rows:
        t.add(features(cands[r["cd_id"]], cards[r["rq_id"]], r["candidate_kind"], "~" in r["rq_id"]), r["bucket"])
    return t


def worst(t: Tally, min_rows: int = MIN_ROWS) -> list[dict]:
    """Gated values over MAX_SHARE (FIT_FEATURES are left out)."""
    return sorted((x for x in t.table(min_rows) if x["top_share"] > MAX_SHARE and x["feature"] not in FIT_FEATURES),
                  key=lambda x: -x["top_share"])
