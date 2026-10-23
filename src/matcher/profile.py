"""The candidate profile: the short form the user fills in next to their resume.

Rendered in plain English from a CandidateCard by code (never by Adaption). It holds
every candidate fact the label rule reads that a resume leaves out on purpose (work
authorization, locations and job preferences), plus the education and experience facts
the rule reads, stated plainly: highest degree and field (or the degree being pursued and
when it is expected), and years of work experience. A real user fills these in on the form. It never shows name, education_country
or language; those are fairness nuisance fields the rule ignores.
"""

from __future__ import annotations

from .schema import CURRENT_YEAR, CandidateCard

WORK_AUTH_TEXT = {
    "citizen_or_pr": "U.S. citizen or permanent resident, no sponsorship needed",
    "f1_cpt": "F-1 student; can work on CPT (Curricular Practical Training), which covers internships and co-ops",
    "f1_opt": "F-1 graduate on OPT (Optional Practical Training)",
    "stem_opt": "F-1 graduate on the STEM OPT extension",
    "needs_h1b_now": "needs H-1B visa sponsorship to start",
    "other_visa": "another visa; whether it allows a given job has to be checked",
}
REMOTE_TEXT = {
    "any": "onsite, hybrid or remote",
    "hybrid_or_remote": "hybrid or remote only",
    "remote_only": "remote only",
}
TYPE_TEXT = {"internship": "internship", "full_time": "full-time", "part_time": "part-time", "contract": "contract"}


def _list(items: tuple[str, ...], empty: str, sep: str = ", ") -> str:
    return sep.join(items) if items else empty


DEGREE_TEXT = {"associate": "Associate degree", "bachelor": "Bachelor's degree", "master": "Master's degree",
               "doctoral": "Doctorate"}


def degree_line(c: CandidateCard) -> str:
    """"Bachelor's degree in Biology, graduated 2021" / "..., expected 2027" / "no college degree"."""
    if c.degree_level == "none":
        return "no college degree"
    text = DEGREE_TEXT[c.degree_level] + (f" in {c.degree_field}" if c.degree_field else "")
    gy = c.graduation_year
    if gy is None:
        return text
    return f"{text}, {'expected' if gy > CURRENT_YEAR else 'graduated'} {gy}"


def degree_label(c: CandidateCard) -> str:
    """"Currently pursuing" for a degree still in progress (it is not yet held), else "Highest degree"."""
    gy = c.graduation_year
    return "Currently pursuing" if c.degree_level != "none" and gy is not None and gy > CURRENT_YEAR \
        else "Highest degree"


def render_profile(c: CandidateCard) -> str:
    """One "Field: value" line per form field; unset fields say so instead of vanishing."""
    floor = f"${c.salary_floor:,} a year" if c.salary_floor is not None else "not set"
    lines = (
        f"{degree_label(c)}: {degree_line(c)}",
        f"Years of work experience: {c.years}",
        f"Work authorization: {WORK_AUTH_TEXT[c.work_auth]}",
        f"Lives in: {c.current_location or 'not given'}",
        f"Willing to relocate to: {_list(c.willing_locations, 'nowhere else', '; ')}",
        f"Work arrangement: {REMOTE_TEXT[c.remote_preference]}",
        f"Job types wanted: {_list(tuple(TYPE_TEXT[t] for t in c.employment_types_wanted), 'any')}",
        f"Minimum salary: {floor}",
        f"Industries wanted: {_list(c.industries_wanted, 'any')}",
        f"Industries to avoid: {_list(c.industries_avoided, 'none')}",
        f"Roles to avoid: {_list(c.roles_avoided, 'none')}",
    )
    return "\n".join(lines)
