"""Plain texts that state every card fact, in wording the visibility probes read.

A rendering this literal should leave `visibility.missing_facts` nothing to report, so any
finding on it is a checker bug (or a crash), not a rendering gap.
"""

from __future__ import annotations

from matcher.schema import CURRENT_YEAR, CandidateCard, RequirementCard

_LEVEL_WORD = {"associate": "associate", "bachelor": "bachelor's", "master": "master's", "doctoral": "doctoral"}
_TYPE = {"internship": "an internship", "full_time": "a full-time role", "part_time": "a part-time role",
         "contract": "a contract role"}
_STANCE = {
    "sponsors": "This employer will sponsor work visas.",
    "will_not_sponsor": "We are unable to sponsor work visas.",
    "will_not_sponsor_or_opt": "We are unable to sponsor work visas and cannot accept OPT or STEM OPT.",
    "unstated": "",
}


def posting_text(r: RequirementCard) -> str:
    lines = [r.title, f"This is {_TYPE[r.employment_type]}, {r.work_mode}."]
    if r.industry:
        lines.append(f"Industry: {r.industry}.")
    if r.locations:
        lines.append("Location: " + "; ".join(r.locations) + ".")
    if r.enrolled_in != "none":
        lines.append(f"Open to students currently enrolled in a {_LEVEL_WORD[r.enrolled_in]} program.")
    if r.degree_level != "none":
        lines.append(f"Requires a {_LEVEL_WORD[r.degree_level]} degree.")
    if r.degree_fields:
        lines.append("Degree fields: " + ", ".join(r.degree_fields) + ".")
    if r.min_years:
        lines.append(f"{r.min_years}+ years of experience.")
    if r.must_skills:
        lines.append("Required skills: " + "; ".join(" or ".join(g) for g in r.must_skills) + ".")
    if r.nice_skills:
        lines.append("Nice to have: " + ", ".join(r.nice_skills) + ".")
    if r.license:
        lines.append(f"Must hold an active {r.license}.")
    if r.salary_max is not None:
        lo = r.salary_min if r.salary_min is not None else r.salary_max
        lines.append(f"Pay: ${lo:,} - ${r.salary_max:,} per year.")
    if r.citizens_only:
        lines.append("Must be a U.S. citizen or permanent resident.")
    else:
        lines.append(_STANCE[r.sponsorship])
    if r.e_verify:
        lines.append("We participate in E-Verify.")
    return "\n".join(x for x in lines if x)


def resume_text(c: CandidateCard) -> str:
    lines = ["Candidate", c.current_location]
    if c.degree_level != "none":
        gy = c.graduation_year
        when = "" if gy is None else f", expected {gy}" if gy > CURRENT_YEAR else f", {gy}"
        lines.append(f"{_LEVEL_WORD[c.degree_level].capitalize()} degree in {c.degree_field}{when}")
    if c.years:
        lines.append(f"{c.years} years of experience.")
    if c.skills:
        lines.append("Skills: " + ", ".join(c.skills))
    if c.licenses:
        lines.append("Licenses: " + ", ".join(c.licenses))
    return "\n".join(x for x in lines if x)
