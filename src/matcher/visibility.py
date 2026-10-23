"""Is every fact behind a label's reasons visible in the prompt?

The rule reads cards; the model reads text (resume, candidate profile, posting). For
each reason in a row's label, `missing_facts` names the check that wrote it
(rules.reasons_by_check) and looks for that check's facts in the text. A fact can be
absent (the posting never states the industry) or contradicted (the resume names a
skill the rule says the candidate lacks).

Categories: `skill_wording` is a skill name the posting or resume words differently;
everything else is a fact the model cannot see.
"""

from __future__ import annotations

import re

from .degrees import card_fields, field_named
from .profile import REMOTE_TEXT, TYPE_TEXT, WORK_AUTH_TEXT, degree_line
from .rules import ALL_SKILLS_MET, CURRENT_YEAR, counted_nice, credential, degree_held, held_skills, \
    reasons_by_check, required_groups
from .schema import SKILL_SYNONYMS, CandidateCard, RequirementCard, group_met, norm, skill_canon, skill_key, \
    skill_meets

# ---------------------------------------------------------------- stance wording

NO_SPONSOR = re.compile(r"(unable to|cannot|can't|can not|not able to|do(es)? not|will not|won't|don't|not allow us to)\s+"
                        r"(currently\s+|presently\s+|be able to\s+)?(sponsor\w*|offer\b[^.]{0,30}sponsorship|provide\b[^.]{0,30}sponsorship)", re.I)
NO_OPT = re.compile(r"(unable to|cannot|can't|not able to|do(es)? not|will not|nor can we)\s+[^.]{0,60}\bOPT\b", re.I)
PERMANENT = re.compile(r"unrestricted|permanent(ly)? (work |employment )?authori|authori[sz]ed to work .{0,40}permanent", re.I)
CITIZENS = re.compile(r"(?<!not )must be (a )?U\.?S\.? citizen|(open|available|limited) (only )?to U\.?S\.? citizens", re.I)


def stance_conflict(card: dict, text: str) -> str | None:
    """Posting text that states a work-authorization stance the card does not have."""
    s = card.get("sponsorship")
    # citizens-only already means no sponsorship (rules._check_work_auth stops at citizens_only before it reads the
    # stance): a no-sponsorship sentence on a citizens-only card with no stance of its own states nothing new
    # (rq_01334~citizens, 'We do not sponsor visas for this role')
    if NO_SPONSOR.search(text) and (s == "sponsors" or s == "unstated" and not card.get("citizens_only")):
        return f"{s} card, text refuses sponsorship"
    if s != "will_not_sponsor_or_opt" and NO_OPT.search(text):
        return f"{s} card, text refuses OPT"
    if not card.get("citizens_only") and CITIZENS.search(text):
        return "citizens_only false, text requires citizenship"
    return None


# ---------------------------------------------------------------- text probes

_NUM_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
              "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
              "nineteen", "twenty")
_LEVEL = {
    "associate": r"associate",
    "bachelor": r"bachelor|\bB\.?S\.?\b|\bB\.?A\.?\b|\bBSc\b|\bBSN\b",
    "master": r"master|\bM\.?S\.?\b|\bM\.?A\.?\b|\bMSc\b|\bMBA\b|\bMSN\b|\bM\.?Ed\b",
    "doctoral": r"doctor|ph\.?\s?d|\bD\.?N\.?P\b|\bPharm\.?\s?D\b|\bM\.?D\.?\b|\bJ\.?D\.?\b|\bEd\.?D\b",
}
_TYPE = {"internship": r"\bintern(ship)?s?\b|co-?op", "full_time": r"full[- ]?time", "part_time": r"part[- ]?time",
         "contract": r"\bcontract"}
_MODE = {"onsite": r"on[- ]?site|in[- ]person|in the office|in-office|on location", "hybrid": r"hybrid",
         "remote": r"remote"}


def _flat(text: str) -> str:
    """Casefold, punctuation and hyphens as spaces (keeping C++, C#, .NET), one space."""
    text = re.sub(r"\.(?=\s|$)", " ", text)
    return " " + skill_key(re.sub(r"[^\w+#.]+", " ", text)) + " "


def has_phrase(text: str, phrase: str) -> bool:
    """Whole phrase, case-insensitive, tolerant of hyphens, punctuation and inserted
    parentheticals ("CNA (Certified Nursing Assistant) certification")."""
    p = _flat(phrase)
    return p in _flat(text) or p in _flat(re.sub(r"\s*\([^)]*\)", "", text))


def industry_forms(industry: str) -> set[str]:
    """The industry label and the forms a posting writes it in: '&' as 'and' (and back), and the last word
    singular ('Hospital & Health Care': 'hospital and health care'; 'Semiconductors': 'a semiconductor company';
    'utilities': 'a utility company')."""
    forms = {industry}
    if "&" in industry:
        forms.add(industry.replace("&", "and"))
    if re.search(r"\sand\s", industry):
        forms.add(re.sub(r"\sand\s", " & ", industry))
    for f in list(forms):
        if re.search(r"ies$", f, re.I):
            forms.add(f[:-3] + "y")
        elif re.search(r"[^s]s$", f, re.I):
            forms.add(f[:-1])
    return forms


def has_industry(text: str, industry: str) -> bool:
    """The posting names its industry (the render gate and the label-side check use this one probe)."""
    return any(has_phrase(text, f) for f in industry_forms(industry))


def has_number(text: str, n: int) -> bool:
    return re.search(rf"(?<![\d,.$]){n}(?![\d,])", text) is not None


def has_years(text: str, n: int) -> bool:
    """"3 years", "3+ years", "three (3) years", "3-year"."""
    pats = [rf"(?<![\d.]){n}\s*\+?\)?\s*-?\s*(years?|yrs?)\b"]
    if n < len(_NUM_WORDS):
        pats.append(rf"\b{_NUM_WORDS[n]}\b\s*(\(\d+\)\s*)?\+?\s*-?\s*(years?|yrs?)\b")
    return any(re.search(p, text, re.I) for p in pats)


def has_money(text: str, v: int) -> bool:
    k = v / 1000
    ks = f"{k:g}"
    pats = [rf"{v:,}", rf"(?<![\d,]){v}(?!\d)", rf"\${ks}\s?[kK]\b", rf"\b{ks}\s?[kK]\b"]
    return any(re.search(p, text) for p in pats)


def _city(place: str) -> str:
    return place.split(",")[0].strip()


def _re(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.I) is not None


# ---------------------------------------------------------------- per-check facts


def _work_auth(c: CandidateCard, r: RequirementCard, card_dict: dict, reason: str, rs: str, pf: str, ps: str):
    if WORK_AUTH_TEXT[c.work_auth] not in pf:
        yield "work_auth", "profile lacks work authorization"
    if r.citizens_only:
        if not _re(ps, r"citizen|permanent residen|green card"):
            yield "work_auth", "posting does not say citizens or permanent residents only"
        return
    if c.work_auth != "needs_h1b_now" and PERMANENT.search(ps):
        # the extraction audit reads "permanent" authorization as citizens or permanent residents only
        yield "work_auth", "posting asks for unrestricted or permanent work authorization; the card is not citizens-only"
    s = r.sponsorship
    if s in ("sponsors", "will_not_sponsor", "will_not_sponsor_or_opt") and not _re(ps, r"sponsor"):
        yield "work_auth", f"posting does not state its sponsorship stance ({s})"
    if s == "will_not_sponsor_or_opt" and not _re(ps, r"\bOPT\b"):
        yield "work_auth", "posting does not say it refuses OPT"
    if "E-Verify" in reason and not _re(ps, r"e-?verify"):
        yield "work_auth", "posting does not mention E-Verify"
    if "CPT" in reason and not _re(ps, _TYPE[r.employment_type]):
        yield "work_auth", f"posting does not state the job type ({r.employment_type})"
    if s == "unstated" and _re(ps, r"\bwill sponsor|\bwe sponsor|offers? (visa )?sponsorship|sponsorship (is )?available"):
        yield "work_auth", "posting offers sponsorship, card is unstated"
    why = stance_conflict(card_dict, ps)
    if why:
        yield "work_auth", f"posting contradicts its stance: {why}"


def _employment_type(c, r, card_dict, reason, rs, pf, ps):
    if not _re(ps, _TYPE[r.employment_type]):
        yield "employment_type", f"posting does not state {r.employment_type}"
    want = ", ".join(TYPE_TEXT[t] for t in c.employment_types_wanted) or "any"
    if f"Job types wanted: {want}" not in pf:
        yield "employment_type", "profile lacks job types wanted"


def _license(c, r, card_dict, reason, rs, pf, ps):
    name = r.license or ""
    if not has_phrase(ps, name):
        yield "license", f"posting does not name {credential(name)}"
    held = reason.startswith("holds")
    if held and not has_phrase(rs, name):
        yield "license", f"resume does not show {name}"
    if not held and has_phrase(rs, name):
        yield "license", f"resume shows {name} the candidate lacks"


# The posting says the role is for current students. 'Enrollment in' counts only before a degree program
# ('Enrollment in a bachelor's or higher degree program'), not a benefits plan or a new-graduate program. The
# employer's own E-Verify enrollment ('Grisham Systems is enrolled in the E-Verify program', rq_01186~sponsors; 'an
# E-Verify enrolled employer') says nothing about students.
ENROLLMENT_IN = (r"\benrollment (?:in|at) (?:[\w'’-]+ ){0,4}?(?:(?:degree|academic|undergraduate|bachelor|doctoral)\b|"
                 r"master(?:['’]s|s)\b|ph\.?d\b|graduate (?:degree|school|studies)\b|associate['’]?s? degree\b)")
NOT_E_VERIFY = r"(?! (?:in|with|into) (?:the )?(?:federal )?e-?verify)"  # after 'enrolled'
ENROLLED = (rf"currently (?:enrolled{NOT_E_VERIFY}|pursuing)|current(ly)? .{{0,30}}students?|(?<!verify )(?<!verify-)enrolled{NOT_E_VERIFY}|"
            rf"students? (pursuing|enrolled)|{ENROLLMENT_IN}")


def _enrollment(c, r, card_dict, reason, rs, pf, ps):
    if not _re(ps, ENROLLED):
        yield "enrollment", "posting does not say it is for current students"
    if degree_line(c) not in pf:
        yield "enrollment", "profile lacks the education line"
    gy = c.graduation_year
    if gy is not None and not has_number(rs, gy):
        yield "enrollment", f"resume does not show graduation year {gy}"
    if gy is not None and gy > CURRENT_YEAR and not _re(rs, r"expected|anticipated|in progress|candidate|current"):
        yield "enrollment", "resume does not mark the graduation as expected"


# The widening tail of a degree-field entry: 'Biology (or a related field)', 'Chemistry or a closely related
# discipline', 'Physics, or equivalent'. The posting may word it its own way; the field itself must be there.
_FIELD_TAIL = re.compile(r"\s*\((?:[^)]*\b(?:related|equivalent|similar|relevant|comparable|other)\b[^)]*)\)|"
                         r",?\s+(?:or|and/or)\s+(?:a |an |any |other |another )?(?:closely |highly |directly )?"
                         r"(?:related|equivalent|similar|relevant|comparable|other)\b.*$", re.I)


def field_core(entry: str) -> str:
    """A degree-field entry without its 'or a related field' tail."""
    return _FIELD_TAIL.sub("", entry).strip(" ,;") or entry


def field_shown(text: str, entry: str) -> bool:
    """The posting names this degree field, its widening tail aside, also with the head word shared in a list:
    'Electrical or Computer Engineering' shows Electrical Engineering, 'Chemical or Biomedical Engineering' shows Chemical
    Engineering, 'Electrical/Computer Engineering' too."""
    core = field_core(entry)
    if has_phrase(text, core):
        return True
    words = core.split()
    if len(words) < 2:
        return False
    first, head = re.escape(" ".join(words[:-1])), re.escape(words[-1])
    return re.search(rf"\b{first}\s*(?:/|,|&|\bor\b|\band\b)[^.;:\n]{{0,60}}?\b{head}\b", text, re.I) is not None


def _degree(c, r, card_dict, reason, rs, pf, ps):
    # a student role that asks for no degree level still checks the field ("studying Biology, a
    # field the role asks for"); the enrollment probe checks its "currently enrolled" wording
    if r.degree_level != "none" and not _re(ps, _LEVEL[r.degree_level]):
        yield "degree", f"posting does not state a {r.degree_level} degree"
    if reason.startswith(("degree field", "studying")):
        # every field the reason lists, not just one: a reason quoting 'Computer Science or Software Engineering or
        # Mathematics' beside a posting that says only 'an applicable discipline' (and 'Software Engineering' in its
        # title) cites a list the model never sees (pr_4023ff03fb20, RTX)
        unseen = [f for f in field_named(card_fields(r)) if not field_shown(ps, f)]
        if unseen:
            yield "degree", f"posting does not name the degree fields ({'; '.join(unseen)})"
        if c.degree_field and not has_phrase(rs, c.degree_field):
            yield "degree", f"resume does not show the degree field {c.degree_field}"
    elif c.degree_level != "none" and not _re(rs, _LEVEL[c.degree_level]):
        yield "degree", f"resume does not show the {c.degree_level} degree"


def _wordings(skill: str) -> set[str]:
    """Every wording of a skill the rule treats as the same one (schema.SKILL_SYNONYMS)."""
    k = skill_canon(skill)
    return {skill} | {x for name, alts in SKILL_SYNONYMS.items() if skill_canon(name) == k for x in (name, *alts)}


def _resume_shows(rs: str, skill: str) -> bool:
    return any(has_phrase(rs, w) for w in _wordings(skill))


def _names_lacked(rs: str, skill: str, c: CandidateCard) -> bool:
    """Does the resume name `skill` as a skill of its own? Mentions inside a longer name the
    candidate does hold ("AutoCAD" in "AutoCAD Civil 3D", "Design" in "PCB design") or inside
    their degree field do not count."""
    text = rs
    for longer in (*c.skills, c.degree_field):
        if longer and skill_key(longer) != skill_key(skill) and has_phrase(longer, skill):
            text = re.sub(re.escape(longer), " ", text, flags=re.I)
    return has_phrase(text, skill)


def _skills(c, r, card_dict, reason, rs, pf, ps):
    # read skills as the rule does (rules.held_skills): any wording of a resume skill, a concrete
    # skill that shows a generic one (Python shows "Programming"), and what the degree field shows;
    # soft skills never count, so they are never checked
    have = held_skills(c)
    implied = degree_held(c)
    if reason.startswith("missing nice-to-have skill: "):
        for s in counted_nice(r):
            if group_met((s,), have):
                continue
            if not has_phrase(ps, s):
                yield "skill_wording", f"posting does not name nice-to-have {s}"
            if _names_lacked(rs, s, c):
                yield "skill_contradiction", f"resume names {s}, which the candidate lacks"
        return
    met_reason = reason == ALL_SKILLS_MET
    for g in required_groups(r):
        met = group_met(g, have)
        if met != met_reason:
            continue  # a missing-skill reason is about the unmet groups only
        for x in g:
            if not has_phrase(ps, x):
                yield "skill_wording", f"posting does not name required {x}"
        if met:
            if group_met(g, implied):
                continue  # the degree field shows it; the degree probe checks the field shows
            shown = [s for s in c.skills if any(skill_meets(x, s) for x in g)]
            if shown and not any(_resume_shows(rs, s) for s in shown):
                yield "skill_wording", f"resume does not name {shown[0]}"
        else:
            for x in g:
                if _names_lacked(rs, x, c):
                    yield "skill_contradiction", f"resume names {x}, which the candidate lacks"


def _years(c, r, card_dict, reason, rs, pf, ps):
    if r.min_years and not has_years(ps, r.min_years):
        yield "years", f"posting does not state {r.min_years} years"
    if c.years and not has_years(rs, c.years):
        yield "years", f"resume does not state {c.years} years"


def _location(c, r, card_dict, reason, rs, pf, ps):
    if reason.startswith(("based in", "location works")):
        if not any(has_phrase(ps, _city(x)) for x in r.locations):
            yield "location", "posting does not name its location"
        for x in (c.current_location, *c.willing_locations):
            if x and x not in pf:
                yield "location", f"profile lacks {x}"
        return
    if not _re(ps, _MODE[r.work_mode]):
        yield "location", f"posting does not say it is {r.work_mode}"
    if f"Work arrangement: {REMOTE_TEXT[c.remote_preference]}" not in pf:
        yield "location", "profile lacks the work arrangement"


def _preferences(c, r, card_dict, reason, rs, pf, ps):
    if reason.startswith("pay"):
        if not has_money(ps, r.salary_max):
            yield "salary", f"posting does not state the top of the pay range ({r.salary_max})"
        if f"${c.salary_floor:,}" not in pf:
            yield "salary", "profile lacks the salary floor"
    elif "an industry you" in reason or "you listed as wanted" in reason:
        if r.industry and not has_industry(ps, r.industry):
            yield "industry", f"posting does not name its industry ({r.industry})"
        for x in (*c.industries_wanted, *c.industries_avoided):
            if x not in pf:
                yield "industry", f"profile lacks {x}"
    elif reason.startswith("title includes"):
        for x in c.roles_avoided:
            stem = norm(x).removesuffix("s")  # the writer may make a plural title singular
            if norm(x) in norm(r.title) and not _re(ps, rf"\b{re.escape(stem)}"):
                yield "role", f"posting title does not show {x}"
            if x not in pf:
                yield "role", f"profile lacks {x}"


def _thin(c, r, card_dict, reason, resume, profile, posting):
    """A thin posting's reason states the posting's own shortness, which the shown posting is: no fact to look for."""
    return iter(())


_FACTS = {
    "thin": _thin,
    "work_auth": _work_auth,
    "employment_type": _employment_type,
    "license": _license,
    "enrollment": _enrollment,
    "degree": _degree,
    "skills": _skills,
    "years": _years,
    "location": _location,
    "preferences": _preferences,
}
# reasons that state no fact of their own
GENERIC_REASONS = frozenset({"meets the stated requirements", "otherwise meets the requirements", "no requirements to miss"})


def missing_facts(c: CandidateCard, r: RequirementCard, reasons: list[str], resume: str, profile: str,
                  posting: str) -> list[tuple[str, str]]:
    """(category, detail) for every fact behind `reasons` that the prompt text does not show."""
    owner = reasons_by_check(c, r)
    card_dict = r.to_dict()
    out: list[tuple[str, str]] = []
    for reason in reasons:
        if reason in GENERIC_REASONS:
            continue
        check = owner.get(reason)
        if check is None:
            out.append(("unknown_reason", reason))
            continue
        for miss in _FACTS[check](c, r, card_dict, reason, resume, profile, posting):
            if miss not in out:
                out.append(miss)
    return out
