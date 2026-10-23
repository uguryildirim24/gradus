"""The label rule: one deterministic function over (candidate, card).

Excluded: an unmet must-have or a hard filter (a work mode the profile rules out is one: remote only for a hybrid or
          onsite role, hybrid or remote only for an onsite one; soft only on a thin posting).
Explore:  every must-have met, but a preference mismatch, a large nice-to-have gap
          or a soft stretch (one year short, overqualified, still finishing the
          degree, an adjacent degree field, uncertain work auth).
For you:  must-haves and preferences all met.

Dates: graduation_year <= CURRENT_YEAR (2026) is a finished degree; later is still
enrolled. Places compare by metro (metros.py). Soft skills listed as requirements
("Communication", "Problem solving") never count for or against anyone; generic skills are
met by concrete ones (Python meets "Programming": schema.skill_meets).

The rule never reads name, country of education or language. The reasons it
writes are the same facts that decided the bucket, so they are faithful.

Work authorization, decided from the extraction audit (Travelers, Graco, Ankura,
Marsh):
- citizens_only covers citizenship, US permanent residency, "permanent" work
  authorization, ITAR and clearance. Only citizen_or_pr passes; every F-1 status,
  CPT included, is excluded. Ankura ("U.S. citizen or permanent resident ... not
  eligible for ... OPT, CPT") and Marsh ("on a permanent basis") are this case.
  ITAR wording that takes a "U.S. person, or eligible to obtain an export license"
  ("eligible to obtain the required authorizations", "otherwise eligible for a federally
  issued export control license": SpaceX, Rocket Lab, Hadrian, K2 Space) is citizens_only
  too. The author decided this (2026-10-19): an export license for an F-1 candidate is rare in
  practice, so every F-1 status gets Skip, for export control. The reason the rule writes
  is still "limited to US citizens or permanent residents", and the real test still scores
  no temporary-status Skip on such text (build_real_test.EXPORT_ROUTE), because that
  reason says more than the text does.
- will_not_sponsor_or_opt is a posting that names OPT or STEM OPT (I-983) as
  unsupported, on top of no sponsorship (Travelers, Graco). It excludes
  needs_h1b_now, f1_opt and stem_opt. f1_cpt on an internship stays eligible,
  because neither posting mentions CPT; a posting that does refuse CPT is in
  practice a permanent-authorization posting and goes to citizens_only.
- will_not_sponsor alone excludes needs_h1b_now only; OPT gets a soft check.

Thin postings (v4 fit layer): a posting that says too little to judge the fit (a short teaser: no skill, degree,
experience or license on its card, and a body under THIN_CHARS characters) is Explore with a reason that says so.
Hard eligibility fails still win (Skip). The card carries it as thin_posting, set by mark_thin from the card and the
posting's own text, never by extraction; on every card without it, label() is what it was.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from .degrees import STEM_DEGREE_FIELDS, canon_field, card_fields, constrains, degree_skills, field_match, field_named, \
    parse_fields
from .metros import join_places, places_match, posting_places
from .onet import license_fits
from .sample import CPT_MAX_AHEAD, INDUSTRIES
from .schema import (
    CURRENT_YEAR,
    DEGREE_LEVELS,
    CandidateCard,
    RequirementCard,
    group_met,
    is_soft_skill,
    norm,
    skill_canon,
)

OVERQUALIFIED_YEARS = 6  # at least this many years beyond the minimum (and 2*min+4) is "overqualified"
INTERN_OVERQUALIFIED_YEARS = 4  # the same for a non-student on an internship
SOFT_YEARS_SHORT = 1  # up to this many years short is a stretch, not a fail (minimums of 2+ only)
NICE_SOFT_MIN = 3  # this many missing nice-to-haves (or more than half, and at least two) makes a stretch
NICE_SOFT_FLOOR = 2  # one missing nice-to-have never decides the label, even when it is the only one

LABEL_WORDS = {"For you": "Keep", "Explore": "Explore", "Excluded": "Skip"}
WORD_TO_BUCKET = {v: k for k, v in LABEL_WORDS.items()}
MAX_REASONS = 4
OTHERWISE_OK = "otherwise meets the requirements"
NO_REQUIREMENTS = "no requirements to miss"
GENERIC_REASONS = frozenset({OTHERWISE_OK, NO_REQUIREMENTS})
# No count: the rule leaves out soft-skill groups ("Communication") that the posting still lists,
# so "5 of 5" would read wrong against a posting that names six.
ALL_SKILLS_MET = "all required skills met"


@dataclass
class _Verdict:
    hard: list[str] = field(default_factory=list)
    soft: list[str] = field(default_factory=list)
    good: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def _degree_rank(level: str) -> int:
    return DEGREE_LEVELS.index(level)


def is_enrolled(c: CandidateCard) -> bool:
    return c.graduation_year is not None and c.graduation_year > CURRENT_YEAR and c.degree_level != "none"


def enrolled_at(c: CandidateCard, level: str) -> bool:
    """Currently pursuing `level` or a higher degree (graduation still ahead)."""
    return is_enrolled(c) and _degree_rank(c.degree_level) >= _degree_rank(level)


LEVEL_TEXT = {"associate": "associate degree", "bachelor": "bachelor's", "master": "master's",
              "doctoral": "doctorate"}
# "a bachelor's degree", as a reason names a degree level
DEGREE_NAME = {"associate": "an associate degree", "bachelor": "a bachelor's degree", "master": "a master's degree",
               "doctoral": "a doctorate"}


REQUIREMENT_NAME = {"associate": "associate degree", "bachelor": "bachelor's degree", "master": "master's degree",
                    "doctoral": "doctoral degree"}


def _or_higher(level: str) -> str:
    return DEGREE_NAME[level] + ("" if level == "doctoral" else " or higher")


# The degree a student has certainly finished, by the one they are pursuing: a graduate student
# holds a bachelor's (a doctoral student only that: many enter a PhD straight from one, and the
# profile cannot show a master's); an undergraduate holds none (a bachelor's does not follow an
# associate degree, and the profile cannot show one).
STUDENT_HELD = {"associate": "none", "bachelor": "none", "master": "bachelor", "doctoral": "bachelor"}


def held_level(c: CandidateCard) -> str:
    """The highest degree a candidate has finished (STUDENT_HELD for a student)."""
    if not is_enrolled(c):
        return c.degree_level
    return STUDENT_HELD[c.degree_level]


def _check_enrollment(c: CandidateCard, r: RequirementCard, v: _Verdict) -> None:
    if r.enrolled_in == "none":
        return
    need = r.enrolled_in
    what = "a degree" if need == "associate" else _or_higher(need)
    if enrolled_at(c, need):
        v.good.append(f"currently pursuing {what}, as the role asks")
    elif c.graduation_year is not None and c.graduation_year <= CURRENT_YEAR:
        v.hard.append(f"for students currently pursuing {what}; you have graduated")
    elif is_enrolled(c):
        v.hard.append(f"for students currently pursuing {what}; you are pursuing {DEGREE_NAME[c.degree_level]}")
    else:
        v.hard.append(f"for students currently pursuing {what}; you are not a student")


def _check_degree(c: CandidateCard, r: RequirementCard, v: _Verdict) -> None:
    level_ok = True
    if r.degree_level != "none":
        need = r.degree_level
        if _degree_rank(c.degree_level) < _degree_rank(need):
            have = (f"you are pursuing {DEGREE_NAME[c.degree_level]}" if is_enrolled(c)
                    else "you have no degree" if c.degree_level == "none" else f"you have {DEGREE_NAME[c.degree_level]}")
            v.hard.append(f"needs {_or_higher(need)}; {have}")
            return
        # a student role that names the level it is for (degree_level >= enrolled_in) is met by pursuing it
        student_role = r.enrolled_in != "none" and _degree_rank(need) >= _degree_rank(r.enrolled_in)
        if is_enrolled(c) and _degree_rank(held_level(c)) < _degree_rank(need) and not student_role:
            if c.degree_level == need:
                v.soft.append(f"needs a finished {LEVEL_TEXT[need]}; you finish your {LEVEL_TEXT[need]} "
                              f"in {c.graduation_year}")
            else:
                v.soft.append(f"needs a finished {LEVEL_TEXT[need]}; you are pursuing a "
                              f"{LEVEL_TEXT[c.degree_level]}, expected {c.graduation_year}")
            level_ok = False
    elif r.enrolled_in == "none" or not enrolled_at(c, r.enrolled_in):
        return  # no degree asked, or the enrollment check already failed
    if constrains(fields := card_fields(r)):
        # quote the posting's "or a related field" too: it is what made a near field a stretch, not a fail
        names = " or ".join(field_named(fields))
        if parse_fields(fields)[2]:
            names += " (or a related field)"
        fit = field_match(c.degree_field, fields)
        if fit == "none":
            v.hard.append(f"degree field should be {names}")
            return
        if fit == "adjacent":
            v.soft.append(f"degree field should be {names}; yours, {c.degree_field}, is close")
            return
        if r.degree_level == "none":
            v.good.append(f"studying {c.degree_field}, a field the role asks for")
            return
    if level_ok and r.degree_level != "none":
        v.good.append(f"meets the {REQUIREMENT_NAME[r.degree_level]} requirement")


def _a(word: str) -> str:
    return f"an {word}" if word[:1] in "aeiou" else f"a {word}"


def _have(level: str) -> str:
    return "no degree" if level == "none" else DEGREE_NAME[level]


def _years(n: int) -> str:
    return f"{n} year" if n == 1 else f"{n} years"


def overqualified_years(c: CandidateCard, r: RequirementCard) -> int | None:
    """Years of experience from which this candidate is overqualified for this role, or None
    when they cannot be. Entry and mid-level roles: OVERQUALIFIED_YEARS beyond the minimum (and
    2*min+4). Internships: INTERN_OVERQUALIFIED_YEARS beyond the minimum, but only for someone
    who is not a student (grad students with years of work are normal there). Student roles
    (enrolled_in set) never: the enrollment check decides who they are for."""
    if r.enrolled_in != "none":
        return None
    if r.employment_type == "internship":
        return None if is_enrolled(c) else r.min_years + INTERN_OVERQUALIFIED_YEARS
    if r.seniority not in ("entry", "mid"):
        return None
    return max(r.min_years + OVERQUALIFIED_YEARS, 2 * r.min_years + 4)


def overqualified(c: CandidateCard, r: RequirementCard) -> bool:
    bar = overqualified_years(c, r)
    return bar is not None and c.years >= bar


def _check_years(c: CandidateCard, r: RequirementCard, v: _Verdict) -> None:
    short = r.min_years - c.years
    if short > 0 and (short > SOFT_YEARS_SHORT or r.min_years < 2):
        v.hard.append(f"needs {r.min_years}+ years of experience; you have {c.years}")
    elif short > 0:
        v.soft.append(f"{short} year short of the {r.min_years}-year minimum")
    else:
        if r.min_years:
            v.good.append(f"{_years(c.years)} of experience covers the {r.min_years}-year minimum")
        if overqualified(c, r):
            asked = f"{r.min_years} asked" if r.min_years else "no minimum asked"
            what = " for an internship" if r.employment_type == "internship" else ""
            v.soft.append(f"you may be overqualified{what} ({c.years} years; {asked})")


def required_groups(r: RequirementCard) -> tuple[tuple[str, ...], ...]:
    """Must-skill groups the rule enforces: groups made only of soft skills are left out."""
    return tuple(g for g in r.must_skills if not all(is_soft_skill(x) for x in g))


def counted_nice(r: RequirementCard) -> tuple[str, ...]:
    """Nice-to-haves the rule counts: soft skills never count."""
    return tuple(s for s in r.nice_skills if not is_soft_skill(s))


def degree_held(c: CandidateCard) -> set[str]:
    """skill_canon keys the candidate's degree field shows (degrees.degree_skills)."""
    return {skill_canon(x) for x in degree_skills(c.degree_field)} if c.degree_level != "none" else set()


def held_skills(c: CandidateCard) -> set[str]:
    """skill_canon keys a candidate holds: their resume skills plus what their degree field shows
    (a Chemistry graduate holds "Chemistry" without listing it)."""
    return {skill_canon(s) for s in c.skills} | degree_held(c)


def nice_gap(c: CandidateCard, r: RequirementCard) -> list[str]:
    have = held_skills(c)
    return [s for s in counted_nice(r) if not group_met((s,), have)]


def nice_is_soft(n_missing: int, n_nice: int) -> bool:
    return n_missing >= NICE_SOFT_FLOOR and (n_missing >= NICE_SOFT_MIN or n_missing * 2 > n_nice)


def _check_skills(c: CandidateCard, r: RequirementCard, v: _Verdict) -> None:
    have = held_skills(c)
    groups = required_groups(r)
    unmet = [g for g in groups if not group_met(g, have)]
    singles = [g[0] for g in unmet if len(g) == 1]
    if singles:
        v.hard.append(f"missing required skill: {', '.join(singles)}")
    for g in unmet:
        if len(g) > 1:
            v.hard.append(f"needs one of: {', '.join(g)}")
    if not unmet and groups:
        v.good.append(ALL_SKILLS_MET)
    nice = nice_gap(c, r)
    if nice_is_soft(len(nice), len(counted_nice(r))):
        v.soft.append(f"missing nice-to-have skill: {', '.join(nice)}")


_CREDENTIAL_WORDS = ("license", "certification", "certificate", "credential", "admission", "permit")


def credential(name: str) -> str:
    """"RN license" stays; "Series 65" reads "Series 65 license"."""
    words = norm(name).split()
    return name if words and words[-1] in _CREDENTIAL_WORDS else f"{name} license"


def _check_license(c: CandidateCard, r: RequirementCard, v: _Verdict) -> None:
    if not r.license:
        return
    if norm(r.license) in {norm(x) for x in c.licenses}:
        v.good.append(f"holds the {credential(r.license)}")
    else:
        v.hard.append(f"requires the {credential(r.license)}")


def _check_employment_type(c: CandidateCard, r: RequirementCard, v: _Verdict) -> None:
    wanted = c.employment_types_wanted
    if wanted and r.employment_type not in wanted:
        kind = r.employment_type.replace("_", "-")
        v.hard.append(f"it is {_a(kind)} role and you want {_join_types(wanted)}")
    elif wanted:
        v.good.append("job type is what you want")


def _join_types(types: tuple[str, ...]) -> str:
    return " or ".join(t.replace("_", "-") for t in types)


def matched_location(c: CandidateCard, r: RequirementCard) -> str | None:
    """The posting place this candidate can work at: one near where they live first, else one
    they would move to. None when no place works, or the role is remote or names no place."""
    if r.work_mode == "remote" or not r.locations:
        return None
    read = posting_places(r.locations)
    for p in (c.current_location, *c.willing_locations):
        for w, x in zip(r.locations, read):
            if p and places_match(p, x):
                return w
    return None


def location_ok(c: CandidateCard, r: RequirementCard) -> bool:
    if r.work_mode == "remote" or not r.locations:
        return True
    return matched_location(c, r) is not None


WORK_MODE_PREFIX = "the role is "  # every work-mode reason starts so (hybrid_rules splits it from the place check)


def mode_ruled_out(c: CandidateCard, r: RequirementCard) -> bool:
    """The candidate's work arrangement rules the role's work mode out: remote only for a hybrid or onsite role,
    hybrid or remote only for an onsite one."""
    pref, mode = c.remote_preference, r.work_mode
    return (pref == "remote_only" and mode != "remote") or (pref == "hybrid_or_remote" and mode == "onsite")


def _check_location(c: CandidateCard, r: RequirementCard, v: _Verdict) -> None:
    """Place, then work mode. A work mode the profile rules out is a hard fail (the rubric's Skip), except on a thin
    posting: a teaser that states nothing about the role cannot rule a mode out (its card's onsite is only the
    default), so there it stays a soft stretch."""
    mode = r.work_mode
    if mode != "remote" and r.locations:
        where = matched_location(c, r)
        if where is None:
            v.hard.append(f"based in {join_places(r.locations)}; you are not there or willing to move")
            return
        v.good.append(f"location works ({where})")
    pref = c.remote_preference
    if mode_ruled_out(c, r):
        out = v.soft if r.thin_posting else v.hard
        if pref == "remote_only":
            out.append(f"{WORK_MODE_PREFIX}{mode}; you said remote only")
        else:
            out.append(f"{WORK_MODE_PREFIX}onsite; you prefer hybrid or remote")
    elif pref != "any":
        v.good.append(f"{mode} matches your preference")


def _check_work_auth(c: CandidateCard, r: RequirementCard, v: _Verdict) -> None:
    auth, stance = c.work_auth, r.sponsorship
    if r.citizens_only:
        if auth == "citizen_or_pr":
            return
        v.hard.append("limited to US citizens or permanent residents")
        v.flags.append("work_auth:citizens_only")
        return
    if auth == "citizen_or_pr":
        return
    if auth == "needs_h1b_now":
        if stance in ("will_not_sponsor", "will_not_sponsor_or_opt"):
            v.hard.append("the employer will not sponsor visas and you need sponsorship now")
            v.flags.append("work_auth:needs_sponsorship_no_sponsor")
        elif stance == "sponsors":
            v.good.append("employer sponsors visas")
        else:
            v.soft.append("the posting does not say whether it sponsors visas")
            v.flags.append("work_auth:sponsorship_unstated")
        return
    if auth == "f1_cpt":
        # CPT is curricular training: internships and co-ops, whatever the hours (a part-time
        # job that is not an internship is no more covered than a full-time one)
        if r.employment_type == "internship":
            v.good.append("CPT covers this internship")
        else:
            v.soft.append("CPT only covers internships and co-ops")
            v.flags.append("work_auth:cpt_not_internship")
        return
    if auth in ("f1_opt", "stem_opt"):
        label = "STEM OPT" if auth == "stem_opt" else "OPT"
        if stance == "will_not_sponsor_or_opt":
            v.hard.append(f"the employer does not accept OPT or STEM OPT and you are on {label}")
            v.flags.append("work_auth:no_opt")
            return
        if stance == "sponsors":
            v.good.append(f"employer sponsors, which helps on {label}")
            return
        extra = ""
        if auth == "stem_opt" and r.e_verify:
            extra = " (employer uses E-Verify, as STEM OPT needs)"
        v.soft.append(f"you are on {label}; check the employer's sponsorship plans{extra}")
        v.flags.append(f"work_auth:{auth}_check_sponsorship")
        return
    # other_visa
    v.soft.append("check that your visa allows this job")
    v.flags.append("work_auth:other_visa_check")


# Posting industry wordings outside sample.INDUSTRIES that mean one of its items.
INDUSTRY_SYNONYMS = {
    "financial services": "finance", "banking": "finance", "investment banking": "finance",
    "asset management": "finance", "investment management": "finance", "capital markets": "finance",
    "proprietary trading": "finance", "trading": "finance", "hedge fund": "finance",
    "insurance": "finance", "accounting": "finance",
    "software": "technology", "internet": "technology", "saas": "technology", "semiconductor": "technology",
    "semiconductors": "technology", "information technology": "technology", "it services": "technology",
    "artificial intelligence": "technology", "ai": "technology", "cloud computing": "technology",
    "computer hardware": "technology", "consumer electronics": "technology", "cybersecurity": "technology",
    "tech": "technology", "computer software": "technology", "telecommunications": "technology",
    "hospital and health care": "healthcare", "hospitals": "healthcare", "health care": "healthcare",
    "medical devices": "healthcare", "health": "healthcare", "hospital": "healthcare",
    "pharmaceuticals": "biotech and pharma", "pharmaceutical": "biotech and pharma",
    "biotechnology": "biotech and pharma", "life sciences": "biotech and pharma", "biotech": "biotech and pharma",
    "pharma": "biotech and pharma",
    "higher education": "education", "k 12 education": "education", "k-12 education": "education",
    "automotive": "manufacturing", "industrial manufacturing": "manufacturing", "chemicals": "manufacturing",
    "industrial automation": "manufacturing", "electronics manufacturing": "manufacturing",
    "management consulting": "consulting", "professional services": "consulting",
    "law": "legal", "legal services": "legal", "law firm": "legal",
    "transportation": "logistics", "supply chain": "logistics", "shipping": "logistics", "freight": "logistics",
    "energy": "utilities", "oil and gas": "utilities", "electric utilities": "utilities",
    "renewable energy": "utilities", "utility": "utilities",
    "media and entertainment": "media", "entertainment": "media", "publishing": "media", "advertising": "media",
    "public sector": "government", "federal government": "government", "state government": "government",
    "local government": "government", "non-profit": "nonprofit", "non profit": "nonprofit",
    "nonprofit organization": "nonprofit", "restaurants": "hospitality", "hotels": "hospitality",
    "travel": "hospitality", "food and beverage": "hospitality", "consumer goods": "retail", "e-commerce": "retail",
    "ecommerce": "retail", "research and development": "research", "scientific research": "research",
    "farming": "agriculture", "environmental": "environmental services", "law enforcement": "public safety",
}
_VOCAB = frozenset(INDUSTRIES)

# Words in a free-text posting industry ("Healthcare Technology", "Defense & Aerospace", "AI Robotics")
# -> the list items, or the outside-the-list industries sample.OUTSIDE_INDUSTRIES names, they mean.
# "-tech" compounds (FinTech, MedTech, EdTech) are technology too; biotech is not.
# An industry can mean several ("Health Insurance": healthcare and finance); a candidate wants it
# when they want any of them. Whole words, read after & becomes "and" and punctuation a space.
_INDUSTRY_WORDS = (
    (r"tech|technology|technologies|software|saas|internet|ai|artificial intelligence|robotics?|semiconductors?|"
     r"computing|cloud|data|cyber ?security|digital|networking|telecom\w*|wireless|broadband|hardware|electronics?|"
     r"iot|quantum|electronic design automation|eda|autonomous|web3|information services|it services|"
     r"neurotechnology|optical|photonics?|identity security|identity centric security|data security|"
     r"communications systems|mission critical communications|geographic information systems|"
     r"enterprise mobility management|(?!bio)\w+tech|gaming|video games", "technology"),
    (r"financ\w*|fintech|banking|bank|trading|investments?|asset management|capital|payments?|insurance|insurtech|"
     r"mortgage|financial markets|capital markets|crypto\w*|defi|digital assets?|tax|accounting|wealth|"
     r"hedge funds?|brokerage", "finance"),
    (r"health\w*|health care|medical|medtech|hospitals?|clinical|dental|managed care|diagnostics", "healthcare"),
    (r"biotech\w*|pharma\w*|biopharma|life sciences?|genomics|diagnostics", "biotech and pharma"),
    (r"education|edtech|universit(y|ies)|academic|schools?|k 12", "education"),
    (r"manufacturing|automotive|industrial|chemicals?|materials|steel|packaging|appliances|tires|hvac|refrigeration|"
     r"3d printing|machinery|equipment|motors?|vehicles?|petrochemicals?|plastics|coatings?|lighting", "manufacturing"),
    (r"consulting|advisory|professional services|business services|agency|engineering services|human resources|"
     r"staffing|recruiting", "consulting"),
    (r"legal|law firms?|law(?! enforcement)", "legal"),
    (r"logistics|supply chain|transportation|shipping|freight|distribution|rail|trucking|warehouse\w*|delivery|"
     r"maritime", "logistics"),
    (r"energy|utilit\w*|oil|natural gas|gas utility|nuclear|renewables?|renewable power|water|electric power|"
     r"power generation|power systems|electric utility|midstream|fusion|batter(y|ies)|solar|wind", "utilities"),
    (r"media|entertainment|publishing|advertising|ad ?tech|marketing|news|gaming|video games|film|music|broadcast\w*",
     "media"),
    (r"government|public sector|federal|national laborator(y|ies)|govtech", "government"),
    (r"non ?profit|nonprofit|religious|charit(y|ies)", "nonprofit"),
    (r"hospitality|restaurants?|hotels?|travel|food service|qsr|events|food and beverage|beverages?|catering|tourism",
     "hospitality"),
    (r"retail|consumer goods|consumer packaged goods|cpg|e ?commerce|apparel|fashion|luxury|grocery|sporting goods|"
     r"toys|consumer", "retail"),
    (r"research|r and d|laborator(y|ies)|scientific|science", "research"),
    (r"agricultur\w*|farming|agritech|agtech|turf", "agriculture"),
    (r"environmental|waste|climate|recycling", "environmental services"),
    (r"public safety|law enforcement|safety|emergency services|mass notification", "public safety"),
    (r"construction|building|plumbing|landscap\w*|aec", "construction"),
    (r"aerospace|defen[cs]e|aviation|space|satellites?|military|national security|drones?|aircraft|air traffic",
     "aerospace and defense"),
    (r"real estate|proptech|reit|housing|propert(y|ies)|self storage", "real estate"),
    (r"sports?|athletics|motorsports|fitness", "sports"),
    (r"architecture|urban planning", "architecture and planning"),
)
_INDUSTRY_RE = tuple((re.compile(rf"\b(?:{pat})\b"), key) for pat, key in _INDUSTRY_WORDS)


def _industry_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.lower().replace("&", " and ")).split())


def industry_key(text: str) -> str | None:
    """The sample.INDUSTRIES item a posting industry means, or None when it is outside the list."""
    k = norm(text.replace("&", " and "))
    if k in _VOCAB:
        return k
    return INDUSTRY_SYNONYMS.get(k)


def industry_ids(text: str) -> frozenset[str]:
    """Everything an industry name means: its list item when it names one, else the list items (and
    outside-the-list industries) its words name, plus its own wording, so an industry nobody's
    words cover only matches itself."""
    key = industry_key(text)
    if key:
        return frozenset({key})
    flat = _industry_text(text)
    return frozenset({flat} | {k for rx, k in _INDUSTRY_RE if rx.search(flat)})


def _role_word_rx(word: str) -> str:
    """A role word or its regular English plural as a whole word: technician(s), coach(es),
    secretary/secretaries. Never a prefix, so "intern" does not match "Internal" or "Internship"."""
    forms = [re.escape(word) + "(?:s|es)?"]
    if re.search(r"[^aeiou]y$", word):
        forms.append(re.escape(word[:-1]) + "ies")
    return rf"\b(?:{'|'.join(forms)})\b"


def role_hits(c: CandidateCard, r: RequirementCard) -> list[str]:
    """Avoided role words that appear in the title as whole words, singular or plural (O*NET titles
    are plural: "Technologists and Technicians")."""
    title = norm(r.title)
    return [x for x in c.roles_avoided if norm(x) and re.search(_role_word_rx(norm(x)), title)]


def _check_preferences(c: CandidateCard, r: RequirementCard, v: _Verdict) -> None:
    if c.salary_floor is not None and r.salary_max is not None:
        if r.salary_max < c.salary_floor:
            v.soft.append(f"pay tops out at ${r.salary_max:,}, under your ${c.salary_floor:,} floor")
        else:
            v.good.append("pay meets your floor")
    # a posting that names no industry is not checked; any other is checked by what it means
    # (industry_ids): "Healthcare Technology" is wanted by someone who wants healthcare or
    # technology; one no word covers ("Infrastructure") is wanted only when listed by name
    ind = industry_ids(r.industry) if r.industry else frozenset()
    if ind and ind & {k for x in c.industries_avoided for k in industry_ids(x)}:
        v.soft.append(f"{r.industry} is an industry you said you avoid")
    elif c.industries_wanted and ind:
        if ind & {k for x in c.industries_wanted for k in industry_ids(x)}:
            v.good.append(f"{r.industry} is an industry you want")
        else:
            v.soft.append(f"{r.industry} is not one you listed as wanted")
    hit = role_hits(c, r)
    if hit:
        v.soft.append(f"title includes {', '.join(hit)}, a role you said you avoid")


def status_problems(c: CandidateCard) -> list[str]:
    """Ways a candidate's work authorization contradicts their education. Empty when consistent."""
    gy, a = c.graduation_year, c.work_auth
    out = []
    enrolled = is_enrolled(c)
    if a in ("f1_cpt", "f1_opt", "stem_opt", "needs_h1b_now") and _degree_rank(c.degree_level) < 2:
        out.append(f"{a} without a bachelor's or higher")
    if a == "f1_cpt" and not enrolled:
        out.append("CPT needs current enrollment")
    if a == "f1_cpt" and enrolled and gy > CURRENT_YEAR + CPT_MAX_AHEAD.get(c.degree_level, gy):
        out.append("CPT comes after a year of study")
    if a == "f1_opt" and gy not in (CURRENT_YEAR - 1, CURRENT_YEAR):
        out.append("OPT is the year after graduating")
    if a == "stem_opt":
        if gy not in (CURRENT_YEAR - 3, CURRENT_YEAR - 2, CURRENT_YEAR - 1):
            out.append("STEM OPT follows OPT: graduated 1-3 years ago")
        if canon_field(c.degree_field) not in STEM_DEGREE_FIELDS:
            out.append("STEM OPT needs a STEM degree field")
        if c.years == 0:
            out.append("STEM OPT follows a year of OPT work")
    if a == "needs_h1b_now" and enrolled:
        out.append("needs H-1B now but still enrolled")
    if c.degree_level == "none" and (gy is not None or c.degree_field):
        out.append("no degree but a graduation year or field")
    return out


def license_problems(c: CandidateCard) -> list[str]:
    """Licenses this candidate's finished education could not lead to (a Medical license with a
    doctorate in Psychology). Empty when every license fits."""
    level = held_level(c)
    return [f"{x} with {_have(level)} in {c.degree_field or 'no field'}" for x in c.licenses
            if not license_fits(x, level, c.degree_field)]


# A posting body under this many characters can be a teaser (fit_pools.SHORT: "teaser" below 700).
THIN_CHARS = 700
THIN_REASON = "the posting says too little to judge the fit: it names no skill, degree, experience or license"


def says_too_little(r: RequirementCard) -> bool:
    """The card asks for nothing the resume could be checked against: no required skill the rule counts, no
    nice-to-have it counts, no degree level or field, no minimum years and no license. Eligibility facts (work
    authorization, enrollment, location, job type) do not count: they decide Skip, not the fit."""
    return not (required_groups(r) or counted_nice(r) or r.degree_level != "none" or r.degree_fields
                or r.min_years or r.license)


def is_thin(r: RequirementCard, text: str) -> bool:
    """says_too_little and the posting's own text (its body, without a Title/Company/Locations header) is under
    THIN_CHARS characters."""
    return says_too_little(r) and len((text or "").strip()) < THIN_CHARS


def mark_thin(r: RequirementCard, text: str) -> RequirementCard:
    """The card with thin_posting set from is_thin. Run it on every extracted card before label(), in training and
    in eval alike."""
    return replace(r, thin_posting=is_thin(r, text))


def _check_thin(c: CandidateCard, r: RequirementCard, v: _Verdict) -> None:
    if r.thin_posting:
        v.soft.append(THIN_REASON)


# Every check, in the order label() runs them; visibility.py names reasons by check. thin runs last, so on every other
# card the reasons and their order are what they were before it existed.
CHECKS = (
    ("work_auth", _check_work_auth),
    ("employment_type", _check_employment_type),
    ("license", _check_license),
    ("enrollment", _check_enrollment),
    ("degree", _check_degree),
    ("skills", _check_skills),
    ("years", _check_years),
    ("location", _check_location),
    ("preferences", _check_preferences),
    ("thin", _check_thin),
)


def reasons_by_check(c: CandidateCard, r: RequirementCard) -> dict[str, str]:
    """Each reason any check writes for this pair -> the check's name."""
    out: dict[str, str] = {}
    for name, check in CHECKS:
        v = _Verdict()
        check(c, r, v)
        for reason in (*v.hard, *v.soft, *v.good):
            out.setdefault(reason, name)
    return out


def label(c: CandidateCard, r: RequirementCard) -> tuple[str, list[str], list[str]]:
    """Return (bucket, reasons, flags). Pure and deterministic."""
    v = _Verdict()
    for _, check in CHECKS:
        check(c, r, v)
    if v.hard:
        return "Excluded", v.hard, v.flags
    if v.soft:
        # "otherwise meets the requirements" reads wrong after a year short, and says more than a thin posting does
        tail = [] if any("year short of" in x for x in v.soft) or THIN_REASON in v.soft else [OTHERWISE_OK]
        return "Explore", [*v.soft, *tail], v.flags
    return "For you", v.good or [NO_REQUIREMENTS], v.flags


def format_completion(bucket: str, reasons: list[str]) -> str:
    """Label word first, then one line of reasons."""
    word = LABEL_WORDS[bucket]
    text = "; ".join(r.strip().rstrip(".") for r in reasons[:MAX_REASONS])
    text = " ".join(text.split())  # one line, no stray whitespace
    return f"{word} — {text}."


def parse_completion(text: str) -> str:
    """Bucket named by a completion's first word."""
    word = text.split(maxsplit=1)[0] if text.split() else ""
    if word not in WORD_TO_BUCKET:
        raise ValueError(f"unknown label word {word!r}")
    return WORD_TO_BUCKET[word]
