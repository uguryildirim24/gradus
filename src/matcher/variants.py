"""Candidates built for one posting: a base that meets everything, then one-edit breaks.

Two kinds of base:
- copy: built from the posting card (`copy_base`; `full_match` is its plain,
  deterministic form). Its profile fields are drawn from the same spread as real
  candidates, its resume carries extra skills of its own, and it graduated 2016-2025.
- repaired: a real sampled candidate pulled just far enough toward the card to pass
  (`repair`): missing must-skills appended to their own skills, years, degree, place
  and preferences fixed, everything else kept (their skill list, relocation list,
  job types, work arrangement, salary floor, nuisance fields).

`make_variants(card, seed, base=...)` adds every single-edit break that applies.
Expected buckets come from construction, never from calling label(), so tests can check
the rule against an independent answer. Every candidate keeps a consistent work
authorization (rules.status_problems): CPT means enrolled, OPT means graduated in the
last year, STEM OPT means a STEM degree 1-3 years back.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from .degrees import GROUPS, STEM_DEGREE_FIELDS, adjacent_fields, canon_field, card_fields, constrains, \
    field_match, parse_fields
from .metros import ANYWHERE_US, METROS, all_places, cities_in, parse_place, place_key, places_match, posting_places
from .onet import LICENSE_EDUCATION, LICENSES, license_fits
from .rules import (
    CURRENT_YEAR,
    counted_nice,
    degree_held,
    held_level,
    industry_ids,
    industry_key,
    is_enrolled,
    license_problems,
    location_ok,
    nice_gap,
    nice_is_soft,
    overqualified,
    overqualified_years,
    required_groups,
    role_hits,
    status_problems,
)
from .sample import ALIAS_RATE, AVOID_WORDS, CONCEPTS, CPT_YEARS, FIELDS, INDUSTRIES, INDUSTRY_W, METRO_NAMES, ROLE_W, TYPE_W, \
    cpt_year, enrolled_year, graduated_year, sample_nuisance, sample_place, sample_willing, short_field, \
    wanted_role_words, weighted_sample
from .schema import (
    DEGREE_LEVELS,
    SKILL_SYNONYMS,
    CandidateCard,
    RequirementCard,
    group_met,
    is_soft_skill,
    norm,
    skill_canon,
    skill_meets,
)

_OTHER_LICENSES = tuple(sorted(set(LICENSES.values())))
# Every degree field the samplers use: a wrong field for one posting is a right one for another.
ALL_FIELDS = tuple(sorted({x for f in FIELDS.values() for x in f["deg"]}))


@dataclass(frozen=True)
class Variant:
    kind: str
    candidate: CandidateCard
    card: RequirementCard
    expected: str
    note: str = ""
    base: str = "copy"  # "copy" or "repaired": what the candidate was built from


def _rank(level: str) -> int:
    return DEGREE_LEVELS.index(level)


def _title(field: str) -> str:
    return " ".join(w if w.isupper() else w.capitalize() for w in field.split())


def _pick(rng: random.Random, pool, avoid: set[str]) -> str:
    options = [x for x in pool if norm(x) not in avoid]
    return rng.choice(options)


def exact_fields(card: RequirementCard) -> list[str]:
    """Degree fields that fully meet the card's degree_fields (display form)."""
    named, groups, _ = parse_fields(card_fields(card))
    out = [_title(x) for x in sorted(named)]
    for g in sorted(groups):
        members = STEM_DEGREE_FIELDS if g == "stem" else GROUPS[g]
        out += [_title(x) for x in sorted(members)]
    return [x for x in dict.fromkeys(out) if field_match(x, card_fields(card)) == "exact"]


def _field_for(card: RequirementCard, rng: random.Random | None, stem: bool = False) -> str:
    """A degree field the card accepts; from the card's own field pool when unconstrained."""
    need = LICENSE_EDUCATION.get(card.license) if card.license else None
    if constrains(card_fields(card)):
        opts = exact_fields(card)
        if need and need.fields:
            opts = [x for x in opts if license_fits(card.license, "doctoral", x)] or opts
        if stem:
            opts = [x for x in opts if canon_field(x) in STEM_DEGREE_FIELDS]
        return (rng.choice(opts) if rng else opts[0]) if opts else ""
    if need and need.fields:  # a license holder studied what the license follows from
        pool = list(need.fields)
    elif card.field in FIELDS:
        pool = list(FIELDS[card.field]["deg"])
    else:
        pool = ["Computer Science", "Business Administration"]
    if stem:  # no STEM field fits a Concierge or Nursing card: no STEM OPT holder for it
        pool = [x for x in pool if canon_field(x) in STEM_DEGREE_FIELDS]
        if not pool:
            return ""
    return rng.choice(pool) if rng else pool[0]


def _home(card: RequirementCard, rng: random.Random | None) -> str:
    """A city that meets the card's location: one of its places, a place in its metro, or a
    city in its state. Never a bare state, metro or country name: people live in a city."""
    if card.work_mode == "remote" or not card.locations:
        return sample_place(rng) if rng else METRO_NAMES[0]
    read = posting_places(card.locations)
    loc = read[0] if rng is None else rng.choice(read)
    p = parse_place(loc)
    if p.kind == "state":
        opts = cities_in(p.state)
        return (rng.choice(opts) if rng else opts[0]) if opts else loc
    if p.kind == "country":
        return sample_place(rng) if rng else METRO_NAMES[0]
    if p.metro and p.kind == "metro":
        return rng.choice(all_places(p.metro)) if rng else METROS[p.metro][0]
    if p.metro and rng is not None and rng.random() < 0.4:
        return rng.choice(all_places(p.metro))
    return loc


def _study_year(rng: random.Random, level: str, auth: str) -> int:
    """An expected graduation year for a student pursuing `level` (past the first year on CPT)."""
    gy = enrolled_year(rng, level)
    return cpt_year(gy, level) if auth == "f1_cpt" else gy


def _base_degree(card: RequirementCard) -> tuple[str, bool]:
    """(degree level, still enrolled) a base candidate needs for this card."""
    if card.enrolled_in != "none":
        need = max(_rank(card.enrolled_in), _rank(card.degree_level) + 1 if card.degree_level != "none" else 0)
        return DEGREE_LEVELS[min(need, 4)], True
    lic = LICENSE_EDUCATION.get(card.license) if card.license else None
    if card.degree_level == "none" and lic and lic.level == "none" and lic.fields is not None:
        return "none", False  # a trade or certificate license on a posting that asks for no degree
    level = card.degree_level if card.degree_level != "none" else "bachelor"
    if lic and _rank(lic.level) > _rank(level):
        level = lic.level
    return level, False


PRE_DEGREE_YEARS = 4  # years of work a graduate may have had before finishing their degree
# An F-1 bachelor's graduate on OPT came to study at about 18: a year of work at most (on campus,
# CPT) before graduating. Master's and doctoral students often worked at home first.
F1_BACHELOR_PRE_YEARS = 1


def license_needs_degree(licenses, degree_level: str) -> bool:
    """Does one of `licenses` follow from a degree (onet.LICENSE_EDUCATION level above none), on a card with one?"""
    if degree_level == "none":
        return False
    return any((n := LICENSE_EDUCATION.get(x)) is not None and n.level != "none" for x in licenses)


def pre_degree_years(c: CandidateCard) -> int:
    """Most years of work `c` can have had before graduating. None for the holder of a license their degree gives: the
    resume dates every year of work as a job (render_cards.job_dates), and the job such a license stands behind cannot
    start before the graduation (cd_v_00823: a residency in 2015, the MD in 2016)."""
    if license_needs_degree(c.licenses, c.degree_level):
        return 0
    if c.work_auth in ("f1_opt", "stem_opt") and c.degree_level == "bachelor":
        return F1_BACHELOR_PRE_YEARS
    return PRE_DEGREE_YEARS


def grad_floor(years: int, lo: int = 2016, pre: int = PRE_DEGREE_YEARS) -> int:
    """Earliest graduation year to draw for someone with `years` of work: `lo`, or earlier when
    their years need it (no more than `pre` of them before graduating)."""
    return min(lo, CURRENT_YEAR - years + pre)


def years_fit(c: CandidateCard) -> bool:
    """Do years of work and the graduation year agree? A graduate has at most pre_degree_years
    more years than years since graduating, and someone who graduated 3+ years ago has worked;
    an F-1 student on CPT has at most CPT_YEARS."""
    if c.degree_level == "none" or c.graduation_year is None:
        return True
    if is_enrolled(c):
        return c.work_auth != "f1_cpt" or c.years <= CPT_YEARS[c.degree_level]
    gap = CURRENT_YEAR - c.graduation_year
    return c.years <= gap + pre_degree_years(c) and not (gap >= 3 and c.years == 0)


def prefs_fit(c: CandidateCard) -> bool:
    """Do the stated preferences agree? Someone who wants internships, or is on CPT, does not avoid "intern"."""
    return not {norm(x) for x in c.roles_avoided} & wanted_role_words(c.employment_types_wanted, c.work_auth)


def _fit_roles(c: CandidateCard) -> CandidateCard:
    """`c` without the avoided role words their wanted job types contradict."""
    if prefs_fit(c):
        return c
    clash = wanted_role_words(c.employment_types_wanted, c.work_auth)
    return replace(c, roles_avoided=tuple(x for x in c.roles_avoided if norm(x) not in clash))


def unique_skills(skills: tuple[str, ...]) -> tuple[str, ...]:
    """Skills with each one listed once (any wording, skill_canon), in their first place."""
    seen: set[str] = set()
    out = []
    for x in skills:
        k = skill_canon(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return tuple(out)


def settle_years(c: CandidateCard, rng: random.Random, min_years: int = 0) -> CandidateCard | None:
    """`c` with years and graduation year in agreement: the graduation year moves, unless the work
    authorization pins it (OPT, STEM OPT, a CPT student), when years shrink, but never below
    `min_years`. None when that cannot be done."""
    if years_fit(c):
        return c
    if is_enrolled(c):
        cap = CPT_YEARS[c.degree_level]
        return replace(c, years=cap) if cap >= min_years else None
    if c.work_auth in ("f1_opt", "stem_opt"):
        cap = CURRENT_YEAR - c.graduation_year + pre_degree_years(c)
        if c.years <= cap:
            return None  # no years three years into STEM OPT, and the year is pinned
        return replace(c, years=cap) if cap >= min_years else None
    return replace(c, graduation_year=graduated_year(rng, c.years, grad_floor(c.years, pre=pre_degree_years(c)), 2025))


def students_can_pass(card: RequirementCard) -> bool:
    """False for a student role no student can fully meet: one for doctoral students that also
    asks for a finished master's, which a doctoral student is not credited with (rules.held_level)."""
    degree, enrolled = _base_degree(card)
    if not enrolled or card.degree_level == "none":
        return True
    student_role = _rank(card.degree_level) >= _rank(card.enrolled_in)
    probe = CandidateCard(degree_level=degree, graduation_year=CURRENT_YEAR + 1)
    return student_role or _rank(held_level(probe)) >= _rank(card.degree_level)


def full_match(card: RequirementCard) -> CandidateCard:
    """The plain base: meets every must-have, sets no preference, holds the first
    alternative of each must-skill group and every counted nice-to-have. When the card asks
    for current enrollment the candidate is still studying; otherwise they graduated."""
    years = card.min_years + 2
    if overqualified(replace(CandidateCard(), years=years), card):
        years = card.min_years
    degree, enrolled = _base_degree(card)
    pre = 0 if card.license and license_needs_degree((card.license,), degree) else PRE_DEGREE_YEARS
    grad = enrolled_year(random.Random(0), degree) if enrolled else \
        graduated_year(random.Random(0), years, grad_floor(years, pre=pre), 2025)
    return CandidateCard(
        field=card.field,
        degree_level=degree,
        degree_field=_field_for(card, None) if degree != "none" else "",
        graduation_year=grad if degree != "none" else None,
        years=years,
        skills=(*(g[0] for g in required_groups(card)), *counted_nice(card)),
        licenses=(card.license,) if card.license else (),
        current_location=_home(card, None),
        work_auth="citizen_or_pr",
    )


def _round_floor(x: float) -> int:
    return max(5000, int(x / 5000) * 5000)


def _floor_for(card: RequirementCard, rng: random.Random) -> int | None:
    """None or a round floor at or under the top of the posting's pay, never its salary_min."""
    if rng.random() < 0.5:
        return None
    if card.salary_max is None:
        return _round_floor(rng.uniform(30_000, 120_000))
    f = _round_floor(card.salary_max * rng.uniform(0.55, 1.0))
    if f == card.salary_min:
        f -= 5000
    return min(f, card.salary_max) if f > 0 else None


def _ind_key(card: RequirementCard) -> str | None:
    return industry_key(card.industry) if card.industry else None


def _ind_ids(card: RequirementCard) -> frozenset[str]:
    """What the posting's industry means to the rule (rules.industry_ids); empty when it names none."""
    return industry_ids(card.industry) if card.industry else frozenset()


def _means(items: tuple[str, ...]) -> set[str]:
    return {k for x in items for k in industry_ids(x)}


def industry_clash(c: CandidateCard) -> bool:
    """Does the profile want and avoid one industry (rules.industry_ids on both lists, so "Software" and
    "technology" meet)?"""
    return bool(_means(c.industries_wanted) & _means(c.industries_avoided))


def _ind_own(card: RequirementCard) -> str:
    """The posting's industry as a candidate lists it: its list item, else its own name in lower case."""
    return _ind_key(card) or card.industry.lower()


def _wanted_types(card: RequirementCard, rng: random.Random) -> tuple[str, ...]:
    r = rng.random()
    if r < 0.45:
        return ()
    if r < 0.75:
        return (card.employment_type,)
    return tuple(sorted({card.employment_type, *weighted_sample(rng, TYPE_W, 1, {card.employment_type})}))


def _wanted_industries(card: RequirementCard, rng: random.Random, avoided: set[str] = frozenset()) -> tuple[str, ...]:
    """Wanted industries for a full match: the posting's own, sometimes one more, none of them an `avoided` id."""
    key = _ind_key(card)
    r = rng.random()
    if r < 0.55:
        return ()
    if not card.industry:  # the posting names no industry: the rule does not check it
        k = rng.randint(1, 2)
        pool = [i for i in INDUSTRIES if not industry_ids(i) & avoided]
        return tuple(rng.sample(pool, min(k, len(pool))))
    own = _ind_own(card)  # an industry outside the list is wanted by its own name
    if r < 0.8:
        return (own,)
    return (own, *weighted_sample(rng, INDUSTRY_W, 1, {key, *avoided}))


def _remote_for(card: RequirementCard, rng: random.Random) -> str:
    if card.work_mode == "remote":
        return rng.choice(("any", "any", "hybrid_or_remote", "remote_only"))
    if card.work_mode == "hybrid":
        return rng.choice(("any", "any", "hybrid_or_remote"))
    return "any"


def _card_skills(card: RequirementCard) -> tuple[str, ...]:
    return (*(x for g in card.must_skills for x in g), *card.nice_skills)


def _meets_card(card: RequirementCard, skill: str) -> bool:
    """Does this resume skill meet any skill the card names (Python meets "Programming")?"""
    return any(skill_meets(x, skill) for x in _card_skills(card))


def _extra_skills(card: RequirementCard, pool: tuple[str, ...] | None) -> list[str]:
    """A candidate's own skills beyond the card: none of them meets anything the card names."""
    src = pool if pool else CONCEPTS.get(card.field, ())
    return [x for x in dict.fromkeys(src) if not _meets_card(card, x) and not is_soft_skill(x)]


def copy_base(card: RequirementCard, rng: random.Random, extra_pool: tuple[str, ...] | None = None) -> CandidateCard:
    """A full match whose label-neutral fields follow the real-candidate spread."""
    degree, enrolled = _base_degree(card)
    years = card.min_years + rng.randint(0, 5)
    if overqualified(replace(CandidateCard(), years=years), card):
        years = card.min_years + rng.randint(0, 3)
    if enrolled:
        years = max(card.min_years, min(years, 3))
        grad = enrolled_year(rng, degree)
    else:
        pre = 0 if card.license and license_needs_degree((card.license,), degree) else PRE_DEGREE_YEARS
        grad = graduated_year(rng, years, grad_floor(years, pre=pre), 2025)
    # skills: an alternative of each required group, most counted nice-to-haves, some of their own
    skills = [rng.choice(g) for g in required_groups(card)]
    nice = list(counted_nice(card))
    rng.shuffle(nice)
    allowed = 0
    while allowed + 1 <= len(nice) and not nice_is_soft(allowed + 1, len(nice)) and rng.random() < 0.5:
        allowed += 1
    skills += nice[allowed:]
    soft_groups = [g for g in card.must_skills if g not in required_groups(card)]
    skills += [g[0] for g in soft_groups if rng.random() < 0.5]
    extra = _extra_skills(card, extra_pool)
    skills += rng.sample(extra, min(len(extra), rng.randint(1, 5)))
    rng.shuffle(skills)
    home = _home(card, rng)
    country, language = sample_nuisance(rng, False)
    avoided: tuple[str, ...] = ()
    if rng.random() < 0.15:
        avoided = weighted_sample(rng, INDUSTRY_W, 1, _ind_ids(card))
    roles: tuple[str, ...] = ()
    if rng.random() < 0.08:
        roles = weighted_sample(rng, ROLE_W, 1, _role_words_in(card))
    field = _field_for(card, rng)
    if rng.random() < ALIAS_RATE:
        field = short_field(field)
    if degree == "none":
        field, grad = "", None
    return _fit_roles(CandidateCard(
        field=card.field,
        degree_level=degree,
        degree_field=field,
        graduation_year=grad,
        years=years,
        skills=tuple(skills),
        licenses=(card.license,) if card.license else (),
        current_location=home,
        willing_locations=sample_willing(rng, home),
        remote_preference=_remote_for(card, rng),
        work_auth="citizen_or_pr",
        salary_floor=_floor_for(card, rng),
        employment_types_wanted=_wanted_types(card, rng),
        industries_wanted=_wanted_industries(card, rng, _means(avoided)),
        industries_avoided=avoided,
        roles_avoided=roles,
        education_country=country,
        language=language,
    ))


# ---------------------------------------------------------------- work authorization

def with_status(c: CandidateCard, status: str, card: RequirementCard, rng: random.Random) -> CandidateCard | None:
    """`c` with work authorization `status` and the education that status implies, changing
    as little else as possible. None when that would also break the card's degree checks."""
    if status in ("citizen_or_pr", "other_visa"):
        return replace(c, work_auth=status)
    level = c.degree_level if _rank(c.degree_level) >= 2 else "bachelor"
    field = c.degree_field or _field_for(card, rng)
    if status == "f1_cpt":
        c = _fit_roles(replace(c, work_auth=status))  # CPT covers internships: "intern" is no role they avoid
        if is_enrolled(c) and _rank(c.degree_level) >= 2:
            return settle_years(replace(c, graduation_year=cpt_year(c.graduation_year, c.degree_level)),
                                rng, card.min_years)
        if card.enrolled_in != "none":
            return None
        if card.degree_level != "none":  # pursuing the next degree up, holding the one asked for
            level = DEGREE_LEVELS[min(max(_rank(card.degree_level) + 1, 2), 4)]
        years = min(c.years, CPT_YEARS[level])
        if years < card.min_years:
            return None  # an F-1 student this experienced is not who the status describes
        out = replace(c, work_auth=status, degree_level=level, degree_field=field,
                      graduation_year=cpt_year(enrolled_year(rng, level), level), years=years)
        if _rank(held_level(out)) < _rank(card.degree_level):
            return None  # a bachelor's student holds no associate, a doctoral one no master's (rules.held_level)
        return out
    if is_enrolled(c) and card.enrolled_in == "none" and _rank(card.degree_level) >= _rank(c.degree_level):
        return None  # finishing the degree the card asks for would add a second stretch
    if status == "stem_opt":
        if canon_field(field) not in STEM_DEGREE_FIELDS:
            field = _field_for(card, rng, stem=True)
            if not field or canon_field(field) not in STEM_DEGREE_FIELDS:
                return None
        # STEM OPT follows a year of OPT work: someone on it has at least one year
        out = replace(c, work_auth=status, degree_level=level, degree_field=field, years=max(c.years, 1),
                      graduation_year=CURRENT_YEAR - rng.choice((3, 2, 1)))
    elif status == "f1_opt":
        out = replace(c, work_auth=status, degree_level=level, degree_field=field,
                      graduation_year=rng.choice((CURRENT_YEAR - 1, CURRENT_YEAR)))
    elif status == "needs_h1b_now":
        gy = c.graduation_year if (c.graduation_year and not is_enrolled(c)) else graduated_year(rng, c.years)
        out = replace(c, work_auth=status, degree_level=level, degree_field=field, graduation_year=gy)
    else:
        raise ValueError(status)
    out = settle_years(out, rng, card.min_years)  # years and graduation year move together
    if out is not None and overqualified(out, card) and not overqualified(c, card):
        return None  # a student made a graduate can be overqualified for an internship: a second stretch
    return out


def auth_outcome(status: str, card: RequirementCard) -> str:
    """The work-auth check alone: "pass", "soft" or "hard" (mirrors rules._check_work_auth)."""
    if status == "citizen_or_pr":
        return "pass"
    if card.citizens_only:
        return "hard"
    stance = card.sponsorship
    if status == "needs_h1b_now":
        return {"sponsors": "pass", "unstated": "soft"}.get(stance, "hard")
    if status == "f1_cpt":
        return "pass" if card.employment_type == "internship" else "soft"
    if status in ("f1_opt", "stem_opt"):
        return {"will_not_sponsor_or_opt": "hard", "sponsors": "pass"}.get(stance, "soft")
    return "soft"


def fix_status(c: CandidateCard, rng: random.Random) -> CandidateCard:
    """Resolve a work authorization that contradicts the education (after a repair moved it)."""
    for _ in range(3):
        probs = status_problems(c)
        if not probs:
            return c
        if c.degree_level == "none" and (c.graduation_year is not None or c.degree_field):
            c = replace(c, graduation_year=None, degree_field="")
            continue
        a, gy = c.work_auth, c.graduation_year
        stem = canon_field(c.degree_field) in STEM_DEGREE_FIELDS
        if _rank(c.degree_level) < 2:
            c = replace(c, work_auth="citizen_or_pr")
        elif is_enrolled(c):
            c = replace(c, work_auth="f1_cpt", graduation_year=cpt_year(gy, c.degree_level))
        elif gy is not None and gy >= CURRENT_YEAR - 1:
            c = replace(c, work_auth="f1_opt")
        elif gy is not None and gy >= CURRENT_YEAR - 3 and stem and c.years >= 1:
            c = replace(c, work_auth="stem_opt")
        else:
            c = replace(c, work_auth="needs_h1b_now" if a != "citizen_or_pr" else a)
    return replace(c, work_auth="citizen_or_pr") if status_problems(c) else c


# ---------------------------------------------------------------- repair

def repair(cand: CandidateCard, card: RequirementCard, rng: random.Random) -> tuple[CandidateCard, str, str] | None:
    """Pull a real candidate just far enough toward `card` to pass every requirement and
    preference. Returns (candidate, expected bucket, note) or None when it cannot be done
    without rewriting them. Expected is Explore only when their work authorization makes
    a clean pass impossible (other visa, OPT with an unstated stance, CPT on a job that is not an
    internship)."""
    c = cand
    fixed: list[str] = []
    if not students_can_pass(card):
        return None
    # education
    need_level, need_enrolled = _base_degree(card)
    if need_enrolled:
        if not (is_enrolled(c) and _rank(c.degree_level) >= _rank(need_level)):
            level = max(need_level, c.degree_level if is_enrolled(c) else need_level, key=_rank)
            cap = CPT_YEARS[level] if c.work_auth != "citizen_or_pr" else 4
            c = replace(c, degree_level=level, graduation_year=_study_year(rng, need_level, c.work_auth),
                        years=max(card.min_years, min(c.years, cap)))
            if c.salary_floor is not None and c.salary_floor > 60_000:  # now a student: a student's floor
                c = replace(c, salary_floor=None if rng.random() < 0.5 else _round_floor(c.salary_floor * 0.5))
            fixed.append("enrollment")
    elif card.degree_level != "none":
        if _rank(held_level(c)) < _rank(card.degree_level):
            c = replace(c, degree_level=max(card.degree_level, c.degree_level, key=_rank))
            c = replace(c, graduation_year=graduated_year(rng, c.years, grad_floor(c.years, pre=pre_degree_years(c)), 2025))
            fixed.append("degree")
    if c.degree_level != "none" and not c.degree_field:
        c = replace(c, degree_field=_field_for(card, rng))
    if c.degree_level != "none" and c.graduation_year is None:
        c = replace(c, graduation_year=graduated_year(rng, c.years, grad_floor(c.years, pre=pre_degree_years(c)), 2025))
    if constrains(card_fields(card)) and (card.degree_level != "none" or card.enrolled_in != "none") \
            and field_match(c.degree_field, card_fields(card)) != "exact":
        f = _field_for(card, rng)
        if not f:
            return None
        c = replace(c, degree_field=f)
        fixed.append("degree field")
    # years
    if c.years < card.min_years:
        c = replace(c, years=card.min_years + rng.randint(0, 3))
        if not is_enrolled(c) and c.graduation_year and c.graduation_year > CURRENT_YEAR - c.years + 3:
            c = replace(c, graduation_year=graduated_year(rng, c.years, 1995, 2025))
        fixed.append("years")
    if overqualified(c, card):
        c = replace(c, years=card.min_years + rng.randint(0, 3))
        fixed.append("years")
    c = fix_status(c, rng)
    c = settle_years(c, rng, card.min_years)
    if c is None:
        return None
    # skills: keep their own, add what the card requires. Only the resume counts here (not the
    # degree field), so a break that changes the degree field never takes a skill with it.
    have = {skill_canon(s) for s in c.skills}
    add = [rng.choice(g) for g in required_groups(card) if not group_met(g, have)]
    nice = [s for s in counted_nice(card) if not group_met((s,), have)]
    rng.shuffle(nice)
    while nice and nice_is_soft(len(nice), len(counted_nice(card))):
        add.append(nice.pop())
    if add:
        own = list(c.skills)
        while len(own) + len(add) > 16:
            drop = [s for s in own if not _meets_card(card, s)]
            if not drop:
                break
            own.remove(rng.choice(drop))
        skills = own[:]
        for s in add:
            skills.insert(rng.randint(0, len(skills)), s)
        c = replace(c, skills=tuple(skills))
        fixed.append("skills")
    if card.license and norm(card.license) not in {norm(x) for x in c.licenses}:
        if not license_fits(card.license, held_level(c), c.degree_field):
            return None  # adding it would give them a license their degree cannot lead to
        c = replace(c, licenses=(*c.licenses, card.license))
        fixed.append("license")
        # the license their degree gives: no year of work before that graduation (pre_degree_years)
        c = settle_years(c, rng, card.min_years)
        if c is None:
            return None
    if license_problems(c):
        return None  # the education fix left a license of theirs without its degree
    # place and work arrangement
    if not location_ok(c, card):
        home = _home(card, rng)
        if rng.random() < 0.5:
            c = replace(c, current_location=home)
        else:
            c = replace(c, willing_locations=(*[x for x in c.willing_locations if x != home], home)[-3:])
        fixed.append("location")
    pref = c.remote_preference
    if (pref == "remote_only" and card.work_mode != "remote") or (pref == "hybrid_or_remote" and card.work_mode == "onsite"):
        c = replace(c, remote_preference="any" if card.work_mode == "onsite" else rng.choice(("any", "hybrid_or_remote")))
        fixed.append("work arrangement")
    # preferences
    if c.employment_types_wanted and card.employment_type not in c.employment_types_wanted:
        types = () if rng.random() < 0.5 else tuple(sorted({*c.employment_types_wanted[:1], card.employment_type}))
        c = replace(c, employment_types_wanted=types)
        fixed.append("job types")
    if not prefs_fit(c):  # now after an internship: "intern" is no longer a role they avoid
        c = _fit_roles(c)
        fixed.append("roles avoided")
    ids = _ind_ids(card)
    if ids & _means(c.industries_avoided):
        c = replace(c, industries_avoided=())
        fixed.append("industries avoided")
    if ids and c.industries_wanted and not ids & _means(c.industries_wanted):
        c = replace(c, industries_wanted=() if rng.random() < 0.4 else (*c.industries_wanted[:2], _ind_own(card)))
        fixed.append("industries wanted")
    if industry_clash(c):  # a kept wanted item the avoided list also names
        c = replace(c, industries_wanted=tuple(x for x in c.industries_wanted
                                               if not industry_ids(x) & _means(c.industries_avoided)))
        fixed.append("industries wanted")
    if role_hits(c, card):
        c = replace(c, roles_avoided=tuple(x for x in c.roles_avoided if x not in role_hits(c, card)))
        fixed.append("roles avoided")
    if c.salary_floor is not None and card.salary_max is not None and c.salary_floor > card.salary_max:
        c = replace(c, salary_floor=_floor_for(card, rng) if rng.random() < 0.7 else None)
        fixed.append("salary floor")
    # work authorization last: it may depend on the education fixed above
    out = auth_outcome(c.work_auth, card)
    if out == "hard":
        alt = with_status(c, "f1_cpt", card, rng) if (
            card.employment_type == "internship" and not card.citizens_only and c.work_auth != "citizen_or_pr"
            and is_enrolled(c)) else None
        c = alt or replace(c, work_auth="citizen_or_pr")
        fixed.append("work authorization")
        out = auth_outcome(c.work_auth, card)
    if status_problems(c) or license_problems(c):
        return None
    return c, ("For you" if out == "pass" else "Explore"), ", ".join(dict.fromkeys(fixed)) or "nothing"


# ---------------------------------------------------------------- breaks

def _role_words_in(card: RequirementCard) -> set[str]:
    return {w for w in AVOID_WORDS if role_hits(CandidateCard(roles_avoided=(w,)), card)}


def _away(card: RequirementCard, rng: random.Random) -> str:
    opts = [m for m in METRO_NAMES if not any(places_match(m, w) for w in posting_places(card.locations))]
    return rng.choice(opts)


def _other_license(rng: random.Random, c: CandidateCard, license: str) -> tuple[str, ...]:
    """In place of `license`: one from the same domain their education fits, else none."""
    need = LICENSE_EDUCATION.get(license)
    near = [x for x in _OTHER_LICENSES if x != license and need and LICENSE_EDUCATION.get(x)
            and LICENSE_EDUCATION[x].domain == need.domain and license_fits(x, held_level(c), c.degree_field)]
    return (rng.choice(near),) if near and rng.random() < 0.6 else ()


def _nice_needed(n: int) -> int | None:
    """Fewest missing nice-to-haves that make a stretch; None when no number does (one listed)."""
    return next((m for m in range(1, n + 1) if nice_is_soft(m, n)), None)


def make_variants(card: RequirementCard, seed: int = 0, base: CandidateCard | None = None,
                  base_kind: str = "copy", extra_pool: tuple[str, ...] | None = None,
                  include_base: bool = True) -> list[Variant]:
    """The base plus every single-edit break that applies to this card. The base must be a
    For you candidate; without one, a copy_base is drawn from the seed."""
    rng = random.Random(f"{seed}:{card.source_posting_id}:{base_kind}")
    if not students_can_pass(card):
        return []  # no For-you base exists: such cards get real candidates only
    if base is None:
        base = copy_base(card, rng, extra_pool)
    out: list[Variant] = [Variant("full_match", base, card, "For you", base=base_kind)] if include_base else []

    def add(kind: str, expected: str, cand: CandidateCard | None = None,
            crd: RequirementCard | None = None, note: str = "") -> None:
        cand = cand or base
        if status_problems(cand) or license_problems(cand) or not prefs_fit(cand) or industry_clash(cand):
            return  # a break must still be a person who could exist
        if unique_skills(cand.skills) != cand.skills:
            cand = replace(cand, skills=unique_skills(cand.skills))
        if cand == base and (crd or card) == card:
            return  # an edit that changed nothing (the base already wanted remote, lived there...)
        out.append(Variant(kind, cand, crd or card, expected, note, base_kind))

    extra = _extra_skills(card, extra_pool)
    held = {skill_canon(s) for s in base.skills}
    unheld = [x for x in extra if skill_canon(x) not in held]  # swap-ins for a removed skill
    enrolled_card = card.enrolled_in != "none"

    # degree level and enrollment
    rank = _rank(card.degree_level)
    if rank > 0:
        lower = DEGREE_LEVELS[rank - 1]
        cand = replace(base, degree_level=lower)
        if lower == "none":
            cand = replace(cand, graduation_year=None, degree_field="")
        elif is_enrolled(base):  # pursuing the lower degree: its own time to finish
            cand = settle_years(replace(cand, graduation_year=_study_year(rng, lower, base.work_auth)), rng)
        else:  # an OPT master's made a bachelor's has fewer years before graduating (pre_degree_years)
            cand = settle_years(cand, rng, card.min_years)
        if cand is not None:
            add("degree_level", "Excluded", cand)
        if not enrolled_card and base.work_auth in ("citizen_or_pr", "other_visa", "f1_cpt"):
            gy = _study_year(rng, card.degree_level, base.work_auth)
            # a CPT student pursuing a lower degree has that degree's CPT years at most (settle_years)
            cand = settle_years(replace(base, degree_level=card.degree_level, graduation_year=gy, work_auth=base.work_auth),
                                rng, card.min_years)
            if cand is not None:
                add("degree_still_enrolled", "Explore", cand, note=f"finishes {gy}")
            if rank < 4 and base.work_auth != "f1_cpt":
                up = DEGREE_LEVELS[rank + 1]
                cand = replace(base, degree_level=up, graduation_year=enrolled_year(rng, up))
                # pursuing the next degree up holds the one asked for, except where it does not
                # (rules.held_level: a bachelor's student holds no associate, a doctoral one no master's)
                if _rank(held_level(cand)) >= rank:
                    add("degree_enrolled_higher", "For you", cand)
    if enrolled_card:
        gy = rng.choice((CURRENT_YEAR - 2, CURRENT_YEAR - 1, CURRENT_YEAR))
        cand = replace(base, graduation_year=gy)
        if base.work_auth == "f1_cpt":
            cand = replace(cand, work_auth="f1_opt", graduation_year=rng.choice((CURRENT_YEAR - 1, CURRENT_YEAR)))
        cand = settle_years(cand, rng)
        if cand is not None:
            add("enrollment_graduated", "Excluded", cand)
        erank = _rank(card.enrolled_in)
        if erank >= 3 or (erank == 2 and base.work_auth in ("citizen_or_pr", "other_visa")):
            lower = DEGREE_LEVELS[erank - 1]
            cand = settle_years(replace(base, degree_level=lower, graduation_year=_study_year(rng, lower, base.work_auth)),
                                rng)
            if cand is not None:
                add("enrollment_level", "Excluded", cand)
    checks_field = constrains(card_fields(card)) and (rank > 0 or enrolled_card)
    if checks_field:
        stem_only = base.work_auth == "stem_opt"
        far = [f for f in ALL_FIELDS if field_match(f, card_fields(card)) == "none"
               and (not stem_only or canon_field(f) in STEM_DEGREE_FIELDS)]
        if far:
            add("degree_field", "Excluded", replace(base, degree_field=rng.choice(far)))
        named, groups, related = parse_fields(card_fields(card))
        near = sorted({x for f in named for x in adjacent_fields(f)})
        near = [_title(x) for x in near if field_match(_title(x), card_fields(card)) != "none"
                and field_match(_title(x), card_fields(card)) != "exact"
                and (not stem_only or x in STEM_DEGREE_FIELDS)]
        if near:
            add("degree_field_adjacent", "Explore", replace(base, degree_field=rng.choice(near)))
        if related:
            fam = sorted({x for f in named for x in adjacent_fields(f)} - set(named))
            fam = [_title(x) for x in fam if field_match(_title(x), card_fields(card)) == "exact"
                   and (not stem_only or x in STEM_DEGREE_FIELDS)]
            if fam:
                add("degree_field_related", "For you", replace(base, degree_field=rng.choice(fam)))

    # years
    def fewer_years(n: int) -> CandidateCard | None:
        return settle_years(replace(base, years=n), rng)  # a graduation year to match

    if card.min_years >= 2:
        for kind, expected, n in (("years_one_short", "Explore", card.min_years - 1),
                                  ("years_far_short", "Excluded", rng.randint(0, card.min_years - 2))):
            cand = fewer_years(n)
            if cand is not None:
                add(kind, expected, cand)
    elif card.min_years == 1:
        cand = fewer_years(0)
        if cand is not None:
            add("years_far_short", "Excluded", cand)
    bar = overqualified_years(base, card)
    oq = bar + rng.randint(0, 4) if bar is not None else 0
    if bar is not None and base.work_auth in ("citizen_or_pr", "other_visa"):
        cand = replace(base, years=oq)
        if base.graduation_year is not None and not is_enrolled(base):
            cand = replace(cand, graduation_year=graduated_year(rng, oq, 1990, 2025))
        add("overqualified", "Explore", cand)

    # skills: swap, rarely delete, so the resume length does not tell the label
    groups = required_groups(card)
    deg = degree_held(base)  # skills the degree field itself shows: deleting them from the resume breaks nothing
    met = [g for g in groups if group_met(g, held) and not group_met(g, deg)]
    if met:
        g = rng.choice(met)
        fresh = iter(rng.sample(unheld, len(unheld)))
        # every skill that meets the group goes, including a language that meets "Programming"
        skills = tuple(next(fresh, "") if any(skill_meets(x, s) for x in g) else s for s in base.skills)
        add("must_skill", "Excluded", replace(base, skills=tuple(s for s in skills if s)), note=f"no {' / '.join(g)}")
        multi = [g for g in met if len(g) > 1]
        if multi:
            g = rng.choice(multi)
            cur = next((s for s in base.skills if skill_canon(s) in {skill_canon(x) for x in g}), None)
            alts = [x for x in g if cur and skill_canon(x) != skill_canon(cur) and skill_canon(x) not in held]
            # never swap out a skill another requirement leans on (Deep learning also meets "Machine learning")
            others = [x for x in _card_skills(card) if skill_canon(x) not in {skill_canon(y) for y in g}]
            if alts and not any(skill_meets(x, cur) for x in others):
                alt = rng.choice(alts)
                add("must_skill_alternative", "For you",
                    replace(base, skills=tuple(alt if s == cur else s for s in base.skills)), note=f"{cur} -> {alt}")
        syn = [(s, a) for s in base.skills for name, alts in SKILL_SYNONYMS.items()
               if skill_canon(s) == skill_canon(name) and skill_canon(s) in {skill_canon(x) for gg in groups for x in gg}
               for a in alts if not a.endswith(" software") and len(a) > 1 and name.lower() not in a.lower()]
        if syn:
            s, a = rng.choice(syn)
            add("must_skill_synonym", "For you", replace(base, skills=tuple(a if x == s else x for x in base.skills)),
                note=f"{s} -> {a}")
    soft_held = [s for s in base.skills if is_soft_skill(s)
                 and skill_canon(s) in {skill_canon(g[0]) for g in card.must_skills if g not in groups}]
    if soft_held:
        add("soft_skill_missing", "For you", replace(base, skills=tuple(s for s in base.skills if s not in soft_held)))
    nice = counted_nice(card)
    if nice:
        missing = nice_gap(base, card)
        fewest = _nice_needed(len(nice))
        need = fewest - len(missing) if fewest is not None else 0
        # nice-to-haves the resume names that nothing else on the card leans on (Python can meet "Programming")
        # and that nothing else they hold also meets (PyTorch meets "Machine learning")
        must_keys = {skill_canon(x) for g in groups for x in g}
        have_nice = [s for s in nice if skill_canon(s) in held and skill_canon(s) not in must_keys
                     and not group_met((s,), deg)
                     and not any(skill_meets(x, s) for x in _card_skills(card) if skill_canon(x) != skill_canon(s))
                     and not any(skill_meets(s, h) for h in base.skills if skill_canon(h) != skill_canon(s))]
        if 0 < need <= len(have_nice):
            gone = {skill_canon(x) for x in rng.sample(have_nice, need)}
            fresh = iter(rng.sample(unheld, len(unheld)))
            skills = tuple(next(fresh, "") if skill_canon(s) in gone else s for s in base.skills)
            add("nice_skill", "Explore", replace(base, skills=tuple(s for s in skills if s)),
                note=f"missing {len(missing) + need} of {len(nice)}")
        if not missing and len(nice) >= 3 and have_nice:
            gone = skill_canon(rng.choice(have_nice))
            add("nice_skill_one_missing", "For you",
                replace(base, skills=tuple(s for s in base.skills if skill_canon(s) != gone)))

    # license: missing it, or holding a nearby one instead (an LPN for an RN posting)
    if card.license:
        add("license", "Excluded", replace(base, licenses=_other_license(rng, base, card.license)))

    # location
    if card.work_mode != "remote" and card.locations and all(parse_place(x).in_us for x in card.locations):
        away = _away(card, rng)
        add("location", "Excluded", replace(base, current_location=away, willing_locations=()))
        add("location_willing", "For you",
            replace(base, current_location=away, willing_locations=(_home(card, rng),)))
        add("location_anywhere", "For you", replace(base, current_location=away, willing_locations=(ANYWHERE_US,)))
        first = posting_places(card.locations)[0]
        p = parse_place(first)
        if p.metro:
            others = [x for x in all_places(p.metro) if place_key(x) != place_key(first)]
            if others:
                add("location_same_metro", "For you",
                    replace(base, current_location=rng.choice(others), willing_locations=()))

    # employment type
    add("employment_type", "Excluded",
        replace(base, employment_types_wanted=tuple(sorted(weighted_sample(rng, TYPE_W, rng.choice((1, 1, 2)),
                                                                           {card.employment_type})))))

    # salary
    if card.salary_max is not None:
        add("salary", "Explore", replace(base, salary_floor=_round_floor(card.salary_max + rng.randint(5_000, 30_000))))

    # preference-only mismatches
    ids = _ind_ids(card)
    if ids:  # an industry outside the list is a mismatch too: no list item names it
        add("pref_industry_wanted", "Explore",
            replace(base, industries_wanted=weighted_sample(rng, INDUSTRY_W, rng.randint(1, 2),
                                                            ids | _means(base.industries_avoided))))
        add("pref_industry_avoided", "Explore",
            replace(base, industries_avoided=(_ind_own(card),),
                    industries_wanted=tuple(x for x in base.industries_wanted if not industry_ids(x) & ids)))
        add("pref_industry_avoided_other", "For you",
            replace(base, industries_avoided=weighted_sample(rng, INDUSTRY_W, 1, ids | _means(base.industries_wanted))))
    # a work mode the profile rules out is an eligibility fail, not a preference (rules._check_location)
    if card.work_mode != "remote":
        add("pref_remote_only", "Excluded", replace(base, remote_preference="remote_only"))
    if card.work_mode == "onsite":
        add("pref_hybrid_or_remote", "Excluded", replace(base, remote_preference="hybrid_or_remote"))
    if card.work_mode == "remote":
        add("pref_remote_match", "For you", replace(base, remote_preference="remote_only"))
    in_title = _role_words_in(card)
    can_avoid = in_title - wanted_role_words(base.employment_types_wanted, base.work_auth)  # no "intern" for an intern-seeker
    if can_avoid:
        add("pref_role_avoided", "Explore", replace(base, roles_avoided=(rng.choice(sorted(can_avoid)),)))
    add("pref_role_avoided_other", "For you",
        replace(base, roles_avoided=weighted_sample(rng, ROLE_W, 1,
                                                    in_title | wanted_role_words(base.employment_types_wanted, base.work_auth))))

    # work authorization (only a base that is a citizen; statuses are swapped in)
    if base.work_auth != "citizen_or_pr":
        return out

    def status(kind: str, st: str, expected: str, crd: RequirementCard | None = None, note: str = "") -> None:
        cand = with_status(base, st, card, rng)
        if cand is not None:
            add(kind, expected, cand, crd, note or st)

    if card.citizens_only:
        for st in rng.sample(("f1_cpt", "f1_opt", "stem_opt", "needs_h1b_now"), 2):
            status(f"work_auth_citizens_only_{st}", st, "Excluded")
        return out
    sponsors = replace(card, sponsorship="sponsors")
    no_sponsor = replace(card, sponsorship="will_not_sponsor")
    no_opt = replace(card, sponsorship="will_not_sponsor_or_opt")
    unstated = replace(card, sponsorship="unstated")
    citizens = replace(card, citizens_only=True, sponsorship="will_not_sponsor"
                       if card.sponsorship == "sponsors" else card.sponsorship)
    if enrolled_card:
        # graduates on OPT or needing H-1B are not students: the enrollment check excludes them
        status("work_auth_opt_student_role", rng.choice(("f1_opt", "stem_opt")), "Excluded")
    else:
        status("work_auth_needs_h1b", "needs_h1b_now", "Excluded", no_sponsor)
        status("work_auth_h1b_sponsors", "needs_h1b_now", "For you", sponsors)
        status("work_auth_h1b_unstated", "needs_h1b_now", "Explore", unstated)
        status("work_auth_f1_opt", "f1_opt", "Explore", unstated)
        status("work_auth_stem_opt", "stem_opt", "Explore", unstated)
        status("work_auth_opt_sponsors", "f1_opt", "For you", sponsors)
        status("work_auth_stem_opt_sponsors", "stem_opt", "For you", sponsors)
        st = rng.choice(("f1_opt", "stem_opt"))
        status("work_auth_no_opt", st, "Excluded", no_opt, note=st)
    if card.employment_type == "internship":
        status("work_auth_cpt_internship", "f1_cpt", "For you")
        status("work_auth_cpt_internship_no_opt", "f1_cpt", "For you", no_opt)
    elif card.employment_type in ("full_time", "contract") and not enrolled_card:
        status("work_auth_cpt_full_time", "f1_cpt", "Explore")
    elif card.employment_type == "part_time" and not enrolled_card:
        status("work_auth_cpt_part_time", "f1_cpt", "Explore")
    add("work_auth_other_visa", "Explore", replace(base, work_auth="other_visa"))
    st = rng.choice(("f1_cpt", "f1_opt", "stem_opt", "needs_h1b_now"))
    status("work_auth_citizens_only", st, "Excluded", citizens, note=st)
    return out
