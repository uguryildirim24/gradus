"""Seeded samplers: RequirementCards from O*NET, CandidateCards with a controlled mix.

Everything is deterministic for a given seed and the same O*NET files.
Cards carry no free text beyond titles and plain skill names; Adaption
renders the posting and resume prose later, joined by id.
"""

from __future__ import annotations

import random
from collections import Counter

from .metros import ANYWHERE_US, METRO_CENTERS, METRO_IDS, METRO_WEIGHTS, METROS, STATE_NAMES, place_key, \
    places_match, state_place
from .onet import LICENSE_EDUCATION, Occupation, license_fits
from .schema import CURRENT_YEAR, CandidateCard, RequirementCard, is_soft_skill, skill_canon

# ---------------------------------------------------------------- fields

# field label -> (sampling weight, SOC matcher, industries, acceptable degree fields)
FIELDS: dict[str, dict] = {
    "Nursing": dict(w=6, ind=("healthcare",), deg=("Nursing", "Health Sciences")),
    "Pharmacy": dict(w=3, ind=("healthcare", "retail"),
                     deg=("Pharmacy", "Pharmaceutical Sciences", "Biology", "Chemistry")),
    "Clinical healthcare": dict(w=9, ind=("healthcare",),
                                deg=("Health Sciences", "Biology", "Biochemistry", "Public Health", "Kinesiology", "Psychology")),
    "Lab and life sciences": dict(w=11, ind=("biotech and pharma", "research", "healthcare"),
                                  deg=("Biology", "Biochemistry", "Chemistry", "Biomedical Science", "Molecular Biology", "Microbiology", "Neuroscience", "Physics")),
    "Social sciences": dict(w=3, ind=("research", "government"),
                            deg=("Psychology", "Sociology", "Economics", "Political Science", "Anthropology", "Statistics")),
    "Business and finance": dict(w=9, ind=("finance", "consulting", "technology"),
                                 deg=("Finance", "Accounting", "Business Administration", "Economics", "Marketing", "Management")),
    "Management": dict(w=3, ind=("finance", "retail", "manufacturing", "healthcare"),
                       deg=("Business Administration", "Management", "Economics", "Industrial Engineering")),
    "Sales and administration": dict(w=5, ind=("retail", "finance", "technology"),
                                     deg=("Business Administration", "Marketing", "Communications", "Management")),
    "Education": dict(w=8, ind=("education",),
                      deg=("Education", "Elementary Education", "English", "Mathematics", "Psychology", "Special Education")),
    "Social services": dict(w=6, ind=("nonprofit", "government", "healthcare"),
                            deg=("Social Work", "Psychology", "Sociology", "Human Services", "Public Health", "Counseling")),
    "Design and media": dict(w=6, ind=("media", "technology", "retail"),
                             deg=("Graphic Design", "Communications", "Journalism", "Film", "Fine Arts", "Marketing", "English")),
    "Protective services and forensics": dict(w=6, ind=("government", "public safety"),
                                              deg=("Criminal Justice", "Forensic Science", "Chemistry", "Biology", "Political Science", "Psychology")),
    "Trades and construction": dict(w=6, ind=("construction", "manufacturing", "utilities"),
                                    deg=("Construction Management", "Industrial Technology", "Engineering Technology", "Electrical Technology")),
    "Hospitality and personal care": dict(w=4, ind=("hospitality", "retail"),
                                          deg=("Hospitality Management", "Business Administration", "Culinary Arts", "Nutrition", "Kinesiology")),
    "IT and data": dict(w=15, ind=("technology", "finance", "healthcare"),
                        deg=("Computer Science", "Information Systems", "Data Science", "Statistics", "Mathematics", "Software Engineering", "Cybersecurity")),
    "Engineering": dict(w=8, ind=("manufacturing", "technology", "construction", "utilities"),
                        deg=("Mechanical Engineering", "Electrical Engineering", "Civil Engineering", "Chemical Engineering", "Industrial Engineering", "Biomedical Engineering", "Physics")),
    "Legal": dict(w=1.5, ind=("legal", "government"),
                  deg=("Political Science", "Criminal Justice", "English", "Philosophy", "Paralegal Studies", "History")),
    "Manufacturing and logistics": dict(w=2.5, ind=("manufacturing", "logistics"),
                                        deg=("Industrial Technology", "Supply Chain Management", "Engineering Technology", "Business Administration")),
    "Agriculture and environment": dict(w=2, ind=("agriculture", "environmental services"),
                                        deg=("Environmental Science", "Agriculture", "Biology", "Geology", "Forestry")),
}
STEM_FIELDS = frozenset({"Lab and life sciences", "IT and data", "Engineering", "Agriculture and environment"})
FIELD_NAMES = tuple(FIELDS)

# Median-ish yearly pay by job zone, scaled by field.
_ZONE_PAY = {1: 30000, 2: 36000, 3: 48000, 4: 70000, 5: 95000}
_FIELD_PAY = {
    "IT and data": 1.35, "Engineering": 1.25, "Business and finance": 1.15, "Pharmacy": 1.3,
    "Nursing": 1.2, "Clinical healthcare": 1.1, "Legal": 1.3, "Management": 1.3,
    "Lab and life sciences": 1.0, "Trades and construction": 1.05, "Education": 0.85,
    "Social services": 0.8, "Hospitality and personal care": 0.75, "Design and media": 0.9,
    "Social sciences": 0.95, "Sales and administration": 0.9,
    "Protective services and forensics": 0.95, "Manufacturing and logistics": 0.9,
    "Agriculture and environment": 0.85,
}
_ONSITE_BIAS = {  # probability the work mode is onsite
    "Nursing": 0.97, "Pharmacy": 0.95, "Clinical healthcare": 0.93, "Trades and construction": 0.95,
    "Hospitality and personal care": 0.95, "Protective services and forensics": 0.92,
    "Manufacturing and logistics": 0.92, "Agriculture and environment": 0.8, "Education": 0.8,
    "Social services": 0.65, "Lab and life sciences": 0.8, "Engineering": 0.55,
}
_INTERN_RATE = {
    "IT and data": 0.38, "Business and finance": 0.38, "Lab and life sciences": 0.33,
    "Design and media": 0.35, "Engineering": 0.35, "Social sciences": 0.35,
    "Sales and administration": 0.25, "Management": 0.15, "Legal": 0.25,
    "Social services": 0.2, "Education": 0.12, "Clinical healthcare": 0.12, "Pharmacy": 0.15,
    "Protective services and forensics": 0.14, "Agriculture and environment": 0.2,
    "Hospitality and personal care": 0.12, "Nursing": 0.05, "Manufacturing and logistics": 0.15,
    "Trades and construction": 0.0,
}
_CITIZENS_ONLY = {"Protective services and forensics": 0.12, "Engineering": 0.06, "IT and data": 0.04}

# Posting and candidate places: metro centers by weight, often a suburb (metros.py).
METRO_NAMES = METRO_CENTERS
METRO_W = METRO_WEIGHTS
SUBURB_RATE = 0.3  # share of places drawn as a suburb instead of the metro's center
STATE_ONLY_RATE = 0.06  # share of onsite/hybrid postings that name only a state

INDUSTRIES = tuple(sorted({i for f in FIELDS.values() for i in f["ind"]}))
# How often each industry is on a sampled posting: random industry picks in variants follow it,
# so a wanted or avoided industry name alone does not tell whether it matched.
INDUSTRY_W = {i: sum(f["w"] / len(f["ind"]) for f in FIELDS.values() if i in f["ind"]) for i in INDUSTRIES}
# How often each employment type is on a sampled posting (for random picks, same reason).
TYPE_W = {"full_time": 50, "internship": 33, "part_time": 9, "contract": 8}


def weighted_sample(rng: random.Random, weights: dict[str, float], k: int, exclude=()) -> tuple[str, ...]:
    """k distinct items drawn by weight, none in exclude."""
    items = {x: w for x, w in weights.items() if x not in exclude}
    out: list[str] = []
    while items and len(out) < k:
        x = rng.choices(list(items), weights=list(items.values()))[0]
        out.append(x)
        del items[x]
    return tuple(out)

# ---------------------------------------------------------------- candidates

# country -> (weight among international, weight among foreign-educated citizens, languages)
COUNTRIES: dict[str, tuple[int, int, tuple[str, ...]]] = {
    "India": (22, 5, ("Hindi", "Telugu", "Tamil", "Gujarati")),
    "China": (18, 4, ("Mandarin",)),
    "South Korea": (5, 1, ("Korean",)),
    "Nigeria": (5, 3, ("Yoruba", "Igbo", "English")),
    "Brazil": (4, 1, ("Portuguese",)),
    "Canada": (5, 1, ("English", "French")),
    "Nepal": (4, 1, ("Nepali",)),
    "Vietnam": (3, 4, ("Vietnamese",)),
    "Turkey": (3, 1, ("Turkish",)),
    "Mexico": (3, 8, ("Spanish",)),
    "Germany": (2, 0, ("German",)),
    "Saudi Arabia": (2, 0, ("Arabic",)),
    "Iran": (2, 1, ("Persian",)),
    "Bangladesh": (3, 1, ("Bengali",)),
    "Pakistan": (3, 1, ("Urdu",)),
    "Japan": (2, 0, ("Japanese",)),
    "Ghana": (2, 1, ("English", "Twi")),
    "Kenya": (2, 1, ("Swahili", "English")),
    "Colombia": (2, 1, ("Spanish",)),
    "Philippines": (2, 6, ("Tagalog", "English")),
    "United Kingdom": (2, 0, ("English",)),
    "Taiwan": (2, 0, ("Mandarin",)),
    "France": (1, 0, ("French",)),
    "Italy": (1, 0, ("Italian",)),
    "Egypt": (2, 1, ("Arabic",)),
    "Ukraine": (1, 1, ("Ukrainian", "Russian")),
    "Russia": (1, 1, ("Russian",)),
    "Indonesia": (2, 0, ("Indonesian",)),
    "Spain": (1, 0, ("Spanish",)),
    "Peru": (1, 1, ("Spanish",)),
    "Ethiopia": (1, 2, ("Amharic",)),
    "Jamaica": (1, 3, ("English",)),
    "Australia": (1, 0, ("English",)),
    "Argentina": (1, 0, ("Spanish",)),
    "Thailand": (1, 0, ("Thai",)),
    "Poland": (1, 0, ("Polish",)),
    "Israel": (1, 0, ("Hebrew",)),
    "Sri Lanka": (1, 0, ("Sinhala",)),
    "South Africa": (1, 0, ("English", "Afrikaans")),
    "Uzbekistan": (1, 0, ("Uzbek", "Russian")),
    "Kazakhstan": (1, 0, ("Russian", "Kazakh")),
    "Dominican Republic": (0, 5, ("Spanish",)),
    "Cuba": (0, 4, ("Spanish",)),
    "Haiti": (0, 3, ("Haitian Creole", "French")),
    "El Salvador": (0, 3, ("Spanish",)),
}
INTL_AUTH = (("f1_cpt", 8), ("f1_opt", 10), ("stem_opt", 7), ("needs_h1b_now", 6), ("other_visa", 4))
INTL_SHARE = 0.35
US_LANGUAGES = (("English", 80), ("Spanish", 12), ("Mandarin", 2), ("Vietnamese", 1), ("Arabic", 1),
                ("Tagalog", 1), ("Korean", 1), ("French", 1), ("Hindi", 1))
EXTRA_LICENSE_RATE = 0.0
# Role words people rule out ("no sales jobs"); the rule matches them as whole words in the title.
# Weights follow how often each word is in a sampled posting title, so an avoided word is about as
# likely to hit the title (Explore) as to miss it (no effect): the word alone tells nothing.
ROLE_W = {"intern": 40, "technician": 9, "analyst": 6, "teacher": 5, "manager": 4, "assistant": 3, "aide": 2,
          "clerk": 2, "sales": 2, "operator": 2, "driver": 1, "representative": 1, "supervisor": 1,
          "cashier": 1, "telemarketer": 1}
AVOID_WORDS = tuple(ROLE_W)


def wanted_role_words(types: tuple[str, ...], work_auth: str = "") -> set[str]:
    """Role words a candidate who wants these job types cannot also avoid: someone after an
    internship does not rule out "intern" titles, and neither does an F-1 student on CPT, whose
    status covers internships and co-ops only."""
    return {"intern"} if "internship" in types or work_auth == "f1_cpt" else set()


def _wchoice(rng: random.Random, items, weights):
    return rng.choices(items, weights=weights, k=1)[0]


def field_of(occ: Occupation) -> str:
    """Our field label for an O*NET occupation, by SOC prefix with a few overrides."""
    s = occ.soc
    if s.startswith(("29-1141", "29-1171", "29-1151", "29-2061", "31-1")):
        return "Nursing"
    if s.startswith(("29-1051", "29-2052", "29-2051")):
        return "Pharmacy"
    if s.startswith("19-4092"):
        return "Protective services and forensics"
    if s.startswith(("19-1", "19-2", "19-4")):
        return "Lab and life sciences" if not s.startswith(("19-1031", "19-1032", "19-1099")) else "Agriculture and environment"
    major = s[:2]
    table = {
        "29": "Clinical healthcare", "31": "Clinical healthcare", "19": "Social sciences",
        "13": "Business and finance", "11": "Management", "41": "Sales and administration",
        "43": "Sales and administration", "25": "Education", "21": "Social services",
        "27": "Design and media", "33": "Protective services and forensics",
        "47": "Trades and construction", "49": "Trades and construction",
        "35": "Hospitality and personal care", "37": "Hospitality and personal care",
        "39": "Hospitality and personal care", "15": "IT and data", "17": "Engineering",
        "23": "Legal", "51": "Manufacturing and logistics", "53": "Manufacturing and logistics",
        "45": "Agriculture and environment",
    }
    return table.get(major, "")


def group_by_field(occs: list[Occupation]) -> dict[str, list[Occupation]]:
    out: dict[str, list[Occupation]] = {f: [] for f in FIELD_NAMES}
    for o in occs:
        f = field_of(o)
        if f in out:
            out[f].append(o)
    return out


def _quotas(total: int) -> dict[str, int]:
    wsum = sum(f["w"] for f in FIELDS.values())
    q = {n: int(total * f["w"] / wsum) for n, f in FIELDS.items()}
    rest = total - sum(q.values())
    for n in sorted(FIELDS, key=lambda n: -FIELDS[n]["w"])[:rest]:
        q[n] += 1
    return q


def _cycle(rng: random.Random, items: list[Occupation], n: int) -> list[Occupation]:
    """n occupations, shuffled passes so each is used before any repeats."""
    out: list[Occupation] = []
    while len(out) < n:
        batch = items[:]
        rng.shuffle(batch)
        out.extend(batch[: n - len(out)])
    return out


# ---------------------------------------------------------------- requirement cards


def singular(title: str) -> str:
    """O*NET titles are plural ("Budget Analysts"); posting titles read singular."""
    if any(t in title for t in (" and ", " or ", "&", "/")):
        return title
    head, sep, tail = title.partition(",")
    words = head.split(" ")
    w = words[-1]
    if w.endswith("ies"):
        w = w[:-3] + "y"
    elif w.endswith("wives"):  # Midwives
        w = w[:-3] + "fe"
    elif w.endswith(("sses", "xes", "ches", "shes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is", "ics")):
        w = w[:-1]
    words[-1] = w
    return " ".join(words) + sep + tail


# Concepts real postings in these fields ask for that O*NET has no element for.
CONCEPTS = {
    "IT and data": ("Data structures", "Algorithms", "Object-oriented programming", "Machine learning",
                    "Deep learning", "Statistics", "Data analysis", "REST APIs", "Distributed systems",
                    "Cloud computing", "Unit testing", "Data visualization", "Natural language processing",
                    "Computer networking", "Operating systems", "Relational databases", "Agile", "Linear algebra",
                    "PyTorch", "TensorFlow", "Pandas", "Spark", "Docker", "Git"),
    "Engineering": ("Circuit design", "PCB design", "Embedded systems", "Signal processing", "CAD",
                    "Finite element analysis", "Thermodynamics", "Control systems", "GD&T", "Fluid mechanics",
                    "Digital logic", "Verilog", "VHDL", "FPGA", "Lab testing", "Root cause analysis", "Simulink",
                    "SolidWorks", "Python", "C"),
    "Lab and life sciences": ("PCR", "Cell culture", "Western blot", "ELISA", "Flow cytometry", "Microscopy",
                              "Aseptic technique", "Good Laboratory Practice", "HPLC", "Mass spectrometry",
                              "DNA extraction", "Gel electrophoresis", "Statistical analysis", "R", "Python"),
    "Business and finance": ("Financial modeling", "Valuation", "Financial analysis", "Forecasting", "GAAP",
                             "Budgeting", "Data analysis", "Market research", "Power BI", "Tableau", "SQL"),
    "Social sciences": ("Survey design", "Statistical analysis", "Qualitative research", "R", "Stata", "SPSS"),
    "Design and media": ("Figma", "Typography", "UX research", "Video editing", "Copywriting", "Social media"),
}
SOFT_MUST = ("Communication", "Problem solving", "Teamwork", "Attention to detail", "Time management",
             "Written and verbal communication", "Analytical skills", "Leadership", "Collaboration")
SOFT_MUST_RATE = 0.2  # share of postings that also "require" a soft skill (the rule ignores it)
# Products and the suite they ship in: never an any-of pair ("Excel or Microsoft Office").
SUITE = frozenset({"Microsoft Excel", "Microsoft Word", "Microsoft PowerPoint", "Microsoft Outlook",
                   "Microsoft Access", "Microsoft Office", "Microsoft Project", "Microsoft Visio"})
TECH_FIELDS = frozenset({"IT and data", "Engineering"})


def skill_pool(occ: Occupation, fld: str) -> list[str]:
    """Concrete skills for an occupation, best first: tech fields lead with hot technologies
    and field concepts, everyone else with knowledge areas and tools. Never abstract skills."""
    from .onet import ABSTRACT_SKILLS

    if fld in TECH_FIELDS:
        lists = [occ.tools, list(CONCEPTS.get(fld, ())), occ.core_knowledge]
    else:
        lists = [occ.core_knowledge, occ.tools, list(CONCEPTS.get(fld, ())), occ.knowledge]
    out: list[str] = []
    seen: set[str] = set()
    for i in range(max(len(x) for x in lists)):
        for lst in lists:
            if i < len(lst) and skill_canon(lst[i]) not in seen and lst[i] not in ABSTRACT_SKILLS \
                    and not is_soft_skill(lst[i]):
                out.append(lst[i])
                seen.add(skill_canon(lst[i]))
    return out


def _n_must(rng: random.Random, fld: str) -> int:
    if fld in TECH_FIELDS:
        return _wchoice(rng, (0, 1, 2, 3, 4, 5, 6), (2, 6, 16, 26, 24, 16, 10))
    return _wchoice(rng, (0, 1, 2, 3, 4, 5), (3, 9, 30, 33, 18, 7))


def _pick_skills(rng: random.Random, occ: Occupation, fld: str) -> tuple[list[tuple[str, ...]], list[str]]:
    """Must-skill groups and nice skills.

    Must-haves are concrete: the occupation's core knowledge areas, its software (hot
    technologies for tech fields) and field concepts. A required tool sometimes comes
    with a peer tool as an any-of group ("Python or R"), never with its own suite.
    Some postings also "require" a soft skill (Communication, ...), as real ones do;
    the rule ignores those. No alternative ever reappears in nice."""
    core = occ.core_knowledge
    tools = [x for x in occ.tools if x not in SUITE] + [x for x in occ.tools if x in SUITE]
    concepts = list(CONCEPTS.get(fld, ()))
    n_must = _n_must(rng, fld)
    if fld in TECH_FIELDS:
        sources = [(tools[:8], 5), (concepts, 3), (core[:2], 1)]
    else:
        sources = [(core[:4], 4), (tools[:5], 3), (concepts, 1)]
    must: list[str] = []
    taken: set[str] = set()
    tries = 0
    while len(must) < n_must and tries < 40:
        tries += 1
        opts = [(lst, w) for lst, w in sources if any(skill_canon(x) not in taken for x in lst)]
        if not opts:
            break
        lst = _wchoice(rng, [o[0] for o in opts], [o[1] for o in opts])
        name = rng.choice([x for x in lst[:8] if skill_canon(x) not in taken])
        must.append(name)
        taken.add(skill_canon(name))
        if name in SUITE:  # one Office product at most
            taken |= {skill_canon(x) for x in SUITE}
    groups: list[tuple[str, ...]] = []
    for name in must:
        peers = [x for x in tools[:8] + concepts if skill_canon(x) not in taken
                 and not (x in SUITE and name in SUITE)]
        if (name in tools or name in concepts) and peers and rng.random() < 0.35:
            alt = rng.choice(peers)
            taken.add(skill_canon(alt))
            groups.append((name, alt))
        else:
            groups.append((name,))
    if rng.random() < SOFT_MUST_RATE:
        groups.append((rng.choice(SOFT_MUST),))
    rest = [x for x in skill_pool(occ, fld)[:20] if skill_canon(x) not in taken]
    k = min(len(rest), _wchoice(rng, (0, 1, 2, 3, 4, 5, 6), (5, 15, 25, 25, 15, 10, 5)))
    nice = rng.sample(rest, k) if k else []
    if rng.random() < 0.25:
        nice.append(rng.choice([x for x in SOFT_MUST if all(x != g[0] for g in groups)]))
    return groups, nice


def _degree_for(rng: random.Random, occ: Occupation, employment: str) -> tuple[str, str]:
    """(degree_level held, enrolled_in). Graduate-level internships are often for
    current master's or PhD students; undergraduate ones for current students."""
    level = occ.degree_level
    if employment == "internship":
        if level in ("master", "doctoral"):
            if rng.random() < 0.5:
                return "bachelor", level
            level = "bachelor"
        held = "none" if rng.random() < 0.55 else level
        if held == "none" and rng.random() < 0.4:
            return "none", "bachelor"
        return held, "none"
    if level == "none" and occ.job_zone >= 3 and rng.random() < 0.3:
        return "associate", "none"
    return level, "none"


def _student_degree(rng: random.Random) -> tuple[str, str]:
    """(degree_level held, enrolled_in) for the student slice."""
    enrolled = _wchoice(rng, ("associate", "bachelor", "master", "doctoral"), (4, 64, 22, 10))
    held = {
        "associate": "none",
        "bachelor": "none",  # no bachelor's student is credited with an associate degree (rules.held_level)
        "master": "bachelor" if rng.random() < 0.7 else "none",
        # never "master": a doctoral student is not credited with one (rules.held_level)
        "doctoral": "none" if rng.random() < 0.25 else "bachelor",
    }[enrolled]
    return held, enrolled


def _seniority(years: int) -> str:
    return "entry" if years <= 1 else "mid" if years <= 4 else "senior" if years <= 8 else "lead"


def _salary(rng: random.Random, occ: Occupation, fld: str, employment: str, years: int):
    if rng.random() < 0.25:
        return None, None
    base = _ZONE_PAY[occ.job_zone] * _FIELD_PAY.get(fld, 1.0)
    base *= 1 + 0.05 * years
    if employment == "internship":
        base = rng.uniform(34000, 62000) * min(1.2, _FIELD_PAY.get(fld, 1.0))
    elif employment == "part_time":
        base *= 0.6
    lo = int(base * rng.uniform(0.88, 1.12) / 1000) * 1000
    hi = int(lo * rng.uniform(1.12, 1.4) / 1000) * 1000
    return lo, hi


def sample_place(rng: random.Random, suburb_rate: float = SUBURB_RATE) -> str:
    """A metro center by weight, or one of its suburbs."""
    mid = _wchoice(rng, METRO_IDS, METRO_WEIGHTS)
    center, _, subs = METROS[mid]
    return rng.choice(subs) if subs and rng.random() < suburb_rate else center


def state_name(place: str) -> str:
    """The whole state a place is in, as a card writes it ("California", "Washington State")."""
    return state_place(place_key(place)[1])


def _locations(rng: random.Random, mode: str) -> tuple[str, ...]:
    if mode == "remote":
        return ()
    if rng.random() < STATE_ONLY_RATE:
        place = sample_place(rng)
        if place_key(place)[1] in STATE_NAMES:
            return (state_name(place),)
    n = 1 if rng.random() < 0.8 else 2
    locs: list[str] = []
    while len(locs) < n:
        m = sample_place(rng)
        if m not in locs:
            locs.append(m)
    return tuple(locs)


# Real-posting wordings of a vocabulary industry (rules.INDUSTRY_SYNONYMS maps them back),
# and a few industries outside the vocabulary: no candidate's wanted list names one unless they
# list it by its own name, so a candidate who lists wanted industries is a mismatch there.
INDUSTRY_WORDING = {
    "finance": ("Financial Services", "Banking", "Insurance", "Asset Management"),
    "technology": ("Software", "Semiconductors", "Information Technology", "Cloud Computing"),
    "healthcare": ("Hospital & Health Care", "Health Care", "Medical Devices"),
    "biotech and pharma": ("Biotechnology", "Pharmaceuticals", "Life Sciences"),
    "education": ("Higher Education", "K-12 Education"),
    "manufacturing": ("Automotive", "Industrial Manufacturing", "Chemicals"),
    "consulting": ("Management Consulting", "Professional Services"),
    "logistics": ("Transportation", "Supply Chain"),
    "utilities": ("Energy", "Renewable Energy"),
    "media": ("Media & Entertainment", "Publishing", "Advertising"),
    "government": ("Public Sector", "Federal Government"),
    "retail": ("Consumer Goods", "E-commerce"),
}
OUTSIDE_INDUSTRIES = ("Aerospace & Defense", "Real Estate", "Sports", "Architecture & Planning")
INDUSTRY_WORDING_RATE = 0.12
OUTSIDE_INDUSTRY_RATE = 0.03


def _industry(rng: random.Random, fld: str) -> str:
    ind = rng.choice(FIELDS[fld]["ind"])
    r = rng.random()
    if r < OUTSIDE_INDUSTRY_RATE:
        return rng.choice(OUTSIDE_INDUSTRIES)
    if r < OUTSIDE_INDUSTRY_RATE + INDUSTRY_WORDING_RATE and ind in INDUSTRY_WORDING:
        return rng.choice(INDUSTRY_WORDING[ind])
    return ind


RELATED_TAILS = ("related field", "a related technical field", "equivalent")


def _degree_fields(rng: random.Random, fld: str, p: float) -> tuple[str, ...]:
    if rng.random() >= p:
        return ()
    pool = FIELDS[fld]["deg"]
    out = list(rng.sample(pool, min(len(pool), rng.randint(2, 4))))
    if rng.random() < 0.3:
        out.append(rng.choice(RELATED_TAILS))
    return tuple(out)


def _license_degree_fields(rng: random.Random, license: str, p: float) -> tuple[str, ...]:
    """Degree fields a licensed posting names: the fields its license follows from (a physician
    posting asks for Medicine, never Psychology). Empty when the license takes any field."""
    need = LICENSE_EDUCATION.get(license)
    if need is None or need.fields is None or rng.random() >= p:
        return ()
    return tuple(rng.sample(need.fields, min(len(need.fields), rng.randint(1, 3))))


def _sponsorship(rng: random.Random, fld: str, occ: Occupation, citizens_only: bool) -> str:
    if fld in ("IT and data", "Engineering", "Lab and life sciences", "Business and finance"):
        w = (30, 20, 5, 45)
    elif occ.job_zone <= 2:
        w = (3, 37, 3, 57)
    else:
        w = (15, 26, 4, 55)
    s = _wchoice(rng, ("sponsors", "will_not_sponsor", "will_not_sponsor_or_opt", "unstated"), w)
    return "will_not_sponsor" if citizens_only and s == "sponsors" else s


def _title(occ: Occupation, employment: str, seniority: str, licensed: bool, rng: random.Random) -> str:
    title = singular(occ.title)
    if employment == "internship":
        return f"{title} Intern"
    if seniority == "senior" and not licensed:
        return f"Senior {title}"
    if seniority == "lead" and not licensed:
        return f"Lead {title}"
    if seniority == "entry" and occ.job_zone >= 4 and rng.random() < 0.3:
        return f"Junior {title}"
    return title


def _work_mode(rng: random.Random, fld: str, employment: str) -> str:
    p_on = _ONSITE_BIAS.get(fld, 0.5)
    mode = _wchoice(rng, ("onsite", "hybrid", "remote"), (p_on, (1 - p_on) * 0.6, (1 - p_on) * 0.4))
    if employment == "internship" and mode == "onsite" and rng.random() < 0.2:
        mode = "hybrid"
    return mode


def make_requirement_card(rng: random.Random, occ: Occupation, fld: str, serial: str) -> RequirementCard:
    licensed = occ.license is not None
    # employment type
    if occ.job_zone >= 3 and not licensed and rng.random() < _INTERN_RATE.get(fld, 0.15):
        employment = "internship"
    else:
        employment = _wchoice(rng, ("full_time", "part_time", "contract"), (78, 10, 12))
        if occ.job_zone <= 2:
            employment = _wchoice(rng, ("full_time", "part_time", "contract"), (60, 32, 8))
    # years, seniority
    if employment == "internship":
        years = 0
    else:
        years = occ.typical_years
        if rng.random() < 0.35:
            years = min(years, rng.choice((0, 1)))
        elif rng.random() < 0.3:
            years = max(0, years + rng.choice((-1, 0, 1, 2)))
        if occ.job_zone >= 4 and rng.random() < 0.25:
            years += rng.randint(2, 6)
        if licensed and years == 0 and rng.random() < 0.3:
            years = 1
    seniority = _seniority(years)
    degree, enrolled = _degree_for(rng, occ, employment)
    if licensed and employment != "internship":
        need = LICENSE_EDUCATION.get(occ.license)
        if need and DEGREE_ORDER.index(need.level) > DEGREE_ORDER.index(degree):
            degree = need.level  # an RN posting asks for at least the degree an RN holds
        degree_fields = _license_degree_fields(rng, occ.license, 0.6) if degree != "none" else ()
    else:
        degree_fields = _degree_fields(rng, fld, 0.6) if degree in ("bachelor", "master", "doctoral") else ()
    if enrolled != "none" and not degree_fields:
        degree_fields = _degree_fields(rng, fld, 0.5)
    must, nice = _pick_skills(rng, occ, fld)
    mode = _work_mode(rng, fld, employment)
    c_only = rng.random() < _CITIZENS_ONLY.get(fld, 0.01)
    lo, hi = _salary(rng, occ, fld, employment, years)
    return RequirementCard(
        source_posting_id=f"onet:{occ.soc}:{serial}",
        title=_title(occ, employment, seniority, licensed, rng),
        field=fld,
        onet_soc=occ.soc,
        seniority=seniority,
        industry=_industry(rng, fld),
        degree_level=degree,
        enrolled_in=enrolled,
        degree_fields=degree_fields,
        min_years=years,
        must_skills=tuple(must),
        nice_skills=tuple(nice),
        license=occ.license if (licensed and (employment != "internship")) else None,
        locations=_locations(rng, mode),
        work_mode=mode,
        sponsorship=_sponsorship(rng, fld, occ, c_only),
        citizens_only=c_only,
        e_verify=rng.random() < 0.18,
        salary_min=lo,
        salary_max=hi,
        employment_type=employment,
    )


# The student slice: internships for current students, weighted toward the fields real
# internship postings come from. A quarter are citizens-only (ITAR, clearance, defense).
STUDENT_SHARE = 0.18
STUDENT_FIELD_W = {
    "IT and data": 30, "Engineering": 20, "Lab and life sciences": 12, "Business and finance": 12,
    "Social sciences": 4, "Design and media": 5, "Clinical healthcare": 3, "Management": 3,
    "Sales and administration": 2, "Agriculture and environment": 3, "Protective services and forensics": 2,
    "Education": 2, "Legal": 2,
}
STUDENT_CITIZENS_ONLY = 0.25


def make_student_card(rng: random.Random, occ: Occupation, fld: str, serial: str) -> RequirementCard:
    held, enrolled = _student_degree(rng)
    must, nice = _pick_skills(rng, occ, fld)
    if rng.random() < 0.12:  # many student postings ask for little beyond enrollment
        must = must[: rng.randint(0, 1)]
    mode = _work_mode(rng, fld, "internship")
    c_only = rng.random() < STUDENT_CITIZENS_ONLY
    lo, hi = _salary(rng, occ, fld, "internship", 0)
    w = (20, 25, 10, 45)
    stance = _wchoice(rng, ("sponsors", "will_not_sponsor", "will_not_sponsor_or_opt", "unstated"), w)
    if c_only and stance == "sponsors":
        stance = "will_not_sponsor"
    return RequirementCard(
        source_posting_id=f"onet:{occ.soc}:s{serial}",
        title=f"{singular(occ.title)} Intern",
        field=fld,
        onet_soc=occ.soc,
        seniority="entry",
        industry=_industry(rng, fld),
        degree_level=held,
        enrolled_in=enrolled,
        degree_fields=_degree_fields(rng, fld, 0.7),
        min_years=0,
        must_skills=tuple(must),
        nice_skills=tuple(nice),
        license=None,
        locations=_locations(rng, mode),
        work_mode=mode,
        sponsorship=stance,
        citizens_only=c_only,
        e_verify=rng.random() < 0.18,
        salary_min=lo,
        salary_max=hi,
        employment_type="internship",
    )


def sample_requirement_cards(occs: list[Occupation], total: int = 1500, seed: int = 20261017):
    """[(id, card)] with ids rq_00001...; field mix follows FIELDS weights, plus a
    STUDENT_SHARE slice of student internships."""
    rng = random.Random(seed)
    groups = group_by_field(occs)
    n_student = round(total * STUDENT_SHARE)
    quotas = _quotas(total - n_student)
    picks: list[tuple[str, Occupation, bool]] = []
    for fld in FIELD_NAMES:
        for occ in _cycle(rng, groups[fld], quotas[fld]):
            picks.append((fld, occ, False))
    wsum = sum(STUDENT_FIELD_W.values())
    sq = {f: int(n_student * w / wsum) for f, w in STUDENT_FIELD_W.items()}
    for f in sorted(STUDENT_FIELD_W, key=lambda f: -STUDENT_FIELD_W[f])[: n_student - sum(sq.values())]:
        sq[f] += 1
    for fld in sorted(STUDENT_FIELD_W):
        pool = [o for o in groups[fld] if o.job_zone >= 3] or groups[fld]
        for occ in _cycle(rng, pool, sq[fld]):
            picks.append((fld, occ, True))
    rng.shuffle(picks)
    seen: Counter[str] = Counter()
    out = []
    for i, (fld, occ, student) in enumerate(picks, start=1):
        seen[occ.soc] += 1
        make = make_student_card if student else make_requirement_card
        out.append((f"rq_{i:05d}", make(rng, occ, fld, f"{seen[occ.soc]}")))
    return out


# ---------------------------------------------------------------- candidate cards

DEGREE_ORDER = ("none", "associate", "bachelor", "master", "doctoral")
# Most years of work an F-1 student on CPT plausibly has, by the degree they are pursuing
# (master's and doctoral students often worked before coming).
CPT_YEARS = {"associate": 2, "bachelor": 2, "master": 5, "doctoral": 6}
# CPT comes after a year of study: an F-1 student on CPT finishes at most this many years from
# now (a bachelor's student past the first year). Master's and doctoral students apply in their
# first year for the summer after it, so they have no cap.
CPT_MAX_AHEAD = {"associate": 1, "bachelor": 3}
_YEAR_W = [18, 12, 11, 9, 8, 7, 5, 4, 4, 3, 3] + [1.5] * 10  # years 0..20
STUDENT_RATE = 0.22  # share of citizen candidates still enrolled
SOFT_ON_RESUME = 0.3  # share of candidates who also list a soft skill or two


def _skills_for(rng: random.Random, occ: Occupation, fld: str, field_pool: list[str],
                all_pool: list[str]) -> tuple[str, ...]:
    mine = skill_pool(occ, fld)[:24]
    got: list[str] = []
    for rank, s in enumerate(mine):
        if rng.random() < (0.7 if rank < 8 else 0.35 if rank < 16 else 0.15):
            got.append(s)
    for _ in range(rng.randint(1, 3)):  # same-field noise
        s = rng.choice(field_pool)
        if s not in got:
            got.append(s)
    if rng.random() < 0.4:  # cross-field noise
        s = rng.choice(all_pool)
        if s not in got:
            got.append(s)
    while len(got) < 4 and len(set(mine)) >= 4:
        s = rng.choice(mine)
        if s not in got:
            got.append(s)
    rng.shuffle(got)
    # O*NET can name one skill twice ("Electronic medical record EMR" and "Patient electronic
    # medical record EMR"): a resume lists it once
    seen: set[str] = set()
    got = [x for x in got[:13] if not (skill_canon(x) in seen or seen.add(skill_canon(x)))]
    if rng.random() < SOFT_ON_RESUME:
        got += rng.sample([x for x in SOFT_MUST if x not in got], rng.randint(1, 2))
    return tuple(got)


def _degree_level(rng: random.Random, occ: Occupation) -> str:
    order = ("none", "associate", "bachelor", "master", "doctoral")
    i = order.index(occ.degree_level)
    r = rng.random()
    if r < 0.68:
        pass
    elif r < 0.84:
        i = min(i + 1, 4)
    else:
        i = max(i - 1, 0)
    if i == 0 and rng.random() < 0.35:
        i = 2  # some people hold more than the job asks
    return order[i]


def graduated_year(rng: random.Random, years: int, lo: int = 1995, hi: int = 2026) -> int:
    """A finished degree's year, consistent with `years` of work after it (roughly)."""
    return max(lo, min(hi, 2026 - years - rng.randint(0, 2)))


def enrolled_year(rng: random.Random, level: str) -> int:
    return 2026 + {"associate": rng.randint(1, 2), "bachelor": rng.randint(1, 4),
                   "master": rng.randint(1, 2), "doctoral": rng.randint(1, 5)}.get(level, 1)


def cpt_year(year: int, level: str) -> int:
    """An expected graduation year an F-1 student on CPT can have (CPT_MAX_AHEAD)."""
    ahead = CPT_MAX_AHEAD.get(level)
    return min(year, CURRENT_YEAR + ahead) if ahead is not None else year


# Some people write their field short ("CS", "EE"); the rule reads it through degrees.FIELD_ALIASES.
SHORT_FIELDS = {"Computer Science": "CS", "Electrical Engineering": "EE", "Mathematics": "Math",
                "Statistics": "Stats", "Economics": "Econ", "Management Information Systems": "MIS",
                "Biomedical Engineering": "BME", "Biochemistry": "Biochem"}
ALIAS_RATE = 0.15


def short_field(field: str) -> str:
    return SHORT_FIELDS.get(field, field)


def stem_field(rng: random.Random, fld: str) -> str:
    from .degrees import is_stem

    pool = [x for x in FIELDS[fld]["deg"] if is_stem(x)]
    return rng.choice(pool or ["Computer Science", "Biology", "Mathematics"])


def sample_nuisance(rng: random.Random, intl: bool) -> tuple[str, str]:
    """(education_country, language): fairness nuisance fields the rule never reads."""
    if intl:
        names = [c for c, v in COUNTRIES.items() if v[0] > 0]
        country = _wchoice(rng, names, [COUNTRIES[c][0] for c in names])
    elif rng.random() < 0.85:
        country = "United States"
    else:
        names = [c for c, v in COUNTRIES.items() if v[1] > 0]
        country = _wchoice(rng, names, [COUNTRIES[c][1] for c in names])
    if country == "United States":
        language = _wchoice(rng, [l for l, _ in US_LANGUAGES], [w for _, w in US_LANGUAGES])
    else:
        langs = COUNTRIES[country][2]
        language = rng.choice(langs) if rng.random() < 0.75 else "English"
    return country, language


def sample_willing(rng: random.Random, current: str, avoid=lambda place: False) -> tuple[str, ...]:
    """Relocation list: empty, one to three metros, or anywhere in the US. Never the metro they
    live in (someone in Newton does not relocate to Boston)."""
    if rng.random() < 0.08:
        return (ANYWHERE_US,)
    willing: list[str] = []
    for _ in range(_wchoice(rng, (0, 1, 2, 3), (45, 25, 20, 10))):
        m = _wchoice(rng, METRO_NAMES, METRO_W)
        if m not in willing and not (current and places_match(m, current)) and not avoid(m):
            willing.append(m)
    return tuple(willing)


def _license_education(rng: random.Random, license: str, degree: str, field: str,
                       auth: str) -> tuple[str, str] | None:
    """(degree level, degree field) for someone who holds `license`, changed as little as
    possible; None when the license cannot go with this work authorization."""
    if license_fits(license, degree, field):
        return degree, field
    need = LICENSE_EDUCATION[license]
    if DEGREE_ORDER.index(degree) > DEGREE_ORDER.index(need.max_level):
        return None  # an entry certificate is not what someone with a graduate degree holds
    level = max(degree, need.level, key=DEGREE_ORDER.index)
    if need.level == "none" and auth == "citizen_or_pr" and rng.random() < 0.6:
        return "none", ""  # trades and certificates: most holders have no degree
    if need.fields is None:
        return level, field
    new_field = rng.choice(need.fields)
    from .degrees import is_stem

    if auth == "stem_opt" and not is_stem(new_field):
        return None
    return level, new_field


def make_candidate_card(rng, occs_by_field, fld: str, occ: Occupation, field_pool, all_pool) -> CandidateCard:
    intl = rng.random() < INTL_SHARE
    auth = "citizen_or_pr"
    if intl:
        stem = fld in STEM_FIELDS
        pairs = [(a, w * (3 if a == "stem_opt" and stem else 0 if a == "stem_opt" else 1)) for a, w in INTL_AUTH]
        auth = _wchoice(rng, [a for a, _ in pairs], [w for _, w in pairs])
    degree = _degree_level(rng, occ)
    if auth in ("f1_cpt", "f1_opt", "stem_opt", "needs_h1b_now") and degree in ("none", "associate"):
        degree = "bachelor"
    # experience and graduation year (graduation_year <= 2026 is a finished degree)
    years = _wchoice(rng, list(range(21)), _YEAR_W)
    gy: int | None
    student = auth == "f1_cpt" or (auth in ("citizen_or_pr", "other_visa") and degree != "none"
                                   and rng.random() < STUDENT_RATE)
    if student:
        years = rng.choice((0, 0, 1, 1, 2, 3)) if degree != "doctoral" else rng.randint(0, 6)
        gy = enrolled_year(rng, degree)
        if auth == "f1_cpt":
            years = min(years, CPT_YEARS[degree])
            gy = cpt_year(gy, degree)
    elif auth == "f1_opt":
        years, gy = rng.choice((0, 0, 1, 2)), rng.choice((2025, 2026, 2026))
    elif auth == "stem_opt":
        years, gy = rng.choice((1, 1, 2, 3)), rng.choice((2023, 2024, 2025, 2025))
    elif auth == "needs_h1b_now":
        years = rng.randint(1, 10)
        gy = graduated_year(rng, years)
    else:
        gy = graduated_year(rng, years)
    if auth in ("f1_opt", "stem_opt") and degree == "bachelor":
        years = min(years, CURRENT_YEAR - gy + 1)  # came at 18: a year of work before graduating at most
    degree_field = ""
    if degree != "none":
        degree_field = stem_field(rng, fld) if auth == "stem_opt" else rng.choice(FIELDS[fld]["deg"])
    else:
        gy = None
    # licenses: a license holder has the education it follows from
    licenses: tuple[str, ...] = ()
    if occ.license and not student and rng.random() < (0.3 if intl else 0.55):
        fit = _license_education(rng, occ.license, degree, degree_field, auth)
        if fit is not None:
            licenses = (occ.license,)
            new_degree, degree_field = fit
            if new_degree != degree:
                gy = None if new_degree == "none" else graduated_year(rng, years) if degree == "none" else gy
                degree = new_degree
    # a license the degree gives: no year of work before that graduation (variants.pre_degree_years; the resume dates
    # every year of work as a job). OPT and STEM OPT pin the graduation year, so their years shrink; anyone else
    # graduated earlier. No random draw, so the other cards stay as they were.
    from .variants import license_needs_degree

    if gy is not None and gy <= CURRENT_YEAR and license_needs_degree(licenses, degree) and years > CURRENT_YEAR - gy:
        if auth in ("f1_opt", "stem_opt"):
            years = CURRENT_YEAR - gy
        else:
            gy = CURRENT_YEAR - years
    # places
    cur = sample_place(rng)
    willing = sample_willing(rng, cur)
    remote_w = (60, 25, 15) if fld != "IT and data" else (35, 35, 30)
    if fld in ("Nursing", "Trades and construction", "Hospitality and personal care"):
        remote_w = (88, 10, 2)
    remote_pref = _wchoice(rng, ("any", "hybrid_or_remote", "remote_only"), remote_w)
    country, language = sample_nuisance(rng, intl)
    # preferences
    base = _ZONE_PAY[occ.job_zone] * _FIELD_PAY.get(fld, 1.0)
    floor = int(base * rng.uniform(0.7, 1.0) / 5000) * 5000 if rng.random() < 0.5 else None
    wanted_types = _wchoice(
        rng,
        ((), ("full_time",), ("internship",), ("full_time", "contract"), ("full_time", "part_time"), ("internship", "full_time")),
        (45, 22, 12 if years <= 1 else 2, 8, 8, 5),
    )
    if student:
        wanted_types = _wchoice(rng, ((), ("internship",), ("internship", "full_time"), ("full_time",)),
                                (30, 40, 20, 10))
        if floor is not None:
            floor = int(floor * 0.6 / 5000) * 5000
    wanted_ind: tuple[str, ...] = ()
    if rng.random() < 0.45:
        n = rng.randint(1, 3)
        pool = list(FIELDS[fld]["ind"]) + list(INDUSTRIES)
        wanted_ind = tuple(dict.fromkeys(rng.sample(pool, min(n, len(pool)))))[:3]
    avoided: tuple[str, ...] = ()
    if rng.random() < 0.18:
        opts = [i for i in INDUSTRIES if i not in wanted_ind and i not in FIELDS[fld]["ind"]]
        avoided = (rng.choice(opts),)
    roles = weighted_sample(rng, ROLE_W, 1, wanted_role_words(wanted_types, auth)) if rng.random() < 0.08 else ()
    if degree_field and rng.random() < ALIAS_RATE:
        degree_field = short_field(degree_field)
    return CandidateCard(
        field=fld,
        degree_level=degree,
        degree_field=degree_field,
        graduation_year=gy,
        years=years,
        skills=_skills_for(rng, occ, fld, field_pool, all_pool),
        licenses=licenses,
        current_location=cur,
        willing_locations=tuple(willing),
        remote_preference=remote_pref,
        work_auth=auth,
        salary_floor=floor,
        employment_types_wanted=wanted_types,
        industries_wanted=wanted_ind,
        industries_avoided=avoided,
        roles_avoided=roles,
        education_country=country,
        language=language,
    )


def sample_candidate_cards(occs: list[Occupation], total: int = 1200, seed: int = 20261018):
    """[(id, card, target_soc)] with ids cd_00001...; target_soc is the occupation each is built around."""
    rng = random.Random(seed)
    groups = group_by_field(occs)
    quotas = _quotas(total)
    field_pools = {f: sorted({s for o in groups[f] for s in skill_pool(o, f)[:12]}) for f in FIELD_NAMES}
    all_pool = sorted({s for f in FIELD_NAMES for s in field_pools[f]})
    picks = []
    for fld in FIELD_NAMES:
        for occ in _cycle(rng, groups[fld], quotas[fld]):
            picks.append((fld, occ))
    rng.shuffle(picks)
    out = []
    for i, (fld, occ) in enumerate(picks, start=1):
        card = make_candidate_card(rng, groups, fld, occ, field_pools[fld], all_pool)
        out.append((f"cd_{i:05d}", card, occ.soc))
    return out
