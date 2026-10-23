"""Degree fields: aliases, families of adjacent fields, STEM fields, vague posting wording.

A posting's degree_fields can name fields ("Computer Science", "EE"), whole groups
("Engineering", "STEM", "a quantitative field") and vague escape hatches ("or a related
field", "equivalent"). `field_match` compares a candidate's degree field against that
list: "exact" (named, an alias, inside a named group, or in a tight cluster with a named
field when the posting says "or a related field"), "adjacent" (a related field without
that wording, or a neighbouring field: a stretch) or "none". A bare "or equivalent" names
no field and widens nothing.

Real postings word fields loosely, so a posting entry is read more widely than a candidate's
field: "Business" is the business group, science, technology, engineering and math named
together mean STEM, and a joint entry ("Math/Statistics", "Related Engineering/Technical
Field") splits into the fields it names. A candidate's field only goes through FIELD_ALIASES.
"""

from __future__ import annotations

import re

from .schema import norm

# alias (normalized) -> canonical field name (normalized)
FIELD_ALIASES = {
    "cs": "computer science", "comp sci": "computer science", "compsci": "computer science",
    "computer sciences": "computer science", "computer science and engineering": "computer science",
    "cse": "computer science", "eecs": "electrical engineering",
    "ee": "electrical engineering", "electrical and computer engineering": "electrical engineering",
    "ece": "electrical engineering", "electrical engineering and computer science": "electrical engineering",
    "ce": "computer engineering", "computer engineering technology": "computer engineering",
    "se": "software engineering",
    "ml": "machine learning", "ai": "machine learning", "artificial intelligence": "machine learning",
    "ai ml": "machine learning", "ai/ml": "machine learning",
    "math": "mathematics", "maths": "mathematics", "applied math": "mathematics",
    "applied mathematics": "mathematics", "pure mathematics": "mathematics",
    "stats": "statistics", "applied statistics": "statistics", "biostatistics": "statistics",
    "ds": "data science", "data analytics": "data science",
    "mis": "information systems", "management information systems": "information systems",
    "computer information systems": "information systems", "it": "information technology",
    "cybersecurity": "cybersecurity", "cyber security": "cybersecurity", "information security": "cybersecurity",
    "me": "mechanical engineering", "mechanical": "mechanical engineering",
    "cheme": "chemical engineering", "bme": "biomedical engineering", "bioengineering": "biomedical engineering",
    "aero": "aerospace engineering", "aeronautical engineering": "aerospace engineering",
    "astronautical engineering": "aerospace engineering", "aerospace": "aerospace engineering",
    "industrial and systems engineering": "industrial engineering", "systems engineering": "industrial engineering",
    "or": "operations research", "operations research and engineering": "operations research",
    "econ": "economics", "biological sciences": "biology", "biological science": "biology",
    "life science": "life sciences", "biochem": "biochemistry", "molecular and cellular biology": "molecular biology",
    "chem": "chemistry", "phys": "physics", "applied physics": "physics", "engineering physics": "physics",
    "business": "business administration", "mba": "business administration", "bba": "business administration",
    "comms": "communications", "communication": "communications", "psych": "psychology",
    "poli sci": "political science", "polisci": "political science", "bsn": "nursing",
    "materials science": "materials science", "materials science and engineering": "materials science",
    "computational biology": "bioinformatics",
    # wording real postings use (data/built/cards/extracted_real_cards.jsonl)
    "electronics engineering": "electrical engineering", "electronic engineering": "electrical engineering",
    "electronics": "electrical engineering", "electrical": "electrical engineering", "bsee": "electrical engineering",
    "bscs": "computer science", "bsme": "mechanical engineering",
    "electrical and electronic engineering": "electrical engineering",
    "electrical and electronics engineering": "electrical engineering",
    "electrical engineering technology": "electrical technology",
    "electronics engineering technology": "electrical technology",
    "material science": "materials science", "computer science engineering": "computer science",
    "computing": "computer science", "software development": "software engineering", "software": "software engineering",
    "information science": "information systems", "information sciences": "information systems",
    "informatics": "information systems", "information management": "information systems",
    "business information systems": "information systems",
    "information science and technology": "information systems",
    "information sciences and technology": "information systems",
    "analytics": "data science", "business analytics": "data science", "data analysis": "data science",
    "data engineering": "data science", "data management": "data science", "data mining": "data science",
    "data": "data science", "data and analytics": "data science",
    "computer vision": "machine learning", "natural language processing": "machine learning",
    "nlp": "machine learning", "deep learning": "machine learning", "ai engineering": "machine learning",
    "ml/ai": "machine learning",
    "operation research": "operations research",
    "operations management": "supply chain management", "supply chain": "supply chain management",
    "logistics": "supply chain management",
    "econometrics": "economics", "commerce": "business administration", "business management": "management",
    "robotics engineering": "robotics", "physical robotics": "robotics", "mechatronics engineering": "mechatronics",
    "manufacturing engineering": "industrial engineering", "biological engineering": "biomedical engineering",
    "quantitative finance": "financial engineering", "computational finance": "financial engineering",
    "financial mathematics": "financial engineering", "mathematical finance": "financial engineering",
    "actuarial": "actuarial science",
    "hci": "human computer interaction", "ui/ux": "human computer interaction", "ux": "human computer interaction",
    "ui ux": "human computer interaction", "user experience": "human computer interaction",
    "user experience design": "human computer interaction", "ux design": "human computer interaction",
    "interaction design": "human computer interaction", "ui/ux engineering": "human computer interaction",
}

# A named group in a posting accepts every field in it as exact.
GROUPS = {
    "engineering": {"electrical engineering", "computer engineering", "mechanical engineering", "civil engineering",
                    "chemical engineering", "industrial engineering", "biomedical engineering",
                    "aerospace engineering", "software engineering", "materials science", "engineering technology",
                    "robotics", "mechatronics", "environmental engineering", "petroleum engineering",
                    "nuclear engineering", "optical engineering"},
    # a bare "Technology" (alone, or "Information Technology or Technology"): computing and technology degrees
    "technology": {"computer science", "computer engineering", "software engineering", "information systems",
                   "information technology", "cybersecurity", "data science", "machine learning",
                   "engineering technology", "electrical technology", "industrial technology"},
    "life sciences": {"biology", "biochemistry", "molecular biology", "microbiology", "biomedical science",
                      "neuroscience", "bioinformatics", "chemistry", "biotechnology"},
    "natural sciences": {"biology", "chemistry", "physics", "biochemistry", "geology", "environmental science"},
    "sciences": {"biology", "chemistry", "physics", "biochemistry", "geology", "environmental science"},
    "quantitative field": {"mathematics", "statistics", "physics", "economics", "computer science", "data science",
                           "operations research", "machine learning", "finance", "financial engineering",
                           "actuarial science"},
    "business": {"business administration", "finance", "accounting", "marketing", "management", "economics"},
}
# "Health Sciences" is a degree of its own (BS in Health Sciences), not a group: a posting that
# names it with Nursing does not take a Biology degree as the same thing.
GROUPS["quantitative"] = GROUPS["quantitative field"]
# Posting wordings of a group, read before FIELD_ALIASES: a posting's "Business" is the business
# group (a Finance major qualifies), while a candidate's "Business" degree is Business Administration.
GROUP_NAMES = {
    "business": "business", "science": "sciences", "sciences": "sciences", "scientific disciplines": "sciences",
    "natural science": "natural sciences", "engineering science": "engineering", "engineering sciences": "engineering",
    "technology": "technology", "tech": "technology", "quantitative fields": "quantitative field",
    "quantitative disciplines": "quantitative field", "quantitative subjects": "quantitative field",
    "quantitative": "quantitative field", "stem": "stem", "stem field": "stem", "stem fields": "stem",
    "stem disciplines": "stem",
}
# A list naming at least this many of science, technology, engineering and math means STEM.
STEM_WORDS_MIN = 3

# Fields a recruiter treats as interchangeable when a posting says "or a related field":
# two fields in one cluster are related (accepted under "related", a stretch without it).
# Clusters are small on purpose; a field can sit in several.
CLUSTERS = (
    {"computer science", "computer engineering", "software engineering"},
    {"computer science", "data science", "machine learning"},
    {"computer science", "cybersecurity"},
    {"cybersecurity", "information technology", "information systems"},
    {"computer engineering", "electrical engineering"},
    {"mathematics", "statistics", "data science"},
    {"statistics", "data science", "machine learning"},
    {"mathematics", "statistics", "operations research"},
    {"operations research", "industrial engineering"},
    {"mechanical engineering", "aerospace engineering"},
    {"chemical engineering", "materials science"},
    {"biology", "molecular biology", "microbiology", "biomedical science", "biotechnology", "biochemistry"},
    {"biology", "neuroscience"},
    {"biochemistry", "chemistry"},
    {"pharmacy", "pharmaceutical sciences"},
    {"health sciences", "public health"},
    {"health sciences", "kinesiology"},
    {"finance", "accounting"},
    {"finance", "economics"},
    {"business administration", "management", "marketing"},
    {"business administration", "finance"},
    {"management", "supply chain management"},
    {"psychology", "counseling"},
    {"psychology", "sociology"},
    {"sociology", "anthropology"},
    {"social work", "human services", "counseling"},
    {"sociology", "social work"},
    {"communications", "journalism"},
    {"graphic design", "fine arts"},
    {"education", "elementary education", "special education"},
    {"environmental science", "forestry"},
    {"environmental science", "geology"},
    {"forestry", "agriculture"},
    {"industrial technology", "engineering technology"},
    {"engineering technology", "electrical technology"},
    {"criminal justice", "forensic science"},
    {"robotics", "mechatronics", "mechanical engineering"},
    {"robotics", "mechatronics", "electrical engineering", "computer engineering"},
    {"financial engineering", "mathematics", "statistics"},
    {"actuarial science", "mathematics", "statistics"},
    {"environmental engineering", "civil engineering"},
)

# Neighbouring fields: close enough that a mismatch is a stretch (Explore), even when the
# posting says "or a related field", but never accepted as the field itself.
NEIGHBOURS = (
    {"computer science", "computer engineering", "software engineering", "information systems", "data science",
     "cybersecurity", "machine learning", "information technology", "electrical engineering", "bioinformatics"},
    # fields only postings name, each next to what it borders (a set makes every pair in it close)
    {"robotics", "computer science"}, {"robotics", "machine learning"},
    {"human computer interaction", "computer science"}, {"human computer interaction", "information systems"},
    {"human computer interaction", "graphic design"}, {"human computer interaction", "psychology"},
    {"financial engineering", "finance"}, {"financial engineering", "mathematics"},
    {"financial engineering", "statistics"}, {"financial engineering", "operations research"},
    {"financial engineering", "economics"}, {"actuarial science", "mathematics", "statistics"},
    {"actuarial science", "finance"},
    {"mathematics", "statistics", "data science", "physics", "operations research", "computer science",
     "machine learning"},
    {"economics", "mathematics", "statistics"},
    {"electrical engineering", "computer engineering", "mechanical engineering", "aerospace engineering", "physics",
     "biomedical engineering", "industrial engineering", "chemical engineering", "civil engineering",
     "materials science", "engineering technology", "electrical technology", "robotics", "mechatronics",
     "environmental engineering", "petroleum engineering", "nuclear engineering", "optical engineering"},
    {"environmental engineering", "environmental science"},
    {"biology", "biochemistry", "molecular biology", "microbiology", "chemistry", "biomedical science", "neuroscience",
     "pharmaceutical sciences", "bioinformatics", "biotechnology", "forensic science"},
    {"biomedical engineering", "biology", "biomedical science"},
    {"nursing", "health sciences", "public health", "kinesiology", "biology", "nutrition"},
    {"pharmacy", "pharmaceutical sciences", "chemistry", "biochemistry"},
    {"finance", "accounting", "business administration", "economics", "marketing", "management",
     "supply chain management"},
    {"hospitality management", "business administration", "management"},
    {"hospitality management", "culinary arts"},
    {"psychology", "sociology", "anthropology", "social work", "human services", "counseling", "public health"},
    {"psychology", "neuroscience"},
    {"criminal justice", "sociology"},
    {"criminal justice", "political science", "paralegal studies"},
    {"communications", "journalism", "english", "marketing"},
    {"graphic design", "fine arts", "film"},
    {"education", "elementary education", "special education"},
    {"environmental science", "geology", "forestry", "agriculture"},
    {"environmental science", "biology"},
    {"industrial technology", "engineering technology", "electrical technology", "construction management",
     "industrial engineering", "civil engineering"},
    {"political science", "history", "philosophy", "paralegal studies"},
    {"english", "history", "philosophy"},
    {"physical therapy", "kinesiology", "health sciences"},
    {"occupational therapy", "kinesiology", "health sciences"},
    {"respiratory therapy", "radiologic technology", "health sciences"},
    {"medical laboratory science", "medical laboratory technology", "biology", "chemistry", "microbiology"},
    {"dental hygiene", "health sciences"},
)
FAMILIES = NEIGHBOURS  # every pair of fields the rule calls close

# Degree fields that qualify for the STEM OPT extension (DHS STEM list, simplified).
STEM_DEGREE_FIELDS = frozenset({
    "computer science", "computer engineering", "software engineering", "information systems", "data science",
    "cybersecurity", "machine learning", "information technology", "bioinformatics",
    "mathematics", "statistics", "physics", "chemistry", "operations research",
    "biology", "biochemistry", "molecular biology", "microbiology", "biomedical science", "neuroscience",
    "biotechnology", "pharmaceutical sciences", "forensic science",
    "electrical engineering", "mechanical engineering", "civil engineering", "chemical engineering",
    "industrial engineering", "biomedical engineering", "aerospace engineering", "materials science",
    "engineering technology", "electrical technology", "industrial technology",
    "environmental science", "geology", "forestry", "agriculture",
    "robotics", "mechatronics", "environmental engineering", "petroleum engineering", "nuclear engineering",
    "optical engineering",
})

# Every canonical field the tables name: a posting entry made of these is split into them.
KNOWN_FIELDS = frozenset(set(FIELD_ALIASES.values()) | {x for g in GROUPS.values() for x in g}
                         | {x for c in CLUSTERS for x in c} | {x for c in NEIGHBOURS for x in c}
                         | STEM_DEGREE_FIELDS)

# Knowledge a degree field stands for beyond its own name (a degree in Chemistry already counts as
# the skill "Chemistry"): what every program in the field teaches, so the rule never asks a
# Biochemistry graduate to list Biology. Skill names as cards write them.
_ENGINEERING = ("electrical engineering", "computer engineering", "mechanical engineering", "civil engineering",
                "chemical engineering", "industrial engineering", "biomedical engineering", "aerospace engineering",
                "software engineering", "materials science", "engineering technology", "electrical technology",
                "industrial technology")
# What each field's coursework covers, as the skill names cards use (O*NET knowledge areas under
# their plain names: "Engineering principles" is Engineering and Technology, "Computer systems" is
# Computers and Electronics, "Sociology" is Sociology and Anthropology). Only what every program in
# the field teaches: Economics does not show "Business management", Nursing does not show "Biology".
_FIELD_SKILL_ROWS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (_ENGINEERING, ("Engineering principles",)),
    (("computer science", "software engineering", "computer engineering"),
     ("Data structures", "Algorithms", "Computer systems")),
    (("information systems", "information technology", "cybersecurity", "electrical engineering",
      "electrical technology"), ("Computer systems",)),
    (("mechanical engineering", "industrial engineering", "aerospace engineering", "engineering technology",
      "industrial technology"), ("Mechanical systems",)),
    (("civil engineering", "construction management"), ("Construction methods",)),
    (("chemical engineering", "pharmacy", "pharmaceutical sciences"), ("Chemistry",)),
    (("biochemistry",), ("Biology", "Chemistry")),
    (("molecular biology", "microbiology", "biomedical science", "biotechnology", "neuroscience"), ("Biology",)),
    (("medicine",), ("Biology", "Clinical medicine")),
    (("dentistry", "nursing", "pharmacy"), ("Clinical medicine",)),
    (("data science",), ("Statistics", "Data analysis", "Computer systems")),
    (("statistics",), ("Data analysis",)),
    (("physics",), ("Mathematics",)),
    (("business administration", "management", "marketing", "hospitality management", "supply chain management"),
     ("Business management",)),
    (("business administration", "marketing"), ("Sales and marketing",)),
    (("business administration", "management"), ("Human resources",)),
    (("finance",), ("Accounting", "Financial modeling")),
    (("counseling",), ("Psychology",)),
    (("counseling", "social work", "human services", "psychology"), ("Counseling",)),
    (("criminal justice",), ("Public safety",)),
    (("paralegal studies",), ("Law and regulations",)),
    (("film", "journalism", "communications"), ("Media production",)),
    (("graphic design",), ("Fine arts", "Design")),
    (("education", "elementary education", "special education"), ("Teaching and training",)),
    (("anthropology",), ("Sociology",)),
    (("agriculture",), ("Food production",)),
)
FIELD_SKILLS: dict[str, tuple[str, ...]] = {}
for _fields, _skills in _FIELD_SKILL_ROWS:
    for _f in _fields:
        FIELD_SKILLS[_f] = tuple(dict.fromkeys((*FIELD_SKILLS.get(_f, ()), *_skills)))

# Wording that widens a named field to its related ones ("Biology or a related field").
_RELATED = re.compile(r"\b(related|similar|relevant|other|comparable|technical discipline|technical field|"
                      r"discipline)\b")
# Wording that names no field: the above, plus "or equivalent" (usually equivalent experience) and
# "or higher". A bare "equivalent" does not widen the named fields.
_VAGUE = re.compile(r"\b(related|equivalent|similar|relevant|other|comparable|technical discipline|"
                    r"technical field|discipline|or higher)\b")


def _field_key(name: str) -> str:
    """Normalized field wording, before FIELD_ALIASES: "&" as "and", no article, no "degree"."""
    key = norm(re.sub(r"[^\w/&+ ]+", " ", name)).replace("&", "and")
    key = re.sub(r"^(a |an |the )", "", key)
    key = re.sub(r"\b(degree|major|program|studies)$", "", key).strip() or key
    if key == "paralegal":
        key = "paralegal studies"
    return key


def canon_field(name: str) -> str:
    key = _field_key(name)
    return FIELD_ALIASES.get(key, key)


def degree_skills(field: str) -> tuple[str, ...]:
    """Skill names a degree in `field` shows: the field itself, then FIELD_SKILLS."""
    if not field:
        return ()
    c = canon_field(field)
    return (c, *FIELD_SKILLS.get(c, ()))


def is_stem(field: str) -> bool:
    return canon_field(field) in STEM_DEGREE_FIELDS


_TECHNICAL = frozenset({"technical", "technical fields", "technical subject", "technical subjects",
                        "technical disciplines", "technical background", "technical backgrounds"})  # "a technical field": any field, as an escape hatch
_SPLIT = re.compile(r"\s*(?:/|,|;|\band\b|\bor\b)\s*")


# "any field", "any area of study": names no field (a posting that says it accepts every field)
_ANY_FIELD = re.compile(r"(?:in )?(?:any|all)(?: academic)?(?: (?:fields?|disciplines?|subjects?|areas?|majors?)(?: areas?)?"
                        r"(?: of stud(?:y|ies))?)?")


def _parse_part(key: str) -> tuple[str, str, bool]:
    """(kind, value, related) for one normalized wording: kind is "named", "group" or "" (vague)."""
    if _ANY_FIELD.fullmatch(key):
        return "", "", False
    related = bool(_RELATED.search(key))
    if _VAGUE.search(key):
        key = _field_key(re.sub(r"\b(or|and|a|an|any|field|fields)\b", " ", _VAGUE.sub(" ", key)))
    key = re.sub(r"\s+fields?$", "", key).strip(" /")
    if key in _TECHNICAL:
        return "", "", True
    if key in GROUP_NAMES:
        return "group", GROUP_NAMES[key], related
    key = FIELD_ALIASES.get(key, key)
    if key in GROUPS:
        return "group", key, related
    if key and not _VAGUE.search(key):
        return "named", key, related
    return "", "", related


def _known(part: tuple[str, str, bool]) -> bool:
    return part[0] == "group" or (part[0] == "named" and part[1] in KNOWN_FIELDS)


def _parse_one(raw: str) -> list[tuple[str, str, bool]]:
    """The (kind, value, related) parts of one entry. An entry that is not a field the tables know
    is split on "/", ",", "and" and "or" ("Math/Statistics", "Related Engineering/Technical Field"),
    a lone word taking the last part's tail ("Electrical or Mechanical Engineering"), when at least
    one part is known; otherwise it stays one name that matches nobody."""
    key = _field_key(raw)
    whole = _parse_part(key)
    if not whole[0] or _known(whole):
        return [whole]
    parts = [x for x in _SPLIT.split(key) if x.strip()]
    if len(parts) < 2:
        return [whole]
    tail = parts[-1].split()[-1] if len(parts[-1].split()) > 1 else ""
    out = []
    for x in parts:
        part = _parse_part(x)
        if not _known(part) and tail and len(x.split()) == 1:
            longer = _parse_part(f"{x} {tail}")
            part = longer if _known(longer) else part
        out.append(part)
    if not any(_known(x) for x in out):
        return [whole]
    return out


_STEM_PARTS = {("group", "sciences"), ("group", "technology"), ("group", "engineering"), ("named", "mathematics")}


def parse_fields(degree_fields: tuple[str, ...]) -> tuple[set[str], set[str], bool]:
    """(named fields, accepted groups, related_ok) from a posting's degree_fields. Science,
    technology, engineering and math named together (STEM_WORDS_MIN of them) also accept STEM."""
    named: set[str] = set()
    groups: set[str] = set()
    related = False
    stem_words: set[tuple[str, str]] = set()
    for raw in degree_fields:
        for kind, value, rel in _parse_one(raw):
            related |= rel
            if kind == "named":
                named.add(value)
            elif kind == "group":
                groups.add(value)
            if (kind, value) in _STEM_PARTS:
                stem_words.add((kind, value))
    if len(stem_words) >= STEM_WORDS_MIN:
        groups.add("stem")
    return named, groups, related


RELATED_ENTRY = "related field"


def card_fields(card) -> tuple[str, ...]:
    """A card's degree_fields as the rule reads them. related_ok (the posting also takes a related field, an
    equivalent degree or STEM; the v2 extraction sets it) counts as a "related field" entry, which v1 and
    synthetic cards carry in the list itself and blueprint.clean_card appends to v2 cards."""
    fields = tuple(card.degree_fields)
    if getattr(card, "related_ok", False) and fields and not parse_fields(fields)[2]:
        return (*fields, RELATED_ENTRY)
    return fields


def field_named(degree_fields: tuple[str, ...]) -> tuple[str, ...]:
    """The entries a reason may quote: everything but purely vague ones ("related field")."""
    return tuple(raw for raw in degree_fields if any(kind for kind, _, _ in _parse_one(raw)))


def constrains(degree_fields: tuple[str, ...]) -> bool:
    named, groups, _ = parse_fields(degree_fields)
    return bool(named or groups)


def field_match(candidate_field: str, degree_fields: tuple[str, ...]) -> str:
    """"exact", "adjacent" or "none". A posting with only vague entries accepts any field."""
    named, groups, related = parse_fields(degree_fields)
    if not named and not groups:
        return "exact"
    c = canon_field(candidate_field) if candidate_field else ""
    if not c:
        return "none"
    if c in named or any(c in (STEM_DEGREE_FIELDS if g == "stem" else GROUPS[g]) for g in groups):
        return "exact"
    targets = set(named) | {x for g in groups for x in (STEM_DEGREE_FIELDS if g == "stem" else GROUPS[g])}
    if any(c in cl and targets & cl for cl in CLUSTERS):
        return "exact" if related else "adjacent"  # "or a related field" accepts a related field
    if any(c in fam and targets & fam for fam in NEIGHBOURS):
        return "adjacent"
    return "none"


def related_fields(field: str) -> set[str]:
    """Canonical fields in a cluster with `field` (accepted under "or a related field"), itself excluded."""
    c = canon_field(field)
    return {x for cl in CLUSTERS if c in cl for x in cl} - {c}


def adjacent_fields(field: str) -> set[str]:
    """Canonical fields close to `field` (related or neighbouring), itself excluded."""
    c = canon_field(field)
    return ({x for fam in NEIGHBOURS if c in fam for x in fam} | related_fields(c)) - {c}
