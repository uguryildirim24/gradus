"""Extraction blueprint v2: what Adaption is told to pull from a posting, and the code that
cleans what comes back. BLUEPRINT (v2, b2b2ed8e) is the live instruction. BLUEPRINT_V3 is fix cycle 2's draft (must_skills,
nice_skills and work_mode lines rewritten), measured on fresh test 1 and not adopted (eval/fix_cycle2.json): it is sent
only by scripts/extract_v3.py, never by the pipelines that import BLUEPRINT.

The output keys are exactly the RequirementCard fields, minus source_posting_id
and onet_soc, which the pipeline fills in. v2 follows the 100-card hand audit in
data/built/audit/extraction_audit.md.

Two layers. The BLUEPRINT asks the model for the right shape. clean_card() then repairs
what a model reliably gets wrong, in code: soft skills out of the required list, slash pairs
as one any-of group, vague degree fields, place names, the closed industry list, and
sponsorship wording. posting_meta() reads country and language from the posting itself, so the
demo and the pair builder can skip non-US and French postings ("fr" means the posting carries French
text, which in this data is Quebec postings, most of them bilingual).
"""

from __future__ import annotations

import json
import re
from dataclasses import fields

from .sample import INDUSTRIES
from .schema import SENIORITIES, RequirementCard, skill_key

BLUEPRINT_VERSION = "v2"
# thin_posting is set by code from the card and the posting text (rules.mark_thin), never extracted
PIPELINE_KEYS = ("source_posting_id", "onet_soc", "thin_posting")
OUTPUT_KEYS = tuple(f.name for f in fields(RequirementCard) if f.name not in PIPELINE_KEYS)

UNIVERSAL = "Extract the hiring requirements from the job posting in the context into one JSON object. Output only the JSON."

BLUEPRINT = (
    """Requirement card blueprint v2. Output exactly one JSON object and nothing else (no prose, no code fences) with exactly these keys:
title (string), field (short occupation name), seniority ("entry"|"mid"|"senior"|"lead"), industry (string: pick exactly one from this list, copied as written: __INDUSTRIES__),
degree_level ("none"|"associate"|"bachelor"|"master"|"doctoral"): the MINIMUM degree the applicant must already HOLD. A degree only being pursued is not held: undergraduate students -> "none"; master's or PhD students -> "bachelor"; PhD students who must already hold a master's -> "master".
enrolled_in ("none"|"associate"|"bachelor"|"master"|"doctoral"): set when the role is only for people currently studying; the LOWEST level of study accepted. "Pursuing a bachelor's or master's" -> "bachelor"; "pursuing a master's or PhD" / "graduate students" -> "master"; "PhD students" / "working towards a PhD" -> "doctoral"; "currently enrolled in a degree program" with no level, or an associate program accepted -> "associate". "Or recent graduates" or "or equivalent experience" -> "none". Otherwise "none".
degree_fields (list of the specific fields the posting names, [] if any field is fine). Name only real fields of study ("Computer Science", "Electrical Engineering"), and only from the required / basic qualifications: fields named under "Preferred" are not degree_fields. Never list "related field", "equivalent", "STEM", "technical discipline" or "quantitative field" as a field: leave them out and set related_ok to true.
related_ok (true only when degree_fields is not empty and the required qualifications also accept "a related field", an equivalent degree, "STEM" or another unnamed field; else false),
min_years (int, the stated MINIMUM years; "ideally", "preferred" or "plus" years are not a minimum; 0 if not stated),
must_skills (list of groups; each group is a list of alternative skill names, short names, and the group is met by any one of them). Take must_skills ONLY from the required / minimum / basic qualifications section: first find that heading. Items under "Preferred", "Nice to have", "Bonus", "Plus" or "Ideal" headings are never must_skills, and bullets under "Responsibilities", "What you will do", "Key responsibilities" or "About the role" are never must_skills, even when they name tools. If the posting has no required-qualifications section, must_skills is []. Make a group of alternatives ONLY when the posting itself offers a choice ("or", "and/or", "one of", "any of", "such as", "e.g.", "like", "similar", "equivalent"): "Java or Python" -> [["Java","Python"]]; "one or more of C, C++, Java" -> [["C","C++","Java"]]; "such as PyTorch, TensorFlow" -> [["PyTorch","TensorFlow"]]. Items joined by "and", or listed one after another as separate requirements, are SEPARATE one-item groups: "Python and SQL" -> [["Python"],["SQL"]]; "proficiency in Python, pandas, NumPy" -> [["Python"],["pandas"],["NumPy"]]. List every concrete language, tool, method or subject the required section names, one item each; do not summarize. A slash pair "A/B" is ONE group of two alternatives: "C/C++" -> [["C","C++"]], "Linux/Unix" -> [["Linux","Unix"]] (except names that are one thing: CI/CD, TCP/IP, A/B testing, R&D). A skill that stands alone is a one-item group: [["SQL"]]. Never take duties, responsibilities, "what you will do", "you will learn" or "opportunity to learn" items. Never put eligibility facts (citizenship, student status, GPA, degree) in must_skills. Leave out vague traits that cannot be checked (e.g. "best practices", "self-direction", "AI"). Soft and generic skills (communication, problem solving, teamwork, collaboration, analytical skills, attention to detail, time management, research, programming, coding, debugging) go to nice_skills, never must_skills; name concrete tools and languages instead.
nice_skills (flat list of skill names from the preferred / nice-to-have / bonus section, plus the soft and generic skills above; anything marked preferred goes here, never in must_skills),
license (string or null): ONLY a license or certification the applicant must ALREADY HOLD to apply (e.g. RN license, CPA, CDL). A license or exam obtained after hire or during onboarding, "if applicable", a security clearance, or the "ability to obtain" one is not a license -> null.
locations (list of "City, ST" strings with full city names, e.g. "New York, NY", "San Francisco, CA", "Washington, DC"; [] if fully remote or unstated), work_mode ("onsite"|"hybrid"|"remote"),
sponsorship ("sponsors"|"will_not_sponsor"|"will_not_sponsor_or_opt"|"unstated"): "will_not_sponsor_or_opt" only when the posting says it will not sponsor AND names OPT, STEM OPT or I-983 as unsupported; "will_not_sponsor" for an explicit no-sponsorship statement, including "without sponsorship now or in the future", "will not require sponsorship" is the candidate's side and also means the employer will not sponsor, "unable to sponsor", "cannot sponsor", "not provide visa sponsorship"; "sponsors" only for an explicit offer to sponsor work visas. Never infer sponsorship from citizenship, export control (ITAR/EAR), security clearance, or "must be authorized to work" wording: those alone -> "unstated".
citizens_only (true when the role is limited to US citizens or US permanent residents: "US citizen", "US citizen or permanent resident", "permanent residency required", "authorized to work on a permanent basis", ITAR/export-control US-person rules, or a security clearance; else false),
e_verify (true if E-Verify is mentioned), salary_min and salary_max (annual pay in US dollars as int, or null): amounts in any currency other than USD (CAD, GBP, EUR, ...) -> null; hourly USD x2080 (e.g. $60/hour -> 124800); base pay, not on-target earnings; null if not stated.
employment_type ("internship"|"full_time"|"part_time"|"contract").
Use only what the posting says; when a field is not stated use its default ("none", [], false, 0, null, "unstated")."""
).replace("__INDUSTRIES__", ", ".join(f'"{i}"' for i in INDUSTRIES))



BLUEPRINT_V2 = BLUEPRINT  # the live text, under the name the v3 draft is compared against

# v3 draft (fix cycle 2, eval/fix_cycle2.json; NOT adopted: on fresh test 1 the full rules agreed with gold less often on
# the rows it flips). v2 with three lines rewritten. must_skills: only what the posting states as required (named
# skills, and required experience areas in a few words); duties, responsibilities, "you will" phrases, credentials, job
# conditions and broad subjects never; a list of examples is one any-of group. nice_skills: preferred items, never
# duties. work_mode: remote when remote is one of the options. Every other line is v2's.
MUST_SKILLS_V3 = (
    'must_skills (list of groups; each group is a list of alternative skill names, short names, and the group '
    'is met by any one of them). A must_skill is something the posting states as REQUIRED of the applicant '
    '(listed under a required / minimum / basic qualifications heading, or in a sentence that says "required", '
    '"must have", "you must have" or "you need"): a named tool, software, platform, programming language, '
    'framework, instrument, technique, method or standard, or a required area of experience named in a few '
    'words ("biomarker development", "month-end close", "field sales leadership"). Find that heading first; if '
    'the posting states nothing as required, must_skills is []. These are never must_skills, wherever they '
    'appear: (1) duties and responsibilities: bullets under "Responsibilities", "What you will do", "Key '
    'responsibilities", "In this role", "About the role", "Day to day" or "Essential functions", and every "you '
    'will ...", "you\'ll ...", "will be responsible for", "you will learn" or "opportunity to learn" sentence, '
    'even when it names a tool: what the hire WILL do is a duty, not a requirement; (2) degrees, diplomas, '
    'licenses, certifications, trainings and exams: they belong to degree_level, degree_fields or license; (3) '
    'conditions of the job: physical demands, schedule, shifts, travel, transportation, location, background '
    'checks; (4) traits, general abilities and broad subjects that cannot be checked on a resume ("strategic '
    'thinking", "comfortable around patients", "mathematical concepts", "statistical foundations", "best '
    'practices", "AI"); (5) eligibility facts (citizenship, student status, GPA). Groups: a list of examples is '
    'ONE group of alternatives: when the posting offers a choice or gives examples ("or", "and/or", "one of", '
    '"one or more of", "any of", "such as", "e.g.", "for example", "including", "like", "or similar", "or '
    'equivalent", examples in parentheses), all the named items are one group: "Java or Python" -> '
    '[["Java","Python"]]; "one or more of C, C++, Java" -> [["C","C++","Java"]]; "experience with an ML '
    'framework such as PyTorch, TensorFlow or JAX" -> [["PyTorch","TensorFlow","JAX"]]; "immunoassay platforms '
    '(ELISA, MSD)" -> [["ELISA","MSD"]]. Only items the posting requires one by one are SEPARATE one-item '
    'groups: "Python and SQL" -> [["Python"],["SQL"]]. Name each required item once, in a few words ("SQL", '
    '"ELISA", "Epic"), never a whole sentence. A slash pair "A/B" is ONE group of two alternatives: "C/C++" -> '
    '[["C","C++"]], "Linux/Unix" -> [["Linux","Unix"]] (except names that are one thing: CI/CD, TCP/IP, A/B '
    'testing, R&D). Soft and generic skills (communication, problem solving, teamwork, collaboration, '
    'analytical skills, attention to detail, time management, research, programming, coding, debugging) and '
    'general office software (Microsoft Office, email) go to nice_skills, never must_skills; name concrete '
    'tools and languages instead.'
)
NICE_SKILLS_V3 = (
    'nice_skills (flat list of named skills the posting marks as preferred, nice to have, bonus, a plus, ideal '
    'or desired, plus the soft and generic skills above; anything marked preferred goes here, never in '
    'must_skills; duties are not nice_skills either),'
)
LOCATIONS_WORK_MODE_V3 = (
    'locations (list of "City, ST" strings with full city names, e.g. "New York, NY", "San Francisco, CA", '
    '"Washington, DC"; [] if fully remote or unstated), work_mode ("onsite"|"hybrid"|"remote"): "remote" when '
    'the role can be done fully remotely, also when remote is one of the listed locations or options; "hybrid" '
    'when the posting states a split between office and remote days; else "onsite",'
)


def _swap_line(text: str, prefix: str, new: str) -> str:
    """text with its one line that starts with prefix replaced by new."""
    lines = text.split("\n")
    hit = [i for i, line in enumerate(lines) if line.startswith(prefix)]
    if len(hit) != 1:
        raise ValueError(f"{len(hit)} blueprint lines start with {prefix!r}")
    lines[hit[0]] = new
    return "\n".join(lines)


BLUEPRINT_V3 = BLUEPRINT_V2.replace("blueprint v2.", "blueprint v3.", 1)
for _prefix, _new in (("must_skills (", MUST_SKILLS_V3), ("nice_skills (", NICE_SKILLS_V3),
                      ("locations (", LOCATIONS_WORK_MODE_V3)):
    BLUEPRINT_V3 = _swap_line(BLUEPRINT_V3, _prefix, _new)

# ------------------------------------------------------------------ skills

# Soft or generic skills no resume lists as a skill. They move to nice_skills.
SOFT_SKILLS = frozenset(
    {
        "communication", "written communication", "verbal communication", "oral communication",
        "interpersonal", "interpersonal communication", "presentation", "presentation skills",
        "problem solving", "critical thinking", "analytical", "analytical thinking", "analysis",
        "teamwork", "team player", "collaboration", "cross functional collaboration",
        "multitasking", "multi tasking", "time management", "organization", "organizational",
        "attention to detail", "detail oriented", "leadership", "adaptability", "flexibility",
        "creativity", "curiosity", "work ethic", "self motivation", "self starter", "initiative",
        "research", "programming", "coding", "debugging", "troubleshooting", "documentation",
        "algorithms", "data structures", "software development", "software engineering",
        "computer science fundamentals", "technical skills", "programming fundamentals", "cs fundamentals", "english",
        "fluent english", "english proficiency", "business proficient in english",
    }
)
_SOFT_PREFIX = re.compile(r"^(strong|excellent|good|great|solid|effective|proven|demonstrated|basic|general|"
                          r"superior|exceptional|outstanding)\s+")
_SOFT_SUFFIX = re.compile(r"\s+(skills?|abilit(y|ies)|capabilit(y|ies)|aptitude)$")
_SOFT_AND = re.compile(r"\s+(?:and|&)\s+")

# "A/B" is a pair of alternatives unless it is one name.
SLASH_NAMES = frozenset({"ci/cd", "tcp/ip", "a/b", "i/o", "r&d", "pl/sql", "n/a", "a/b testing", "q/a", "d/a", "a/d",
                         "ui/ux", "ux/ui", "cad/cam", "front-end/back-end", "frontend/backend", "pl/i"})
# Slash pairs that are one product under another spelling.
SLASH_RENAME = {"onnx/runtime": "ONNX Runtime"}
# Slash pairs where the posting wants both: alone in a group they become two required groups;
# inside a group of alternatives they stay two alternatives.
SLASH_BOTH = frozenset({"html/css", "css/html"})


def soft_key(name: str) -> str:
    k = skill_key(name)
    for _ in range(2):
        k = _SOFT_SUFFIX.sub("", _SOFT_PREFIX.sub("", k)).strip()
    return k


def is_soft(name: str) -> bool:
    k = soft_key(name)
    if k in SOFT_SKILLS:
        return True
    parts = [p.strip() for p in _SOFT_AND.split(k) if p.strip()]
    return len(parts) > 1 and all(p in SOFT_SKILLS for p in parts)


def soft_in_text(name: str, text: str) -> bool:
    """Whether the posting names a soft skill. The blueprint lists eleven soft words as examples and the
    model often copies the whole list into nice_skills ("time management" on 294 cards whose posting never
    says it); clean_card drops a soft nice skill the posting does not name. "Strong analytical skills"
    counts when the posting says "analytical"; "problem-solving" matches "problem solving"."""
    names = {soft_key(name), *(p.strip() for p in _SOFT_AND.split(soft_key(name)) if p.strip())}
    for k in filter(None, names):
        # one word: its stem counts ("communication" in "communicate", "collaboration" in "collaborative")
        pat = re.escape(k[:max(5, len(k) - 3)]) if " " not in k else _name_pattern(k) + r"s?(?![\w])"
        if re.search(r"(?<![\w])" + pat, text, re.I):
            return True
    return False


def _split_slash(name: str) -> list[str]:
    n = re.sub(r"\s*/\s*", "/", " ".join(name.split()))
    if n.casefold() in SLASH_RENAME:
        return [SLASH_RENAME[n.casefold()]]
    if "/" not in n or n.casefold() in SLASH_NAMES or "(" in n or " " in n:
        return [" ".join(name.split())]
    parts = [p for p in n.split("/") if p]
    if len(parts) < 2 or any(len(p) > 24 for p in parts):
        return [n]
    return parts


def _clean_name(name: str) -> str:
    """Trim spaces and punctuation; a leading dot stays (".NET")."""
    return " ".join(str(name).replace("–", "-").split()).lstrip(" ,;:").rstrip(" .,;:")


def _flat(x) -> list:
    """Skill names from a value the model may nest (["Swift", ["Kotlin", "C++"]], nice [["collaboration"]]):
    every leaf, in order, so no name reads "['Kotlin', 'C++']"."""
    if isinstance(x, (list, tuple)):
        return [y for v in x for y in _flat(v)]
    return [x]


def clean_skills(must, nice) -> tuple[list[list[str]], list[str]]:
    """Groups with slash pairs split, soft skills moved out, duplicates removed. Nested lists are flattened:
    inside a must group they are more alternatives, in nice_skills more names."""
    groups: list[list[str]] = []
    moved: list[str] = []
    seen_groups: set[frozenset[str]] = set()
    expanded: list[list[str]] = []
    nice = _flat(nice or [])
    for g in must or []:
        items = _flat(g)
        if len(items) == 1 and re.sub(r"\s*/\s*", "/", _clean_name(items[0])).casefold() in SLASH_BOTH:
            expanded.extend([p] for p in re.split(r"\s*/\s*", _clean_name(items[0])))
        else:
            expanded.append(items)
    for items in expanded:
        alts: list[str] = []
        for it in items:
            alts.extend(_split_slash(_clean_name(it)))
        alts = [a for a in dict.fromkeys(a for a in alts if a)]
        keep = [a for a in alts if not is_soft(a)]
        moved.extend(a for a in alts if is_soft(a))
        if not keep:
            continue
        key = frozenset(skill_key(a) for a in keep)
        if key not in seen_groups:
            seen_groups.add(key)
            groups.append(keep)
    out_nice: list[str] = []
    seen: set[str] = set()
    have = {skill_key(a) for g in groups for a in g}
    for n in [*(nice or []), *moved]:
        n = _clean_name(n)
        k = skill_key(n)
        if n and k not in seen and k not in have:
            seen.add(k)
            out_nice.append(n)
    return groups, out_nice


# ------------------------------------------------------------------ sections

_PREF_HEAD = re.compile(r"prefer|nice[- ]to[- ]have|bonus|\bplus(es)?\b|desired|good to have|extra credit|a plus|stand out|"
                        r"helpful|might also have|at least some of|some of the following|sets? you apart|"
                        r"we'?d love (?:it )?(?:if|for) you|love (?:it )?if you|we'?d love to see|even better if|extra points",
                        re.I)
# Duty headings: tasks the hire will do ("Responsibilities", "What you'll do", "Role:", "Job Summary") ...
_DUTY_HEAD = re.compile(r"responsib|what you.?ll do|what you will do|duties|your impact|day[- ]to[- ]day|"
                        r"you will|you.?ll |\blearn\b|areas? of focus|focus areas|job functions|essential functions|"
                        r"^role$|job summary|position summary|accountabilit",
                        re.I)
# ... and headings that describe the job ("About the Role", "The Work"): kind "about". Postings with no
# requirements heading often state their requirements under these, so only task headings drop skills there.
_ABOUT_HEAD = re.compile(r"the role|in this role|about the (role|internship|job|position|team|opportunity)|"
                         r"what we.?re building|the work|your role|role summary", re.I)
_REQ_STRONG = re.compile(r"what you.?ll bring|what you will bring|you bring|you have|bring/have|competenc|"
                         r"requirements?\b|required\b|qualifications?\b|who you are|what we.?re looking|looking for|skills needed|"
                         r"\bfit,? if\b|thrive (?:here )?if|excel\b.{0,40}\bif you|to succeed|we seek|need to see|you.?ll need|you need|"
                         r"you will need|do you need|what it takes|who we want|who we look for|essentials|you will have|"
                         r"capabilities|for success|be successful|(?:great|good|strong|right|ideal) (?:fit|match)|makes? you\b|about you\b|"
                         r"^(?:you are|you're|you)\W*$|you.?ll bring|you will bring|bring with you|come with|"
                         r"what (?:we|you) look for|already have|you.?ll have|experience you need|you should have|seeking|"
                         r"expect(?!\w* (?:of|from) us)(?<!you can expect)", re.I)
_REQ_HEAD = re.compile(r"require|qualif|must|minimum|basic|what you bring|who you are|about you|you have|looking for|"
                       r"skills|experience|what we.?re looking|eligib|candidate|background|profile|^education\b", re.I)
# The header lines extract_postings puts on top of every posting ("Title: Machine Learning Intern").
_HEADER_LINE = re.compile(r"^\s*(?:title|company|locations?):", re.I)


_SMALL = {"and", "of", "to", "the", "for", "with", "in", "a", "an", "or", "you", "we", "your", "our", "on", "at", "do", "will"}
# Common headings that postings write in sentence case ("What you bring", "Preferred qualifications",
# "What we need to see:"); a line that is only one of these is a heading even when not title case.
_KNOWN_HEAD = re.compile(
    r"^(?:what (?:you|we)(?:'ll| will| would| are| do| should)?(?: be)? (?:bring|need|do|doing|have|looking for|"
    r"need to see|love to see|like to see|get|look for)\b.{0,30}|who (?:you are|we (?:look|are looking) for|we want)|"
    r"(?:basic|minimum|preferred|required|desired|additional|key|bonus|your) (?:qualifications|requirements|skills|"
    r"experience)\b.{0,20}|qualifications|requirements|responsibilities|ways to stand out.{0,30}|nice to haves?|"
    r"bonus points|helpful experience|what it takes|what experience you need|what you already have|"
    r"(?:what )?you(?:'ll| will) (?:come with|have|need)|what could set you apart|required for the role|you should have)\W*$",
    re.I)


# A sentence-case requirements heading with no colon ("Essential experience and job requirements"): a short
# line that opens like a heading and names requirements, but is no item ("Experience with Python required").
_PLAIN_REQ_START = re.compile(r"^(?:essential|required|requirements|qualifications|minimum|basic|skills|experience|"
                              r"education|key|candidate|ideal|desirable|desired|preferred)\b", re.I)


def _plain_req_head(t: str) -> bool:
    return len(t.split()) <= 7 and bool(_PLAIN_REQ_START.match(t)) and bool(_REQ_STRONG.search(t) or _PREF_HEAD.search(t)) \
        and not re.search(r"\b(?:with|in|of|using)\b|(?:required|preferred)$", t, re.I)


def _head_kind(line: str, after_bullet: bool = False) -> str | None:
    """Heading kind for one line, or None. Postings often put a bullet marker alone on a line and
    the item below it, so a line right after a bare marker is an item, never a heading, unless the
    line is only a common heading (_KNOWN_HEAD). Curly apostrophes count as straight ones."""
    t = line.replace("\u2019", "'").strip().strip("*#:- \t")
    if _HEADER_LINE.match(line):
        return None
    if after_bullet and _KNOWN_HEAD.match(t) and not line.lstrip().startswith(("-", "*", "\u2022")):
        after_bullet = False  # an empty bullet ("-" alone) right before "Required qualifications"
    # a long heading ending in a colon that names the requirements ("What Life Fitness / Hammer Strength is
    # looking for in our Embedded Software Engineering Intern:") is still a heading
    long_ok = line.rstrip().endswith(":") and len(t.split()) <= 16 and len(t) <= 120 \
        and bool(_REQ_STRONG.search(t) or _PREF_HEAD.search(t))
    if after_bullet or not t or ((len(t) > 80 or len(t.split()) > 11) and not long_ok) \
            or line.lstrip().startswith(("-", "*", "\u2022")) or t.endswith("."):
        return None
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z&']*", t) if w.casefold() not in _SMALL]
    titled = all(w[0].isupper() for w in words) or t.isupper()
    if not (line.rstrip().endswith(":") or (len(t.split()) <= 7 and titled) or _KNOWN_HEAD.match(t)
            or _plain_req_head(t)):
        return None
    if _PREF_HEAD.search(t):
        return "pref"
    if _REQ_STRONG.search(t):
        return "req"
    if _DUTY_HEAD.search(t):
        return "duty"
    if _ABOUT_HEAD.search(t):
        return "about"
    if _REQ_HEAD.search(t):
        return "req"
    return None


_PREF_LINE = re.compile(r"\b(is a plus|are a plus|a plus\b|nice to have|preferred|bonus|desirable|ideally|preferabl[ey]|an asset|"
                        r"is an asset|encouraged to apply)\b", re.I)
# Cues that look forward: "CAD software, ideally SolidWorks", "a language, preferably Python or Java".
_FORWARD_PREF = re.compile(r"ideally|preferabl[ey]", re.I)
# A sentence that opens with the cue ("Preferred: Docker", "Ideally, Rust") is preferred as a whole.
_LEAD_PREF = re.compile(r"^\W*(?:preferred|bonus|nice to have|ideally|a plus|desirable)\b", re.I)
_REQ_WORD = re.compile(r"\b(?:required|is a must|are a must|must have|necessary|mandatory)\b", re.I)
# Sentence ends: ". " or "; ", and a period glued to the next sentence ("Python and SQL.Solid understanding").
_SENT_SPLIT = re.compile(r"(?<=[.;])\s+|(?<=[A-Za-z0-9)]\.)(?=[A-Z][a-z]{2})")


def _depth0_boundary(sent: str, end: int, chars: str = ",;:") -> int:
    """Index of the last of `chars` before `end` outside parentheses, or -1."""
    depth, best = 0, -1
    for i, ch in enumerate(sent[:end]):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch in chars and depth == 0:
            best = i
    return best


def pref_spans(sent: str) -> list[tuple[int, int]]:
    """The stretches of one sentence that a "preferred / is a plus" cue covers. The cue governs its own
    clause, not the whole line: in "Must have Python and SQL (experience with Databricks is a plus)" only
    the parenthesis is preferred; in "statistics and hypothesis testing, with exposure to model-building
    considered a plus" only the clause after the comma; a bare "(preferred)" covers the item before it
    ("C (preferred) or C++": C). A cue at the end of a list ("Docker, Kubernetes, and AWS is a plus")
    covers the whole list, and a sentence that opens with the cue is preferred as a whole."""
    if _LEAD_PREF.match(sent):
        return [(0, len(sent))]
    spans: list[tuple[int, int]] = []
    for m in _PREF_LINE.finditer(sent):
        p = m.start()
        o, c = sent.rfind("(", 0, p), sent.rfind(")", 0, p)
        if o > c:  # inside a parenthesis
            close = sent.find(")", m.end())
            close = len(sent) if close < 0 else close
            inner = re.sub(r"\b(?:is|are|a|an|highly|strongly|but|very|much)\b", "", sent[o + 1:p])
            if not re.search(r"\w", inner) and not re.search(r"\w", sent[m.end():close]):
                b = max(sent.rfind(ch, 0, o) for ch in ",;:(")  # "(preferred)": the item before it
                for w in (" or ", " and ", " and/or "):
                    k = sent.rfind(w, 0, o)
                    if k >= 0:
                        b = max(b, k + len(w) - 1)
                spans.append((b + 1, close + 1))
            else:
                b = max(sent.rfind(ch, o + 1, p) for ch in ",;")
                if re.match(r"\s*(?:and|or|&)\b", sent[b + 1:p]):  # "(C++, Rust, or another language is a plus)"
                    b = sent.rfind(";", o + 1, p)
                spans.append((max(o + 1, b + 1), close))
            continue
        if _FORWARD_PREF.fullmatch(m[0]):  # the cue covers what follows it, to the end of the clause
            e = re.search(r"[;)]|\.(?:\s|$)", sent[m.end():])
            spans.append((p, m.end() + e.start() if e else len(sent)))
            continue
        b = _depth0_boundary(sent, p)
        if b < 0 or re.match(r"\s*(?:and|or|&)\b", sent[b + 1:p]):  # no clause break, or a list tail
            b = _depth0_boundary(sent, p, ";:")
        req = list(_REQ_WORD.finditer(sent, b + 1, p))
        if req:  # "SQL required and Python/Spark preferred": the cue starts after the required part
            b = req[-1].end() - 1
        end = len(sent) if re.match(r"\s*[:\-\u2013]", sent[m.end():]) else m.end()  # "Bonus: X, Y" mid-line
        spans.append((b + 1, end))
    spans.sort()
    out: list[tuple[int, int]] = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


def sections(text: str) -> list[tuple[str, str]]:
    """Posting text -> [(kind, text)] with kind req, pref, duty, about or other. A heading is a short line
    naming the section; lines below it belong to it until the next heading. Inside a section, the
    clause a "preferred / is a plus" cue covers (pref_spans) is its own pref piece."""
    segs: list[tuple[str, list[str]]] = [("other", [])]
    prev = ""
    for line in text.split("\n"):
        kind = _head_kind(line, prev in {"-", "*", "\u2022", "\u25cf", "o"})
        if kind:
            segs.append((kind, []))
        else:
            segs[-1][1].append(line)
        if line.strip():
            prev = line.strip()
    out: list[tuple[str, str]] = []
    for k, lines in segs:
        plain: list[str] = []
        for line in lines:
            for sent in _SENT_SPLIT.split(line):
                spans = pref_spans(sent) if _PREF_LINE.search(sent) else []
                if not spans:
                    plain.append(sent)
                    continue
                rest, last = [], 0
                for a, b in spans:
                    rest.append(sent[last:a])
                    last = b
                rest.append(sent[last:])
                kept = " ".join(x.strip() for x in rest if x.strip())
                if kept:
                    plain.append(kept)
                out.append((k, "\n".join(plain)))
                plain = []
                out.append(("pref", " ".join(sent[a:b] for a, b in spans)))
        out.append((k, "\n".join(plain)))
    return out


def _name_pattern(name: str) -> str:
    """A skill name as a regex: words may be joined by spaces or hyphens ("end-to-end" = "end to end")."""
    return r"[\s\-]+".join(re.escape(w) for w in re.split(r"[\s\-]+", name.strip()) if w)


_SHORT_EDGE_L, _SHORT_EDGE_R = r"(?<![\w+#&])", r"(?![\w+#&-])"


def _found_in(name: str, segs: list[tuple[str, str]]) -> set[str]:
    n = name.strip()
    if len(n) <= 2:  # "C", "R", "Go", "C#": exact case, whole word ("C" is not in "C++", "R&D" or "C-suite")
        pat = re.compile(_SHORT_EDGE_L + re.escape(n) + _SHORT_EDGE_R)
        return {k for k, t in segs if pat.search(t)} or {"req"}  # not found anywhere: keep it required
    pat = re.compile(r"(?<![\w])" + _name_pattern(n) + r"(?![\w])", re.I)
    hit = {k for k, t in segs if pat.search(t)}
    if hit:
        return hit
    toks = [re.escape(w[:-1] if w.endswith("s") and len(w) > 4 else w) for w in re.findall(r"[A-Za-z0-9+#.]+", n) if len(w) > 3]
    if len(toks) >= 2:  # "Deep Learning Frameworks" for "deep learning framework"
        pats = [re.compile(r"(?<![\w])" + t, re.I) for t in toks]
        return {k for k, t in segs if all(p.search(t) for p in pats)}
    return set()


_DEGREE_SENT = re.compile(r"degree|major|bachelor|master|ph\.?d|pursuing|studying|enrolled|discipline|field|background in|coursework|"
                          r"candidate for|graduat", re.I)


def degree_segs(segs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Only the sentences that talk about a degree: a field named in a duty or a bio is no degree field."""
    out = []
    for k, t in segs:
        if k == "pref" and t.strip():  # a preferred clause ("hard sciences preferred - physics, chemistry")
            out.append((k, t))
            continue
        keep = "\n".join(x for x in re.split(r"(?<=[.;])\s+|\n", t) if _DEGREE_SENT.search(x))
        if keep:
            out.append((k, keep))
    return out


def where(alts: list[str], segs: list[tuple[str, str]]) -> str:
    """"keep" or "pref": where a group's alternatives are mentioned. Named under preferred headings
    or on a "... is a plus" line and nowhere else: "pref" (goes to nice_skills). Anything else, and
    names found nowhere (the model renamed them), stay. Names found only under duty headings are
    handled by duty_only(), which needs a requirements section in the same posting. Names of two
    characters or fewer ("C#", "R") are matched in exact case (_found_in)."""
    kinds: set[str] = set()
    for a in alts:
        kinds |= _found_in(a, segs)
    return "pref" if kinds == {"pref"} else "keep"


def duty_only(alts: list[str], segs: list[tuple[str, str]]) -> bool:
    """A required group whose names the posting mentions only under duty headings ("Primary
    Responsibilities", "What you'll do") or preferred ones, in a posting that does have a
    requirements section: a task the hire will do, not a skill the applicant must bring. Names of
    two letters or fewer ("C", "R", "Go") are matched in exact case."""
    kinds: set[str] = set()
    for a in alts:
        kinds |= _found_in(a, segs)
    if any(k == "req" and t.strip() for k, t in segs):
        return bool(kinds & {"duty", "about"}) and kinds <= {"duty", "about", "pref"}
    # No requirements heading found: keep. Many postings write their requirements under a heading the
    # code cannot read ("What We Look For" at Affirm, "What experience you need" at Equifax), and
    # dropping duty-only names there lost real requirements on about half of the changed cards.
    return False


_AND_OR = re.compile(r"\band/or\b", re.I)
_ONE_OF = re.compile(r"\b(?:at least one|one or more|any one|any|one|either)"
                     r"(?: (?:\w+ )?(?:areas?|fields?|topics?|domains?|disciplines?))? of\b", re.I)  # "one or more areas of"
_ANYOF_LIST = re.compile(r"\b(?:at least one|one or more|any one|any|one) of the following\b|"
                         r"\b(?:one or more|any|at least one) of these\b", re.I)
_BULLET = re.compile(r"^\s*(?:[-*\u2022\u25cf\u00b7o]|\d+[.)])\s+")


_CHOICE_LEAD = re.compile(r"\b(?:such as|including|like|e\.g\.,?)\s", re.I)
# The item after "and/or" ends at punctuation or where the sentence goes on ("C++ as well as ...",
# "R to support ...", "mixed integer programming and convex optimisation").
_ITEM_END = re.compile(r"[,.;:()]|\s[\u2013\u2014]\s|\s(?:as well as|and|to|for|with|in|on|including|plus|preferred)\s", re.I)
# A bullet that opens a new requirement ends a "one or more of the following" list.
_REQ_LEAD = re.compile(r"^(?:understanding|experience|knowledge|ability|strong|proficien\w*|excellent|must|"
                       r"demonstrated|proven|solid|working knowledge|basic|good|hands-on|comfort\w*|exposure|you|"
                       r"bachelor|master|ph\.?d|currently|pursuing|enrolled|at least|minimum)\b", re.I)


def _inline_scopes(line: str) -> list[str]:
    """The stretch of one line a choice word covers. "A, B, and/or C": from the start of the list
    (line start or the last colon) to the end of the item after "and/or", so in "Linux and/or
    Android, Claude Code and OpenAI API" only Linux and Android are a choice; an example list in parentheses
    on either side belongs to its item ("ML frameworks (e.g., PyTorch, TensorFlow) and/or distributed systems
    (e.g., Ray, Spark)" is one choice). "one of A, B or C": from the cue to the end of the sentence."""
    out = []
    for m in _AND_OR.finditer(line):
        opened = line.rfind("(", 0, m.start())
        if opened > line.rfind(")", 0, m.start()):  # "statistical programming (Python and/or R)"
            close = line.find(")", m.end())
            out.append(line[opened + 1: close if close >= 0 else len(line)])
            continue
        start = max(line.rfind(c, 0, m.start()) for c in ":;") + 1
        start = max(start, line.rfind(". ", 0, m.start()) + 1)
        # not a lead inside a closed parenthesis: "ML frameworks (e.g., PyTorch, TensorFlow) and/or ..."
        lead = [c for c in _CHOICE_LEAD.finditer(line, start, m.start()) if ")" not in line[c.end():m.start()]]
        sent = m.end() + re.search(r"[.;](?:\s|$)|$", line[m.end():]).start()
        if lead and _CHOICE_LEAD.search(line, m.end(), sent):
            # "Graphics APIs such as OpenGL, Vulkan, or DirectX, and/or shading languages including GLSL and HLSL"
            # (Intel 8d699dd6): the and/or joins two example lists, so the choice holds both to the sentence end
            out.append(line[start:sent])
            continue
        if lead:  # "simulation tools such as FineSim and/or HSPICE": the choice starts after "such as"
            start = lead[-1].end()
        end = _ITEM_END.search(line, m.end())
        stop = end.start() if end else len(line)
        if end and end[0] == "(" and line[:m.start()].rstrip().endswith(")") and line.find(")", stop) > 0:
            # "(e.g., PyTorch, TensorFlow) and/or distributed systems (e.g., Ray, Spark)": examples on both sides
            stop = line.find(")", stop) + 1
        out.append(line[start:stop])
    for m in _ONE_OF.finditer(line):
        end = re.search(r"[.;:]|$", line[m.end():])
        out.append(line[m.start():m.end() + end.start()])
    return out


_COMMA_OR = re.compile(r"(?:,\s*|\s+)or\s+", re.I)
# The word before the first item of a list: "training in econometrics, ...", "experience with C/C++, ...".
_LIST_LEAD = re.compile(r"\b(?:with|in|of|using|including|such as|like|e\.g\.,?|i\.e\.,?|and|or|for|on)\s", re.I)


# Before the items of a list without the Oxford comma: "experience in C/C++, Java or Python", "exposure to
# network programming, ... or GUI programming", "experience with LLMs or Agents".
_PLAIN_LEAD = re.compile(r"\b(?:with|in|of|using|including|such as|like|e\.g\.,?|i\.e\.,?|for|on|to)\s", re.I)


def _plain_or_scope(line: str, m: re.Match, seg: int) -> str | None:
    """"A, B or C" / "with A or B" (no comma before "or"): the list runs from the last lead word ("in", "with",
    "to") before "or". The list must be plain: short comma items with no "and", "&", other "or" or a parenthesis
    with commas in it ("Python (NumPy, Pandas) or R"); a short parenthesis inside an item ("network programming
    (TCP/IP)") is fine. Without a lead word, a list of two or more commas from the clause start."""
    pre = line[seg:m.start()]
    lead = list(_PLAIN_LEAD.finditer(pre))
    if lead:
        pre, seg = pre[lead[-1].end():], seg + lead[-1].end()
    elif pre.count(",") < 2:
        return None
    flat = re.sub(r"\([^(),]{1,30}\)", "", pre)
    if re.search(r"[()]|\band\b|&|\bor\b", flat) or any(len(x.split()) > 5 for x in flat.split(",")):
        return None
    end = _ITEM_END.search(line, m.end())
    after = line[m.end():end.start() if end else len(line)]
    if len(after.split()) > 5:
        return None
    return line[seg:end.start() if end else len(line)]


def _comma_or_scopes(line: str) -> list[str]:
    """"A, B, or C": a plain list ending in ", or" offers a choice between its items ("Strong training in
    econometrics, empirical finance, statistics, or quantitative economics"; "C/C++, Python, or Rust").
    The scope runs from the first item (after the last lead word such as "in" or "with" before the
    first comma) to the end of the item after "or". A list that also says "and" ("Python, SQL, and
    Spark, or equivalent") is left alone: which items are the choice is unclear. Lists without the Oxford
    comma ("C/C++, Java or Python") go through _plain_or_scope."""
    out = []
    for m in _COMMA_OR.finditer(line):
        opened = line.rfind("(", 0, m.start())
        seg = max(max(line.rfind(c, 0, m.start()) for c in ":;"), line.rfind(". ", 0, m.start()),
                  opened if opened > line.rfind(")", 0, m.start()) else -1) + 1
        if not m[0].startswith(","):
            sc = _plain_or_scope(line, m, seg)
            if sc:
                out.append(sc)
            continue
        pre = re.sub(r"\b(e\.g|i\.e)\.,", r"\1. ", line[seg:m.start()])  # the comma in "e.g.," is no list comma
        choice = [c for c in _CHOICE_LEAD.finditer(pre) if ")" not in pre[c.end():]]  # not "HDLs (such as X), ..."
        if choice:  # "APIs such as Crossref, Semantic Scholar, or SSRN": the list starts after "such as"
            seg, pre = seg + choice[-1].end(), pre[choice[-1].end():]
        # commas, lead words and "and" inside a closed parenthesis belong to its item: "circuit design (PLLs and
        # clocking, SerDes), advanced packaging technologies, or thermal analysis" (NVIDIA d58843db) is one list
        comma = _paren_blank(pre).find(",")
        head = pre[:comma] if comma >= 0 else pre
        lead = list(_LIST_LEAD.finditer(_paren_blank(head)))
        start = seg + (lead[-1].end() if lead else 0)
        end = _ITEM_END.search(_paren_blank(line), m.end())
        scope = line[start:end.start() if end else len(line)]
        if re.search(r"\band\b(?!/)|&", _paren_blank(scope).replace("and/or", "")):
            continue
        out.append(scope)
    return out


def _paren_blank(s: str) -> str:
    """`s` with the inside of every closed parenthesis blanked (same length, so positions hold)."""
    return re.sub(r"\([^()]*\)", lambda m: "(" + " " * (len(m[0]) - 2) + ")", s)


def anyof_blocks(text: str) -> list[str]:
    """Stretches of the posting that offer a choice between the items they name: the scope of an
    "and/or" or "one of" inside a line (_inline_scopes), or a line with "one or more of the
    following" together with the items under it. Bullet items run until the first line that is not
    a bullet (blank lines between bullets are fine); plain-line items run until a blank line. At most
    10 items, never past a heading."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        ors = _or_bullets(lines, i)
        if ors:  # "- Strong proficiency in C, Java, or Kotlin." then "- Or experience in ..." bullets (Relay c4d18418)
            out.append("\n".join([line, *ors]))
        if not _ANYOF_LIST.search(line):
            out.extend(_inline_scopes(line))
            i += 1
            continue
        cue = _ANYOF_LIST.search(line)
        rest = line[cue.end():]
        # the items follow on the same line; "one or more of the following (key) areas:" is a header (Astranis
        # 32ef9b27, Acron 52d30fe4, Tenstorrent 23bd538f)
        if re.search(r"\w{3}", rest) and not re.fullmatch(r"\s*(?:[A-Za-z-]+\s+){0,2}[A-Za-z-]+\s*:\s*", rest):
            end = re.search(r"[.;]\s|$", line[cue.end():])
            out.append(line[cue.start():cue.end() + end.start()])
            i += 1
            continue
        block, j, bullets = [line], i + 1, None
        while j < len(lines) and len(block) <= 10:
            nxt = lines[j]
            if not nxt.strip():
                if bullets is False:
                    break
                j += 1
                continue
            is_bullet = bool(_BULLET.match(nxt))
            if bullets is None:
                bullets = is_bullet
            if is_bullet != bullets or (not is_bullet and _head_kind(nxt)):
                break
            if len(block) > 1 and _REQ_LEAD.search(_BULLET.sub("", nxt).strip()):
                break
            block.append(nxt)
            j += 1
        out.append("\n".join(block))
        i = j
    return out


def _or_bullets(lines: list[str], i: int) -> list[str]:
    """The bullets right after bullet i (blank lines between are fine) that open with "Or ": alternatives to it."""
    if not _BULLET.match(lines[i]) or re.match(r"or\b", _BULLET.sub("", lines[i]).strip(), re.I):
        return []
    out = []
    for nxt in lines[i + 1:]:
        if not nxt.strip():
            continue
        if not (_BULLET.match(nxt) and re.match(r"or\s", _BULLET.sub("", nxt).strip(), re.I)):
            break
        out.append(nxt)
    return out


def _named_in(name: str, block: str) -> bool:
    n = name.strip()
    block = block.replace("\u2010", "-").replace("\u2011", "-")  # "multi\u2010threaded"
    if not n:
        return False
    flags = 0 if len(n) <= 2 else re.I  # "C", "R", "Go": exact case
    return bool(re.search(r"(?<![\w+#])" + _name_pattern(n) + r"(?![\w+#])", block, flags))


_LAST_OR = re.compile(r"(?<!,)\s+or\s+(?!.*\sor\s)", re.I)


def _local_or(block: str, groups: list[list[str]]) -> bool:
    """"HTML, CSS, JavaScript or TypeScript" with JavaScript and TypeScript already one group: the model read
    the "or" as a choice between the two items around it only, which a list without the Oxford comma allows.
    Left as the model read it."""
    m = _LAST_OR.search(block)
    if not m:
        return False
    before = block[:m.start()].rsplit(",", 1)[-1]
    after = block[m.end():]
    return any(any(_named_in(a, before) for a in g) and any(_named_in(a, after) for a in g) for g in groups)


def _joined_by_and(groups: list[list[str]], text: str) -> bool:
    """(Lists without the Oxford comma only.) Two of the groups asked for together elsewhere ("Demonstrated aptitude in Math and Statistics" next to a
    degree line "in Math, Statistics, Physics or equivalent"): both are required, so the list is no choice."""
    pats = [[r"(?<![\w+#])" + _name_pattern(a) + r"(?![\w+#])" for a in g] for g in groups]
    for x in range(len(groups)):
        for y in range(len(groups)):
            if x != y and any(re.search(a + r",?\s+(?:and|&)\s+" + b, text, re.I) for a in pats[x] for b in pats[y]):
                return True
    return False


def _covered(block: str, groups: list[list[str]]) -> bool:
    """"with LLMs or Agents": both sides of a lone "or" are the groups' names. In "training foundation models or
    efficient machine learning" they are not, and "machine learning" there is no alternative to foundation models."""
    rest = block.replace("\u2010", "-")
    for a in sorted((a for g in groups for a in g), key=len, reverse=True):
        rest = re.sub(r"(?<![\w+#])" + _name_pattern(a) + r"(?![\w+#])", " ", rest, flags=0 if len(a) <= 2 else re.I)
    return not re.sub(r"\b(?:or|a|an|the)\b|[^\w]", "", rest, flags=re.I)


def _named_in_choice(name: str, block: str) -> bool:
    """_named_in, or (a name of two or more words) its words in order on one line of the block with at most three
    other words between two of them: "power delivery/regulation" names "power regulation", "Analog or digital
    circuit design" names "Analog circuit design" and "RTL design and debugging" names "RTL debugging" (Acron
    52d30fe4, Tenstorrent 23bd538f). "agentic, tool-using, or multi-turn conversational LLM systems" does not
    name "Agentic systems"."""
    if _named_in(name, block):
        return True
    words = [w for w in re.split(r"[\s\-]+", name.strip()) if w]
    if len(words) < 2:
        return False
    pat = re.compile(r"(?<![\w+#])" + r"(?:\W+[\w+#]+){0,3}?\W+".join(re.escape(w) for w in words) + r"(?![\w+#])", re.I)
    return any(pat.search(line) for line in block.split("\n"))


def merge_anyof(groups: list[list[str]], text: str) -> list[list[str]]:
    """Groups the model split although the posting offers a choice ("Proficiency with C/C++,
    Python, Golang, and/or Javascript/Typescript"; "experience in one or more of the following:"
    and a bullet list; "C/C++, Python, or Rust") become one group of alternatives. A group joins a
    block only when every one of its names is in that block, and a block merges only two or more groups."""
    out = [list(g) for g in groups]
    # "A, B, or C" lists count outside duty sections only: "Perform validation using Python, C++, or Rust"
    # is a task, and its list says nothing about which skills the applicant may choose between. A posting
    # with no requirements section mixes its requirements into the duties, so there every line counts.
    segs = sections(text)
    has_req = any(k == "req" and t.strip() for k, t in segs)
    lists = [sc for k, t in segs if k not in ("duty", "about") or not has_req for line in t.split("\n")
             for sc in _comma_or_scopes(line)]
    plain = {sc for sc in lists if not re.search(r",\s*or\s", sc, re.I)}  # lists without the Oxford comma
    for block in [*anyof_blocks(text), *lists]:
        idx = [k for k, g in enumerate(out) if g and all(_named_in_choice(a, block) for a in g)]
        if len(idx) < 2:
            continue
        if block in lists and (_local_or(block, [out[k] for k in idx]) or
                               ("," not in block and not _covered(block, [out[k] for k in idx]))):
            continue
        if block in plain and _joined_by_and([out[k] for k in idx], text):
            continue
        merged = list(dict.fromkeys(a for k in idx for a in out[k]))
        out[idx[0]] = merged
        for k in idx[1:]:
            out[k] = []
        out = [g for g in out if g]
    return out


_SLASH_COMPOUND = re.compile(r"\b([A-Za-z]+) ([a-z]{3,})/([a-z]{4,})\b")
_OR_ITEMS = re.compile(r",\s*(?:or\s+)?|\s+or\s+|\s+and/or\s+", re.I)
# "or other / similar / a comparable language": an open alternative, not an item the posting names.
_OPEN_ITEM = re.compile(r"(?:an?\s+|any\s+)?(?:other|another|similar|equivalent|comparable|related|relevant)\b", re.I)


def demote_or_siblings(groups: list[list[str]], nice: list[str], text: str) -> tuple[list[list[str]], list[str]]:
    """A required group offered as one choice of an or-list whose other items the card does not require
    ("algorithms, software development, or systems engineering fundamentals" with the first two generic, so in
    nice_skills): the posting accepts an item the card treats as optional, so this group is no hard requirement
    either and goes to nice_skills. Only when every requirement line naming the group is such a list: GE Vernova
    asks for generative AI in "large language models, RAG, or generative AI" and also in "machine learning
    concepts and generative AI technologies", so it stays. Returns (kept groups, demoted names)."""
    segs = sections(text)
    has_req = any(k == "req" and t.strip() for k, t in segs)
    lines = [ln for k, t in segs if (k == "req" if has_req else k != "pref") for ln in t.split("\n") if ln.strip()]
    # "chip design/development" names "chip development" too
    lines = [_SLASH_COMPOUND.sub(r"\1 \2/\1 \3", ln) for ln in lines]
    names = [a for h in groups for a in h]
    must = {skill_key(a) for a in names}
    others = [n for n in nice if skill_key(n) not in must]

    def sibling(sc: str) -> bool:  # an item of the list that is an optional skill and names no required one
        if len(re.findall(r"(?<![/\w])or\b", sc, re.I)) > 1:  # "X, Y, or Z through academic, research, or project work"
            return False
        for item in _OR_ITEMS.split(sc):
            item = item.strip(" ()")
            if not item or len(item.split()) > 5 or _OPEN_ITEM.match(item) or any(_named_in(a, item) for a in names):
                continue
            if any(_named_in(n, item) for n in others):
                return True
        return False

    def items(sc: str) -> str:  # "data structures, with proficiency in C++, Python, or X": the list after "in"
        last_or = max(sc.rfind(" or "), sc.rfind(",or "))
        lead = [m for m in _PLAIN_LEAD.finditer(sc) if m.start() < last_or]
        return sc[lead[-1].end():] if lead else sc

    keep, out = [], []
    for g in groups:
        hits = [ln for ln in lines if all(_named_in(a, ln) for a in g)]
        cov = hits and all(any(all(_named_in(a, sc) for a in g) and sibling(sc)
                               for sc in map(items, _comma_or_scopes(ln))) for ln in hits)
        (out.extend(g) if cov else keep.append(g))
    return keep, out


_PHRASE_START = re.compile(
    r"^(ability|able|willingness|desire|passion|interest|knowledge|understanding|familiarity|experience|exposure|"
    r"proficiency|comfort|building|writing|developing|designing|creating|using|working|managing|implementing|"
    r"maintaining|deploying|shipping|optimizing|leading|collaborating|communicating|analyzing|supporting|driving|"
    r"owning|applying|leveraging|delivering|solving|learning|translating|conducting|performing|ensuring|providing|"
    r"making|handling)\b")
_PHRASE_END = re.compile(r"\b(concepts?|fundamentals|principles|best practices|mindset)$")
# Category words a posting uses for a whole area, not a skill a resume would list as is: "databases",
# "AI tools", "testing". A must group made only of these is a hard Skip for nearly every resume.
# Matched on skill_key, which turns hyphens into spaces ("AI-assisted" -> "ai assisted").
_GENERIC_BARE = frozenset({"ai", "data", "analytics", "databases", "database", "testing", "coding", "programming",
                           "software", "technology", "technologies", "tools", "automation", "stakeholder engagement",
                           "stakeholder management"})
_GENERIC_TAIL = re.compile(
    r"^(ai (assisted|powered)|ai|software|data|modern|emerging|technical|server based|web based|cloud based|"
    r"business|office|productivity|internal|new|various|relevant|common|standard|industry standard|low code)\s+"
    r"(tools|technologies|applications|solutions|platforms|systems|methods|techniques)$")


def is_phrase(a: str) -> bool:
    """A name that reads like a phrase rather than a skill a resume would list (see phrase_groups)."""
    k = skill_key(a)
    return len(k.replace("/", " ").split()) > 3 or bool(_PHRASE_START.search(k) or _PHRASE_END.search(k)) \
        or k in _GENERIC_BARE or bool(_GENERIC_TAIL.match(k))


def phrase_groups(groups: list[list[str]]) -> list[list[str]]:
    """Required groups whose every name reads like a phrase rather than a skill a resume would list:
    more than three words, or starting with a gerund or "ability"/"knowledge of"-type words, or
    ending in "concepts"/"fundamentals", or only a category word ("databases", "AI tools"). On the 149
    hand-checked v2 cards this shape test catches about as many real requirements as made-up ones, so
    it only flags (meta.must_phrases); it does not move anything. A known-skill vocabulary test (O*NET
    tool, knowledge and skill names) was worse: on the 99 audit cards it would demote 58 real groups
    ("SQL", "Excel", "Java") to catch 32 made-up ones."""
    return [list(g) for g in groups if g and all(is_phrase(a) for a in g)]


# Lines that name what the hire will learn or what the company uses, not what an applicant brings:
# "you will receive training in", "interest in robotics", "Our tech stack is a blend of Python and C++".
_LEARN_LINE = re.compile(r"\b(?:you(?:'|\u2019)?ll learn|you will learn|learn(?:ing)? (?:about|how|to)|opportunit(?:y|ies) to|"
                         r"receive (?:\w+ )?training|training (?:in|on)\b|interest(?:ed)? in|passion(?:ate)? (?:for|about)|"
                         r"you(?:'|\u2019)?ll gain|you will gain|gain (?:\w+ )?(?:experience|exposure))", re.I)
_STACK_LINE = re.compile(r"\b(?:(?:our|the team'?s|current) tech(?:nology)? stack|tech(?:nology)? stack(?: is| includes|:)|"
                         r"our stack|stack is|we use|we work with|we build with|built (?:on|with)|"
                         r"tools we use|our tools|our tech)\b", re.I)
# "updating timelines", "tracking progress": a task, not a skill. Gerunds that name a field or a skill stay.
_SKILL_GERUNDS = frozenset({"engineering", "programming", "testing", "modeling", "modelling", "marketing", "accounting",
                            "scripting", "networking", "computing", "processing", "manufacturing", "machining",
                            "welding", "soldering", "budgeting", "forecasting", "auditing", "consulting", "drafting",
                            "rendering", "routing", "banking", "trading", "underwriting", "bookkeeping", "nursing",
                            "imaging", "sequencing", "cloning", "pipetting", "staining", "plating", "printing",
                            "molding", "casting", "packaging", "pricing", "mining", "lending", "recruiting",
                            "learning", "rigging", "texturing", "lighting", "editing", "sketching", "storyboarding",
                            "branding", "copywriting", "blogging", "coaching", "tutoring", "counseling", "surveying",
                            "estimating", "scheduling", "costing", "hedging", "investing", "clustering", "parsing",
                            "tracing", "profiling", "benchmarking", "fuzzing", "hashing", "caching", "indexing",
                            "load balancing", "logging", "monitoring", "alerting", "debugging", "troubleshooting"})
NICE_MAX = 8  # nice-to-haves a card keeps (synthetic cards have 1 to 3; a large list makes every resume a stretch)


def nice_segs(segs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Sections with learn and tech-stack sentences tagged "learn" (unless the sentence says preferred)."""
    out = []
    for k, t in segs:
        for sent in re.split(r"(?<=[.;])\s+|\n", t):
            if not sent.strip():
                continue
            learn = k != "pref" and (_LEARN_LINE.search(sent) or _STACK_LINE.search(sent)) and not _PREF_LINE.search(sent)
            out.append(("learn" if learn else k, sent))
    return out


def trim_nice(nice: list[str], segs: list[tuple[str, str]]) -> list[str]:
    """Nice-to-haves a recruiter would score: the preferred list, not every tool the posting names.
    Drops names found only under duty / about headings or on a learn / interest / tech-stack line,
    and phrase-like names (is_phrase). Keeps at most NICE_MAX of the others, those the posting names
    under a preferred heading or cue first, then those in the requirements, then the rest. Soft
    skills (which the rules never count) stay apart from the cap."""
    ns = nice_segs(segs)
    ranked, soft = [], []
    for pos, a in enumerate(nice):
        if is_soft(a):
            soft.append(a)
            continue
        kinds = _found_in(a, ns)
        first = skill_key(a).split()[0] if skill_key(a) else ""
        gerund = first.endswith("ing") and first not in _SKILL_GERUNDS and len(skill_key(a).split()) > 1
        if (kinds and kinds <= {"duty", "about", "learn"}) or is_phrase(a) or gerund:
            continue
        rank = 0 if "pref" in kinds else 1 if "req" in kinds else 2
        ranked.append((rank, pos, a))
    keep = {a for _, _, a in sorted(ranked)[:NICE_MAX]}
    return [a for a in nice if a in keep or a in soft]


_SENT_ANY = re.compile(r"(?<![A-Z]\.[A-Z]\.)(?<=[.;])\s+|\n")
# A choice between the names of a sentence: "or", "and/or", "one of", "such as", "e.g.", "like", a slash.
_CHOICE_WORD = re.compile(r"\bor\b|and/or|\bone of\b|\bany of\b|\beither\b|such as|e\.g\.|\blike\b|\bsimilar\b|"
                          r"\bequivalent\b|\bincluding\b|\bexamples? (?:include|are)\b|\bfor example\b|/", re.I)


def split_and_lists(groups: list[list[str]], text: str) -> list[list[str]]:
    """A group of alternatives the posting lists with "and" and no choice word ("utilizing C, C++, and
    Python") is separate requirements: each name becomes its own group. Kept as one group when any
    sentence naming all of them offers a choice."""
    sents = [x for x in _SENT_ANY.split(text) if x.strip()]
    out: list[list[str]] = []
    for g in groups:
        if len(g) < 2:
            out.append(g)
            continue
        hits = [x for x in sents if all(_named_in(a, x) for a in g)]
        choice = any(_CHOICE_WORD.search(x) for x in sents if sum(_named_in(a, x) for a in g) >= 2) or any(
            all(_named_in(a, p) for a in g) for x in hits for p in re.findall(r"\(([^()]*)\)", x))  # "(C#, Visual Basic)": examples
        if hits and not choice and any(re.search(r"\band\b|&", x) for x in hits):
            out.extend([a] for a in g)
        else:
            out.append(g)
    return out


# A list of examples: "tools such as Git, Confluence, and Jira", "(e.g., pandas, NumPy)", "Useful examples include C, ...".
_EXAMPLE_CUE = re.compile(r"\b(?:such as|e\.g\.,?|for example,?|for instance,?|examples? (?:include|are)|including|like)\s*:?\s*",
                          re.I)


# A category a parenthesis can list examples of: "programming languages (C, C++, Python)".
_CATEGORY_END = re.compile(r"\b(?:languages?|tools?|frameworks?|librar(?:y|ies)|software|packages?|platforms?|applications?|"
                           r"apps|suites?|programs?|methods?|techniques?|equipment|instruments?|technolog(?:y|ies)|stack|"
                           r"databases?|environments?|formats?|systems?|skills?|concepts?|principles?|practices?)\W*$", re.I)


def example_spans(sent: str, parents: list[str] | None = None) -> list[tuple[str, str]]:
    """(clause before the list, the list) for each list of examples in one sentence: after a cue (_EXAMPLE_CUE),
    or a parenthesis that holds a list after a category or one of `parents` ("programming languages (C, Python)",
    "Microsoft Office Suite (Outlook, Word, Excel)"). Inside a parenthesis
    the list ends at ")" and the clause is what stands before "("; otherwise the list ends at "(", ";", ")",
    "as well as", the next cue or the end of the sentence. The clause starts after the last comma, semicolon or
    colon before it (one further back when the cue follows a comma: "fundamentals, including ...")."""
    out = []
    def clause(head: str) -> str:
        b = _depth0_boundary(head, len(head))
        if b >= 0 and not re.search(r"\w", head[b + 1:]):
            b = _depth0_boundary(head, b)
        return head[b + 1:]
    for m in _EXAMPLE_CUE.finditer(sent):
        o, c = sent.rfind("(", 0, m.start()), sent.rfind(")", 0, m.start())
        if o > c:
            close = sent.find(")", m.end())
            out.append((clause(sent[:o]), sent[m.end(): close if close >= 0 else len(sent)]))
        else:
            e = re.search(r"[;()]|\.(?:\s|$)|\bas well as\b|\b(?:such as|e\.g\.)", sent[m.end():])
            out.append((clause(sent[:m.start()]), sent[m.end(): m.end() + e.start() if e else len(sent)]))
    for m in re.finditer(r"\(([^()]*,[^()]*)\)", sent):  # a bare list after a category or a required name
        head = clause(sent[:m.start()])
        if _CATEGORY_END.search(head) or any(_named_in(a, head) for a in parents or []):
            out.append((head, m[1]))
    return out


def merge_abbreviations(groups: list[list[str]], text: str) -> list[list[str]]:
    """Two single-name groups the posting writes as one name and its short form ("direct preference optimization
    (DPO)"): one group. The short form is an acronym, so "Arduino (C++)" stays two requirements."""
    out = [list(g) for g in groups]
    i = 0
    while i < len(out):
        j = 0
        while j < len(out):
            if i != j and len(out[i]) == 1 and len(out[j]) == 1 and len(out[i][0].split()) >= 2 \
                    and re.fullmatch(r"[A-Z][A-Z0-9/&-]{1,7}s?", out[j][0]) and re.search(
                    _name_pattern(out[i][0]) + r"\s*\(\s*" + _name_pattern(out[j][0]) + r"\s*\)", text, re.I):
                out[i] = [out[i][0], out[j][0]]
                del out[j]
                if j < i:
                    i -= 1
                continue
            j += 1
        i += 1
    return out


def merge_examples(groups: list[list[str]], text: str, segs: list[tuple[str, str]]) -> tuple[list[list[str]], list[str]]:
    """Single-name groups that are only examples in one list ("lab equipment such as oscilloscopes, logic analyzers
    and LCR meters" is not three requirements): one group of alternatives, or, when the clause before the list
    names another required group ("Python for data analysis (e.g., pandas, NumPy)"), dropped as examples of it.
    A name the requirements also ask for outside an example list stays its own group. Returns (groups, dropped)."""
    groups = merge_abbreviations(groups, text)
    names = [a for g in groups for a in g]
    req_text = [t for k, t in segs if k in ("req", "other")]
    def outside(name: str) -> bool:  # named in the requirements outside any example list
        for t in req_text:
            for x in _SENT_ANY.split(t):
                rest = x
                for _, span in example_spans(x, names):
                    rest = rest.replace(span, " ")
                if _named_in(name, rest):
                    return True
        return False
    dropped: list[str] = []
    for x in _SENT_ANY.split(text):
        for head, span in example_spans(x, names):
            members = [k for k, g in enumerate(groups) if len(g) == 1 and _named_in(g[0], span) and not outside(g[0])]
            if len(members) < 2:
                continue
            parent = any(_named_in(a, head) for k, g in enumerate(groups) if k not in members for a in g)
            if parent:
                dropped += [groups[k][0] for k in members]
                groups = [g for k, g in enumerate(groups) if k not in members]
            else:
                first = members[0]
                groups[first] = [groups[k][0] for k in members]
                groups = [g for k, g in enumerate(groups) if k == first or k not in members]
    return groups, dropped


def _slash_pair_only(a: str, b: str, text: str) -> bool:
    """The posting writes a and b as a slash pair ("Git/GitHub") and names neither of them elsewhere
    ("Python, SQL and Power BI ... troubleshoot DAX/SQL" names SQL on its own: two requirements)."""
    if re.sub(r"\s*/\s*", "/", f"{a}/{b}").casefold() in SLASH_BOTH | SLASH_NAMES:
        return False
    pa, pb = _name_pattern(a), _name_pattern(b)
    pair = re.compile(r"(?<![\w])(?:" + pa + r"\s*/\s*" + pb + r"|" + pb + r"\s*/\s*" + pa + r")(?![\w])", re.I)
    if not pair.search(text):
        return False
    rest = pair.sub(" ", text)
    return not (_named_in(a, rest) or _named_in(b, rest))


def merge_slash_pairs(groups: list[list[str]], text: str) -> list[list[str]]:
    """Two groups the model split although the posting writes them as one slash pair ("Git/GitHub",
    "ETL/Data Processing"): one group of alternatives, as clean_skills does for a single "A/B" group."""
    out = [list(g) for g in groups]
    i = 0
    while i < len(out):
        j = i + 1
        while j < len(out):
            pair = next(((a, b) for a in out[i] for b in out[j] if _slash_pair_only(a, b, text)), None)
            if pair:
                out[i] = list(dict.fromkeys([*out[i], *out[j]]))
                del out[j]
            else:
                j += 1
        i += 1
    return out


def _tokens_after(name: str, s: str) -> bool:
    """Every word of `name` (plural or not) is in `s`: "fact tables" in "fact and dimension tables"."""
    toks = [w[:-1] if w.endswith("s") and len(w) > 4 else w for w in re.findall(r"[A-Za-z0-9+#.]+", name) if len(w) > 2]
    return bool(toks) and all(re.search(r"(?<![\w])" + re.escape(t), s, re.I) for t in toks)


# Spreadsheet features a resume lists as "Excel", never one by one.
_EXCEL_FEATURES = frozenset({"xlookup", "vlookup", "hlookup", "index match", "pivot tables", "pivot table", "pivottables",
                             "pivots", "basic formulas", "formulas", "excel formulas", "conditional formatting",
                             "excel functions", "functions", "charts", "data validation", "lookups"})


def drop_subfeatures(groups: list[list[str]], text: str) -> tuple[list[list[str]], list[str]]:
    """Single-name groups that only spell out another required name ("Microsoft Excel including pivot
    tables, XLOOKUP, basic formulas"; "data modeling concepts, including fact and dimension tables"):
    names of two or more words after "including" in a sentence that names a required group before it,
    and spreadsheet features when the card requires Excel. Returns (groups, dropped names)."""
    keys = [{skill_key(a) for a in g} for g in groups]
    has_excel = any(any("excel" in k for k in ks) for ks in keys)
    drop: set[int] = set()
    for k, g in enumerate(groups):
        if len(g) != 1:
            continue
        if has_excel and skill_key(g[0]) in _EXCEL_FEATURES:
            drop.add(k)
            continue
        if len(skill_key(g[0]).split()) < 2:
            continue  # "Python, including pandas and NumPy": pandas is a tool a resume lists, keep it
        for x in _SENT_ANY.split(text):
            m = re.search(r"\bincluding\b", x, re.I)
            if not m or not _tokens_after(g[0], x[m.end():]) or _named_in(g[0], x[:m.start()]):
                continue
            if any(j != k and j not in drop and any(_named_in(a, x[:m.start()]) for a in h) for j, h in enumerate(groups)):
                drop.add(k)
                break
    return [g for k, g in enumerate(groups) if k not in drop], [groups[k][0] for k in sorted(drop)]


def single_example(g: list[str], segs: list[tuple[str, str]]) -> bool:
    """A one-name group the requirements name only as the one example of a category ("robotics frameworks
    such as ROS2 and physics simulation ..."): the category is required, not the example."""
    if len(g) != 1:
        return False
    n = g[0]
    flags = 0 if len(n) <= 2 else re.I
    pat = re.compile(r"(?:such as|e\.g\.,?|like|for example,?)\s+" + _name_pattern(n) +
                     r"(?=\s*(?:$|[.;)]|\s+and\s+(?-i:[a-z])|\s+(?:is|are|to|for|in|on|with)\b))", flags)
    any_pat = re.compile(r"(?<![\w+#])" + _name_pattern(n) + r"(?![\w+#])", flags)
    mentions = [(k, t) for k, t in segs if k in ("req", "other", "about") and any_pat.search(t)]
    return bool(mentions) and all(len(any_pat.findall(t)) == len(pat.findall(t)) for _, t in mentions)


# Words that ask the applicant for something: a stack line with one of them is a requirement
# ("eagerness to work in a TypeScript environment" is no ask: "experience" there is not checked).
_ASK_WORD = re.compile(r"\b(?:proficien\w*|(?<!foundational )experience[ds]? (?:with|in|using)|knowledge|familiar\w*|understanding|must|required|skills?|"
                       r"ability|comfortable|fluen\w*)\b", re.I)


def stack_only(g: list[str], segs: list[tuple[str, str]]) -> bool:
    """A group the posting names only on tech-stack lines ("Our tech stack is a blend of Python and C++")
    or "you will learn / receive training in" lines: what the team uses, not what the applicant brings."""
    lines = []
    for k, t in segs:
        for sent in re.split(r"(?<=[.;])\s+|\n", t):
            if sent.strip():
                tag = "stack" if (_STACK_LINE.search(sent) or re.search(
                    r"you(?:'|\u2019)?ll learn|you will learn|receive (?:\w+ )?training|opportunit(?:y|ies) to learn|"
                    r"eager\w* to (?:work|learn)", sent, re.I)) \
                    and not _PREF_LINE.search(sent) and not _ASK_WORD.search(sent) else k
                lines.append((tag, sent))
    kinds: set[str] = set()
    for a in g:
        kinds |= _found_in(a, lines)
    return "stack" in kinds and kinds <= {"stack", "duty", "about", "pref"}


MUST_MANY = 8  # a card with this many required groups is flagged (meta.must_many): real postings rarely mean it


# ------------------------------------------------------------------ degree fields

_VAGUE_FIELD = re.compile(
    r"\b(related|relevant|equivalent|similar|comparable|stem(?! cells?)|technical (discipline|field|degree)|"
    r"quantitative (fields?|disciplines?|backgrounds?|subjects?|areas?|degrees?)|"
    r"other (technical|engineering|scientific))\b|^(analytical|technology|quantitative|scientific|technical)$"
)


def is_vague_field(name: str) -> bool:
    k = skill_key(name)
    return bool(_VAGUE_FIELD.search(k)) or k in {"technical", "technology related", "scientific", "science technology"}


# ------------------------------------------------------------------ places

_CITY_FIX = {
    "nyc": "New York", "new york city": "New York", "manhattan": "New York", "brooklyn": "New York",
    "sf": "San Francisco", "san fran": "San Francisco", "dc": "Washington", "washington d.c.": "Washington",
    "washington dc": "Washington", "washington, d.c.": "Washington", "la": "Los Angeles", "philly": "Philadelphia",
}
_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}
_ABBR = set(_STATES.values())


def clean_location(loc: str) -> str | None:
    """One location string -> "City, ST" when it is a US city; other places come back trimmed;
    "Remote" and bare states are dropped (None)."""
    s = " ".join(str(loc).replace("–", "-").split()).strip(" .;")
    if not s or s.casefold() in {"remote", "united states", "usa", "us", "u.s.", "nationwide", "anywhere"}:
        return None
    s = re.sub(r"\s*\((remote|hybrid|on-?site)[^)]*\)", "", s, flags=re.I)
    s = re.sub(r",?\s*(USA|United States|U\.S\.A?\.?)$", "", s, flags=re.I)
    if s.casefold() == "remote":
        return None
    if "," not in s:
        if s.casefold() in _STATES or s.upper() in _ABBR:
            return None  # a bare state is no city
        return _CITY_FIX.get(s.casefold(), s)
    city, rest = [p.strip() for p in s.split(",", 1)]
    st = rest.strip(" .")
    st = _STATES.get(st.casefold(), st.upper() if st.upper() in _ABBR else st)
    city = _CITY_FIX.get(city.casefold(), city)
    if city.casefold() == "washington" and st in {"DC", "D.C", "D.C."}:
        st = "DC"
    if st.upper() == "D.C.":
        st = "DC"
    return f"{city}, {st}"


# ------------------------------------------------------------------ industry

_INDUSTRY_WORDS = (
    ("biotech and pharma", ("biotech", "pharma", "drug", "life science", "genom", "therapeutic")),
    ("healthcare", ("health", "medical", "hospital", "clinical", "medic")),
    ("finance", ("financ", "bank", "insur", "invest", "payment", "accounting", "capital", "trading", "fintech", "asset")),
    ("education", ("educat", "universit", "school", "academ", "learning")),
    ("government", ("government", "federal", "public sector", "defense", "defence", "military", "aerospace", "space")),
    ("public safety", ("public safety", "law enforcement", "security services", "emergency")),
    ("legal", ("legal", "law ")),
    ("consulting", ("consult", "advisory", "professional services")),
    ("logistics", ("logistic", "transport", "shipping", "supply chain", "freight", "delivery", "aviation", "airline")),
    ("utilities", ("energy", "utilit", "oil", "gas", "power", "solar", "nuclear", "mining", "renewable")),
    ("environmental services", ("environment", "sustainab", "climate")),
    ("construction", ("construct", "engineering services", "real estate", "infrastructure")),
    ("agriculture", ("agricult", "farm", "food", "beverage")),
    ("hospitality", ("hospitality", "travel", "restaurant", "hotel", "tourism")),
    ("retail", ("retail", "e-commerce", "ecommerce", "consumer", "apparel", "marketplace")),
    ("media", ("media", "entertainment", "gaming", "game", "advertis", "marketing", "publishing", "sports")),
    ("nonprofit", ("nonprofit", "non-profit", "foundation", "charity")),
    ("research", ("research",)),
    ("manufacturing", ("manufactur", "automotive", "industrial", "automation", "electronics", "hardware", "robotic",
                       "semiconductor", "machinery", "materials", "chemical")),
    ("technology", ("technolog", "software", "artificial", "ai", "cloud", "cyber", "internet", "data", "saas",
                    "telecom", "it ", "computing", "platform")),
)


def snap_industry(text: str, fallback: str = "technology") -> tuple[str, bool]:
    """The model's industry -> the closed list. Second value: True when the text matched the list
    directly or by keyword, False when the fallback was used."""
    t = " ".join(str(text).casefold().replace("&", " and ").split())
    if t in INDUSTRIES:
        return t, True
    for ind in INDUSTRIES:
        if re.search(rf"\b{re.escape(ind)}\b", t):
            return ind, True
    padded = f"{t} "
    for ind, words in _INDUSTRY_WORDS:
        if any(w in padded for w in words):
            return ind, True
    return fallback, False


# ------------------------------------------------------------------ license / sponsorship / salary

_LICENSE_NOT_HELD = re.compile(
    r"clearance|if applicable|ability to obtain|able to obtain|obtain|upon hire|within \d+|sie\b|series \d|vetting|eligib",
    re.I)

_NO_SPONSOR = re.compile(
    r"(without|not (?:require|need)|no)\s+(?:visa\s+)?sponsorship|"
    r"(?:will not|won'?t|unable to|cannot|can ?not|not able to|does not|do not|doesn'?t|not)\s+(?:be able to\s+)?"
    r"(?:provide\s+|offer\s+|support\s+)?(?:visa\s+|immigration\s+|employment\s+|work\s+)?(?:sponsor|sponsorship)|"
    r"not\s+(?:eligible|available)\s+for\s+(?:visa\s+|immigration\s+)?sponsorship|"
    r"(?:sponsorship|sponsor)\s+(?:is|are)\s+not\s+(?:available|offered|provided)|"
    r"\bno\b[^.\n]{0,80}\b(?:student )?visa\b[^.\n]{0,30}\bassistance|"
    # "would not require visa support including H-1B, CPT/OPT/STEM", "Must not require visa support"
    r"\b(?:not|without|no)\s+(?:\w+\s+){0,3}?(?:visa|immigration)\s+support", re.I)
# OPT named as unsupported ("including OPT or STEM OPT", "(H1B, OPT/CPT, TN etc)"); "OPT" and "CPT" only in capitals,
# so "opt in" / "opt out" do not count. A refusal of "time-limited visa status" covers OPT too.
_OPT_WORDS = re.compile(r"(?-i:\bOPT\b|\bCPT\b)|\bstem[- ]opt\b|\bi-?983\b|optional practical training|curricular practical training|"
                        r"time[- ]limited visa", re.I)
_YES_SPONSOR = re.compile(r"(?:will|can|do|does|may)\s+(?:provide\s+|offer\s+)?(?:visa\s+)?sponsor|sponsorship\s+(?:is\s+)?available",
                          re.I)
# "Sponsor" about visas: a visa, immigration or work-authorization word near it ...
_VISA_CTX = re.compile(r"visa|immigra|h-?1-?b|work(?:ing)? authori|authori[sz]ed to work|eligib\w* to work|legal(?:ly)? (?:right|authori)|"
                       r"right to work|employment (?:visa|authori|based|eligib)|for employment|now or in the future|green card|"
                       r"permanent resid|citizen|\bf-?1\b|(?-i:\bOPT\b|\bCPT\b|\bTN\b)|work permit|work in the (?:u\.?s|united states)",
                       re.I)
# ... unless the sentence is about another kind of sponsor: "U.S. government sponsors", "sponsorship for an export
# license", "communicate progress to sponsors and/or management", or says only that future sponsorship "is not guaranteed".
_OTHER_SPONSOR = re.compile(r"export licen|(?:government|project|executive|business|client|event|corporate|industry|program|"
                            r"research|study) sponsors?\b|sponsors? (?:and|or|and/or) (?:management|stakeholders)|"
                            r"to (?:the )?sponsors\b|not guaranteed|no guarantee", re.I)


# An ATS label: "Sponsorship: Yes", "VISA Sponsorship:\nNo", "Is Sponsorship Available?\nNo".
_SPONSOR_LABEL = re.compile(r"\b(?:is )?(?:visa |immigration |h-?1-?b )?sponsorship(?: available)?\s*[:?]\s*(yes|no|not available|available)\b",
                            re.I)
_SPONSOR_NEG = re.compile(r"\b(?:no|not|unable|cannot|can't|won't|neither|nor)\b", re.I)


def _sponsor_sents(text: str) -> list[tuple[str, bool]]:
    """Every sentence as (sentence, about visa sponsorship). About visa sponsorship: "sponsor" (not "sponsored"
    benefits) with a visa word in the sentence or within about 150 characters, a refusal of "sponsorship"
    ("no relocation assistance or sponsorship is provided"), or an ATS label ("Sponsorship: Yes"); not a bare
    heading ("Sponsorship") and no sign of another kind of sponsor (_OTHER_SPONSOR)."""
    text = _SPONSOR_LABEL.sub(lambda m: "Visa sponsorship is " + ("available." if m[1].lower() in ("yes", "available")
                                                                  else "not available."), text)
    out, pos = [], 0
    for s in _AUTH_SENT.split(text):
        i = text.find(s, pos)
        pos = i + len(s) if i >= 0 else pos
        refuse = bool(_NO_SPONSOR.search(s))
        visa = refuse and bool(_VISA_CTX.search(s))  # "No company-sponsored ... student visa ... assistance"
        bare = re.sub(r"\bsponsored\b", "", s, flags=re.I)
        if not visa and re.search(r"sponsor", bare, re.I) and not _OTHER_SPONSOR.search(s):
            win = text[max(0, i - 150): i + len(s) + 150] if i >= 0 else s
            neg = re.search(r"\bsponsorship\b", bare, re.I) and _SPONSOR_NEG.search(bare)
            heading = len(s.split()) <= 3 and not neg and not _VISA_CTX.search(s)  # "Sponsorship", not "- Visa Sponsorship"
            visa = not heading and bool(_VISA_CTX.search(win) or refuse or neg or _YES_SPONSOR.search(s))
        out.append((s, visa))
    return out


def visa_sponsor_sents(text: str) -> list[str]:
    """Sentences that speak about visa sponsorship (see _sponsor_sents)."""
    return [s for s, visa in _sponsor_sents(text) if visa]


def _opt_named(ss: list[tuple[str, bool]]) -> bool:
    """OPT, STEM OPT, I-983 or CPT in a visa-sponsorship sentence or the sentence before or after it, where
    that sentence does not let F-1 students in ("F1 on CPT accepted")."""
    for k, (_, visa) in enumerate(ss):
        if not visa:
            continue
        for x, _ in ss[max(0, k - 1): k + 2]:
            if _OPT_WORDS.search(x) and not (_F1_ADMITTED.search(x) and not _NEGATION.search(x)):
                return True
    return False


def repair_sponsorship(value: str, text: str) -> str:
    """Sponsorship from the posting text. Only sentences about visa sponsorship count (visa_sponsor_sents);
    "will_not_sponsor_or_opt" needs OPT, STEM OPT, I-983 or CPT named next to them (_opt_named)."""
    ss = _sponsor_sents(text)
    visa = [s for s, v in ss if v]
    if value in ("sponsors", "will_not_sponsor", "will_not_sponsor_or_opt") and not visa:
        return "unstated"  # inferred from citizenship or authorization wording, or another kind of sponsor
    if value == "unstated" and any(_NO_SPONSOR.search(s) for s in visa) and not any(_YES_SPONSOR.search(s) for s in visa):
        value = "will_not_sponsor"
    if value in ("will_not_sponsor", "will_not_sponsor_or_opt"):
        return "will_not_sponsor_or_opt" if _opt_named(ss) else "will_not_sponsor"
    return value


# Wording that asks the applicant to be studying now. Company blurbs say "university", "college",
# "co-op" and "studying" too ("each university's long-term goals", "while studying for your
# qualification" in an apprenticeship), so those words alone are no cue: the posting must say
# enrolled, pursuing a degree, working toward one, a graduation date, a year of study, and the like.
_STUDENT_WORDS = re.compile(
    r"\b(enrolled|enrol?lment|matriculated|working towards?|currently attending|"
    # "candidate for a degree", not "the right candidate for this role"
    r"candidates? for (?:an? |the |their |your )?(?:\S+ ){0,3}(?:degree|bachelor|master|ph\.?d\.?|doctora|bs\b|ms\b|b\.s\.|m\.s\.)|"
    r"near(?:ly)? complet\w*|graduat(?:e|ing) (?:with|from)\b|in (?:the )?(?:progress|process) of (?:obtaining|completing|earning)|"
    r"(?:on-?going|in-progress|in progress)\s+(?:\S+\s+){0,3}degree|majoring in|"
    r"(?:undergraduate|graduate|degree|bachelor'?s|master'?s|ph\.?d\.?|doctoral)(?:\s+or\s+(?:undergraduate|graduate))?\s+(?:degree\s+)?candidates?|"
    r"(?:degree|graduation|bachelor'?s|master'?s|ph\.?d\.?)\s+(?:is\s+|date\s+)?expected|"
    r"towards? (?:a|an|the|your|their) (?:\S+ ){0,3}(?:degree|bachelor\S*|master\S*|ph\.?d\.?|bs\w*|b\.s\.)|"
    r"(?:college|university) (?:juniors?|seniors?|sophomores?|freshm[ae]n)|"
    r"studying\b[^.\n]{0,60}\b(?:degree|bachelor|master|ph\.?d\.?|computer|engineering|science|mathematics|math|business|"
    r"finance|economics|physics|statistics|accounting)|"
    r"p[ue]rsu(?:ing|e|it)\b[^.\n]{0,80}\b(?:degree|bachelor|master|ph\.?d\.?|doctora|b\.?s\w*|b\.?a\b|m\.?s\b|studies|"
    r"program|undergrad|graduat\w*|education|major|minor)|"
    r"(?:\d+|one|two|three|four|five)(?:\.\d)?\+? years? (?:of study )?towards?|"
    # "Current undergraduate, master's, or PhD students"
    r"current(?:ly)?(?: an?)?(?: full[- ]time| part[- ]time| matriculated| registered)? (?:[\w'\u2019]+,? (?:or |and )?){0,4}students?|"
    # "Working on a BS degree", "Seeking a Bachelor's degree in CS"
    r"working on (?:a|an|your|their) (?:\S+ ){0,3}(?:degree|bachelor\S*|master\S*|ph\.?d\.?|bs\b|ms\b|b\.s\.|m\.s\.)|"
    r"seeking (?:a|an) (?:bachelor|master|ph\.?d|doctora|associate|undergraduate|graduate|bs\b|ms\b|b\.s\.|m\.s\.)\S*"
    r"(?: \S+)? degree|"
    # "Bachelor's degree in CS (3rd year and later)"
    r"(?:\d(?:st|nd|rd|th)|first|second|third|fourth|fifth)[- ]year (?:and|or) (?:later|above|beyond|up|higher)|"
    r"(?:be(?:ing)?|you are|are you) (?:an?|currently an?) (?:\S+ ){0,4}students?|"
    r"students? (?:currently )?(?:enrolled|pursuing|in good standing)|placement students?|"
    r"completing (?:a|an|your|their) [^.\n]{0,40}\bdegree|"
    r"return(?:ing)? to (?:school|studies|university|college|campus|your studies)|"
    r"(?:undergraduate|graduate|ph\.?d\.?|doctoral|master'?s|ms|m\.s\.|bs|b\.s\.|university|college|engineering|bachelor'?s)"
    r"(?:\s*(?:/|or|and)\s*(?:ph\.?d\.?|ms|m\.s\.|master'?s|bs|undergraduate|graduate))? students?|"
    r"(?:\d(?:st|nd|rd|th)|first|second|third|fourth|fifth|final|penultimate)[- ]year\b[^.\n]{0,60}"
    r"\b(?:students?|stud(?:y|ies)|undergrad\w*|degree|program|bachelor|university|college)|"
    r"penultimate|sophomores?|juniors?\b(?! (?:data|software|engineer|developer|analyst|consultant|associate|level|role|position))|"
    r"senior (?:year|standing)|rising (?:juniors?|seniors?|sophomores?)|"
    r"expected (?:graduation|to graduate|bachelor|master|ph\.?d\.?|degree)|anticipated (?:graduation|date)|"
    r"scheduled to (?:obtain|receive|graduate)|graduat(?:e|es|ing|ion)\b[^.\n]{0,40}\b20\d\d|graduation (?:date|year|term)|"
    r"current(?:ly)? (?:\S+ ){0,3}studies\b|currently studying)",
    re.I)
# "Students" alone counts, except in company blurbs: a compound ("Student Success Center", "the full
# student journey", "student-athletes"), a mission or hiring line ("helping students learn", "hiring
# co-op students ... is core to how we build"), or "students who ..." said of students in general.
_BARE_STUDENT = re.compile(r"\bstudents?\b(?![- ]?(?:journey|success|experience|loans?|debt|athletes?|affairs|services|life|"
                           r"housing|union|cent(?:er|re)s?|organi[sz]ations?|body|outcomes|data|information|records|"
                           r"engagement|retention|portal|platform|achievement|populations?|recruit\w*)\b|\s+who\b)", re.I)
_STUDENT_BLURB = re.compile(r"(?:help\w*|hir\w*|serv\w*|teach\w*|support\w*|prepar\w*|empower\w*|millions of|thousands of|k-12|"
                            r"educat\w*|connect\w*|reach\w*|enabl\w*|for all|of our|our|their|every|whose)\s+(?:\w+\s+)?$", re.I)


# Study the posting only prefers: "(Master's or PhD preferred)", "; PhD candidates preferred", "Master's degree
# candidates preferred". Taken out before looking for a study cue.
_PREF_PAREN = re.compile(r"\([^()]*\b(?:preferred|preferabl\w*|ideally|a plus|desired)\b[^()]*\)", re.I)
_PREF_CANDIDATES = re.compile(r"[^.;:\n()]{0,60}?\b(?:candidates?|students?)\s+(?:are |is )?(?:strongly |highly )?"
                              r"(?:preferred|a plus|desired)\b", re.I)


def student_cue(text: str) -> bool:
    """Whether the posting asks the applicant to be studying now (see _STUDENT_WORDS, _BARE_STUDENT). Study the
    posting only prefers ("PhD candidates preferred") does not count."""
    text = _PREF_CANDIDATES.sub(" ", _PREF_PAREN.sub(" ", text))
    if _STUDENT_WORDS.search(text):
        return True
    return any(not _STUDENT_BLURB.search(text[max(0, m.start() - 40):m.start()]) for m in _BARE_STUDENT.finditer(text))


# "Recently completed" counts only for a degree or a program, not "a research project you recently completed".
_RECENT_GRAD = re.compile(r"recent(ly)?[ -](?:(?:ph\.?d|bachelor'?s|master'?s|college|university|ms|bs) )?(?:graduat|(?:complet|acquir|obtain|earn|receiv)\w*\b[^.\n]{0,40}?"
                          r"\b(?:degree|program|bachelor|master|ph\.?d|doctora|diploma|school|studies|education|university|college|"
                          r"bs|ms|ba|b\.s|m\.s|mba|qualification))|or graduates|graduated within|graduated\b[^.\n]{0,60}\bwithin (?:the )?(?:past|last)|"
                          r"\bor graduated? within|just graduated", re.I)
# "New grad" lets graduates in only as an alternative ("students or new grads", "new grads are also welcome"),
# not in a title ("Intern New Grad"), a blurb ("our interns and new graduates") or "new grads who join full-time".
_NEW_GRAD_OK = re.compile(r"\b(?:or|and/or|also)\s+(?:\w+\s+){0,3}new grad|"
                          r"\bnew grad\w*\b[^.\n]{0,40}\b(?:are also|also|welcome to apply|are welcome|are eligible|may (?:also )?apply|"
                          r"can (?:also )?apply)\b", re.I)
# Master's study named next to the PhD ("Pursuing PhD or Master's", "predicted PhD or master's degree",
# "penultimate year of graduate study", "enrolled in a graduate degree program").
_MASTER_STUDY = re.compile(r"master['\u2019]?s|\bmasters\b|\bm\.?sc?\b|\bmfe\b|\bmba\b|(?<![3-6]-year )(?<![3-6] year )graduate (?:degree )?program|"
                           r"\badvanced degree",
                           re.I)
_STUDY_WORD = re.compile(r"pursu|enrol|candidate|predicted|expected|final[- ]year|penultimate|current|student|studying|"
                         r"working towards?", re.I)
# "Pursuing or obtained a Bachelor's", "working toward, or recently acquired, a degree": a graduate qualifies too.
_OR_HOLD = re.compile(r"\b(pursuing|working towards?|enrolled|studying)\b[^.\n]{0,30}\bor (?:have |has |already )?"
                      r"(?:recently )?(obtained|acquired|earned|completed|received|hold|holds|holding|possess)\b", re.I)
# "... or can demonstrate equivalent experience", "(or equivalent experience)": no degree study needed.
# "or equivalent degree / field / program" is about the degree, not a way around it.
_OR_EQUIV = re.compile(r"\bor (?:can )?(?:demonstrate |show )?(?:an? )?equiv\w*\s+(?!degree|program|field|discipline|major|"
                       r"qualification|course|area|subject|of\b)(?:\w+[- ]){0,2}(?:experience|learning|work|background|knowledge|"
                       r"training|skills?)\b", re.I)
# Only study wording: in "Bachelor's degree, or equivalent work experience" the equivalent stands in for a held degree.
_ENROLL_SENT = re.compile(r"pursu|enrol|working toward|studying|student", re.I)
# "enrolled in a program and have completed one course or equivalent experience": the equivalent is for the course.
_AND_MORE = re.compile(r"\band (?:have|has|had|be|must|should)\b", re.I)


def _equiv_instead_of_study(sent: str) -> bool:
    """An "or equivalent experience" that stands in for the study itself, not for a later requirement."""
    cue, eq = _ENROLL_SENT.search(sent), _OR_EQUIV.search(sent)
    return bool(cue and eq and cue.start() < eq.start() and not _AND_MORE.search(sent, cue.end(), eq.start()))


# Words that name a level of study. Without one, "currently enrolled in a degree program", "graduating
# in 2028" or "in their penultimate year" accepts any program, which the blueprint calls "associate".
# "Master" alone is no level ("masters of a diverse set of commodities", "master problem-solvers"), and
# "MS" / "BA" count only next to degree words ("MS degree", "MS or PhD"), not in "MS SQL" or "MS&A".
_LEVEL_WORD = re.compile(r"bachelor|baccalaureate|undergrad|upperclass\w*|\bb\.?tech\b|\bbs(?:ee|ce|cs|me|ie|che|cpe|e)\b|"
                         r"\bb\.?s\.?c?\b|\bb\.?a\.?(?=\s*(?:or\b|/|,|degree|in\b|student|program|\())|\bbeng\b|\bbs/ba\b|\bba/bs\b|"
                         r"sophomore|"
                         r"\bjunior(s|\s+(year|standing|or|and))\b|\bsenior(s|\s+(year|standing))\b|rising (junior|senior)|"
                         r"\b(3rd|4th|third|fourth)[- ](or \w+ )?year|four[- ]year|4[- ]year|\b(junior|senior)[- ]status\b|"
                         r"master['\u2019]s|masters? (degree|program|student|candidate|level)|master of (science|arts|engineering)|"
                         r"\bm\.?sc?\.?(?=\s*(?:or\b|/|,|degree|in\b|student|candidate|program|\())|\bmba\b|ph\.?d|doctora|"
                         r"graduate (students?|degrees?|stud)", re.I)
_VISA_SENT = re.compile(r"visa|f-?1|\bopt\b|\bcpt\b|sponsor|work authori", re.I)
_HOLD_MASTERS = re.compile(r"(holds?|holding|completed|have|has|having|earned|received|possess(?:es)?)\s+(?:already\s+)?"
                           r"(a|an|their)?\s*master", re.I)
_ANY_DEGREE_WORD = re.compile(r"bachelor|master|ph\.?d|doctora|degree", re.I)


_LEVEL_RANK = {"none": 0, "associate": 1, "bachelor": 2, "master": 3, "doctoral": 4}
_NAMES_BACHELOR = re.compile(r"bachelor|baccalaureate|undergrad\w*|\bb\.?s\.?c?\b|\bb\.?a\b|\bbeng\b|\bbs/ba\b", re.I)
_NAMES_MASTER = re.compile(r"master['\u2019]?s|\bmasters\b|master of|\bm\.?sc?\b|\bmba\b|\bmeng\b", re.I)
_NAMES_PHD = re.compile(r"ph\.?d|doctora", re.I)
_DEGREE_ASK = re.compile(r"degree|\bin\s+(?:a\s+)?[A-Za-z]|required|must|hold|possess|minimum|accept", re.I)
# "Pursuing an advanced degree, PhD preferred": what is preferred is the PhD, not the master's study.
_PHD_PREFERRED = re.compile(r"\b(?:an? )?(?:ph\.?d\.?|doctora\w*)(?: degree| candidates?| students?)?\s*(?:is |are )?"
                            r"(?:strongly |highly )?(?:preferred|a plus|desired)\b", re.I)


# "either equivalent practical experience or a Bachelor's degree": no degree is needed.
_EQUIV_EXP = re.compile(r"\bequivalen\w*\s+(?:\w+[- ]){0,2}(?:experience|work|training|background)\b", re.I)
# "Completion of at least two years of an undergraduate or graduate degree program": years of study, no degree held.
_YEARS_OF_STUDY = re.compile(r"\byears? (?:of|in|into) (?:an? |the |your |their )?(?:\S+ ){0,3}"
                             r"(?:undergrad\w*|college|university|degree|program|bachelor\S*|stud(?:y|ies)|coursework)", re.I)


def _held_level(level: str, text: str) -> str | None:
    """The degree a posting asks the applicant to hold when the model read it as study ("Bachelor's Degree in
    Computer Science", "PhD degree in computer science", "Master's degree in Electrical Engineering"): the lowest
    level named in a requirement sentence that names `level` ("PhD or master's in CS" is a master's). Preferred
    text does not count (a "Preferred" section, "preferably a May 2028 grad"), nor a degree with "or equivalent
    experience" (no degree is then needed), a sentence that also accepts study ("under pursuit or obtained") or
    asks for years of study ("two years of an undergraduate program"). None when no such sentence."""
    named = {"doctoral": _NAMES_PHD, "master": _NAMES_MASTER, "bachelor": _NAMES_BACHELOR}[level]
    sents = [x for k, t in sections(text) if k != "pref" for x in re.split(r"(?<=[.;\n])\s*", t)
             if not _HEADER_LINE.match(x)]
    found, hit = [], False
    for x in sents:
        if not _DEGREE_ASK.search(x) or _PREF_LINE.search(x) or "$" in x or _OR_EQUIV.search(x) \
                or _EQUIV_EXP.search(x) or _ENROLL_SENT.search(x) or _YEARS_OF_STUDY.search(x):
            continue  # "should be under pursuit or have been obtained": study counts too, so no degree is required
        lv = [k for k, pat in (("bachelor", _NAMES_BACHELOR), ("master", _NAMES_MASTER), ("doctoral", _NAMES_PHD))
              if pat.search(x)]
        if lv:  # "PhD / MSC preferred. BSc with high gpa ... is also accepted": a bachelor's
            found.append(min(lv, key=_LEVEL_RANK.get))
            hit = hit or bool(named.search(x))
    return min(found, key=_LEVEL_RANK.get) if hit else None


def repair_enrollment(d: dict, text: str) -> None:
    """Level of study, from the posting wording. The model reads 'internship' or 'co-op' as 'student',
    'or recent graduates' as 'master', reads a bachelor's into postings that name no level, and forgets
    that a master's or PhD student already holds a bachelor's. The held-degree adjustments for a graduate
    student run only when students qualify; when the posting asks for no study, a degree the model read as
    study is the degree required ("Bachelor's Degree in Computer Science", "PhD degree in computer science")."""
    enrolled = d.get("enrolled_in", "none")
    sents = [s for s in re.split(r"(?<=[.;\n])\s*", text) if not _HEADER_LINE.match(s)]
    if enrolled == "doctoral" and any(_MASTER_STUDY.search(x) and _STUDY_WORD.search(x) and not _HOLD_MASTERS.search(x)
                                      and not _PREF_LINE.search(x) and "$" not in x  # not a pay table
                                      for x in (_PREF_PAREN.sub(" ", _PHD_PREFERRED.sub(" ", s)) for s in sents)):
        enrolled = d["enrolled_in"] = "master"  # "Pursuing PhD or Master's": a master's student qualifies
    elif enrolled == "doctoral" and any(re.search(r"\badvanced degree", t, re.I) and not _PREF_LINE.search(t)
                                        for k, t in sections(text) if k == "req"):
        enrolled = d["enrolled_in"] = "master"  # "Requirements: Advanced degree (preferably PhD)": a master's is one
    if enrolled == "associate" and any(re.search(r"\bundergrad", s, re.I) and student_cue(s) for s in sents):
        enrolled = d["enrolled_in"] = "bachelor"  # "undergraduate studies" is a bachelor's
    if enrolled == "none":
        return
    if not student_cue(text):
        d["enrolled_in"] = "none"
        if not _ANY_DEGREE_WORD.search(text):
            d["degree_level"] = "none"
        elif enrolled in ("bachelor", "master", "doctoral"):
            held = _held_level(enrolled, text)
            if held and _LEVEL_RANK[held] > _LEVEL_RANK.get(d.get("degree_level", "none"), 0):
                d["degree_level"] = held
        return
    if any((_RECENT_GRAD.search(s) or _NEW_GRAD_OK.search(s)) and not _VISA_SENT.search(s) for s in sents) or any(
            _OR_HOLD.search(s) or _equiv_instead_of_study(s) for s in sents):
        d["enrolled_in"] = "none"  # graduates qualify too; a student still does, so the held degree below stands
    elif enrolled == "bachelor" and not _LEVEL_WORD.search(text):
        d["enrolled_in"] = enrolled = "associate"
    if enrolled in ("master", "doctoral") and d.get("degree_level") == "none":
        d["degree_level"] = "bachelor"
    if enrolled == "doctoral" and d.get("degree_level") == "master" and not _HOLD_MASTERS.search(text):
        d["degree_level"] = "bachelor"
    if enrolled == "master" and d.get("degree_level") in ("master", "doctoral"):
        d["degree_level"] = "bachelor"  # a master's student holds a bachelor's


def _has_sponsor_word(text: str) -> bool:
    """"Sponsor" in the posting, not counting benefits ("company-sponsored", "sponsored memberships")."""
    return bool(re.search(r"sponsor", re.sub(r"\bsponsored\b", "", text, flags=re.I), re.I))


# Permanent work authorization rules out every F-1 status: "permanently authorized", "permanent authorization to
# work", "authorized to work in the U.S. on a permanent basis" (the gap may hold "U.S." and a line break),
# "authorized to work permanently", "permanent US work authorization", "ongoing, permanent authorization".
# "Permanent resident" alone is no cue: "citizen, permanent resident or otherwise authorized" lets others in.
_PERMANENT_AUTH = re.compile(r"permanent(?:ly)?,? (?:legally )?authori[sz](?:ed|ation)|"
                             r"authori[sz](?:ed|ation) to work\b[\s\S]{0,80}?(?:\bpermanent basis|\bpermanently\b)|"
                             r"\bpermanent,? (?:legal )?(?:u\.?s\.? |us )?(?:work authori[sz]ation|right to work)|"
                             r"\bongoing,? permanent\b|\b(?:work|employed) permanently\b", re.I)
_CPT_REFUSED = re.compile(r"(cpt|curricular practical)[^.\n]{0,160}(not eligible|ineligible|not be considered|do not (meet|qualify))|"
                          r"(not eligible|ineligible|not be considered|do not qualify)[^.\n]{0,120}(cpt|curricular practical)", re.I)


# U.S. person, citizenship and clearance rules the blueprint counts as citizens_only: "This job requires a U.S.
# Person", "open only to U.S. Persons", "a U.S. Person is required", "U.S. citizenship is required", "Active
# Secret clearance", "ability to obtain and maintain a Top Secret clearance". Not "may require" or "preferred".
# The author's decision (2026-10-19): ITAR wording that takes a "U.S. person, or eligible to obtain an export license" ("eligible
# to obtain the required authorizations from the U.S. Department of State": SpaceX, CesiumAstro, Rocket Lab, ... 22
# training cards) is citizens_only True. An export licence for an F-1 candidate is rare in practice, so F-1 candidates
# (CPT, OPT, STEM OPT) get Skip, for export control; the real test leaves those Skips out (build_real_test.EXPORT_ROUTE,
# rules.py), because the rule's reason 'limited to US citizens or permanent residents' says more than the text.
_US = r"u\.?\s?s\.?"
_US_PERSON_ONLY = re.compile(
    r"\b(?:requires?|required to be|must be|open only to|only open to|limited to|restricted to|available only to)\b"
    r".{0,40}\b" + _US + r"\W{0,2}persons?\b|\b" + _US + r"\W{0,2}persons?\b.{0,80}\b(?:is|are|status is) required\b|"
    r"\b" + _US + r" citizenship (?:is |will be )?(?:required|mandatory)|"
    r"\bmust (?:be|hold) (?:an? )?(?:\(\w+\)\s*)?" + _US + r" citizen|"
    r"\brequires? " + _US + r" citizenship|\b(?:only|open to) " + _US + r" citizens\b|"
    r"\b(?:requires?|subject to)\b.{0,60}\b(?:itar|international traffic in arms)|"
    r"\bmust\b.{0,30}\b(?:qualify as|be) (?:an? )?\W?" + _US + r"\W{0,2}persons?\b|"
    r"\b(?:active|current|must (?:have|hold|possess|be able to obtain|obtain)|ability to obtain|able to obtain|"
    r"obtain and maintain|eligib\w* (?:to obtain|for)|requires?)\b.{0,40}\b(?:secret|top secret|ts/sci|"
    r"security|dod|government|federal) (?:security )?clearance", re.I)
# Sentences for the work-authorization checks: no break inside "U.S." or "C.F.R.".
_AUTH_SENT = re.compile(r"(?<![A-Z]\.[A-Z]\.)(?<=[.;])\s+|\n")
# Soft: "may require", "certain positions ... may need to be U.S. Persons" (Oshkosh), "may involve access to
# export-controlled information" (Moderna), "... or an alien authorized to work in the U.S." (UPS).
_SOFT_REQ = re.compile(r"may (?:be )?requir|might requir|could requir|prefer|a plus|desir|not required|is not a requirement|"
                       r"if (?:applicable|required)|some (?:of these )?positions|may be subject|might be subject|"
                       r"\bcertain (?:positions|roles|jobs)\b|may need to|may involve|might involve|"
                       r"\bor\b.{0,50}\b(?:authori[sz]ed to work|work authori[sz]ation|otherwise authori[sz]ed|eligible to work)",
                       re.I)
# ATS labels: "US Citizenship Required:\nNo" says nothing; "...:\nYes" says required.
_AUTH_LABEL = re.compile(r"((?:u\.?s\.? )?citizenship|clearance) required\s*:\s*\n?\s*(yes|no|none|not required)\b", re.I)
# "F1 on CPT accepted for masters", "international students on OPT are eligible": F-1 students may apply.
_F1_ADMITTED = re.compile(r"\b(?:f-?1|cpt|opt|international students?)\b[^.\n]{0,40}\b(?:accepted|eligible|welcome|"
                          r"considered|may apply|can apply)\b|\b(?:accept|welcome|consider)\w*\b[^.\n]{0,40}\b(?:f-?1|cpt)\b",
                          re.I)
_NEGATION = re.compile(r"\b(?:not|cannot|can't|unable|ineligible|no)\b", re.I)
_FOREIGN = re.compile(r"\b(?:uk|united kingdom|canad\w*|british)\b", re.I)
# _status_rule: the words of a citizens-or-permanent-residents rule, and what makes a sentence one
_STATUS_HOLDER = re.compile(r"citizen|\b" + _US + r"\W{0,2}persons?\b|permanent resid|green card", re.I)
_STATUS_WORD = re.compile(_STATUS_HOLDER.pattern + r"|clearance|\bitar\b|international traffic in arms|export[- ]control"
                          r"|lawfully admitted", re.I)
_STATUS_ASK = re.compile(r"requir|\bmust\b|\bonly\b|restricted to|limited to|contingent|pre-?requisite|eligib|\bneed"
                         r"|abilit(?:y|ies) to (?:obtain|get|hold|maintain)|able to (?:obtain|get|hold|maintain)"
                         r"|policy is to", re.I)
_STATUS_NOISE = re.compile(r"without regard|regardless of|citizenship status|citizenship and immigration services"
                           r"|protected (?:veteran|class|characteristic)|national origin", re.I)
_STATUS_NONE = re.compile(r"\bno\b[^.\n]{0,20}\bclearance|clearance (?:is )?not\b|\b(?:not|nor)\b[^.\n]{0,30}\b(?:requir|need)",
                          re.I)


def repair_work_auth(d: dict, text: str) -> None:
    """Sponsorship needs sponsorship words next to visa words ("sponsored" memberships do not count),
    and permanent authorization, a refusal of CPT students, a U.S. person rule, required citizenship or
    a required clearance means citizens and permanent residents only, unless the posting lets F-1
    students in ("F1 on CPT accepted")."""
    if d.get("sponsorship") in ("sponsors", "will_not_sponsor", "will_not_sponsor_or_opt") and not _has_sponsor_word(text):
        d["sponsorship"] = "unstated"
    sents = _AUTH_SENT.split(_AUTH_LABEL.sub(lambda m: f"{m[1]} is required." if m[2].lower() == "yes" else "", text))
    admitted = any(_F1_ADMITTED.search(s) and not _NEGATION.search(s) for s in sents)
    if admitted:
        d["citizens_only"] = False
        return
    if not d.get("citizens_only") and (_PERMANENT_AUTH.search(text) or _CPT_REFUSED.search(text) or any(
            _US_PERSON_ONLY.search(s) and not _SOFT_REQ.search(s) and not _NEGATION.search(s)
            and not _FOREIGN.search(s) for s in sents)):
        d["citizens_only"] = True
    # the model's True needs a stated rule too: Axon ae5e81fe names none, the National Laboratory of the Rockies
    # ff8232ae only a PIV badge paragraph (its sister posting 3f47069f is False), and Oshkosh ("Certain positions
    # ... may need to be U.S. Persons"), UPS ("... or an alien authorized to work") and Draper be5ac22c ("may be
    # required to obtain ... a clearance") only soft or open ones
    if d.get("citizens_only") and not _PERMANENT_AUTH.search(text) and not _CPT_REFUSED.search(text) \
            and not any(_status_rule(s) for s in sents):
        d["citizens_only"] = False


def _status_rule(s: str) -> bool:
    """A sentence that states a citizens-or-permanent-residents rule: a status word (citizen, U.S. person,
    permanent resident, green card, clearance, ITAR, export control) with a requirement word, or a short line
    naming the status alone ("U.S. Citizen or Permanent Residency" under Requirements). EEO lines, USCIS by name,
    soft or open wording (_SOFT_REQ), "no clearance is required" and foreign rules are not."""
    if not _STATUS_WORD.search(s) or _STATUS_NOISE.search(s) or _SOFT_REQ.search(s) or _FOREIGN.search(s):
        return False
    if _STATUS_NONE.search(s):
        return False
    return bool(_STATUS_ASK.search(s) or (_STATUS_HOLDER.search(s) and len(s.split()) <= 12))


# Remote: a role-specific statement, not boilerplate ("If this position is listed as remote ...") or a company
# trait ("remote-friendly"). Hybrid: the word, or a days-in-office schedule.
_REMOTE_NOISE = re.compile(r"(?:(?:remote|hybrid|on-?site)\s*:\s*)?if (?:this|the) (?:position|role|job) is (?:listed|posted|designated) as (?:remote|hybrid)[^.\n]*|"
                           r"remote[- ]friendly|for remote (?:roles|positions|employees)[^.\n]*|"
                           r"remote (?:work )?(?:options?|opportunities) (?:may|might|are)[^.\n]*", re.I)
_REMOTE_CUE = re.compile(r"\b(?:fully|100%|completely|entirely|100 percent) remote|remote[- ]first|"
                         r"(?:this|the) (?:\w+ )?(?:role|position|internship|job|opportunity|program) (?:is|will be|can be) "
                         r"(?:\w+ )?remote|\bremote (?:position|role|internship|opportunity|work(?:er)?|program|eligible)\b|"
                         r"work (?:from home|remotely)|work[- ]from[- ]home|\bwfh\b|virtual internship|"
                         r"(?:location|work (?:location|arrangement|model|type)|workplace type|remote type)\W+remote\b|"
                         r"\(remote\)|\bremote\s*[-,/(]\s*(?:us|usa|united states)\b|\bremote in\b|\bor remote(?:ly)?\b|open to (?:us |u\.s\. )?remote|"
                         r"(?:as well as|and) remote(?:ly)?\b|(?:^|\n)\W*remote\s*:|"
                         r"(?:telecommuting options?|work flexibility|work model|location type|work type)\W+remote\b", re.I)
_HYBRID_CUE = re.compile(r"\bhybrid\b|\b(?:\d|one|two|three|four)(?:\s*(?:-|to|or)\s*(?:\d|two|three|four|five))?\+? "
                         r"days?\b[^.\n]{0,40}\b(?:in[- ](?:the )?office|on[- ]?site|in[- ]person|at (?:the|our) office)|"
                         r"\b(?:in[- ]office|on[- ]?site|in[- ]person)\b[^.\n]{0,80}\b(?:\d|one|two|three|four)\+? days?\b|"
                         r"\d+% (?:of (?:our|your|the) time|in[- ]office|in[- ]person)|\b\d0/\d0 work model|in[- ]person time and remote|"
                         r"on[- ]?site part[- ]time|part[- ]time on[- ]?site|"
                         r"flexible work (?:arrangement|schedule|model)|partially remote|remote and (?:in[- ]office|on[- ]?site)|"
                         r"(?:on[- ]?site|in[- ]office) and remote|split (?:your time )?between|"
                         r"\b(?:remote|on[- ]?site|office|in[- ]office) days?\b|"
                         r"\bin[- ](?:the[- ])?office\b[^.\n]{0,40}\b(?:mondays?|tuesdays?|wednesdays?|thursdays?|fridays?)\b", re.I)


def repair_work_mode(d: dict, text: str, ats_locations: list[str] | None = None) -> None:
    """The model invents remote and hybrid on silent postings (Booz Allen's "If this position is listed as
    remote ..." boilerplate, an ATS location of just "United States"). Remote switches off the location
    check, so it needs an ATS location that says remote or a role-specific remote statement; hybrid
    needs the word or a days-in-office schedule; otherwise the posting is onsite."""
    mode = d.get("work_mode")
    if mode not in ("remote", "hybrid"):
        return
    body = _REMOTE_NOISE.sub(" ", text)
    remote = any(re.search(r"remote|work from home|virtual", str(x), re.I) for x in ats_locations or []) \
        or bool(_REMOTE_CUE.search(body))
    hybrid = bool(_HYBRID_CUE.search(body))
    if mode == "remote" and not remote:
        d["work_mode"] = "hybrid" if hybrid else "onsite"
    elif mode == "hybrid" and not hybrid and not remote:  # "our Sunnyvale office or remotely": hybrid stays
        d["work_mode"] = "onsite"


_STEM_WORDS = {"science", "technology", "engineering", "mathematics", "math"}
_VAGUE_SKILL = re.compile(r"^(building|reason about|advising|shipping|designing|developing|working with|managing)\s+\w+|\bexperience$",
                          re.I)


def repair_vague(d: dict) -> None:
    """STEM spelled out is no field. Phrases that are experience statements are no skill."""
    fields_ = d.get("degree_fields") or []
    if len({skill_key(f) for f in fields_} & _STEM_WORDS) >= 3:
        d["degree_fields"] = [f for f in fields_ if skill_key(f) not in _STEM_WORDS]
        d["related_ok"] = True
    if d.get("must_skills"):
        keep = []
        for g in d["must_skills"]:
            g2 = [a for a in g if not _VAGUE_SKILL.search(a.strip())]
            if g2:
                keep.append(g2)
        d["must_skills"] = keep


# ------------------------------------------------------------------ country and language

_CA = re.compile(r"\b(canada|ontario|quebec|qu[eé]bec|british columbia|alberta|toronto|vancouver|montr[eé]al|ottawa|calgary|waterloo)\b", re.I)
_UK = re.compile(r"\b(united kingdom|england|scotland|wales|ireland|london|dublin|edinburgh|manchester|cambridge, uk|uk)\b", re.I)
_EU = re.compile(r"\b(france|paris|germany|munich|berlin|spain|madrid|barcelona|switzerland|zurich|z[uü]rich|zug|netherlands|"
                 r"amsterdam|leiden|italy|sweden|poland|israel|india|bangalore|bengaluru|hyderabad|singapore|japan|tokyo|"
                 r"korea|seoul|china|beijing|shanghai|australia|sydney|brazil|(?<!new )mexico)\b", re.I)
_CA_PROV = re.compile(r",\s*(ON|BC|AB|QC|MB|NS|NB|SK)\s*$")
_US_STATE_TAIL = re.compile(r",\s*(%s)\s*$" % "|".join(sorted(_ABBR)))
_US_WORDS = re.compile(r"united states|\busa\b|\bu\.s\.", re.I)
# "Remote - EMEA": a remote posting outside the US.
_NON_US_REGION = re.compile(r"\b(emea|apac|latam|europe|asia|africa|middle east|anz)\b", re.I)
# ATS shorthands and US cities that come without a state ("NYC", "SF", "South SF").
_US_BARE = {"nyc", "sf", "la", "dc", "new york", "san francisco", "los angeles", "south sf", "south san francisco", "washington",
            "philadelphia", "silicon valley", "bay area", "sf bay area"}
# French function words, lowercase only (case-sensitive). English postings never use them; "experience"
# and "stage" are English words too and must not count.
_FRENCH = re.compile(r"\b(nous|vous|notre|votre|avec|dans|chez|compétences|équipe|poste|sont|pour les)\b")
FRENCH_MIN_HITS = 4


def location_country(loc: str) -> str | None:
    """One ATS or card location -> "US", "CA", "UK", "other", or None (cannot tell, or "Remote")."""
    x = " ".join(str(loc).split())
    if not x:
        return None
    if _US_WORDS.search(x):
        return "US"
    if _US_STATE_TAIL.search(x):  # "Toronto, CA" is Canada, not California
        return "CA" if re.search(r",\s*CA\s*$", x) and _CA.search(x) else "US"
    if _CA_PROV.search(x) or _CA.search(x):
        return "CA"
    if _UK.search(x):
        return "UK"
    if _EU.search(x):
        return "other"
    k = x.casefold().strip(" .")
    city = (clean_location(x) or "").casefold()
    if k in _STATES or k in _US_BARE or city in _US_BARE or _US_STATE_TAIL.search(clean_location(x) or ""):
        return "US"
    return None


def posting_meta(locations: list[str], text: str, card_locations: list[str] | None = None) -> dict:
    """Country and language from the posting itself. us: at least one US location, or none stated
    with no foreign place named (remote US postings). When the ATS locations say nothing usable,
    the card's cleaned locations ("City, ST") decide."""
    locs = [str(x) for x in locations or []]
    kinds = [location_country(x) for x in locs]
    if not any(kinds) and card_locations:
        locs = [str(x) for x in card_locations]
        kinds = [location_country(x) for x in locs]
    us = [x for x, k in zip(locs, kinds) if k == "US"]
    if us:
        country = "US"
    elif "CA" in kinds:
        country = "CA"
    elif "UK" in kinds:
        country = "UK"
    elif "other" in kinds:
        country = "other"
    else:
        country = "US" if not locs or any(re.search(r"remote", x, re.I) and not _NON_US_REGION.search(x) for x in locs) \
            else "unknown"
    hits = len(_FRENCH.findall(text[:4000]))
    return {"country": country, "us": country == "US", "mixed": bool(us) and len(us) < len(locs),
            "language": "fr" if hits >= FRENCH_MIN_HITS else "en"}


def training_ok(meta: dict | None) -> bool:
    """Whether a real card's row may go into training pairs: a US posting in English with no
    phrase-like required group (meta.must_phrases), fewer than MUST_MANY required groups (meta.must_many),
    not a second copy of an identical posting (meta.dup_of) and not a second row of one company's description
    under another title or location line (meta.same_description_as), and not a posting of the real-posting test set
    or one with its description (meta.in_real_test). Rows without meta (synthetic cards) are fine."""
    if not meta:
        return True
    return bool(meta.get("us", True)) and meta.get("language", "en") == "en" and not meta.get("must_phrases") \
        and not meta.get("dup_of") and not meta.get("must_many") and not meta.get("same_description_as") \
        and not meta.get("in_real_test")


# ------------------------------------------------------------------ the card

RELATED_FIELD = "related field"
# ATS shorthands -> "City, ST".
_BARE_CITY_STATE = {"nyc": "New York, NY", "new york": "New York, NY", "sf": "San Francisco, CA",
                    "san francisco": "San Francisco, CA", "la": "Los Angeles, CA", "los angeles": "Los Angeles, CA",
                    "dc": "Washington, DC", "south sf": "South San Francisco, CA",
                    "south san francisco": "South San Francisco, CA", "philadelphia": "Philadelphia, PA"}


def _ats_city(loc: str) -> str | None:
    """One ATS location -> "City, ST" for a US city, else None (states, countries, "Remote")."""
    c = clean_location(loc)
    if not c:
        return None
    if "," not in c:
        return _BARE_CITY_STATE.get(c.casefold())
    return c if re.search(r", [A-Z]{2}$", c) and location_country(c) == "US" else None


def repair_locations(d: dict, posting_locations: list[str] | None, text: str | None = None) -> None:
    """An onsite or hybrid card with no location loses the location check: take the ATS's US cities. A
    card city with no state ("Overland Park") takes the state of the ATS city with that name, or else the state the
    posting text names after it ("Fort Myers, Florida"). A state-only
    ATS location ("California") stays no location: a card city the model made up for it (Highmark's 51 state
    capitals, "Georgia" -> "Atlanta, GA", "Washington" -> "Washington, DC") is dropped unless the posting body
    names that city (for Washington, DC: "Washington, DC" or "D.C.")."""
    states = [x for x in posting_locations or [] if " ".join(str(x).split()).strip(" .").casefold() in _STATES]
    if states and d.get("locations") and text is not None:
        body = "\n".join(ln for ln in text.split("\n") if not _HEADER_LINE.match(ln))
        named = {c.casefold() for c in (clean_location(x) for x in posting_locations or [] if x not in states) if c}

        def stated(loc: str) -> bool:
            if loc.casefold() in named or (loc.split(",")[0].casefold() in {n.split(",")[0] for n in named}):
                return True
            city = loc.split(",")[0].strip()
            if city.casefold() == "washington":
                return bool(re.search(r"washington,?\s*d\.?c\b|\bd\.c\.", body, re.I))
            return bool(re.search(r"(?<!\w)" + _name_pattern(city) + r"(?!\w)", body, re.I))

        d["locations"] = [loc for loc in d["locations"] if stated(loc)]
    ats = [c for c in (_ats_city(x) for x in posting_locations or []) if c]
    if not ats or d.get("work_mode") == "remote" or "locations" not in d:
        return
    if not d["locations"]:
        d["locations"] = list(dict.fromkeys(ats))
        return
    fixed = []
    for loc in d["locations"]:
        if "," not in loc:
            loc = next((a for a in ats if a.split(",")[0].casefold() == loc.casefold()), loc)
        if "," not in loc and text:  # "Fort Myers" with ATS "Fort Worth, TX": the state the posting gives the city
            m = re.search(r"(?i:" + _name_pattern(loc) + r"),?\s+((?i:" + "|".join(sorted(_STATES, key=len, reverse=True))
                          + r")|" + "|".join(sorted(_ABBR)) + r")\b", text)  # "TX", not "or"
            if m:
                st = m[1] if m[1] in _ABBR else _STATES[m[1].casefold()]
                loc = f"{loc}, {st}"
        fixed.append(loc)
    d["locations"] = list(dict.fromkeys(fixed))


_RATE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:usd\s*)?(?:/|per|an|a)\s*(hour|hr|week|wk|month|mo)\b", re.I)
_PER_YEAR = {"hour": 2080, "hr": 2080, "week": 52, "wk": 52, "month": 12, "mo": 12}


_PERIOD_WORD = re.compile(r"\b(hour|hourly|hr|week|weekly|month|monthly)\b", re.I)
_PERIOD_YEAR = {"hour": 2080, "hourly": 2080, "hr": 2080, "week": 52, "weekly": 52, "month": 12, "monthly": 12}
_AMOUNT = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)(?![\d,]*\s*[kKmMbB]\b)")


PAY_FLOOR = 20000  # an annual pay under this is a monthly or per-term figure unless the posting names the period


def repair_pay(d: dict, text: str) -> None:
    """Annual pay the model worked out from an hourly, weekly or monthly figure, recomputed in code: the model's
    arithmetic is off by a few dollars (Hadrian 96051 for $46.15/hour x 2080 = 95992). The period is the one
    the line names ("the target hourly range is $46.15 - $50.71", "$1,800 per week")."""
    rates = [float(m[1].replace(",", "")) * _PER_YEAR[m[2].lower()] for m in _RATE.finditer(text)]
    for line in re.split(r"\n|(?<=[.;])\s+(?=[A-Z])", text):
        per = _PERIOD_WORD.search(line)
        if per:
            rates += [float(m[1].replace(",", "")) * _PERIOD_YEAR[per[1].lower()] for m in _AMOUNT.finditer(line)]
    stated = {round(float(m[1].replace(",", ""))) for m in _AMOUNT.finditer(text)}
    for k in ("salary_min", "salary_max"):
        v = d.get(k)
        if not isinstance(v, (int, float)) or v <= 0 or round(v) in stated:  # an annual figure the posting states
            continue
        near = [r for r in rates if abs(r - v) / v < 0.01]
        if near:
            d[k] = int(round(min(near, key=lambda r: abs(r - v))))
    for k in ("salary_min", "salary_max"):
        v = d.get(k)
        # "$10,000-$10,000 USD" for a summer intern is monthly or per term, not a year's pay: under PAY_FLOOR a figure
        # counts only when the posting gives its period ("$1,500 monthly stipend" x 12, "$8.25/hour" x 2080)
        if isinstance(v, (int, float)) and 0 < v < PAY_FLOOR and not any(abs(r - v) / v < 0.01 for r in rates):
            d["salary_min"] = d["salary_max"] = None


_YEARS = re.compile(r"\b(\d{1,2})(?:\.\d)?\s*(?:\+|(?:-|to|or more|plus)\s*\d*)?\s*\+?\s*(?:years?|yrs?)\b", re.I)


def repair_min_years(d: dict, segs: list[tuple[str, str]]) -> None:
    """A minimum the posting states only as preferred ("Preferred Qualifications: 1 or more years relevant work
    experience") is no minimum: min_years drops to the largest number of years asked outside preferred text, or 0.
    Only sentences about experience count ("18 years or older" is no experience)."""
    v = d.get("min_years")
    if not isinstance(v, int) or v <= 0:
        return
    req, pref = set(), set()
    for k, t in segs:
        for sent in re.split(r"(?<=[.;])\s+|\n", t):
            if re.search(r"experience", sent, re.I):
                (pref if k == "pref" else req).update(int(m[1]) for m in _YEARS.finditer(sent))
    if v in pref and v not in req:
        d["min_years"] = max([n for n in req if n < v], default=0)


# Seniority words a model writes outside the four levels ("director": a v3 completion on fresh test 1), by level.
_SENIORITY_WORDS = {"lead": ("director", "head", "principal", "executive", "vp", "vice president", "manager", "staff"),
                    "entry": ("junior", "associate", "intern", "internship", "graduate", "trainee"),
                    "senior": ("sr", "senior level", "expert"), "mid": ("intermediate", "mid level", "mid-level")}


def repair_seniority(d: dict) -> None:
    """A seniority outside SENIORITIES becomes the level its word names, else "mid" (the card default)."""
    if "seniority" not in d:
        return
    v = str(d["seniority"] or "").strip().casefold()
    d["seniority"] = v if v in SENIORITIES else next((lvl for lvl, ws in _SENIORITY_WORDS.items() if v in ws), "mid")


def clean_card(data: dict, text: str | None = None, posting_locations: list[str] | None = None) -> dict:
    """Repair a model-produced card dict. `text` is the posting text; with it, sponsorship is
    cross-checked against the wording and non-US salaries are nulled. `posting_locations` are the
    ATS locations, a better country signal than the card's. Returns a new dict."""
    d = dict(data)
    if "must_skills" in d or "nice_skills" in d:
        d["must_skills"], d["nice_skills"] = clean_skills(d.get("must_skills"), d.get("nice_skills"))
    if "degree_fields" in d:
        names = [_clean_name(x) for x in d["degree_fields"] or []]
        vague = [x for x in names if is_vague_field(x)]
        keep = [x for x in dict.fromkeys(names) if x and not is_vague_field(x)]
        d["degree_fields"] = keep
        if vague or d.get("related_ok"):
            d["related_ok"] = True
        else:
            d["related_ok"] = False
    if "locations" in d:
        locs = [clean_location(x) for x in d["locations"] or []]
        d["locations"] = list(dict.fromkeys(x for x in locs if x))
    if "industry" in d:
        d["industry"] = snap_industry(d["industry"])[0]
    if d.get("license"):
        lic = _clean_name(d["license"])
        d["license"] = None if _LICENSE_NOT_HELD.search(lic) else lic
    for k in ("salary_min", "salary_max"):
        v = d.get(k)
        if isinstance(v, (int, float)) and 0 < v <= 400:  # an hourly rate left as is
            d[k] = int(round(v * 2080))
    if d.get("salary_min") and d.get("salary_max") and d["salary_min"] > d["salary_max"]:
        d["salary_min"], d["salary_max"] = d["salary_max"], d["salary_min"]
    repair_vague(d)
    repair_seniority(d)
    if text is not None:
        repair_enrollment(d, text)
        repair_work_auth(d, text)
        repair_work_mode(d, text, posting_locations)
        segs = sections(text)
        repair_min_years(d, segs)
        if "must_skills" in d:
            keep, demote = [], []
            for g in d["must_skills"]:
                w = where(g, segs)
                ok = w == "keep" and not duty_only(g, segs) and not stack_only(g, segs) and not single_example(g, segs)
                (keep if ok else demote).append(g)
            keep = merge_slash_pairs(split_and_lists(keep, text), text)
            keep, sub = drop_subfeatures(merge_anyof(keep, text), text)
            keep, ex = merge_examples(keep, text, segs)
            demote += [[a] for a in ex + sub]
            keep, sib = demote_or_siblings(keep, [*d.get("nice_skills", []), *(a for g in demote for a in g)], text)
            d["must_skills"] = keep
            demote += [[a] for a in sib]
            have = {skill_key(a) for a in d.get("nice_skills", [])}
            d["nice_skills"] = [*d.get("nice_skills", []),
                                *(a for g in demote for a in g if skill_key(a) not in have)]
        if "nice_skills" in d:
            d["nice_skills"] = trim_nice([a for a in d["nice_skills"] if not is_soft(a) or soft_in_text(a, text)], segs)
        dsegs = degree_segs(segs)
        if d.get("degree_fields"):
            fields_ = [f for f in d["degree_fields"] if where([f], dsegs) == "keep"]
            d["degree_fields"] = fields_
        if not d.get("degree_fields"):
            d["related_ok"] = False
        if d.get("sponsorship"):
            d["sponsorship"] = repair_sponsorship(d["sponsorship"], text)
        meta = posting_meta(posting_locations if posting_locations is not None else d.get("locations", []), text,
                            d.get("locations"))
        if not meta["us"] and meta["country"] != "unknown":
            d["salary_min"] = d["salary_max"] = None  # other currencies
    if text is not None:
        repair_locations(d, posting_locations, text)
        repair_pay(d, text)
    if d.get("related_ok") and d.get("degree_fields"):
        # the rules read "or a related field" from an entry in degree_fields (degrees.parse_fields), as on the
        # v1 and synthetic cards, and the rendered posting then says it too; related_ok keeps the flag
        d["degree_fields"] = [*d["degree_fields"], RELATED_FIELD]
    return d


def parse_card(posting_id: str, text: str, posting: str | None = None,
               posting_locations: list[str] | None = None) -> RequirementCard:
    """One extraction output -> a validated, cleaned RequirementCard. Raises on bad keys or values.
    `posting` is the posting text, used for the sponsorship cross-check. related_ok may be left out."""
    data = json.loads(text)
    data.setdefault("related_ok", False)
    keys = set(data)
    if any(not isinstance(g, list) for g in data.get("must_skills", [])):
        raise ValueError("must_skills must be a list of groups (lists of alternatives)")
    for k in ("must_skills", "nice_skills"):
        if any(not isinstance(a, str) for a in _flat(data.get(k) or [])):
            raise ValueError(f"{k}: every skill name must be a string")
    if keys != set(OUTPUT_KEYS):
        raise ValueError(f"keys differ: missing {set(OUTPUT_KEYS) - keys}, extra {keys - set(OUTPUT_KEYS)}")
    data = clean_card(data, posting, posting_locations)
    return RequirementCard.from_dict({**data, "source_posting_id": posting_id, "onet_soc": None})
