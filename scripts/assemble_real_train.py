"""Step 3 of the real-posting build: screen the real-posting pairs, then the training CSV and the real dev set.

The pairs in data/built/real_train/{train,dev}.jsonl (build_real_train.py assemble) carry labels from rules.label on
the extracted card. Candidates are built to meet that card, so a requirement the posting states and the card leaves
out turns a Skip into a Keep or an Explore (the label read: Keep 68-76% right, Skip 33/33). Skip rows are not
screened: a row the card already fails stays failed.

  assemble_real_train.py screen
      Reads every non-Skip row against the full posting text and the candidate card and drops it when a screen hits
      (SCREENS, in that order; the first hit is recorded):
        location        the posting's own locations (the Title/Company/Locations header) name places, none remote, and
                        none matches where the candidate lives or will move (metros.places_match, the rule's matcher)
        student_only    a graduated candidate, and the posting asks for current students (enrolled, pursuing,
                        returning to school, student worker) with no word for recent graduates
        class_standing  the posting asks for junior/senior standing or years of study done, and an enrolled
                        bachelor's candidate is not that far along (junior: expected 2029 or later, 2030 for "rising";
                        senior: 2028 or later, 2029 for "rising"; "sophomore" in the sentence adds a year; a sentence
                        that names a year is left to grad_window)
        grad_window     the posting gives graduation years (or "within a year") that the candidate's year is outside
        prior_internship  the posting requires a prior internship or co-op and the resume names none
        school_only     the posting is open only to one school's students and the resume does not name that school
        permanent_auth  a candidate who is not a citizen or permanent resident, and the posting asks for unlimited or
                        permanent authorization or says CPT/OPT/F-1 students do not qualify (rules.py reads that
                        wording as citizens_only). "No sponsorship now or in the future" alone is the rule's
                        will_not_sponsor: CPT on an internship stays Keep and OPT an Explore, as in the real test
        no_sponsor      a candidate who needs H-1B now, and the posting says it does not sponsor (other_visa stays the
                        rule's soft check)
        citizen_only    a candidate who is not a citizen or permanent resident, and the posting requires U.S.
                        citizenship, a U.S. person, a security clearance or ITAR status
        industry_conflict  the candidate avoids an industry whose words the posting uses 2 or more times (3 for
                        technology, 4 for research; the rule's industry word lists; pref_industry_avoided rows already
                        say so and are not screened)
        soft_nice_skill nice_skill rows whose missing nice-to-have is a soft skill (the prompt says soft skills
                        never decide the label)
        broad_degree_field  degree_field_adjacent rows on a posting that takes any technical, STEM or engineering
                        discipline (an engineering field only for "engineering discipline")
      Writes data/built/real_train/screened_{train,dev}.jsonl (kept rows) and screen_dropped.jsonl (dropped rows,
      with the screen and the posting text that hit), and prints counts per screen, split, bucket and kind.
  assemble_real_train.py assemble
      Training CSV: every row of data/built/training/smoke.csv (md5 checked first) and every screened real train row,
      prompt and completion only, in one seeded shuffle, written to data/built/real_train/train_mix.csv. Real dev:
      data/built/real_train/real_dev.jsonl with data/built/real_test/test.jsonl's keys (screened rows are dropped, not
      marked decidable False: that flag means the prompt does not show a fact). Before writing, the leakage, duplicate
      and shortcut checks (checks) go to data/built/real_train/checks.json; a leak or a duplicate stops the build,
      shortcut scores are reported. Counts go to assemble_report.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import build_real_test as brt  # noqa: E402
from matcher.metros import parse_place, places_match  # noqa: E402
from matcher.pairs import prompt_parts  # noqa: E402
from matcher.rules import _INDUSTRY_RE, _industry_text, industry_ids  # noqa: E402
from matcher.schema import CURRENT_YEAR, CandidateCard, is_soft_skill  # noqa: E402

OUT = ROOT / "data/built/real_train"
SYNTH = ROOT / "data/built/training/smoke.csv"
SYNTH_MD5 = "d6ebe08672212a77d32ca4d4f5460572"
TEST = ROOT / "data/built/real_test/test.jsonl"
SPLITS = ("train", "dev")
SKIP = "Excluded"
CITIZEN = "citizen_or_pr"
TEST_KEYS = ("pair_id", "rq_id", "base_rq_id", "cd_id", "bucket", "variant_kind", "completion", "flags", "decidable",
             "missing", "prompt")


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- posting sentences

_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
PREFERRED = re.compile(r"\bprefer|\bplus\b|\bbonus\b|nice to have|nice-to-have|\bideal(ly)?\b|\bdesired?\b|"
                       r"\badvantage|\bencourag", re.I)
EEO = re.compile(r"without regard|regardless of|discriminat|equal (employment )?opportunit|protected (veteran|status|"
                 r"class|characteristic)|affirmative action", re.I)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(text) if s and s.strip()]


def first_hit(rx: re.Pattern, text: str, skip_preferred: bool = True, skip_eeo: bool = False) -> str | None:
    """The first sentence of `text` that rx matches (not a preferred or EEO sentence when asked), or None."""
    for s in sentences(text):
        if rx.search(s) and not (skip_preferred and PREFERRED.search(s)) and not (skip_eeo and EEO.search(s)):
            return s[:300]
    return None


# ---------------------------------------------------------------- screens

def _place(text: str) -> str:
    """"Lake St Louis, MO" and "Lake Saint Louis, MO" are one place to places_match."""
    return re.sub(r"\bSt\.? (?=[A-Z])", "Saint ", text)


def screen_location(c: CandidateCard, raw: dict) -> str | None:
    locs = [x for x in (raw.get("locations") or []) if x and x.strip()]
    if not locs or any(re.search(r"remote|anywhere|united states|nationwide|\busa?\b", x, re.I) for x in locs):
        return None
    locs = [_place(re.sub(r"\bSF\b", "San Francisco, CA", x) if x.strip() != "SF" else x) for x in locs]
    mine = [_place(m) for m in (c.current_location, *c.willing_locations) if m]
    if any(parse_place(m).kind == "country" for m in mine if m):
        return None
    if any(places_match(m, x) for m in mine if m for x in locs):
        return None
    return f"posting locations {locs}; candidate in {c.current_location}, will move to " \
           f"{list(c.willing_locations) or 'nowhere else'}"


STANDING_JUNIOR = re.compile(
    r"\bjunior[- ](?:level[- ])?(?:standing|status|class(?:ification)?|year)\b|\brising juniors?\b|"
    r"\bjuniors? (?:or|and|/|,) (?:seniors?|graduate)|\bat least (?:a )?junior\b|\bjuniors? or above\b|"
    r"\b(?:completed|complete|finished|completion of) (?:at least |a minimum of |minimum )?(?:two|2|three|3|four|4|"
    r"sixty|60) (?:full )?(?:academic )?(?:years?|semesters?|credit hours|credits) of (?:college|university|"
    r"undergraduate|your|their|academic|coursework|study|studies|school|post-secondary)|"
    r"\b(?:entering|going into|in) (?:your|their|the) (?:junior|third|3rd) year\b|\bthird[- ]year (?:students?|undergrad)",
    re.I)
STANDING_SENIOR = re.compile(
    r"\bsenior[- ](?:standing|status|class(?:ification)?)\b|\brising seniors?\b|\b(?:entering|going into|in) (?:your|their|"
    r"the) (?:senior|fourth|4th|final) year\b|\bfinal[- ]year (?:students?|undergrad)", re.I)
# Standing by the internship (summer CURRENT_YEAR + 1), not at application: one year further along.
RISING = re.compile(r"\brising\b|\bentering\b|\bgoing into\b|\bby (?:the )?(?:summer|start|internship)", re.I)


def screen_class_standing(c: CandidateCard, text: str) -> str | None:
    """An enrolled bachelor's candidate not yet at the standing the posting asks. In the fall of CURRENT_YEAR a
    student expected in CURRENT_YEAR + 2 is a junior and one expected in CURRENT_YEAR + 1 a senior; "rising" (by the
    summer internship) adds a year. A sentence that names a year is left to grad_window."""
    gy = c.graduation_year
    if c.degree_level != "bachelor" or not gy or gy <= CURRENT_YEAR:
        return None
    for s in sentences(text):
        if PREFERRED.search(s) or YEAR.search(s):
            continue
        # the lowest standing the sentence names decides: "Junior or Senior status" takes a junior,
        # "a Sophomore, Junior or Senior standing" a sophomore
        extra = 1 if RISING.search(s) else 0
        if STANDING_JUNIOR.search(s):
            need = CURRENT_YEAR + (4 if re.search(r"sophomore", s, re.I) else 3)
        elif STANDING_SENIOR.search(s):
            need = CURRENT_YEAR + 2
        else:
            continue
        if gy >= need + extra:
            return f"expected {gy}; posting: {s[:300]}"
    return None


GRAD = re.compile(r"\bgraduat\w*|\bdegree completion\b|\bcomplet\w* (?:of )?(?:your|their|the|a) (?:\w+ )?degree", re.I)
YEAR = re.compile(r"\b(20[2-3]\d)\b")
WITHIN = re.compile(r"(?:graduat\w*|degree completion|complet\w* (?:of )?(?:your|their|the|a) (?:\w+ )?degree)"
                    r"[^.]{0,40}\bwithin (?:one|a|1|twelve|12|six|6) (?:year|months)", re.I)
LATER = re.compile(r"\bnot (?:\w+ ){0,3}?(?:before|earlier|prior)|\bno earlier\b|or (?:later|after|beyond)|"
                   r"and (?:later|after|beyond)|onwards?|\bafter\b|\bthrough at least\b|\bat least\b", re.I)
EARLIER = re.compile(r"or (?:earlier|before|sooner)|\bbefore\b|no later than|\bby\b|\bprior to\b", re.I)


def grad_years_clause(s: str) -> tuple[list[int], str] | None:
    """The years right after a graduation word in a sentence, and the clause from 30 characters before that word
    ("must not graduate before Fall 2027": the "not" is part of it). None when no year follows a graduation word."""
    m = GRAD.search(s)
    if not m:
        return None
    ys = [int(y) for y in YEAR.findall(s[m.start(): m.end() + 90])]
    return (ys, s[max(0, m.start() - 30): m.end() + 90]) if ys else None


def screen_grad_window(c: CandidateCard, text: str) -> str | None:
    gy = c.graduation_year
    if not gy:
        return None
    for s in sentences(text):
        if PREFERRED.search(s):
            continue
        if gy > CURRENT_YEAR + 2 and WITHIN.search(s):
            return f"graduation {gy}; posting: {s[:300]}"
        got = grad_years_clause(s)
        if not got:
            continue
        ys, clause = got
        head = clause[clause.lower().find("graduat") if "graduat" in clause.lower() else 0: clause.find(str(ys[0]))]
        if re.search(r"intern|summer|co-?op|program|start", head, re.I) and \
                not re.search(r"graduat\w* (?:in|between|by|from|date|no|before|after|on or)", clause, re.I):
            continue  # "graduating students for our Summer 2027 program": the year is the program's
        lo, hi = min(ys), max(ys)
        if LATER.search(clause):
            hi = 9999
        elif EARLIER.search(clause) and len(ys) == 1:
            lo = 0
        if not lo <= gy <= hi:
            return f"graduation {gy}; posting: {s[:300]}"
    return None


ENROLLED = re.compile(
    r"\b(?:currently|actively) (?:enrolled|pursuing|attending)\b|\bmust be (?:a |an )?(?:current(?:ly)? )?(?:enrolled|"
    r"student|pursuing|undergraduate|graduate student|attending)\b|\bstudent workers?\b|\bcurrent(?:ly enrolled)? "
    r"(?:\w+ )?students?\b|\benrolled (?:full[- ]time |part[- ]time |as a student )?(?:in|at) (?:an? |the )?(?:accredited )?"
    r"(?:\w+ )?(?:college|university|bachelor|master|undergraduate|graduate|degree|program|school|institution)|"
    r"\breturning to (?:school|your studies|university|college|campus)\b|\bremain(?:ing)? enrolled\b", re.I)
GRADUATES_OK = re.compile(r"recent(?:ly)? (?:grad|complet|finish|earn|obtain|receiv)|graduated|new grads?\b|alumni|early[- ]career|completed (?:a |an |your |their )?"
                          r"(?:\w+ )?degree|\bor (?:a )?(?:recent )?graduate\b|\bor (?:have )?(?:completed|graduated|earned|hold)|"
                          r"degree holders?|equivalent (?:\w+ )?(?:experience|learning|work|training)", re.I)


NOT_ENROLLED = re.compile(r"\b(?:cannot|can ?not|not|n't|never|no longer)\b", re.I)


def screen_student_only(c: CandidateCard, text: str) -> str | None:
    """A graduated candidate on a posting that asks for current students (the card left enrolled_in empty)."""
    if not c.graduation_year or c.graduation_year > CURRENT_YEAR:
        return None
    for s in sentences(text):
        if ENROLLED.search(s) and not GRADUATES_OK.search(s) and not PREFERRED.search(s) and not EEO.search(s) \
                and not NOT_ENROLLED.search(s):
            return f"graduated {c.graduation_year}; posting: {s[:300]}"
    return None


PRIOR_INTERN = re.compile(
    r"\b(?:at least|minimum of|a minimum of|one or more|1\+|two or more) (?:one |1 |two |2 )?(?:previous |prior |past |"
    r"relevant )?(?:professional )?(?:internships?|co-?ops?)\b|"
    r"\b(?:previous|prior|past) (?:\w+ )?(?:internships?|co-?ops?)(?: experience)? (?:is |are )?(?:required|a must|"
    r"necessary|mandatory)|\bmust have (?:completed|had|held) (?:at least one |a |one |an )?(?:previous |prior )?"
    r"(?:\w+ )?(?:internship|co-?op)|\b(?:requires?|required:?) (?:at least one |a |one )?(?:previous |prior )"
    r"(?:\w+ )?(?:internship|co-?op)", re.I)
HAS_INTERN = re.compile(r"\bintern\b|\binternship\b|\bco-?op\b", re.I)


def screen_prior_internship(resume: str, text: str) -> str | None:
    s = first_hit(PRIOR_INTERN, text)
    if s and not HAS_INTERN.search(resume):
        return f"resume names no internship; posting: {s}"
    return None


_NAME = r"(?:[Tt]he )?[A-Z][\w&.'-]*(?: (?:of|at|and|&|the|[A-Z][\w&.'-]*))*"
SCHOOL_ONLY = re.compile(
    r"(?i:\b(?:only|exclusively) (?:open|available) to (?:current(?:ly enrolled)? |enrolled )?(?:\w+ ){0,3}?(?:students|"
    r"undergraduates|graduates|alumni)(?: who are)? (?:currently )?(?:enrolled )?(?:at|of|from|in|attending) )(?P<a>" + _NAME + ")|"
    r"(?i:\bopen (?:only )?to (?:current(?:ly enrolled)? |enrolled )?(?:\w+ ){0,3}?(?:students|undergraduates)(?: who are)? "
    r"(?:currently )?(?:enrolled )?(?:at|of|from|attending) )(?P<b>" + _NAME + ")|"
    r"(?i:\bmust be (?:a |an )?(?:current(?:ly)? )?(?:enrolled )?(?:full[- ]time )?(?:\w+ ){0,3}?(?:student|undergraduate)s? "
    r"(?:at|of|enrolled (?:at|in)|attending) )(?P<c>" + _NAME + ")|"
    r"\b(?P<d>[A-Z][\w&.'-]*(?: (?:of|at|[A-Z][\w&.'-]*))*) (?i:students only)\b")
SCHOOL_FILLER = frozenset("the of at and university college institute school state students student campus".split())
NOT_SCHOOL = re.compile(r"^(?:the )?(?:U\.?S\.?|United States|an? |accredited|any|this|our|a |all|North America|US)\b", re.I)


def screen_school_only(resume: str, text: str) -> str | None:
    for s in sentences(text):
        m = SCHOOL_ONLY.search(s)
        if not m or PREFERRED.search(s):
            continue
        name = next(g for g in m.groups() if g)
        if NOT_SCHOOL.search(name):
            continue
        words = [w for w in re.findall(r"[A-Za-z&]+", name) if w.lower() not in SCHOOL_FILLER and len(w) > 1]
        if not words:
            continue
        if not all(re.search(rf"\b{re.escape(w)}\b", resume, re.I) for w in words[:3]):
            return f"school {name!r} not on the resume; posting: {s[:300]}"
    return None


PERMANENT = re.compile(
    r"\bunlimited (?:amount of )?time\b|\bwithout (?:any |a )?time limit|\bunrestricted (?:right|authori[sz]ation|"
    r"work|employment)|\bpermanent(?:ly)? (?:authori[sz]ed|work authori[sz]ation|right to work|employment "
    r"authori[sz]ation)|\bon a permanent basis\b|\bindefinite(?:ly)?\b[^.]{0,40}authori|"
    r"(?-i:\b(?:CPT|OPT|F-1|F1)\b)[^.]{0,100}(?:not eligible|do not qualify|does not qualify|not qualify|ineligible|"
    r"not (?:be )?accepted|cannot be considered|not be considered|not supported|unable to (?:support|accept|consider))|"
    r"(?:not eligible|do not qualify|ineligible|not accept|cannot accept|not (?:be )?considered)[^.]{0,100}"
    r"(?-i:\b(?:CPT|OPT|F-1|F1)\b)", re.I)
NO_SPONSOR = re.compile(
    r"(?:will not|won't|unable to|not able to|cannot|can't|can ?not|does not|do not|doesn't|don't|not (?:currently )?"
    r"(?:offer|provide|available)|\bno\b|is not|are not) (?:\w+ ){0,4}?sponsor|sponsorship (?:is |will |are )?(?:not|n't)"
    r" (?:be )?(?:available|offered|provided|considered|possible)|without (?:the )?(?:need for |requiring |needing )?"
    r"(?:\w+ )?(?:visa |employer |company )?sponsorship|not (?:require|need)(?:ing)? (?:\w+ )?sponsorship|"
    r"\bno (?:visa |h-?1-?b )?sponsorship", re.I)
NOT_REFUSAL = re.compile(r"not guaranteed|\bFTE\b|full[- ]time (?:conversion|roles?|positions?|employment|hire)|"
                         r"return offer|conversion", re.I)
CITIZEN_RX = re.compile(
    r"\bU\.?\s?S\.? citizens?(?:hip)?\b|\bUS citizens?(?:hip)?\b|\bUnited States citizens?(?:hip)?\b|\bU\.?\s?S\.? persons?\b"
    r"|\bsecurity clearance\b|\bclearance\b[^.]{0,40}(?:required|obtain|eligib)|\bITAR\b|\bgreen card\b|"
    r"\bpermanent residen", re.I)
NEED_CUE = re.compile(r"\bmust\b|\brequire|\bonly\b|\beligib|\bneed|\bnecessary|\bable to obtain|\bobtain\b|"
                      r"\bmaintain\b|\bmandatory|\bqualif|\bpossess", re.I)


def screen_permanent_auth(c: CandidateCard, text: str) -> str | None:
    """A candidate who is not a citizen or permanent resident, on a posting that asks for unlimited or permanent
    authorization or says CPT/OPT/F-1 students do not qualify: rules.py reads that wording as citizens_only."""
    if c.work_auth == CITIZEN:
        return None
    for s in sentences(text):
        if PERMANENT.search(s) and not EEO.search(s):
            return f"{c.work_auth}; posting: {s[:300]}"
    return None


def screen_no_sponsor(c: CandidateCard, text: str) -> str | None:
    """needs_h1b_now on a posting that does not sponsor (the rule's will_not_sponsor; other_visa stays a soft check)."""
    if c.work_auth != "needs_h1b_now":
        return None
    for s in sentences(text):
        if NO_SPONSOR.search(s) and not EEO.search(s) and not NOT_REFUSAL.search(s):
            return f"{c.work_auth}; posting: {s[:300]}"
    return None


def screen_citizen_only(c: CandidateCard, text: str) -> str | None:
    if c.work_auth == CITIZEN:
        return None
    for s in sentences(text):
        if CITIZEN_RX.search(s) and NEED_CUE.search(s) and not EEO.search(s) and not PREFERRED.search(s):
            return f"{c.work_auth}; posting: {s[:300]}"
    return None


IND_RX = defaultdict(list)
for _rx, _key in _INDUSTRY_RE:
    IND_RX[_key].append(_rx)
INDUSTRY_MIN = 2  # the label read counted 3; a posting that names healthcare twice ("transform healthcare delivery")
# is a healthcare job. Words most postings use anyway ("research", "data", "software") need more.
INDUSTRY_MIN_BY = {"research": 4, "technology": 3}


def screen_industry(c: CandidateCard, raw: dict, kind: str) -> str | None:
    if kind == "pref_industry_avoided" or not c.industries_avoided:
        return None
    flat = _industry_text(f"{raw.get('title', '')} {raw.get('company', '')} {raw.get('description', '')}")
    for ind in c.industries_avoided:
        for key in industry_ids(ind):
            n = sum(len(rx.findall(flat)) for rx in IND_RX.get(key, ()))
            if n >= INDUSTRY_MIN_BY.get(key, INDUSTRY_MIN):
                return f"avoids {ind}; its words appear {n} times in the posting"
    return None


SOFT_RX = re.compile(
    r"communicat|teamwork|team player|collaborat|attention to detail|detail[- ]oriented|\bethic|integrity|"
    r"\blearn|prioriti|multi-?task|time management|organi[sz]ation(?:al)? skill|\borganized\b|interpersonal|"
    r"leadership|(?<!8D )problem[- ]solving|critical thinking|self-?start|motivat|curios|passion|\bdrive\b|initiative|"
    r"adaptab|flexib|big[- ]picture|professionalism|positive attitude|work ethic|judg(?:e)?ment|independen|"
    r"ownership|accountab|creativ|resourceful|enthusias|growth mindset|eager|willing", re.I)


def missing_nice(completion: str) -> list[str]:
    m = re.search(r"missing nice-to-have skills?: (.*?);", completion)
    return [x.strip() for x in m.group(1).split(",")] if m else []


def screen_soft_nice(kind: str, completion: str) -> str | None:
    if kind != "nice_skill":
        return None
    soft = [x for x in missing_nice(completion) if is_soft_skill(x) or SOFT_RX.search(x)]
    return f"soft nice-to-have counted: {soft}" if soft else None


BROAD_TECH = re.compile(r"\btechnical (?:discipline|field|major|degree|area)s?\b|\b(?:any|other|related|a|an) (?:technical|"
                        r"STEM|quantitative|scientific) (?:discipline|field|major|degree|area)s?\b|\bSTEM\b|"
                        r"\b(?:any|all) (?:majors|disciplines|fields of study)\b", re.I)
BROAD_ENG = re.compile(r"\b(?:any |an |other |related )?engineering (?:discipline|field|major)s?\b", re.I)


DEGREE_WORDS = re.compile(r"\bdegree|\bmajor|\bpursuing|\bstudying|\bstudents?\b|\bbachelor|\bmaster|\bph\.?d|"
                          r"\b[BM]\.?S\.?\b|\bundergrad|\bgraduate", re.I)


def screen_broad_degree(c: CandidateCard, kind: str, text: str) -> str | None:
    """degree_field_adjacent rows whose posting takes any technical/STEM field (or, for an engineering field, any
    engineering discipline) in a sentence about the degree."""
    if kind != "degree_field_adjacent":
        return None
    eng = "engineering" in (c.degree_field or "").lower()
    for s in sentences(text):
        if DEGREE_WORDS.search(s) and (BROAD_TECH.search(s) or eng and BROAD_ENG.search(s)):
            return f"field {c.degree_field}; posting: {s[:300]}"
    return None


SCREENS = ("location", "student_only", "class_standing", "grad_window", "prior_internship", "school_only", "permanent_auth",
           "no_sponsor", "citizen_only", "industry_conflict", "soft_nice_skill", "broad_degree_field")


def screen_row(row: dict, cand: CandidateCard, raw: dict) -> tuple[str, str] | None:
    """(screen, detail) for the first screen a non-Skip row fails, or None (Skip rows always pass)."""
    if row["bucket"] == SKIP:
        return None
    text = raw.get("description", "")
    resume = prompt_parts(row["prompt"])[0]
    kind = row["variant_kind"]
    checks = (
        ("location", lambda: screen_location(cand, raw)),
        ("student_only", lambda: screen_student_only(cand, text)),
        ("class_standing", lambda: screen_class_standing(cand, text)),
        ("grad_window", lambda: screen_grad_window(cand, text)),
        ("prior_internship", lambda: screen_prior_internship(resume, text)),
        ("school_only", lambda: screen_school_only(resume, text)),
        ("permanent_auth", lambda: screen_permanent_auth(cand, text)),
        ("no_sponsor", lambda: screen_no_sponsor(cand, text)),
        ("citizen_only", lambda: screen_citizen_only(cand, text)),
        ("industry_conflict", lambda: screen_industry(cand, raw, kind)),
        ("soft_nice_skill", lambda: screen_soft_nice(kind, row["completion"])),
        ("broad_degree_field", lambda: screen_broad_degree(cand, kind, text)),
    )
    for name, f in checks:
        if why := f():
            return name, why
    return None


def screen(a: argparse.Namespace | None = None) -> dict:
    cands = {r["pair_id"]: r["candidate"] for r in read_jsonl(OUT / "rows.jsonl")}
    raw = brt_raw()
    dropped, rep = [], {}
    for s in SPLITS:
        rows = read_jsonl(OUT / f"{s}.jsonl")
        kept = []
        for r in rows:
            hit = screen_row(r, CandidateCard.from_dict(cands[r["pair_id"]]), raw[r["rq_id"]])
            if hit:
                dropped.append({k: r[k] for k in ("pair_id", "rq_id", "cd_id", "bucket", "variant_kind", "completion")}
                               | {"split": s, "screen": hit[0], "detail": hit[1]})
            else:
                kept.append(r)
        brt.write_jsonl(OUT / f"screened_{s}.jsonl", kept)
        d = [x for x in dropped if x["split"] == s]
        rep[s] = {"rows_in": len(rows), "rows_kept": len(kept), "dropped": len(d),
                  "dropped_by_screen": dict(Counter(x["screen"] for x in d).most_common()),
                  "dropped_by_bucket": dict(Counter(x["bucket"] for x in d)),
                  "in_by_bucket": dict(Counter(r["bucket"] for r in rows)),
                  "kept_by_bucket": dict(Counter(r["bucket"] for r in kept)),
                  "non_skip_drop_share": round(len(d) / max(sum(r["bucket"] != SKIP for r in rows), 1), 3),
                  "dropped_by_kind": dict(Counter(x["variant_kind"] for x in d).most_common())}
        print(f"{s}: {len(rows)} rows, {len(d)} dropped ({rep[s]['non_skip_drop_share']:.1%} of non-Skip); "
              f"by screen {rep[s]['dropped_by_screen']}; kept buckets {rep[s]['kept_by_bucket']}")
    brt.write_jsonl(OUT / "screen_dropped.jsonl", dropped)
    (OUT / "screen_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def brt_raw() -> dict[str, dict]:
    return {"ats_" + p["simplify_id"]: p for p in read_jsonl(brt.RAW_POSTINGS)}


# ---------------------------------------------------------------- assemble

CSV_OUT = OUT / "train_mix.csv"
DEV_OUT = OUT / "real_dev.jsonl"
CHECKS_OUT = OUT / "checks.json"
SEED = 0
WORD = {"For you": "Keep", "Explore": "Explore", "Excluded": "Skip"}
NEAR_JACCARD = 0.5  # a resume or posting sharing half its 6-word windows with another is a near copy (build_real_train)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def read_synth(path: Path = SYNTH, want: str | None = SYNTH_MD5) -> list[dict]:
    """smoke.csv's rows, refusing a file whose md5 is not the one the earlier runs trained on."""
    if want and md5(path) != want:
        raise SystemExit(f"{path} md5 {md5(path)}, expected {want}: the synthetic rows changed; nothing written")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def frame_problems(rows: list[dict], synth: list[dict]) -> list[str]:
    """Rows not in the synthetic rows' frame: the same instruction up to RESUME:, the four blocks in order, the
    prompt ending in "Answer:", a completion starting with the bucket's word and " — "."""
    head = synth[0]["prompt"].split("\nRESUME:\n", 1)[0]
    bad = []
    for r in rows:
        p, c = r["prompt"], r["completion"]
        try:
            prompt_parts(p)
        except (IndexError, ValueError):
            bad.append(f"{r.get('pair_id')}: blocks")
            continue
        if p.split("\nRESUME:\n", 1)[0] != head or not p.endswith("Answer:"):
            bad.append(f"{r.get('pair_id')}: instruction or ending")
        if "bucket" in r and not c.startswith(WORD[r["bucket"]] + " — "):
            bad.append(f"{r.get('pair_id')}: completion word")
    return bad


def mix(synth: list[dict], real: list[dict], seed: int = SEED) -> list[tuple[str, str]]:
    """Every synthetic row and every screened real train row, (prompt, completion), in one seeded shuffle (nothing
    says Adaption shuffles a file, and a shuffled file trains the same if it does)."""
    rows = [(r["prompt"], r["completion"]) for r in synth] + [(r["prompt"], r["completion"]) for r in real]
    import random
    random.Random(seed).shuffle(rows)
    return rows


def write_csv(path: Path, rows: list[tuple[str, str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "completion"])
        w.writerows(rows)
    return md5(path)


def near_pairs(a: dict[str, str], b: dict[str, str], jac: float = NEAR_JACCARD) -> list[dict]:
    """(id in a, id in b, Jaccard) for every pair of texts sharing at least `jac` of their 6-word windows (exact
    copies included), found through an index of b's windows."""
    from matcher.nearcopy import shingles
    sb = {k: shingles(v) for k, v in b.items()}
    inv: dict[str, list[str]] = defaultdict(list)
    for k, s in sb.items():
        for x in s:
            inv[x].append(k)
    out = []
    for ka, va in a.items():
        sa = shingles(va)
        shared: Counter = Counter()
        for x in sa:
            for kb in inv.get(x, ()):
                shared[kb] += 1
        for kb, n in shared.items():
            j = n / len(sa | sb[kb])
            if j >= jac:
                out.append({"a": ka, "b": kb, "jaccard": round(j, 3)})
    return out


def baselines(train: list[dict], dev: list[dict]) -> list[dict]:
    """leakage.run_baselines (length-only, bag-of-words on each block, shuffled-posting control) on the rows'
    resume, profile and posting as the prompt holds them."""
    import pandas as pd
    from matcher import leakage as L

    def frame(rows):
        parts = [prompt_parts(r["prompt"]) for r in rows]
        return pd.DataFrame({L.ID: [r["pair_id"] for r in rows], L.LABEL: [r["bucket"] for r in rows],
                             L.RESUME: [p[0] for p in parts], L.PROFILE: [p[1] for p in parts],
                             L.POSTING: [p[2] for p in parts]})
    return L.run_baselines(frame(train), frame(dev), seed=SEED)


def random_f1(labels: list[str]) -> dict:
    """Expected macro-F1 of guessing at random: uniformly, and by the labels' own shares."""
    n = Counter(labels)
    tot = sum(n.values())
    p = {k: v / tot for k, v in n.items()}
    uni = sum(2 * p[k] * (1 / len(p)) / (p[k] + 1 / len(p)) for k in p) / len(p)
    prior = sum(p[k] for k in p) / len(p)  # F1 of a prior-matched guess on class k is p_k
    return {"uniform": round(uni, 3), "prior_matched": round(prior, 3)}


def checks(train: list[dict], dev: list[dict], synth: list[dict], csv_rows: list[tuple[str, str]]) -> dict:
    import build_real_train as brr
    from build_real_train import ctx_of
    test = read_jsonl(TEST)
    raw = brt_raw()
    out: dict = {}
    # 1. postings: no test posting, no near copy of one, and no dev posting near a train one
    test_ids = {r["rq_id"] for r in test}
    real_ids = {s: {r["rq_id"] for r in rows} for s, rows in (("train", train), ("dev", dev))}
    idx = brr.TestIndex({i: raw[i] for i in test_ids})
    out["postings"] = {
        "train_postings": len(real_ids["train"]), "dev_postings": len(real_ids["dev"]),
        "test_postings_in_train": len(real_ids["train"] & test_ids), "test_postings_in_dev": len(real_ids["dev"] & test_ids),
        "dev_postings_in_train": len(real_ids["dev"] & real_ids["train"]),
        "near_test_in_train": sorted(i for i in real_ids["train"] if idx.near(raw[i])),
        "near_test_in_dev": sorted(i for i in real_ids["dev"] if idx.near(raw[i])),
        "dev_near_train": [p for p in near_pairs({i: raw[i].get("description", "") for i in real_ids["dev"]},
                                                 {i: raw[i].get("description", "") for i in real_ids["train"]})],
    }
    # 2. resumes: no test resume context; no resume text equal or near to a test resume (real and synthetic), and
    # none shared between real dev and real train or the synthetic rows
    cands = {r["pair_id"]: r["candidate"] for r in read_jsonl(OUT / "rows.jsonl")}
    test_ctx = {ctx_of(r["card"]) for r in read_jsonl(brt.OUT / "resume_cards.jsonl")}
    ctx = {s: {ctx_of(cands[r["pair_id"]]) for r in rows} for s, rows in (("train", train), ("dev", dev))}
    res = lambda rows, key: {key(r): prompt_parts(r["prompt"])[0] for r in rows}  # noqa: E731
    t_res = res(test, lambda r: r["cd_id"])
    tr_res, dv_res = res(train, lambda r: r["cd_id"]), res(dev, lambda r: r["cd_id"])
    sy_res = {f"synth_{i}": prompt_parts(r["prompt"])[0] for i, r in enumerate(synth)}
    sy_res = {k: v for k, v in sy_res.items() if v}  # dedupe not needed; ids differ
    out["resumes"] = {
        "train_resumes": len(tr_res), "dev_resumes": len(dv_res), "test_resumes": len(t_res),
        "test_context_in_train": len(ctx["train"] & test_ctx), "test_context_in_dev": len(ctx["dev"] & test_ctx),
        "dev_context_in_train": len(ctx["dev"] & ctx["train"]),
        "near_test_resume": {"train": near_pairs(tr_res, t_res), "dev": near_pairs(dv_res, t_res),
                             "synthetic": near_pairs(sy_res, t_res)},
        "dev_resume_near_train": near_pairs(dv_res, tr_res),
        "dev_resume_near_synthetic": near_pairs(dv_res, {k: v for k, v in sy_res.items()}),
    }
    # 3. duplicates
    prompts = Counter(p for p, _ in csv_rows)
    pairs = Counter(csv_rows)
    labels = defaultdict(set)
    for p, c in csv_rows:
        labels[p].add(c.split()[0])
    test_prompts = {r["prompt"] for r in test}
    dev_prompts = Counter(r["prompt"] for r in dev)
    out["duplicates"] = {
        "csv_rows": len(csv_rows), "csv_prompts_seen_twice": sum(v > 1 for v in prompts.values()),
        "csv_rows_seen_twice": sum(v > 1 for v in pairs.values()),
        "csv_prompts_with_two_labels": sum(len(v) > 1 for v in labels.values()),
        "csv_prompts_in_test": len(set(prompts) & test_prompts), "csv_prompts_in_real_dev": len(set(prompts) & set(dev_prompts)),
        "real_dev_prompts_seen_twice": sum(v > 1 for v in dev_prompts.values()),
        "real_dev_prompts_in_test": len(set(dev_prompts) & test_prompts),
    }
    # 4. shortcut guessers on the real part (train -> real dev), and the same guessers on the real test for reference
    dev_labels = [r["bucket"] for r in dev]
    test_dec = [r for r in test if r.get("decidable", True)]
    out["shortcuts"] = {
        "real_dev": {"rows": len(dev), "random_guess_macro_f1": random_f1(dev_labels), "baselines": baselines(train, dev)},
        "real_test_reference": {"rows": len(test_dec), "random_guess_macro_f1": random_f1([r["bucket"] for r in test_dec]),
                                "baselines": baselines(train, test_dec)},
    }
    # the test's side headlines: the resume-dependent kinds and the kinds whose label the posting states outright
    from matcher.evalstats import RESUME_KINDS
    for name, kinds in (("real_test_resume_kinds", RESUME_KINDS), ("real_test_core_kinds", brt.CORE_KINDS)):
        sub = [r for r in test_dec if r["variant_kind"] in kinds]
        if len({r["bucket"] for r in sub}) > 1:
            out["shortcuts"][name] = {"rows": len(sub), "labels": dict(Counter(r["bucket"] for r in sub)),
                                      "random_guess_macro_f1": random_f1([r["bucket"] for r in sub]),
                                      "baselines": baselines(train, sub)}
    return out


def problems_of(ch: dict) -> list[str]:
    """Leaks and duplicates that must stop the build (shortcut scores are reported, not gated)."""
    p, r, d = ch["postings"], ch["resumes"], ch["duplicates"]
    bad = []
    for k in ("test_postings_in_train", "test_postings_in_dev", "dev_postings_in_train", "test_context_in_train",
              "test_context_in_dev", "dev_context_in_train"):
        if (p | r)[k]:
            bad.append(f"{k}: {(p | r)[k]}")
    for k in ("near_test_in_train", "near_test_in_dev", "dev_near_train", "dev_resume_near_train"):
        if (p | r)[k]:
            bad.append(f"{k}: {len((p | r)[k])}")
    for k, v in r["near_test_resume"].items():
        if v:
            bad.append(f"near_test_resume.{k}: {len(v)}")
    for k in ("csv_rows_seen_twice", "csv_prompts_with_two_labels", "csv_prompts_in_test", "csv_prompts_in_real_dev",
              "real_dev_prompts_seen_twice", "real_dev_prompts_in_test"):
        if d[k]:
            bad.append(f"{k}: {d[k]}")
    return bad


def assemble(a: argparse.Namespace | None = None) -> dict:
    synth = read_synth()
    train = read_jsonl(OUT / "screened_train.jsonl")
    dev = read_jsonl(OUT / "screened_dev.jsonl")
    if not train or not dev:
        sys.exit("no screened rows: run `screen` first")
    bad = frame_problems(train + dev, synth)
    if bad:
        sys.exit(f"{len(bad)} real rows out of the synthetic frame, e.g. {bad[:3]}; nothing written")
    rows = mix(synth, train)
    ch = checks(train, dev, synth, rows)
    stop = problems_of(ch)
    CHECKS_OUT.write_text(json.dumps(ch, indent=1))
    if stop:
        sys.exit(f"checks failed, CSV not written ({CHECKS_OUT.relative_to(ROOT)}): {stop}")
    digest = write_csv(CSV_OUT, rows)
    brt.write_jsonl(DEV_OUT, [{k: r[k] for k in TEST_KEYS} for r in dev])
    lab = lambda rs: dict(Counter(WORD[r["bucket"]] for r in rs))  # noqa: E731
    rep = {"csv": str(CSV_OUT.relative_to(ROOT)), "csv_md5": digest, "csv_rows": len(rows),
           "synthetic_rows": len(synth), "synthetic_md5": SYNTH_MD5, "real_train_rows": len(train),
           "real_train_postings": len({r["rq_id"] for r in train}),
           "labels": {"synthetic": dict(Counter(r["completion"].split()[0] for r in synth)), "real_train": lab(train),
                      "csv": dict(Counter(c.split()[0] for _, c in rows))},
           "real_dev": str(DEV_OUT.relative_to(ROOT)), "real_dev_md5": md5(DEV_OUT), "real_dev_rows": len(dev),
           "real_dev_postings": len({r["rq_id"] for r in dev}), "real_dev_labels": lab(dev),
           "real_train_kinds": dict(Counter(r["variant_kind"] for r in train).most_common()),
           "prompt_chars_median": {"synthetic": sorted(len(r["prompt"]) for r in synth)[len(synth) // 2],
                                   "real_train": sorted(len(r["prompt"]) for r in train)[len(train) // 2],
                                   "real_dev": sorted(len(r["prompt"]) for r in dev)[len(dev) // 2]},
           "prompt_chars_max": max(len(p) for p, _ in rows)}
    (OUT / "assemble_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "real_train_kinds"}, indent=1))
    print(f"checks: {CHECKS_OUT.relative_to(ROOT)}; nothing failed")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("screen")
    sub.add_parser("assemble")
    a = ap.parse_args()
    {"screen": screen, "assemble": assemble}[a.cmd](a)


if __name__ == "__main__":
    main()
