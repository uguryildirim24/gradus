"""train_mix3.csv for run d: the cleaned real-posting train rows of clean_real_checked, three profile draws per row,
with two more screens and the three rows the data review read as wrong taken out.

What it does, in order (train rows only; the dev files clean_real_checked writes are left alone):

  1. relabel    clean_real_checked.relabel_rows on every rendered train row of a hand-checked posting, with
                --draws 3: three draws of the profile fields no resume shows, same resume and posting (draw 0 is
                train_mix2's draw). The build's own gates run as there (guards, assemble_real_train's text screens,
                visibility, duplicate candidates).
  2. known bad  every draw of the three train rows the data review read as wrong (KNOWN_BAD, by the row's old pair
                id: Xcel Energy x2 and the UT Austin research assistant Keep). The screens below catch all three on
                their own; the build stops if they don't.
  3. restriction  RESTRICTED: the train postings with a requirement the card cannot hold, in the dev side's kinds
                and with its rule (clean_real_checked.py dev-v2, checked_real_dev_v2*.jsonl). One school, must be
                local (here also a list of states it is not open to), class standing (here also a PhD year), a
                graduation window (also "a term left after the internship" and "graduated within a year"), a
                degree level the rules do not encode: the posting loses every row, Skips included. An experience
                cap or a degree ceiling: the posting goes only when a non-Skip row's candidate (any draw) is past
                it. A GPA is not a restriction there, so not here. Read by hand from the checkers' flags
                (cardcheck/checked_*.jsonl) and a text scan of every train posting (restriction_scan); each entry
                quotes the posting, and the build stops if a quote is not in the shown posting.
  4. class standing  screen_class_standing: the posting asks a class standing ("junior or higher", "sophomore
                standing or above", "completed the first year", "3rd year completed", "first- or second-year
                undergraduates", "rising juniors") at a term ("as of Fall 2027", "by Summer 2027", rising, or now)
                that an enrolled bachelor's candidate's expected year does not reach (or passes, for a closed
                list), or a graduated candidate on a posting for students of a standing. Non-Skip rows only. Run on
                every posting, so it also covers phrasings RESTRICTED did not list.
  5. assemble   the kept rows cut to the smallest label (clean_real_checked.balance, round-robin over postings,
                seeded), smoke.csv's 6,891 rows (md5 checked before reading, and every one of its records checked
                byte for byte in the output), one seeded shuffle (assemble_real_train.mix), prompt and completion.
  6. report     label mix, real share, the ask gap of the real Keeps against the synthetic Keeps, the leakage and
                duplicate checks (assemble_real_train.checks against checked_real_dev.jsonl; a leak stops the
                build), and the profile-, resume- and posting-only guessers on the real part in employer folds.

  build_train_mix3.py build [--seed 0] [--draws 3]
      Writes data/built/real_train/train_mix3.csv and data/built/real_train/mix3/{rows.jsonl, report.json}.
      Deterministic, local files only; loads no model and calls nothing.

RESTRICTED holds train postings only; the dev side's own table is clean_real_checked.RESTRICTIONS.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import assemble_real_train as art  # noqa: E402
import build_real_test as brt  # noqa: E402
import clean_real_checked as ccr  # noqa: E402
from matcher.schema import CURRENT_YEAR, DEGREE_LEVELS, CandidateCard  # noqa: E402

RT = ROOT / "data/built/real_train"
OUT = RT / "mix3"
CSV_OUT = RT / "train_mix3.csv"
DEV_FILE = RT / "checked_real_dev.jsonl"
SKIP = "Excluded"
WORD = ccr.WORD

# the data review's wrong train rows (train_mix2), by old pair id: every draw of them goes
KNOWN_BAD = {
    "pr_5887ce950851": "Xcel Energy Data Analyst Intern: 'Junior or higher (as of Fall 2027)', candidate expects 2030",
    "pr_202a5c99a6fa": "Xcel Energy Data Analyst Intern: 'Junior or higher (as of Fall 2027)', candidate expects 2030",
    "pr_51e3cee2a506": "UT Austin Remote Sensing RA: 'Current UT student ... has completed the first year', "
                       "candidate expects 2030",
}


# ---------------------------------------------------------------- the candidate


def gy(c: CandidateCard) -> int:
    return c.graduation_year or 0


# ---------------------------------------------------------------- restriction screen (posting level)

# The dev side's rule (clean_real_checked.py dev-v2, which writes checked_real_dev_v2*.jsonl): a posting with one of
# REMOVE_KINDS loses every row, Skips included; a CHECK_KINDS restriction is exact on the candidate card, so it
# removes the posting only when a non-Skip row's candidate (any draw) is past it. A GPA, program dates or a weekly
# schedule are not restrictions there (no profile shows a GPA, and the test is built the same way), so not here.
REMOVE_KINDS = ("school_only", "class_standing", "grad_window", "degree_level", "must_be_local")
CHECK_KINDS = ("degree_ceiling", "exp_cap")


@dataclass(frozen=True)
class Restriction:
    kind: str
    quote: str  # words from the shown posting (build_real_test.posting_text)
    note: str = ""
    limit: object = None  # CHECK_KINDS: the degree level or the most years of work the posting takes


RESTRICTED: dict[str, Restriction] = {
    "ats_6f1a27e9-3515-4a43-a644-80dbddacc4bb": Restriction(
        "school_only", "A current undergraduate or graduate student at Arizona State University"),
    "ats_f7d43f88-b142-46f2-9334-796092e21c57": Restriction(
        "school_only", "Current UT student in undergraduate programs", "also: has completed the first year"),
    "ats_58f69c1f-a80d-4906-8d64-29ad188aa643": Restriction(
        "school_only", "Summer Intern - Youngstown State University", "the title names one university"),
    "ats_ef6779b9-6706-4217-8b3a-09ef98f4faf4": Restriction(
        "must_be_local", "students attending Nebraska universities who reside in the Omaha, NE area"),
    "ats_a9b15888-79e1-4d34-b682-700189ca37f3": Restriction(
        "must_be_local", "We are seeking a local student who can start in June 2027",
        "part time during the semesters; also: graduation May 2028 or later"),
    "ats_9c708edf-68f9-43ac-a86d-b27868f9d200": Restriction(
        "must_be_local", "This position is not available for candidates living in the following states",
        "remote, but not for residents of AK, CA, HI, NM, NY, OR, WA, WY"),
    "ats_168a9b2f-3f11-497a-a04f-62c40059bff5": Restriction(
        "class_standing", "Are a 4th or 5th year senior Ph.D student", "a PhD year"),
    "ats_453db16f-11ff-4cf7-85fb-5f41453a4483": Restriction(
        "class_standing", "Must currently be at least a second-year Ph.D. student", "a PhD year"),
    "ats_01adde08-7298-47e1-8394-bb946ac7ec13": Restriction("class_standing", "sophomore standing or above"),
    "ats_01effa4f-a101-46d3-ae7c-516e03b8b401": Restriction(
        "class_standing", "3rd Year of bachelor's degree completed", "also: at most 24 months of work"),
    "ats_03ddbf84-5e23-46af-b5c3-428a7ef7f009": Restriction("class_standing", "Junior or senior standing by Summer 2027"),
    "ats_18eeb2a6-97b5-4ca5-9d64-d93fadcd08be": Restriction(
        "class_standing", "Current student, Junior or higher (as of Fall 2027)"),
    "ats_1beafc6b-26c4-4dd9-9396-8b9792b26456": Restriction("class_standing", "completion of at least sophomore year"),
    "ats_5e14584d-3148-449a-b903-28b77771e883": Restriction(
        "class_standing", "Sophomore or Junior year standing by the start of the internship"),
    "ats_613293d2-6943-4f9a-ab4a-7bb7883891fc": Restriction(
        "class_standing", "Must be a first-year (class of 2030) or second year (class of 2029) undergraduate"),
    "ats_954f1915-886d-44d2-b63c-900fb3b88410": Restriction(
        "class_standing", "third- and fourth-year undergraduate students and recent graduates"),
    "ats_240adc3d-e80f-42bc-9a7b-559c2acdc5f5": Restriction(
        "grad_window", "Rising Junior or Senior graduating in 2027 or 2028"),
    "ats_7b684d8f-d5d1-4f2a-ae66-c9f8f79aec21": Restriction(
        "grad_window", "Rising Junior or Senior graduating in 2027 or 2028"),
    "ats_2599beb7-cad8-4b9b-9020-7baf41143bfb": Restriction(
        "grad_window", "with an expected graduation in 2027 or 2028"),
    "ats_037d751e-e964-4164-ad1e-5cd1b7b6cb57": Restriction(
        "grad_window", "Must be graduating between December 2027 and August 2029"),
    "ats_93b4a798-697c-42d2-a104-cda1bb152b83": Restriction(
        "grad_window", "Must be graduating between December 2027 and August 2029"),
    "ats_827ccc8c-5daf-4ca9-90a9-badca54261a8": Restriction(
        "grad_window", "Must have at least one additional quarter/semester of school remaining"),
    "ats_6682ab3e-72f8-49a9-928c-220d51a81e0d": Restriction(
        "grad_window", "Planning to graduate between December 2027 and June 2029", "also: at most 2 years of work"),
    "ats_2149ec11-ebea-44e9-95ff-a9ef69ecacdf": Restriction(
        "grad_window", "with a graduation date of December 2027 or later"),
    "ats_463d369a-081e-447b-9b9b-02cc6a0c5f79": Restriction(
        "grad_window", "returning to school following the internship"),
    "ats_0d13979f-34e6-4863-983f-b931e8619e05": Restriction(
        "grad_window", "for at least one semester following the conclusion of the internship"),
    "ats_e86150cf-386f-4eac-99a4-224ea7eacfc3": Restriction(
        "grad_window", "expected graduation date between Dec 2026 - 2028"),
    "ats_1dba15c8-8b9b-480b-9237-6a5a7bf57e11": Restriction("grad_window", "completion date in 2028 or later"),
    "ats_08aebc55-1f7d-4e55-b07f-6a07f7e62c71": Restriction(
        "grad_window", "Graduating class of 2029 pursuing a Masters or Doctorate degree"),
    "ats_7f13fb69-b688-4ec7-811b-29b870ae8668": Restriction(
        "grad_window", "have recently graduated within 1 year of the position start date"),
    "ats_2a0b4fae-5de5-4ba8-b55d-c4bba930145b": Restriction(
        "degree_level", "Must a recent graduate or current student pursuing their master's"),
    "ats_e7524d34-3342-448c-83e8-2de9a3502b9b": Restriction(
        "degree_level", "Currently pursuing or recently completed a PhD"),
    "ats_29379ac0-6351-43a9-a3a3-cd3b286670cb": Restriction(
        "exp_cap", "Requires less than 1 year of related experience", limit=0),
}

# read and left in: the line is preferred, opens the role, is a GPA (not a restriction on the dev side either), or
# the card holds it
REVIEWED_NOT_RESTRICTED = {
    "ats_0f457023-4491-454a-bc5c-fc8ea237a9a1": "Rockwell: graduation window and 'within 50 miles' are under Preferred",
    "ats_0421e841-beed-4315-83cf-cbc7ffc68617": "WhiteWater: 'preferably a May 2028 grad'",
    "ats_b10edb5a-e520-4da4-988d-a8a730bef747": "GE Vernova: the 3.0 GPA sits under 'What Will Make You Stand Out'",
    "ats_4060f153-ab15-44d4-9f86-1bf555dbf0f7": "Home Depot: 'Preferred GPA 3.0 or above'",
    "ats_1a70f3d5-9b7c-4830-81a5-6da2c80e7c16": "Applied Materials: 'GPA of 3.0 or above preferred'",
    "ats_59c4465f-c0d4-4d19-b808-988e46c72c88": "Insperity: 'GPA of 3.25 or higher preferred'",
    "ats_b07a5da6-2c9f-4bf1-bf90-18ffff8232ae": "National Laboratory of the Rockies: a 3.0 GPA minimum (no GPA kind)",
    "ats_c9792cfd-c5b4-4224-b707-ce3b840830f4": "iRhythm: 'student or new graduate', no window stated",
    "ats_3d2ef8a9-a5da-4120-9c53-50ceec3023cd": "Persona AI: 'recent graduates are also welcome' opens the role",
}


def quote_problems(raw: dict) -> list[str]:
    """RESTRICTED entries whose quote is not in the shown posting (a typo would screen nothing)."""
    norm = lambda t: re.sub(r"\s+", " ", t)  # noqa: E731
    return [i for i, r in RESTRICTED.items() if i not in raw or norm(r.quote) not in norm(brt.posting_text(raw[i]))]


def past_limit(kind: str, limit, c: CandidateCard) -> bool:
    """clean_real_checked.past_limit: studying above the ceiling, or more years than the cap."""
    if kind == "degree_ceiling":
        return gy(c) > CURRENT_YEAR and DEGREE_LEVELS.index(c.degree_level) > DEGREE_LEVELS.index(limit)
    if kind == "exp_cap":
        return c.years > limit
    raise ValueError(kind)


def removed_postings(rows: list[dict], table: dict[str, Restriction] = RESTRICTED) -> dict[str, str]:
    """posting id -> why, for the postings the restriction screen removes from `rows` (every draw counts for a
    CHECK_KINDS limit)."""
    out = {}
    for i in sorted({r["rq_id"] for r in rows} & set(table)):
        t = table[i]
        if t.kind in REMOVE_KINDS:
            out[i] = f"{t.kind}: '{t.quote}'"
        elif t.kind in CHECK_KINDS:
            past = sum(past_limit(t.kind, t.limit, CandidateCard.from_dict(r["candidate"]))
                       for r in rows if r["rq_id"] == i and r["bucket"] != SKIP)
            if past:
                out[i] = f"{t.kind}: '{t.quote}' ({past} non-Skip rows past {t.limit})"
        else:
            raise ValueError(f"{i}: unknown restriction kind {t.kind}")
    return out


SCAN = re.compile(
    r"\b(?:junior|sophomore|freshm[ae]n|rising|first[- ]year|second[- ]year|third[- ]year|fourth[- ]year|"
    r"[1-4](?:st|nd|rd|th)[- ]year|class of|GPA|grade point|only (?:open|available)|open only|residen\w*|"
    r"local (?:student|candidate)s?|recent(?:ly)? grad\w*|new grad\w*|graduat\w*[^.]{0,60}20[23]\d|"
    r"20[23]\d[^.]{0,40}graduat\w*|semester (?:of school )?remaining|return(?:ing)? to school|no more than|"
    r"maximum of|less than (?:\d+|one) years?|not available for candidates)", re.I)


def restriction_scan(text: str) -> list[str]:
    """The posting sentences a reader should check for a requirement the card cannot hold (for the report: a
    posting with a hit that is in neither RESTRICTED nor REVIEWED_NOT_RESTRICTED was not read)."""
    return [s[:200] for s in art.sentences(text) if SCAN.search(s) and not art.EEO.search(s)]


# ---------------------------------------------------------------- class standing (row level)

LEVEL = {"freshman": 1, "freshmen": 1, "first": 1, "1st": 1, "sophomore": 2, "sophomores": 2, "second": 2, "2nd": 2,
         "junior": 3, "juniors": 3, "third": 3, "3rd": 3, "senior": 4, "seniors": 4, "fourth": 4, "4th": 4,
         "final": 4}
# an ordinal counts only as a year ("first-year", "third- and fourth-year"), a class word alone
_LV = r"(?:freshm[ae]n|sophomores?|juniors?|seniors?|(?:first|second|third|fourth|final|1st|2nd|3rd|4th)" \
      r"(?=[- ]year|-?\s*(?:and|or|&|/|,)\s*\w+-?\s*(?:and|or|&|/|,)?\s*\w*[- ]year))"
_YR = r"(?:[- ]year)?(?:\s*\(\s*class of 20\d\d\s*\))?"
# a list of standings: "Junior or Senior", "sophomores, juniors, and seniors", "third- and fourth-year"
_LIST = rf"{_LV}{_YR}(?:-?\s*(?:,\s*|\s+|/)(?:(?:or|and|&)\s+)?{_LV}{_YR})*"
STANDING = re.compile(
    rf"\b(?P<list>{_LIST})\s*"
    r"(?P<ctx>(?:year\s+)?(?:standing|status|class(?:ification)?|level)\b|"
    r"(?:year\s+)?(?:undergrad\w*|students?|bachelor\S*)\b|"
    r"(?:or|and) (?:higher|above|beyond|greater|later|up)\b|\+)", re.I)
RISING_LIST = re.compile(rf"\b(?:rising|entering|incoming)\s+(?P<list>{_LIST})\b", re.I)
OPEN_TO = re.compile(rf"\b(?:open (?:only )?to|only|exclusively for|seeking|looking for)\s+(?:current\s+)?(?P<list>{_LIST})\b",
                     re.I)
AT_LEAST = re.compile(rf"\b(?:at least|minimum(?: of)?|a minimum of)\s+(?:an?\s+)?(?P<list>{_LV}){_YR}\b", re.I)
_NUM = {"one": 1, "1": 1, "two": 2, "2": 2, "three": 3, "3": 3, "four": 4, "4": 4}
COMPLETED = re.compile(
    r"\b(?:ha(?:s|ve)\s+)?(?:completed|complete|completion of|finished)\s+(?:at least\s+|a minimum of\s+|minimum\s+|"
    r"the\s+|their\s+|your\s+|his or her\s+)*"
    rf"(?:(?P<lv>{_LV})[- ]year|(?P<lvw>freshm[ae]n|sophomore|junior)\s+year|"
    r"(?P<n>one|two|three|four|1|2|3|4)\s+(?:full\s+)?(?:academic\s+)?(?P<unit>years?|semesters?))\b", re.I)
COMPLETED_AFTER = re.compile(rf"\b(?P<lv>{_LV})\s+year of (?:\w+'?s?\s+)?(?:degree|study|college|undergrad\w*)\s+"
                             r"(?:completed|complete|finished)\b", re.I)
TERM = re.compile(r"\b(?:as of|by|in|for|during|before|prior to|starting)\s+(?:the\s+)?(?P<season>fall|autumn|spring|"
                  r"summer|winter)\s+(?:of\s+)?(?P<year>20[2-3]\d)\b", re.I)
SOON = re.compile(r"\brising\b|\bentering\b|\bincoming\b|\bgoing into\b|\bby the (?:start|summer|time)|"
                  r"\bby (?:the )?(?:start of )?(?:the )?internship", re.I)
NOT_UNDERGRAD = re.compile(r"\b(?:ph\.?\s?d|doctoral|doctorate|master'?s|mba|graduate student|law|medical)\b", re.I)
OPEN_ENDED = re.compile(r"\b(?:or|and) (?:higher|above|beyond|greater|later|up)\b|\+|\bat least\b|\bminimum\b|"
                        r"\bcompleted|\bcompletion\b", re.I)
JOB_LEVEL = re.compile(r"\b(?:junior|senior)\s+(?:engineer|developer|analyst|scientist|manager|leader|staff|"
                       r"executive|leadership|management|director|member|associate|consultant|designer|level\b(?! standing))",
                       re.I)


def standing_at(year_grad: int, fall: int) -> int:
    """A May-graduating bachelor's student's standing in the fall of `fall`: 4 senior, 3 junior, 2 sophomore,
    1 freshman (0 or less: not started; 5 or more: graduated)."""
    return 5 - (year_grad - fall)


def ref_fall(s: str) -> int:
    """The fall the sentence's standing is counted at: a stated term (spring and winter belong to the fall
    before), "rising" or "by the internship" (the fall after the summer CURRENT_YEAR + 1 internship), else now."""
    m = TERM.search(s)
    if m:
        y = int(m.group("year"))
        return y - 1 if m.group("season").lower() in ("spring", "winter") else y
    return CURRENT_YEAR + 1 if SOON.search(s) else CURRENT_YEAR


def standing_need(s: str) -> tuple[int, int, int] | None:
    """(lowest standing, highest standing, fall counted at) the sentence asks of a bachelor's student, or None."""
    if art.PREFERRED.search(s) or NOT_UNDERGRAD.search(s):
        return None
    fall = ref_fall(s)
    m = COMPLETED_AFTER.search(s) or COMPLETED.search(s)
    if m:
        gd = m.groupdict()
        if gd.get("lv"):
            done = LEVEL[gd["lv"].lower()]
        elif gd.get("lvw"):
            done = LEVEL[gd["lvw"].lower().rstrip("s") if gd["lvw"].lower() != "freshmen" else "freshman"]
        else:
            n = _NUM[gd["n"].lower()]
            done = n // 2 if gd["unit"].lower().startswith("semester") else n
        if done < 1:
            return None
        # completed k years by the fall counted at: the standing that fall is k + 1
        return done + 1, 99, fall
    for rx in (RISING_LIST, STANDING, AT_LEAST, OPEN_TO):
        m = rx.search(s)
        if not m:
            continue
        if JOB_LEVEL.search(s[max(0, m.start() - 5): m.end() + 30]):
            return None
        levels = [LEVEL[w.lower()] for w in re.findall(_LV, re.sub(r"\(\s*class of 20\d\d\s*\)", "", m.group("list")), re.I)]
        if not levels:
            return None
        if rx is RISING_LIST:
            fall = max(fall, CURRENT_YEAR + 1)
        lo = min(levels)
        tail = s[m.start(): m.end() + 25]
        hi = 99 if (rx is AT_LEAST or OPEN_ENDED.search(tail) or max(levels) == 4) else max(levels)
        return lo, hi, fall
    return None


def screen_class_standing(c: CandidateCard, text: str) -> str | None:
    """An enrolled bachelor's candidate whose standing at the counted term is outside what a sentence asks, or a
    graduated candidate on a sentence that asks a standing of current students (and does not admit graduates)."""
    for s in art.sentences(text):
        need = standing_need(s)
        if not need:
            continue
        lo, hi, fall = need
        y = gy(c)
        if not y:
            continue
        if y <= CURRENT_YEAR:
            if c.degree_level in ("bachelor", "associate") and not art.GRADUATES_OK.search(s) and \
                    re.search(r"\bcurrent|\benrolled|\bstudents?\b|\bundergrad", s, re.I) and \
                    not re.search(r"recent|graduates?\b|alumni", s, re.I):
                return f"graduated {y}; posting asks a standing of students: {s[:240]}"
            continue
        if c.degree_level != "bachelor":
            continue
        st = standing_at(y, fall)
        if st < lo or st > hi:
            return f"expected {y}: standing {st} in fall {fall}, posting asks {lo}-{hi if hi < 99 else 'up'}: {s[:240]}"
    return None


# ---------------------------------------------------------------- build


def extra_screens(r: dict, raw: dict, gone: dict[str, str]) -> tuple[str, str] | None:
    """The screens this build adds, on a kept row: known bad, restriction (`gone`: removed_postings), class
    standing."""
    if r["old_pair_id"] in KNOWN_BAD:
        return "known_bad", KNOWN_BAD[r["old_pair_id"]]
    if r["rq_id"] in gone:
        return "restriction", gone[r["rq_id"]]
    c = CandidateCard.from_dict(r["candidate"])
    if r["bucket"] != SKIP and (why := screen_class_standing(c, raw[r["rq_id"]].get("description", ""))):
        return "class_standing", why
    return None


def synth_bytes_check(csv_path: Path, synth_path: Path = art.SYNTH) -> dict:
    """Every smoke.csv record, serialized as csv.writer writes it, appears in the output as the same bytes, and the
    output holds nothing else beyond the real rows."""
    def records(p: Path) -> list[str]:
        with open(p, newline="") as f:
            rows = list(csv.reader(f))[1:]
        out = []
        for row in rows:
            b = io.StringIO()
            csv.writer(b).writerow(row)
            out.append(b.getvalue())
        return out
    raw_out = csv_path.read_bytes()
    syn = records(synth_path)
    with open(synth_path, "rb") as f:
        syn_raw = f.read()
    # the re-serialized record must be a slice of both files (so the bytes, not only the values, match)
    not_in_synth = sum(rec.encode() not in syn_raw for rec in syn)
    out_set = Counter(records(csv_path))
    missing = sum(1 for rec in syn if out_set[rec] < 1 or rec.encode() not in raw_out)
    return {"synthetic_records": len(syn), "missing_in_output": missing, "not_byte_equal_in_smoke": not_in_synth,
            "output_records": sum(out_set.values())}


def dev_rows_for_checks() -> list[dict]:
    """checked_real_dev.jsonl's rows with the old pair id the leakage checks look resumes up by."""
    rows = ccr.read_jsonl(DEV_FILE)
    old = {r["pair_id"]: r["old_pair_id"] for r in ccr.read_jsonl(ccr.OUT / "rows.jsonl") if r["split"] == "dev"}
    return [r | {"pair_id": old.get(r["pair_id"], r["pair_id"])} for r in rows]


def build(a: argparse.Namespace) -> dict:
    splits = json.loads(ccr.SPLITS_FILE.read_text())
    cards, card_rep = ccr.load_checked("extracted", splits=splits)
    rows, cands, raw, store, ctx_of = ccr.load_inputs()
    if (bad := quote_problems(raw)):
        raise SystemExit(f"RESTRICTED quotes not in the posting: {bad}")
    resume_of = lambda c: store.get(ctx_of(c.to_dict()))  # noqa: E731
    _, step2 = ccr.relabel_rows({"train": rows["train"], "dev": []}, cands, cards, raw, resume_of, a.seed, a.draws)
    gated = [r for r in step2 if r["stage"] == "kept"]

    gone = removed_postings(gated)
    recs, kept = [], []
    for r in gated:
        hit = extra_screens(r, raw, gone)
        recs.append(r | {"mix3_stage": f"{hit[0]}" if hit else "kept", "mix3_detail": hit[1] if hit else ""})
        if not hit:
            kept.append(r)
    # the screens alone must catch the known-bad rows
    missed = [r["pair_id"] for r in gated if r["old_pair_id"] in KNOWN_BAD and r["rq_id"] not in gone and
              not screen_class_standing(CandidateCard.from_dict(r["candidate"]), raw[r["rq_id"]].get("description", ""))]
    if missed:
        raise SystemExit(f"the screens alone miss known-bad rows {missed}")
    standing_alone = Counter(
        (WORD[r["bucket"]], r["rq_id"] in gone) for r in gated if r["bucket"] != SKIP and
        screen_class_standing(CandidateCard.from_dict(r["candidate"]), raw[r["rq_id"]].get("description", "")))

    real = ccr.balance(kept, a.seed)
    synth = art.read_synth()
    if (bad := art.frame_problems(real, synth)):
        raise SystemExit(f"{len(bad)} rows out of the synthetic frame, e.g. {bad[:3]}; nothing written")
    csv_rows = art.mix(synth, real, a.seed)

    dev = dev_rows_for_checks()
    ch = art.checks(ccr.lineage(real), dev, synth, csv_rows)
    ch.pop("shortcuts", None)
    stop = art.problems_of(ch)
    csv_prompts = {p for p, _ in csv_rows}
    other_dev = {p: ccr.read_jsonl(RT / f"{p}.jsonl") for p in ("checked_real_dev_builder", "checked_real_dev_v2_builder",
                                                                "checked_real_dev_v2", "real_dev")}
    ch["other_dev_prompts_in_csv"] = {k: len({r["prompt"] for r in v} & csv_prompts) for k, v in other_dev.items()}
    stop += [f"{k} prompts in csv: {v}" for k, v in ch["other_dev_prompts_in_csv"].items() if v]
    dev_posts = {r["rq_id"] for v in [dev, *other_dev.values()] for r in v}
    ch["real_postings_in_any_dev"] = len({r["rq_id"] for r in real} & dev_posts)
    if ch["real_postings_in_any_dev"]:
        stop.append(f"train postings in a dev file: {ch['real_postings_in_any_dev']}")
    emp = ccr.employer_overlap(real, dev, raw)
    if emp["train_dev"]:
        stop.append(f"employer in train and dev: {emp['train_dev'][:5]}")
    if stop:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "report.json").write_text(json.dumps({"leakage": ch, "problems": stop}, indent=1, default=str))
        raise SystemExit(f"checks failed, nothing else written: {stop}")

    md5 = art.write_csv(CSV_OUT, csv_rows)
    syn_check = synth_bytes_check(CSV_OUT)
    if syn_check["missing_in_output"] or syn_check["not_byte_equal_in_smoke"] or \
            syn_check["output_records"] != len(synth) + len(real):
        raise SystemExit(f"synthetic rows not carried byte for byte: {syn_check}")

    by_stage = Counter(r["mix3_stage"] for r in recs)
    lab = lambda rs: dict(Counter(WORD[r["bucket"]] for r in rs))  # noqa: E731
    scan = {}
    train_posts = sorted({r["rq_id"] for r in step2})
    for i in train_posts:
        hits = restriction_scan(raw[i].get("description", ""))
        if hits and i not in RESTRICTED and i not in REVIEWED_NOT_RESTRICTED:
            scan[i] = hits[:3]
    rep = {
        "seed": a.seed, "draws": a.draws, "cards": card_rep,
        "relabel_gates": ccr.gate_counts(step2)["train"],
        "extra_screens": {
            "rows_in": len(gated), "labels_in": lab(gated), "dropped_by": dict(by_stage),
            "dropped_labels": {s: lab([r for r in recs if r["mix3_stage"] == s]) for s in by_stage if s != "kept"},
            "restriction_rows_by_kind": dict(Counter(r["mix3_detail"].split(":")[0] for r in recs
                                                     if r["mix3_stage"] == "restriction")),
            "restriction_postings_listed": len(RESTRICTED),
            "restriction_postings_with_rows": len({r["rq_id"] for r in gated} & set(RESTRICTED)),
            "restriction_postings_removed": {i: gone[i] for i in sorted(gone)},
            "restriction_postings_checked_not_removed": sorted(({r["rq_id"] for r in gated} & set(RESTRICTED)) - set(gone)),
            "class_standing_screen_alone": {f"{b}, {'on a removed posting' if g else 'elsewhere'}": n
                                            for (b, g), n in sorted(standing_alone.items())},
            "postings_lost_every_row": sorted({r["rq_id"] for r in gated} - {r["rq_id"] for r in kept}),
            "class_standing_hits": [{"pair_id": r["pair_id"], "rq_id": r["rq_id"], "bucket": WORD[r["bucket"]],
                                     "detail": r["mix3_detail"]} for r in recs if r["mix3_stage"] == "class_standing"],
            "scan_hits_not_read": scan,
            "kept": len(kept), "kept_labels": lab(kept), "kept_postings": len({r["rq_id"] for r in kept}),
            "kept_old_rows": len({r["old_pair_id"] for r in kept}),
        },
        "mix": {"synthetic_rows": len(synth), "synthetic_md5": art.SYNTH_MD5, "synthetic_check": syn_check,
                "real_rows": len(real), "real_labels": lab(real), "real_postings": len({r["rq_id"] for r in real}),
                "real_old_rows": len({r["old_pair_id"] for r in real}),
                "draws_per_old_row": dict(Counter(Counter(r["old_pair_id"] for r in real).values())),
                "balance_cut": {WORD[b]: n - lab(real).get(WORD[b], 0) for b, n in
                                Counter(r["bucket"] for r in kept).items()},
                "csv_rows": len(csv_rows), "csv_labels": dict(Counter(c.split()[0] for _, c in csv_rows)),
                "real_share": round(len(real) / len(csv_rows), 4), "csv_md5": md5},
        "ask_gap_keep": ccr.ask_gap_report({"real_kept_after_screens": kept, "real_in_csv": real}, synth),
        "guessers_real_part_employer_folds": ccr.grouped_cv(real, raw, seed=a.seed),
        "random_guess_real_part": art.random_f1([r["bucket"] for r in real]),
        "employer_overlap": emp,
        "leakage": ch,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    in_csv = {r["pair_id"] for r in real}
    brt.write_jsonl(OUT / "rows.jsonl", [{k: v for k, v in r.items() if k != "prompt"} | {"in_csv": r["pair_id"] in in_csv}
                                         for r in recs])
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("extra_screens", "mix", "ask_gap_keep", "guessers_real_part_employer_folds",
                                          "random_guess_real_part")}, indent=1, default=str))
    print(f"wrote {CSV_OUT.relative_to(ROOT)} ({md5}), {OUT.relative_to(ROOT)}/rows.jsonl, report.json")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--draws", type=int, default=3)
    a = ap.parse_args()
    {"build": build}[a.cmd](a)


if __name__ == "__main__":
    main()
