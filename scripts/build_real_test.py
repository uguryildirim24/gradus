"""The real-posting test: the 87 hand-checked posting cards, real text, candidates built to match. The rows cover 86 of
them: every resume of ats_5f3d33fb-4b34-47a8-a127-86e779a876b6 failed the render gates on all 3 tries.

  build_real_test.py build [--seed 0]
      Pairs each corrected card (data/built/audit/test_cards.jsonl) with candidates and writes
        data/built/real_test/rows.jsonl         one row per (posting, candidate): labels, no prompt yet
        data/built/real_test/resume_cards.jsonl the candidate cards whose resumes must be rendered
        data/built/real_test/resume_ids.txt     their ids, for `render_cards.py resumes estimate|run`
        data/built/real_test/dropped.jsonl      every row left out, with the reason
        data/built/real_test/report.json        counts: bucket mix, per kind, drops
      Re-runnable: it recomputes every label with the current rules.label(), so run it again
      after the rules branch merges.
  build_real_test.py assemble
      Joins rendered resume texts (render_cards' context hash) with render_profile and the real
      posting text through build_prompt, and writes data/built/real_test/test.jsonl, the file
      the model evaluator scores. Rows with no rendered resume are listed and left out. Every row
      is checked again against the rendered resume; one whose label the resume does not show
      (graduation year, degree, a held skill, a swapped-out skill it still names) gets
      decidable False and its misses, and is kept out of the headline.
  build_real_test.py score PREDICTIONS.json [--draws 2000]
      PREDICTIONS.json is an evaluator --out file (or any {"rows": [{pair_id, pred}]}). Prints
      accuracy per variant kind, Keep recall on the core and the added Keep kinds, and the
      headline macro-F1 with a 95% bootstrap CI that resamples postings (the 86 with rows), not pairs, given
      again without the added Keep kinds and on the resume-dependent kinds alone. Rows test.jsonl
      marks not decidable are left out (local_eval.py and baselines.py leave them out too).

Rows:
- Every variant from variants.make_variants whose posting card is the corrected card itself
  (the variants that edit the posting card, such as work_auth_needs_h1b, would contradict the real
  text and are not used), plus the full match crossed with every work-authorization status. Each
  status gets the education it implies (variants.with_status: CPT means enrolled, OPT graduated last
  year or this, STEM OPT a STEM graduate of 1-3 years back with a year of work, H-1B now not
  enrolled), so a crossing whose status needs another education has its own resume (on a posting for
  students the OPT and H-1B rows are graduates, so their Skip is the enrollment one).
- Every candidate is a person who could exist, checked as build_pairs checks the training ones
  (person_problems: work authorization against education, license against degree, years of work
  against the graduation year, internships wanted with "intern" avoided, a skill listed twice, an
  industry wanted and avoided). The posting fit (graduation window, 0 years on an internship) can
  break that; settle() repairs it, and a row it cannot repair is dropped (candidate_inconsistent).
  build fails, writing nothing, if a kept row still has one.
- The candidate is built from the posting: its graduation year sits inside the graduation
  window the text states, internship cards get 0 years, and a preference (industry wanted, salary
  floor) is only set when the posting text shows the fact it is compared with.
- A row is dropped when the shown text contradicts its label or says something the card cannot
  hold (guard(): a "far" city in the posting's metro, a wanted job type the posting names (also as
  part-time hours), a generic, example-only, duties-only or slash-paired missing skill, a missing
  skill on an any-of line or list the candidate meets through another item, "must be local", a cap
  on years, sponsorship wording such as "now or in the future" against an F-1 or visa status, a
  required minimum GPA behind a Keep or Explore, a graduation year outside the stated window, a
  missing skill a held one implies (pandas -> Python), "or equivalent" beside the degree an
  enrollment or degree Skip rests on, a degree level a below-level Skip names that the text never
  names ("undergraduate" alone takes an associate student; a pay-tier line does not count), a
  graduated candidate's Skip beside a "graduation YEAR or later" line that year meets, a sponsorship
  Skip whose only "sponsor" word is a company-sponsored benefit, a location Skip on a hybrid card whose
  text names no mode or office, a location Skip whose text says "remote" and no in-person mode, a
  graduated candidate's Skip beside "or equivalent experience" with no graduation window or
  already-graduated line, a degree-field Skip on "or a related discipline" for a candidate whose
  skills are fields the line names, a must-skill named only under an "ideal candidate" or
  "preferred" heading, an overqualified "0 asked" beside a stated range of years, CPT on a
  student-worker role, a citizens-only Skip for a temporary status on ITAR text that also takes anyone eligible for
  an export license or authorization, a Keep or Explore whose undergraduate's graduation year falls outside the
  class level the text asks for (class_window: "Rising Junior/Senior", "junior or senior", "completion of at
  least three years", "sophomore year must be complete"), a still-enrolled Explore or a later finisher beside a
  degree deadline ("Receive a Ph.D. no later than spring 2027"), a pay Explore beside a stated figure at or above
  the floor ("$98k base; $111k OTE")).
  Location and job-type edits are re-picked first, so most of these rows are replaced, not lost.
  The avoided-industry variant drops the wanted industry, so the profile does not want and avoid it.
- A graduation window with a month or season ("on or after December 2027", "by June 2028") is met
  by a year only when every month of it fits: the card holds no month and the resume writer picks
  one. The candidate gets such a year; on a window no whole year fits ("December 2027 - May 2028")
  the Keep and Explore rows are dropped. assemble checks a month the resume names against it.
- The candidate's graduation year is fitted to the window the text states ("graduating in 2028",
  "scheduled to obtain a degree by Summer 2029"); on an internship whose card says enrolled_in none
  but whose text is for current students, it is CURRENT_YEAR + 2, so the candidate is a student. A
  bachelor's student is then fitted into the class level the text asks for (a rising junior or senior
  after the Summer 2027 internship graduates in 2028 or 2029; a junior or senior now in 2027 or 2028).
- The For you share is reached by construction: besides the full match, the any-of skill swaps
  and the CPT crossings, it counts relocate, pref_mode_match, industry_list and up to three more
  any-of swaps per posting. It was 15% until the strict-citizenship guard; it is 12.7% now (98 of
  773; report.json is the live count) and 8.2% without the added kinds. score() reports Keep recall on the core and the added
  kinds apart.
- A row is dropped when its label hinges on a fact the posting text, as build_prompt cuts it,
  does not state (visibility.missing_facts, posting side only). An Excluded row is kept when at
  least one of its hard reasons is visible. A row is also dropped when rules.label() disagrees with
  the bucket variants.py built the variant to have (the label is then not what the edit meant).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from matcher import textclean  # noqa: E402
from matcher.pairs import build_prompt, prompt_parts  # noqa: E402
from matcher.profile import render_profile  # noqa: E402
from matcher.metros import place_key, places_match, posting_places  # noqa: E402
from matcher.rules import counted_nice, format_completion, held_skills, label, license_problems, required_groups, \
    status_problems  # noqa: E402
from matcher.sample import INDUSTRIES, METRO_NAMES  # noqa: E402
from matcher.schema import CURRENT_YEAR, EMPLOYMENT_TYPES, WORK_AUTH, CandidateCard, RequirementCard, group_met, norm, \
    skill_key  # noqa: E402
from matcher.sample import graduated_year  # noqa: E402
from matcher.variants import full_match, grad_floor, industry_clash, make_variants, pre_degree_years, prefs_fit, \
    settle_years, unique_skills, with_status, years_fit  # noqa: E402
from matcher.visibility import has_money, has_phrase, missing_facts  # noqa: E402

AUDIT = ROOT / "data/built/audit/test_cards.jsonl"
RAW_POSTINGS = ROOT / "data/raw/ats/postings.jsonl"
OUT = ROOT / "data/built/real_test"
BUCKETS = ("For you", "Explore", "Excluded")

# Kinds whose label rests on text the posting states outright (the rest also lean on the
# profile or on the posting's silence). The headline macro-F1 covers every kept row; this subset is
# nearly all Skip, so it is reported as accuracy only.
CORE_KINDS = ("must_skill", "must_skill_alternative", "degree_level", "degree_field", "enrollment_graduated",
              "enrollment_level", "location", "employment_type", "work_auth_citizens_only", "license",
              "work_auth_opt_student_role",
              *(f"work_auth_citizens_only_{s}" for s in ("f1_cpt", "f1_opt", "stem_opt", "needs_h1b_now")))

# Keeps that test reading a requirement (an any-of skill swap or a synonym, a related field, a soft or a
# single nice-to-have skill the rule does not count, CPT on an internship) and the full match; the
# easy profile-only Keep kinds (evalstats.ADDED_KEEP_KINDS) are reported apart.
CORE_KEEP_KINDS = ("full_match", "must_skill_alternative", "must_skill_synonym", "degree_field_related",
                   "degree_enrolled_higher", "soft_skill_missing", "nice_skill_one_missing", "auth_f1_cpt",
                   "work_auth_cpt_internship", "work_auth_cpt_internship_no_opt")
# The added Keep kinds and the resume-dependent kinds (evalstats, so every scorer uses one list).
from matcher.evalstats import ADDED_KEEP_KINDS, RESUME_KINDS  # noqa: E402
# Every kind variants.make_variants can make with the posting card unchanged (a stance variant keeps the card
# when the card already has that stance), the work-auth crossings and the added kinds; the report lists those
# with no kept row as coverage gaps.
VARIANT_KINDS = (
    "full_match", "degree_level", "degree_still_enrolled", "degree_enrolled_higher", "enrollment_graduated",
    "enrollment_level", "degree_field", "degree_field_adjacent", "degree_field_related", "years_one_short",
    "years_far_short", "overqualified", "must_skill", "must_skill_alternative", "must_skill_synonym",
    "soft_skill_missing", "nice_skill", "nice_skill_one_missing", "license", "location", "location_willing",
    "location_anywhere", "location_same_metro", "employment_type", "salary", "pref_industry_wanted",
    "pref_industry_avoided", "pref_industry_avoided_other", "pref_remote_only", "pref_hybrid_or_remote",
    "pref_remote_match", "pref_role_avoided", "pref_role_avoided_other",
    *(f"work_auth_citizens_only_{s}" for s in ("f1_cpt", "f1_opt", "stem_opt", "needs_h1b_now")),
    "work_auth_opt_student_role", "work_auth_needs_h1b", "work_auth_h1b_sponsors", "work_auth_h1b_unstated",
    "work_auth_f1_opt", "work_auth_stem_opt", "work_auth_opt_sponsors", "work_auth_stem_opt_sponsors",
    "work_auth_no_opt", "work_auth_cpt_internship", "work_auth_cpt_internship_no_opt", "work_auth_cpt_full_time",
    "work_auth_cpt_part_time", "work_auth_other_visa", "work_auth_citizens_only")
KNOWN_KINDS = (*VARIANT_KINDS, *(f"auth_{s}" for s in WORK_AUTH), *ADDED_KEEP_KINDS)
# Kinds whose candidate lives away from the posting, and those of them that relocate to it.
RELOCATE_KINDS = ("relocate", "location_willing", "location_anywhere")
AWAY_KINDS = ("location", *RELOCATE_KINDS)
# Keeps whose remote preference the posting's mode meets.
MODE_MATCH_KINDS = ("pref_mode_match", "pref_remote_match")
THIN = 5  # a kind with fewer kept rows than this is reported as thin coverage

# Posting-side wording the stock regex in visibility.py misses for "for current students".
ENROLL_EXTRA = re.compile(
    r"actively pursuing|pursuing (a|an|their|your|the)\b|rising (sophomore|junior|senior)|returning to (school|university)"
    r"|(undergraduate|graduate|college|university|masters?|bachelor'?s|phd|doctoral) students?|enrolled|working toward"
    r"|currently (attending|completing|seeking a)|upcoming graduat|\bclass of 20", re.I)
ENROLL_MISS = "posting does not say it is for current students"

# "graduate" as in "graduate degree" or "graduate student" names a level, not a graduation date.
# "Scheduled to obtain a Bachelor's degree ... by Summer 2028" also names a graduation date.
# "Expected" and "anticipated" count only next to a graduation or degree word ("Expected in-office
# schedule ... summer of 2027" names no graduation date); "completion" alone (of the internship)
# never does.
GRAD_WORD = re.compile(r"\bgraduat(es|ed|ing|ion)\b|\bgraduate\b(?!\s+(degree|students?|programs?|school|level|studies|or)\b)"
                       r"|class of|\b(expected|anticipated)\b(?=[^.\n]{0,30}\b(graduat|degree|completion of (your|the|a|their) "
                       r"(degree|studies|program)))"
                       r"|\bto (obtain|earn|receive|complete) (a|an|your|their|the)\b[^.\n]{0,60}\bdegree\b", re.I)
# A posting for students that also takes recent graduates: a graduated candidate fits it.
RECENT_GRAD = re.compile(r"recent(ly)?\s+(graduat|acquired|completed|obtained|earned)|or (a )?recent grad", re.I)
YEAR = re.compile(r"\b(20\d\d)\b")


# ---------------------------------------------------------------- inputs

def read_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_test_cards(path: Path = AUDIT) -> dict[str, RequirementCard]:
    return {r["id"]: RequirementCard.from_dict(r["card"]) for r in read_jsonl(path)}


def posting_text(p: dict) -> str:
    """The posting under the Title/Company/Locations header, built as the demo and training build theirs
    (textclean.real_posting_text: every ATS location, the body without markdown or HTML entities)."""
    return textclean.real_posting_text(p)


def load_postings(path: Path = RAW_POSTINGS) -> dict[str, str]:
    return {"ats_" + p["simplify_id"]: posting_text(p) for p in read_jsonl(path)}


def shown_posting(text: str) -> str:
    """The posting as the model sees it: build_prompt's fit (textclean.fit_posting, 6,000 chars: the
    header, every work-authorization sentence and the requirement lines first) applied."""
    prompt = build_prompt("", "", text)
    return prompt.split("JOB POSTING:\n", 1)[1].rsplit("\n\nAnswer:", 1)[0]


# ---------------------------------------------------------------- row guards
# Labels the posting text contradicts, or that the card schema cannot express. Each guard names a
# row the rule labels one way while a careful reader of the shown text would label it another.

# Places one commute apart that the label rule keeps in separate metros (metros.py has Baltimore as its
# own metro and no Des Moines metro). The rule's own metro table (metros.METROS and metros.NEARBY) decides
# every other pair; a candidate living in one of these is never picked as "away" from a posting in the other.
COMMUTE_GROUPS = (
    ("Washington, DC", "Baltimore, MD"),
    ("Des Moines, IA", "West Des Moines, IA", "Johnston, IA"),
)
_COMMUTE_OF = [[place_key(p) for p in g] for g in COMMUTE_GROUPS]


def _same_key(a: tuple[str, str], b: tuple[str, str]) -> bool:
    return a[0] == b[0] and (not a[1] or not b[1] or a[1] == b[1])


def same_metro(a: str, b: str) -> bool:
    """The label rule's place match (metros.places_match), or one commute apart (COMMUTE_GROUPS)."""
    if places_match(a, b):
        return True
    pa, pb = place_key(a), place_key(b)
    return any(any(_same_key(pa, x) for x in g) and any(_same_key(pb, x) for x in g) for g in _COMMUTE_OF)


def away_ok(city: str, card: RequirementCard, shown: str) -> bool:
    """A place the candidate can live in and still be away from the posting: not in the same metro
    as any posting location, and not a city the shown posting names."""
    return not any(same_metro(city, x) for x in posting_places(card.locations)) and \
        not has_phrase(shown, city.split(",")[0])


# The stated job type phrases (visibility's own patterns).
# Part-time also when the posting offers part-time hours without the word ("up to 20 hours/week
# during the semesters", "a minimum of 20 hours", "flexibility regarding hours").
# Full-time also as 35-40 hours a week ("40 hours/week over the summer", "up to 40 hours per week").
TYPE_PHRASE = {"internship": r"\bintern(ship)?s?\b|co-?op",
               "full_time": r"full[- ]?time|\b(3[5-9]|40)\s*(\+\s*)?(hours|hrs)\b",
               "part_time": r"part[- ]?time|up to (1\d|2\d|3[0-4]) (hours|hrs)|minimum of \d+ (hours|hrs)"
                            r"|\b(1\d|2\d|3[0-4])\s*(-\s*\d+\s*)?(hours|hrs)\s*(/|per|a|each)\s*week"
                            r"|flexib\w+[^.\n]{0,40}\b(hours|schedule)|during the (school|academic) year|during the semesters?"
                            r"|during the (fall|spring) and (fall|spring)",
               "contract": r"\bcontract"}

# "Must be local", "commutable distance": a relocation Keep is not what the text says.
LOCAL_ONLY = re.compile(r"must be local|be local to|local (candidate|student|intern|applicant|resident)s?\b|live within"
                        r"|reside within|must (currently )?reside|currently (live|reside)|commutable|commuting distance", re.I)
# A "Work Model" (or flexible-work) section the fit's hard cut ends inside, before it names a
# mode: the shown text hides the work-mode policy a location Skip or relocate Keep rests on.
MODE_SECTION = re.compile(r"^\W*(work model|work arrangements?|flexible work)\W*$", re.I | re.M)
ANY_MODE = re.compile(r"on[- ]?site|in[- ]person|in the office|in-office|hybrid|remote", re.I)
MODE_WORD = {"onsite": r"on[- ]?site|in[- ]person|in the office|in-office", "hybrid": r"hybrid"}
_NUM = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
YEARS_CAP = re.compile(rf"(no|not) more than {_NUM}\s*\+?\s*(\(\d+\)\s*)?(full[- ]time )?(years?|yrs)|at most {_NUM}\s*(years?|yrs)"
                       rf"|(less|fewer) than {_NUM}\s*(years?|yrs)|maximum of {_NUM}\s*(years?|yrs)", re.I)

# Wording the card's sponsorship field cannot hold. Against it any temporary status reads Skip,
# not Keep or Explore, unless the text names F-1, CPT or OPT as eligible (DoorDash does).
NO_TEMPORARY = re.compile(r"temporary visa|any employer|(now|currently|presently) (or|and) (at any (time|point) )?in the future"
                          r"|with no restrictions|without restriction|without (the )?(need|requirement)"
                          r"( for| of)? (current or future |any )?(employer |company |visa |immigration )?sponsorship", re.I)
MUST_BE_AUTHORIZED = re.compile(r"(must|should|need to) (be|have) (legally )?(authoriz|eligible to work)|legally authorized to work"
                                r"|authorized to work in the (U\.?S|United States)", re.I)
F1_WELCOME = re.compile(r"(F-?1|\bCPT\b|\bOPT\b)[^.\n]{0,80}\b(are|is) (eligible|welcome)|(?<!not )(?<!in)eligible[^.\n]{0,40}(F-?1|\bCPT\b)",
                        re.I)
TEMPORARY = ("f1_cpt", "f1_opt", "stem_opt", "needs_h1b_now", "other_visa")
# The profile says "U.S. citizen or permanent resident". A posting that asks for U.S. citizenship
# itself (or a security clearance, which needs it) and never names permanent residents, green cards
# or "U.S. persons" does not take a permanent resident, so its Keep and Explore rows for that profile
# are not what the text says. The card's citizens_only covers both kinds, the profile cannot tell
# them apart, so those rows are dropped; the Skip rows stay.
STRICT_CITIZEN = re.compile(r"citizenship (is )?(required|a pre-?requisite)|citizenship required|must (be|have) (a )?U\.?S\.? citizen"
                            r"|(only|restricted to|limited to|open only to|available only to) U\.?S\.? citizens|citizen to apply"
                            r"|^\W*U\.?S\.? citizenship\b|\b(secret|top secret|TS/SCI|DoD|security) clearance", re.I | re.M)
PR_ACCEPTED = re.compile(r"permanent resid|green card|U\.?S\.? persons?\b|asylee|refugee|protected individual", re.I)
# A plain-OPT Skip needs the text to name OPT itself, not only STEM OPT or Form I-983 (plain OPT uses no I-983).
PLAIN_OPT = re.compile(r"\bOPT\b|(?i:optional practical training)")

# Required "skills" a resume's silence does not show missing: the words for these are in nearly
# every posting and in few resumes. A group made only of them is not a hard Skip reason.
SOFT_SKILLS = frozenset(skill_key(s) for s in (
    "communication", "written communication", "verbal communication", "communication skills",
    "problem solving", "technical problem solving", "problem solving skills", "leadership", "teamwork",
    "math", "mathematics", "programming", "coding", "algorithms", "microsoft office", "excel", "microsoft excel",
    "office tools", "ms office", "troubleshooting", "debugging", "documentation", "technical documentation",
    "software engineering", "software development", "software development best practices", "best practices",
    "statistics", "model evaluation", "self motivation", "resolution skills", "organizational skills",
    "interpersonal skills", "attention to detail", "time management", "initiative"))
# A skill named only as an example, or as something to learn, is not a requirement.
HEDGE = re.compile(r"interested in|\binterest in\b|or (strong )?interest\b|exposure to|such as|e\.g\.|for example|eager to learn"
                   r"|willing to learn|desire to learn|you(['’]ll| will) learn|curious|fundamentals"
                   r"|basic (understanding|knowledge)|or similar|\bcoursework\b", re.I)
# A line that offers a choice: holding any one listed item meets it.
ANY_OF = re.compile(r"\bor\b|and/or|one or more|such as|at least one|any of|e\.g\.|\betc\b|including|/", re.I)
PAREN_LIST = re.compile(r"\([^)]*(,| or |/)[^)]*\)")  # "(Docker, Kubernetes)": examples of what the line asks for
PAREN = re.compile(r"\([^()]*\)")
LIST_ANY_OF = re.compile(r"\bone of\b|one or more|at least one|any of|(two|three|several|some) of the", re.I)
CHOOSE = re.compile(r"at least one|one or more", re.I)
NICE_WORD = re.compile(r"prefer|\bplus\b|bonus|nice to have|desired|ideal", re.I)
ASK_SECTION = re.compile(r"qualif|requir|prefer|skills|you have|you bring|bring|looking for|ideal|need|must|bonus"
                         r"|\bplus\b|nice|desired|about you|education|experience", re.I)
LONG_LIST = 5  # a required line naming this many held skills is not missed for one more
LONG_NICE_LIST = 4  # the same for a preferred line
# A minimum GPA the card, profile and resume cannot show; required unless the line or its section
# says preferred.
GPA_MIN = re.compile(r"\bGPA\b[^\n]{0,40}\d\.\d|\d\.\d+[^\n]{0,30}\bGPA\b", re.I)
PREFERRED = re.compile(r"prefer|desired|nice to have|stand out|bonus|\bplus\b|ideal|not required", re.I)
_BULLET = r"[-·•*▪●]"
NEEDED = re.compile(r"requir|\bmust\b", re.I)
ASK_WORD = re.compile(r"\bexperience|\bknowledge|\bproficien|\bfamiliar|\bskill|requir|\bmust\b|\babilit|\bunderstanding"
                      r"|\bbackground|\bfluen|\binterest|\bable to", re.I)


# A missing skill the held ones already imply: a reader who sees pandas, NumPy and scikit-learn
# does not read "no Python" as a hard Skip. Missing skill -> held skills that show it.
_PY_LIBS = ("pandas", "numpy", "scikit learn", "sklearn", "pytorch", "tensorflow", "keras", "django", "flask", "fastapi",
            "jupyter", "matplotlib", "scipy", "pyspark")
_GEN_AI = ("generative ai", "genai", "prompt engineering", "ai enabled tools", "ai tools", "llms", "llm",
           "large language models", "agentic ai", "ai agents", "ai assisted development", "ai coding tools")
_GEN_AI_HELD = (*_GEN_AI, "rag", "retrieval augmented generation", "langchain", "langgraph", "llamaindex", "transformers",
                "hugging face", "openai api", "llm fine tuning", "fine tuning llms")
# A reader takes ML, computer-vision and generative-AI work as Python work.
_PY_WORK = ("machine learning", "deep learning", "computer vision", "image processing", "natural language processing",
            "nlp", "generative ai", "foundation models", "vision language models", "vlms", "large language models",
            "llms", "agentic ai", "reinforcement learning", "data science")
_EXCEL = ("excel", "microsoft excel", "ms excel", "advanced excel", "pivottables", "pivot tables", "power query",
          "power pivot", "vlookup", "excel formulas", "xlookup")
IMPLIED_BY = {"python": (*_PY_LIBS, *_PY_WORK), **{k: _GEN_AI_HELD for k in _GEN_AI}, **{k: _EXCEL for k in _EXCEL},
              "machine learning": ("scikit learn", "pytorch", "tensorflow", "keras", "deep learning", "xgboost"),
              "deep learning": ("pytorch", "tensorflow", "keras"),
              "sql": ("postgresql", "mysql", "sqlite", "sql server", "t sql", "pl/sql", "snowflake", "bigquery"),
              "javascript": ("typescript", "react", "node.js", "vue.js", "angular", "next.js"),
              # circuit design is bench work: scopes, meters and supplies come with it (Mill ac1d9d95)
              "test and measurement equipment": ("analog, digital and power circuit design", "circuit design",
                                                 "analog circuit design", "pcb design", "power electronics")}


# A citizens-only Skip rests on a shown line that asks for the status: U.S. citizenship, U.S. person status,
# permanent residency, a green card, a clearance or permanent work authorization, beside a requirement word or as
# a bullet under a requirements heading. A definition ("As defined in the ITAR, U.S. Persons include ..."), an EEO
# line ("without regard to ... citizenship status") or a "Clearance required: No" field is not one.
CITIZEN_REASON = "limited to US citizens or permanent residents"
CITIZEN_MISS = "posting shows no line that asks for citizenship, U.S. person status, permanent residency or a clearance"
STATUS_WORD = re.compile(r"citizen|U\.?S\.? persons?\b|permanent resid|green card|clearance|on a permanent basis"
                         r"|permanent (work )?authoriz", re.I)
STATUS_ASK = re.compile(r"requir|\bmust\b|\bonly\b|restricted to|limited to|contingent|pre-?requisite|eligib|\bneed"
                        r"|abilit(y|ies) to (obtain|get|hold|maintain)|able to (obtain|get|hold|maintain)", re.I)
STATUS_NOT_ASK = re.compile(r"without regard|regardless of|citizenship status|citizenship and immigration services|^\W*as defined|persons?\W* include|:\s*(no|none|n/a)\W*$",
                            re.I)
STATUS_HEADING = re.compile(r"requir|qualif|you have|you bring|you need|must|eligib|ITAR|export|clearance|citizen", re.I)
# An onsite Explore ("the role is onsite; you said remote only") rests on the text stating the work mode. The
# visibility check takes any "onsite" / "in-office" substring, so a perk or an errand counted as the mode: Axos
# a1902097 "Access to Onsite Gym", Trimble bee2e809 "on-site field visits", Rundoo 21ae5c99 "in-office lunch",
# Cboe dd3566b3 "in-office lunch" and "in office networking events" (final review, 2026-10-20). A mention right
# before a perk, an event or a visit, or right after "catered", is not the mode.
ONSITE_REASON = "the role is onsite"
ONSITE_MISS = "posting does not state the role is onsite (only a perk, event or visit is onsite)"
ONSITE_WORD = re.compile(r"on[- ]?site|in[- ]person|in the office|in[- ]office|on location", re.I)
ONSITE_PERK_AFTER = re.compile(r"\s+(?:[\w-]+\s+)?(?:lunch(?:es)?|meals?|breakfast|dinners?|food|snacks?|coffee|gym|fitness|"
                               r"caf[eé]s?|cafeteria|kitchen|parking|child ?care|clinic|amenit\w*|perks?|field visits?|"
                               r"visits?|networking|events?|happy hours?|social)\b", re.I)
ONSITE_PERK_BEFORE = re.compile(r"\bcatered\s+$", re.I)


def onsite_mode_shown(shown: str) -> bool:
    """The shown text says the work is onsite: an onsite word that is not a perk, an event or a visit."""
    return any(not ONSITE_PERK_AFTER.match(shown, m.end()) and not ONSITE_PERK_BEFORE.search(shown[:m.start()])
               for m in ONSITE_WORD.finditer(shown))


# ITAR wording that also takes anyone eligible for an export license or authorization (K2 Space "or otherwise
# eligible for a federally issued export control license", Hadrian, Rocket Lab, SpaceX "or (be) eligible to obtain
# the required authorizations from the U.S. Department of State"). The card stays citizens_only (the author's decision,
# 2026-10-19: rules.py), but the reason "limited to US citizens or permanent residents" says more than this text, so the real
# test scores no temporary-status Skip on it. "Eligible to obtain a Secret clearance" is no such route.
EXPORT_ROUTE = re.compile(r"\b(?:or|otherwise)\s+(?:be\s+)?eligible\s+(?:to\s+obtain|for)\s+(?:the\s+|an?\s+)?"
                          r"(?:required\s+|federally[- ]issued\s+)?(?:export\s+(?:control\s+)?)?(?:authori[sz]ations?|licen[cs]es?)\b",
                          re.I)


def citizen_requirement_shown(shown: str) -> bool:
    """The shown text has a line that asks for citizenship (or permanent residency, U.S. person status, a
    clearance, permanent work authorization): see STATUS_WORD."""
    heading = ""
    for line in shown.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.endswith(":") and len(s) < 90:  # RTX's "U.S. Citizen, U.S. Person, or Immigration Status Requirements:"
            heading = s
            continue
        if not STATUS_WORD.search(s) or STATUS_NOT_ASK.search(s):
            continue
        if STATUS_ASK.search(s) or (s.startswith(("-", "•", "*", "·")) and STATUS_HEADING.search(heading)):
            return True
    return False


def strict_citizen(shown: str) -> bool:
    """The shown text asks for U.S. citizenship (or a clearance) and names no permanent resident
    outside a heading (RTX's "U.S. Citizen, U.S. Person, or Immigration Status Requirements:" heads
    a "U.S. citizenship is required" line)."""
    body = "\n".join(line for line in shown.split("\n") if not line.strip().endswith(":"))
    return bool(STRICT_CITIZEN.search(body)) and not PR_ACCEPTED.search(body)


def names_plain_opt(shown: str) -> bool:
    """The text names OPT outside "STEM OPT" and "I-983/STEM OPT"."""
    return bool(PLAIN_OPT.search(re.sub(r"STEM[- ]OPT|I-983", " ", shown, flags=re.I)))


def mode_hidden_by_cut(shown: str) -> bool:
    """A work-mode section heading is cut off by a hard cut before it names a mode (fit_posting cuts hard, marked
    [truncated], only when the key text alone is over budget)."""
    heads = list(MODE_SECTION.finditer(shown))
    if not heads or not shown.rstrip().endswith("[truncated]"):
        return False
    # "whether it occurs in person or virtually" is about collaboration, not a mode
    tail = re.sub(r"in[- ]person or virtual\w*|virtual\w* or in[- ]person", " ", shown[heads[-1].end():], flags=re.I)
    # the section must run to the cut: a later heading ("Benefits:") means the section ended in view
    later_heading = any(line.strip().endswith(":") for line in tail.split("\n"))
    return not later_heading and not ANY_MODE.search(tail)


def implied_by_held(missing: str, held: tuple[str, ...]) -> str | None:
    """A held skill that shows the missing one (IMPLIED_BY), or None."""
    by = IMPLIED_BY.get(skill_key(missing), ())
    return next((h for h in held if skill_key(h) in by and skill_key(h) != skill_key(missing)), None)


# "B.S. in Computer Science or equivalent (experience)": a lower degree does not rule the candidate out.
OR_EQUIVALENT = re.compile(r"\bor (an? )?equivalent\b", re.I)
_LEVEL_WORD = {"associate": r"associate", "bachelor": r"bachelor|\bB\.?S\.?\b|\bB\.?A\.?\b|\bBSc\b",
               "master": r"master|\bM\.?S\.?\b|\bMBA\b|\bMSc\b", "doctoral": r"doctor|ph\.?\s?d"}
# A reason that names a degree level ("pursuing a bachelor degree or higher; you are not", "needs a
# bachelor degree or higher; you have an associate") needs the text to name that level. "Undergraduate"
# alone does not: an associate degree is an undergraduate degree. A pay line ("Bachelor's: $26.75 to
# $33.50 per hour") names a pay tier, not a requirement.
NAMED_LEVEL = {"bachelor": r"bachelor|baccalaureate|\bB\.?S\.?(E\.?|E\.?E\.?)?\b|\bB\.?A\.?\b|\bBSc\b|\bfour[- ]year\b"
                           r"|\b4[- ]year (degree|college|university|program)",
               "master": r"master|\bM\.?S\.?\b(?!\s*(Word|Office|Excel|Teams|Project|Access|Outlook|PowerPoint|SQL|Azure"
                         r"|Dynamics|365|Visio))|\bMBA\b|\bMSc\b|\bM\.?Eng\b|advanced degree"
                         r"|(?<!under)graduate (degree|program|studies|student)",
               "doctoral": r"doctor|ph\.?\s?d"}
LEVEL_REASON = re.compile(r"pursuing (?:an? )?(bachelor|master|doctor)(?:'s degree|al degree|ate)[^;]*; you are (?:not|pursuing)"
                          r"|needs (?:an? )?(bachelor|master|doctor)(?:'s degree|al degree|ate)")
PAY_LINE = re.compile(r"\$\s?\d")
# A Skip for a candidate who has graduated, beside a line that a graduate of that year meets
# ("Expected graduation date of Fall 2025 or later"): the text pulls the other way.
GRAD_OR_LATER = re.compile(r"graduat[^.\n]{0,60}?\b(20\d\d)\b[^.\n]{0,15}?(or later|or after|and later|onwards?|or beyond)",
                           re.I)
# A sponsorship refusal names sponsoring itself, not a "company-sponsored" benefit (McKesson: "No
# company-sponsored housing, relocation, student visa or Green Card assistance").
SPONSOR_WORD = re.compile(r"sponsor(?!ed\b)", re.I)
# A location Skip on a hybrid card whose text names no mode or office ("Locations: South SF" only):
# the card's mode is not in the shown text.
MODE_OR_OFFICE = re.compile(r"on[- ]?site|in[- ]person|hybrid|remote|based (in|at|out of)|\boffice\b", re.I)
IN_PERSON_WORD = re.compile(r"on[- ]?site|in[- ]person|hybrid|based (in|at|out of)|\boffice\b|relocat", re.I)
# A graduated candidate's Skip beside "or equivalent experience" needs the text to shut graduates out
# another way: a line for those who have graduated (Astranis: "If you have already graduated ...,
# please apply to be an Associate Engineer"), or a future graduation window (graduation_years).
ALREADY_GRADUATED = re.compile(r"\b(already|have|has) graduated\b|\bgraduated students?\b|\bnot (open|eligible) to graduates\b",
                               re.I)
# A years-of-experience range asked of the candidate ("Ideally you will have 2-3 years' work
# experience"), beside an overqualified reason that says 0 asked. A company's "60 years of experience"
# is not a range.
YEARS_RANGE_ASKED = re.compile(r"\b[1-9]\d?\s*(-|–|to)\s*\d+\s*years?['’]?\s*(of\s+)?(work|relevant|related|professional"
                               r"|industry)?\s*experience", re.I)
# A list heading that makes its bullets wishes, not requirements ("The ideal candidate has:").
SOFT_HEADING = re.compile(r"ideal candidate|nice to have|preferred|bonus|plus\b", re.I)
RELATED_FIELD = re.compile(r"or (an? )?(other )?related (discipline|field|area|major)", re.I)


def unnamed_levels(shown: str, reasons: list[str]) -> list[str]:
    """Degree levels a below-level reason names that the shown text (pay lines aside) never names."""
    body = "\n".join(line for line in shown.split("\n") if not PAY_LINE.search(line))
    levels = [{"doctor": "doctoral"}.get(lv, lv) for r in reasons if (m := LEVEL_REASON.search(r))
              for lv in [m.group(1) or m.group(2)]]
    return [lv for lv in levels if not re.search(NAMED_LEVEL[lv], body, re.I)]


# A student-worker program is not an internship to a reader, but CPT may still cover it.
STUDENT_WORKER = re.compile(r"student[- ]worker|student employee|student assistant", re.I)
# CPT is granted through the school for one employer: "authorized to work ... for anyone other than
# current employer and/or education institution" (ConocoPhillips) shuts it out though it never names CPT.
ANY_OTHER_EMPLOYER = re.compile(r"authori[sz]ed to work[^.\n]{0,120}?\bother than (your |the |their )?(current )?employer", re.I)


def or_equivalent_degree(shown: str, level: str) -> bool:
    """"Or equivalent" follows the asked degree level within its line (120 characters on; by line, as
    "B.S. or M.S." would cut a sentence split)."""
    pat = _LEVEL_WORD.get(level)
    return bool(pat) and any(OR_EQUIVALENT.search(line[m.start(): m.end() + 120])
                             for line in shown.split("\n") for m in re.finditer(pat, line, re.I))


def under_soft_heading(shown: str, name: str) -> bool:
    """Every line naming the skill sits in a list whose heading makes it a wish ("The ideal candidate
    has:", "Preferred:")."""
    lines = shown.split("\n")
    hits = [n for n, line in enumerate(lines) if has_phrase(line, name)]
    return bool(hits) and all((j := lead_in(lines, n)) is not None and SOFT_HEADING.search(lines[j]) for n in hits)


def unmet_groups(cand: CandidateCard, card: RequirementCard) -> list[tuple[str, ...]]:
    """Required groups the rule finds unmet: read as rules.label reads them (synonyms, a concrete skill
    that shows a generic one, the degree field's skills; soft-skill groups are not enforced)."""
    have = held_skills(cand)
    return [g for g in required_groups(card) if not group_met(g, have)]


def missing_nice(cand: CandidateCard, card: RequirementCard) -> list[str]:
    """Counted nice-to-haves the rule finds missing (rules.nice_gap)."""
    have = held_skills(cand)
    return [s for s in counted_nice(card) if not group_met((s,), have)]


def missing_skill_names(cand: CandidateCard, card: RequirementCard) -> list[str]:
    return [x for g in unmet_groups(cand, card) for x in g] + missing_nice(cand, card)


def sentences(shown: str) -> list[str]:
    return re.split(r"(?<=[.;!?])\s+|\n+", shown)


def lead_in(lines: list[str], n: int) -> int | None:
    """The line ending in ':' that introduces the list line n sits in ("one of the following areas:")."""
    for j in range(n - 1, max(-1, n - 40), -1):
        if lines[j].strip().endswith(":"):
            return j
    return None


def list_block(lines: list[str], j: int) -> str:
    """The lines a lead-in introduces, up to the next line ending in ':'."""
    out = []
    for line in lines[j + 1: j + 40]:
        if line.strip().endswith(":"):
            break
        out.append(line)
    return "\n".join(out)


def anyof_met(shown: str, missing: str, held: tuple[str, ...]) -> bool:
    """The missing skill is one choice of several and the candidate holds another: a sentence naming
    it offers a choice and names a held skill, or it is an item of a list introduced as "one of the
    following" whose other items include a held skill."""
    others = [h for h in held if skill_key(h) != skill_key(missing)]
    lines = shown.split("\n")
    for n, line in enumerate(lines):
        for sent in sentences(line):
            if not has_phrase(sent, missing):
                continue
            if (ANY_OF.search(sent) or PAREN_LIST.search(sent)) and any(has_phrase(sent, h) for h in others):
                return True
            j = lead_in(lines, n)
            if j is not None and LIST_ANY_OF.search(lines[j]) and any(has_phrase(list_block(lines, j), h) for h in others):
                return True
    return False


def asked(shown: str, name: str) -> bool:
    """Some sentence naming the skill asks for it (experience with, knowledge of, a plus...) or
    sits in a qualifications section; a skill named only in the duties or the team description is
    not something the posting asks the candidate to have. A name the text never spells out is left
    to visibility.missing_facts (True here)."""
    named, asks = asking_sentences(shown, name)
    return bool(asks) or not named


def choice_unclear(shown: str, name: str) -> bool:
    """A line naming the skill asks for "at least one" or "one or more" of something: whether the
    candidate's other skills count the card cannot say."""
    return any(CHOOSE.search(s) for s in sentences(shown) if has_phrase(s, name))


def _name_re(name: str) -> str:
    return r"(?<![\w+#.])" + re.escape(name) + r"(?![\w+#])"


def slash_neighbour(shown: str, missing: str, held: tuple[str, ...]) -> bool:
    """The missing skill stands next to a held one across a slash ("C/C++", "JavaScript/ TypeScript")."""
    keys = {skill_key(h) for h in held if skill_key(h) != skill_key(missing)}
    flags = re.I if len(missing) > 2 else 0
    for m in re.finditer(_name_re(missing), shown, flags):
        after = re.match(r"\s*/\s*([\w+#.]+(?: [\w+#.]+)?)", shown[m.end():])
        before = re.search(r"([\w+#.]+(?: [\w+#.]+)?)\s*/\s*$", shown[:m.start()])
        for g in (after, before):
            if g and any(skill_key(g.group(1)) == k or skill_key(g.group(1)).split()[:1] == [k]
                         or skill_key(g.group(1)).split()[-1:] == [k] for k in keys):
                return True
    return False


def in_long_list(shown: str, missing: str, held: tuple[str, ...], n: int = LONG_LIST) -> bool:
    """A sentence naming the missing skill names n or more held skills."""
    return any(sum(has_phrase(s, h) for h in held) >= n for s in sentences(shown) if has_phrase(s, missing))


def section_of(lines: list[str], n: int) -> str:
    """The nearest short line above line n that is not a bullet item: the section heading, roughly."""
    for j in range(n - 1, -1, -1):
        line = lines[j].strip()
        if not line or re.match(_BULLET, line) or len(line) >= 90:
            continue
        prev = next((lines[k].strip() for k in range(j - 1, -1, -1) if lines[k].strip()), "")
        if re.fullmatch(_BULLET, prev):  # the text of a bullet whose marker sits on its own line
            continue
        return line
    return ""


def required_gpa(shown: str) -> str | None:
    """The first line stating a minimum GPA as a requirement, or None."""
    lines = shown.split("\n")
    for n, line in enumerate(lines):
        if GPA_MIN.search(line) and not PREFERRED.search(line) and not PREFERRED.search(section_of(lines, n)):
            return line.strip()
    return None


def graduation_years(text: str) -> set[int]:
    """Every future graduation year the posting names: a year within a line's reach (40 characters
    before, 120 after) of a graduation word."""
    years: set[int] = set()
    for line in text.split("\n"):
        for m in GRAD_WORD.finditer(line):
            years.update(int(y) for y in YEAR.findall(line[max(0, m.start() - 40): m.end() + 120]))
    return {y for y in years if CURRENT_YEAR < y <= CURRENT_YEAR + 6}


# Graduation dates with a month or season ("on or after December 2027", "between Fall 2027 and
# Spring 2029"). The card holds a year only, and the resume writer picks the month, so a year is safe
# only when every month of it meets the bound. A token's months as (first, last); winter may sit on
# either side of New Year.
_SEASON = {"spring": (1, 6), "summer": (5, 9), "fall": (8, 12), "autumn": (8, 12), "winter": (-1, 14)}
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_MONTH_WORD = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?|spring|summer|fall|autumn|winter"
DATE_TOKEN = re.compile(rf"\b((?:{_MONTH_WORD})(?:\s*(?:/|or|-)\s*(?:{_MONTH_WORD}))*)\s*(?:of\s+)?,?\s*(20\d\d)\b|\b(20\d\d)\b",
                        re.I)
OPEN_LATE = re.compile(r"or (later|after|beyond|thereafter)|and (later|beyond|after)|on or after|\bnot (graduat\w* )?before"
                       r"|\b(after|beyond)\b", re.I)
OPEN_EARLY = re.compile(r"\bby\b|on or before|or (sooner|earlier|before)|no later than|(?<!not )(?<!graduate )\bbefore\b", re.I)


def _month_span(word: str) -> tuple[int, int]:
    w = word.lower().rstrip(".")
    if w in _SEASON:
        return _SEASON[w]
    m = _MONTHS.index(w[:3]) + 1
    return m, m


def _date_tokens(reach: str) -> list[tuple[int, int, bool]]:
    """(first, last, has a month) as absolute month numbers (year * 12 + month) for each date."""
    out = []
    for m in DATE_TOKEN.finditer(reach):
        if m.group(3):
            y = int(m.group(3))
            out.append((y * 12 + 1, y * 12 + 12, False))
            continue
        y = int(m.group(2))
        spans = [_month_span(w) for w in re.split(r"\s*(?:/|\bor\b|-)\s*", m.group(1)) if w.strip()]
        out.append((y * 12 + min(a for a, _ in spans), y * 12 + max(b for _, b in spans), True))
    return out


def graduation_bounds(text: str) -> list[tuple[int | None, int | None, int | None, int | None]]:
    """For each line stating a graduation date with a month or season (and some future year):
    (earliest first, earliest last, latest first, latest last) absolute months; the lower pair is
    None when the line says "by" / "or sooner", the upper pair None for "or later" / "not before"."""
    out = []
    for line in text.split("\n"):
        for g in GRAD_WORD.finditer(line):
            before = line[max(0, g.start() - 40): g.start()]
            after = re.split(r"(?<=\d)\.|\.\s+(?=[A-Z])", line[g.start(): g.end() + 120], maxsplit=1)[0]
            reach = before + after
            toks = _date_tokens(reach)
            if not any(t[2] for t in toks) or not any(t[1] > (CURRENT_YEAR + 1) * 12 for t in toks):
                continue
            late, early = OPEN_LATE.search(reach), OPEN_EARLY.search(reach)
            lo = min(toks, key=lambda t: t[0])
            hi = max(toks, key=lambda t: t[1])
            out.append((*((None, None) if early and not late else (lo[0], lo[1])),
                        *((None, None) if late and not early else (hi[0], hi[1]))))
            break  # one bound per line
    return out


def month_safe_years(text: str) -> set[int] | None:
    """Years every month of which meets every month-bounded graduation line, or None when the
    posting states no graduation date with a month or season."""
    bounds = graduation_bounds(text)
    if not bounds:
        return None
    ok = set()
    for y in range(CURRENT_YEAR - 6, CURRENT_YEAR + 7):
        if all((lo_last is None or y * 12 + 1 >= lo_last) and (hi_first is None or y * 12 + 12 <= hi_first)
               for _, lo_last, hi_first, _ in bounds):
            ok.add(y)
    return ok


RESUME_GRAD = re.compile(rf"\b({_MONTH_WORD})\s*(?:of\s+)?,?\s*(20\d\d)\b", re.I)


def resume_month_ok(resume: str, posting: str, year: int | None) -> str | None:
    """Why the resume's graduation date does not show the posting's month-bounded window, or None.
    A month the resume names must sit inside every bound; with no month shown, the year must be safe
    at every edge (month_safe_years)."""
    bounds = graduation_bounds(posting)
    if not bounds or year is None:
        return None
    shown = []
    for line in resume.split("\n"):
        if re.search(r"graduat|expected|anticipated|class of|\b(B\.?S|B\.?A|M\.?S|bachelor|master|ph\.?d|associate)", line, re.I):
            shown += [(int(m.group(2)), _month_span(m.group(1))) for m in RESUME_GRAD.finditer(line)
                      if int(m.group(2)) == year]
    for y, (a, b) in shown:
        for lo_first, _, _, hi_last in bounds:
            if (lo_first is not None and y * 12 + b < lo_first) or (hi_last is not None and y * 12 + a > hi_last):
                return f"resume graduation date is outside the posting's window ({year})"
    if not shown and year not in (month_safe_years(posting) or set()):
        return f"resume shows no graduation month and {year} meets the posting's month window only in part"
    return None


# Class-level wording names a graduation window the card's year-only fields cannot hold: "Must be a Rising
# Junior/Senior", "Currently at the junior or senior undergraduate level", "Completion of at least three
# years of an undergraduate ... program", "Sophomore year must be complete by the start of the summer
# internship", "at a minimum, entering their Sophomore by the start of summer 2027". Now is fall
# CURRENT_YEAR, and a four-year bachelor's student in class k (freshman 1 ... senior 4) graduates in
# CURRENT_YEAR + 5 - k; after the summer S of the internship, a rising class-k student graduates in
# S + 5 - k. The window binds undergraduates; a graduate student fits it when the line names graduate
# students too, and only "completed years" wording is met by any graduate (rules.label reads none of this).
_CLASS = r"(?:freshm[ae]n|sophomores?|juniors?|seniors?)"
_STAFF = r"(?!\s+(?:level\s+)?(?:engineers?|developers?|staff|team|members?|management|managers?|leaders?|leadership|scientists?"\
         r"|analysts?|designers?|associates?|colleagues|employees|executives|professionals)\b)"
CLASS_LIST = re.compile(rf"\b({_CLASS}(?:\s*(?:/|,|\bor\b|\band\b|&)\s*(?:an?\s+)?{_CLASS})+)\b{_STAFF}", re.I)
RISING = re.compile(rf"\brising\s+({_CLASS}(?:\s*(?:/|,|\bor\b|\band\b|&)\s*(?:an?\s+)?{_CLASS})*)\b", re.I)
_N_WORD = {"one": 1, "two": 2, "three": 3, "four": 4, "1": 1, "2": 2, "3": 3, "4": 4}
YEARS_DONE = re.compile(r"\b(?:completion of|completed|complete|finished)\s+(?:at least\s+)?(one|two|three|four|[1-4])\s+"
                        r"(?:full\s+)?(?:academic\s+)?years?\s+of\s+(?:an?\s+|your\s+|their\s+)?"
                        r"(?:undergraduate|college|university|bachelor)", re.I)
CLASS_DONE = re.compile(r"\b(freshman|sophomore|junior)\s+year\s+(?:must\s+be\s+|is\s+|will\s+be\s+)?(?:complete|completed|finished)\b"
                        r"|\b(?:completion of|completed|complete|finished)\s+(?:your\s+|their\s+|the\s+)?(freshman|sophomore|junior)\s+year\b",
                        re.I)
ENTERING = re.compile(r"\bentering\s+(?:their|the|your|his or her)?\s*(sophomore|junior|senior)\b", re.I)
AT_LEAST = re.compile(r"at (a )?minimum|at least|or (higher|above|beyond)", re.I)
GRAD_STUDENT = re.compile(r"\bgraduate (student|program|degree|level)|master|\bph\.?\s?d|doctora|advanced degree|\(?or higher\b", re.I)
SUMMER = re.compile(r"\bsummer\s*(?:of\s+)?(20\d\d)\b", re.I)
# "Receive a Ph.D. no later than spring 2027": the degree the card asks for, finished by a date.
DEGREE_BY = re.compile(r"\b(?:receive|complete|obtain|earn|finish|be awarded|conferr\w*)\w*\s+(?:an?\s+|the\s+|your\s+|their\s+)?"
                       r"(?:ph\.?\s?d\.?|doctora\w*|master'?s?|bachelor'?s?|degree)[^.\n]{0,40}?\b(?:no later than|by|before|prior to)\s+"
                       rf"(?:(?:{_MONTH_WORD})\s+)?(?:of\s+)?(20\d\d)\b", re.I)


def internship_summer(shown: str) -> int:
    """The summer the posting is for: the first future "Summer 20XX" it names, else next summer."""
    years = [int(y) for y in SUMMER.findall(shown) if int(y) > CURRENT_YEAR]
    return years[0] if years else CURRENT_YEAR + 1


def class_rank(word: str) -> int:
    """Freshman 1, sophomore 2, junior 3, senior 4."""
    return {"f": 1, "so": 2, "j": 3, "se": 4}[next(k for k in ("f", "so", "j", "se") if word.lower().startswith(k))]


def _ranks(words: str) -> list[int]:
    return [class_rank(w) for w in re.findall(_CLASS, words, re.I)]


def class_window(shown: str) -> tuple[set[int], bool] | None:
    """(graduation years an undergraduate student may have, whether a graduate student fits too), from the
    class-level lines the shown text states, or None when it states none. Lines under a preferred wording
    and pay-tier lines ("Intern/Junior/Senior: $34.00/hour") do not count."""
    summer = internship_summer(shown)
    students = set(range(CURRENT_YEAR + 1, CURRENT_YEAR + 7))
    allowed: set[int] | None = None
    grads_ok = True
    for line in shown.split("\n"):
        if PREFERRED.search(line) or PAY_LINE.search(line):
            continue
        sets: list[set[int]] = []
        rest = line
        for m in RISING.finditer(line):  # rising class k after the summer: graduates summer + 5 - k
            sets.append({summer + 5 - k for k in _ranks(m.group(1))})
            rest = rest.replace(m.group(0), " ")
        for m in CLASS_LIST.finditer(rest):  # in class k now (fall CURRENT_YEAR)
            sets.append({CURRENT_YEAR + 5 - k for k in _ranks(m.group(1))})
        for m in YEARS_DONE.finditer(line):  # N years done (read as now, the stricter reading)
            sets.append({y for y in students if y <= CURRENT_YEAR + 4 - _N_WORD[m.group(1).lower()]})
        for m in CLASS_DONE.finditer(line):  # class k finished by the summer
            sets.append({y for y in students if y <= summer + 4 - class_rank(m.group(1) or m.group(2))})
        for m in ENTERING.finditer(line):  # entering class k by the summer
            top = summer + 5 - class_rank(m.group(1))
            sets.append({y for y in students if y <= top} if AT_LEAST.search(line) else {top})
        if not sets:
            continue
        ok = set().union(*sets)
        if not (GRAD_STUDENT.search(line) or YEARS_DONE.search(line) or CLASS_DONE.search(line)):
            grads_ok = False
        allowed = ok if allowed is None else allowed & ok
    return None if allowed is None else (allowed, grads_ok)


def class_problem(cand: CandidateCard, shown: str) -> str | None:
    """Why a student candidate's graduation year does not fit the posting's class-level wording, or None."""
    if not is_student(cand):
        return None
    win = class_window(shown)
    if win is None:
        return None
    years, grads_ok = win
    if cand.degree_level in ("bachelor", "associate"):
        if cand.graduation_year not in years:
            return f"candidate graduates in {cand.graduation_year}; the posting's class level allows {sorted(years) or 'no year'}"
    elif not grads_ok:
        return "posting asks for an undergraduate class level; the candidate is a graduate student"
    return None


MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s?([kK]\b)?")


def money_figures(shown: str) -> list[int]:
    """Every yearly-sized dollar figure the shown text states ("$98k base; $111k OTE" gives 98,000 and
    111,000); hourly figures under $1,000 are left out."""
    out = []
    for m in MONEY.finditer(shown):
        v = float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)
        if v >= 1000:
            out.append(int(v))
    return out


def degree_deadline(shown: str) -> int | None:
    """The year by which the posting wants the degree finished ("Receive a Ph.D. no later than spring
    2027"), or None."""
    years = [int(m.group(1)) for m in DEGREE_BY.finditer(shown)]
    return min(years) if years else None


EXAMPLE =re.compile(r"such as|e\.g\.|for example|including|\blike\b", re.I)


def held_example(shown: str, missing: str, held: tuple[str, ...]) -> bool:
    """A sentence names the missing skill and then gives a held skill as its example
    ("cheminformatics tools (e.g., RDKit)", "deep learning frameworks such as PyTorch"): the
    candidate shows the missing skill through the held one."""
    others = [h for h in held if skill_key(h) != skill_key(missing)]
    for s in sentences(shown):
        # the example runs to the end of its clause: "tools, including Claude, and data analysis"
        cuts = [(m.start(), re.split(r",\s*and\b|;|\)", s[m.end():], maxsplit=1)[0]) for m in EXAMPLE.finditer(s)]
        cuts += [(m.start(), m.group(0)) for m in PAREN_LIST.finditer(s)]
        # the example belongs to the phrase just before it: "Python and the data stack (pandas, ...)"
        # gives examples of the data stack, not of Python
        head = lambda at: re.split(r",\s*|\band\b|\bor\b", s[:at].rstrip(" ,"))[-1]  # noqa: E731
        if any(has_phrase(head(at), missing) and any(has_phrase(tail, h) for h in others) for at, tail in cuts):
            return True
    return False


def paren_with_held(shown: str, missing: str, held: tuple[str, ...]) -> bool:
    """The missing skill sits in one parenthetical with a held one ("Python and its ML ecosystem
    (NumPy and pandas)" for a candidate with NumPy): the brackets spell out a broader ask the
    candidate meets in part, so whether one missing item is a hard Skip the card cannot say."""
    others = [h for h in held if skill_key(h) != skill_key(missing)]
    return any(has_phrase(m.group(0), missing) and any(has_phrase(m.group(0), h) for h in others)
               for m in PAREN.finditer(shown))


def asking_sentences(shown: str, name: str) -> tuple[list[str], list[str]]:
    """(sentences naming the skill, those of them that ask for it: an ask word, a "plus", or a line
    in a qualifications section). A title, a team name or a duty asks for nothing."""
    named, asks = [], []
    lines = shown.split("\n")
    for n, line in enumerate(lines):
        for sent in sentences(line):
            if has_phrase(sent, name):
                named.append(sent)
                if ASK_WORD.search(sent) or NICE_WORD.search(sent) or ASK_SECTION.search(section_of(lines, n)):
                    asks.append(sent)
    return named, asks


def paren_example(shown: str, name: str) -> bool:
    """Every sentence asking for the skill names it only inside a parenthetical list of examples
    ("programming skills (Python, C++, PyTorch)")."""
    named, asks = asking_sentences(shown, name)
    asks = asks or named
    return bool(asks) and all(any(has_phrase(m.group(0), name) for m in PAREN_LIST.finditer(s)) for s in asks)


def hedged(shown: str, name: str) -> bool:
    """Every sentence of the shown posting that asks for `name` (or, when none does, every sentence
    naming it) names it as an example or something to learn, with no "required" or "must" in the
    same sentence."""
    named, asks = asking_sentences(shown, name)
    asks = asks or named
    return bool(asks) and all(HEDGE.search(s) and not NEEDED.search(s) for s in asks)


def guard(kind: str, cand: CandidateCard, card: RequirementCard, bucket: str, shown: str) -> str | None:
    """Why this row's label is not what the shown text says, or None when it holds."""
    if kind in AWAY_KINDS and not away_ok(cand.current_location, card, shown):
        return "candidate city is in the posting's metro or named in it"
    if kind == "employment_type":
        wanted = [t for t in cand.employment_types_wanted if re.search(TYPE_PHRASE[t], shown, re.I)]
        if wanted:
            return f"posting names the wanted job type ({wanted[0]})"
    names = missing_skill_names(cand, card)
    if any(has_phrase(s, m) or has_phrase(m, s) for s in cand.skills for m in names):
        return "a held skill contains the missing skill's name, or the reverse"
    if kind in ("must_skill", "nice_skill") and any(slash_neighbour(shown, m, cand.skills) for m in names):
        return "posting pairs the missing skill with a held one across a slash"
    if kind == "nice_skill":
        gone = missing_nice(cand, card)
        if any(skill_key(m) in SOFT_SKILLS for m in gone):
            return "missing skill is a soft or generic word"
        if any(anyof_met(shown, m, cand.skills) for m in gone):
            return "posting lists the missing nice-to-have skill as one choice among others the candidate holds"
        if any(hedged(shown, m) or paren_example(shown, m) for m in gone):
            return "posting names the missing skill only as an example or something to learn"
        if any(in_long_list(shown, m, cand.skills, LONG_NICE_LIST) for m in gone):
            return "missing skill is one item of a long list the candidate otherwise holds"
        if any(choice_unclear(shown, m) for m in gone):
            return "posting asks for at least one of a kind the card cannot spell out"
        if any(not asked(shown, m) for m in gone):
            return "posting names the missing skill only in the duties, not as something to have"
    if kind in ("must_skill", "nice_skill"):
        implied = [(m, h) for m in names if (h := implied_by_held(m, cand.skills))]
        if implied:
            return f"a held skill implies the missing one ({implied[0][1]} -> {implied[0][0]})"
    if kind in ("enrollment_level", "degree_level") and or_equivalent_degree(shown, card.enrolled_in if kind ==
                                                                              "enrollment_level" else card.degree_level):
        return "posting takes an equivalent in place of the degree"
    if kind == "enrollment_graduated" and or_equivalent_degree(shown, card.enrolled_in) and not (
            ALREADY_GRADUATED.search(shown) or (cand.graduation_year is not None and any(
                cand.graduation_year < y for y in graduation_years(shown)))):
        return "posting takes an equivalent in place of the degree, and no line shuts out graduates"
    if kind == "degree_field" and any(RELATED_FIELD.search(line) and any(has_phrase(line, s) for s in cand.skills)
                                      for line in shown.split("\n")):
        return "posting takes a related field, and the candidate's skills are fields it names"
    reasons = label(cand, card)[1] if bucket == "Excluded" else []
    if cand.work_auth in TEMPORARY and any(CITIZEN_REASON in r for r in reasons) and EXPORT_ROUTE.search(shown):
        return "posting also takes anyone eligible for an export license; the reason says citizens or permanent residents only"
    if (lv := unnamed_levels(shown, reasons)):
        return f"the reason names a {lv[0]} degree; the shown text never names that level"
    if kind == "enrollment_graduated" and cand.graduation_year is not None and any(
            int(m.group(1)) <= cand.graduation_year for m in GRAD_OR_LATER.finditer(shown)):
        return "posting takes a graduation year the candidate's meets ('YEAR or later')"
    if any("will not sponsor" in r for r in reasons) and not SPONSOR_WORD.search(shown):
        return "posting names no sponsorship refusal, only a company-sponsored benefit"
    if kind == "location" and bucket == "Excluded" and card.work_mode == "hybrid" and not MODE_OR_OFFICE.search(shown):
        return "hybrid card, but the shown text names no work mode or office, only the location header"
    if (kind == "location" and bucket == "Excluded" and card.work_mode != "remote"
            and re.search(r"remote(?! sensing)", shown, re.I) and not IN_PERSON_WORD.search(shown)):
        return "the shown text names remote (Rundoo's interview steps) and never an in-person mode"
    if cand.work_auth == "f1_cpt" and bucket != "For you" and STUDENT_WORKER.search(shown):
        return "posting is a student-worker role; whether CPT covers it the card cannot say"
    if cand.work_auth == "f1_cpt" and bucket != "Excluded" and ANY_OTHER_EMPLOYER.search(shown):
        return "posting asks for authorization to work for anyone other than the current employer or school, which CPT is not"
    if kind == "must_skill":
        groups = unmet_groups(cand, card)
        if groups and all(all(skill_key(x) in SOFT_SKILLS for x in g) for g in groups):
            return "missing skill is a soft or generic word"
        if groups and any(all(hedged(shown, x) or paren_example(shown, x) for x in g) for g in groups):
            return "posting names the missing skill only as an example or something to learn"
        if groups and any(any(in_long_list(shown, x, cand.skills) for x in g) for g in groups):
            return "missing skill is one item of a long list the candidate otherwise holds"
        if groups and any(any(held_example(shown, x, cand.skills) for x in g) for g in groups):
            return "posting gives a held skill as an example of the missing one"
        if groups and any(any(paren_with_held(shown, x, cand.skills) for x in g) for g in groups):
            return "posting names the missing skill in one parenthetical with a held one"
        if groups and any(not any(asked(shown, x) for x in g) for g in groups):
            return "posting names the missing skill only in the duties, not as something to have"
        if groups and any(all(under_soft_heading(shown, x) for x in g) for g in groups):
            return "posting names the missing skill only under an ideal-candidate or preferred heading"
    if bucket != "Excluded":
        gpa = required_gpa(shown)
        if gpa:
            return "posting requires a minimum GPA the card cannot hold"
        safe = month_safe_years(shown)
        yrs = graduation_years(shown)
        if safe is not None and not any(y > CURRENT_YEAR for y in safe):
            return "posting's graduation window has month edges no year-only card meets"
        if safe is not None and cand.graduation_year not in safe:
            return "candidate graduation year meets the posting's month window only in part"
        if yrs and cand.graduation_year is not None and not min(yrs) <= cand.graduation_year <= max(yrs) and safe is None:
            return "candidate graduation year is outside the window the posting states"
        why = class_problem(cand, shown)
        if why:
            return why
        if (cand.salary_floor is not None and card.salary_max is not None and card.salary_max < cand.salary_floor
                and any(v >= cand.salary_floor for v in money_figures(shown))):
            return "posting also states a pay figure at or above the floor (total pay or another range)"
        by = degree_deadline(shown)
        if by is not None and kind == "degree_still_enrolled":
            return f"posting takes the degree finished by {by}, which the rule's 'still enrolled' reason cannot read"
        if by is not None and is_student(cand) and cand.graduation_year > by:
            return f"candidate finishes the degree in {cand.graduation_year}; the posting wants it by {by}"
    if kind in RELOCATE_KINDS and LOCAL_ONLY.search(shown):
        return "posting asks for local candidates"
    if kind in AWAY_KINDS and mode_hidden_by_cut(shown):
        return "the cut ends inside the posting's work-model section, before it names a mode"
    if card.citizens_only and cand.work_auth == "citizen_or_pr" and bucket != "Excluded" and strict_citizen(shown):
        return "posting asks for U.S. citizenship and names no permanent resident; the profile says citizen or permanent resident"
    if cand.work_auth == "f1_opt" and "work_auth:no_opt" in label(cand, card)[2] and not names_plain_opt(shown):
        return "posting bars STEM OPT or Form I-983 only; plain OPT it does not name"
    if kind in MODE_MATCH_KINDS:
        rejected = {"hybrid_or_remote": ("onsite",), "remote_only": ("onsite", "hybrid")}.get(cand.remote_preference, ())
        if any(re.search(MODE_WORD[m], shown, re.I) for m in rejected):
            return "posting also names a work mode the preference rejects"
    if kind == "overqualified" and YEARS_CAP.search(shown):
        return "posting caps years of experience"
    if kind == "overqualified" and card.min_years == 0 and YEARS_RANGE_ASKED.search(shown):
        return "posting asks a range of years the reason's '0 asked' does not match"
    welcome = F1_WELCOME.search(shown)
    named = welcome and re.search({"f1_cpt": r"CPT|F-?1", "f1_opt": r"OPT", "stem_opt": r"OPT"}.get(cand.work_auth, "$^"),
                                  welcome.group(0), re.I)
    if cand.work_auth in TEMPORARY and not card.citizens_only and bucket != "Excluded" and not named:
        if NO_TEMPORARY.search(shown):
            return "posting bars temporary visas or future sponsorship, which the card cannot hold"
        if cand.work_auth == "needs_h1b_now" and MUST_BE_AUTHORIZED.search(shown):
            return "posting requires current work authorization"
    return None


# ---------------------------------------------------------------- candidates

def graduation_window(text: str) -> int | None:
    """The latest future graduation year the posting names (graduation_years), or None."""
    years = graduation_years(text)
    return max(years) if years else None


def base_candidate(card: RequirementCard, text: str) -> tuple[CandidateCard, CandidateCard, list[str]]:
    """(raw full match, full match fitted to the posting text, notes on what was fitted)."""
    raw = full_match(card)
    shown = shown_posting(text)
    notes: list[str] = []
    fitted = raw
    win = graduation_window(shown)
    if win is not None:
        fitted = replace(fitted, graduation_year=win)
        if win != raw.graduation_year:
            notes.append(f"graduation {raw.graduation_year}->{win} (posting window)")
    elif card.enrolled_in != "none":
        notes.append("enrolled card, no graduation window in the text")
    elif card.employment_type == "internship" and ENROLL_EXTRA.search(shown) and not RECENT_GRAD.search(shown):
        # the card does not ask for enrollment but the text does: a 2026 graduate is not a student
        # during the internship
        fitted = replace(fitted, graduation_year=CURRENT_YEAR + 2)
        notes.append(f"graduation {raw.graduation_year}->{CURRENT_YEAR + 2} (posting is for current students)")
    safe = sorted(y for y in (month_safe_years(shown) or ()) if y > CURRENT_YEAR)
    if safe and fitted.graduation_year not in safe:
        y = min(safe, key=lambda y: abs(y - (fitted.graduation_year or CURRENT_YEAR)))
        notes.append(f"graduation {fitted.graduation_year}->{y} (every month of it meets the posting's month window)")
        fitted = replace(fitted, graduation_year=y)
    cw = class_window(shown)
    if cw and is_student(fitted) and fitted.degree_level in ("bachelor", "associate") and fitted.graduation_year not in cw[0]:
        pool = sorted(cw[0] & set(safe)) if safe else sorted(cw[0])
        if pool:
            y = min(pool, key=lambda y: abs(y - fitted.graduation_year))
            notes.append(f"graduation {fitted.graduation_year}->{y} (the posting's class level)")
            fitted = replace(fitted, graduation_year=y)
    if card.employment_type == "internship" and fitted.years != card.min_years:
        fitted = replace(fitted, years=card.min_years)
        notes.append(f"years {raw.years}->{card.min_years} (internship)")
    if not years_fit(fitted) and not is_student(fitted):
        # a graduate of three or more years back with no years of work: an internship's graduate is a recent one
        gy = graduated_year(random.Random(0), fitted.years, grad_floor(fitted.years, pre=pre_degree_years(fitted)),
                            CURRENT_YEAR - 1)
        notes.append(f"graduation {fitted.graduation_year}->{gy} ({fitted.years} years of work)")
        fitted = replace(fitted, graduation_year=gy)
    # the full match wants the posting's industry and pay when the shown text states them (since v3/rules
    # variants.full_match sets no preference), so the preference rows have a stated fact to test
    if card.industry and has_phrase(shown, card.industry):
        fitted = replace(fitted, industries_wanted=(card.industry,))
    elif card.industry:
        fitted = replace(fitted, industries_wanted=())
        notes.append("industry preference left unset (posting does not name it)")
    if card.salary_min is not None and card.salary_max and has_money(shown, card.salary_max):
        fitted = replace(fitted, salary_floor=card.salary_min)
    elif card.salary_min is not None:
        fitted = replace(fitted, salary_floor=None)
        notes.append("salary floor left unset (posting does not state the pay)")
    if label(fitted, card)[0] != "For you":  # a preference the rule reads otherwise: leave both unset
        fitted = replace(fitted, industries_wanted=(), salary_floor=None)
        notes.append("industry and pay preferences left unset (the rule does not read them as met)")
    return raw, fitted, notes


def is_student(c: CandidateCard) -> bool:
    return c.graduation_year is not None and c.graduation_year > CURRENT_YEAR


def person_problems(c: CandidateCard) -> list[str]:
    """Ways the candidate is not a person who could exist, as build_pairs checks its candidates."""
    out = [*status_problems(c), *license_problems(c)]
    if not years_fit(c):
        out.append("years of work do not fit the graduation year")
    if not prefs_fit(c):
        out.append("wants internships and avoids 'intern' titles")
    if unique_skills(c.skills) != c.skills:
        out.append("lists a skill twice")
    if industry_clash(c):
        out.append("wants and avoids one industry")
    return out


def settle(cand: CandidateCard, card: RequirementCard, rng: random.Random) -> tuple[CandidateCard, list[str]]:
    """(the candidate as a person who could exist, the problems left). Skills are listed once, the
    education follows the work authorization (variants.with_status), years of work follow the graduation
    year (variants.settle_years). A candidate with problems left is dropped by build_rows."""
    c = cand
    if unique_skills(c.skills) != c.skills:
        c = replace(c, skills=unique_skills(c.skills))
    if status_problems(c):
        c = with_status(c, c.work_auth, card, rng) or c
    if not years_fit(c):
        c = settle_years(c, rng, card.min_years) or c
    return c, person_problems(c)


def fit(cand: CandidateCard, raw: CandidateCard, fitted: CandidateCard) -> CandidateCard:
    """Carry the posting fit into a variant: fields the variant left at the raw full-match value
    take the fitted value; fields the variant edited stay."""
    changes = {f: getattr(fitted, f) for f in cand.__dataclass_fields__
               if getattr(cand, f) == getattr(raw, f) and getattr(fitted, f) != getattr(raw, f)}
    return replace(cand, **changes) if changes else cand


def cd_id_of(cand: CandidateCard) -> str:
    return "cd_rt_" + hashlib.sha1(json.dumps(cand.to_dict(), sort_keys=True).encode()).hexdigest()[:10]


def pair_id_of(rq_id: str, cd_id: str) -> str:
    return "pr_" + hashlib.sha1(f"{rq_id}|{cd_id}".encode()).hexdigest()[:12]


MAX_ALTERNATIVES = 3  # extra any-of skill Keeps per posting, so no posting dominates


def extra_keep_variants(card: RequirementCard, fitted: CandidateCard, shown: str,
                        rng: random.Random) -> list[tuple[str, CandidateCard]]:
    """More ways to be a Keep, so the real test is not mostly Explore and Skip. Each changes a rule
    input and keeps every requirement and every preference met; none adds or removes a skill, so
    resume length does not tell the label."""
    out: list[tuple[str, CandidateCard]] = []
    if card.work_mode != "remote" and card.locations:
        away = [m for m in METRO_NAMES if away_ok(m, card, shown)]
        if away:
            out.append(("relocate", replace(fitted, current_location=rng.choice(away),
                                            willing_locations=(card.locations[0],))))
    pref = {"hybrid": "hybrid_or_remote", "remote": "remote_only"}.get(card.work_mode)
    if pref:
        out.append(("pref_mode_match", replace(fitted, remote_preference=pref)))
    if card.industry:
        other = rng.choice([i for i in INDUSTRIES if norm(i) != norm(card.industry)])
        out.append(("industry_list", replace(fitted, industries_wanted=(other, card.industry))))
    # any-of groups: holding another listed alternative still meets the group (a few per posting)
    held = {skill_key(s) for s in fitted.skills}
    alts = [(g[0], a) for g in card.must_skills if len(g) > 1 for a in g[1:] if skill_key(a) not in held]
    for first, alt in rng.sample(alts, min(len(alts), MAX_ALTERNATIVES)):
        skills = tuple(alt if s == first else s for s in fitted.skills)
        out.append(("must_skill_alternative", replace(fitted, skills=skills)))
    return out


# Coarse industries (sample.INDUSTRIES) a posting's own industry belongs to or sits next to: a
# candidate who wants one of them could read the posting as wanted, so pref_industry_wanted does not draw it.
RELATED_INDUSTRY = {
    "manufacturing": r"manufactur|semiconductor|aerospace|aviation|defen[cs]e|automotive|industrial|robotic|equipment"
                     r"|electronic|consumer packaged|oil|energy|nuclear|autonomous",
    "government": r"government|defen[cs]e|national security|intelligence|nuclear|air traffic|public|aerospace",
    "public safety": r"security|defen[cs]e|safety|intelligence|air traffic",
    "biotech and pharma": r"pharma|biotech|research|health|life science",
    "research": r"research|r&d|laborator|defen[cs]e research",
    "healthcare": r"health|pharma|medical|fitness",
    "finance": r"financ|bank|trading|invest|insurance|capital",
    "technology": r"tech|software|cloud|data|ai\b|semiconductor|robotic|wireless|gaming|autonomous|electronic",
    "utilities": r"utilit|energy|oil|nuclear|wireless|broadband|waste",
    "logistics": r"logistic|air traffic|aviation|automotive|transport|waste",
    "environmental services": r"environment|waste|energy",
    "consulting": r"consult|professional services",
    "media": r"media|entertainment|gaming",
    "retail": r"retail|consumer",
    "construction": r"construct",
    "agriculture": r"agricult",
    "education": r"educat",
    "legal": r"legal",
}


def industry_related(coarse: str, industry: str) -> bool:
    """The coarse industry is the posting's industry, shares a word with it, or is a known parent of it."""
    a, b = set(norm(coarse).split()) - {"and", "&"}, set(norm(industry).split()) - {"and", "&"}
    pat = RELATED_INDUSTRY.get(norm(coarse))
    return bool(a & b) or bool(pat and re.search(pat, industry, re.I))


def refit(kind: str, cand: CandidateCard, card: RequirementCard, shown: str, rng: random.Random) -> CandidateCard:
    """Re-pick a variant's edit when the stock pick contradicts the shown text: a "far away" city in
    the posting's metro, or a wanted job type the posting also names. When nothing fits, the row
    stays as it is and guard() drops it."""
    if kind in AWAY_KINDS and not away_ok(cand.current_location, card, shown):
        away = [m for m in METRO_NAMES if away_ok(m, card, shown)]
        if away:
            return replace(cand, current_location=rng.choice(away))
    if kind == "pref_industry_wanted" and any(industry_related(w, card.industry) for w in cand.industries_wanted):
        others = [i for i in INDUSTRIES if not industry_related(i, card.industry)]
        if others:  # its own draw, so the posting's other picks (relocation city, ...) do not move
            pick = random.Random(f"{card.source_posting_id}:industry_wanted").choice(others)
            return replace(cand, industries_wanted=(pick,))
    if kind == "pref_industry_avoided" and card.industry and norm(card.industry) in {norm(x) for x in cand.industries_wanted}:
        # the stock variant keeps the full match's wanted industry: the profile would want and avoid it
        return replace(cand, industries_wanted=tuple(x for x in cand.industries_wanted if norm(x) != norm(card.industry)))
    if kind == "employment_type" and any(re.search(TYPE_PHRASE[t], shown, re.I) for t in cand.employment_types_wanted):
        others = [t for t in EMPLOYMENT_TYPES if t != card.employment_type and not re.search(TYPE_PHRASE[t], shown, re.I)]
        if others:
            return replace(cand, employment_types_wanted=(rng.choice(others),))
    return cand


def candidates_for(card: RequirementCard, text: str, seed: int = 0) -> list[dict]:
    """Every (kind, candidate, built-to-be bucket or None) for one posting, deduplicated."""
    raw, fitted, notes = base_candidate(card, text)
    shown = shown_posting(text)
    rng = random.Random(f"{seed}:{card.source_posting_id}:real")
    srng = random.Random(f"{seed}:{card.source_posting_id}:settle")  # its own draws, so the other picks do not move
    out: list[dict] = []
    seen: set[str] = set()

    def add(kind: str, cand: CandidateCard, expected: str | None, note: str = "") -> None:
        cand, problems = settle(cand, card, srng)
        cid = cd_id_of(cand)
        if cid in seen:
            return
        seen.add(cid)
        out.append(dict(kind=kind, candidate=cand, expected=expected, note=note, cd_id=cid, problems=problems))

    # the variants are built on the raw full match (not a sampled copy), so fit() can carry the posting fit
    for v in make_variants(card, seed, base=raw):
        if v.card == card:
            add(v.kind, refit(v.kind, fit(v.candidate, raw, fitted), card, shown, rng), v.expected, v.note)
    for status in WORK_AUTH:
        if status != fitted.work_auth:
            add(f"auth_{status}", replace(fitted, work_auth=status), None)
    for kind, cand in extra_keep_variants(card, fitted, shown, rng):
        add(kind, cand, "For you")
    for o in out:
        o["fit_notes"] = notes
    return out


# ---------------------------------------------------------------- visibility

def readable_misses(cand: CandidateCard, card: RequirementCard, reasons: list[str], resume: str, profile: str,
                    shown: str, hard: bool, side=lambda m: True) -> list[tuple[str, str]]:
    """Misses (restricted to `side`) that decide whether the label can be read from the text.

    Excluded rows are readable when any one hard reason is fully visible; Explore and For you rows
    need every reason visible."""
    def one(rs: list[str]) -> list[tuple[str, str]]:
        miss = [m for m in missing_facts(cand, card, rs, resume, profile, shown) if side(m)]
        if any(m[1] == ENROLL_MISS for m in miss) and ENROLL_EXTRA.search(shown):
            miss = [m for m in miss if m[1] != ENROLL_MISS]
        if any(CITIZEN_REASON in r for r in rs) and not citizen_requirement_shown(shown) and side(("work_auth", CITIZEN_MISS)):
            miss.append(("work_auth", CITIZEN_MISS))
        if any(r.startswith(ONSITE_REASON) for r in rs) and not onsite_mode_shown(shown) and side(("location", ONSITE_MISS)):
            miss.append(("location", ONSITE_MISS))
        return miss

    if not hard:
        return one(reasons)
    per = [one([r]) for r in reasons]
    return [] if any(not m for m in per) else [m for ms in per for m in ms]


def posting_misses(cand: CandidateCard, card: RequirementCard, reasons: list[str], profile: str, shown: str,
                   hard: bool) -> list[tuple[str, str]]:
    """Posting-side misses only: the build runs before any resume is rendered."""
    return readable_misses(cand, card, reasons, "", profile, shown, hard, lambda m: m[1].startswith("posting"))


# ---------------------------------------------------------------- build

def build_rows(cards: dict[str, RequirementCard], postings: dict[str, str], seed: int = 0) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    dropped: list[dict] = []
    for rq_id in sorted(cards):
        card, text = cards[rq_id], postings[rq_id]
        shown = shown_posting(text)
        for c in candidates_for(card, text, seed):
            cand = c["candidate"]
            bucket, reasons, flags = label(cand, card)
            row = dict(pair_id=pair_id_of(rq_id, c["cd_id"]), rq_id=rq_id, base_rq_id=rq_id, cd_id=c["cd_id"],
                       bucket=bucket, variant_kind=c["kind"], reasons=reasons, flags=flags,
                       completion=format_completion(bucket, reasons), candidate=cand.to_dict(),
                       fit_notes=c["fit_notes"], note=c["note"], built_as=c["expected"])
            if c["problems"]:
                dropped.append(row | {"drop": "candidate_inconsistent", "detail": c["problems"]})
                continue
            if c["expected"] is not None and bucket != c["expected"]:
                dropped.append(row | {"drop": "label_disagrees_with_construction",
                                      "detail": f"built as {c['expected']}, label() says {bucket}"})
                continue
            why = guard(c["kind"], cand, card, bucket, shown)
            if why:
                dropped.append(row | {"drop": "text_contradicts_label", "detail": why})
                continue
            profile = render_profile(cand)
            miss = posting_misses(cand, card, reasons, profile, shown, hard=bucket == "Excluded")
            if miss:
                # was it readable before build_prompt fitted the posting to its budget?
                late = not posting_misses(cand, card, reasons, profile, text, hard=bucket == "Excluded")
                dropped.append(row | {"drop": "past_truncation" if late else "posting_does_not_state",
                                      "detail": [list(m) for m in miss][:4],
                                      "categories": sorted({m[0] for m in miss})})
                continue
            kept.append(row | {"missing": []})
    return kept, dropped


def report_of(kept: list[dict], dropped: list[dict], n_postings: int,
              cards: dict[str, RequirementCard] | None = None) -> dict:
    kinds: dict[str, Counter] = defaultdict(Counter)
    for r in kept:
        kinds[r["variant_kind"]][r["bucket"]] += 1
    n = len(kept)
    return {
        "n_rows": n,
        "n_postings_with_rows": len({r["base_rq_id"] for r in kept}),
        "n_postings": n_postings,
        "posting_employment_types": dict(Counter(c.employment_type for c in cards.values())) if cards else None,
        "bucket_counts": {b: sum(r["bucket"] == b for r in kept) for b in BUCKETS},
        "bucket_share": {b: round(sum(r["bucket"] == b for r in kept) / max(n, 1), 3) for b in BUCKETS},
        "per_kind": {k: dict(v) | {"n": sum(v.values())} for k, v in sorted(kinds.items())},
        "core_rows": sum(r["variant_kind"] in CORE_KINDS for r in kept),
        "dropped": dict(Counter(r["drop"] for r in dropped)),
        "dropped_by_category": dict(Counter(c for r in dropped for c in r.get("categories", []))),
        "dropped_by_kind": dict(Counter(r["variant_kind"] for r in dropped)),
        "dropped_by_guard": dict(Counter(r["detail"] for r in dropped if r["drop"] == "text_contradicts_label")),
        "unique_resume_cards": len({r["cd_id"] for r in kept}),
        "for_you_share_without_added_keep_kinds": round(
            sum(r["bucket"] == "For you" and r["variant_kind"] not in ADDED_KEEP_KINDS for r in kept)
            / max(sum(r["variant_kind"] not in ADDED_KEEP_KINDS for r in kept), 1), 3),
        "resume_kind_rows": sum(r["variant_kind"] in RESUME_KINDS for r in kept),
        "employment_type_by_wanted": dict(Counter(",".join(r["candidate"]["employment_types_wanted"])
                                                  for r in kept if r["variant_kind"] == "employment_type")),
        "keep_rows": {"core": sum(r["bucket"] == "For you" and r["variant_kind"] in CORE_KEEP_KINDS for r in kept),
                      "added": sum(r["bucket"] == "For you" and r["variant_kind"] in ADDED_KEEP_KINDS for r in kept),
                      "note": "the For you share is reached by construction; the added (profile-only) kinds are "
                              + ", ".join(ADDED_KEEP_KINDS)},
        "coverage_gaps": {"no_rows": sorted(k for k in KNOWN_KINDS if k not in kinds and not k.startswith("auth_citizen")),
                          "thin": {k: sum(v.values()) for k, v in sorted(kinds.items()) if sum(v.values()) < THIN},
                          "not_in_the_card": {
                              "gpa_minimum": "no GPA field; Keep and Explore rows on postings that require a "
                                             "minimum GPA are dropped",
                              "experience_months": "internship cards hold 0 years; '3+ months of experience' lines "
                                                   "(which count projects and labs) are taken as met",
                              "graduation_month": "the card holds a year; Keep and Explore rows on postings whose "
                                                  "graduation window no whole year fits are dropped"}},
        "assumptions": ["a location Skip on a posting whose text names no work mode (only 'Locations: City, ST') "
                        "takes the card's on-site mode as stated (about a third of the location Skips); on a hybrid "
                        "card such a Skip is dropped",
                        "degree_level has one row (doctoral needed, master held); associate-candidate rows on master or PhD "
                        "postings, whose 'needs a bachelor degree' clause the text never states, are dropped; quote no "
                        "degree-level accuracy",
                        "a job-type Skip on an internship whose text states no hours follows the rule that an "
                        "internship is not a part-time or full-time job, not an hours statement",
                        "degree_field has no rows: its one row (a theology PhD with ML skills on 'or a related "
                        "discipline') is dropped",
                        "must_skill Skips are few (small n; quote no must_skill accuracy alone): most stock missing skills are generic, example-only, offered as "
                        "one of several, or implied by a held skill, and are dropped",
                        "the real postings are nearly all internships (posting_employment_types), so experience-years "
                        "rules have no rows; most rows are decided from the profile and the posting header "
                        "(resume_kind_rows counts the rest)",
                        "Keep and Explore rows for the 'citizen or permanent resident' profile are dropped on postings "
                        "that ask for U.S. citizenship (or a clearance) and name no permanent resident",
                        "a must_skill Skip stands when the required line softens its list with 'or related subjects' "
                        "or gives examples with 'e.g.' and the candidate holds none of the named items (Micron 5a07cf07 "
                        "'Digital Logic Design, CMOS Circuits, ... or related subjects'; HPE d6a60616 'programming or "
                        "scripting knowledge (e.g., Python, Java, C++)'): the card cannot judge which held skills a "
                        "reader would count as related",
                        "a remote-only or hybrid-or-remote Explore says 'the role is onsite' from the card's on-site "
                        "mode; where the shown text calls only a perk, event or visit onsite (Rundoo 21ae5c99 "
                        "'in-office lunch', Axos a1902097 'Onsite Gym') the row is not decidable "
                        "(onsite_mode_shown), and a build drops it"],
    }


def build(a: argparse.Namespace) -> None:
    cards = load_test_cards()
    postings = load_postings()
    missing = [i for i in cards if i not in postings]
    if missing:
        sys.exit(f"no posting text for {missing[:5]}")
    kept, dropped = build_rows(cards, postings, a.seed)
    bad = [(r["pair_id"], p) for r in kept if (p := person_problems(CandidateCard.from_dict(r["candidate"])))]
    print(f"kept candidates who could not exist (work authorization against education, license, years, "
          f"preferences, a skill twice, an industry wanted and avoided): {len(bad)}")
    if bad:
        sys.exit(f"build check failed, nothing written: {len(bad)} rows, e.g. {bad[:3]}")
    rep = report_of(kept, dropped, len(cards), cards)
    resume_cards = {r["cd_id"]: {"id": r["cd_id"], "card": r["candidate"]} for r in kept}
    write_jsonl(OUT / "rows.jsonl", kept)
    write_jsonl(OUT / "dropped.jsonl", dropped)
    write_jsonl(OUT / "resume_cards.jsonl", list(resume_cards.values()))
    (OUT / "resume_ids.txt").write_text("\n".join(resume_cards) + "\n")
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    print_report(rep)
    print(f"wrote {OUT.relative_to(ROOT)}/{{rows,dropped,resume_cards}}.jsonl, resume_ids.txt, report.json")
    print_resume_estimate(list(resume_cards.values()))


def print_report(rep: dict) -> None:
    print(f"Keep rows: {rep['keep_rows']['core']} core, {rep['keep_rows']['added']} added ({rep['keep_rows']['note']}); "
          f"For you share without the added kinds {rep['for_you_share_without_added_keep_kinds']:.1%}")
    print(f"resume-dependent rows {rep['resume_kind_rows']}; employment_type Skips by wanted type "
          f"{rep['employment_type_by_wanted']}; postings by type {rep['posting_employment_types']}")
    print("coverage gaps: no rows for", rep["coverage_gaps"]["no_rows"], "| thin:", rep["coverage_gaps"]["thin"])
    print(f"{rep['n_rows']} rows over {rep['n_postings_with_rows']} of {rep['n_postings']} postings, before the resume "
          f"render (assemble counts the postings left with rows); "
          f"buckets {rep['bucket_counts']} shares {rep['bucket_share']}")
    print("rows by kind (For you / Explore / Excluded):")
    for k, v in rep["per_kind"].items():
        print(f"  {k:28s} {v.get('For you', 0):4d} {v.get('Explore', 0):4d} {v.get('Excluded', 0):4d}")
    print("dropped:", rep["dropped"], "| by category:", rep["dropped_by_category"])
    print("dropped by guard:", rep["dropped_by_guard"])


def print_resume_estimate(resume_cards: list[dict]) -> None:
    """Unique resume contexts (what Adaption would render) and the credit estimate at 0.01 a row."""
    try:
        from render_cards import ctx_hash, ctx_str, texts_by_ctx
    except Exception as e:  # render_cards needs the adaption package
        print("resume context count skipped:", e)
        return
    ctxs = {ctx_hash(ctx_str("resumes", r["card"])) for r in resume_cards}
    have = ctxs & set(texts_by_ctx("resumes"))
    todo = len(ctxs) - len(have)
    print(f"resumes: {len(resume_cards)} candidate cards -> {len(ctxs)} unique contexts, {len(have)} already rendered, "
          f"{todo} to render (~{max(todo * 0.01, 1):.1f} credits at 0.01 a row, 1-credit floor). Not run.")
    print("To render (render_cards.load_cards reads resume_cards.jsonl): "
          "render_cards.py resumes estimate data/built/real_test/resume_ids.txt")


# ---------------------------------------------------------------- assemble

def assemble_rows(rows: list[dict], cards: dict[str, RequirementCard], postings: dict[str, str],
                  resume_of) -> tuple[list[dict], list[str], Counter]:
    """(test rows, pair ids with no rendered resume, rows not decidable per miss category).

    Each row is checked again with the rendered resume in hand (visibility.missing_facts on all
    three texts): graduation year shown and marked expected, degree shown, held skills named, a
    swapped-out skill not named, and on a posting with a month-bounded graduation window, a graduation
    month the resume names inside it (resume_month_ok). A row whose label the full prompt does not show is kept with
    decidable False and its misses listed, so the evaluator reports it apart."""
    out, missing, undecidable = [], [], Counter()
    keep = ("pair_id", "rq_id", "base_rq_id", "cd_id", "bucket", "variant_kind", "completion", "flags")
    for r in rows:
        resume = resume_of(r)
        if not resume:
            missing.append(r["pair_id"])
            continue
        cand, card = CandidateCard.from_dict(r["candidate"]), cards[r["rq_id"]]
        prompt = build_prompt(resume, render_profile(cand), postings[r["rq_id"]])
        resume, profile, shown = prompt_parts(prompt)  # as the prompt holds them, like assemble_training
        miss = readable_misses(cand, card, r["reasons"], resume, profile, shown, hard=r["bucket"] == "Excluded")
        # a Keep or Explore also says the graduation date fits; a month the card cannot hold may not
        why = resume_month_ok(resume, shown, cand.graduation_year) if r["bucket"] != "Excluded" else None
        if why:
            miss.append(("enrollment", why))
        undecidable.update({m[0] for m in miss})
        out.append({k: r[k] for k in keep} | {"decidable": not miss, "missing": [list(m) for m in miss],
                                              "prompt": prompt})
    return out, missing, undecidable


def assemble(a: argparse.Namespace, resume_texts: dict[str, str] | None = None) -> list[dict]:
    from render_cards import ctx_hash, ctx_str, texts_by_ctx
    rows = read_jsonl(OUT / "rows.jsonl")
    if resume_texts is not None:
        resume_of = lambda r: resume_texts.get(r["cd_id"])  # noqa: E731
    else:
        store = texts_by_ctx("resumes")
        resume_of = lambda r: store.get(ctx_hash(ctx_str("resumes", r["candidate"])))  # noqa: E731
    out, missing, undecidable = assemble_rows(rows, load_test_cards(), load_postings(), resume_of)
    write_jsonl(OUT / "test.jsonl", out)
    n_bad = sum(not r["decidable"] for r in out)
    n_post = len({r["base_rq_id"] for r in out})
    n_dec = len({r["base_rq_id"] for r in out if r["decidable"]})
    print(f"test.jsonl: {len(out)} rows with a prompt over {n_post} postings ({n_dec} with a decidable row; build's "
          f"count is before the resume render), {len(missing)} left out for want of a rendered resume; "
          f"{n_bad} not decidable from the rendered text, by category: {dict(undecidable)}")
    return out


# ---------------------------------------------------------------- score

def score(a: argparse.Namespace) -> dict:
    """Score an evaluator --out file against rows.jsonl, decidable rows of test.jsonl only. The bootstrap resamples
    the postings that have rows (86: one of the 87 cards lost every resume to the render gates), not pairs."""
    preds = {r["pair_id"]: r["pred"] for r in json.loads(Path(a.predictions).read_text())["rows"]}
    rows = [r for r in read_jsonl(OUT / "rows.jsonl") if r["pair_id"] in preds]
    test = OUT / "test.jsonl"
    undecidable = {r["pair_id"] for r in read_jsonl(test) if not r.get("decidable", True)} if test.exists() else set()
    rows = [r for r in rows if r["pair_id"] not in undecidable]
    res = score_rows(rows, preds, a.draws)
    print(f"{res['n']} rows scored of {len(preds)} predictions ({len(undecidable)} not decidable from the rendered "
          f"text, left out)")
    print(f"headline macro-F1 {res['headline']['macro_f1']:.3f} "
          f"[{res['headline']['ci'][0]:.3f}, {res['headline']['ci'][1]:.3f}] over {res['headline']['postings']} postings; "
          f"accuracy {res['headline']['accuracy']:.3f}")
    c = res["core"]
    print(f"core kinds (text the posting states outright; almost all Skip, so accuracy only): "
          f"accuracy {c['accuracy']:.3f} on {c['n']} rows")
    for name, v in res["keep_recall"].items():
        print(f"Keep recall, {name} Keep kinds: {v['recall']:.3f} on {v['n']} rows")
    for name, d in res["side_headlines"].items():
        if d:
            print(f"macro-F1, {name.replace('_', ' ')}: {d['macro_f1']['value']:.3f} "
                  f"[{d['macro_f1']['ci95'][0]:.3f}, {d['macro_f1']['ci95'][1]:.3f}], accuracy {d['accuracy']:.3f} "
                  f"on {d['n']} rows")
    for k, v in res["per_kind"].items():
        print(f"  {k:28s} n={v['n']:4d} accuracy {v['accuracy']:.3f}")
    return res


def score_rows(rows: list[dict], preds: dict[str, str], draws: int = 2000, seed: int = 0) -> dict:
    """Headline numbers; `preds` maps pair_id -> Keep/Explore/Skip."""
    from matcher.evalstats import WORD, cluster_bootstrap, macro_f1_of, side_headlines

    def stats(sub: list[dict]) -> dict:
        gold = [WORD[r["bucket"]] for r in sub]
        pred = [preds[r["pair_id"]] for r in sub]
        groups = [r["base_rq_id"] for r in sub]
        point, lo, hi = cluster_bootstrap(gold, pred, groups, macro_f1_of, draws, seed)
        return {"n": len(sub), "postings": len(set(groups)), "macro_f1": point, "ci": [lo, hi],
                "accuracy": sum(g == p for g, p in zip(gold, pred)) / max(len(sub), 1)}

    per_kind: dict[str, dict] = {}
    for k in sorted({r["variant_kind"] for r in rows}):
        sub = [r for r in rows if r["variant_kind"] == k]
        per_kind[k] = {"n": len(sub), "accuracy": sum(WORD[r["bucket"]] == preds[r["pair_id"]] for r in sub) / len(sub)}
    core = [r for r in rows if r["variant_kind"] in CORE_KINDS]
    core_acc = sum(WORD[r["bucket"]] == preds[r["pair_id"]] for r in core) / max(len(core), 1)
    recall = {}
    for name, ks in (("core", CORE_KEEP_KINDS), ("added", ADDED_KEEP_KINDS)):
        sub = [r for r in rows if r["bucket"] == "For you" and r["variant_kind"] in ks]
        recall[name] = {"n": len(sub), "recall": sum(preds[r["pair_id"]] == "Keep" for r in sub) / max(len(sub), 1)}
    side = side_headlines([{"gold": WORD[r["bucket"]], "pred": preds[r["pair_id"]], "variant_kind": r["variant_kind"],
                            "base_rq_id": r["base_rq_id"]} for r in rows], draws)
    return {"n": len(rows), "headline": stats(rows), "core": {"n": len(core), "accuracy": core_acc},
            "keep_recall": recall, "per_kind": per_kind, "side_headlines": side}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=0)
    sub.add_parser("assemble")
    s = sub.add_parser("score")
    s.add_argument("predictions")
    s.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    {"build": build, "assemble": assemble, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
