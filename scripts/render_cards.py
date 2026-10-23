"""Render cards into text with Adaption adapt (universal-prompt mode + blueprint).

  resumes:  CandidateCard JSON  -> one-page resume      -> data/built/rendered/resumes.jsonl
  postings: RequirementCard JSON -> realistic job posting -> data/built/rendered/postings.jsonl

Usage:
  render_cards.py resumes|postings estimate|run|report|pick|signoff IDS_FILE [--max-credits 30] [--redo] [--out-dir DIR]
  (report: gate numbers for the ids that already have text, or with --rows FILE for the rows of FILE
   by id; pick: write a stratified check batch of --n ids to IDS_FILE; signoff: check the rendered check
   batch (CHECK_IDS) and, with --read-by NAME, write its sign-off; none of these spends anything)

The check batch comes first: `run` on any id outside CHECK_IDS (check_resume_ids.txt, check_posting_ids.txt) is refused
until `signoff` passes for the current blueprint and check contexts. A blueprint or context change makes the old
sign-off stale.

Sign-off rules (signoff_problems; each kind on its own):
  1. Every check context has a kept text that passes every fatal gate under the current gate code, kept within 3 tries
     (MAX_REDO + 1, retries included) and rendered by the current writer (a text kept from before a blueprint change
     is no try of this blueprint: --redo).
  2. Every stratum (POSTING_STRATA / RESUME_STRATA) has a kept check text that passes the current gates.
  3. Every kept check text re-gates clean under the current gate code.
  4. Every stock phrase (ECHO) is in at most 15% of the kept check texts.
  5. At least 55% of first tries (tries sent without retry feedback, by the current writer) pass every fatal gate,
     re-gated under the current gate code. This is a cost floor, not a correctness bar: at 55% the full render costs
     about 1.54 tries a context, about 190 credits. Correctness is the gates plus a person's read of every kept text.
  6. Rows that no record ties to a writer (no stamp, no dataset, no batch with one writer) are counted in the numbers;
     any of them that fails the current gates, or sits on a check context with no counted try, blocks the sign-off.
  7. A person, named by --read-by, has read SIGNOFF_READ[kind]; before that, an independent reader has read every
     kept check text next to its card (the sign-off records where that read is).

Rendered rows are {"id", "ctx", "text"}: ctx is the hash of the exact context sent to adapt, and
a card's text is the row whose ctx matches the card's current context (`rendered_texts`). So a
card that changed has no text until it is rendered again, and cards with identical contexts share
one text (Adaption drops exact-duplicate rows anyway, so each batch uploads one row per unique
context). A blueprint change does not invalidate texts: pass --redo for that (it moves the ids'
current rows to rendered/replaced_<kind>.jsonl and their failed tries to rendered/replaced_quarantine_<kind>.jsonl,
in estimate mode too, so the next run renders them again and counts only the new blueprint's tries).
IDS_FILE has one card id per line (cd_*, cd_v_*, rq_*, rq_*~kind). `estimate` prints the cost and
stops; `run` estimates, refuses when the estimate is over --max-credits, then runs, waits until
every row is processed, downloads and appends rows. Each batch keeps a state file in data/adaption/
so a rerun resumes (no second upload or second adapt). Adapt bills at least 1 credit per run.

The QA gates are enforced, not just reported. At download every text is gated (`fatal`: every
resume_problems / posting_problems tag and a stance conflict, except leftover markdown); a text that
fails goes to rendered/quarantine_<kind>.jsonl, never into <kind>.jsonl, so the next `run` of the same
ids renders it again. That retry is a counted try, and its context is the card's context with one key appended,
fix_from_last_try (FEEDBACK_KEY): one line per fatal tag of the last failed try, re-gated under the current gates,
quoting the offending sentence (or naming the missing fact) and saying what the card says instead, e.g.
  Your last version wrote: "Applicants must not require sponsorship now or in the future". The card says:
  work_authorization says only: '<the stance wording>' (no time scope, no other status, nothing added).
The card's facts are unchanged and the text is stored under the card's own context (base_ctx). A context that
failed MAX_REDO + 1 times is not rendered again and has no text.
`rendered_texts` (what assemble_training reads) gates again and leaves out any text that fails, writing
the ids to rendered/dropped_<kind>.jsonl, so a text rendered before a gate existed cannot reach training.
A plain `run` (and `estimate`) first re-gates the kept texts of the ids asked: one that fails a gate added after it was
rendered moves to the quarantine file with its new tags (stale_to_quarantine), so it is a failed try of its context,
counted toward the 3 tries, and the context is rendered again with its findings; a context with no try left stays
without text and is listed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import io
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adaption import Adaption  # noqa: E402

from matcher import textclean  # noqa: E402
from matcher.pairs import build_prompt  # noqa: E402
from matcher.schema import CURRENT_YEAR  # noqa: E402
from matcher.visibility import (_MODE, _NUM_WORDS, _TYPE, ENROLLED, ENROLLMENT_IN, NOT_E_VERIFY, has_industry,  # noqa: E402
                                has_money, has_number, has_phrase, has_years, stance_conflict)

CARDS = ROOT / "data/built/cards"
PAIRS = ROOT / "data/built/pairs"
REAL_TEST = ROOT / "data/built/real_test"
OUT = ROOT / "data/built/rendered"
STATE = ROOT / "data/adaption"

# Today, for the writer. The label rule's year (CURRENT_YEAR) is the same calendar: a graduation year up to
# and including it is a finished degree. Without this sentence the writer thought it was 2024 and wrote
# finished degrees as "Expected 2026".
TODAY = f"Today is October {CURRENT_YEAR}."

# Resume rules are render_test.py's, plus consistency rules the label depends on
# (every skill and license verbatim, years stated, enrolled students shown as expected).
RESUME_UNIVERSAL = (
    "Write a realistic one-page resume as plain text for the job seeker described by the JSON card in the context. "
    "Use only the facts in the card. Invent plausible but generic employer names and bullet points consistent with those facts."
)
RESUME_BLUEPRINT = (
    f"{TODAY} Write plain text only: no markdown, no asterisks, no pound signs, no bold; use '-' for bullets. "
    "Header: if the card has a 'contact' entry, start with it exactly as given (name, then the contact details, which are fake); "
    "when it says \"layout\": \"one line\", the first line is the name, current_location, phone and email (and the "
    "LinkedIn or address if given) separated by ' | ', and the next line starts the resume; "
    "otherwise the header is the single word 'Candidate'. Without a 'contact' entry never include a name, email, phone number, "
    "street address, URL, photo, age, gender, nationality, or religion. Do not mention work authorization, visa status, salary "
    "expectations, preferred locations, remote preference, industries wanted or avoided, or roles avoided: those are private "
    "preferences, not resume content. Mention the country of education only as the school's location. Write in English. "
    "Vary layout, section order, headings and tone across resumes; omit any section the card gives nothing for, and never write "
    "'None', 'None listed' or 'N/A'.\n"
    "Consistency rules:\n"
    "- skills and licenses: list every skill and every license from the card, using the card's exact wording. Do not use the "
    "names of other skills anywhere, not even in bullet points (a skill that is not listed is a skill this person does not have).\n"
    "- jobs: the paid jobs, most recent first, each with its title and dates. Write exactly one job per entry with that title "
    "and exactly those dates (month and year), the latest ending 'Present' (it is October " + str(CURRENT_YEAR) + "); that "
    "job's employer is in employer_city. Use the given titles (a specialty such as 'Pediatric' may go before a title); never "
    "write a licensed job title (Registered Nurse, Physician, Pharmacist, CPA, Teacher, Attorney and the like) that is not a "
    "given title. Under a work or experience heading list no other "
    "dated job, internship, assistantship, practicum or rotation; unpaid projects, coursework and volunteering go under their "
    "own headings. Without jobs list no jobs at all, under any heading: no assistantship, fellowship, internship or other "
    "role title with dates, not even under research or projects (a graduate student lists research projects, coursework and "
    "volunteer work, with no job title); never write '0 years'.\n"
    "- summary_skills: the skills the summary names, word for word; the summary names no other skill or skill area (no "
    "'data analysis', 'data management' or 'research skills' unless skills lists it).\n"
    "- years_of_experience: say the number once, in the summary, in your own words (for example 'five years in hospital "
    "pharmacy' or 'over 3 years as a lab technician'); it is a whole number of years that the dates may exceed by some "
    "months, so never round it up ('nearly 4 years' for 3 is wrong); never write 'years of paid experience' or 'paid "
    "professional experience'.\n"
    "- List no license, certification or certificate that is not in licenses (no CPR, first aid, OSHA or safety "
    "certificates); without licenses write no license or certification section.\n"
    "- Name no software, tool, programming language or product that is not in skills (no 'Microsoft Office suite' when only "
    "Microsoft Word is listed, no Excel when Excel is not listed).\n"
    "- Paraphrase the context; never copy its sentences into the resume.\n"
    "- fix_from_last_try (only on a second or third try): what your last version of this resume got wrong, quoted, and what "
    "the card says instead. Write the whole resume again from the card, fix every point, never write those sentences again, "
    "and never mention this key, the card or a last version.\n"
    "- education: follow the card's education_status sentence exactly. A degree it calls finished is a past graduation date "
    "(for example 'May 2025'); never write expected, anticipated, in progress or currently enrolled next to it. A degree it calls "
    "in progress is written 'Expected <month> <year>'. If degree_field is given, write it with the degree (for example "
    "'Bachelor of Science in Nursing'). If no degree_field is given, write the degree by level only (for example "
    "'Bachelor of Science' or 'Associate degree') with no 'in <subject>' after it, no major, no concentration and no BSN, MSN "
    "or MBA. Never use the card's field label as a major or degree name. "
    "Name the school like a real college or university with its city and state; never write 'Local Community College', "
    "'Springfield University' or a placeholder, and never write 'University Location: United States'.\n"
    "- The line right after the name (or after 'Candidate') is current_location exactly as written in the card, without "
    "adding a state (on a one-line contact header it is the second item of that line)."
)

POSTING_UNIVERSAL = (
    "Write one realistic job posting as plain text for the job described by the JSON card in the context. "
    "Every requirement in the card must appear in the posting in natural language, and the posting must not add any hard requirement "
    "that is not in the card."
)
POSTING_BLUEPRINT = (
    f"{TODAY} Write plain text only, like text pasted from an employer's careers page: no markdown, no asterisks, no pound signs, "
    "no bold, no tables; use '-' for list items. Line 1 is the job title alone. Line 2 is the company name from the context, alone. "
    "Then a blank line and the posting. "
    "Vary the structure and the wording of headings from posting to posting (a short intro, 'What you will do', 'About you', "
    "'Minimum qualifications', 'Nice to have'), and skip any heading you do not need. Every posting has one requirements part "
    "that names every required skill: a list under a heading, or, in a posting of plain paragraphs, one paragraph that says "
    "what applicants must have. "
    "Vary tone (corporate, startup, hospital, school district, government agency, small business). "
    "Write 450 to 900 words in English. Vary the opening and the closing too: no stock lines such as 'We are currently "
    "seeking', 'In this role, you will', 'To be considered for this' or 'we encourage you to apply'.\n"
    "The context lists the facts of this job. Put every listed fact in the posting in natural language. A fact that is not listed "
    "is absent: say nothing about its topic. Never write what is NOT required or NOT stated: no 'no degree required', "
    "'no experience necessary', 'no license required', 'regardless of citizenship', 'salary is commensurate / competitive / "
    "discussed at interview', and no 'not stated', 'not specified', 'available on request' or 'decided case by case'.\n"
    "How to read the context:\n"
    "- title: use it as the title line (a plural title may be made singular).\n"
    "- company: line 2, exactly as given; use this name wherever the posting names the company.\n"
    "- duties_opener: begin the part about the job's duties with these words, then go on in your own words.\n"
    "- industry: copy it into the company description letter for letter, keeping any '&' and any final 's' (for example "
    "'a Glass & Ceramics company', 'a company in the Textiles industry', 'a healthcare company').\n"
    "- level, employment_type (internship, full-time, part-time, contract): say the job type plainly.\n"
    "- program_term: the term of the internship; write it as given. Write no other year or season.\n"
    "- minimum_degree: the lowest degree the applicant must hold (a higher one also counts).\n"
    "- completed_degree (an internship): the lowest degree the applicant must hold (a higher one also counts), asked "
    "for as any job asks for a degree: the degree itself, not a program or studies.\n"
    "- who_can_apply (an internship): who may apply. This internship has no school requirement: say nothing about "
    "students, enrollment or graduation.\n"
    "- must_be_student_in: only students currently enrolled in this kind of program qualify. Without it, say nothing at all "
    "about students, enrollment or graduation.\n"
    "- fields_of_study: the degree (or, only with must_be_student_in, the program the student is in) must be in one of "
    "these fields; name every one. Without it or degree_subjects, name no subject or major for the degree (a required "
    "skill such as "
    "'Glaciology' is a skill, not a degree subject).\n"
    "- degree_subjects (an internship with completed_degree): the finished degree must be in one of these subjects; "
    "name every one.\n"
    "- degree_field: the degree (or, with must_be_student_in, the student's program) may be in any field; say so in "
    "your own words or leave it out, and name no subject or major.\n"
    "- Mention a degree, a field of study, students or graduates only when minimum_degree, completed_degree, "
    "must_be_student_in, fields_of_study or degree_subjects is given.\n"
    "- minimum_years_experience: the years of work the applicant must have and what counts; state the number and that "
    "any kind of work counts, and give the experience no field, setting or domain. Absent means say nothing about "
    "experience.\n"
    "- required_skills: every entry is a separate hard requirement. An entry {\"any_one_of\": [...]} means one of them is "
    "enough: write them as alternatives in one sentence ('experience with X or Y') and never say or suggest that all of them "
    "are needed. Put every required skill in one requirements list or paragraph (under 'Requirements' or 'Minimum "
    "qualifications', or in sentences with 'must' or 'required'), each name exactly as given, even a name that reads oddly "
    "such as 'Codes and standards'. You may mention them again among the duties, but a required skill that appears only "
    "among the duties or only under a preferred heading is wrong.\n"
    "- preferred_skills are preferred, not required: put them under a 'preferred' or 'nice to have' line, never under requirements.\n"
    "- required_license: a credential the applicant must already hold. Absent means say nothing about licenses or certifications.\n"
    "- locations, work_mode (onsite, hybrid, remote): name the work location(s) and the arrangement.\n"
    "- pay: state it exactly as given, with every figure given (hourly and yearly). Absent means say nothing about pay.\n"
    "- work_authorization, when present, lists the posting's work-authorization statements: state each one in your own words, "
    "keep its exact scope, and add nothing to it (a statement that does not mention OPT means the posting says nothing about OPT). "
    "Never add a scope word to it: no 'unrestricted', 'full', 'permanent', 'without restriction', 'any status' or "
    "'regardless of immigration status'. "
    "When it is absent, write nothing about sponsorship, visas, OPT, citizenship, E-Verify or work authorization.\n"
    "- fix_from_last_try (only on a second or third try): what your last version of this posting got wrong, quoted, and what "
    "the card says instead. Write the whole posting again from the card, fix every point, never write those sentences again, "
    "and never mention this key, the card or a last version.\n"
    "The context gives bare facts and short statements: word them your own way, as real postings do, and never copy its key "
    "names or its sentences.\n"
    "Never add requirements that are not in the context: no extra degrees, years, licenses, certifications, clearances, citizenship, "
    "work-authorization statements, background checks, travel, shifts, or required skills. Benefits, team description and an "
    "equal-opportunity line are fine."
)

RESUME_FIELDS = ("field", "degree_field", "skills", "licenses", "current_location", "education_country", "language")
POSTING_DROP = ("source_posting_id", "onet_soc", "related_ok")  # related_ok: card-side flag, not posting text
LEVEL_TEXT = {"none": "no college degree", "associate": "associate degree", "bachelor": "bachelor's degree",
              "master": "master's degree", "doctoral": "doctoral degree"}

KINDS = {
    "resumes": dict(universal=RESUME_UNIVERSAL, blueprint=RESUME_BLUEPRINT, length="detailed"),
    "postings": dict(universal=POSTING_UNIVERSAL, blueprint=POSTING_BLUEPRINT, length="detailed"),
}

csv.field_size_limit(10**8)


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_cards(kind: str) -> dict[str, dict]:
    if kind == "resumes":
        # real test candidates (build_real_test.py build) and real-posting train/dev candidates (build_real_train.py
        # build) render through the same check batch and gates
        files = [CARDS / "candidate_cards.jsonl", PAIRS / "variant_candidates.jsonl", REAL_TEST / "resume_cards.jsonl",
                 ROOT / "data/built/real_train/resume_cards.jsonl"]
    else:
        # v2 real cards come last, so they win over the v1 card of the same posting id (the pairs use v2)
        files = [CARDS / "requirement_cards.jsonl", PAIRS / "variant_requirements.jsonl", CARDS / "extracted_real_cards.jsonl",
                 CARDS / "extracted_real_cards_v2.jsonl"]
    out: dict[str, dict] = {}
    for p in files:
        for row in read_jsonl(p):
            out[row["id"]] = row["card"]
    return out


# The stance goes in as plain sentences, not enum names: with the enums in the context the
# writer added "cannot support OPT" to 27 of ~160 will_not_sponsor postings (v1 blueprint). Each
# stance has several wordings of the same scope (picked by card hash), because the writer copies the
# one it is given: "now or in the future" was in 29% of the pilot, and with three wordings a stance
# 'Visa sponsorship is not available for this position.' was in 21% of the check batch (limit 15%).
# Each wording reads as its own stance, and only that one, to visibility's and the extraction's
# patterns (test_work_auth_wordings_read_as_their_own_stance); "offer" never sits next to "sponsorship"
# in a refusal (SPONSOR_OFFERED does not see the negation). No wording says "now or in the future": it is a
# stock phrase of its own (ECHO), and the writer copied it from 3 of 3 contexts that had it.
WORK_AUTH_TEXT = {
    "sponsors": ("The employer offers visa sponsorship (for example H-1B).",
                 "Visa sponsorship is available for this role, for example H-1B.",
                 "The company will sponsor an employment visa, such as H-1B, for the person hired.",
                 "The employer can sponsor work visas (for example H-1B) for this job.",
                 "H-1B and other employment visa sponsorship is available for this position.",
                 "The company offers sponsorship for employment visas such as H-1B."),
    # each wording says the scope the rule means, so the writer has nothing to add: no sponsorship to take this job,
    # nothing about OPT, CPT, students or the future (6 of 39 r4 check postings added 'now or in the future' to their own
    # closing sentence 'Applicants must already possess the legal right to work ... without the need for sponsorship')
    "will_not_sponsor": ("The employer does not sponsor employment visas such as H-1B for this role, so applicants must be "
                         "able to take the job without visa sponsorship.",
                         "The company cannot sponsor a work visa for this position; the person hired must be able to start "
                         "work without one.",
                         "The employer will not sponsor H-1B or other employment visas for this position, so it is open to "
                         "applicants who need no sponsorship to take it.",
                         "The employer is not able to sponsor employment visas, so the person hired must not need visa "
                         "sponsorship to take this job.",
                         "The company won't sponsor work visas such as H-1B; applicants must be able to do this job without "
                         "visa sponsorship.",
                         "The employer does not sponsor visas for this job: applicants need to be able to hold it without "
                         "employer sponsorship."),
    # no sponsorship and no OPT / STEM OPT, and nothing wider: the I-983 training plans of the old wordings were
    # generalised to 'student work authorization support', which also refuses CPT (rq_00693~no_opt)
    "will_not_sponsor_or_opt": ("The employer does not sponsor employment visas such as H-1B and does not hire people "
                                "working on F-1 OPT or STEM OPT, so applicants must be able to take the job without either.",
                                "The company cannot sponsor work visas for this role and cannot employ anyone on OPT or STEM "
                                "OPT; the person hired must be able to start work without them.",
                                "The employer will not sponsor H-1B or other employment visas and does not accept OPT or STEM "
                                "OPT work authorization, so applicants must be able to do this job without visa sponsorship "
                                "or OPT.",
                                "The employer is unable to sponsor employment visas and cannot take on hires who would work on "
                                "F-1 OPT or STEM OPT; applicants need to be able to hold the job without either.",
                                "The company won't sponsor work visas such as H-1B, and this job cannot be done on OPT or STEM "
                                "OPT: applicants must be able to work in it without sponsorship or OPT."),
}
CITIZENS_TEXT = ("Applicants must be U.S. citizens or permanent residents.",
                 "This position is open only to U.S. citizens and lawful permanent residents (green card holders).",
                 "Only U.S. citizens or permanent residents may be considered.",
                 "You must be a U.S. citizen or a lawful permanent resident to be considered.",
                 "The position is limited to U.S. citizens and permanent residents.")
E_VERIFY_TEXT = ("The company participates in E-Verify.",
                 "The employer uses E-Verify to confirm the employment eligibility of new hires.",
                 "This employer is an E-Verify participant.",
                 "New hires' employment eligibility is confirmed through E-Verify.",
                 "The company is enrolled in E-Verify.")

# An internship with no student requirement: who may apply, as a fact of its own. Without one the writer made the
# internship a student role (6 of the 12 such check postings wrote 'must be currently enrolled'; the blueprint line
# against it alone did not hold). Several wordings, picked by card hash, so no one wording becomes a stock line;
# none names students, school, enrollment, graduation or a degree, and none says 'anyone' or 'all': a wording that
# opens the door to everyone invited the writer to list who is not shut out ('There are no restrictions based on
# current student status or specific degree programs', rq_00105). Ten wordings: with four, one ('candidates who meet
# this listing's requirements') was in 6 of 39 r5 check postings (15.4%, stock-phrase cap 15%). None uses a word the
# requirement gates read as a requirement of its own (must, minimum, qualifications, necessary, essential).
OPEN_TO_TEXT = ("applicants who meet the requirements below",
                "people who meet the listed requirements",
                "job seekers who meet the stated requirements",
                "those who meet the requirements described here",
                "applicants who fit the requirements that follow",
                "people who match the requirements in this posting",
                "individuals who satisfy the requirements listed here",
                "interested applicants who meet the requirements set out below",
                "people who can meet the requirements outlined below",
                "applicants whose background fits the requirements below")
# wordings no context carries any more, still counted as stock phrases (the writer learned them)
RETIRED_OPEN_TO_TEXT = ("candidates who meet this listing's requirements",)

# How the duties part starts, picked by card hash: the writer's own opener 'In this role, you will' was in 31% of the r13
# check postings (stock-phrase limit 15%), and a negative line in the blueprint did not stop it, so the context gives
# an opener to write. Twenty wordings, so even one copied every time stays near 5% (and under the 15% cap on a
# 39-row check batch); each is counted as a stock phrase
# (ECHO), and none is a requirement, preference, applicant-profile or student wording to the gates.
DUTY_OPENERS = ("Day to day, you will", "Your main responsibilities include", "On a typical day, you can expect to",
                "The work centers on", "Here is what the job involves:", "You will spend most of your time",
                "Core duties of the position include", "Each week, the person we hire will", "What the work looks like:",
                "Responsibilities for this position cover", "Your week will be spent", "The job itself comes down to",
                "Most of the job is", "Expect your days to include", "The person in this job will", "A normal week here means",
                "The day-to-day work includes", "You can expect to spend your days", "This job is mostly about",
                "From week to week, you will")

# A degree (or a student's program) with no field on the card: any field counts. Said as a value of its own, because a
# context with a degree and no field gave the writer nothing to say and it invented a subject (5 of 38 check postings;
# 26% of training postings have a degree and no field, 5% a student requirement and no field). Several wordings, picked
# by card hash, so no one wording becomes a stock phrase; none says 'major' (FIELD_SAID reads 'major in').
ANY_FIELD_TEXT = ("any field", "any discipline", "any subject area", "any area of study", "any academic field")

# The minimum years as a value the writer can write, scope included: with the bare number the writer gave the years a
# domain ('one year of relevant experience in a clinical or support setting', rq_01180; years_domain_added failed 9 of
# 75 r13 tries). The rule counts any work. Several wordings, picked by card hash, so none is a stock line; none says
# 'any field' (ANY_FIELD_TEXT's), and each has the number as has_years reads it ('at least 3 years').
YEARS_TEXT = ("at least {n} of work experience of any kind", "at least {n} of work experience; any kind of job counts",
              "at least {n} of experience in any kind of work", "at least {n} of work history of any type",
              "at least {n} of prior work in any line of work", "at least {n} of employment in any kind of job")


def years_text(n: int, key: str) -> str:
    return _pick(YEARS_TEXT, key + "|years").format(n=f"{n} year" if n == 1 else f"{n} years")


def _h(key: str) -> int:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)


FIRST = ("Maria", "James", "Priya", "Wei", "Aisha", "Daniel", "Sofia", "Omar", "Hannah", "Luis", "Mei", "Tyrone",
         "Elena", "Kofi", "Grace", "Rahul", "Chloe", "Mateo", "Fatima", "Noah")
LAST = ("Gonzalez", "Patel", "Nguyen", "Johnson", "Kim", "Okafor", "Rossi", "Hassan", "Walker", "Silva", "Chen", "Murphy",
        "Ivanov", "Santos", "Cohen", "Brown", "Yamamoto", "Haddad", "Reyes", "Foster")
STREETS = ("Maple Street", "Oak Avenue", "Cedar Road", "Elm Drive", "Lake Boulevard", "Pine Lane", "Hill Court")
CONTACT_SHARE = 0.15


def fake_contact(card: dict) -> dict | None:
    """A fake name and contact block for about 15% of resumes (by card hash). The writer puts it at the
    top; textclean.redact must strip it again, so the redaction path is exercised on training data."""
    key = json.dumps(card, sort_keys=True)
    if (_h(key + "|contact") % 1000) / 1000 >= CONTACT_SHARE:
        return None
    first, last = FIRST[_h(key + "|f") % len(FIRST)], LAST[_h(key + "|l") % len(LAST)]
    n = _h(key + "|n")
    contact = {"name": f"{first} {last}", "email": f"{first}.{last}{n % 90 + 10}@example.com".lower(),
               "phone": f"({200 + n % 700}) 555-{n % 9000 + 1000:04d}"}
    if n % 2:
        contact["linkedin"] = f"linkedin.com/in/{first}-{last}-{n % 900 + 100}".lower()
    if n % 3 == 0:
        contact["address"] = f"{n % 900 + 10} {STREETS[n % len(STREETS)]}, {card.get('current_location', '')}".strip(", ")
    if _h(key + "|layout") % 2:  # the most common real header: "Name | City, ST | phone | email" on one line
        contact["layout"] = "one line"
    return contact


def education_status(card: dict) -> str | None:
    """The graduation year as a sentence, against the label rule's calendar."""
    gy, level = card.get("graduation_year"), card.get("degree_level", "bachelor")
    if level == "none":
        return ("No college degree: education is a high school diploma, certificates or on-the-job training only; "
                "never list an associate, bachelor's or other degree and never write 'expected' or 'in progress'")
    if gy is None:
        return None
    if gy <= CURRENT_YEAR:
        when = f" (a month before October {CURRENT_YEAR})" if gy == CURRENT_YEAR else ""
        return (f"Finished the {LEVEL_TEXT[level]} in {gy}. This is in the past: write a plain graduation date{when} in {gy}, "
                "never 'expected', 'anticipated', 'in progress' or 'currently enrolled'")
    prior, now = "", ""
    if (by := prior_bachelor_year(card)) is not None:
        # the label rule (v3/rules held_level) credits a master's or doctoral student with a finished bachelor's;
        # the resume must show it, or a 'needs a bachelor's' Keep has nothing to point at
        subject = f" in {card['degree_field']}" if card.get("degree_field") else " (by level only, no subject)"
        prior = f". Also list a finished bachelor's degree{subject} with a past graduation date in {by}"
        # the degree in progress is the subject the rule reads ('studying X'): with the subject named only for the
        # bachelor's, the writer gave the master's another one (an MBA for Criminal Justice, cd_v_01236)
        now = f" in {card['degree_field']}" if card.get("degree_field") else ""
    return (f"Still enrolled in the {LEVEL_TEXT[level]}{now}; expected graduation in {gy}. Write 'Expected <month> {gy}' "
            f"and no job that ends after today{prior}")


def prior_bachelor_year(card: dict) -> int | None:
    """The year of the finished bachelor's an enrolled master's or doctoral student lists: two years before
    a master's ends, five before a doctorate ends, never later than this year (None for anyone else)."""
    gy, level = card.get("graduation_year"), card.get("degree_level", "bachelor")
    if gy is None or gy <= CURRENT_YEAR or level not in ("master", "doctoral"):
        return None
    return min(CURRENT_YEAR, gy - (2 if level == "master" else 5))


# O*NET knowledge areas whose name is also a degree subject (degrees.KNOWN_FIELDS, plus Geography and Design, which
# check postings also wrote as one). As a bare required skill the writer made them the degree's subject ('Your degree
# program must be in one of the following fields: Sociology, Philosophy, or Counseling', rq_00437; 'degree in
# Geography', rq_01081), so the context words them as knowledge the applicant has. The card keeps the name, and the
# required-skill gate and visibility look for that name, which 'knowledge of sociology' contains.
SUBJECT_KNOWLEDGE = frozenset({"Biology", "Chemistry", "Physics", "Mathematics", "Psychology", "Sociology", "Philosophy",
                               "Counseling", "History", "Accounting", "Fine arts", "Business management", "Geography",
                               "Design"})


def skill_name(name: str) -> str:
    """A required skill as the context words it: a subject-like knowledge area as 'knowledge of <it>'."""
    return f"knowledge of {name.lower()}" if name in SUBJECT_KNOWLEDGE else name


def skill_entries(groups) -> list:
    """One entry per required group: a name, or {"any_one_of": [names]} for an any-of group."""
    out: list = []
    for g in groups:
        g = [skill_name(x) for x in ([g] if isinstance(g, str) else g)]
        out.append(g[0] if len(g) == 1 else {"any_one_of": g})
    return out


def pay_text(card: dict) -> str | None:
    lo, hi = card.get("salary_min"), card.get("salary_max")
    if lo is None and hi is None:
        return None
    lo, hi = lo if lo is not None else hi, hi if hi is not None else lo
    yearly = f"${lo:,} to ${hi:,} a year" if lo != hi else f"${lo:,} a year"
    if card.get("employment_type") in ("internship", "part_time"):
        a, b = lo / 2080, hi / 2080
        hourly = f"${a:.2f} to ${b:.2f} per hour" if lo != hi else f"${a:.2f} per hour"
        return f"{hourly} (about {yearly})"
    return yearly


# Internship terms that have not started yet (it is October 2026).
TERMS = ("Summer 2027", "Summer 2027", "Spring 2027", "Winter 2027", "Fall 2027")

DEGREE_NAME = {"associate": "associate degree", "bachelor": "bachelor's degree", "master": "master's degree",
               "doctoral": "doctoral degree"}
STUDENT_IN = {"associate": "any degree program (associate or higher)", "bachelor": "a bachelor's or higher degree program",
              "master": "a master's or doctoral program", "doctoral": "a doctoral program"}

# Fictional company names: coined stems plus a word for the industry, picked by card hash (the
# writer's own names repeated: 10 of 100 pilot postings were 'Apex ...', 7 'Nexus ...').
CO_STEM = ("Varrow", "Quillon", "Tessaly", "Brindle", "Corvane", "Halvet", "Lumara", "Ostrel", "Pellam", "Rivane", "Sorrel",
           "Tamsin", "Ulvar", "Veltro", "Wexley", "Yarrowby", "Zenthe", "Ambrel", "Calloway", "Dunmere", "Elstow", "Fenwold",
           "Grisham", "Harlow", "Ivelle", "Juniper Ridge", "Kestrel Bay", "Larchmont", "Merrin", "Northam", "Orrin", "Prescot",
           "Redfern", "Stilwell", "Thornbury", "Upland", "Vantor", "Willowmere", "Ashgrove", "Bellcourt", "Cedarline", "Dovecote",
           "Emberly", "Foxhollow", "Glenmora", "Holloway", "Ironvale", "Lindqvist", "Marsten", "Oakhurst")
CO_WORD = {
    "healthcare": ("Health", "Medical Group", "Care Partners", "Clinic", "Health System"),
    "technology": ("Software", "Systems", "Labs", "Technologies", "Digital"),
    "education": ("Academy", "School District", "Learning", "College", "Schools"),
    "retail": ("Goods", "Market", "Stores", "Supply", "Outfitters"),
    "finance": ("Financial", "Capital", "Bank", "Partners", "Advisors"),
    "government": ("County Services", "Public Works", "City Office", "Regional Authority", "State Agency"),
    "research": ("Research Institute", "Labs", "Research Center", "Scientific", "Institute"),
    "manufacturing": ("Manufacturing", "Industries", "Works", "Fabrication", "Components"),
    "consulting": ("Consulting", "Advisory", "Group", "Partners", "Associates"),
    "construction": ("Builders", "Construction", "Contracting", "Building Co.", "Constructors"),
    "public safety": ("Public Safety", "Emergency Services", "Fire District", "Safety Services", "Response"),
    "utilities": ("Energy", "Power", "Water", "Utilities", "Electric"),
    "biotech and pharma": ("Biosciences", "Therapeutics", "Pharma", "Bio", "Life Sciences"),
    "hospitality": ("Hospitality", "Hotels", "Resorts", "Dining Group", "Inns"),
    "media": ("Media", "Studios", "Press", "Broadcasting", "Creative"),
    "nonprofit": ("Foundation", "Alliance", "Community Services", "Trust", "Network"),
    "agriculture": ("Farms", "Agriculture", "Growers", "Ag", "Harvest Co."),
    "logistics": ("Logistics", "Freight", "Transport", "Distribution", "Shipping"),
    "legal": ("Law Group", "Legal", "LLP", "Law Offices", "Attorneys"),
    "environmental services": ("Environmental", "Ecology", "Green Services", "Environmental Services", "Earthworks"),
}


def company_name(card: dict) -> str:
    key = json.dumps(card.get("source_posting_id", ""), sort_keys=True) + card.get("title", "")
    words = CO_WORD.get(card.get("industry", ""), ("Group", "Company", "Partners", "Services", "Inc."))
    return f"{CO_STEM[_h(key + '|co') % len(CO_STEM)]} {words[_h(key + '|cw') % len(words)]}"


def _pick(options: tuple[str, ...], key: str) -> str:
    return options[_h(key) % len(options)]


MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December")
NOW_MONTH = CURRENT_YEAR * 12 + 9  # October of CURRENT_YEAR, as months since year 0


def month_name(m: int) -> str:
    return f"{MONTHS[m % 12]} {m // 12}"


EXTRA_MONTHS = 6  # the dates span card['years'] plus 0 to 5 months


def licensed_by_degree(card: dict) -> bool:
    """Does the card hold a license that its finished degree gives (onet.LICENSE_EDUCATION level above none)? Then no
    job starts before that graduation (sample / variants keep years <= CURRENT_YEAR - graduation_year), and job_dates
    starts the first job no earlier than July of the graduation year."""
    from matcher.onet import LICENSE_EDUCATION

    gy = card.get("graduation_year")
    if gy is None or gy > CURRENT_YEAR or card.get("degree_level", "none") == "none":
        return False
    return any((n := LICENSE_EDUCATION.get(x)) is not None and n.level != "none" for x in card.get("licenses", ()))


def _job_bounds(card: dict) -> list[int]:
    """job_dates' month bounds: the first job's start, the later starts, and NOW_MONTH."""
    y = int(card.get("years") or 0)
    if y <= 0:
        return []
    key = json.dumps(card, sort_keys=True)
    n = 1 if y <= 2 else 2 if y <= 6 else 3
    total = 12 * y + _h(key + "|extra") % EXTRA_MONTHS
    if licensed_by_degree(card):  # the first licensed job starts after the graduation that gives the license (cd_v_00823)
        total = max(12 * y, min(total, NOW_MONTH - (card["graduation_year"] * 12 + 6)))
    start = NOW_MONTH - total
    cuts = []
    for i in range(1, n):
        span = max(total // (3 * n), 1)
        cuts.append(start + total * i // n + _h(key + f"|cut{i}") % (2 * span + 1) - span)
    return [start, *cuts, NOW_MONTH]


def job_dates(card: dict) -> list[str]:
    """The card's years as dated jobs, most recent first, ending 'Present' (October CURRENT_YEAR).
    1 job up to 2 years, 2 up to 6, else 3, split near evenly by card hash; the jobs touch. Their span
    is card['years'] plus 0 to 5 months (by card hash), so the start month varies like real resumes
    and the whole years are always the card's: years is a floor ('2 years' covers 2.0 to 2.9), and
    years_mismatch checks 0 <= span - years < 1. At most 5 extra months, so the dates never read as
    nearly a year more than the card (2 years 11 months against a 3-year posting). The writer was asked to make dates add up and missed
    in 48 of 98 pilot resumes, so the dates are computed here."""
    bounds = _job_bounds(card)
    n = len(bounds) - 1
    jobs = [f"{month_name(bounds[i])} to {'Present' if i == n - 1 else month_name(bounds[i + 1] - 1)}" for i in range(n)]
    return jobs[::-1]


# Job titles, given to the writer with each job's dates: left to itself it wrote titles a license stands behind with no
# license on the card ('Registered Nurse', cd_v_00286; 'Elementary Teacher', cd_v_00940) and a resident physician a year
# before the MD (cd_v_00823). A card's licensed title comes from its license, for a job that starts after the
# graduation that gives it; every other job gets a plain title of its field, picked by card hash. No title here is a
# PROTECTED_TITLES title without its license, or names a skill (test_job_titles_are_safe).
FIELD_TITLES = {
    "Nursing": ("Patient Care Technician", "Health Unit Coordinator", "Patient Care Associate", "Unit Secretary"),
    "Pharmacy": ("Pharmacy Clerk", "Pharmacy Assistant", "Pharmacy Services Associate", "Pharmacy Cashier"),
    "Clinical healthcare": ("Medical Assistant", "Patient Services Coordinator", "Clinical Research Coordinator",
                            "Medical Office Specialist"),
    "Lab and life sciences": ("Laboratory Assistant", "Research Assistant", "Research Associate", "Laboratory Technician"),
    "Social sciences": ("Research Assistant", "Program Analyst", "Research Coordinator", "Policy Assistant"),
    "Business and finance": ("Business Analyst", "Accounts Assistant", "Operations Analyst", "Finance Associate"),
    "Management": ("Operations Manager", "Project Coordinator", "Program Manager", "Operations Supervisor"),
    "Sales and administration": ("Administrative Assistant", "Sales Associate", "Office Coordinator", "Front Office Assistant"),
    "Education": ("Teaching Assistant", "Tutor", "Instructional Aide", "Education Program Coordinator"),
    "Social services": ("Case Manager", "Community Outreach Coordinator", "Program Assistant", "Family Support Specialist"),
    "Design and media": ("Content Producer", "Media Coordinator", "Production Assistant", "Creative Assistant"),
    "Protective services and forensics": ("Loss Prevention Associate", "Emergency Dispatcher", "Evidence Technician",
                                          "Safety Coordinator"),
    "Trades and construction": ("Construction Worker", "Maintenance Technician", "Building Maintenance Worker",
                                "Site Assistant"),
    "Hospitality and personal care": ("Guest Services Associate", "Front Desk Agent", "Server", "Shift Lead"),
    "IT and data": ("IT Support Specialist", "Systems Analyst", "Technical Support Analyst", "Application Support Analyst"),
    "Engineering": ("Engineering Technician", "Project Engineer", "Engineering Associate", "Test Engineer"),
    "Legal": ("Legal Assistant", "Paralegal", "Legal Secretary", "Compliance Assistant"),
    "Manufacturing and logistics": ("Production Associate", "Logistics Coordinator", "Warehouse Associate", "Quality Inspector"),
    "Agriculture and environment": ("Field Technician", "Environmental Technician", "Farm Assistant", "Conservation Assistant"),
}
LICENSE_TITLES = {
    "RN license": ("Registered Nurse", "Staff Nurse", "Charge Nurse"), "NP license": ("Nurse Practitioner",),
    "LPN license": ("Licensed Practical Nurse",), "CNA certification": ("Certified Nursing Assistant",),
    "Pharmacy technician certification": ("Certified Pharmacy Technician",),
    "Pharmacist license": ("Pharmacist", "Staff Pharmacist", "Clinical Pharmacist"),
    "CPA license": ("Certified Public Accountant",), "Series 65": ("Investment Adviser Representative",),
    "State teaching license": ("Teacher", "Classroom Teacher"), "CDL Class A": ("Truck Driver",),
    "FAA commercial pilot certificate": ("Commercial Pilot",), "Journeyman electrician license": ("Journeyman Electrician",),
    "Plumber license": ("Plumber",), "EPA 608 certification": ("HVAC Technician",),
    "Water operator license": ("Water Treatment Operator",), "EMT certification": ("Emergency Medical Technician",),
    "Firefighter certification": ("Firefighter",), "Police academy certification": ("Police Officer", "Patrol Officer"),
    "Security guard license": ("Security Officer",), "RRT license": ("Respiratory Therapist",),
    "ARRT certification": ("Radiologic Technologist",), "Phlebotomy certification": ("Certified Phlebotomist",),
    "Clinical laboratory scientist license": ("Clinical Laboratory Scientist",),
    "MLT certification": ("Medical Laboratory Technician",), "Physical therapist license": ("Physical Therapist",),
    "Occupational therapist license": ("Occupational Therapist",), "Medical license": ("Physician", "Staff Physician"),
    "Dental license": ("Dentist", "General Dentist"), "Dental hygiene license": ("Dental Hygienist",),
    "Veterinary license": ("Veterinarian", "Associate Veterinarian"),
    "Veterinary technician credential": ("Veterinary Technician",), "Bar admission": ("Attorney", "Associate Attorney"),
    "LCSW license": ("Licensed Clinical Social Worker",), "Cosmetology license": ("Cosmetologist",),
}
_ROLE_NOUN = re.compile(r"\b(?:Engineer|Researcher|Developer|Analyst|Scientist|Manager|Fellow|Intern|Designer|Consultant|"
                        r"Specialist|Associate|Coordinator|Assistant|Architect|Administrator|Technician|Officer|Lead|Director)$")


def _plain_titles(card: dict) -> tuple[str, ...]:
    fld = card.get("field", "")
    if fld in FIELD_TITLES:
        return FIELD_TITLES[fld]
    # a real test card's own field ('Software Engineer'), when it reads as a title, no license stands behind it and it
    # names no skill the card lacks
    t = fld.strip() if _ROLE_NOUN.search(fld.strip()) else f"{fld.strip()} Associate"
    line = f"Candidate\n\nExperience\n{t}\nMay 2024 to Present\n"
    return (t,) if t.strip() and not protected_title(t) and not lacked_skill_named(card, line) else ("Associate",)


def jobs_of(card: dict) -> list[dict]:
    """The context's jobs, most recent first: {"title", "dates"}, and employer_city (current_location) on the job that
    ends 'Present'. A job gets the card's licensed title when it starts on or after January of the graduation of a
    degree-given license (any time for a license that needs no degree)."""
    from matcher.onet import LICENSE_EDUCATION

    bounds, dates = _job_bounds(card), job_dates(card)
    if not dates:
        return []
    key = json.dumps(card, sort_keys=True)
    lic = next((x for x in card.get("licenses", ()) if x in LICENSE_TITLES), None)
    need = LICENSE_EDUCATION.get(lic) if lic else None
    gy = card.get("graduation_year")
    plain = _plain_titles(card)
    out = []
    for i, d in enumerate(dates):  # dates are most recent first; bounds oldest first
        start = bounds[len(dates) - 1 - i]
        licensed = lic is not None and (need is None or need.level == "none" or gy is not None and gy <= CURRENT_YEAR
                                        and start >= gy * 12)
        pool = LICENSE_TITLES[lic] if licensed else plain
        job = {"title": pool[_h(key + f"|title{i}") % len(pool)], "dates": d}
        if i == 0 and card.get("current_location"):
            job["employer_city"] = card["current_location"]
        out.append(job)
    return out


SUMMARY_SKILLS = 3


def summary_skills(card: dict) -> list[str]:
    """Up to SUMMARY_SKILLS of the card's own skills, picked by card hash, for the summary to name: left to itself the
    writer filled the summary with skills the card lacks ('data analysis' in 18 of 25 lacked_skill_named hits, r4)."""
    from matcher.schema import is_soft_skill

    pool = [x for x in card.get("skills", ()) if not is_soft_skill(x)] or list(card.get("skills", ()))
    key = json.dumps(card, sort_keys=True)
    return sorted(pool, key=lambda x: _h(key + "|sum|" + x))[:SUMMARY_SKILLS]



def context_of(kind: str, card: dict) -> dict:
    """What the writer sees. Bare facts, not finished sentences (the writer copied context sentences
    like 'Applicants must already hold a' into 64% of pilot postings); work-authorization statements
    as plain sentences (enum names made it add "cannot support OPT" to 27 of ~160 will_not_sponsor
    postings). An unset field is left out entirely."""
    if kind == "resumes":
        ctx = {k: card[k] for k in RESUME_FIELDS if k in card and card[k] not in ("", None, [], ())}
        if card.get("degree_level", "bachelor") == "none":
            ctx.pop("degree_field", None)
        else:
            ctx["degree"] = LEVEL_TEXT[card.get("degree_level", "bachelor")]
        if (st := education_status(card)):
            ctx["education_status"] = st
        ctx["years_of_experience"] = card.get("years", 0)
        if (jobs := jobs_of(card)):
            ctx["jobs"] = jobs
        if (best := summary_skills(card)):
            ctx["summary_skills"] = best
        if (c := fake_contact(card)):
            ctx["contact"] = c
        return ctx
    key = json.dumps(card.get("source_posting_id", ""), sort_keys=True) + card.get("title", "")
    # the opener is picked by the whole card (not the card-side related_ok flag), so the variants of one base posting
    # do not all share one
    whole = json.dumps({k: v for k, v in card.items() if k != "related_ok"}, sort_keys=True)
    ctx = {"title": card["title"], "company": company_name(card), "duties_opener": _pick(DUTY_OPENERS, whole + "|duty")}
    if card.get("industry"):
        ctx["industry"] = card["industry"]
    if card.get("seniority") and card["seniority"] != "mid":
        ctx["level"] = card["seniority"]
    ctx["employment_type"] = {"full_time": "full-time", "part_time": "part-time"}.get(card.get("employment_type", "full_time"),
                                                                                       card.get("employment_type"))
    # an internship that asks for no enrollment: its degree is a finished one, under a key that says so (with
    # 'minimum_degree' 3 of 7 such check postings asked for students pursuing the degree instead), its fields are the
    # finished degree's subjects, and it has no program term (next to 'Fall 2027' the writer wrote 'for students
    # currently enrolled', rq_00105; open internships were 6 of the 12 failed check postings)
    open_intern = card.get("employment_type") == "internship" and card.get("enrolled_in", "none") == "none"
    if card.get("employment_type") == "internship" and not open_intern and _h(key + "|term") % 100 < 60:
        ctx["program_term"] = TERMS[_h(key + "|t2") % len(TERMS)]
    if card.get("degree_level", "none") != "none":
        ctx["completed_degree" if open_intern else "minimum_degree"] = DEGREE_NAME[card["degree_level"]]
    if card.get("enrolled_in", "none") != "none":
        ctx["must_be_student_in"] = STUDENT_IN[card["enrolled_in"]]
    if open_intern:
        ctx["who_can_apply"] = _pick(OPEN_TO_TEXT, key + "|open")
    if card.get("degree_fields"):
        finished = open_intern and card.get("degree_level", "none") != "none"
        ctx["degree_subjects" if finished else "fields_of_study"] = list(card["degree_fields"])
    elif card.get("degree_level", "none") != "none" or card.get("enrolled_in", "none") != "none":
        ctx["degree_field"] = _pick(ANY_FIELD_TEXT, key + "|anyfield")
    if card.get("min_years", 0) > 0:
        ctx["minimum_years_experience"] = years_text(card["min_years"], whole)
    if card.get("must_skills"):
        ctx["required_skills"] = skill_entries(card["must_skills"])
    if card.get("nice_skills"):
        ctx["preferred_skills"] = list(card["nice_skills"])
    if card.get("license"):
        ctx["required_license"] = card["license"]
    if card.get("locations"):
        ctx["locations"] = list(card["locations"])
    ctx["work_mode"] = {"onsite": "onsite", "hybrid": "hybrid", "remote": "fully remote"}[card.get("work_mode", "onsite")]
    if (pay := pay_text(card)):
        ctx["pay"] = pay
    wa = [_pick(WORK_AUTH_TEXT[card["sponsorship"]], key + "|wa")] if card.get("sponsorship") in WORK_AUTH_TEXT else []
    if card.get("citizens_only"):
        wa.append(_pick(CITIZENS_TEXT, key + "|cit"))
    if card.get("e_verify"):
        wa.append(_pick(E_VERIFY_TEXT, key + "|ev"))
    if wa:
        ctx["work_authorization"] = wa
    return ctx


def client() -> Adaption:
    os.environ.setdefault("ADAPTION_API_KEY", Path("~/.config/adaption/api_key").expanduser().read_text().strip())
    return Adaption()


def adapt_kwargs(kind: str) -> dict:
    k = KINDS[kind]
    return dict(
        column_mapping={"context": ["card"], "universal_prompt": k["universal"]},
        recipe_specification={"recipes": {"prompt_rephrase": False, "deduplication": False, "reasoning_traces": False}},
        brand_controls={"blueprint": k["blueprint"], "length": k["length"], "hallucination_mitigation": False},
        training_type="instruction_dataset",
    )


def download_rows(c: Adaption, did: str) -> list[dict]:
    res = c.datasets.download(did, file_format="csv")
    data = res.read() if hasattr(res, "read") else res.content
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))


LEAK_RESUME = re.compile(r"\b(visa|OPT|CPT|H-?1B|sponsor(ship|ed|s)?|green card|U\.?S\.? citizen\w*|work authori[sz]ation|salary)\b|@|https?://|\(\d{3}\)|\d{3}-\d{3}-\d{4}", re.I)
LEAK_WORDS = re.compile(r"\b(visa|OPT|CPT|H-?1B|sponsor(ship|ed|s)?|green card|U\.?S\.? citizen\w*|work authori[sz]ation|salary)\b", re.I)

STALE_WORD = re.compile(r"expected|anticipated|in progress|currently (?:enrolled|pursuing|completing)|pursuing|candidate for|"
                        r"completing|final[- ]year|graduating|(?:senior|junior|sophomore|freshman),? class of|"
                        r"(?:to|will) be conferred|conferral expected", re.I)
ARTIFACTS = re.compile(r"none listed|\bN/A\b|local community college|springfield university|university location|"
                       r"university located in|\[(?:email|phone|address|name|contact|city|location)\b[^\]\n]{0,25}\]|"
                       r"possesses 0 years|\b0 years\b|layout:\s*one line|\"layout\"|\bone[- ]line layout\b", re.I)
DEGREE_WORDS = re.compile(r"bachelor|associate(?:['’]s)? (?:degree|of)|\bA\.?A\.?S?\b|\bmaster(?:['’]s)?\b(?! (?:of the|new|the|safety))|\bdoctor(?:ate|al)\b|"
                          r"\bB\.?S\.?\b|\bB\.?A\.?\b|\bM\.?S\.?\b|\bPh\.?D|expected graduation|anticipated", re.I)


def stale_expected(text: str) -> list[str]:
    """Lines that call a degree unfinished next to a year that is this year or earlier."""
    bad = []
    for line in text.split("\n"):
        for m in STALE_WORD.finditer(line):
            years = [(abs(y.start() - m.start()), int(y.group())) for y in re.finditer(r"\b(?:19|20)\d\d\b", line)]
            near = [(d, y) for d, y in years if d <= 45]
            if near and min(near)[1] <= CURRENT_YEAR:
                bad.append(line.strip()[:100])
                break
    return bad


_MON = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\.?")
_DATE = rf"(?:{_MON},?\s+(?:19|20)\d\d|\d{{1,2}}/(?:19|20)\d\d|(?:19|20)\d\d)"
DATE_RANGE = re.compile(rf"(?P<a>{_DATE})\s*(?:–|—|-|to|until|through)\s*(?P<b>{_DATE}|present|current|now|today|ongoing)", re.I)
WORK_HEAD = re.compile(r"experience|employment|work history|career history|positions|professional background|work\b", re.I)
NOT_WORK_HEAD = re.compile(r"volunteer|leadership|project|community|activit|extracurricular|education|skill|summary|profile|research", re.I)
_SECTION_KW = (r"(?:experience|employment|history|positions|background|education|skills?|licen[sc]\w*|certif\w*|projects?|"
               r"volunteer\w*|summary|profile|objective|awards?|honors?|publications?|languages?|interests?|coursework|training|"
               r"activities|leadership|references|competenc\w*|information|affiliations?|memberships?|development|"
               r"qualifications|credentials|tools|technolog\w*|proficienc\w*|research)")
# A heading ends in a section word, optionally '& X' / 'and X' ('Skills and Licenses'); a job title
# with such a word inside ('Certified Nursing Assistant', 'Senior Training Coordinator') is not one.
SECTION_LINE = re.compile(rf"(?:[A-Za-z&/'’-]+\s+){{0,3}}{_SECTION_KW}(?:\s+(?:&|and|/)\s+(?:[A-Za-z'’-]+\s*){{1,3}})?", re.I)


def _month(tok: str, start: bool) -> int:
    tok = tok.strip().rstrip(".").lower()
    if tok in ("present", "current", "now", "today", "ongoing"):
        return NOW_MONTH
    y = int(re.search(r"(?:19|20)\d\d", tok).group())
    if (m := re.match(r"(\d{1,2})/", tok)):
        mi = int(m.group(1)) - 1
    elif (m := re.match(r"[a-z]+", tok)):
        mi = next(i for i, name in enumerate(MONTHS) if name.lower().startswith(m.group()[:3]))
    else:  # a bare year: mid-year
        mi = 5
    return y * 12 + mi + (0 if start else 1)  # an end month counts in full


def sections(text: str) -> list[tuple[str, list[str]]]:
    """(heading, lines) blocks; the lines before the first heading have heading ''."""
    out: list[tuple[str, list[str]]] = [("", [])]
    for ln in text.split("\n"):
        s = ln.strip().rstrip(":").strip()
        if s and len(s.split()) <= 6 and not re.search(r"\d", s) and s[0].isupper() and SECTION_LINE.fullmatch(s):
            out.append((s, []))
        else:
            out[-1][1].append(ln)
    return out


JOB_TITLE = re.compile(r"\b(?:assistant|associate|specialist|coordinator|technician|technologist|analyst|intern|instructor|"
                       r"fellow|aide|clerk|manager|officer|engineer|scientist|nurse|teacher|tutor|developer|consultant|"
                       r"representative|supervisor|administrator|worker|operator|lead|director|researcher)\b", re.I)
SCHOOL_HEAD = re.compile(r"educat|academic background|degrees?\b|volunteer|community|leadership|activit|extracurricular|"
                         r"skill|summary|profile|objective|award|honor|certif|licen", re.I)


def work_ranges(text: str) -> list[tuple[int, int]]:
    """Dated ranges of jobs, as (start, end) months: every range under a work or experience heading,
    and a range under another heading (research, projects) whose line or the two lines above name a
    job title ('Graduate Research Assistant'). Education and volunteer sections never count."""
    rng = []
    for head, lines in sections(text):
        if not head or SCHOOL_HEAD.search(head) and not WORK_HEAD.search(head) or re.search(r"volunteer", head, re.I):
            continue
        is_work = bool(WORK_HEAD.search(head)) and not NOT_WORK_HEAD.search(head)
        for i, ln in enumerate(lines):
            for m in DATE_RANGE.finditer(ln):
                if not is_work and not JOB_TITLE.search(" ".join(lines[max(0, i - 2):i + 1])):
                    continue
                a, b = _month(m.group("a"), True), _month(m.group("b"), False)
                if b > a:
                    rng.append((a, b))
    return rng


def work_span_years(text: str) -> float:
    """Months covered by the union of the dated work ranges, in years."""
    total, cur = 0, None
    for a, b in sorted(work_ranges(text)):
        if cur is None or a > cur[1]:
            if cur:
                total += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        total += cur[1] - cur[0]
    return total / 12


def years_mismatch(card: dict, text: str) -> bool:
    """The dated jobs do not span the card's whole years: card years is a floor, so a years=2 card needs
    2.0 to 2.9 years of dates (1.5 years would fail a 2-year minimum; 3.1 would pass a 3-year one)."""
    y = int(card.get("years") or 0)
    if y == 0:
        return bool(work_ranges(text))
    return not 0 <= round(work_span_years(text) - y, 4) < 1


# Tools: the brand-named or technical entries of the skill vocabulary, plus short names a resume uses.
TOOL_PREFIX = re.compile(r"^(?:Microsoft|Adobe|Google|Oracle|SAP|IBM|Autodesk|Salesforce|Epic|Intuit|Apple|Atlassian|ESRI|MEDITECH|"
                         r"Allscripts|Blackboard|Corel|Dassault|Bentley|Henry Schein|The MathWorks|National Instruments|PTC|"
                         r"GE Healthcare|eClinicalWorks|Avid|Python|JavaScript|C\+\+|C#|Perl|Linux|UNIX|SAS|WordPress|Zoom|"
                         r"Structured query language|Hypertext markup language|LexisNexis|Schoology|Nearpod|Padlet|Kahoot)", re.I)
TOOL_ALIASES = {"Microsoft Excel": ("Excel",), "Microsoft PowerPoint": ("PowerPoint",), "Microsoft Outlook": ("Outlook",),
                "Microsoft SharePoint": ("SharePoint",), "Microsoft Access": ("MS Access",), "Adobe Photoshop": ("Photoshop",),
                "Adobe Illustrator": ("Illustrator",), "Adobe InDesign": ("InDesign",), "Adobe Acrobat": ("Acrobat",),
                "Structured query language SQL": ("SQL",), "Hypertext markup language HTML": ("HTML",),
                "The MathWorks MATLAB": ("MATLAB",), "Autodesk AutoCAD": ("AutoCAD",), "Intuit QuickBooks": ("QuickBooks",),
                "Atlassian Confluence": ("Confluence",), "Dassault Systemes SolidWorks": ("SolidWorks",),
                "IBM SPSS Statistics": ("SPSS",), "ESRI ArcGIS software": ("ArcGIS",), "Oracle Java": ("Java",),
                "Salesforce software": ("Salesforce",), "SAP software": ("SAP",), "MEDITECH software": ("MEDITECH",),
                "Microsoft Office software": ("Microsoft Office", "MS Office", "Office suite", "Office 365", "Microsoft 365"),
                "Google Workspace software": ("Google Workspace", "G Suite"), "Epic Systems": ("Epic EHR", "Epic Systems"),
                "Microsoft Windows": ("Windows",)}


# Hot Technology names that are also everyday words or names: never read as a tool ("Go to", "Ruby").
AMBIGUOUS_TOOLS = frozenset({"Go", "Swift", "Ruby"})


def _software_names() -> frozenset[str]:
    """O*NET Hot Technology names as cards show them (onet.display_name: "SQL", "Excel", "AutoCAD", "Git",
    "Docker"), and the short forms TOOL_ALIASES lists. Software category labels ("Inventory control",
    "Scheduling software") are left out: real resumes use them as duty words."""
    from matcher.onet import load_occupations

    try:
        occs = load_occupations()
    except (FileNotFoundError, OSError):  # no O*NET files: the prefix rule alone
        occs = []
    names = {n for o in occs for n in o.hot_tech} | {a for v in TOOL_ALIASES.values() for a in v}
    return frozenset(names - AMBIGUOUS_TOOLS)


def _tool_vocab() -> list[str]:
    """Tool names on the cards: vendor-prefixed O*NET names (TOOL_PREFIX, TOOL_ALIASES) and, since v3/rules
    writes skills by their plain names, every card name that is an O*NET software name ("SQL", "Tableau",
    "Docker"). Those plain names are matched in their own capitals (PLAIN_TOOLS), so "react", "outlook" or
    "spark" in a sentence is no tool."""
    names = set()
    for kind in ("resumes", "postings"):
        for c in load_cards(kind).values():
            for g in c.get("skills", ()) or ():
                names.add(g)
            for g in list(c.get("must_skills", ()) or ()) + list(c.get("nice_skills", ()) or ()):
                names.update([g] if isinstance(g, str) else g)
    soft = _software_names()
    aliases = {a for v in TOOL_ALIASES.values() for a in v}  # short forms keep their old, any-case match
    plain = {n for n in names if n in soft and len(n) > 1 and not (TOOL_PREFIX.match(n) or n in TOOL_ALIASES)}
    PLAIN_TOOLS.clear()
    PLAIN_TOOLS.update(plain - aliases)
    return sorted(n for n in names if TOOL_PREFIX.match(n) or n in TOOL_ALIASES or n in plain)


_VOCAB: list[str] | None = None
PLAIN_TOOLS: set[str] = set()  # plain tool names in the vocab (_tool_vocab), matched case-sensitively


def _word(text: str, name: str) -> bool:
    flags = 0 if name in PLAIN_TOOLS else re.I
    return re.search(rf"(?<![\w+#]){re.escape(name)}(?![\w+#])", text, flags) is not None


def _tool_key(name: str) -> str:
    """'Microsoft Office software', 'Microsoft Office Suite' and 'MS Office' are one tool."""
    k = re.sub(r"\b(?:software|suite|systems?|tools?|programs?|applications?|apps?)\b", "", name.lower())
    k = re.sub(r"\bms\b", "microsoft", k)
    return " ".join(k.replace("-", " ").split())


def extra_tools(card: dict, text: str, vocab: list[str] | None = None) -> list[str]:
    """Tool names in the resume that the card's skills do not have ('Microsoft Excel' or 'Excel' when
    only Word is listed; 'Microsoft Office suite' when Office is not listed)."""
    global _VOCAB
    if vocab is None:
        _VOCAB = _VOCAB if _VOCAB is not None else _tool_vocab()
        vocab = _VOCAB
    have = [_tool_key(x) for x in card.get("skills", ())]
    out: list[str] = []
    said: set[str] = set()  # tool keys already reported: "Excel" and "Microsoft Excel" are one tool
    for tool in vocab:
        if _tool_key(tool) in have:
            continue
        keys = {_tool_key(tool), *(_tool_key(a) for a in TOOL_ALIASES.get(tool, ()))}
        keys |= {_tool_key(k) for k, v in TOOL_ALIASES.items() if tool in v}
        if keys & said:
            continue
        for name in (tool, *TOOL_ALIASES.get(tool, ())):
            if not _word(text, name):
                continue
            # covered when the mention is part of a listed skill ("Microsoft Office" inside "Microsoft Office
            # software") or a listed skill is part of the mention ("Python" inside "Python Scripting")
            spans = list(re.finditer(rf"(?<![\w+#]){re.escape(name)}(?![\w+#])(?:[ ]+[\w+#]+){{0,2}}", text, re.I))
            key = _tool_key(name)
            if any(key and key in h for h in have) or all(any(h and h in _tool_key(m.group()) for h in have) for m in spans):
                continue
            out.append(name)
            said |= keys
            break
    return out


# Credentials the writer may add on its own (a 'CPR certified' line, a 'Laboratory Safety Certification'): a license
# the card does not have is a fact the label never saw, and a posting that requires it would be labelled against it.
ADDON_LICENSES = ("CPR", "BLS", "ACLS", "PALS", "First Aid", "OSHA 10", "OSHA 30", "ServSafe", "HAZWOPER")
LICENSE_EXPAND = {"RN": "Registered Nurse", "CNA": "Certified Nursing Assistant", "NP": "Nurse Practitioner",
                  "LPN": "Licensed Practical Nurse", "LCSW": "Licensed Clinical Social Worker", "CPA": "Certified Public Accountant",
                  "EMT": "Emergency Medical Technician", "MLT": "Medical Laboratory Technician",
                  "RRT": "Registered Respiratory Therapist", "ARRT": "Registered Technologist", "Pharmacy technician": "CPhT"}
_LIC_GENERIC = re.compile(r"\b(?:license|licence|certification|certificate|credential|admission)\b", re.I)
_LIC_HEAD = re.compile(r"licen[sc]|certif|credential", re.I)
_LICENSE_VOCAB: list[str] | None = None


def _license_names(lic: str) -> list[str]:
    """'RN license' -> ['RN license', 'RN', 'Registered Nurse']."""
    k = _LIC_GENERIC.sub("", lic).strip(" ,")
    out = [lic] + ([k] if k and k != lic else [])
    return out + ([LICENSE_EXPAND[k]] if k in LICENSE_EXPAND else [])


def _license_vocab() -> list[str]:
    cards = [*load_cards("resumes").values()]
    posts = [*load_cards("postings").values()]
    return sorted({x for c in cards for x in c.get("licenses", ())} | {c["license"] for c in posts if c.get("license")})


def extra_licenses(card: dict, text: str, vocab: list[str] | None = None) -> list[str]:
    """Licenses and certifications in the resume that the card does not have: a card license name of another
    card or a common add-on (CPR, BLS, First Aid, OSHA 10, ...) anywhere, and any credential line under a
    license or certification heading that names none of the card's licenses or skills."""
    global _LICENSE_VOCAB
    if vocab is None:
        _LICENSE_VOCAB = _LICENSE_VOCAB if _LICENSE_VOCAB is not None else _license_vocab()
        vocab = _LICENSE_VOCAB
    have = [n for x in card.get("licenses", ()) for n in _license_names(x)]
    new = [x for x in (*vocab, *ADDON_LICENSES) if not any(has_phrase(x, h) or has_phrase(h, x) for h in have)]
    out = [x for x in new if (_addon_outside_titles(text, x) if x in ADDON_LICENSES else has_phrase(text, x))]
    for head, lines in sections(text):
        if not _LIC_HEAD.search(head):
            continue
        # a skill under a heading that holds skills too ('Skills & Certifications') is a skill; under a heading of
        # credentials only it is a credential the card does not have ('CPR Visual Estimator' under 'Certifications')
        skills_too = re.search(r"skill|competenc|tool|technolog|proficienc|software", head, re.I)
        for ln in lines:
            x = ln.strip(" -•\t")
            if (not x or len(x) > 90 or ARTIFACTS.search(x) or any(has_phrase(x, n) for n in have)
                    or skills_too and any(has_phrase(x, k) for k in card.get("skills", ()))):
                continue
            if _credential_entry(x):
                out.append(x[:80])
    return out


# capitals that are not a credential: a state ('Relocated to Chicago, IL'), a GPA, a country
_NOT_CRED_TOKEN = set(textclean._STATES.split("|")) | {"GPA", "USA", "US", "UK", "EU", "QPA"}


def _credential_entry(x: str) -> bool:
    """A line under a license heading that reads as a credential: a credential word, or a capital token
    ('BLS', 'ACLS') other than a state or a GPA, in a short line that is not 'No licenses ...' or a city."""
    if re.match(r"no\b|none\b", x, re.I) or textclean.CITY_ST.match(x):
        return False
    if re.search(r"(?i:licen[sc]\w*|certif\w*|credential\w*)", x):
        return True
    return len(x) <= 70 and any(t not in _NOT_CRED_TOKEN for t in re.findall(r"\b[A-Z]{2,5}\b", x))


def _addon_outside_titles(text: str, name: str) -> bool:
    """The add-on credential appears other than as the start of a job title ('CPR Visual Estimator')."""
    for m in re.finditer(rf"(?<![\w+#]){re.escape(name)}(?![\w+#])", text, re.I):
        if not re.match(rf"(?:[ ]+[A-Z][\w&-]*){{0,2}}[ ]+{textclean._ROLE}\b", text[m.end():m.end() + 60], re.I):
            return True
    return False


DEGREE_HEAD = (r"\b(?:Bachelor(?:['’]s)?(?: of [A-Z][a-z]+)?|Master(?:['’]s)?(?: of [A-Z][a-z]+)?|Associate(?:['’]s)?(?: of [A-Z][a-z]+)?|"
               r"Doctor(?:ate)?(?: of [A-Z][a-z]+)?|B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?|A\.?A\.?S?\.?|A\.?S\.?|Ph\.?D\.?)")
MAJOR_AFTER = re.compile(DEGREE_HEAD + r"(?:[ ]+degree)?[ ]*(?:,[ ]*|[ ]+in[ ]+|[ ]*[-–—][ ]*)(?!(?:19|20)\d\d)([A-Z][A-Za-z&]+)")
MAJOR_DEGREES = re.compile(r"\b(?:BSN|MSN|MBA|B\.?S\.?N\.?|M\.?S\.?N\.?|M\.?B\.?A\.?|B\.?S\.?E\.?E|MPH|MSW|BSW|MEd|M\.Ed)\b")
_NOT_MAJOR = {"Expected", "Anticipated", "Graduated", "May", "June", "December", "August", "January", "University", "College",
              "Institute", "School", "Community", "State", "Honors", "Cum", "Magna", "Summa", "GPA", "Dean", "Minor"}


def major_named(card: dict, text: str) -> list[str]:
    """For a card with a degree but no degree_field: majors the resume names after a degree."""
    if card.get("degree_level", "none") == "none" or card.get("degree_field"):
        return []
    edu = "\n".join(ln for head, lines in sections(text) if not head or re.search(r"educat|academic|degree", head, re.I)
                    for ln in lines)
    hits = [m.group(1) for m in MAJOR_AFTER.finditer(edu) if m.group(1) not in _NOT_MAJOR
            and not re.match(r"(?:[A-Z][\w.&'-]*,?[ ]+){0,4}(?:University|College|Institute|School|Academy|Polytechnic)",
                             edu[m.start(1):m.start(1) + 80])]
    hits += MAJOR_DEGREES.findall(text)
    field = card.get("field", "")
    if field and re.search(DEGREE_HEAD + r"[^\n]{0,30}(?i:" + re.escape(field) + ")", text):
        hits.append(field)
    return hits


def city_missing(card: dict, text: str) -> bool:
    city = card.get("current_location", "")
    head = "\n".join(textclean.redact(textclean.strip_markdown(text)).split("\n")[:3])
    return bool(city) and city not in head


# Job titles a license stands behind: the title says the person holds the license, so a title with none of its
# licenses on the card is a license the label never saw ('Registered Nurse' on a card with no license, cd_v_00286: a
# posting that requires an RN license is a Skip the resume contradicts; 'Elementary Teacher' with no teaching
# license, cd_v_00940). Each entry: the title (matched as a whole title segment, case-sensitive), the card licenses
# that cover it, and the degree level the license needs (a job with the title that starts before that degree's
# graduation is a job the person could not hold: a resident physician a year before the MD, cd_v_00823).
_CAP = r"(?:[A-Z][\w'’&.-]*\s+)"  # a capitalised word before the title ('Senior', 'Pediatric', 'Elementary')
_NURSES = ("RN license", "LPN license", "NP license")
PROTECTED_TITLES: tuple[tuple[str, tuple[str, ...], str | None], ...] = (
    (rf"(?!(?:Student|Graduate|Volunteer|Aspiring|Assistant|Associate)\b){_CAP}{{0,3}}(?<!Student )(?<!Graduate )Nurse(?:\s+(?:Manager|Supervisor|"
     r"Educator|Lead|Coordinator|Clinician|Specialist|Navigator|Case Manager))?|RN(?:\s+[A-Z][\w-]*){0,2}|"
     rf"{_CAP}{{1,2}}RN|R\.N\.", _NURSES, None),
    (rf"{_CAP}{{0,2}}Nurse Practitioner|NP|FNP(?:-[A-Z]+)?|APRN", ("NP license",), "master"),
    (r"Licensed (?:Practical|Vocational) Nurse|LPN|LVN", ("LPN license",), None),
    (r"Certified Nurs(?:ing|e) Assistant|Certified Nurse Aide|CNA", ("CNA certification",), None),
    (rf"(?!(?:Student|Aspiring)\b){_CAP}{{0,3}}(?:Physician|Surgeon|Hospitalist|Psychiatrist|Pediatrician|Internist|"
     r"Anesthesiologist|Radiologist|Cardiologist|Dermatologist|Neurologist|Oncologist|Obstetrician|Gynecologist)|"
     r"Medical Doctor", ("Medical license",), "doctoral"),
    (rf"(?!(?:Student|Intern|Aspiring)\b){_CAP}{{0,2}}Pharmacist|PharmD|Pharm\.D\.", ("Pharmacist license",), "doctoral"),
    (r"Certified Pharmacy Technician|CPhT", ("Pharmacy technician certification",), None),
    (rf"(?!(?:Student|Substitute|Assistant|Associate|Preschool|Pre-K|Volunteer|Guest|Head Start|Teacher)\b){_CAP}{{0,3}}"
     r"Teacher", ("State teaching license",), None),
    (r"(?:Licensed )?Clinical Social Worker|LCSW", ("LCSW license",), "master"),
    (r"Clinical Laboratory Scientist|CLS", ("Clinical laboratory scientist license",), None),
    (r"(?:Licensed |Armed |Unarmed )?Security (?:Guard|Officer)", ("Security guard license",), None),
    (rf"{_CAP}?Occupational Therapist|OTR/L", ("Occupational therapist license",), "master"),
    (rf"{_CAP}?Physical Therapist|DPT", ("Physical therapist license",), "doctoral"),
    (r"(?:[A-Z][\w-]*\s+)?(?:Police Officer|Patrol Officer|Police Detective)|Deputy Sheriff|Sheriff['’]?s Deputy|State Trooper",
     ("Police academy certification",), None),
    (r"(?:[A-Z][\w-]*\s+)?Fire ?fighter", ("Firefighter certification",), None),
    (r"(?:Licensed |Journeyman |Master )?Plumber", ("Plumber license",), None),
    (r"Certified Public Accountant|CPA", ("CPA license",), None),
    (rf"{_CAP}{{0,2}}(?:Attorney|Lawyer)|Attorney at Law", ("Bar admission",), "doctoral"),
    (r"Medical Laboratory Technician|Medical Lab Technician|MLT", ("MLT certification", "Clinical laboratory scientist license"), None),
    (r"(?:Registered |Certified )?Respiratory Therapist|RRT", ("RRT license",), None),
    (r"EMT(?:-[A-Z]\w*)?|Emergency Medical Technician(?:-\w+)?|Paramedic", ("EMT certification",), None),
    (r"Certified Phlebotomist|Certified Phlebotomy Technician", ("Phlebotomy certification",), None),
    (r"(?:Licensed )?Cosmetologist", ("Cosmetology license",), None),
    (r"(?:Licensed |Journeyman |Master )?Electrician", ("Journeyman electrician license",), None),
    (rf"{_CAP}?Dentist|DDS|DMD", ("Dental license",), "doctoral"),
    (rf"{_CAP}?Veterinarian|DVM", ("Veterinary license",), "doctoral"),
    (r"(?:Licensed|Registered|Certified|Credentialed) Veterinary Technician|LVT|RVT|CVT", ("Veterinary technician credential",), None),
    (r"(?:Registered )?Radiologic Technologist|RT\(R\)", ("ARRT certification",), None),
)
_PROTECTED = [(re.compile(t), lic, lvl) for t, lic, lvl in PROTECTED_TITLES]
_TITLE_SPLIT = re.compile(r"\s*(?:\||,|;|–|—|\s-\s|/|\(|\)|•|·)\s*")


def _title_segments(line: str) -> list[str]:
    """The parts of a non-bullet line that could be a job title ('Registered Nurse | Bayfront Hospital, Tampa, FL')."""
    s = line.strip()
    if not s or s.startswith(("-", "•", "*", "·", "▪")):
        return []
    return [p.strip(" .:") for p in _TITLE_SPLIT.split(s) if p.strip(" .:")]


def protected_title(line: str) -> list[tuple[str, tuple[str, ...], str | None]]:
    """(title, licenses that cover it, degree it needs) for each protected title segment of `line`."""
    return [(seg, lic, lvl) for seg in _title_segments(line) for rx, lic, lvl in _PROTECTED if rx.fullmatch(seg)]


def title_without_license(card: dict, text: str) -> list[str]:
    """Job titles in the resume that a license stands behind (PROTECTED_TITLES) with none of those licenses on the card.
    License and certification sections are left to extra_licenses."""
    have = set(card.get("licenses", ()))
    out = []
    for head, lines in sections(text):
        if _LIC_HEAD.search(head):
            continue
        for ln in lines:
            out += [seg for seg, lic, _ in protected_title(ln) if not have & set(lic)]
    return out


def job_entries(text: str) -> list[dict]:
    """Dated work entries (the ranges work_ranges counts): start and end month and the entry's own non-bullet lines
    (title, employer, place): up to three lines above the dates back to the previous entry, the dates line and a
    non-bullet line right after it ('Educational Support Specialist / May 2025 to Present / Al-Noor Learning Center,
    Philadelphia, PA')."""
    out = []
    for head, lines in sections(text):
        if not head or SCHOOL_HEAD.search(head) and not WORK_HEAD.search(head) or re.search(r"volunteer", head, re.I):
            continue
        is_work = bool(WORK_HEAD.search(head)) and not NOT_WORK_HEAD.search(head)
        last = -1
        for i, ln in enumerate(lines):
            for m in DATE_RANGE.finditer(ln):
                if not is_work and not JOB_TITLE.search(" ".join(lines[max(0, i - 2):i + 1])):
                    continue
                own = [x for x in lines[max(last + 1, i - 3):i] if x.strip() and not x.strip().startswith(("-", "•", "*"))]
                after = lines[i + 1] if i + 1 < len(lines) else ""
                if after.strip() and not after.strip().startswith(("-", "•", "*")) and not DATE_RANGE.search(after):
                    own.append(after)
                out.append({"start": _month(m.group("a"), True), "end": _month(m.group("b"), False),
                            "present": m.group("b").lower() in ("present", "current", "now", "today", "ongoing"),
                            "lines": own + [ln]})
            if DATE_RANGE.search(ln):
                last = i
    return out


def licensed_title_too_early(card: dict, text: str) -> list[str]:
    """Protected titles dated before the graduation of the degree their license needs, on a card whose own (finished)
    degree is that degree: 'Resident Physician, June 2015 to July 2018' with an MD in 2016 (cd_v_00823). The dates are
    the card's job_dates, and the context gives each job its title (jobs_of: a plain title before the graduation), so this
    catches a writer that swapped in a licensed title."""
    gy, level = card.get("graduation_year"), card.get("degree_level", "none")
    if gy is None or gy > CURRENT_YEAR or level not in ("master", "doctoral"):
        return []
    out = []
    for e in job_entries(text):
        if e["start"] >= gy * 12:
            continue
        out += [seg for ln in e["lines"] for seg, lic, lvl in protected_title(ln) if lvl == level]
    return out


_COUNTRIES: frozenset[str] | None = None


def _countries() -> frozenset[str]:
    """Country names a resume writes after a city: the cards' education countries and a few more."""
    global _COUNTRIES
    if _COUNTRIES is None:
        names = {c.get("education_country", "") for c in load_cards("resumes").values()}
        names |= {"Saudi Arabia", "China", "India", "South Korea", "Korea", "Japan", "Canada", "Mexico", "Brazil", "Nigeria",
                  "United Kingdom", "UK", "England", "Germany", "France", "Turkey", "Türkiye", "Vietnam", "Philippines",
                  "Pakistan", "Egypt", "Iran", "Taiwan", "Australia", "Spain", "Italy", "Russia", "Ukraine", "Kenya", "Ghana"}
        _COUNTRIES = frozenset(n for n in names if n and n.lower() not in ("united states", "usa", "us"))
    return _COUNTRIES


def _entry_places(line: str) -> list[str]:
    """'City, ST' / 'City, State' / 'City, Country' pairs in an entry line (every pair of adjacent comma or bar parts
    whose second part is a state or a country: 'Capital Insight Group, Washington, DC' gives both
    'Capital Insight Group, Washington' and 'Washington, DC')."""
    parts = [p.strip(" .") for p in re.split(r"\s*(?:,|\||–|—|\s-\s|•|·)\s*", line) if p.strip(" .")]
    out = []
    for a, b in zip(parts, parts[1:]):
        b2 = re.sub(r"\s+\d{5}(?:-\d{4})?$", "", b)
        if (re.fullmatch(rf"(?:{textclean._STATES})", b2) or re.fullmatch(rf"(?:{STATE_NAMES})", b2)
                or b2 in _countries()) and re.match(r"[A-Z]", a) and not DATE_RANGE.search(a):
            out.append(f"{a}, {b2}")
    return out


_REMOTE_PART = re.compile(r"(?:fully |100% )?remote(?:ly)?(?: (?:role|position|work|basis|us|usa))?", re.I)  # not 'Remote Sensing'


def present_job_elsewhere(card: dict, text: str) -> list[str]:
    """The place of a job that ends 'Present', when the entry names one and it is outside current_location's metro
    (metros.places_match): someone living in Portland, OR does not hold a current job in Riyadh (cd_v_01354), and
    Hampton, VA is not Arlington, VA's metro (cd_rt_673458ad45). An entry with no place, or 'Remote', passes; earlier
    jobs may be anywhere."""
    from matcher.metros import places_match

    cur = card.get("current_location", "")
    if not cur:
        return []
    out = []
    for e in job_entries(text):
        if not e["present"] or any(_REMOTE_PART.fullmatch(p) for x in e["lines"] for p in _TITLE_SPLIT.split(x.strip())):
            continue
        places = [p for x in e["lines"] for p in _entry_places(x)]
        city = cur.split(",")[0].strip().lower()
        if places and not any(places_match(cur, p) or p.split(",")[0].strip().lower() == city for p in places):
            out.append(places[-1])
    return out


_POSTING_SKILLS: list[str] | None = None


def _posting_skill_vocab() -> list[str]:
    """Skill names a label can cite: every required or preferred skill on a posting card that the rule counts (no
    soft skills)."""
    from matcher.schema import is_soft_skill

    names = set()
    for i, c in load_cards("postings").items():
        if not i.startswith("rq_"):  # the training cards (real-test cards hold free-text names: 'Teams', 'IL')
            continue
        for g in list(c.get("must_skills", ()) or ()) + list(c.get("nice_skills", ()) or ()):
            names.update([g] if isinstance(g, str) else g)
    tools = set(_VOCAB if _VOCAB is not None else _tool_vocab())  # tool names are extra_tools' (case-sensitive there)
    return sorted(n for n in names if len(n) > 1 and not is_soft_skill(n) and n not in tools)


def lacked_skill_named(card: dict, text: str, vocab: list[str] | None = None) -> list[str]:
    """Skill names a posting can require that the resume names although the rule says the candidate lacks them
    (rules.held_skills: the card's skills in any wording, what a concrete skill shows, what the degree field shows):
    'data analysis' on a card without Data analysis (cd_00704), 'Cellular Biology' in a coursework list without
    Biology (cd_v_00678). A posting that requires the skill gets a 'missing' reason the resume contradicts; the
    label-side check (visibility._names_lacked) reads the text the same way."""
    global _POSTING_SKILLS
    from matcher.rules import held_skills
    from matcher.schema import CandidateCard, group_met, skill_key
    from matcher.visibility import _names_lacked

    if vocab is None:
        _POSTING_SKILLS = _POSTING_SKILLS if _POSTING_SKILLS is not None else _posting_skill_vocab()
        vocab = _POSTING_SKILLS
    c = CandidateCard.from_dict({k: v for k, v in card.items() if k in CandidateCard.__dataclass_fields__})
    have = held_skills(c)
    items = {skill_key(x) for x in _list_items(text)}
    return [x for x in vocab if not group_met((x,), have)
            and (skill_key(x) in items if x in LIST_ONLY_SKILLS else _names_lacked(text, x, c))]


# Skill names that are also everyday words ('design and distribute surveys', 'track patient history', 'accounting
# for'): they count only as a list entry of their own ('Skills: Design, Photoshop').
LIST_ONLY_SKILLS = frozenset({"Design", "History", "Accounting"})


def _list_items(text: str) -> list[str]:
    """Every '-' item and every comma, semicolon or bar separated part of a line, stripped."""
    out = []
    for ln in text.split("\n"):
        s = ln.strip().lstrip("-•*·▪ ").strip()
        s = re.sub(r"^[A-Z][\w &/]{0,30}:\s*", "", s)  # 'Skills: Design, Photoshop'
        out += [p.strip(" .") for p in re.split(r"\s*[,;|]\s*", s) if p.strip(" .")]
    return out


# visibility's probes, tightened where they pass on text that is not the fact: _LEVEL's bare MA / MS / MD are
# also state abbreviations ('Boston, MA' satisfied a master's degree) and 'associate' is also a job title;
# _enrollment's 'candidate' is the redacted header of every resume and 'current' is 'current role'.
LEVEL_WORDS = {
    "associate": r"associate(?:['’]s)?\s+(?:degree|of\b|in\b)|\bA\.A\.S?\.?|\bA\.S\.|\bAAS\b",
    "bachelor": r"bachelor|\bB\.?S\.?\b|\bB\.?A\.?\b|\bBSc\b|\bBSN\b",
    "master": r"\bmaster(?:['’]?s)?\b|\bM\.S\.|\bM\.A\.|\bMSc\b|\bMBA\b|\bMSN\b|\bM\.?Ed\b|\bMPH\b",
    "doctoral": r"doctor|ph\.?\s?d|\bD\.?N\.?P\b|\bPharm\.?\s?D\b|\bM\.D\.|\bJ\.D\.|\bEd\.?D\b",
}
EXPECTED_MARK = re.compile(r"\bexpected\b|anticipated|in progress|currently (?:enrolled|pursuing)|candidate for|"
                           r"(?:degree|ph\.?\s?d\.?|doctoral|master['’]?s) candidate", re.I)


def expected_near(text: str, year: int) -> bool:
    """An 'Expected' marker within 60 characters of the graduation year."""
    return any(EXPECTED_MARK.search(text[max(0, m.start() - 60):m.end() + 60])
               for m in re.finditer(rf"(?<!\d){year}(?!\d)", text))
_YEARS_NUM = "|".join(_NUM_WORDS)
STATED_YEARS = re.compile(rf"(?<![\d.])(\d+|\b(?:{_YEARS_NUM})\b)\s*(?:\(\d+\)\s*)?\+?\s*-?\s*(?:years?|yrs?)\b"
                          r"(?!\s+(?:degree|program|college|university|institution|school|course|curriculum|term|plan|contract)\b)(?![- ]olds?\b)", re.I)


def stated_years(text: str) -> list[int]:
    """Every 'N years' / 'five years' / '3+ years' number in the text ('4-year degree' does not count)."""
    out = []
    for m in STATED_YEARS.finditer(text):
        tok = m.group(1).lower()
        out.append(int(tok) if tok.isdigit() else _NUM_WORDS.index(tok))
    return out


def resume_missing_facts(card: dict, text: str) -> list[str]:
    """Card facts the label rule reads that the resume does not show (the probes of
    matcher.visibility, so a text that passes here passes assemble_training's missing_facts on the
    resume side). Every skill and license verbatim, the degree level and field, the graduation year,
    'Expected' for a degree still in progress, the finished bachelor's of an enrolled master's or doctoral
    student (prior_bachelor_year), and the years as a number. The writer is told to
    write each of them; a text without one would carry a label its resume does not support."""
    out = []
    if any(not has_phrase(text, s) for s in card.get("skills", ())):
        out.append("missing_skill")
    if any(not has_phrase(text, s) for s in card.get("licenses", ())):
        out.append("missing_license")
    level = card.get("degree_level", "bachelor")
    if level != "none":
        if not re.search(LEVEL_WORDS[level], text, re.I):
            out.append("missing_degree_level")
        if card.get("degree_field") and not has_phrase(text, card["degree_field"]):
            out.append("missing_degree_field")
        gy = card.get("graduation_year")
        if gy is not None and not has_number(text, gy):
            out.append("missing_graduation_year")
        if gy is not None and gy > CURRENT_YEAR and not expected_near(text, gy):
            out.append("missing_expected_marker")
        if (by := prior_bachelor_year(card)) is not None and not any(  # the bachelor's and its year together
                has_number(text[max(0, m.start() - 120):m.end() + 120], by) for m in re.finditer(LEVEL_WORDS["bachelor"], text, re.I)):
            out.append("missing_prior_bachelor")
    y = int(card.get("years") or 0)
    if y and not has_years(text, y):
        out.append("missing_years")
    if any(n > y for n in stated_years(text)):  # 'nearly 4 years' for 3: the dates allow it, the label does not
        out.append("years_overstated")
    return out


# The degree an enrolled student is still pursuing carries the card's degree_field: the rule reads it as the subject being
# studied ('studying Criminal Justice', rules._check_degree), so a posting for students of that subject gets a 'meets'
# label. The writer put the subject on the finished bachelor's only and named another program for the degree in progress:
# 'Master of Business Administration, Expected May 2027' for Criminal Justice (cd_v_01236), 'Master of Arts in Education'
# for Elementary Education (cd_v_00940). missing_degree_field passed, because the bachelor's line names the subject.
DEGREE_ABBREV_SUBJECT = {"MBA": "Business Administration", "MSN": "Nursing", "BSN": "Nursing", "MPH": "Public Health",
                         "MSW": "Social Work", "BSW": "Social Work", "MEd": "Education", "M.Ed": "Education",
                         "EdD": "Education", "Ed.D": "Education", "PharmD": "Pharmacy", "Pharm.D": "Pharmacy",
                         "DNP": "Nursing", "JD": "Law", "J.D": "Law", "MD": "Medicine", "M.D": "Medicine"}
_EDU_HEAD = re.compile(r"educat|academic|degree|qualification", re.I)


def _subject_shown(entry: str, field: str) -> bool:
    if has_phrase(entry, field):
        return True
    return any(re.search(rf"(?<![\w.]){re.escape(a)}\b", entry) and (has_phrase(sub, field) or has_phrase(field, sub))
               for a, sub in DEGREE_ABBREV_SUBJECT.items())


def program_field_missing(card: dict, text: str) -> list[str]:
    """For a card still enrolled (graduation_year after this year) with a degree_field: the in-progress degree entries of
    the education section (a degree line with an 'Expected' mark and the graduation year on it or on the line next to
    it, up to the next degree line or blank line) that do not name the degree_field. A summary sentence is not an entry."""
    gy, field, level = card.get("graduation_year"), card.get("degree_field"), card.get("degree_level", "none")
    if not field or level == "none" or gy is None or gy <= CURRENT_YEAR:
        return []
    blocks = [lines for head, lines in sections(text) if _EDU_HEAD.search(head)] or [text.split("\n")]
    level_rx = re.compile(LEVEL_WORDS[level], re.I)
    any_degree = re.compile("|".join(LEVEL_WORDS.values()), re.I)
    out = []
    for lines in blocks:
        for i, ln in enumerate(lines):
            if len(ln) > 140 or not EXPECTED_MARK.search(ln):
                continue
            near = " ".join(lines[max(0, i - 1):i + 2])
            if not has_number(near, gy):
                continue
            # the degree line: this one, or one of the two above it in the same entry, or the one right after it
            cand = [j for j in (i, i - 1, i - 2) if j >= 0 and lines[j].strip() and len(lines[j]) <= 140
                    and level_rx.search(lines[j]) and all(lines[k].strip() for k in range(j, i + 1))]
            if not cand and i + 1 < len(lines) and level_rx.search(lines[i + 1]):
                cand = [i + 1]
            if not cand:
                continue
            d = cand[0]
            end = max(d, i)
            while end + 1 < len(lines) and end + 1 - d < 4 and lines[end + 1].strip() and not any_degree.search(lines[end + 1]):
                end += 1
            entry = "\n".join(lines[min(d, i):end + 1])
            if not _subject_shown(entry, field):
                out.append(" / ".join(x.strip() for x in lines[min(d, i):end + 1] if x.strip())[:160])
    return list(dict.fromkeys(out))


# A certification or certificate the card's licenses do not hold, stated as the person's own: 'Certified in CPR Visual
# Estimator techniques' for a card whose only CPR is the software skill 'CPR Visual Estimator' (cd_v_00840; licenses
# empty), 'Secondary Education Certification' (cd_00966). extra_licenses reads license names and a license heading's
# lines; this reads the claim anywhere. Someone else's credential ('under the supervision of licensed pharmacists',
# 'verifying visitor credentials') and a record kept for others ('certification logs') are no claim.
_CRED_NOUNS = r"(?:certifications?|certificates?|credentials?|licen[sc]es?|licensure)"
_NOT_HELD_AFTER = (r"(?!\s+(?:logs?|records?|files?|data|tracking|requirements?|audits?|process\w*|documents?|documentation|"
                   r"paperwork|renewals?|programs?|courses?|coursework|classes|training|deadlines?|compliance|status|"
                   r"databases?|systems?|forms?|exams?|examinations?|preparation|prep|study)\b)")
CRED_CLAIM = re.compile(
    r"\bcertified (?:in|as|for|by)\b|"
    r"\b(?:hold(?:s|ing)?|held|earn(?:ed|ing|s)?|obtain(?:ed|ing|s)?|receiv(?:ed|ing|es)|complet(?:ed|ing|es)|"
    r"achiev(?:ed|ing|es)|pass(?:ed|ing|es)|possess(?:es|ing)?|maintain(?:s|ed|ing)?|acquir(?:ed|ing|es)|attain(?:ed|ing|s)?)"
    rf"\b(?:\s+[\w'’&/+-]+){{0,5}}?\s+{_CRED_NOUNS}\b{_NOT_HELD_AFTER}|"
    r"\b(?:certifications?|certificates?|licen[sc]es?)\s*:|"
    # a line that is a credential's name ('Google Data Analytics Certificate', 'Secondary Education Certification')
    r"(?-i:^\W*(?:[A-Z][\w+&/.'’-]*\s+){1,5}(?:Certification|Certificate|Certified|License|Licence)\b)"
    rf"(?-i:{_NOT_HELD_AFTER})", re.I | re.M)


def credential_claimed(card: dict, text: str) -> list[str]:
    """Lines that state a certification, certificate or license as the person's own and name none of the card's
    licenses (by name, short name or licensed title)."""
    lic = card.get("licenses", ())
    have = [n for x in lic for n in _license_names(x)] + [t for x in lic for t in LICENSE_TITLES.get(x, ())]
    out = []
    for ln in text.split("\n"):
        if not CRED_CLAIM.search(ln) or re.match(r"\W*(?:no|none|not)\b", ln, re.I) or any(has_phrase(ln, h) for h in have):
            continue
        out.append(ln.strip()[:120])
    return out


# A higher degree the card does not have: 'Currently pursuing advanced studies in cybersecurity' on an enrolled bachelor's
# card (cd_v_00639_r2_..., 112 kept texts) reads as a graduate student, and pairs that cite the bachelor's ('for students
# pursuing a master's degree or higher; you are pursuing a bachelor's degree', cd_v_00938_r0_enrollment_level) then
# contradict the resume. On a master's card only a doctorate is higher. Course names ('Advanced Programming') are not one.
HIGHER_STUDY = re.compile(r"\b(?:advanced|post-?graduate)[- ](?:stud(?:y|ies)|academic\w*|degrees?|education)\b|"
                          r"\bgraduate[- ](?:stud(?:y|ies)|degrees?|education|school|program(?:me)?s?|training|coursework|level)\b|"
                          r"\b(?:pursuing|completing|working toward|enrolled in) (?:a |an |my |his |her |their )?"
                          r"(?:master|doctor|ph\.?\s?d|graduate)", re.I)
DOCTOR_STUDY = re.compile(r"\b(?:pursuing|completing|working toward|enrolled in) (?:a |an |my |his |her |their )?"
                          r"(?:doctor\w*|ph\.?\s?d)|\bdoctoral (?:stud(?:y|ies)|program(?:me)?s?|candidate|research)\b", re.I)


def higher_degree_implied(card: dict, text: str) -> list[str]:
    lvl = card.get("degree_level", "bachelor")
    rx = HIGHER_STUDY if lvl in ("none", "associate", "bachelor") else DOCTOR_STUDY if lvl == "master" else None
    return [_clip(_sentence_at(text, m.start())) for m in rx.finditer(text)] if rx else []


# Pair-aware: a skill the card's own pairs say it lacks ('missing required skill: Accounting', 'needs one of: Digital
# logic, CMOS, Computer architecture, ...') is never named in its resume. lacked_skill_named reads the training postings'
# skill names only, so a real-test resume was never checked against its real posting's names (cd_rt_076c07da45 lists
# 'Computer Architecture'), and an everyday-word skill (LIST_ONLY_SKILLS) counted only as a list entry ('three years in
# public accounting', cd_v_00270_must_skill). The names come from the pairs of every card with this card's content
# (pairs/manifest.jsonl, real_test/rows.jsonl).
SKILL_REASON = re.compile(r"^(?:missing required skill|missing nice-to-have skill|needs one of): ")
_PAIR_LACKED: dict[str, set[str]] | None = None


def _card_key(card: dict) -> str:
    return json.dumps(card, sort_keys=True)


def pair_lacked() -> dict[str, set[str]]:
    """card content -> the skill names its pairs' reasons say it lacks."""
    global _PAIR_LACKED
    if _PAIR_LACKED is None:
        cards = load_cards("resumes")
        by: dict[str, set[str]] = defaultdict(set)
        for r in read_jsonl(PAIRS / "manifest.jsonl") + read_jsonl(REAL_TEST / "rows.jsonl"):
            c = cards.get(r.get("cd_id", ""))
            if c is None:
                continue
            for x in r.get("reasons", ()):
                if (m := SKILL_REASON.match(x)):
                    by[_card_key(c)].update(n.strip() for n in x[m.end():].split(", ") if len(n.strip()) > 2)
        _PAIR_LACKED = dict(by)
    return _PAIR_LACKED


def pair_lacked_named(card: dict, text: str, lacked: set[str] | None = None) -> list[str]:
    from matcher.schema import CandidateCard
    from matcher.visibility import _names_lacked

    names = lacked if lacked is not None else pair_lacked().get(_card_key(card), set())
    if not names:
        return []
    c = CandidateCard.from_dict({k: v for k, v in card.items() if k in CandidateCard.__dataclass_fields__})
    return sorted(n for n in names if _names_lacked(text, n, c))


# A card with no jobs lists no dated role anywhere: 'Retail Associate / Green Valley Market / June 2024 - August 2025' or
# 'Academic Tutor, August 2023 - Present' under Volunteer Experience (cd_v_00027_r0_years_far_short: its pair says 'needs
# 1+ years of experience; you have 0'). years_vs_dates counts work sections only; this reads Volunteer, Projects,
# Leadership and the rest too. The Education block keeps its own dates.
_EDU_ONLY = re.compile(r"educat|academic background|\bdegrees?\b|coursework", re.I)


def no_job_dated_role(card: dict, text: str) -> list[str]:
    if jobs_of(card):
        return []
    out = []
    for head, lines in sections(text):
        if head and _EDU_ONLY.search(head) and not re.search(r"project|research|experience|volunteer", head, re.I):
            continue
        for i, ln in enumerate(lines):
            if DATE_RANGE.search(ln) and JOB_TITLE.search(" ".join(lines[max(0, i - 2):i + 1])):
                out.append(_clip(" / ".join(x.strip() for x in lines[max(0, i - 2):i + 1] if x.strip())))
    return out


# A letter of another script in an English text: 'greeted visitor接待' (8 kept resumes, cd_00370), '现场' and '秩序'. No
# card holds one, so the writer added it. Accented Latin letters ('Bogotá', 'résumé') are Latin; dashes, quotes and
# bullets are punctuation, not letters. CJK punctuation and fullwidth forms count too. A character the card itself holds
# is the card's.
_LETTER_OK = frozenset({"MICRO SIGN", "FEMININE ORDINAL INDICATOR", "MASCULINE ORDINAL INDICATOR"})


def _foreign_char(ch: str) -> bool:
    if ord(ch) < 0x80 or 0x2100 <= ord(ch) <= 0x214F:  # letterlike symbols ('ℹ', '№') are symbols, not a script
        return False
    if 0x3000 <= ord(ch) <= 0x303F or 0xFF00 <= ord(ch) <= 0xFFEF:
        return True
    name = unicodedata.name(ch, "")
    return unicodedata.category(ch).startswith("L") and not name.startswith("LATIN") and name not in _LETTER_OK


def non_latin_script(card: dict, text: str) -> list[str]:
    """The words of `text` that hold a letter of a non-Latin script (CJK, Cyrillic, Greek, Arabic and the like) the card
    does not hold."""
    own = set(json.dumps(card, ensure_ascii=False))
    return list(dict.fromkeys(w for w in re.findall(r"\S+", text) if any(_foreign_char(ch) and ch not in own for ch in w)))


def resume_problems(card: dict, text: str) -> list[str]:
    ctx = context_of("resumes", card)
    out = resume_missing_facts(card, textclean.redact(textclean.strip_markdown(text)))
    if stale_expected(text):
        out.append("stale_expected")
    if "contact" in ctx:
        red = textclean.redact(text)
        street = ctx["contact"].get("address", "").split(",")[0]
        if textclean.has_contact(red) or ctx["contact"]["name"] in red or street and street in red:
            out.append("redact_leaves_contact")
    elif LEAK_RESUME.search(text) or LEAK_WORDS.search(text):
        out.append("leak_term")
    if card.get("degree_level") == "none" and DEGREE_WORDS.search(text):
        out.append("degree_for_none")
    if ARTIFACTS.search(text):
        out.append("artifact")
    if textclean.has_markdown(textclean.strip_markdown(text)):
        out.append("markdown_after_clean")
    if years_mismatch(card, textclean.strip_markdown(text)):
        out.append("years_vs_dates")
    if city_missing(card, text):
        out.append("city_missing")
    if major_named(card, textclean.strip_markdown(text)):
        out.append("major_for_blank_field")
    if extra_tools(card, textclean.strip_markdown(text)):
        out.append("extra_tool")
    if extra_licenses(card, textclean.strip_markdown(text)):
        out.append("extra_license")
    if credential_claimed(card, textclean.redact(textclean.strip_markdown(text))):
        out.append("certification_added")
    if program_field_missing(card, textclean.redact(textclean.strip_markdown(text))):
        out.append("program_field_missing")
    if title_without_license(card, textclean.strip_markdown(text)):
        out.append("title_without_license")
    if licensed_title_too_early(card, textclean.strip_markdown(text)):
        out.append("licensed_title_too_early")
    if present_job_elsewhere(card, textclean.strip_markdown(text)):
        out.append("present_job_elsewhere")
    if lacked_skill_named(card, textclean.redact(textclean.strip_markdown(text))):
        out.append("lacked_skill_named")
    if higher_degree_implied(card, textclean.strip_markdown(text)):
        out.append("higher_degree_implied")
    if pair_lacked_named(card, textclean.redact(textclean.strip_markdown(text))):
        out.append("pair_lacked_named")
    if no_job_dated_role(card, textclean.strip_markdown(text)):
        out.append("no_job_dated_role")
    if non_latin_script(card, text):
        out.append("non_latin_script")
    return out


# 'X is not required' in every wording the writer used or might ('is not mandated / expected / a requirement',
# 'you do not need a degree', 'no prior experience is needed'), for a fact the card leaves unset.
_NOT_NEEDED = (r"(?:(?:is|are)(?: not|n't)|not) (?:strictly |absolutely |always )?(?:required|necessary|needed|mandated|mandatory|expected|"
               r"essential|compulsory|a (?:requirement|must|prerequisite|necessity))")
_OPTIONAL = r"(?:is |are )?optional\b"
_DONT_NEED = r"(?:do not|don't|does not|doesn't|will not|won't|not) (?:need|require|have to have|be required to (?:have|hold))"


def _neg(noun: str, no_forms: str, over: bool = True, optional_gap: str = r"[^.\n]") -> re.Pattern:
    """'no X', 'X is not required / optional', 'you do not need X', 'without X' and (over=True) 'over X'
    ('we value skills over formal credentials'; not for experience: 'over two years of experience').
    optional_gap is what may stand between X and 'optional' (for experience not a skill: 'experience with
    Excel is optional' is a preferred skill, not 'no experience needed')."""
    alt = (rf"|\b(?:over|instead of|in place of|rather than) (?:a |any )?(?:[a-z-]+ ){{0,2}}{noun}\b" if over else "")
    return re.compile(rf"\bno (?:[a-z-]+ ){{0,2}}{no_forms}|\b{noun}\b[^.\n]{{0,30}}?\b(?:{_NOT_NEEDED})|"
                      rf"\b{noun}\b(?:{optional_gap}){{0,30}}?\b{_OPTIONAL}|{_DONT_NEED}\b[^.\n]{{0,25}}?\b{noun}\b|"
                      rf"\bwithout (?:a |any )?(?:prior |previous )?{noun}\b{alt}", re.I)


NEG_DEGREE = _neg(r"(?:degrees?|diplomas?|(?:formal |college |university )education|formal qualifications?|formal credentials?)",
                  r"(?:degree|education|qualifications?|diploma)\b")
NEG_EXP = _neg(r"experience", r"(?:prior |previous |relevant )?(?:work |professional |industry )?experience\b", over=False,
               optional_gap=r"(?!\b(?:with|in|using|of|on|across)\b)[^.\n]")
NEG_LICENSE = _neg(r"(?:licen[sc](?:es?|ure|ing)|certifications?|credentials?)", r"(?:licen[sc]\w*|certifications?|credentials?)\b")
NEG_AUTH = re.compile(r"regardless of (?:your |their |an applicant's |the applicant's |a candidate's )?(?:current )?(?:citizenship|immigration|visa|"
                      r"national origin and|work authori)|(?:all|any) (?:immigration|visa|citizenship) status(?:es)?|"
                      r"(?:visa|sponsorship|work authori[sz]ation|citizenship)[^.\n]{0,60}"
                      r"(?:not (?:stated|specified|addressed)|case[- ]by[- ]case|(?:on|upon) request|to be determined)", re.I)
NEG_PAY = re.compile(r"commensurate|competitive (?:salary|pay|compensation|wage)|(?:salary|pay|compensation|wage)s? (?:is |are |will be )?[^.\n]{0,40}"
                     r"(?:discussed|negotiat|determined|depends|based on (?:experience|qualifications))|not (?:disclosed|specified|stated|listed)", re.I)
NEG_ANY = re.compile(r"not stated|not specified|available (?:on|upon) request|decided case|case[- ]by[- ]case", re.I)
# Any work-authorization topic: an unset card's posting says nothing about it (OPT, CPT in upper case only: 'opt in';
# 'immigration' only as a status: an officer's duties may name immigration law).
AUTH_TOPIC = re.compile(r"sponsor|\bvisas?\b|(?-i:\b(?:OPT|CPT)\b)|\bH-?1-?B\b|\b[FJ]-?1\b|e-?verify|work authori[sz]\w*|authori[sz]ed to work|"
                        r"authori[sz]ation to work|employment authori[sz]ation|citizenship|U\.?S\.? citizens?|immigration (?:status|sponsorship|support)|"
                        r"international (?:students?|candidates?|applicants?|graduates?)|work permits?|foreign nationals?|"
                        r"(?:legally|lawfully) (?:permitted|authori[sz]ed|eligible|able) to work|eligib\w+ to work|green card|"
                        r"permanent residen\w+|right to work|"
                        # citizens (not 'senior citizens'), residence in or outside the country, any country
                        r"(?<!senior )(?<!fellow )(?<!good )(?<!our )(?<!the )(?<!local )(?<!community )(?<!area )(?<!its )"
                        r"(?<!both )(?<!of )(?<!all )(?<!for )(?<!serve )(?<!serving )(?<!protect )(?<!protecting )(?<!private )"
                        r"(?<!everyday )(?<!ordinary )(?<!elderly )(?<!older )(?<!concerned )(?<!their )(?<!help )"
                        # civic-education wording ('shaping global citizens', 'raising informed citizens')
                        r"(?<!global )(?<!informed )(?<!engaged )(?<!responsible )"
                        r"\bcitizens?\b(?! of (?!the U\.?S|the United States|America|the US\b))|\breside (?:in|within|inside) (?:the )?(?:U\.?S\.?A?\b|US\b|United States|America|country)|"
                        r"\b(?:non-?)?(?:U\.?S\.?|US|American) residen(?:ts?|cy|ce)\b|\bresiden(?:ts?|cy|ce) (?:in|of) (?:the )?(?:U\.?S\.?\b|US\b|United States)|"
                        r"\b(?:from|in) (?:any|all|other) (?:country|countries|nations?)\b|\bany country\b|\boutside (?:of )?the (?:U\.?S\.?\b|US\b|United States)", re.I)
ANY_OF = re.compile(r"\b(?:or|either|one of|any of|any one|and/or|alternatively)\b|/", re.I)


def alt_phrased(group, text: str) -> bool:
    """True when a sentence names two or more of the group's alternatives joined by or/either/one of."""
    for sent in re.split(r"(?<=[.!?:;])\s+|\n", text):
        names = [x for x in group if has_phrase(sent, x)]
        if len(names) >= 2 and ANY_OF.search(sent):
            return True
    return False


def multi_groups(card: dict) -> list:
    return [g for g in card.get("must_skills", ()) if not isinstance(g, str) and len(g) >= 2]


# Scope words that turn a stance into a stricter one ("unrestricted" or "full" authorization shuts
# out OPT and CPT students). The first part is visibility.PERMANENT on main (commit bee68ca).
SCOPE_WORDS = re.compile(r"unrestricted|permanent(ly)? (work |employment )?authori|authori[sz]ed to work .{0,40}permanent|"
                         r"\bfull (?:work |employment )?authori[sz]ation|without (?:any )?restriction|"
                         r"regardless of (?:their |your |an applicant's |the applicant's )?(?:current )?(?:immigration|visa|citizenship) status|"
                         r"any (?:immigration|visa|work authori[sz]ation) status", re.I)
# 'needs' and 'requirements' alone are nouns in duty sentences ("the specific needs of different species",
# "customer requirements"); only the verb forms and "requirements for / include / are" mark a requirement.
# 'requiring' marks one only before the applicant's ability ("using Zoom, requiring comfort and competence with
# this digital platform", rq_01180), not a task's need ("tickets requiring escalation in Jira").
REQ_MARK = re.compile(r"\b(?:must|required|requires?|requirements? (?:for|include|are)|need(?:s)? to|you(?:['’]ll| will)? need|"
                      r"requiring (?:(?:a|an|strong|solid|real|full|some|both|your)[ ,]+(?:and )?){0,3}(?:comfort|competence|proficiency|"
                      r"familiarity|fluency|knowledge|expertise|experience|skill|mastery|command|ability)|"
                      r"will need|needed (?:to|for)|candidates? needs?|applicants? needs?|proficien(?:cy|t) (?:in|with)|experience (?:with|in|using)|"
                      r"necessary|essential|mandatory|minimum|qualifications?|demands?|should (?:be|have|possess)|"
                      r"looking for (?:candidates|someone|applicants|people)|demonstrated|proven|"
                      # the applicant profile of a requirements paragraph ('Successful applicants will demonstrate a solid
                      # knowledge of history', rq_00475); under a preferred heading it stays preferred (PREF_HEADING)
                      r"(?:successful|ideal|qualified) (?:candidate|applicant)s? will (?:have|demonstrate|possess|bring|show))\b", re.I)
PREF_MARK = re.compile(r"\b(?:prefer\w*|nice[- ]to[- ]have|bonus|a plus|is a plus|ideally|desir\w+|helpful|an advantage|advantageous|"
                       r"ideal (?:candidate|applicant)s? would (?:have|bring|possess))\b", re.I)
REQ_HEADING = re.compile(r"require|qualif|must|what you(?:['’]ll)? need|you(?:['’]ll)? need|what you bring|you have|who you are|"
                         r"about you|skills|competenc|looking for|essential", re.I)
PREF_HEADING = re.compile(r"prefer|nice|bonus|plus|desired|ideal|helpful", re.I)


# A heading that opens the duties, never a requirements list, even right after a requirement sentence.
DUTY_HEAD = re.compile(r"responsib|\bdut(?:y|ies)\b|day[- ]to[- ]day|what you(?:['’]ll| will)? do|\btasks?\b|typical (?:day|week)|"
                       r"a day in|\bweek\b|job involves|work looks like|the work|the role|the job|this role|this job|your role|"
                       r"how you(?:['’]ll| will)? spend|"
                       + "|".join(re.escape(o.lower().rstrip(":")) for o in DUTY_OPENERS), re.I)


def _sentences_with_heading(text: str) -> list[tuple[str, str]]:
    """(sentence, current heading) for the posting body. A paragraph too long to be a heading that ends in ':' leads
    into a list ('Technical competencies are essential. The required skills for this position include:', rq_00693):
    its last sentence joins the heading for the '-' items right after it, and the heading alone holds again after the list.
    A short heading line ending in ':' right after a requirement sentence ('... all candidates must possess specific
    foundational knowledge.' then 'Successful applicants will have:', rq_00437) leads its list the same way, unless it
    opens the duties or the preferred skills (DUTY_HEAD, PREF_HEADING)."""
    out, head, lead, prev = [], "", "", ""
    for ln in textclean.strip_markdown(text).split("\n"):
        s = ln.strip()
        if not s:
            continue
        if not s.startswith("-"):
            lead = ""
        if not s.startswith("-") and (s.endswith(":") or textclean._is_heading(s)) and len(s) <= 70:
            head = s
            if not s.endswith(":") or len(s) < 40:
                if (s.endswith(":") and prev and REQ_MARK.search(prev) and not PREF_MARK.search(prev)
                        and not DUTY_HEAD.search(s) and not PREF_HEADING.search(s)):
                    lead = prev
                prev = ""
                continue
        sents = re.split(r"(?<=[.!?;])\s+", s)
        for sent in sents:
            out.append((sent, f"{head} | {lead}" if lead else head))
        if not s.startswith("-") and s.endswith(":") and len(s) > 70:
            lead = sents[-1]
            if (len(sents) > 1 and len(lead) < 40 and REQ_MARK.search(sents[-2]) and not PREF_MARK.search(sents[-2])
                    and not DUTY_HEAD.search(lead) and not PREF_HEADING.search(lead)):
                lead = f"{sents[-2]} {lead}"  # '... must possess specific knowledge. In particular:'
        prev = "" if s.startswith("-") else sents[-1]
    return out


def required_as_duty_only(card: dict, text: str) -> list:
    """Required skill groups that no sentence states as a requirement: every sentence naming the
    group lacks a requirement word (and sits under no requirement heading) or is preferred."""
    sents = _sentences_with_heading(text)
    bad = []
    for g in card.get("must_skills", ()):
        names = [g] if isinstance(g, str) else list(g)
        ok = False
        for sent, head in sents:
            if not any(has_phrase(sent, n) for n in names):
                continue
            lead = head.split(" | ", 1)[1] if " | " in head else ""  # a list's lead-in sentence (_sentences_with_heading)
            pref = (PREF_MARK.search(sent) or PREF_HEADING.search(head) and not REQ_HEADING.search(head.split("pref")[0])
                    and not (lead and REQ_MARK.search(lead) and not PREF_MARK.search(lead)))
            if (REQ_MARK.search(sent) or REQ_HEADING.search(head) and not PREF_HEADING.search(head)
                    or lead and REQ_MARK.search(lead) and not PREF_MARK.search(lead)) and not pref:
                ok = True
                break
        if not ok:
            bad.append(g)
    return bad


# The applicant described as a student or graduate (not the pupils of a school employer).
# The employer's E-Verify enrollment is no student statement ('is enrolled in the E-Verify program', rq_01186~sponsors);
# 'Enrollment in a bachelor's degree program' is one (visibility.ENROLLMENT_IN, the label side's wording).
STUDENT_TOPIC = re.compile(r"recent (?:college )?grad\w*|new grads?\b|\b(?:current|enrolled|university|college) students?\b|"
                           r"\bstudents? or (?:recent )?grad\w*|\b(?:candidate|applicant)s? (?:is|are|must be|should be) (?:a )?students?\b|"
                           rf"ideal for students|\bcurrently (?:enrolled{NOT_E_VERIFY}|pursuing)\b|"
                           r"\benrolled in (?!(?:the |an? )?(?:federal )?e-?verify\b)(?:a |an )?(?:[\w'’-]+ ){0,2}"
                           rf"(?:program|degree|college|university)\b|{ENROLLMENT_IN}", re.I)
GRADUATE_TOPIC = re.compile(r"\b(?:bachelor['’]?s?|master['’]?s?|associate['’]?s?|doctoral|college|university) degree\b|"
                            r"\bdegree (?:in|from)\b|\bgraduates? (?:of|from)\b|\bPh\.?\s?D\b|\bdoctorate\b|\b(?:graduate|advanced|undergraduate) "
                            r"degree\b|\bhigh school (?:diploma|education|graduate)\b|(?-i:\bGED\b)", re.I)
TRAVEL = re.compile(r"\btravel\w*\b", re.I)
# A subject the degree (or the program) must be in. Only a card with degree_fields may say one: a card without them accepts
# any field, so a text that asks for one carries a requirement its label never read. The writer turns required skills
# into degree subjects ('Your degree program must be in one of the following fields: Sociology, Philosophy, or Counseling'
# for required skills Sociology, Philosophy and Counseling, rq_00437; 5 of 38 check postings) and words it many ways
# ('Your course of study must be in actuarial science', rq_00357~unstated). The capitalised branch is case-sensitive, so
# 'a degree in Geography' counts and 'a degree in order to' does not; the lowercase ones skip a stoplist of words that
# start no subject ('a degree in hand', 'enrolled in an accredited program').
_NOT_SUBJECT = (r"(?:any|a|an|the|order|addition|progress|hand|place|good|which|all|their|your|our|one|this|that|related|"
                r"relevant|similar|lieu|or|and|to|it|its|each|every|some|such|process|accordance|conjunction|combination|"
                r"question|line)\b")
_NOT_PROGRAM = (r"(?:degree|bachelor['’]?s?|master['’]?s?|graduate|undergraduate|accredited|academic|doctoral|associate['’]?s?|"
                r"four|two|higher|college|university|formal|full|part|qualifying|relevant|related|post|current|similar|"
                r"certificate|diploma|eligible|approved|recognized|recognised|educational|internship|training|study|"
                r"apprenticeship|credit|school|program)\b")
_NOT_MAJORS = (r"(?:All|Any|Some|Most|Many|Other|Related|Relevant|Similar|Both|These|Those|The|Our|Your|Their|Different|"
               r"Various|Multiple|Several|Select|Eligible|Approved|Qualifying|Undergraduate|Graduate|Academic|Oil|Energy|Tech)\b")
_NOT_MAJORS_LOW = (r"(?:all|any|some|most|many|other|related|relevant|similar|both|these|those|the|our|your|their|different|"
                   r"various|multiple|several|select|eligible|approved|qualifying|undergraduate|graduate|academic|oil|energy|"
                   r"tech|of|and|or|from|across|with|including|for|to|in|industry|technical|specific|particular|certain|major|minor)\b")
FIELD_SAID = re.compile(
    r"\b(?:degree(?! of\b)|major|studies|study(?<!to study)(?<!will study))\b[^.\n]{0,25}\b(?:in|of|within) "
    r"(?:one of the following "
    r"(?:fields|disciplines|areas)|the (?:fields?|discipline|area|subject) of)"
    r"|(?:one of the following|one of these|any of these|following|these) (?:academic )?(?:fields|disciplines|majors)\b"
    r"|\b(?:field|area) of (?:study|studies)\b[^.,;\n]{0,20}\b(?:must|should|needs? to|is required|required|has to)"
    r"|\b(?:course|program|programme|area|field|line) of stud(?:y|ies)\b[^.,;\n]{0,20}\b(?:must|should|needs? to|has to|"
    r"aligns?|match(?:es)?|relates?|fits?)\b"
    r"|\bstudies\b[^.,;\n]{0,12}\b(?:must|should)\b[^.,;\n]{0,20}\b(?:be in|align|match|relate|fit)"
    r"|\bcourse of study\b[^.,;\n]{0,12}\b(?:is|in)\b"
    r"|\b(?:degree|major|studies|academic (?:focus|background|training|discipline|preparation))\b (?:program )?(?:must|should|needs? to) "
    r"be (?:in|within|related)"
    r"|\b(?:your|their|whose) (?:field|area) of study is\b|\b(?:these|those) (?:same )?(?:academic )?fields\b"
    r"|\b(?:degree|major)\b in (?:either|one of)\b"
    r"|\bmajor(?:ing|s|ed)? in\b"
    r"|\b(?:degree|major|studies|study)\b[^.\n]{0,15}\bin (?:a |an )?(?:related|relevant|similar|technical|scientific|"
    r"appropriate|applicable|quantitative|corresponding) (?:field|discipline|area|subject)"
    rf"|\benrolled in (?:a |an )(?!{_NOT_PROGRAM})[a-z][\w-]*(?: [a-z][\w-]*)? (?:program|programme|major)\b"
    rf"|\b(?:students?|you|are|is|be) (?:currently )?studying (?!(?:toward|towards|for|at|in|on|under|with|abroad|full|part|"
    rf"to|or|and)\b|{_NOT_SUBJECT})[a-z]"
    rf"|\b(?:degree|major|bachelor['’]?s|master['’]?s|doctorate|ph\.?\s?d\.?) in (?!{_NOT_SUBJECT})[a-z]"
    rf"|\b(?:bachelor|master|associate|doctor)(?:['’]?s)? of (?:applied )?(?:science|arts) in (?!{_NOT_SUBJECT})[a-z]"
    rf"|(?-i:\b(?!{_NOT_MAJORS})[A-Z][a-z]+(?: (?:and )?[A-Z][a-z]+)? majors\b|\b[Aa]n? [A-Z][a-z]+(?: (?:and )?[A-Z][a-z]+)? major\b)"
    rf"|(?-i:\b(?!{_NOT_MAJORS_LOW})[a-z]+ majors\b)"
    r"|(?-i:\b(?:[Dd]egree|B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?|BSc|MSc|[Ss]tudying|[Mm]ajor) in "
    r"(?!(?:Fall|Spring|Summer|Winter|Autumn|May|June|December|20\d\d)\b)[A-Z][a-z]+)", re.I)
# 'any field' (and 'no particular major') names no subject: taken out before FIELD_SAID reads a text, so 'a bachelor's
# degree in any field of study is required' and 'students majoring in any discipline' pass.
ANY_FIELD = re.compile(r"\b(?:in |of )?(?:any|no (?:particular|specific|single|one)) (?:academic )?(?:field|discipline|subject|"
                       r"area|major)s?(?: area)?(?: of stud(?:y|ies))?\b", re.I)


def field_said(text: str) -> bool:
    """The text names a subject for the degree or the program, or says the field must match (FIELD_SAID)."""
    return FIELD_SAID.search(ANY_FIELD.sub(" ", textclean.strip_markdown(text))) is not None


ENROLLED_MARK = re.compile(ENROLLED, re.I)  # visibility._enrollment's
SPONSOR_OFFERED = re.compile(r"\bwill sponsor|\bwe sponsor|offers? (visa )?sponsorship|sponsorship (is )?available", re.I)


def posting_missing_facts(card: dict, text: str) -> list[str]:
    """Card facts the label rule reads that the posting does not show (the probes of
    matcher.visibility, so a text that passes here passes assemble_training's missing_facts on the
    posting side): every name of every required group and every preferred skill, the license, the
    years, both pay figures, every location's city, the work mode, the job type, the industry, the
    degree level and every field of study, the student requirement and each work-authorization
    statement. The context lists each of them; a text without one would carry a label its posting
    does not support."""
    out = []
    if any(not has_phrase(text, n) for g in card.get("must_skills", ()) for n in ([g] if isinstance(g, str) else g)):
        out.append("missing_required_skill")
    if any(not has_phrase(text, s) for s in card.get("nice_skills", ())):
        out.append("missing_preferred_skill")
    if card.get("license") and not has_phrase(text, card["license"]):
        out.append("missing_license")
    if card.get("min_years") and not has_years(text, card["min_years"]):
        out.append("missing_years")
    if any(v is not None and not has_money(text, v) for v in (card.get("salary_min"), card.get("salary_max"))):
        out.append("missing_pay")
    if any(not has_phrase(text, x.split(",")[0].strip()) for x in card.get("locations", ())):
        out.append("missing_location")
    if not re.search(_MODE[card.get("work_mode", "onsite")], text, re.I):
        out.append("missing_work_mode")
    if not re.search(_TYPE[card.get("employment_type", "full_time")], text, re.I):
        out.append("missing_job_type")
    if card.get("industry") and not has_industry(text, card["industry"]):
        out.append("missing_industry")
    if card.get("degree_level", "none") != "none" and not re.search(LEVEL_WORDS[card["degree_level"]], text, re.I):
        out.append("missing_degree_level")
    if any(not has_phrase(text, f) for f in card.get("degree_fields", ())):
        out.append("missing_field_of_study")
    if card.get("enrolled_in", "none") != "none" and not ENROLLED_MARK.search(text):
        out.append("missing_student_requirement")
    s = card.get("sponsorship", "unstated")
    if s in WORK_AUTH_TEXT and not re.search(r"sponsor", text, re.I):
        out.append("missing_sponsorship")
    if s == "will_not_sponsor_or_opt" and not re.search(r"\bOPT\b", text):
        out.append("missing_opt")
    if s == "unstated" and SPONSOR_OFFERED.search(text):
        out.append("sponsorship_added")
    if card.get("citizens_only") and not re.search(r"citizen|permanent resident|green card", text, re.I):
        out.append("missing_citizens_only")
    if card.get("e_verify") and not re.search(r"e-?verify", text, re.I):
        out.append("missing_e_verify")
    return out


# Absence gates: a hard fact the posting states that the card does not have (the posting side of the resume's
# extra_tool / extra_license / years_overstated). The label never saw it, so a posting with it is labelled
# against a requirement the model can read and the rule cannot.
MONEY = re.compile(r"(?:\$|\bUSD\s?)\s?(\d[\d,]*(?:\.\d+)?)(\s?[kK]\b)?(?!\s*(?:million|billion|mm\b|[MB]\b))|"
                   # no dollar sign: 'Pay: 20 an hour', '45k to 55k per year' (not '401k', not '5k run')
                   r"(?<![\d$.,])\b(\d{1,3}(?:\.\d{1,2})?)(?:\s?dollars)?\s?(?:an|per|/|each)\s?(?:hour|hr)\b|"
                   r"(?<![\d$.,])\b(\d{2,3}(?:\.\d)?)\s?[kK]\b(?=\s*(?:-|–|to)\s*\$?\d{2,3}(?:\.\d)?\s?[kK]\b|"
                   r"\s*(?:per|a|an|/)\s*(?:year|yr|annum)\b|\s*(?:annually|salary|base)\b)", re.I)


def money_figures(text: str) -> list[float]:
    out = []
    for m in MONEY.finditer(text):
        if m.group(1):
            v = float(m.group(1).replace(",", ""))
            out.append(v * 1000 if m.group(2) else v)
        elif m.group(3):
            out.append(float(m.group(3)))
        else:
            out.append(float(m.group(4)) * 1000)
    return out


def pay_added(card: dict, text: str) -> list[float]:
    """Dollar figures that are not the card's pay (yearly, or hourly at 2,080 hours for the hourly contexts);
    any dollar figure when the card has no pay. Budgets in millions do not count."""
    allowed = [v for v in (card.get("salary_min"), card.get("salary_max")) if v is not None]
    allowed += [v / 2080 for v in allowed]
    return [v for v in money_figures(text) if not any(abs(v - a) <= max(0.01, 0.01 * a) for a in allowed)]


def years_added(card: dict, text: str) -> list[int]:
    """Stated 'N years' numbers other than the card's minimum ('3-5 years' for 3, 'two years' for none)."""
    y = int(card.get("min_years") or 0)
    return [n for n in stated_years(text) if n != y]


# 'Prior experience in a similar setting is required' with no number: an experience requirement the card (no
# minimum years) does not have. Experience with one of the card's skills is the skill requirement, not this.
_EXP_REQUIRED = re.compile(r"\b(?:prior|previous|relevant|related|professional|work|hands-on|industry) (?:[\w-]+ ){0,3}?experience\b"
                           r"([^.;\n]{0,50}?)\b(?:is|are) (?:strictly |absolutely )?(?:required|necessary|a must|mandatory|needed|essential)\b|"
                           r"\bmust (?:have|possess|bring) (?:some |prior |previous |relevant |related )(?:[\w-]+ ){0,3}?experience\b()", re.I)
_EXP_NEG = re.compile(r"\b(?:no|not|n't|never|without|while)\b", re.I)


def experience_added(card: dict, text: str) -> list[str]:
    if card.get("min_years"):
        return []
    skills = [n for g in card.get("must_skills", ()) for n in ([g] if isinstance(g, str) else g)] + list(card.get("nice_skills", ()))
    out = []
    for m in _EXP_REQUIRED.finditer(text):
        sent = _sentence_at(text, m.start())
        gap = m.group(1) or ""
        if _EXP_NEG.search(sent) or stated_years(sent) or any(has_phrase(gap, s) for s in skills):
            continue
        out.append(m.group())
    return out


# A sentence that states a requirement or a preference (the words, a heading or a 'Requirements:' label);
# duty sentences are not.
REQ_LABEL = re.compile(r"^\W*(?:[A-Za-z]+ ){0,2}(?:requirements?|qualifications?|must[- ]haves?)\s*:", re.I)


def _req_or_pref_sentences(text: str) -> list[str]:
    return [s for s, head in _sentences_with_heading(text)
            if REQ_MARK.search(s) or PREF_MARK.search(s) or REQ_LABEL.search(s) or REQ_HEADING.search(head)
            or PREF_HEADING.search(head)]


POSTING_ADDON_LICENSES = (*ADDON_LICENSES, "PMP", "CompTIA", "Security+", "driver's license", "driving license",
                          "clearance", "background check", "drug screen", "TS/SCI", "Public Trust")
_CRED_NOUN = re.compile(r"\b(?:licen[sc]es?|licensure|certifications?|certificates?|credentials?|clearances?|"
                        r"background checks?|drug (?:screen|test)\w*)\b", re.I)


# Not a credential the applicant holds: the degree itself ('in addition to these educational credentials'), the
# employer's tools ('we provide the necessary software licenses') and the closing line's 'we look forward to
# reviewing your credentials' (rq_01223~citizens).
_NOT_CRED = re.compile(r"\b(?:educational|academic|education|degree|school|university|college|doctoral|doctorate|graduate|"
                       r"master['’]?s|bachelor['’]?s|associate['’]?s)(?: and \w+)? credentials?\b|"
                       r"\b(?:review|reviewing|consider|considering|evaluate|evaluating|learning more about) (?:your|their) "
                       r"(?:(?:application|resume|experience|background)(?:,| and| or)? ){0,3}credentials\b|"
                       r"\b(?:software|product|tool|seat|user) licen[sc]es?\b|\blicen[sc]es? (?:and|for) (?:the )?(?:software|hardware|tools?)\b", re.I)
_BARE_CRED = re.compile(r"credentials?", re.I)
# The applicant's record, not a license: 'credentials' as the thing they send in ('submit your credentials for review',
# rq_00868) or the employer reviews. Anywhere else a bare 'credentials' is a license or certification requirement
# ('must hold professional credentials in nursing', 'must hold relevant credentials').
_RECORD_CRED = re.compile(r"\b(?:submit|send|provide|attach|upload|forward|share|include|enclose)\w*\s+(?:(?:your|their|all|any|"
                          r"the|a copy of|copies of|supporting|relevant)\s+){0,3}credentials?\b|"
                          r"\bcredentials?\s+(?:for|to)\s+(?:our\s+|your\s+)?(?:review|verification|evaluation)\b", re.I)
_CRED_NAMED = re.compile(r"\b((?:[A-Z][\w+/-]*[ ]){0,2}[A-Z][A-Z0-9+/-]{1,7})[ ]+(?i:certifi\w*|licen[sc]\w*|credential\w*|clearance)")


def license_added(card: dict, text: str, vocab: list[str] | None = None) -> list[str]:
    """Credentials a requirement or preference sentence names that the card does not have: a license name of
    any card or a common add-on (CPR, a driver's license, a clearance, a background check), or any sentence
    with a credential noun ('PMP certification is required') that does not name the card's license."""
    global _LICENSE_VOCAB
    if vocab is None:
        _LICENSE_VOCAB = _LICENSE_VOCAB if _LICENSE_VOCAB is not None else _license_vocab()
        vocab = _LICENSE_VOCAB
    have = _license_names(card["license"]) if card.get("license") else []
    out = []
    for s in _req_or_pref_sentences(text):
        s = _NOT_CRED.sub(" ", s)
        if NEG_LICENSE.search(s) and not card.get("license"):
            continue  # neg_license's
        hits = [x for x in (*vocab, *POSTING_ADDON_LICENSES) if has_phrase(s, x)
                and not any(has_phrase(x, h) or has_phrase(h, x) for h in have)]
        if not hits and _CRED_NOUN.search(s) and not any(has_phrase(s, h) for h in have):
            # a licensed card's own license is often 'licensure' or 'your certification': there only a named
            # credential ('PMP certification') counts
            named = [m.group(1) for m in _CRED_NAMED.finditer(s) if not any(has_phrase(h, m.group(1)) for h in have)]
            # a bare 'credentials' the applicant sends in is their record, no license ('submit your credentials for
            # review', rq_00868; 'educational and professional credentials', rq_01152~sponsors, is _NOT_CRED's); a bare
            # 'credentials' they must hold is a license requirement ('professional credentials in nursing')
            record = all(_BARE_CRED.fullmatch(m.group()) for m in _CRED_NOUN.finditer(s)) and _RECORD_CRED.search(s)
            hits = named if have or record else [s[:80]]
        out += hits
    return out


def extra_required_skills(card: dict, text: str, vocab: list[str] | None = None) -> list[str]:
    """Tools a requirement or preference sentence names that are not among the card's skills
    (extra_tools on those sentences)."""
    skills = [n for g in card.get("must_skills", ()) for n in ([g] if isinstance(g, str) else g)]
    skills += list(card.get("nice_skills", ()))
    sents = _EXCEL_VERB.sub(lambda m: (m.group(1) or "") + "do well", "\n".join(_req_or_pref_sentences(text)))
    return extra_tools({"skills": skills}, sents, vocab)


# 'excel' the verb, not the tool ('help you excel in creating', 'to excel at'): lower case (no re.I), before
# in/at/as/when/while/within or after to/you/will/can/who.
_EXCEL_VERB = re.compile(r"\bexcel(?:s|led|ling)?\b(?=\s+(?:in|at|as|when|while|within)\b)|"
                         r"\b((?:to|you|will|can|who) )excel(?:s|led|ling)?\b"
                         r"(?!\s+(?:spreadsheets?|skills?|formulas?|files?|workbooks?|macros?|pivot|vba|software|proficiency))")


# A time scope on a sponsorship statement: 'does not require employer sponsorship now or in the future' (rq_01180)
# shuts out an OPT holder who needs an H-1B later, which the card's 'won't sponsor' does not (the rule reads
# will_not_sponsor as a hard stop for needs_h1b_now only).
SPONSOR_TIME = re.compile(r"\b(?:now or in the future|(?:currently|presently) or in the future|future\b|"
                          r"now or later|at any (?:time|point)|at a later (?:date|time|point)|going forward|long[- ]term|"
                          r"(?<!what)ever\b(?!-))", re.I)


def sponsor_time_scope(card: dict, text: str) -> list[str]:
    """Work-authorization sentences with a time scope the card does not give: on a refusal stance or a citizens-only card,
    a later scope ('now or in the future', 'current or future visa support', rq_00319, rq_00608), with any work-status word
    (sponsorship, a visa, a work permit, work authorization), not only 'sponsor'; on any card with a stance, 'at this
    time' or 'currently unable' ('We are unable to sponsor visas ... at this time', 14 citizens-only texts), which says the
    stance may change. A sponsoring card's 'now or in the future' narrows nothing and is left alone."""
    refusal = card.get("citizens_only") or str(card.get("sponsorship", "")).startswith("will_not")
    has_wa = refusal or card.get("sponsorship") == "sponsors" or card.get("e_verify")
    out = []
    for s in re.split(r"(?<=[.!?;])\s+|\n", text):
        if not WORK_STATUS.search(s):
            continue
        if refusal and SPONSOR_TIME.search(s) or has_wa and AT_THIS_TIME.search(s):
            out.append(s[:100])
    return out


WORK_STATUS = re.compile(r"sponsor|\bvisas?\b|work permits?|(?:work|employment) authori[sz]\w*|immigration", re.I)
AT_THIS_TIME = re.compile(r"\bat (?:this|the present|the current|present)(?: time| moment)?\b|\bat the moment\b|\bfor now\b|"
                          r"\bcurrently (?=(?:unable|not able|cannot|can't|does not|do not|doesn't|don't|will not|won't|not)\b)",
                          re.I)


# A refusal wider than the card's: 'without the need for visa sponsorship or student-based work authorization
# programs' (rq_00693~no_opt) also refuses CPT, which no stance refuses (the rule keeps a CPT intern eligible when the
# posting is silent on it; a posting that refuses CPT is a citizens-only case).
STUDENT_AUTH = re.compile(r"(?-i:\bCPT\b)|curricular practical|\bstudent[- ](?:based (?:work |visa|authori|program)|visas?\b|"
                          r"work (?:authori[sz]|permit|visa)\w*)|"
                          r"\bstudent visas?\b|\bF-?1 (?:status|visas?|visa holders?|students?(?! (?:on|using|through|with) (?:STEM )?OPT))\b|"
                          r"\binternational students?\b|\b(?:any|all) (?:forms? of )?(?:student|temporary) (?:work )?authori[sz]\w*", re.I)
_REFUSAL = re.compile(r"\b(?:not|no|cannot|can't|unable|won't|will not|without|nor|ineligible|exclud\w*|does not|do not|"
                      r"neither)\b", re.I)


def student_auth_refused(card: dict, text: str) -> list[str]:
    """Work-authorization sentences that refuse a student status the card's stance does not name (STUDENT_AUTH next
    to a refusal word). 'F-1 OPT' and 'STEM OPT' are the will_not_sponsor_or_opt stance's own words."""
    out = []
    for s in re.split(r"(?<=[.!?;])\s+|\n", text):
        if STUDENT_AUTH.search(s) and _REFUSAL.search(s):
            out.append(s[:100])
    return out


# A place on a license the card names without one: 'you must be authorized to practice in Florida' for 'RN license'
# (rq_00817) is a Florida license, which a nurse licensed elsewhere does not hold. A 'State teaching license' names the
# state of the job: the job's own state is that license.
def license_scope_added(card: dict, text: str) -> list[str]:
    from matcher.metros import STATE_NAMES as CODE_NAME

    lic = card.get("license") or ""
    if not lic:
        return []
    states = rf"(?:{STATE_NAMES})"
    rx = re.compile(rf"\b(?:authori[sz]ed|licensed|eligible|able|permitted|qualified|certified) to (?:practice|work as)\b[^.\n]{{0,40}}?"
                    rf"\bin (?:the state of )?(?P<a>{states})\b|\blicen[sc]\w*\b[^.\n]{{0,25}}?\b(?:in|by|from|of|with) (?:the )?"
                    rf"(?:state of |State of )?(?P<b>{states})\b|\b(?P<c>{states})(?: [Ss]tate)?(?: [A-Z]{{2,5}}| [a-z]+)? licen[sc]|"
                    rf"\b(?:compact|multi-?state|multistate) (?:\w+ )?licen[sc]|\bNurse Licensure Compact\b")
    own = {CODE_NAME.get(x.rsplit(",", 1)[-1].strip().lower(), "").lower() for x in card.get("locations", ()) if "," in x}
    out = []
    for m in rx.finditer(text):
        st = (m.group("a") or m.group("b") or m.group("c") or "").lower()
        if st and re.search(r"\bstate\b", lic, re.I) and st in own:
            continue
        if st and re.search(rf"\b{re.escape(st)}\b", lic, re.I):
            continue
        out.append(m.group()[:80])
    return out


# A level on a required or preferred skill the card names without one: 'Show advanced ability in using Microsoft Word'
# (rq_01297), 'Advanced skills in Microsoft Excel' (rq_01384). 'Proficiency', 'strong', 'solid' and 'experience with'
# stay allowed.
SKILL_LEVEL = r"(?:advanced|expert(?:-level)?|extensive|senior[- ]level|in-depth|exceptional|superior|mastery|master-level)"
# the same level as a noun next to the skill: 'Deep learning expertise is a strict requirement' (rq_00897), 'expertise in
# SQL', 'mastery of Python'
SKILL_LEVEL_AFTER = r"(?:expertise|mastery)"


def skill_level_added(card: dict, text: str) -> list[str]:
    names = [n for g in card.get("must_skills", ()) for n in ([g] if isinstance(g, str) else g)] + list(card.get("nice_skills", ()))
    names = sorted({n for x in names for n in (x, skill_name(x))}, key=len, reverse=True)
    out = []
    for s in _req_or_pref_sentences(text):
        for n in names:
            if re.search(SKILL_LEVEL, n, re.I):
                continue
            m = re.search(rf"\b{SKILL_LEVEL}\b(?:[ ,]+[\w'’/-]+){{0,4}}?[ ]+(?:of |in |with |using |in using )?{re.escape(n)}\b", s, re.I) \
                or re.search(rf"\b{re.escape(n)}[ ]+{SKILL_LEVEL_AFTER}\b|\b{SKILL_LEVEL_AFTER}[ ]+(?:in|with|of|on)[ ]+{re.escape(n)}\b", s, re.I)
            if m and not re.search(r"[.;:]", m.group()):
                out.append(m.group()[:80])
                break
    return out


# Years of experience in a named domain: 'at least seven years of professional experience in an educational or related
# field' (rq_01297) narrows the card's plain minimum, which counts any work. A relevant / related / similar field or
# role, and experience in a card skill, are the card's own.
_YEARS_DOMAIN = re.compile(r"\byears?\b(?:\s+of)?\s+(?:[\w-]+\s+){0,3}?experience\s+(?P<q>(?:in|within|across|as)\s+[^.;:\n]{1,70})", re.I)
_PLAIN_DOMAIN = re.compile(r"(?:in|within|across)\s+(?:any|all)\s+(?:kinds?|types?|lines?|forms?|sorts?|fields?|industr(?:y|ies)|"
                           r"sectors?|jobs?|roles?|occupations?|settings?|areas?|capacit(?:y|ies)|work|positions?)\b|"
                           r"(?:in|within|across)\s+(?:a|an|the|any)?\s*(?:(?:relevant|related|similar|comparable|applicable|"
                           r"professional|equivalent|appropriate)(?:\s+or\s+(?:relevant|related|similar|comparable))?\s+)+"
                           r"(?:fields?|capacit(?:y|ies)|roles?|positions?|settings?|areas?|environments?|work|jobs?|disciplines?)\b"
                           r"|(?:in|within)\s+(?:this|the|a|an)\s+(?:field|role|position|job|area)\b", re.I)


def years_domain_added(card: dict, text: str) -> list[str]:
    if not card.get("min_years"):
        return []
    skills = [n for g in card.get("must_skills", ()) for n in ([g] if isinstance(g, str) else g)] + list(card.get("nice_skills", ()))
    out = []
    for m in _YEARS_DOMAIN.finditer(text):
        q = m.group("q")
        if _PLAIN_DOMAIN.match(q) or any(has_phrase(q[:60], n) or has_phrase(q[:60], skill_name(n)) for n in skills):
            continue
        out.append(m.group()[:90])
    return out


# Sentences that describe the applicant ('The ideal candidate will demonstrate strong communication skills', rq_00475):
# a skill named there is asked for, like one under a requirements heading.
PROFILE_MARK = re.compile(r"\bideal (?:candidate|applicant)s?\b|\bsuccessful (?:candidate|applicant)s?\b|\byou should\b|"
                          r"\bwe (?:seek|are seeking|look for|are looking for|value|want)\b|\bour team values\b|"
                          r"\bcandidates? (?:who|with|will)\b|\bapplicants? (?:who|with|will)\b", re.I)


def _profile_sentences(text: str) -> list[str]:
    return [s for s, _ in _sentences_with_heading(text) if PROFILE_MARK.search(s)]


# A skill asked of the applicant: an ability noun with the skill as its object ('a strong foundation in data analysis',
# rq_01362; 'demonstrated skills in data analysis'; 'background in counseling and psychology'), or the skill before one
# ('data analysis skills'). A skill named as a tool's purpose ('SPSS for statistical analysis'), the company's mission or
# a department ('our public safety mission', 'our human resources department') is no ask.
_ABILITY = (r"(?:foundation|background|grounding|experience|skills?|proficiency|knowledge|expertise|familiarity|competenc[ey]|"
            r"abilit(?:y|ies)|capabilit(?:y|ies)|exposure|training|understanding|command|fluency|mastery|aptitude)")


# What the job gives, not what it asks ('gain hands-on experience in the public safety sector', 'designed to provide
# experience in human resources functions'); a skill name that names a tool kind ('data analysis tools, specifically
# Microsoft Excel or SPSS') is the card's tools.
_GIVEN = re.compile(r"\b(?:gain\w*|provid\w*|offer\w*|build\w*|develop\w*|get|getting|receiv\w*|earn\w*|detailing|describing)\b"
                    r"(?:\s+[\w'’-]+){0,3}\s*$", re.I)


def _asked_skill(sent: str, name: str) -> bool:
    n = r"[\W_]+".join(re.escape(w) for w in name.split())
    for m in re.finditer(rf"\b{_ABILITY}\s+(?:in|with|of|on|using)\s+(?:(?!(?:for|to|the company|our)\b)[\w'’-]+,?\s+){{0,3}}?{n}\b"
                         rf"(?!\s+(?:tools?|software|platforms?|programs?|applications?|systems?)\b)|"
                         rf"\b{n}\s+(?:skills?|experience|expertise|knowledge|background|abilit(?:y|ies)|proficiency)\b", sent, re.I):
        if not _GIVEN.search(sent[:m.start()]):
            return True
    return False


def extra_plain_skills(card: dict, text: str) -> tuple[list[str], list[str]]:
    """(skills, soft skills) that a requirement, preference or applicant-profile sentence asks of the applicant and the
    card does not list: a non-tool skill of the training vocabulary asked as an ability (_asked_skill; 'a strong
    foundation in data analysis', rq_01362), and a soft skill named at all ('strong communication skills', rq_00475).
    The rule never counts a soft skill, so that second list is reported, not fatal. Duty sentences are left alone."""
    from matcher.schema import skill_canon

    global _POSTING_SKILLS
    _POSTING_SKILLS = _POSTING_SKILLS if _POSTING_SKILLS is not None else _posting_skill_vocab()
    have = [n for g in card.get("must_skills", ()) for n in ([g] if isinstance(g, str) else g)] + list(card.get("nice_skills", ()))
    have_k = {skill_canon(x) for x in (*have, *card.get("degree_fields", ()))}  # a degree subject is the card's own
    sents = list(dict.fromkeys(_req_or_pref_sentences(text) + _profile_sentences(text)))
    clean = []
    for s in sents:
        for n in have:  # a card skill's own name, with the words that describe it ('Linux operating systems', 'SAS
            # statistical software'), is no other skill
            for x in (n, skill_name(n)):  # 'Supervisory control and data acquisition (SCADA) software' is the card's name
                name = r"[\W_]+".join(re.escape(w) for w in x.split())
                s = re.sub(rf"{name}(?:\s+(?!(?:and|or|for|to|with|in|of|as|is|are|at|on)\b)[a-z][\w-]*){{0,2}}", " ", s, flags=re.I)
        clean.append(s)
    hard = [x for x in _POSTING_SKILLS if skill_canon(x) not in have_k and x not in LIST_ONLY_SKILLS
            and any(_asked_skill(s, x) for s in clean)]
    soft = [x for x in SOFT_PLAIN if skill_canon(x) not in have_k and any(has_phrase(s, x) for s in clean)]
    return hard, soft


# Soft skills as a posting writes them (schema.SOFT_SKILLS keys; the rule never counts them, but a posting that asks
# for one the card does not have states a requirement its label never read).
SOFT_PLAIN = ("communication skills", "leadership", "teamwork", "problem solving", "problem-solving", "time management",
              "attention to detail", "interpersonal skills", "collaboration skills")


_NEGATED = re.compile(r"\b(?:not|no|non|never|isn't|is not|cannot|can't|without|rather than|instead of)\b[^.\n]{0,25}$", re.I)
_NEGATED_AFTER = re.compile(r"^[^.;\n]{0,40}?\b(?:(?:is|are)(?: not|n't) (?:available|offered|possible|permitted|allowed|an option|supported)|"
                            r"not (?:available|offered|possible|an option))", re.I)
MODE_SAID = {
    "remote": re.compile(r"\bremote (?:position|role|job|work|opportunity|basis|option|setting|environment|arrangement|schedule|"
                         r"employment|employee|days?|team member)\b|\b(?:work|working) remotely\b|\bfully remote\b|"
                         r"\b100% remote\b|\bremote[- ](?:first|friendly|eligible)\b|\bwork[- ]from[- ]home\b|"
                         r"\btelecommut\w*|\bWFH\b|\b(?:role|position|job) is remote\b|\bis (?:a )?remote\b|\bwork[- ]from[- ]anywhere\b", re.I),
    "hybrid": re.compile(r"\bhybrid\b(?!\s+(?:cloud|vehicles?|cars?|seeds?|corn|IT|infrastructure|apps?|mobile|integrated|"
                         r"solutions?|systems?|methods?|approach|models? of care|courses?|classes|learning|instruction|meetings?|"
                         r"events?|conferences?|sessions?))|"
                         # part of the week in the office: 'from our office three days a week', '2 days per week on-site'
                         r"\b(?:[1-4]|one|two|three|four) days? (?:a|per|each) week (?:in|at|from) (?:the|our) office\b|"
                         r"\b(?:in|at|from) (?:the|our) office (?:[1-4]|one|two|three|four) days? (?:a|per|each) week\b|"
                         r"\b(?:[1-4]|one|two|three|four) days? (?:a|per|each) week (?:on[- ]?site|in[- ]office|in person)\b", re.I),
    # the job's own place, not a duty ('engage remote and on-site employees', 'refer patients for in-person care')
    "onsite": re.compile(r"\b(?:on[- ]?site|in[- ]person|in[- ]office) (?:position|role|job|work|presence|requirement|schedule|basis|"
                         r"days?|attendance)\b|\b(?:fully|100%|entirely|exclusively|based|located|work|working|report) (?:on[- ]?site|in[- ]office|"
                         r"in the office|in person)\b|\b(?:is|are|be) (?:an )?(?:on[- ]?site|in[- ]office|in[- ]person)\b|"
                         r"\bwork (?:from|at|in) (?:our|the) (?:office|offices)\b(?! (?:[1-4]|one|two|three|four) days?)", re.I),
}
FULLY = {"hybrid": re.compile(r"\b(?:fully|100%|entirely|completely) (?:remote|on[- ]?site|in[- ]office|in[- ]person)\b", re.I)}


def _said(rx: re.Pattern, text: str) -> bool:
    return any(not _NEGATED.search(text[max(0, m.start() - 40):m.start()]) and not _NEGATED_AFTER.search(text[m.end():m.end() + 80])
               for m in rx.finditer(text))


def mode_conflict(card: dict, text: str) -> list[str]:
    """Work modes the posting states other than the card's (a hybrid card may name both of its parts,
    but not 'fully remote'); a negated mention ('this is not a remote role') is not one."""
    mode = card.get("work_mode", "onsite")
    if mode == "hybrid":
        return ["fully"] if _said(FULLY["hybrid"], text) else []
    return [m for m, rx in MODE_SAID.items() if m != mode and _said(rx, text)]


TYPE_SAID = {
    # not a past internship ('prior internship experience', 'moved beyond internships')
    "internship": re.compile(r"(?<!prior )(?<!previous )(?<!past )(?<!beyond )(?<!completed )(?<!including )(?<!completed an )(?<!completed a )\binternships?\b(?! experience)|"
                             r"\bco-?op (?:position|role|program|term)\b", re.I),
    # not a later job ('convert to full-time employment', 'potential for future full-time roles')
    "full_time": re.compile(r"(?<!convert to )(?<!transition to )(?<!advancement to )(?<!future )(?<!into )"
                            r"\bfull[- ]?time\b(?! (?:students?|enrollment|enrolled|study|studies|program))", re.I),
    "part_time": re.compile(r"\bpart[- ]?time\b(?! (?:students?|enrollment|enrolled|study|studies|program))", re.I),
    "contract": re.compile(r"\bcontract (?:position|role|job|basis|assignment|employee|opportunity|work)\b|\bcontract[- ]to[- ]hire\b|"
                           r"\b\d+[- ]month contract\b|\btemporary (?:position|role|job|assignment|basis)\b|\btemp[- ]to[- ]hire\b|"
                           r"\bfixed[- ]term\b|\bseasonal (?:position|role|job|employment|assignment|basis|hire)\b", re.I),
}


# 'Full-time' / 'part-time' that is not this job's type: the pay conversion the context itself asks for ('per hour
# (about $X a year)' -> 'about $65,000 a year when prorated for full-time equivalent work'), a later career
# ('prepare you for a future career as a full-time technologist') or the other staff ('as our full-time staff').
_HOURS_TYPE = re.compile(r"\$|\bUSD\b|equivalen\w*|prorat\w*|annuali[sz]ed|\bannual\b|\ba year\b|\bper year\b|\bper annum\b|"
                         r"based on (?:a |an )?(?:standard |typical |regular )?(?:full|part)[- ]?time\b", re.I)
_LATER_TYPE = re.compile(r"\b(?:future|eventual|later|potential|possible|convert\w*|transition\w*|path(?:way)?s?|lead\w* to|"
                         r"prepare\w*|after (?:graduation|the internship)|upon (?:graduation|completion))\b[^.\n]{0,40}$", re.I)
_TYPE_JOB = re.compile(r"\s+(?:paid |summer )?(?:position|role|job|internship|employment|opportunity|employee|hire|contract)\b", re.I)
_OTHER_STAFF = re.compile(r"^[^.\n]{0,3}(?:staff|employees|colleagues|counterparts|team members|workers|coworkers|co-workers)\b", re.I)
# 'A full-time internship' (rq_00433) is a job type of its own: candidate cards want full-time, part-time, internship or
# contract work (employment_types_wanted), and an internship card gives no hours. (Until the r6 sample read this was let
# through as the internship's hours; that reader found it in 3 of 120 texts and it now counts.)


def _sentence_at(text: str, i: int) -> str:
    a = max(text.rfind("\n", 0, i), *(text.rfind(p, 0, i) for p in (". ", "! ", "? ")))
    ends = [j for j in (text.find("\n", i), *(text.find(p, i) for p in (". ", "! ", "? "))) if j >= 0]
    return text[a + 1:min(ends) if ends else len(text)]


def _type_said(kind: str, rx: re.Pattern, text: str, card_type: str = "") -> bool:
    for m in rx.finditer(text):
        if _NEGATED.search(text[max(0, m.start() - 40):m.start()]) or _NEGATED_AFTER.search(text[m.end():m.end() + 80]):
            continue
        if card_type == "internship" and re.fullmatch(r"fixed[- ]term", m.group(), re.I):
            continue  # an internship is fixed-term by nature ('join our development team for a fixed term', rq_01362)
        if kind in ("full_time", "part_time"):
            job = _TYPE_JOB.match(text, m.end())  # 'a full-time internship paying $20 an hour' is the job's type
            if (not job and _HOURS_TYPE.search(_sentence_at(text, m.start())) or _LATER_TYPE.search(text[max(0, m.start() - 60):m.start()])
                    or re.search(r"\b(?:our|other|fellow|the) $", text[max(0, m.start() - 7):m.start()], re.I)
                    and _OTHER_STAFF.search(text[m.end():m.end() + 20])):
                continue
        return True
    return False


def type_conflict(card: dict, text: str) -> list[str]:
    """Job types the posting states other than the card's ('part-time' in a full-time posting, 'full-time'
    in an internship, 'a full-time internship' too); 'full-time students', a full-time pay conversion, a later full-time
    job and, on an internship card, 'a fixed-term internship' are not one."""
    t = card.get("employment_type", "full_time")
    return [k for k, rx in TYPE_SAID.items() if k != t and _type_said(k, rx, text, t)]


STATE_NAMES = ("Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|"
               "Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|"
               "Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|"
               "Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|"
               "West Virginia|Wisconsin|Wyoming|D\\.?C\\.?")
CITY_STATE = re.compile(rf"\b((?:[A-Z][a-z.'-]+[ ]){{0,2}}[A-Z][a-z.'-]+),[ ]*(?:(?-i:{textclean._STATES})\b|(?:{STATE_NAMES})\b)")


_WORD_STATES = {"OR", "IN", "ME", "OK", "HI"}  # also words: 'Microsoft Excel, OR manage our social media'
_CITIES: set[str] | None = None


def _known_cities() -> set[str]:
    global _CITIES
    if _CITIES is None:
        _CITIES = {x.split(",")[0].strip().lower() for kind, f in (("postings", "locations"), ("resumes", "current_location"))
                   for c in load_cards(kind).values() for x in ([c.get(f)] if isinstance(c.get(f), str) else c.get(f) or [])}
    return _CITIES


def location_added(card: dict, text: str, cities: set[str] | None = None) -> list[str]:
    """'City, ST' / 'City, State' places the posting names that are not among the card's locations (with OR, IN,
    ME, OK or HI only for a city some card has)."""
    have = {x.split(",")[0].strip().lower() for x in card.get("locations", ())}
    out = []
    for m in CITY_STATE.finditer(text):
        city, st = m.group(1), m.group(0).rsplit(",", 1)[1].strip()
        if any(city.lower().endswith(h) or h.endswith(city.lower()) for h in have):
            continue
        if st in _WORD_STATES and not any(city.lower().endswith(c) for c in (cities if cities is not None else _known_cities())):
            continue
        out.append(city)
    return out


def _sents(text: str) -> list[str]:
    return [x for x in re.split(r"(?<=[.!?;])\s+|\n", text) if x.strip()]


# A work-authorization scope wider than the card's stance. will_not_sponsor is a hard stop for needs_h1b_now only: OPT
# and STEM OPT get a soft check (rules._check_work_auth), and STEM OPT needs the employer's E-Verify and its I-983
# training plan. 'This policy applies to all candidates regardless of their current location or status' (rq_00868),
# 'We are unable to assist with immigration processes' (rq_01180) and 'We are unable to assist with work authorization
# processes, so candidates should ensure they have the necessary rights to work independently' (rq_00957) refuse what
# an OPT or STEM OPT hire needs, the old OPT / I-983 over-generalisation (rq_00693~no_opt). An equal-opportunity line
# ('regardless of race, color ... or veteran status') is no stance. 'Legal right to work in the United States' alone is
# no widening: an OPT holder has it.
_EEO = re.compile(r"\b(?:race|colou?r|religion|sex|gender|age|disabilit\w*|veterans?|orientation|national origin|marital|"
                  r"genetic|ancestry|creed|pregnan\w*|identity|expression)\b", re.I)
# (an educational or enrollment status is the student gate's, ENROLL_SAID)
REGARDLESS_STATUS = re.compile(r"\bregardless of\b(?![^.\n]{0,40}?\b(?:educational|enrollment|academic|student)\b)[^.\n]{0,50}?"
                               r"\bstatus(?:es)?\b", re.I)
# (no visa: refused help with a visa is the refused sponsorship of the stance itself, 'we cannot assist with visa procurement')
AUTH_PROCESS = re.compile(r"\b(?:immigration|work[- ]authori[sz]ation|employment[- ]authori[sz]ation|work[- ]permit)\s+"
                          r"(?:process\w*|paperwork|procedures?|matters|applications?|filings?|petitions?|documentation|support|"
                          r"assistance|help|services|steps|cases?|needs)\b|"
                          r"\b(?:assist\w*|help\w*|support\w*|aid\w*|guid\w*|facilitat\w*|handl\w*|process\w*)\b(?:\s+[\w'’-]+){0,3}?"
                          r"\s+(?:with|in|through|on|for)\s+(?:[\w'’-]+\s+){0,3}?(?:immigration|work[- ]authori[sz]ation|"
                          r"employment[- ]authori[sz]ation)\b", re.I)
STEM_OPT_NEEDS = re.compile(r"\bI-?983\b|\btraining plans?\b", re.I)
WORK_ON_OWN = re.compile(r"\b(?:rights?|authori[sz]ation|eligib\w+|permission) to work (?:independently|on (?:their|your) own)\b|"
                         r"\bauthori[sz]ed to work (?:independently|immediately|on (?:their|your) own)\b|"
                         r"\bwithout (?:any )?(?:employer|company|our|its) (?:involvement|assistance|support|help|participation)\b", re.I)


def auth_scope_added(card: dict, text: str) -> list[str]:
    """Sentences that widen the card's work-authorization scope: 'regardless of ... status' (not an equal-opportunity
    list), refused help with immigration or work-authorization processes, a refused I-983 / training plan (STEM OPT's
    employer part; the will_not_sponsor_or_opt stance's own), and work authorization the hire must use on their own or at
    once. A citizens-only card is left out: no help for a non-citizen is what it already means."""
    if card.get("citizens_only"):
        return []
    out = []
    for s in _sents(text):
        if REGARDLESS_STATUS.search(s) and not _EEO.search(s):
            out.append(s[:120])
        elif _REFUSAL.search(s) and (AUTH_PROCESS.search(s) or STEM_OPT_NEEDS.search(s)
                                     and card.get("sponsorship") != "will_not_sponsor_or_opt"):
            out.append(s[:120])
        elif WORK_ON_OWN.search(s):
            out.append(s[:120])
    return out


# Several locations are places the job is based: the rule matches a candidate on any one of them (rules.matched_location).
# 'splitting time between our offices in Dublin, OH and Kissimmee, FL' (rq_00646) turns 'either city' into 'both' and a
# hybrid job into office-to-office travel. A hybrid job split between one office and home is the card's own.
_SPLIT = re.compile(r"\b(?:split\w*|divid\w*|rotat\w*|alternat\w*|shar\w* (?:your|their|his or her) time|travel\w*|commut\w*|"
                    r"across)\b", re.I)
_ONE_OF = re.compile(r"\b(?:one of|either|any (?:one )?of|whichever|choice of|choose)\b", re.I)
_SITES = r"(?:offices|locations|sites|campuses|cities|branches|facilities|clinics|hospitals|labs|stores|plants|warehouses)"
SPLIT_SITES = re.compile(rf"\b(?:split\w*|divid\w*|rotat\w*|alternat\w*|travel\w*|commut\w*)\b[^.\n]{{0,30}}\bbetween (?:(?:our|the|both|"
                         rf"these|two|its|all|three|several|multiple)\s+)*(?:[\w-]+\s+)?{_SITES}\b"
                         # 'between our on-site locations and remote duties' is office and home
                         rf"(?!\s+(?:and|&)\s+(?:[\w-]+\s+){{0,2}}(?:remote\w*|home|virtual\w*|telework\w*|off-?site))|"
                         rf"\b(?:at|in|across|between) (?:both|each of (?:our|the|these|its)|all (?:of )?(?:our|the|these|its)) "
                         rf"(?:[\w-]+\s+)?{_SITES}\b|\ball (?:of )?(?:our|the|its|these) (?:[\w-]+\s+)?{_SITES}\b|"
                         rf"\bbetween (?:both|the two|our two) {_SITES}\b|\btravel\w* between\b", re.I)


def locations_split(card: dict, text: str) -> list[str]:
    """For a card with two or more locations: sentences that put the person in more than one of them (split, divide,
    rotate or travel between the card's cities or 'our offices', work at both / each of them)."""
    locs = [x.split(",")[0].strip() for x in card.get("locations", ())]
    if len(locs) < 2:
        return []
    out = []
    for s in _sents(text):
        if _ONE_OF.search(s) or any(re.search(rf"\b{re.escape(a)}\b[^.\n]{{0,30}}\bor\b[^.\n]{{0,15}}\b{re.escape(b)}\b", s)
                                    for a in locs for b in locs if a != b):
            continue  # 'one of our Texas locations: Austin or Houston' is the card's reading
        named = [x for x in locs if has_phrase(s, x)]
        between = any(re.search(rf"\bbetween\b[^.\n]{{0,40}}\b{re.escape(x)}\b", s) for x in named)
        if SPLIT_SITES.search(s) or len(named) >= 2 and (_SPLIT.search(s) or between):
            out.append(s[:120])
    return out


# No card limits how recently a degree was finished: 'designed for individuals who have recently completed their
# undergraduate education' (rq_01399, a finished bachelor's) adds a recent-graduate rule the label never read. On a
# student card a 'recent graduates' door is a second, wrong one.
RECENT_GRAD = re.compile(r"\brecent(?:ly)?\s+(?:graduat\w*|grads?\b|complet\w*|finish\w*|earn\w*|obtain\w*|receiv\w*|conferr\w*)"
                         r"(?:\s+[\w'’-]+){0,4}?\s+(?:degrees?|bachelor\w*|master\w*|associate\w*|undergraduate|graduate|studies|"
                         r"education|diplomas?|coursework)\b|\brecent (?:college |university |school )?graduat\w*|\brecent grads?\b|"
                         r"\bnew(?:ly)?[- ](?:graduates?|graduated|grads?)\b|\bfresh (?:graduates?|grads?)\b|"
                         r"\bjust (?:graduated|finished (?:school|college|university|(?:a|your|their) degree)|completed (?:a|your|their) degree)\b|"
                         r"\bgraduated (?:with)?in the (?:last|past)\b", re.I)


def recent_grad_added(card: dict, text: str) -> list[str]:
    return [s[:120] for s in _sents(text) if RECENT_GRAD.search(s)]


# The applicant's whole education described as over: 'designed for individuals who have completed their formal
# education' (rq_00980, rq_00539, rq_00454), 'who have finished their academic training' (rq_00559), 'who have already
# graduated'. No card says that: the rule only ever reads a degree level, which a student pursuing a higher degree
# already holds (a master's student meets a finished-bachelor's internship), and a student card asks for the opposite.
# So the wording is fatal on every card. A completion that names its level ('completed a bachelor's degree', 'completed
# their undergraduate education', 'graduates from any discipline', 'who have graduated with a bachelor's') is the
# card's degree requirement, not this.
_EDU_ADJ = r"(?:formal|academic|foundational|full|entire|educational|school|scholastic)"
EDUCATION_DONE = re.compile(
    rf"\b(?:completed|finished|concluded)\s+(?:all\s+(?:of\s+)?)?(?:their|your|his or her|his/her|one['’]s)\s+"
    rf"(?:(?:{_EDU_ADJ}\s+){{0,2}}(?:education|studies|schooling)\b|"
    rf"(?:{_EDU_ADJ}\s+){{1,2}}(?:training|journey|groundwork|requirements|preparation|careers?|programs?|path)\b)|"
    r"\b(?:who|that|once you|after you|you)\s+(?:have\s+|has\s+|had\s+)?(?:already\s+)?graduated\b"
    r"(?!\s+(?:with|from|in|by)\b)|"
    r"\bgraduates only\b|\bonly (?:open to |for )?graduates\b(?!\s+(?:of|from|with|in)\b)|"
    r"\bno longer (?:a\s+)?(?:students?|enrolled|in school|studying|attending)\b|\bout of school\b|\bleft school\b|"
    r"\bdone with (?:school|their studies|your studies)\b", re.I)


def education_done_added(card: dict, text: str) -> list[str]:
    return [s[:120] for s in _sents(text) if EDUCATION_DONE.search(s)]


# Enrollment or student status on a card with no student rule, in any wording, a negation included: 'We do not require
# current enrollment in a school program' (rq_00633), 'no restrictions based on student status' (rq_00105). STUDENT_TOPIC
# reads the positive forms; this reads the rest. The employer's E-Verify enrollment, benefits enrollment and the pupils'
# enrollment a school employer manages ('student enrollment records') are no statement about the applicant.
_ENROLL_NOT_APPLICANT = (r"(?!\s+(?:records?|data|process\w*|numbers?|figures?|management|services?|office|forms?|reports?|growth|"
                         r"trends?|targets?|goals?|events?|periods?|systems?|documentation|paperwork|files?|counts?|rates?|"
                         r"campaigns?|activities|requests?|applications?|inquir\w*|support|tasks?|procedures?|drives?|levels?|"
                         r"specialists?|coordinators?|staff|team)\b)")
ENROLL_SAID = re.compile(
    rf"\b(?:current|active|school|student|academic|university|college|program|degree)[- ]enrollment\b{_ENROLL_NOT_APPLICANT}|"
    rf"\benrollment (?:in|at) (?:a |an |any )?(?:(?!benefits?|health|retirement|insurance|401)[\w'’-]+ ){{0,2}}?(?:schools?|degree|"
    rf"college|university|courses?|institution|studies|classes|(?:school|degree|academic|study|educational|university|college) "
    rf"programs?)\b|"
    r"\bstudent (?:status|requirements?)\b|\b(?:limited|restricted) to (?:current |enrolled )?students\b|"
    r"\b(?:do(?:es)? not|don't|need not|no|not|without|nor) (?:[\w'’-]+ ){0,3}?(?:enrol(?:led|lment)|be (?:a )?students?|"
    rf"currently studying){NOT_E_VERIFY}\b{_ENROLL_NOT_APPLICANT}|"
    r"\bregardless of [^.\n]{0,25}\b(?:student|enrollment|academic|educational) (?:enrollment )?status\b", re.I)


def student_said(card: dict, text: str) -> str | None:
    """The first sentence about the applicant as a student or graduate on a card with no student requirement (STUDENT_TOPIC,
    ENROLL_SAID; GRADUATE_TOPIC too when the card has no degree either); None when the card has a student rule."""
    if card.get("enrolled_in", "none") != "none":
        return None
    if (m := STUDENT_TOPIC.search(text) or ENROLL_SAID.search(text)):
        return _clip(_sentence_at(text, m.start()))
    if card.get("degree_level", "none") == "none" and not card.get("degree_fields") and (m := GRADUATE_TOPIC.search(text)):
        return _clip(_sentence_at(text, m.start()))
    return None


# A fully remote card with no locations names no place: 'from any location within the United States' (rq_00566) and
# 'from any location within the country' (rq_01384) add a country limit the rule never reads (location_ok is true for any
# remote card). A work-authorization sentence ('authorized to work within the United States') is that gate's, not this.
REMOTE_REGION = re.compile(rf"\b(?:within|anywhere in|from anywhere in|from any (?:location|place|state|city|part|where) (?:in|within|of|across)|"
                           rf"located in|based in|resid(?:e|ing) in|liv(?:e|ing) in|residents? of)\s+(?:the\s+)?"
                           rf"(?:continental |contiguous |mainland )?(?:United States|U\.?S\.?A?\b|US\b|USA\b|America\b|country\b|nation\b|"
                           rf"lower 48|(?:{STATE_NAMES})\b|North America|Canada|Europe|EU\b|EMEA|(?:Eastern|Central|Mountain|Pacific) "
                           rf"(?:time|standard)|(?:EST|CST|MST|PST|ET|CT|MT|PT)\b)|"
                           rf"\b(?:U\.?S\.?|US|USA)[- ]based\b", re.I)
_REMOTE_WORK = re.compile(r"\b(?:remote\w*|work\w*|location|based|resid\w*|live|living|from home|telecommut\w*|anywhere)\b", re.I)
_AUTH_SENT = re.compile(r"sponsor|authori[sz]|eligib|\bvisas?\b|citizen|permanent resid|green card|e-?verify", re.I)


def remote_region_added(card: dict, text: str) -> list[str]:
    if card.get("work_mode") != "remote" or card.get("locations"):
        return []
    return [s[:120] for s in _sents(text) if REMOTE_REGION.search(s) and _REMOTE_WORK.search(s) and not _AUTH_SENT.search(s)]


def _card_skill_names(card: dict) -> tuple[list[str], list[str]]:
    must = [n for g in card.get("must_skills", ()) for n in ([g] if isinstance(g, str) else g)]
    return must, list(card.get("nice_skills", ()))


# A trait asked of the applicant: 'a proactive approach to safety ... is essential' (rq_00526), 'The fully remote nature
# of the role requires self-discipline' (rq_01009), 'Our ideal candidate is detail-oriented' (rq_00065~citizens). No card
# has these as a requirement (a soft skill on a card is a preferred skill, and the rule never counts one), so a posting
# that makes one a must states a requirement its label never read. A card's own soft skill may be named; as a must only
# when the card has it as a must. A thing that is organized ('an organized inventory') is no trait of the applicant.
# Reported, not fatal (NONFATAL): the label does not depend on it.
TRAIT = re.compile(r"\b(?:detail[- ]oriented|attention to detail|meticulous\w*|"
                   r"(?<!system )(?<!systems )(?<!infrastructure )(?<!network )(?<!service )(?<!equipment )(?<!grid )"
                   r"(?<!product )(?<!data )(?<!supply )(?<!process )(?<!operational )(?<!power )(?<!site )(?<!asset )"
                   r"(?<!drive )(?<!improve )(?<!enhance )(?<!ensure )(?<!efficiency and )(?<!safety and )"
                   r"(?:reliability|dependability)(?! (?:of|in|within|across|for) (?:the |our |its )?(?!(?:team|you)\b)[a-z])|"
                   r"(?:be|is|are|remain|stay|being|someone|person|individual|candidate|professional|team member)"
                   r" (?:highly |very |consistently |)(?:reliable|dependable|resilient)\b|"
                   r"(?:strong |solid )?work ethic|self[- ]disciplin\w*|self[- ]motivat\w*|self[- ]starter|punctual\w*|patience|"
                   r"calm under pressure|composure|proactive|adaptab(?:le|ility)|adapt quickly|"
                   r"(?:highly |well[- ]|be |stay |remain |are |is )organi[sz]ed\b(?!\s+(?:workspace|work ?space|inventory|records?|"
                   r"files?|environment|area|storage|manner|system|way))|organi[sz]ational skills|enjoys? the process|etiquette|"
                   r"professionalism|eager(?:ness)? to learn|willingness to learn|positive attitude|resilience(?! of)|diligen(?:ce|t))\b",
                   re.I)
TRAIT_FRAME = re.compile(r"\b(?:must|requires?|required|demands?|essential|necessary|necessity|"
                         r"should (?:be|have|possess|demonstrate|bring)|ideal (?:candidate|applicant)s?|"
                         r"looking for (?:someone|a candidate|an individual|individuals|candidates|people|a person)|"
                         r"successful (?:candidate|applicant)s? (?:will|should|must|is|are)|crucial|paramount|non-negotiable|"
                         r"indispensable|expected to (?:be|have|demonstrate|show|bring))\b", re.I)
_TRAIT_MUST = re.compile(r"\b(?:must|requires?|required|demands?|essential|necessary|necessity|crucial|paramount|non-negotiable|"
                         r"indispensable)\b", re.I)


def trait_required(card: dict, text: str) -> list[str]:
    must, nice = _card_skill_names(card)
    out = []
    for s in _sents(text):
        if not TRAIT_FRAME.search(s) or re.match(r"\W*if you\b", s, re.I):
            continue  # 'If you have the required background and are eager to learn, apply' invites, it asks nothing
        for m in TRAIT.finditer(s):
            w = re.sub(r"^(?:(?:highly|well[- ]|be|stay|remain|are|is|being|someone|person|individual|candidate|professional|"
                       r"team member) )+(?:(?:highly|very|consistently) )?", "", m.group(), flags=re.I)
            if any(has_phrase(x, w) or has_phrase(w, x) for x in must):
                continue
            if any(has_phrase(x, w) or has_phrase(w, x) for x in nice) and not _TRAIT_MUST.search(s):
                continue  # the card's preferred soft skill, named without a must
            out.append(s[:160])
            break
    return out


# A working condition or hard requirement no card has: 'reliable transportation to our location is necessary'
# (rq_00754), 'The role requires standing for long periods' (rq_01446~no_sponsor), a background check, a drug test,
# fingerprinting (a driver's license or a clearance is license_added's). And a subject foundation on an any-field degree: 'While your degree may be in
# any discipline, you must possess a solid foundation in the life sciences' (rq_00193).
CONDITION = re.compile(r"\b(?:reliable|dependable|own|personal) (?:transportation|vehicle|car)\b|\bdriving record\b|"
                       r"\blift(?:ing)? (?:up to |at least )?\d+\s*(?:lbs?|pounds)\b|"
                       r"\b(?:heavy lifting|lift(?:ing)? heavy)\b|\bstand(?:ing)? (?:for )?(?:long|extended|prolonged) (?:periods|hours)\b|"
                       r"\bon (?:your|their) feet\b|\bphysical(?:ly)? (?:stamina|demanding|fit|fitness|requirements?|endurance)\b|"
                       r"\bbackground (?:check|screening|investigation)s?\b|\bdrug (?:test\w*|screen\w*)\b|"
                       r"\bfingerprint\w*\b", re.I)  # a driver's license or a clearance is license_added's
_COND_FRAME = re.compile(r"\b(?:must|required?|requires|requirements?|necessary|essential|mandatory|need(?:ed|s)? to|"
                         r"expected to|should be able|ability to|able to|condition of|subject to|demands?|involves?|may require|"
                         r"comfortable)\b", re.I)
ANY_FIELD_FOUNDATION = re.compile(r"\b(?:foundation|background|grounding|footing|training|coursework|knowledge|understanding)\s+"
                                  r"(?:in|of)\s+(?!(?:any|this|the role|the job|your|their|our|a related|related)\b)"
                                  r"(?:the\s+)?(?P<x>[a-z][\w-]*(?:\s+[a-z][\w-]*){0,2}\s+(?:sciences?|studies|principles|"
                                  r"concepts|theory|fundamentals))\b", re.I)


OPEN_FIELD = re.compile(r"\bany (?:academic )?(?:field|discipline|subject|major|area of study)\b|\b(?:discipline|field|major|subject)\b"
                        r"[^.\n]{0,40}\b(?:is |are |remains? |can be |may be )(?:open|flexible|any)\b|\bregardless of (?:your |the )?"
                        r"(?:major|field|discipline|academic background)\b", re.I)


def condition_added(card: dict, text: str) -> list[str]:
    cj = json.dumps(card).lower()
    out = []
    for s, head in _sentences_with_heading(text):
        m = CONDITION.search(s)
        if m and (_COND_FRAME.search(s) or REQ_HEADING.search(head or "")) and m.group().split()[0].lower() not in cj:
            out.append(s[:160])
    has_degree = card.get("degree_level", "none") != "none" or card.get("enrolled_in", "none") != "none"
    if has_degree and not card.get("degree_fields"):
        for s in _sents(text):
            m = ANY_FIELD_FOUNDATION.search(s)
            if m and OPEN_FIELD.search(s) and not any(
                    has_phrase(m.group("x"), n) or has_phrase(n, m.group("x")) for n in sum(_card_skill_names(card), [])):
                out.append(s[:160])
    return out


# Hours and pay terms a candidate card can match on (employment_types_wanted, salary_floor) that the card does not give:
# flexible hours on a part-time job ('suitable for those seeking flexible scheduling', rq_00586), standard business hours,
# weekend or shift rotations (rq_01221~no_opt), and a pay range made conditional ('pay depends on experience',
# 'commensurate with qualifications', rq_00114~unstated). The pay conversion the context asks for ('about $X a year based
# on a full-time schedule', '40 hours per week') is the card's pay, not a term.
FLEX_HOURS = re.compile(r"\bflexib\w* (?:hours|schedul\w*|work hours|working hours|timing|availability)\b|"
                        r"\bset (?:your|their) own (?:hours|schedule)\b|\bschedule (?:is|can be) flexible\b|"
                        r"\bhours (?:are|can be) flexible\b", re.I)
SCHEDULE = re.compile(r"\b(?:standard |regular |normal |typical |core )?business hours\b|"
                      r"\bweekends?\b(?! (?:events?|programs?|camps?|workshops?|activities))|\bevenings?\b(?! (?:events?|classes|programs?))|"
                      r"\bovernight\b|\b(?:rotating|night|evening|weekend|overnight|early|late|split|12-hour|8-hour|10-hour|double) shifts?\b|"
                      r"\bshift work\b|\b(?:shift|weekend|schedule|on-call|call) rotations?\b|\bovertime\b|\bon[- ]call\b|"
                      r"\b(?:\d{1,2}|twenty|thirty|forty|fifty)[- ]hours? (?:per|a|each|every) week\b|"
                      r"\b(?:\d{1,2}|forty|thirty)[- ]hour (?:work ?)?week\b|\bMonday (?:through|to|-|–) Friday\b|"
                      r"\b9(?: ?a\.?m\.?)? ?(?:-|–|to) ?5(?: ?p\.?m\.?)?\b", re.I)
_SCHEDULE_DUTY = re.compile(r"\b(?:scheduling|schedules (?!(?:include|includes|are|is|will|may)\b)|manag\w*|coordinat\w*|planning|"
                            r"assign\w*|covering|staffing|overse\w*|organi[sz]\w*)\s+"
                            r"(?:\w+\s+){0,2}?(?:shifts?|rotations?|on-call|weekend)\b|\b(?:crop|job|staff|clinical|patient|"
                            r"inventory|stock|product) rotations?\b|\brotation of\b", re.I)
# The hours a duty is done for customers in are the store's, not the worker's schedule: 'assist customers during business
# hours' (rq_00962). Only that span is skipped; any other schedule word in the sentence still counts.
_CUSTOMER_HOURS = re.compile(r"\b(?:assist\w*|serv\w*|help\w*|greet\w*|support\w*|answer\w*)\s+(?:\w+\s+)?(?:customers|"
                             r"patrons|shoppers|guests|visitors|the public)\s+during (?:our |the )?business hours\b", re.I)
PAY_WORD = re.compile(r"\$|\bpay\b|\bsalary\b|\bcompensation\b|\bwages?\b|\bhourly rate\b|\bstipend\b|\bpay rate\b|\bearn\w*\b", re.I)
PAY_TERMS = re.compile(r"\bcommensurate with (?!the (?:responsibilities|scope|duties|level of responsibility|role|position)\b)|"
                       r"\bdepend\w* (?:up)?on\b(?=[^.\n]{0,60}\b(?:experience|qualifications?|performance|skills?|background|"
                       r"education|credentials|candidate|interview|fit)\b)|"
                       r"\b(?:based|determined|set) (?:up)?on (?:the |your |a |an |each )?(?:candidate|applicant|experience|qualifications?|"
                       r"skills?|performance|selected|individual|education|level of|prior|demonstrated|interview)|"
                       r"\bperformance[- ](?:based|related)\b|\bnegotiable\b|\bDOE\b|\b(?:plus|and|with|including) (?:a |an )?"
                       r"(?:performance |annual |signing |sign-on )?(?:bonus\w*|commissions?)\b|\bincentive pay\b", re.I)


def hours_added(card: dict, text: str) -> list[str]:
    out = []
    flexible_ok = card.get("employment_type", "full_time") == "full_time"  # the full-time card's 'flexible' is no type
    for s in _sents(text):
        if not flexible_ok and FLEX_HOURS.search(s):
            out.append(s[:160])
        elif (m := SCHEDULE.search(_CUSTOMER_HOURS.sub(" ", s))) and not _SCHEDULE_DUTY.search(s) and not (
                _HOURS_TYPE.search(s) and re.search(r"hour|week", m.group(), re.I)):
            out.append(s[:160])
    return out


def pay_terms_added(card: dict, text: str) -> list[str]:
    if card.get("salary_min") is None:
        return []  # NEG_PAY's
    return [s[:160] for s in _sents(text) if PAY_WORD.search(s) and PAY_TERMS.search(s)]


# A multi-city card is based in any one of its cities (rules.matched_location), and the text must say so: 'Work will be
# conducted entirely onsite at our locations in Coral Gables, FL and Miami, FL' (rq_01245~no_opt) reads as both. One
# sentence joins the places with 'or' or 'one of / either / any of', or offers openings at them: plural openings or
# something available ('We have openings in Boston, MA and Davis, CA', 'Positions are available at our offices in A and
# B', 'The available cities are A and B', 'available in two locations: A and B') reads as one or the other. The offer
# must be in the clause that leads into the first place, with no 'based' in it ('based at our offices in A and B'
# still reads as both), and the sentence says nothing of both, across or each.
_ABBREV_DOT = re.compile(r"\b(St|Ft|Mt|Pt)\.\s")
_OPENINGS = re.compile(r"\b(?:openings|positions|roles|opportunities|vacancies|jobs|available)\b", re.I)
_BOTH = re.compile(r"\b(?:both|across|each|all)\b", re.I)


def _openings_at(s: str, named: list[str]) -> bool:
    """The clause leading into the first named place offers openings there (see above)."""
    at = [m.start() for x in named if (m := re.search(re.escape(_ABBREV_DOT.sub(r"\1 ", x + " ").strip()), s, re.I))]
    if len(at) < len(named) or _BOTH.search(s):
        return False
    lead = re.split(r"[,;]", s[:min(at)])[-1]
    return bool(_OPENINGS.search(lead)) and not re.search(r"\bbased\b", lead, re.I)


def locations_not_alternatives(card: dict, text: str) -> bool:
    locs = [x.split(",")[0].strip() for x in card.get("locations", ())]
    if len(locs) < 2:
        return False
    plain = _ABBREV_DOT.sub(r"\1 ", text)
    for s in _sents(plain):
        named = [x for x in locs if has_phrase(s, _ABBREV_DOT.sub(r"\1 ", x + " ").strip())]
        if len(named) >= 2 and (_ONE_OF.search(s) or re.search(r"\bor\b", s) or _openings_at(s, named)):
            return False
        if named and _ONE_OF.search(s) or re.search(rf"\b(?:one|any|either) of (?:our |the |these |its |those )?(?:\w+\s+)?{_SITES}\b", s):
            return False
    return True


# A citizens-only card gives no reason for the rule: 'due to the nature of our projects and contractual obligations'
# (rq_00130), 'As a county government position' (rq_00374~citizens), 'restricted by federal regulations'. The rule's
# reason is the restriction itself ('limited to US citizens or permanent residents'); a reason the card does not have
# (export control, a contract, a law) is a fact of its own. A sentence about E-Verify is the card's E-Verify.
CIT_SENT = re.compile(r"citizen|permanent resident|green card|U\.?S\.? persons?|eligib\w+|work authori[sz]\w*|"
                      r"authori[sz]ed to work|restricted|limited to", re.I)
CIT_REASON = re.compile(r"\bdue to\b|\bbecause\b|\bowing to\b|\bin (?:order )?to comply\b|\bcomply with\b|\bcompliance with\b|"
                        r"\b(?:required|mandated|dictated|governed|restricted) by\b|\bcontractual\b|\b(?:our|government|federal|client|its|"
                        r"defen[cs]e) contracts?\b|\bnature of (?:our|the|this)\b|"
                        r"\b(?:federal|state|government(?:al)?|legal|statutory|regulatory) (?:regulations?|mandates?|guidelines|laws?|rules|"
                        r"requirements|obligations|standards|policies|contracts?)\b|\bregulations\b|\bexport\b|\bITAR\b|\bclearance\b|"
                        r"\bsensitive\b|\bclassified\b|\bnational security\b|\bas (?:a|an) (?:county|city|state|federal|government|"
                        r"public|municipal)\b", re.I)


def citizens_reason_added(card: dict, text: str) -> list[str]:
    if not card.get("citizens_only"):
        return []
    return [s[:160] for s in _sents(text.replace("U.S. ", "U.S. ")) if CIT_SENT.search(s) and CIT_REASON.search(s)
            and not re.search(r"e-?verify", s, re.I)]


# A citizens-only card with no sponsorship stance says nothing about sponsoring: 'We are unable to sponsor other visa
# types or work authorizations at this time' (rq_00130) writes will_not_sponsor where the card says unstated.
# Reported, not fatal (NONFATAL): citizens-only already excludes everyone who needs sponsorship, so no label moves.
SPONSOR_REFUSED = re.compile(r"\b(?:unable|cannot|can't|do not|does not|don't|doesn't|won't|will not|not able|not|no longer|never)\b"
                             r"[^.\n]{0,40}?\bsponsor\w*|\bwithout (?:the need for |requiring |needing |any )?(?:\w+ )?sponsorship\b|"
                             r"\bno (?:visa |work visa |employment visa )?sponsorship\b|\bsponsorship (?:is|will) not\b", re.I)


def no_sponsor_added(card: dict, text: str) -> list[str]:
    if not card.get("citizens_only") or card.get("sponsorship") != "unstated":
        return []
    return [s[:160] for s in _sents(text) if SPONSOR_REFUSED.search(s)]


# A hybrid card's remote days have no place of their own: 'completing tasks remotely from a location of your choice'
# (rq_00330) makes the hybrid job location-free.
HYBRID_ANYWHERE = re.compile(r"\b(?:from )?anywhere\b|\bany location\b|\bwherever\b|\blocation of (?:your|their) choice\b|"
                             r"\bany (?:state|city|place)\b", re.I)


def hybrid_anywhere(card: dict, text: str) -> list[str]:
    if card.get("work_mode") != "hybrid":
        return []
    return [s[:160] for s in _sents(text) if HYBRID_ANYWHERE.search(s) and _REMOTE_WORK.search(s) and not _AUTH_SENT.search(s)]


def posting_extra_facts(card: dict, text: str) -> list[str]:
    """Tags for hard facts the posting adds that its card does not have."""
    out = []
    if years_added(card, text):
        out.append("years_added")
    if experience_added(card, text):
        out.append("experience_added")
    if pay_added(card, text):
        out.append("pay_added")
    if license_added(card, text):
        out.append("license_added")
    if extra_required_skills(card, text):
        out.append("extra_required_skill")
    if mode_conflict(card, text):
        out.append("mode_conflict")
    if type_conflict(card, text):
        out.append("type_conflict")
    if location_added(card, text) or remote_region_added(card, text):
        out.append("location_added")
    if locations_split(card, text):
        out.append("locations_split")
    if auth_scope_added(card, text):
        out.append("auth_scope_added")
    if recent_grad_added(card, text):
        out.append("recent_grad_added")
    if education_done_added(card, text):
        out.append("education_done_added")
    if sponsor_time_scope(card, text):
        out.append("sponsor_time_scope")
    if student_auth_refused(card, text):
        out.append("student_auth_refused")
    if license_scope_added(card, text):
        out.append("license_scope_added")
    if skill_level_added(card, text):
        out.append("skill_level_added")
    if years_domain_added(card, text):
        out.append("years_domain_added")
    if trait_required(card, text):
        out.append("trait_required")
    if condition_added(card, text):
        out.append("condition_added")
    if hours_added(card, text):
        out.append("hours_added")
    if pay_terms_added(card, text):
        out.append("pay_terms_added")
    if locations_not_alternatives(card, text):
        out.append("locations_not_alternatives")
    if citizens_reason_added(card, text):
        out.append("citizens_reason_added")
    if no_sponsor_added(card, text):
        out.append("no_sponsor_added")
    if hybrid_anywhere(card, text):
        out.append("hybrid_anywhere")
    hard, soft = extra_plain_skills(card, text)
    if hard:
        out.append("extra_plain_skill")
    if soft:
        out.append("soft_skill_added")
    return out


def posting_problems(card: dict, text: str) -> list[str]:
    out = posting_missing_facts(card, textclean.strip_markdown(text))
    out += posting_extra_facts(card, textclean.strip_markdown(text))
    if multi_groups(card) and not all(alt_phrased(g, text) for g in multi_groups(card)):
        out.append("any_of_as_all_of")
    unset_degree = card.get("degree_level", "none") == "none" and card.get("enrolled_in", "none") == "none"
    if unset_degree and NEG_DEGREE.search(text):
        out.append("neg_degree")
    if not card.get("min_years") and NEG_EXP.search(text):
        out.append("neg_experience")
    if not card.get("license") and NEG_LICENSE.search(text):
        out.append("neg_license")
    has_wa = context_of("postings", card).get("work_authorization")
    if not has_wa and (NEG_AUTH.search(text) or AUTH_TOPIC.search(text)):
        out.append("work_auth_topic_unset" if not NEG_AUTH.search(text) else "neg_work_auth")
    if has_wa and NEG_AUTH.search(text):
        out.append("neg_work_auth")
    if not card.get("e_verify") and re.search(r"e-?verify", text, re.I):
        out.append("e_verify_unset")
    if card.get("salary_min") is None and (NEG_PAY.search(text)):
        out.append("neg_pay")
    if NEG_ANY.search(text):
        out.append("neg_meta")
    if "program_term" not in context_of("postings", card) and re.search(r"\b(?:spring|summer|fall|winter|autumn)\s+20\d\d\b|\b20[2-3]\d\b", text, re.I):
        out.append("year_unset")
    if textclean.has_markdown(textclean.strip_markdown(text)):
        out.append("markdown_after_clean")
    company = textclean.split_synthetic(text)[1]
    if not company:
        out.append("no_company_line")
    elif company != company_name(card):
        out.append("company_not_given")
    if has_wa and not card.get("citizens_only") and SCOPE_WORDS.search(text):
        out.append("scope_word")
    if required_as_duty_only(card, text):
        out.append("required_skill_not_stated_required")
    if student_said(card, text):
        out.append("student_topic_unset")
    if not card.get("degree_fields") and field_said(text):
        out.append("degree_field_added")
    if TRAVEL.search(text) and not TRAVEL.search(json.dumps(card)):
        out.append("travel_added")
    if non_latin_script(card, text):
        out.append("non_latin_script")
    return out


# Every problem tag keeps a text out of training, except leftover markdown (build_prompt strips it again), a soft
# skill the card does not list ('strong communication skills') and a soft trait made a requirement ('attention to detail
# is essential'): both reported, but the rule never counts a soft skill and pairs.INSTRUCTION tells the model soft skills
# do not decide the label, so no label rests on them, and real postings almost always ask for traits (trait_required
# failed 17% of the full render). Also no_sponsor_added: a citizens-only card already excludes everyone who needs
# sponsorship, so a no-sponsorship sentence on it changes no label (as 82efb17 decided for stance_conflict). A tag not
# listed here is fatal too, so a new gate is enforced the day it is added.
NONFATAL = frozenset({"markdown_after_clean", "soft_skill_added", "trait_required", "no_sponsor_added"})
MAX_REDO = 2  # a context whose text fails the gates is rendered again at most twice (3 tries), then dropped


def problems(kind: str, card: dict, text: str) -> list[str]:
    """Every gate tag for one text; postings also get 'stance_conflict' (matcher.visibility)."""
    if kind == "resumes":
        return resume_problems(card, text)
    out = posting_problems(card, text)
    if stance_conflict(card, text):
        out.append("stance_conflict")
    return out


def fatal(kind: str, card: dict, text: str) -> list[str]:
    """The tags that keep this text out of training (empty: the text may be used)."""
    return [t for t in problems(kind, card, text) if t not in NONFATAL]


# ---------------------------------------------------------------- retry feedback
# A context that failed the gates is sent again (at most MAX_REDO times; each retry is a counted try) with one more key,
# FEEDBACK_KEY, appended after the card's own facts: one line per fatal tag of the last failed try, re-gated under the
# current gate code, quoting the offending sentence (or naming the missing fact) and saying in one plain line what the
# card says instead, in the context's own wording. The card's facts are unchanged, and the stored ctx hash is the hash of
# the context without the key (base_ctx), so a retry's text is the card's text like any other.
FEEDBACK_KEY = "fix_from_last_try"
QUOTE_MAX = 200


def _clip(s: str) -> str:
    s = " ".join(s.split())
    return s if len(s) <= QUOTE_MAX else s[:QUOTE_MAX - 3].rstrip() + "..."


def _sentence_of(text: str, snippet: str) -> str:
    """The sentence (or line) of `text` that holds `snippet`; the snippet itself when it is not found verbatim."""
    snippet = snippet.strip().rstrip(".")
    i = text.find(snippet) if snippet else -1
    if i < 0 and snippet:
        i = text.lower().find(snippet.lower())
    return _clip(_sentence_at(text, i) if i >= 0 else snippet)


def _match_sentence(rx: re.Pattern, text: str) -> str | None:
    m = rx.search(text)
    return _clip(_sentence_at(text, m.start())) if m else None


def _names(groups) -> list[str]:
    return [n for g in groups for n in ([g] if isinstance(g, str) else g)]


def _join(xs) -> str:
    xs = [str(x) for x in xs]
    return ", ".join(xs) if xs else "none"


def _wa_line(ctx: dict) -> str:
    wa = ctx.get("work_authorization")
    if not wa:
        return "the card has no work-authorization statement: write nothing about sponsorship, visas, OPT, citizenship or E-Verify"
    return "work_authorization says only: " + " ".join(f"'{x}'" for x in wa) + " (no time scope, no other status, nothing added)"


NON_LATIN_SAYS = ("write in English, in Latin letters only: no Chinese, Japanese, Korean or other non-Latin characters, "
                  "not even one word")


def _resume_findings(card: dict, text: str, tags: list[str]) -> list[dict]:
    ctx = context_of("resumes", card)
    plain = textclean.strip_markdown(text)
    red = textclean.redact(plain)
    skills = list(card.get("skills", ()))
    out = []
    for t in tags:
        f: dict = {"tag": t}
        if t == "missing_skill":
            f["missing"] = _join(s for s in skills if not has_phrase(red, s))
            f["card_says"] = "skills: list every one, in the card's exact wording"
        elif t == "missing_license":
            f["missing"] = _join(s for s in card.get("licenses", ()) if not has_phrase(red, s))
            f["card_says"] = "licenses: list every one, in the card's exact wording"
        elif t in ("missing_degree_level", "missing_degree_field", "missing_graduation_year", "missing_expected_marker",
                   "missing_prior_bachelor", "stale_expected"):
            what = {"missing_degree_level": f"the {ctx.get('degree', 'degree')}",
                    "missing_degree_field": f"the degree field {card.get('degree_field')}",
                    "missing_graduation_year": f"the year {card.get('graduation_year')}",
                    "missing_expected_marker": f"'Expected <month> {card.get('graduation_year')}'",
                    "missing_prior_bachelor": f"the finished bachelor's of {prior_bachelor_year(card)}"}.get(t)
            if t == "stale_expected":
                f["quote"] = _clip((stale_expected(text) or [""])[0])
            else:
                f["missing"] = what
            f["card_says"] = f"education_status: '{ctx.get('education_status', '')}'"
        elif t == "missing_years":
            f["missing"] = f"{card.get('years')} years"
            f["card_says"] = f"years_of_experience: {ctx.get('years_of_experience')}, said once in the summary"
        elif t == "years_overstated":
            y = int(card.get("years") or 0)
            m = next((m for m in STATED_YEARS.finditer(red) if (int(m.group(1)) if m.group(1).isdigit()
                                                               else _NUM_WORDS.index(m.group(1).lower())) > y), None)
            f["quote"] = _clip(_sentence_at(red, m.start())) if m else None
            f["card_says"] = f"years_of_experience: {y}, never rounded up"
        elif t == "years_vs_dates":
            f["quote"] = _clip("; ".join(m.group(0) for m in DATE_RANGE.finditer(plain))) or None
            f["card_says"] = "jobs: write exactly these dates, one job per entry: " + "; ".join(
                j["dates"] for j in ctx.get("jobs", ())) if ctx.get("jobs") else "no jobs: list no dated job"
        elif t == "redact_leaves_contact":
            f["card_says"] = "contact: write it once, at the top, exactly as given, and nowhere else"
        elif t == "leak_term":
            f["quote"] = _match_sentence(LEAK_RESUME, text) or _match_sentence(LEAK_WORDS, text)
            f["card_says"] = "no work authorization, visa, salary or contact details on the resume"
        elif t == "degree_for_none":
            f["quote"] = _match_sentence(DEGREE_WORDS, text)
            f["card_says"] = f"education_status: '{ctx.get('education_status', '')}'"
        elif t == "artifact":
            f["quote"] = _match_sentence(ARTIFACTS, text)
            f["card_says"] = "leave out a section the card gives nothing for; name a real-sounding school with its city"
        elif t == "city_missing":
            f["missing"] = card.get("current_location")
            f["card_says"] = f"the line after the header is current_location exactly: {card.get('current_location')}"
        elif t == "major_for_blank_field":
            f["quote"] = _sentence_of(plain, (major_named(card, plain) or [""])[0])
            f["card_says"] = f"degree: {ctx.get('degree')}, by level only, with no subject or major"
        elif t == "extra_tool":
            f["quote"] = _sentence_of(plain, (extra_tools(card, plain) or [""])[0])
            f["card_says"] = "skills (the only tools and software to name): " + _join(skills)
        elif t == "extra_license":
            f["quote"] = _sentence_of(plain, (extra_licenses(card, plain) or [""])[0])
            f["card_says"] = "licenses: " + _join(card.get("licenses", ()))
        elif t in ("title_without_license", "licensed_title_too_early"):
            hits = (title_without_license if t == "title_without_license" else licensed_title_too_early)(card, plain)
            f["quote"] = _sentence_of(plain, (hits or [""])[0])
            f["card_says"] = ("jobs: use these job titles: " + "; ".join(f"{j['title']} ({j['dates']})" for j in ctx.get("jobs", ()))
                              + "; licenses: " + _join(card.get("licenses", ())))
        elif t == "present_job_elsewhere":
            hits = present_job_elsewhere(card, plain)
            entry = next((e for e in job_entries(plain) if e["present"] and hits and any(
                hits[0].split(",")[0] in x for x in e["lines"])), None)  # the job entry, not the school in the same city
            f["quote"] = _clip(" / ".join(x.strip() for x in entry["lines"])) if entry else _sentence_of(plain, (hits or [""])[0])
            f["card_says"] = f"the job that ends 'Present' is in {card.get('current_location')}"
        elif t == "program_field_missing":
            f["quote"] = _clip((program_field_missing(card, red) or [""])[0].rstrip(".")) or None
            f["card_says"] = (f"the {ctx.get('degree', 'degree')} still in progress (the one marked 'Expected "
                              f"<month> {card.get('graduation_year')}') is in {card.get('degree_field')}: write that "
                              f"subject on that degree's own line")
        elif t == "certification_added":
            f["quote"] = _clip((credential_claimed(card, red) or [""])[0].rstrip(".")) or None
            f["card_says"] = ("licenses: " + _join(card.get("licenses", ())) + "; list no other certification, certificate "
                              "or license, and a skill name is a skill, never a certification")
        elif t == "lacked_skill_named":
            hits = lacked_skill_named(card, red)
            f["quote"] = _sentence_of(red, hits[0]) if hits else None
            f["named"] = _join(hits)
            f["card_says"] = "skills (the only skills to name, anywhere): " + _join(skills)
        elif t == "pair_lacked_named":
            hits = pair_lacked_named(card, red)
            f["quote"] = _sentence_of(red, hits[0]) if hits else None
            f["named"] = _join(hits)
            f["card_says"] = ("skills (the only skills to name, anywhere, in any wording, coursework included): " + _join(skills)
                              + "; the person does not have " + _join(hits))
        elif t == "higher_degree_implied":
            f["quote"] = (higher_degree_implied(card, plain) or [None])[0]
            f["card_says"] = (f"degree: {ctx.get('degree', 'none')}; education_status: '{ctx.get('education_status', '')}': "
                              "call the studies by that degree only, never advanced, graduate or doctoral studies")
        elif t == "non_latin_script":
            f["quote"] = _sentence_of(plain, (non_latin_script(card, plain) or [""])[0])
            f["card_says"] = NON_LATIN_SAYS
        elif t == "no_job_dated_role":
            f["quote"] = (no_job_dated_role(card, plain) or [None])[0]
            f["card_says"] = ("no jobs: list no dated role anywhere, volunteer, project, tutoring and internship roles "
                              "included; dates belong only to the education entry")
        else:
            f["card_says"] = f"the check '{t}' failed"
        out.append(f)
    return out


def _posting_findings(card: dict, text: str, tags: list[str]) -> list[dict]:
    ctx = context_of("postings", card)
    plain = textclean.strip_markdown(text)
    must, nice = _names(card.get("must_skills", ())), list(card.get("nice_skills", ()))
    out = []
    for t in tags:
        f: dict = {"tag": t}
        if t == "missing_required_skill":
            f["missing"] = _join(n for n in must if not has_phrase(plain, n))
            f["card_says"] = "required_skills: name every one exactly as given, in the requirements part"
        elif t == "missing_preferred_skill":
            f["missing"] = _join(n for n in nice if not has_phrase(plain, n))
            f["card_says"] = "preferred_skills: name every one exactly as given, under a preferred line"
        elif t == "missing_license":
            f["missing"] = card.get("license")
            f["card_says"] = f"required_license: {ctx.get('required_license')}"
        elif t == "missing_years":
            f["missing"] = f"{card.get('min_years')} years"
            f["card_says"] = f"minimum_years_experience: {ctx.get('minimum_years_experience')}"
        elif t == "missing_pay":
            f["missing"] = "the pay figures"
            f["card_says"] = f"pay: {ctx.get('pay')}, every figure as given"
        elif t == "missing_location":
            f["missing"] = _join(x for x in card.get("locations", ()) if not has_phrase(plain, x.split(",")[0].strip()))
            f["card_says"] = "locations: " + _join(ctx.get("locations", ()))
        elif t in ("missing_work_mode", "mode_conflict"):
            if t == "missing_work_mode":
                f["missing"] = ctx.get("work_mode")
            else:
                f["quote"] = next((q for k in mode_conflict(card, plain) if (q := _match_sentence(MODE_SAID[k], plain))), None)
            f["card_says"] = f"work_mode: {ctx.get('work_mode')}, and no other arrangement"
        elif t in ("missing_job_type", "type_conflict"):
            if t == "missing_job_type":
                f["missing"] = ctx.get("employment_type")
            else:
                f["quote"] = next((q for k in type_conflict(card, plain) if (q := _match_sentence(TYPE_SAID[k], plain))), None)
            f["card_says"] = f"employment_type: {ctx.get('employment_type')}, and no other job type, hours or term"
        elif t == "missing_industry":
            f["missing"] = card.get("industry")
            f["card_says"] = f"industry: copy '{ctx.get('industry')}' letter for letter"
        elif t == "missing_degree_level":
            f["missing"] = ctx.get("minimum_degree") or ctx.get("completed_degree")
            f["card_says"] = f"the degree: {ctx.get('minimum_degree') or ctx.get('completed_degree')}"
        elif t == "missing_field_of_study":
            f["missing"] = _join(x for x in card.get("degree_fields", ()) if not has_phrase(plain, x))
            f["card_says"] = "fields of study: " + _join(card.get("degree_fields", ()))
        elif t == "missing_student_requirement":
            f["missing"] = "the student requirement"
            f["card_says"] = f"must_be_student_in: {ctx.get('must_be_student_in')}"
        elif t in ("missing_sponsorship", "missing_opt", "missing_citizens_only", "missing_e_verify"):
            f["missing"] = {"missing_sponsorship": "the sponsorship statement", "missing_opt": "the OPT statement",
                            "missing_citizens_only": "the citizenship statement", "missing_e_verify": "the E-Verify statement"}[t]
            f["card_says"] = _wa_line(ctx)
        elif t == "auth_scope_added":
            f["quote"] = _sentence_of(plain, (auth_scope_added(card, plain) or [""])[0])
            f["card_says"] = (_wa_line(ctx) + "; say nothing about help with immigration or work-authorization processes, "
                              "about anyone's status, or about when or how the person hired is authorized to work")
        elif t in ("sponsorship_added", "work_auth_topic_unset", "neg_work_auth", "scope_word", "e_verify_unset",
                   "stance_conflict", "sponsor_time_scope", "student_auth_refused"):
            if t == "sponsor_time_scope":
                q = _sentence_of(plain, (sponsor_time_scope(card, plain) or [""])[0])
            elif t == "student_auth_refused":
                q = _sentence_of(plain, (student_auth_refused(card, plain) or [""])[0])
            else:
                rx = {"sponsorship_added": SPONSOR_OFFERED, "neg_work_auth": NEG_AUTH, "scope_word": SCOPE_WORDS,
                      "e_verify_unset": re.compile(r"e-?verify", re.I)}.get(t)
                q = _match_sentence(rx, text) if rx else None
                if t == "work_auth_topic_unset":
                    q = _match_sentence(AUTH_TOPIC, text)
                if t == "stance_conflict":
                    from matcher.visibility import CITIZENS, NO_OPT, NO_SPONSOR
                    q = _match_sentence(NO_SPONSOR, text) or _match_sentence(NO_OPT, text) or _match_sentence(CITIZENS, text)
            f["quote"] = q
            f["card_says"] = _wa_line(ctx)
        elif t in ("years_added", "experience_added", "years_domain_added", "neg_experience"):
            if t == "years_added":
                bad = set(years_added(card, plain))
                m = next((m for m in STATED_YEARS.finditer(plain) if (int(m.group(1)) if m.group(1).isdigit()
                                                                      else _NUM_WORDS.index(m.group(1).lower())) in bad), None)
                f["quote"] = _clip(_sentence_at(plain, m.start())) if m else None
            elif t == "experience_added":
                f["quote"] = _sentence_of(plain, (experience_added(card, plain) or [""])[0])
            elif t == "years_domain_added":
                f["quote"] = _sentence_of(plain, (years_domain_added(card, plain) or [""])[0])
            else:
                f["quote"] = _match_sentence(NEG_EXP, text)
            f["card_says"] = (f"minimum_years_experience: {ctx['minimum_years_experience']}" if ctx.get("minimum_years_experience")
                              else "the card asks for no experience: say nothing about experience")
        elif t in ("pay_added", "neg_pay"):
            if t == "pay_added":
                bad = pay_added(card, plain)
                m = next((m for m in MONEY.finditer(plain) if bad and money_figures(m.group()) and
                          any(abs(money_figures(m.group())[0] - b) < 0.01 for b in bad)), None)
                f["quote"] = _clip(_sentence_at(plain, m.start())) if m else None
            else:
                f["quote"] = _match_sentence(NEG_PAY, text)
            f["card_says"] = f"pay: {ctx['pay']}, and no other figure" if ctx.get("pay") else \
                "the card gives no pay: say nothing about pay"
        elif t in ("license_added", "license_scope_added", "neg_license"):
            hits = (license_added if t == "license_added" else license_scope_added)(card, plain) if t != "neg_license" else []
            f["quote"] = _sentence_of(plain, hits[0]) if hits else _match_sentence(NEG_LICENSE, text)
            f["card_says"] = (f"required_license: {ctx['required_license']}, with no state, place or other credential"
                              if ctx.get("required_license") else "the card asks for no license or certification: say nothing about them")
        elif t in ("extra_required_skill", "extra_plain_skill", "skill_level_added"):
            if t == "skill_level_added":
                f["quote"] = _sentence_of(plain, (skill_level_added(card, plain) or [""])[0])
            else:
                hits = extra_required_skills(card, plain) if t == "extra_required_skill" else extra_plain_skills(card, plain)[0]
                f["named"] = _join(hits)
                f["quote"] = next((s for s, _ in _sentences_with_heading(plain) for h in hits[:1] if has_phrase(s, h)), None)
            req = [" or ".join(g["any_one_of"]) if isinstance(g, dict) else g for g in ctx.get("required_skills", ())]
            f["card_says"] = ("required_skills: " + _join(req) + "; preferred_skills: " + _join(nice)
                              + " (named as given, with no level and no other skill asked of the applicant)")
        elif t in ("any_of_as_all_of", "required_skill_not_stated_required"):
            groups = multi_groups(card) if t == "any_of_as_all_of" else required_as_duty_only(card, text)
            f["missing"] = "; ".join(" or ".join([g] if isinstance(g, str) else g) for g in groups)
            f["card_says"] = ("an any_one_of entry: one of them is enough, written as alternatives in one sentence"
                              if t == "any_of_as_all_of" else
                              "required_skills: every entry goes in the requirements list or a 'must' sentence, not only in the duties")
        elif t in ("neg_degree", "student_topic_unset", "degree_field_added"):
            rx = {"neg_degree": NEG_DEGREE, "degree_field_added": FIELD_SAID}.get(t)
            q = _match_sentence(rx, ANY_FIELD.sub(" ", plain) if t == "degree_field_added" else text) if rx else (
                student_said(card, text))
            f["quote"] = q
            f["card_says"] = "; ".join(f"{k}: {ctx[k]}" for k in ("minimum_degree", "completed_degree", "must_be_student_in",
                                                                 "who_can_apply", "degree_field", "fields_of_study",
                                                                 "degree_subjects") if k in ctx) or \
                "the card asks for no degree and no student status: say nothing about degrees, students or graduation"
            if t == "student_topic_unset" and "must_be_student_in" not in ctx:
                f["card_says"] += ("; the job has no school requirement: say nothing about students, enrollment or student "
                                   "status, not even that they are not required")
        elif t == "year_unset":
            f["quote"] = _match_sentence(re.compile(r"\b(?:spring|summer|fall|winter|autumn)\s+20\d\d\b|\b20[2-3]\d\b", re.I), text)
            f["card_says"] = f"program_term: {ctx['program_term']}" if ctx.get("program_term") else "write no year or season"
        elif t in ("no_company_line", "company_not_given"):
            f["quote"] = _clip(textclean.split_synthetic(text)[1] or "") or None
            if not f["quote"]:
                f["missing"] = "the company line"
            f["card_says"] = f"line 2 is the company name, alone: {ctx.get('company')}"
        elif t == "location_added":
            f["quote"] = _sentence_of(plain, (location_added(card, plain) or remote_region_added(card, plain) or [""])[0])
            f["card_says"] = ("locations: " + _join(ctx.get("locations", ())) + ", and no other place" if ctx.get("locations")
                              else "work_mode: fully remote, with no place: name no country, state, region or time zone for "
                                   "where the person may work")
        elif t == "locations_split":
            f["quote"] = _sentence_of(plain, (locations_split(card, plain) or [""])[0])
            f["card_says"] = ("locations: " + " or ".join(ctx.get("locations", ())) + ": the job is based in one of these places "
                              "and the person works in one of them; say it is based in one or another of them, never that "
                              "the person splits time, rotates or travels between them or works at all of them")
        elif t == "recent_grad_added":
            f["quote"] = _sentence_of(plain, (recent_grad_added(card, plain) or [""])[0])
            f["card_says"] = ("; ".join(f"{k}: {ctx[k]}" for k in ("minimum_degree", "completed_degree", "must_be_student_in",
                                                                   "who_can_apply") if k in ctx) or "no degree and no students")
            f["card_says"] += "; nothing about how recently a degree was finished: no recent, new or fresh graduates"
        elif t == "non_latin_script":
            f["quote"] = _sentence_of(plain, (non_latin_script(card, plain) or [""])[0])
            f["card_says"] = NON_LATIN_SAYS
        elif t == "education_done_added":
            f["quote"] = _sentence_of(plain, (education_done_added(card, plain) or [""])[0])
            f["card_says"] = ("; ".join(f"{k}: {ctx[k]}" for k in ("minimum_degree", "completed_degree", "must_be_student_in",
                                                                   "who_can_apply") if k in ctx) or "no degree and no students")
            f["card_says"] += ("; never say the applicant's education or studies are finished or that they have graduated: "
                               "name the degree itself, if the card asks for one")
        elif t == "travel_added":
            f["quote"] = _match_sentence(TRAVEL, text)
            f["card_says"] = "the card has no travel: say nothing about travel"
        elif t == "neg_meta":
            f["quote"] = _match_sentence(NEG_ANY, text)
            f["card_says"] = "a fact that is not in the context is absent: say nothing about its topic"
        elif t == "trait_required":
            f["quote"] = _sentence_of(plain, (trait_required(card, plain) or [""])[0])
            f["card_says"] = ("required_skills: " + _join(must) + "; preferred_skills: " + _join(nice) + ": ask nothing else of "
                              "the applicant; no personal trait (detail-oriented, reliable, proactive, self-disciplined, "
                              "adaptable, organized) is required, essential or what the ideal candidate is")
        elif t == "condition_added":
            f["quote"] = _sentence_of(plain, (condition_added(card, plain) or [""])[0])
            f["card_says"] = ("the card has no working condition: no transportation, driver's license, lifting, standing, "
                              "physical demands, background check, drug test or clearance, and no subject foundation beyond "
                              + (", ".join(f"{k}: {ctx[k]}" for k in ("degree_field", "fields_of_study", "degree_subjects")
                                           if k in ctx) or "the card's skills"))
        elif t == "hours_added":
            f["quote"] = _sentence_of(plain, (hours_added(card, plain) or [""])[0])
            f["card_says"] = (f"employment_type: {ctx.get('employment_type')}, and nothing about hours or schedule: no "
                              "flexible hours, business hours, weekends, evenings, shifts, rotations, overtime or on-call")
        elif t == "pay_terms_added":
            f["quote"] = _sentence_of(plain, (pay_terms_added(card, plain) or [""])[0])
            f["card_says"] = (f"pay: {ctx.get('pay')}, as a plain range: it does not depend on experience, qualifications or "
                              "performance, and there is no bonus or commission")
        elif t == "locations_not_alternatives":
            f["missing"] = "that the job is in one of " + " or ".join(ctx.get("locations", ()))
            f["card_says"] = ("locations: " + " or ".join(ctx.get("locations", ())) + ": write one sentence that names them "
                              "as alternatives ('in A or B', 'at one of our offices in A or B'), never 'in A and B'")
        elif t == "citizens_reason_added":
            f["quote"] = _sentence_of(plain, (citizens_reason_added(card, plain) or [""])[0])
            f["card_says"] = (_wa_line(ctx) + "; give no reason for it: no law, regulation, contract, government role, "
                              "export control or the nature of the work")
        elif t == "no_sponsor_added":
            f["quote"] = _sentence_of(plain, (no_sponsor_added(card, plain) or [""])[0])
            f["card_says"] = _wa_line(ctx) + "; the card says nothing about sponsorship: write nothing about it"
        elif t == "hybrid_anywhere":
            f["quote"] = _sentence_of(plain, (hybrid_anywhere(card, plain) or [""])[0])
            f["card_says"] = ("work_mode: hybrid, " + ("in " + " or ".join(ctx["locations"]) if ctx.get("locations") else "")
                              + ": the remote days are not 'from anywhere' or 'a location of your choice'")
        else:
            f["card_says"] = f"the check '{t}' failed"
        out.append(f)
    return out


def findings(kind: str, card: dict, text: str) -> list[dict]:
    """One finding per fatal tag of `text` under the current gate code, in fatal()'s order: {"tag", "quote" (the
    offending sentence) or "missing" (the missing fact), "card_says" (one plain line in the context's own wording)}."""
    tags = list(dict.fromkeys(fatal(kind, card, text)))
    return (_resume_findings if kind == "resumes" else _posting_findings)(card, text, tags)


def feedback_lines(fs: list[dict]) -> list[str]:
    """The retry context's FEEDBACK_KEY value: one line per finding."""
    out = []
    for f in fs:
        if f.get("quote"):
            got = f"Your last version wrote: \"{f['quote']}\""
            if f.get("named"):
                got += f" (it names {f['named']})"
        elif f.get("missing"):
            got = f"Your last version left out {f['missing']}"
        elif f.get("named"):
            got = f"Your last version named {f['named']}"
        else:
            got = "Your last version got this wrong"
        out.append(f"{got}. The card says: {f['card_says']}.")
    return list(dict.fromkeys(out))


def retry_context(kind: str, card: dict, last_text: str | None, last_tags: list[str] = ()) -> str:
    """The context a retry sends: the card's context with FEEDBACK_KEY appended (the findings of the last failed try,
    re-gated now; its stored tags when it has no text or now passes)."""
    ctx = context_of(kind, card)
    fs = findings(kind, card, last_text) if last_text else []
    if not fs:
        fs = (_resume_findings if kind == "resumes" else _posting_findings)(card, last_text or "",
                                                                             [t for t in last_tags if t not in NONFATAL])
    lines = feedback_lines(fs)
    return json.dumps(ctx | {FEEDBACK_KEY: lines}, ensure_ascii=False) if lines else ctx_str(kind, card)


def base_ctx(sent: str) -> str:
    """The card's own context string of a sent context (a retry's without FEEDBACK_KEY): what its text is stored under."""
    if f'"{FEEDBACK_KEY}"' not in sent:
        return sent
    d = json.loads(sent)
    d.pop(FEEDBACK_KEY, None)
    return json.dumps(d, ensure_ascii=False)


def quarantine_path(kind: str) -> Path:
    return OUT / f"quarantine_{kind}.jsonl"


def tries(kind: str) -> Counter:
    """ctx hash -> number of rendered texts for it that failed the gates (from the quarantine file)."""
    return Counter(r["ctx"] for r in read_jsonl(quarantine_path(kind)))


def last_failed(kind: str) -> dict[str, dict]:
    """ctx hash -> its last failed try (the quarantine row), for the retry's findings; {} for a context with none."""
    out: dict[str, dict] = {}
    for r in read_jsonl(quarantine_path(kind)):
        out[r["ctx"]] = r
    return defaultdict(dict, out)


def assert_clean(kind: str, cards: dict[str, dict], texts: dict[str, str]) -> None:
    """Raise if any text in `texts` (card id -> text) has a fatal tag. For an assembly step to call on
    the texts it is about to use."""
    bad = {i: tags for i, t in texts.items() if (tags := fatal(kind, cards[i], t))}
    if bad:
        raise ValueError(f"{len(bad)} {kind} texts fail the gates, e.g. {dict(list(bad.items())[:5])}")


def qa(kind: str, cards: dict[str, dict], rows: list[dict]) -> None:
    """Cheap text checks. Prints one line; returns nothing (use `report` for gates)."""
    probs = {r["id"]: (resume_problems if kind == "resumes" else posting_problems)(cards[r["id"]], r["text"]) for r in rows}
    c = Counter(t for v in probs.values() for t in v)
    print(f"QA {kind} ({len(rows)}): rows with a problem {sum(bool(v) for v in probs.values())}", dict(c))
    if kind == "postings":
        conflicts = [r["id"] for r in rows if stance_conflict(cards[r["id"]], r["text"])]
        print(f"QA postings: stance conflicts {len(conflicts)}", conflicts[:10])


# Context wording the writer copied into the pilot; real postings and resumes do not use it.
# Entries are regexes on lower-cased text without commas.
REQUIRED_SKILL_LABEL = r"required skill(?!s| set)"  # the old context label, not a 'Required Skills' heading
ECHO = {
    "postings": ("already hold", "already completed", r"a completed (?:associate|bachelor|master|doctoral)", REQUIRED_SKILL_LABEL, "field of study must be", "program term", "currently enrolled in and pursuing",
                 "now or in the future", "any one of these alternatives", "regardless of their current immigration status",
                 "must_be_student_in", "minimum_degree", "completed_degree", "who_can_apply", "fields of study:", "minimum years",
                 # the writer's own habits (real postings: 0.2%, 6.8%, 1.7%, 5.0%)
                 "we are currently seeking", "in this role you will", "to be considered for this", "we encourage you to apply",
                 # the retry feedback's own words
                 "fix_from_last_try", "last version", "the card says"),
    "resumes": ("years of paid", "paid professional experience", "paid experience", "years_of_experience", "job_dates", "layout",
                "employer_city", "summary_skills",
                # education_status wording
                "this is in the past", "still enrolled in the", "a month before october", "write a plain graduation date",
                "also list a finished", "by level only",
                "on-the-job training only",
                # the retry feedback's own words
                "fix_from_last_try", "last version", "the card says"),
}
# Work-authorization wordings no context carries any more: still counted, because the writer learned them as stock
# lines ('Visa sponsorship is not available for this position.' was in 20.8% of a check batch).
RETIRED_WORK_AUTH_TEXT = ("Visa sponsorship is not available for this position.",
                          "The employer cannot sponsor employment visas now or in the future.",
                          "The employer cannot sponsor employment visas now or in the future, and cannot support F-1 OPT or "
                          "STEM OPT employment, including I-983 training plans.",
                          "Visa sponsorship, such as H-1B, is available for this role.",
                          "The employer sponsors employment visas (for example H-1B) for this position.",
                          "No visa sponsorship is available, and the role cannot be filled through F-1 OPT or STEM OPT, "
                          "including I-983 training plans.",
                          "The employer cannot sponsor employment visas for this role.",
                          "The employer does not sponsor employment visas such as H-1B for this role.",
                          "The company is unable to sponsor work visas for this job.",
                          "The employer will not sponsor H-1B or other employment visas for this position.",
                          "The employer is not able to sponsor employment visas for the person hired.",
                          "The company won't sponsor work visas such as H-1B for this position.",
                          "The employer cannot sponsor employment visas and cannot support F-1 OPT or STEM OPT employment, "
                          "including I-983 training plans.",
                          "The employer does not sponsor employment visas such as H-1B and does not accept F-1 OPT or STEM OPT "
                          "work authorization (no I-983 training plans).",
                          "The company will not sponsor work visas, and it cannot hire through F-1 OPT or STEM OPT, including "
                          "I-983 training plans.",
                          "The employer is unable to sponsor employment visas and does not take part in F-1 OPT or STEM OPT "
                          "employment or I-983 training plans.",
                          "The company won't sponsor H-1B or other work visas and will not employ anyone on F-1 OPT or STEM OPT"
                          " (no I-983 training plans).")
# who may apply to an internship without a student requirement, copied word for word
ECHO["postings"] += tuple(re.escape(t.lower().replace(",", "")) for t in (*OPEN_TO_TEXT, *RETIRED_OPEN_TO_TEXT))
# the duty openers, and the key name
ECHO["postings"] += tuple(re.escape(t.lower().replace(",", "").rstrip(":")) for t in DUTY_OPENERS) + ("duties_opener",)
# the any-field wordings, and the key name
ECHO["postings"] += tuple(re.escape(t) for t in ANY_FIELD_TEXT) + ("degree_field", "degree_subjects")
# the years wordings, any number
ECHO["postings"] += tuple(re.escape(t.lower().replace(",", "")).replace(re.escape("{n}"), r"\d+ years?")
                          .replace(r"\ ", " ") for t in YEARS_TEXT)
# the work-authorization statements copied word for word (each wording of each stance)
ECHO["postings"] += tuple(re.escape(t.lower().replace(",", "").rstrip(".")) for ts in (*WORK_AUTH_TEXT.values(), CITIZENS_TEXT,
                                                                                      E_VERIFY_TEXT, RETIRED_WORK_AUTH_TEXT)
                          for t in ts)
ECHO_MAX = 0.15
# The writer's own openers and closers the r5 check read counted over the 39 kept postings ('we are looking for' 41%,
# 'while not mandatory' 38%, 'to qualify for this' 38%, 'we look forward to' 31%, 'candidates who bring additional' 28%,
# 'to succeed in this' 26%, 'must meet specific educational and' 20%). Reported next to the stock phrases, not counted
# against ECHO_MAX this round: no context value covers them yet, and a blueprint line would change the writer.
ECHO_WATCH = {"postings": ("we are looking for", "while not mandatory", "to qualify for this", "we look forward to",
                           "candidates who bring additional", "to succeed in this", "must meet specific educational and"),
              "resumes": ()}


def echo_rates(kind: str, texts: list[str]) -> dict:
    """Share of texts with each stock phrase, and the 5 most common 5-word phrases (for a read)."""
    n = max(len(texts), 1)
    low = [" ".join(t.lower().replace(",", "").replace("’", "'").split()) for t in texts]  # ’ as ' ("listing’s")
    rates = {p: round(sum(re.search(p, t) is not None for t in low) / n, 3) for p in ECHO[kind]}
    grams: Counter = Counter()
    for t in low:
        w = re.findall(r"[a-z0-9'’$-]+", t)
        grams.update({" ".join(w[i:i + 5]) for i in range(len(w) - 4)})
    top = [(g, round(c / n, 3)) for g, c in grams.most_common(8)]
    watch = {p: round(sum(re.search(re.escape(p), t) is not None for t in low) / n, 3) for p in ECHO_WATCH.get(kind, ())}
    return {"stock_phrase_rates": rates, "max_stock_phrase": max(rates.values()) if rates else 0,
            "stock_phrase_gate": (max(rates.values()) <= ECHO_MAX and rates.get(REQUIRED_SKILL_LABEL, 0) == 0) if rates else True,
            # report only: not in max_stock_phrase or stock_phrase_gate
            "watched_phrase_rates": watch, "watched_over_cap": sorted(p for p, r in watch.items() if r > ECHO_MAX),
            "top_5grams": top}


def report(kind: str, cards: dict[str, dict], want: list[str], rows_file: str | None = None) -> dict:
    """Gate numbers for the rendered texts of `want` (no credits). With rows_file, the texts are the
    rows of that jsonl by card id, whatever context they were rendered from (an old pilot)."""
    if rows_file:
        by_id = {r["id"]: r["text"] for r in read_jsonl(Path(rows_file))}
        texts = {i: by_id[i] for i in want if i in by_id}
    else:
        texts = rendered_texts(kind, {i: cards[i] for i in want}, clean=False)
    rows = [(i, cards[i], t) for i, t in texts.items()]
    out: dict = {"kind": kind, "asked": len(want), "rendered": len(rows), "source": rows_file or "current contexts"}
    if not rows_file:  # texts that failed the gates at download and never entered the store
        q = tries(kind)
        cur = {i: ctx_hash(ctx_str(kind, cards[i])) for i in want}
        out["quarantined_tries"] = sum(q[h] for h in set(cur.values()))
        out["given_up_after_redo"] = sorted(i for i, h in cur.items() if q[h] > MAX_REDO and i not in texts)
        out["first_tries"] = first_tries(kind, cards, want)
    out["raw_markdown_rows"] = sum(textclean.has_markdown(t) for _, _, t in rows)
    probs = {i: (resume_problems if kind == "resumes" else posting_problems)(c, t) for i, c, t in rows}
    out["rows_with_a_problem"] = sum(bool(v) for v in probs.values())
    out["problem_counts"] = dict(Counter(p for v in probs.values() for p in v).most_common())
    if kind == "resumes":
        out["with_contact_header"] = sum("contact" in context_of(kind, c) for _, c, _ in rows)
        spans = [abs(work_span_years(textclean.strip_markdown(t)) - int(c.get("years") or 0)) for _, c, t in rows]
        out["years_vs_dates_pass_rate"] = round(1 - sum("years_vs_dates" in v for v in probs.values()) / max(len(rows), 1), 3)
        out["years_gap_median"] = sorted(spans)[len(spans) // 2] if spans else None
    else:
        groups = [(g, t) for _, c, t in rows for g in multi_groups(c)]
        ok = sum(alt_phrased(g, t) for g, t in groups)
        out["multi_name_groups"] = len(groups)
        out["groups_phrased_as_alternatives"] = ok
        out["alternatives_share"] = ok / len(groups) if groups else None
        req = [(g, c, t) for _, c, t in rows for g in c.get("must_skills", ())]
        not_req = sum(len(required_as_duty_only(c, t)) for _, c, t in rows)
        out["required_groups"] = len(req)
        out["required_groups_stated_as_required"] = len(req) - not_req
        neg = ("neg_degree", "neg_experience", "neg_license", "neg_work_auth", "neg_pay", "neg_meta", "work_auth_topic_unset", "e_verify_unset")
        out["negation_statements_for_unset_fields"] = sum(sum(t in neg for t in v) for v in probs.values())
        words = sorted(len(t.split()) for _, _, t in rows)
        out["words_p10_p50_p90"] = [words[len(words) // 10], words[len(words) // 2], words[9 * len(words) // 10]] if words else None
        out["stance_conflicts"] = sum(bool(stance_conflict(c, t)) for _, c, t in rows)
        out["work_auth_mix"] = dict(Counter(f"{c.get('sponsorship')}{'+citizens_only' if c.get('citizens_only') else ''}"
                                            f"{'+e_verify' if c.get('e_verify') else ''}" for _, c, _ in rows).most_common())
    out |= echo_rates(kind, [t for _, _, t in rows])
    out["problem_ids"] = {i: v for i, v in probs.items() if v}
    return out


# Strata a check batch must cover: each needs its own rows, because a stratum with no row is a
# wording nobody has read (the first pilot had 0 citizens-only postings).
POSTING_STRATA = {
    "citizens_only": lambda c: c.get("citizens_only") and not c.get("e_verify"),
    "citizens_only+e_verify": lambda c: c.get("citizens_only") and c.get("e_verify"),
    "sponsors+e_verify": lambda c: c.get("sponsorship") == "sponsors" and c.get("e_verify"),
    "will_not_sponsor": lambda c: c.get("sponsorship") == "will_not_sponsor" and not c.get("citizens_only"),
    "will_not_sponsor_or_opt": lambda c: c.get("sponsorship") == "will_not_sponsor_or_opt",
    "any_of_group": lambda c: bool(multi_groups(c)),
    "student_only": lambda c: c.get("enrolled_in", "none") != "none",
    "degree_fields": lambda c: bool(c.get("degree_fields")),
    "license": lambda c: bool(c.get("license")),
    "pay": lambda c: c.get("salary_min") is not None or c.get("salary_max") is not None,
    "internship_term": lambda c: c.get("employment_type") == "internship" and "program_term" in context_of("postings", c),
    "remote": lambda c: c.get("work_mode") == "remote",
    "hybrid": lambda c: c.get("work_mode") == "hybrid",
}
RESUME_STRATA = {
    "years_0": lambda c: int(c.get("years") or 0) == 0,
    "years_1_2": lambda c: 1 <= int(c.get("years") or 0) <= 2,
    "years_3_6": lambda c: 3 <= int(c.get("years") or 0) <= 6,
    "years_7_plus": lambda c: int(c.get("years") or 0) >= 7,
    "still_enrolled": lambda c: (c.get("graduation_year") or 0) > CURRENT_YEAR,
    "enrolled_graduate": lambda c: prior_bachelor_year(c) is not None,
    "contact_header": lambda c: fake_contact(c) is not None,
}


def pick_ids(kind: str, cards: dict[str, dict], n: int, per_stratum: int = 3) -> list[str]:
    """A check batch of n ids: per_stratum ids from every stratum first, then the rest, all in card-id
    hash order (deterministic)."""
    strata = POSTING_STRATA if kind == "postings" else RESUME_STRATA
    order = sorted(cards, key=lambda i: _h(i + "|pick"))
    if kind == "postings":
        order = [i for i in order if i.startswith("rq_")]  # synthetic cards only, never real-test text
    out: list[str] = []
    for name, test in strata.items():
        got = [i for i in order if i not in out and test(cards[i])][:per_stratum]
        out += got
    out += [i for i in order if i not in out][:max(n - len(out), 0)]
    return out


# The check batch gates the full render: `run` on any id outside it is refused until `signoff` has written a
# sign-off for the current blueprint and check-batch contexts (a changed blueprint or context needs a new one).
CHECK_IDS = {"resumes": ROOT / "data/built/training/check_resume_ids.txt",
             "postings": ROOT / "data/built/training/check_posting_ids.txt"}
# Share of first tries (sent without retry feedback) that pass every fatal gate, re-gated under the current gate code. A
# cost floor, not a correctness bar: correctness is the gates plus a person's read of every kept text. At 55% and the
# retries' measured pass rate (about 80% or better), the full render costs about 1 + 0.45 + 0.45 * 0.2 = 1.54 tries a
# context, about 190 credits against a balance of 518 on 2026-10-19.
SIGNOFF_MIN_PASS = 0.55
SIGNOFF_RULES = __doc__[__doc__.index("Sign-off rules"):__doc__.index("  7. A person")] + \
    __doc__[__doc__.index("  7. A person"):].split("\n\n")[0]
# what the person signing off has read (the gates cannot check wording); --read-by is that person's own name, not
# an agent's or a script's: an agent that renders the batch does not sign it off
SIGNOFF_READ = {"resumes": "5 of the contact-header resumes after redact",
                "postings": "the work-authorization block of every citizens_only, citizens_only+e_verify and "
                            "sponsors+e_verify row"}
# who read every kept check text next to its card, before the person's read (an agent reader, so not --read-by)
FULL_READ = "every kept check text, by an independent Sonnet reader; its report is the read step of workflow wf_cb2125d4-e37"


def is_reader_name(name: str | None) -> bool:
    """A person's name for --read-by: letters, not an agent, a bot or a script."""
    return bool(name and re.fullmatch(r"[A-Za-z][A-Za-z.'’ -]{1,40}", name)
                and not re.search(r"(?i)\b(?:claude|agent|bot|ai|gpt|codex|script|auto\w*|ci|test|nobody|none)\b", name))


def _rel(p: Path) -> Path:
    return p.relative_to(ROOT) if p.is_relative_to(ROOT) else p


def check_ids(kind: str) -> list[str]:
    p = CHECK_IDS[kind]
    return [x.strip() for x in p.read_text().split() if x.strip()] if p.exists() else []


def signoff_path(kind: str) -> Path:
    return CHECK_IDS[kind].with_name(f"check_signoff_{kind}.json")


def check_fingerprint(kind: str, cards: dict[str, dict], ids: list[str]) -> str:
    """The writer's instructions plus the check batch's contexts: what a sign-off vouches for."""
    k = KINDS[kind]
    parts = [k["universal"], k["blueprint"], k["length"], *(f"{i}\t{ctx_str(kind, cards[i])}" for i in ids)]
    return hashlib.sha1("\x00".join(parts).encode()).hexdigest()[:16]


def _writer_batches(kind: str, cards: dict[str, dict]) -> tuple[dict[str, bool], dict[tuple[str, str], set[bool]], dict]:
    """Which batches the current writer rendered, for rows written before gate_download stamped a 'writer' on each row.
    A batch's csv tag is batch_tag(ids, contexts with their try counts, writer), so for a batch rendered at try 0 (a
    first render or a --redo, the only kind whose rows --redo moves to the replaced files) the tag recomputed with
    writer_fingerprint(kind) says whether it is the current writer's; every run of one batch has the same writer.
    Returns (dataset id -> current writer?, (id, ctx hash) -> {current writer?} over the batches that sent it, and
    (id, text) -> dataset id from the downloads)."""
    by_ds: dict[str, bool] = {}
    by_pair: dict[tuple[str, str], set[bool]] = {}
    writer = writer_fingerprint(kind)
    for sp in STATE.glob(f"render_{kind}_*.json"):
        try:
            st = json.loads(sp.read_text())
            ids = st["ids"]
            if "writer" in st:
                mine = st["writer"] == writer
            else:
                mine = all(x in cards for x in ids) and batch_tag(
                    ids, [f"{ctx_str(kind, cards[x])}#0" for x in ids], writer) == sp.stem.rsplit("_", 1)[-1]
            if st.get("dataset_id"):
                by_ds[st["dataset_id"]] = mine
            for r in csv.DictReader(open(ROOT / st["csv"], newline="")):
                by_pair.setdefault((r["row_id"], ctx_hash(base_ctx(r["card"]))), set()).add(mine)
        except (KeyError, OSError, ValueError):
            continue
    texts = {}
    for p in STATE.glob("*/download.jsonl"):
        for r in read_jsonl(p):
            texts[(r.get("row_id", ""), (r.get("enhanced_completion") or "").strip())] = p.parent.name
    return by_ds, by_pair, texts


def recorded_tries(kind: str, cards: dict[str, dict], ids: list[str]) -> tuple[list[dict], list[dict]]:
    """Every recorded try of the ids' current contexts by the current writer, re-gated under the current gate code:
    the stored texts, the quarantined ones, and the replaced and replaced-quarantine rows the current writer rendered
    (a --redo under the same blueprint moves tries aside, it does not undo them; _writer_batches tells an unstamped
    row's writer by its dataset, or by its batches when every batch that sent its context had one writer). Returns (tries, unattributed): each
    try is {"id", "ctx", "source", "tags"} with tags = fatal() now (a row without its text keeps its stored tags);
    the other rows are rows of another writer (writer_known: a kept text from before a blueprint change that had no --redo
    is not a try of the new blueprint) or of no writer the records can show (writer_known False), re-gated too, reported,
    not counted. 'feedback' is the number of retry feedback lines the try was sent (0: a first try)."""
    by_ctx = {ctx_hash(ctx_str(kind, cards[i])): i for i in ids}
    writer = writer_fingerprint(kind)
    mine = None
    tries_out, other = [], []
    for name in (f"{kind}.jsonl", quarantine_path(kind).name, f"replaced_{kind}.jsonl", f"replaced_quarantine_{kind}.jsonl"):
        for r in read_jsonl(OUT / name):
            if r["ctx"] not in by_ctx:
                continue
            known = True
            if "writer" in r:
                ok = r["writer"] == writer
            else:  # by its dataset (stamped, or found by its text), else by the batches that sent its context
                by_ds, by_pair, texts = mine = mine if mine is not None else _writer_batches(kind, cards)
                ds = r.get("dataset_id") or texts.get((r["id"], (r.get("text") or "").strip()))
                pair = by_pair.get((r["id"], r["ctx"]))
                if ds and ds in by_ds:
                    ok = by_ds[ds]
                elif pair in ({True}, {False}):
                    ok = pair == {True}
                else:  # no stamp, no dataset, no batch with one writer: the row cannot be tied to a writer
                    ok, known = False, False
            row = {"id": r["id"], "ctx": r["ctx"], "source": name.removesuffix(".jsonl"), "feedback": r.get("feedback", 0)}
            card = cards.get(r["id"]) or cards[by_ctx[r["ctx"]]]
            row["tags"] = fatal(kind, card, r["text"]) if r.get("text") else [t for t in r.get("tags", ["no_text"])
                                                                                   if t not in NONFATAL]
            if not ok:
                other.append(row | {"writer_known": known})
                continue
            tries_out.append(row)
    return tries_out, other


def first_tries(kind: str, cards: dict[str, dict], ids: list[str]) -> dict:
    """Over the check ids' current contexts: every recorded try (recorded_tries) re-gated under the current gate code,
    the share that passes every fatal gate, overall and for each tag, and the same share as the tags stored at render
    time had it (stored_pass_rate). 'kept' counts contexts with a stored text, 'kept_passing' those whose stored text
    passes now."""
    store = texts_by_ctx(kind)
    hashes = {ctx_hash(ctx_str(kind, cards[i])): i for i in ids}
    rows, other = recorded_tries(kind, cards, ids)
    first = [r for r in rows if not r["feedback"]]  # sent without retry feedback (every try before retries had it)
    retry = [r for r in rows if r["feedback"]]
    n = len(first)
    passing = sum(not r["tags"] for r in first)
    tags = Counter(t for r in first for t in set(r["tags"]))
    q = [r for r in read_jsonl(quarantine_path(kind)) if r["ctx"] in hashes]
    kept = sum(h in store for h in hashes)
    kept_ok = sum(h in store and not fatal(kind, cards[i], store[h]) for h, i in hashes.items())
    unknown = [r for r in other if not r["writer_known"]]
    counted = {r["ctx"] for r in rows}
    return {"contexts": len(hashes), "kept": kept, "kept_passing": kept_ok, "quarantined": len(q),
            "tries": n, "passing_tries": passing, "pass_rate": round(passing / n, 3) if n else None,
            "retries": len(retry), "passing_retries": sum(not r["tags"] for r in retry),
            "retry_pass_rate": round(sum(not r["tags"] for r in retry) / len(retry), 3) if retry else None,
            "tries_per_context": round(len(rows) / len(counted), 3) if counted else None,
            "stored_pass_rate": round(kept / (kept + len(q)), 3) if kept + len(q) else None,
            "tries_by_source": dict(Counter(r["source"] for r in rows)),
            "passing_by_source": dict(Counter(r["source"] for r in rows if not r["tags"])),
            "other_writer_rows": len(other) - len(unknown),
            "unattributed_rows": len(unknown), "unattributed_by_source": dict(Counter(r["source"] for r in unknown)),
            "unattributed_failing": sum(bool(r["tags"]) for r in unknown),
            "unattributed_contexts_without_try": len({r["ctx"] for r in unknown} - counted),
            "tag_pass_rate": {t: round(1 - c / n, 3) for t, c in tags.most_common()}}


def signoff_problems(kind: str, cards: dict[str, dict], ids: list[str]) -> tuple[dict, list[str]]:
    """The check batch's numbers and every reason they do not allow the full render (SIGNOFF_RULES)."""
    ft = first_tries(kind, cards, ids)
    texts = rendered_texts(kind, {i: cards[i] for i in ids}, clean=False)
    bad = {i: tags for i, t in texts.items() if (tags := fatal(kind, cards[i], t))}
    good = {i: t for i, t in texts.items() if i not in bad}
    rows, _ = recorded_tries(kind, cards, ids)
    mine = Counter(r["ctx"] for r in rows)
    kept_mine = {r["ctx"] for r in rows if r["source"] == kind}  # contexts whose stored text is a try of this writer
    why = []
    hashes = {i: ctx_hash(ctx_str(kind, cards[i])) for i in ids}
    no_text = sorted({i for i in ids if i not in good})
    if no_text:
        why.append(f"{len(no_text)} of {len(ids)} check contexts have no kept text that passes the current gates: {no_text[:10]}")
    late = sorted(i for i in good if mine[hashes[i]] > MAX_REDO + 1)
    if late:
        why.append(f"{len(late)} check contexts were kept after more than {MAX_REDO + 1} tries: {late[:10]}")
    stale = sorted(i for i in good if hashes[i] not in kept_mine)
    if stale:
        why.append(f"{len(stale)} check contexts keep a text of another writer, not a try of this blueprint (render them with "
                   f"--redo): {stale[:10]}")
    if ft["tries"] == 0:
        return ft, why or ["nothing rendered"]
    if ft["pass_rate"] < SIGNOFF_MIN_PASS:
        why.append(f"{ft['pass_rate']:.0%} of first tries pass every gate, re-gated under the current gate code "
                   f"(< {SIGNOFF_MIN_PASS:.0%})")
    echo = echo_rates(kind, list(good.values()))
    if not echo["stock_phrase_gate"]:
        why.append(f"stock phrases over {ECHO_MAX:.0%}: "
                   f"{ {p: r for p, r in echo['stock_phrase_rates'].items() if r > ECHO_MAX} }")
    if bad:
        why.append(f"{len(bad)} kept check texts fail the current gates: {dict(list(bad.items())[:5])}")
    strata = POSTING_STRATA if kind == "postings" else RESUME_STRATA
    empty = [n for n, test in strata.items() if not any(test(cards[i]) for i in good)]
    if empty:
        why.append(f"strata with no kept check text that passes the current gates: {empty}")
    if ft["unattributed_failing"] or ft["unattributed_contexts_without_try"]:
        why.append(f"{ft['unattributed_rows']} recorded rows cannot be tied to a writer: {ft['unattributed_failing']} of them "
                   f"fail the current gates, and {ft['unattributed_contexts_without_try']} sit on a check context with no "
                   "counted try")
    return ft | {"stock_phrase_rates": echo["stock_phrase_rates"], "watched_phrase_rates": echo["watched_phrase_rates"]}, why


def require_signoff(kind: str, cards: dict[str, dict], want: list[str]) -> str | None:
    """None when `run` may render `want`: every id is in the check batch, or the check batch has a
    sign-off for the current blueprint and contexts. Otherwise the reason it may not."""
    ids = check_ids(kind)
    if not set(want) - set(ids):
        return None
    if not ids:
        return f"no check batch at {_rel(CHECK_IDS[kind])}: `pick` one and render it first"
    p = signoff_path(kind)
    if not p.exists():
        return (f"{len(set(want) - set(ids))} ids are outside the check batch and it has no sign-off: render "
                f"{_rel(CHECK_IDS[kind])}, read {SIGNOFF_READ[kind]}, then run `signoff`")
    so = json.loads(p.read_text())
    if so.get("fingerprint") != check_fingerprint(kind, cards, ids):
        return (f"the sign-off in {_rel(p)} is for another blueprint or other contexts: render the check "
                "batch again and sign off again")
    return None


def ctx_str(kind: str, card: dict) -> str:
    return json.dumps(context_of(kind, card), ensure_ascii=False)


def ctx_hash(context: str) -> str:
    return hashlib.sha1(context.encode()).hexdigest()[:16]


def texts_by_ctx(kind: str) -> dict[str, str]:
    """ctx hash -> text, first row wins."""
    out: dict[str, str] = {}
    for r in read_jsonl(OUT / f"{kind}.jsonl"):
        out.setdefault(r["ctx"], r["text"])
    return out


def rendered_texts(kind: str, cards: dict[str, dict] | None = None, clean: bool = True) -> dict[str, str]:
    """card id -> text, for every card whose current context has been rendered and whose text passes
    the gates (`fatal`). A text that fails is left out, its id printed and written to
    rendered/dropped_<kind>.jsonl with its tags (rows of ids outside `cards` stay in that file); so a training set built from this never holds a text
    a gate flags. clean=False returns every rendered text (for reports)."""
    cards = cards if cards is not None else load_cards(kind)
    store = texts_by_ctx(kind)
    out = {}
    dropped: list[dict] = []
    for i, card in cards.items():
        h = ctx_hash(ctx_str(kind, card))
        if h not in store:
            continue
        if clean and (tags := fatal(kind, card, store[h])):
            dropped.append({"id": i, "ctx": h, "tags": tags})
            continue
        out[i] = store[h]
    if clean:
        # the file is the drop list of every id any call has checked: this call replaces the rows of its own
        # ids (a call on a subset no longer wipes the rest) and leaves the others
        p = OUT / f"dropped_{kind}.jsonl"
        if dropped or p.exists():
            keep = [r for r in read_jsonl(p) if r["id"] not in cards]
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w") as f:
                for r in keep + dropped:
                    f.write(json.dumps(r) + "\n")
        if dropped:
            print(f"rendered_texts {kind}: dropped {len(dropped)} texts that fail the gates "
                  f"({dict(Counter(t for r in dropped for t in r['tags']).most_common())}); ids in "
                  f"{_rel(p)}: {[r['id'] for r in dropped[:20]]}")
    return out


# build_prompt's posting budget in characters (its max_posting_tokens default, ~4 chars a token)
POSTING_BUDGET = inspect.signature(build_prompt).parameters["max_posting_tokens"].default * 4


def training_postings(cards: dict[str, dict] | None = None) -> dict[str, str]:
    """card id -> the posting text a training prompt gets: the gated rendered text (rendered_texts) under the
    Title/Company/Locations header with 0-3 bank paragraphs (textclean.training_posting), the same format
    real test and demo postings get from textclean.real_posting_text. The bank is textclean.training_bank:
    rebuilt when its fingerprint is stale, real-test companies left out, every paragraph re-checked, and an
    empty bank raises. At the merge, assemble_training.build() (1) uses this in place of
    rendered_texts("postings", ...), (2) counts truncation against POSTING_BUDGET (6,000 chars, not 1200*4), and
    (3) runs missing_facts on redact(strip_markdown(resume)) and fit_posting(strip_markdown(posting), POSTING_BUDGET),
    the text the prompt holds, so a posting whose must-skill group falls past the budget (7 of the 726
    over-budget real postings on their v2 cards) is dropped, not kept with a label citing a skill the model
    cannot see."""
    cards = cards if cards is not None else load_cards("postings")
    bank = textclean.training_bank(ROOT)
    out, dropped = {}, {}
    for i, t in rendered_texts("postings", cards).items():
        full = textclean.training_posting(cards[i], t, i, bank)
        # the prompt holds the posting cut to build_prompt's budget: every card fact must survive the cut
        if (miss := posting_missing_facts(cards[i], textclean.fit_posting(textclean.strip_markdown(full), POSTING_BUDGET))):
            dropped[i] = miss
            continue
        out[i] = full
    if dropped:
        print(f"training_postings: dropped {len(dropped)} postings that lose a card fact at the {POSTING_BUDGET}-char "
              f"budget: {dict(list(dropped.items())[:10])}")
    return out


def wait_done(c: Adaption, did: str, settle_polls: int = 4, interval: float = 60.0):
    """Wait until every row is processed. Right after upload the dataset already reads
    "succeeded", so status alone is not enough: wait for processed_rows to reach total_rows,
    or to stop moving for settle_polls polls (rows the pipeline dropped)."""
    last, still = -1, 0
    while True:
        st = c.datasets.get_status(did)
        n, tot = (st.progress.processed_rows, st.progress.total_rows) if st.progress else (0, 0)
        print(time.strftime("%H:%M:%S"), st.status, f"{n}/{tot}", flush=True)
        if st.status == "failed":
            return st, n, tot
        if st.status == "succeeded":
            if tot and n >= tot:
                return st, n, tot
            still = still + 1 if n == last else 0
            if still >= settle_polls:
                return st, n, tot
        last = n
        time.sleep(interval)


def gate_download(kind: str, cards: dict, rows: list[dict], sent: dict[str, str], store: dict[str, str],
                  did: str = "") -> tuple[list[dict], list[dict]]:
    """Split downloaded rows into store rows ({"id", "ctx", "text"}) and quarantine rows (with "tags").
    A text that fails a gate never enters the store (`store` is updated in place with the good ones), so
    the next `run` renders its context again."""
    new, bad = [], []
    writer = writer_fingerprint(kind)  # recorded_tries counts a row as the current writer's try by this stamp
    for r in rows:
        rid, text = r.get("row_id", ""), (r.get("enhanced_completion") or "").strip()
        h = ctx_hash(base_ctx(sent[rid])) if rid in sent else None  # a retry's text is stored under the card's own context
        if not (h and text and h not in store):
            continue
        fb = len(json.loads(sent[rid]).get(FEEDBACK_KEY, ())) if FEEDBACK_KEY in sent[rid] else 0  # 0: a first try
        if (tags := fatal(kind, cards[rid], text) if rid in cards else ["no_card"]):
            bad.append({"id": rid, "ctx": h, "tags": tags, "dataset_id": did, "writer": writer, "feedback": fb, "text": text})
            continue
        new.append({"id": rid, "ctx": h, "text": text, "dataset_id": did, "writer": writer, "feedback": fb})
        store[h] = text
    return new, bad


def collect(c: Adaption, kind: str, cards: dict, state_path: Path, out_path: Path) -> None:
    state = json.loads(state_path.read_text())
    did = state["dataset_id"]
    st, n, tot = wait_done(c, did)
    print("DONE", st)
    if st.status != "succeeded":
        sys.exit(f"adapt ended with status {st.status}")
    rows = download_rows(c, did)
    raw = STATE / did
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "download.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sent = {r["row_id"]: r["card"] for r in csv.DictReader(open(ROOT / state["csv"], newline=""))}
    store = texts_by_ctx(kind)
    new, bad = gate_download(kind, cards, rows, sent, store, did)
    with out_path.open("a") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if bad:
        with quarantine_path(kind).open("a") as f:
            for r in bad:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"quarantined {len(bad)} texts that fail the gates "
              f"({dict(Counter(t for r in bad for t in r['tags']).most_common())}): {[r['id'] for r in bad[:20]]}; "
              f"`run` again renders them again (at most {MAX_REDO} redos per context)")
    batch_ids = state["ids"]
    missing = [x for x in batch_ids if ctx_hash(ctx_str(kind, cards[x])) not in store]
    state.update(finished=time.time(), processed=n, total=tot, complete=True,
                 rendered=state.get("rendered", 0) + len(new), unrendered=missing)
    state_path.write_text(json.dumps(state, indent=1))
    print(f"{did}: processed {n}/{tot}; wrote {len(new)} new texts to {out_path.relative_to(ROOT)}; "
          f"{len(batch_ids) - len(missing)} of {len(batch_ids)} batch ids now have text, {len(missing)} without")
    qa(kind, cards, [{"id": r["id"], "text": r["text"]} for r in new + bad])


def _move_rows(src: Path, dst: Path, hashes: set[str]) -> int:
    """Move the rows of `src` whose ctx is in `hashes` to the end of `dst`; returns how many moved."""
    keep, moved = [], []
    for r in read_jsonl(src):
        (moved if r["ctx"] in hashes else keep).append(r)
    if moved:
        with dst.open("a") as f:
            for r in moved:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with src.open("w") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(moved)


def redo_contexts(kind: str, cards: dict[str, dict], want: list[str]) -> None:
    """--redo, for a changed blueprint: the ids' current texts move to rendered/replaced_<kind>.jsonl and their failed
    tries to rendered/replaced_quarantine_<kind>.jsonl, so the next `run` renders every one of them again and its first
    tries (the sign-off pass rate, the MAX_REDO count) are the new blueprint's alone. A failed try under the old
    blueprint is no try of the new one."""
    hashes = {ctx_hash(ctx_str(kind, cards[x])) for x in want}
    n = _move_rows(OUT / f"{kind}.jsonl", OUT / f"replaced_{kind}.jsonl", hashes)
    q = _move_rows(quarantine_path(kind), OUT / f"replaced_quarantine_{kind}.jsonl", hashes)
    print(f"--redo: moved {n} rows to replaced_{kind}.jsonl and {q} failed tries to replaced_quarantine_{kind}.jsonl")


def stale_to_quarantine(kind: str, cards: dict[str, dict], want: list[str]) -> list[dict]:
    """A kept text of a wanted context that fails a gate added after it was rendered is a failed try of that context: its
    row moves from <kind>.jsonl to quarantine_<kind>.jsonl with its re-gated tags and stale_gate true (its writer,
    dataset_id and feedback count unchanged, so recorded_tries counts it as the try it was). The rest of `run` then
    treats it as any failed try: tries() counts it toward MAX_REDO, and the retry sends its findings (last_failed). A
    text that passes stays. Returns the moved rows."""
    by_ctx: dict[str, str] = {}
    for x in want:
        by_ctx.setdefault(ctx_hash(ctx_str(kind, cards[x])), x)
    p = OUT / f"{kind}.jsonl"
    keep, moved = [], []
    for r in read_jsonl(p):
        i = by_ctx.get(r["ctx"])
        if i is not None and r.get("text"):
            own = r["id"] in cards and ctx_hash(ctx_str(kind, cards[r["id"]])) == r["ctx"]
            if (tags := fatal(kind, cards[r["id"]] if own else cards[i], r["text"])):
                moved.append(r | {"tags": tags, "stale_gate": True})
                continue
        keep.append(r)
    if moved:
        with quarantine_path(kind).open("a") as f:
            for r in moved:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with p.open("w") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{len(moved)} kept texts fail a gate added after they were rendered "
              f"({dict(Counter(t for r in moved for t in r['tags']).most_common())}): moved to {quarantine_path(kind).name} "
              f"as failed tries, to render again with their findings: {[r['id'] for r in moved[:20]]}")
    return moved


def writer_fingerprint(kind: str) -> str:
    """The writer's instructions (universal prompt, blueprint, length)."""
    k = KINDS[kind]
    return hashlib.sha1("\x00".join((k["universal"], k["blueprint"], k["length"])).encode()).hexdigest()[:16]


def batch_tag(ids: list[str], contexts: list[str], writer: str = "") -> str:
    """Names a batch's csv and state file. It includes the contexts, so the same ids with new contexts
    (a changed blueprint-side card context) are a new batch, not a rerun of the old upload, and the writer's
    instructions, so a --redo under a new blueprint never picks up the old batch's state file and its old texts."""
    return hashlib.sha1(("\n".join(ids) + "\x00" + "\n".join(contexts) + ("\x00" + writer if writer else "")).encode()).hexdigest()[:10]


def unique_contexts(kind: str, cards: dict[str, dict], todo: list[str]) -> list[str]:
    """One id per distinct context, the first in todo order: the rows a batch uploads. Adapt drops exact-duplicate rows
    without a word (batch a1ab39c6b8 sent 593 rows with 6 duplicate contexts and got 587 back, one per duplicate pair
    missing), so a batch never sends one; the ids that share a context share its text (rendered_texts)."""
    return list({ctx_str(kind, cards[x]): x for x in reversed(todo)}.values())[::-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=KINDS)
    ap.add_argument("mode", choices=("estimate", "run", "report", "pick", "signoff"))
    ap.add_argument("ids_file")
    ap.add_argument("--max-credits", type=float, default=30.0)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--out-dir", help="where rendered rows and batch files go (default data/built/rendered)")
    ap.add_argument("--rows", help="report only: gate the rows of this jsonl by card id, ignoring their context hash")
    ap.add_argument("--n", type=int, default=30, help="pick only: batch size (every stratum gets 3 ids first)")
    ap.add_argument("--read-by", help="signoff only: the name of the person who read the check texts (see SIGNOFF_READ)")
    a = ap.parse_args()
    global OUT
    if a.out_dir:
        OUT = (ROOT / a.out_dir).resolve()

    cards = load_cards(a.kind)
    if a.mode == "pick":  # writes IDS_FILE: a stratified check batch
        ids = pick_ids(a.kind, cards, a.n)
        Path(a.ids_file).write_text("\n".join(ids) + "\n")
        print(f"wrote {len(ids)} {a.kind} ids to {a.ids_file}")
        return
    if a.mode == "signoff":  # IDS_FILE is ignored: the check batch is CHECK_IDS[kind]
        ids = check_ids(a.kind)
        if not ids:
            sys.exit(f"no check batch at {CHECK_IDS[a.kind]}")
        nums, why = signoff_problems(a.kind, cards, ids)
        print(json.dumps(nums, indent=1))
        if why:
            sys.exit("no sign-off: " + "; ".join(why))
        if not is_reader_name(a.read_by):
            sys.exit(f"no sign-off: pass --read-by with the name of the person who read {SIGNOFF_READ[a.kind]}")
        signoff_path(a.kind).write_text(json.dumps({"fingerprint": check_fingerprint(a.kind, cards, ids), "ids": len(ids),
                                                    "read_by": a.read_by, "read": SIGNOFF_READ[a.kind], "full_read": FULL_READ, "numbers": nums,
                                                    "time": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
        print(f"signed off: {_rel(signoff_path(a.kind))}")
        return
    out_path = OUT / f"{a.kind}.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    want = list(dict.fromkeys(x.strip() for x in Path(a.ids_file).read_text().split() if x.strip()))
    missing = [x for x in want if x not in cards]
    if missing:
        sys.exit(f"{len(missing)} ids have no card, e.g. {missing[:5]}")
    if a.mode == "run" and (why := require_signoff(a.kind, cards, want)):
        sys.exit(f"not running: {why}")
    if a.mode == "report":
        r = report(a.kind, cards, want, a.rows)
        probs = r.pop("problem_ids")
        print(json.dumps(r, indent=1))
        print("problem rows:", json.dumps(dict(list(probs.items())[:40])))
        return
    if a.redo:
        redo_contexts(a.kind, cards, want)
    c = client()
    # finish any launched batch first, so a rerun never pays twice for the same rows
    for sp in sorted(STATE.glob(f"render_{a.kind}_*.json")):
        st = json.loads(sp.read_text())
        if st.get("run_id") and not st.get("complete") and a.mode == "run":
            print("resuming", sp.name)
            collect(c, a.kind, cards, sp, out_path)
    # a kept text that a newer gate fails is a failed try now (in estimate mode too, so the estimate is of the batch a
    # run sends): the full render keeps improving as gates are added
    stale = {r["ctx"] for r in stale_to_quarantine(a.kind, cards, want)}

    store = texts_by_ctx(a.kind)
    todo = [x for x in want if ctx_hash(ctx_str(a.kind, cards[x])) not in store]
    failed = tries(a.kind)
    gave_up = [x for x in todo if failed[ctx_hash(ctx_str(a.kind, cards[x]))] > MAX_REDO]
    if gave_up:
        print(f"{len(gave_up)} ids failed the gates {MAX_REDO + 1} times and are not rendered again (they stay out of "
              f"training): {gave_up[:20]}")
        if (used := [x for x in gave_up if ctx_hash(ctx_str(a.kind, cards[x])) in stale]):
            print(f"{len(used)} of them had a kept text that a newer gate fails and no try left: {used[:20]}")
        todo = [x for x in todo if x not in set(gave_up)]
    uniq = unique_contexts(a.kind, cards, todo)
    print(f"{a.kind}: {len(want)} ids asked, {len(want) - len(todo) - len(gave_up)} already have text for their current "
          f"card, {len(todo)} to render as {len(uniq)} unique contexts "
          f"({sum(failed[ctx_hash(ctx_str(a.kind, cards[x]))] > 0 for x in uniq)} of them redos of a failed text)")
    if not todo:
        return

    # a retry sends the card's context plus the last failed try's findings (retry_context); the try count is part of
    # the tag, so a redo of a failed text is a new upload, not a rerun of the old one
    last = last_failed(a.kind)
    sent = {}
    for x in todo:
        h = ctx_hash(ctx_str(a.kind, cards[x]))
        sent[x] = retry_context(a.kind, cards[x], last[h].get("text"), last[h].get("tags", ())) if failed[h] else \
            ctx_str(a.kind, cards[x])
    print(f"{sum(FEEDBACK_KEY in sent[x] for x in uniq)} of {len(uniq)} rows carry the last try's findings ({FEEDBACK_KEY})")
    tag = batch_tag(todo, [f"{sent[x]}#{failed[ctx_hash(ctx_str(a.kind, cards[x]))]}" for x in todo],
                    writer_fingerprint(a.kind))
    batch_csv = OUT / "batches" / f"{a.kind}_{tag}.csv"
    state_path = STATE / f"render_{a.kind}_{tag}.json"
    batch_csv.parent.mkdir(parents=True, exist_ok=True)
    if not batch_csv.exists():
        with batch_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row_id", "card"])
            for x in uniq:
                w.writerow([x, sent[x]])
    state = json.loads(state_path.read_text()) if state_path.exists() else \
        {"kind": a.kind, "rows": len(uniq), "ids": todo, "csv": str(batch_csv.relative_to(ROOT)), "writer": writer_fingerprint(a.kind)}
    if "dataset_id" not in state:
        up = c.datasets.upload_file(str(batch_csv), name=f"render-{a.kind}-{tag}")
        state["dataset_id"] = up.dataset_id
        state_path.write_text(json.dumps(state, indent=1))
        print("UPLOADED", up.dataset_id)
    did = state["dataset_id"]
    for _ in range(120):  # adapt answers 409 until the import has a row_count
        st = c.datasets.get_status(did)
        if st.row_count or st.status == "failed":
            break
        time.sleep(5)
    kw = adapt_kwargs(a.kind)
    est = c.datasets.adapt(did, estimate=True, **kw)
    print("ESTIMATE", est)
    state["estimate_credits"] = est.estimated_credits_consumed
    state_path.write_text(json.dumps(state, indent=1))
    if a.mode == "estimate":
        return
    if est.estimated_credits_consumed > a.max_credits:
        sys.exit(f"estimate {est.estimated_credits_consumed} > --max-credits {a.max_credits}; not running")
    r = c.datasets.adapt(did, estimate=False, **kw)
    print("RUN", r)
    state.update(run_id=r.run_id, started=time.time())
    state_path.write_text(json.dumps(state, indent=1))
    time.sleep(30)
    collect(c, a.kind, cards, state_path, out_path)


if __name__ == "__main__":
    main()
