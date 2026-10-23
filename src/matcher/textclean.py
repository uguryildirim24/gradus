"""Make the text the model reads look like real resumes and real postings.

Everything here is deterministic code, never Adaption:

  strip_markdown   Adaption writes ** and '*   ' bullets; real text has almost none.
  redact           a real resume has a name and contact block; the model sees "Candidate".
  compose_posting  one posting format ("Title:/Company:/Locations:" header + body) for
                   training and for real postings; ATS-style locations in some rows;
                   EEO, accommodation and privacy paragraphs harvested from real postings
                   (training_bank; nothing that could contradict a card).
  fit_posting      cut a long posting to a budget without losing the work-authorization
                   sentences, the pay / years / degree / enrollment / work-mode lines or the
                   requirement sentences (a plain cut drops those); card_fact_survival measures
                   it against the facts extracted from the real postings.

Run `python -m matcher.textclean REPORT` to see how fit_posting does on the real postings.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from pathlib import Path

# ------------------------------------------------------------------ markdown


def strip_markdown(text: str) -> str:
    """Plain text: no **bold**, _italics_, #headings, '*   ' bullets, [links](url), rules or code fences."""
    t = html.unescape(text).replace("\r\n", "\n").replace(" ", " ")
    t = re.sub("[\u200b-\u200d\u2060\ufeff]", "", t)  # zero-width characters
    t = re.sub(r"```[a-z]*\n?", "", t)
    t = re.sub(r"\[([^\]\n]+)\]\((?:https?://|mailto:)[^)\s]*\)", r"\1", t)
    t = re.sub(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$", "", t, flags=re.M)  # horizontal rules
    t = re.sub(r"^[ \t]*#{1,6}[ \t]+", "", t, flags=re.M)
    t = re.sub(r"^([ \t]*)[*+•·▪●◦][ \t]+", r"\1- ", t, flags=re.M)  # bullets
    t = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t, flags=re.S)
    t = re.sub(r"__(.+?)__", r"\1", t, flags=re.S)
    t = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])", r"\1", t)  # *italic*
    t = re.sub(r"(?<![\w_])_(?!\s)([^_\n]+?)(?<!\s)_(?![\w_])", r"\1", t)
    t = re.sub(r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(\|[ \t]*:?-{3,}:?[ \t]*)*\|?[ \t]*$", "", t, flags=re.M)  # table rules
    t = t.replace("`", "")
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


MARKDOWN_LEFTOVER = re.compile(r"\*\*|__|^\s*#{1,6}\s|^\s*[*+]\s{1,4}\S|\]\(https?://|^\s*```", re.M)


def has_markdown(text: str) -> bool:
    return MARKDOWN_LEFTOVER.search(text) is not None


# ------------------------------------------------------------------ redaction

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# US numbers, and international ones written with a leading +country code ('+90 532 123 45 67', '+44 20 7946 0958')
PHONE = re.compile(r"(?<![\w.])(?:\+?\d{1,2}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?![\w])"
                   r"|(?<![\w+])\(?\+\d{1,3}\)?(?:[\s.-]?\(?\d{1,4}\)?){2,5}(?![\w])")
# a scheme, www, a profile host, or a hosted personal site. Not a bare 'name.io': 'Socket.io' is a skill.
# A path stops before trailing punctuation ('github.com/jane/repo; analysis' keeps the ';').
_PATH = r"(?:\S*[^\s;,.:!?)\]'\"])?"
URL = re.compile(rf"(?:https?://|www\.){_PATH}|(?:linkedin|github|gitlab|behance|dribbble|twitter|x|medium|kaggle|orcid)\.(?:com|io|dev|org)/{_PATH}"
                 rf"|\b[\w-]+\.(?:github\.io|gitlab\.io|vercel\.app|netlify\.app|wordpress\.com|wixsite\.com)(?:/{_PATH})?", re.I)
STREET_SUFFIX = (r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl|Parkway|Pkwy|Terrace|Circle)")
STREET = re.compile(rf"\b\d{{1,6}}[ ]+(?:[NSEW]\.?[ ]+)?(?:[A-Z][\w.'-]*[ ]+){{1,4}}(?i:{STREET_SUFFIX})\b\.?"
                    r"(?:[ ,]+(?i:Apt|Suite|Ste|Unit|#)\.?[ ]*[\w-]+)?")
_STATES = ("AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|"
           "PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC|PR")
# outside the header a street address must look like one: a city and state, or a ZIP, after it
# ("2 Oak Street clinics" and "Migrated 3 Google Drive accounts" are resume content)
STREET_BODY = re.compile(STREET.pattern + rf"(?=,?[ ]+(?:[A-Z][\w.'-]*[ ]*){{1,3}},[ ]*(?:{_STATES})\b|,?[ ]+\d{{5}}\b)")
STREET_HEAD = re.compile(STREET.pattern + r"(?=,[ ]*[A-Z])")
ZIP = re.compile(rf"(?<=[ ,](?:{_STATES}))[ ,]+\d{{5}}(?:-\d{{4}})?\b")
CITY_STATE = re.compile(rf"^[A-Z][A-Za-z.'\- ]+,\s*(?:{_STATES})$")  # "Boston, MA" (not "Jane Doe, RN")
# A bare lower-case domain ('janedoe.com'; 'Socket.io' and 'ASP.NET' are capitalised skills). In the body .io, .ai and
# .tech are left alone: there they name a project or a product ('built ledger.io') more often than a contact.
DOMAIN = re.compile(rf"(?<![\w@/.-])(?:[a-z0-9-]+\.)+(?:com|org|net|io|dev|me|tech|gov|edu|ai|co|us|info)(?:/{_PATH})?(?![\w.])")
DOMAIN_BODY = re.compile(rf"(?<![\w@/.-])(?:[a-z0-9-]+\.)+(?:com|org|net|dev|me|gov|edu|co|us|info)(?:/{_PATH})?(?![\w.])")
SEP = re.compile(r"\s*[|•·▪●]\s*|\s+[–—]\s+|\s{3,}")
SECTION = re.compile(
    r"^(?:professional\s+|career\s+|executive\s+)?(summary|profile|objective|about( me)?|education|experience|work experience|"
    r"employment|skills|technical skills|core competencies|licenses?|certifications?|projects?|coursework|volunteer\w*|"
    r"activities|leadership|awards|honors|publications|languages|interests|references)\b.*$", re.I)
CITY_ST = re.compile(r"^[A-Z][A-Za-z.'\- ]+,\s*(?:[A-Z]{2}|[A-Z][a-z]+)(?:\s+\d{5})?$")
_HONORIFIC = re.compile(r"^(?:Dr|Mr|Ms|Mrs|Mx|Prof)\.?\s+")
_PARTICLE = {"de", "da", "del", "della", "der", "den", "di", "du", "la", "le", "van", "von", "bin", "binti", "al", "el", "y", "e"}
# a credential after the name ('RN', 'BSN', 'MSc', 'Pharm.D.', 'PhD'): two capitals or a dot, at most 9 characters
_CRED = re.compile(r"[A-Z][A-Za-z.]{0,8}")
DOC_TITLE = re.compile(r"^(?:resume|résumé|curriculum vitae|cv)$", re.I)
# A headline in the name's place ("Data Analyst", "Registered Nurse, BSN", "SENIOR DATA ANALYST"): up to 5 words with
# no digit or separator, the part before any comma ending in a role noun. A name never ends in one of these.
_ROLE = (r"(?:analyst|engineer|developer|scientist|technician|technologist|nurse|manager|assistant|specialist|coordinator|"
         r"designer|consultant|teacher|educator|pharmacist|accountant|administrator|representative|professional|student|"
         r"graduate|intern|researcher|therapist|paralegal|chemist|biologist|programmer|officer|aide|supervisor|director|"
         r"associate|instructor|tutor|advisor|counselor|worker|operator|planner|economist|statistician|librarian|"
         r"electrician|mechanic|machinist|welder|caregiver|clinician|physician|hygienist|dietitian|psychologist|"
         r"leader|lead|executive|strategist|auditor|inspector|estimator|recruiter|writer|editor)s?")
HEADLINE = re.compile(rf"^(?:[A-Za-z&/+.'’()-]+\s+){{0,6}}{_ROLE}(?:\s*,[^\d|•·▪●@]*)?$", re.I)
# words that are never part of a name: a headline in the name's place ('Biology Major', 'Aspiring Biochemist',
# 'Bilingual (English/Spanish) Customer Service Representative') stays in the resume
_NOT_NAME = {
    "major", "minor", "aspiring", "bilingual", "trilingual", "multilingual", "senior", "junior", "entry", "entry-level",
    "certified", "licensed", "registered", "experienced", "recent", "motivated", "dedicated", "candidate", "resume",
    "biology", "chemistry", "biochemistry", "physics", "mathematics", "math", "economics", "nursing", "pharmacy", "finance",
    "accounting", "marketing", "engineering", "computer", "science", "sciences", "data", "software", "mechanical",
    "electrical", "civil", "chemical", "health", "healthcare", "public", "business", "administration", "psychology",
    "sociology", "english", "spanish", "customer", "service", "services", "sales", "clinical", "medical", "research",
    "laboratory", "lab", "information", "technology", "student", "graduate", "undergraduate", "biomedical", "environmental",
    "teaching", "programming", "consulting", "banking", "manufacturing", "recruiting", "writing", "editing", "tutoring",
    "university", "college", "institute", "school", "academy", "polytechnic", "hospital", "center", "centre",
    "and", "of", "in", "for", "with", "the", "&", "/",
}
# State codes that are also credentials after a name ('John Smith, MD', 'Ana Ruiz, PA', 'Wei Chen, MS'): 'X Y, MD'
# is a place only when the place table knows it.
_CRED_STATES = {"md", "pa", "ms", "ma"}


def _is_place(s: str) -> bool:
    """'New York, NY', 'Los Angeles, California', 'Grand Rapids, MI': a US city and state, not a name with a
    credential. Real resumes that lost their name line (the demo's own pass) can open with one."""
    from .metros import STATE_CAPITALS, STATES, metro_of

    name, _, rest = s.partition(",")
    code = " ".join(rest.strip().rstrip(".").lower().split())
    code = STATES.get(code, code)
    if not name.strip() or code not in set(STATES.values()):
        return False
    return code not in _CRED_STATES or metro_of(s) is not None or s.strip() in STATE_CAPITALS.values()


def _is_name_token(t: str, lower_ok: bool = False) -> bool:
    w = t.strip(".,")
    if w.lower() in _NOT_NAME or re.search(r"(?:ists?|ology|ogy|ics)$", w.lower()) or re.fullmatch(_ROLE, w, re.I):
        return False
    return bool(w) and (w[0].isupper() or lower_ok) and all(c.isalpha() or c in ".'’-" for c in w)


PRONOUNS = re.compile(r"\s*\((?:she|he|they|ze|xe|any)(?:\s*/\s*\w+){1,2}\)", re.I)  # 'Jane Doe (she/her)'


def _name_cred(s: str) -> tuple[bool, str, str]:
    """(is the line a person's name, the credentials after it as ', RN, BSN', a headline after it).
    'Maria Gonzalez, RN, BSN', 'Dr. Wei Chen', 'José de la Cruz', 'JANE DOE' and 'jane doe' are names, and
    'Jane Doe, Data Analyst' is a name with a headline; 'Biology Major' and 'Boston, MA' are not names."""
    s = PRONOUNS.sub("", s).strip()
    if _is_place(s):
        return False, "", ""
    name, _, rest = s.partition(",")
    toks = _HONORIFIC.sub("", name.strip()).split()
    caps = [t for t in toks if t.lower() not in _PARTICLE]
    lower_ok = name == name.lower()  # 'jane doe'
    if not 2 <= len(caps) <= 4 or len(toks) > 6 or not all(_is_name_token(t, lower_ok) for t in caps):
        return False, "", ""
    if rest.strip() and HEADLINE.match(rest.strip()):
        return True, "", rest.strip()
    creds = [c.strip() for c in rest.split(",")] if rest.strip() else []
    if not all(_CRED.fullmatch(c) and (sum(x.isupper() for x in c) >= 2 or "." in c) for c in creds):
        return False, "", ""
    return True, "".join(f", {c}" for c in creds), ""


CONTACT_WORD = re.compile(r"\b(?:email|e-mail|phone|tel|mobile|cell|linkedin|github|portfolio|website|address)\b", re.I)


def _scrub_line(line: str, header: bool = True) -> str:
    for rx in (EMAIL, URL, DOMAIN if header else DOMAIN_BODY, STREET if header else STREET_BODY, PHONE):
        line = rx.sub("", line)
    line = ZIP.sub("", line)
    return re.sub(r"(\S)[ \t]+([,;])", r"\1\2", re.sub(r"\b(at|in|to|from)[ \t]*,[ \t]*", r"\1 ", line))  # 'at , Boston' -> 'at Boston'


def _clean_contact_line(line: str) -> str:
    """Drop contact segments from one header line; keep what is left (a city)."""
    parts = [p.strip(" ,;:-") for p in SEP.split(line)]
    keep = []
    for p in parts:
        if not p:
            continue
        if re.fullmatch(r"(?:email|e-mail|phone|tel|mobile|cell|linkedin|github|portfolio|website|web|address)\s*:?", p, re.I):
            continue
        q = _scrub_line(re.sub(r"^(?:email|e-mail|phone|tel|mobile|cell|linkedin|github|portfolio|website|web|address)\s*:\s*", "", p, flags=re.I)).strip(" ,;:-|")
        if q and re.search(r"[A-Za-z]{2}", q):
            keep.append(q)
    return " | ".join(keep)


def _contactish(s: str) -> bool:
    return "@" in s or bool(re.search(r"\d", s) or CONTACT_WORD.search(s) or URL.search(s) or DOMAIN.search(s))


def _header_len(lines: list[str]) -> int:
    """How many top lines form the header block: up to 8, ending at a blank line or a section heading."""
    n = 0
    for i, ln in enumerate(lines[:8]):
        s = ln.strip()
        if i == 0 and SECTION.match(s):
            break
        if i > 0 and (not s or SECTION.match(s) and len(s) < 60):
            break
        n = i + 1
    return n


def redact(resume_text: str) -> str:
    """The resume the model sees: the top name line becomes "Candidate" (a credential after the name stays:
    'Candidate, RN'; a first line that is not a name, like 'Biology Major', stays as written); contact segments (email,
    phone, URL, street, ZIP) of the header lines go and the rest of those lines (city, headline,
    summary) stays as written; emails, phones and URLs are removed everywhere, and a street address
    in the body only when a city and state or a ZIP follows it."""
    lines = resume_text.strip().split("\n")
    n = _header_len(lines)
    block, rest = lines[:n], lines[n:]
    out = ["Candidate"]
    name_at = 1 if block and DOC_TITLE.match(block[0].strip()) else 0  # "Resume" above the name
    for i, ln in enumerate(block):
        s = ln.strip()
        if i < name_at:
            continue
        first_seg = SEP.split(s, maxsplit=1)[0].strip(" ,;:-")
        if i == name_at and HEADLINE.match(s) and not SEP.search(s) and not _name_cred(s)[0]:
            out.append(s)  # no name line: the resume opens with its headline, which stays
            continue
        if i == name_at:
            is_name, cred, headline = _name_cred(s)
            seg_name, seg_cred, seg_headline = (True, "", "") if first_seg.lower() == "candidate" else _name_cred(first_seg)
            if s.lower() == "candidate":
                continue
            if is_name:
                out[0] = "Candidate" + cred  # 'Maria Gonzalez, RN' -> 'Candidate, RN'
                out += [headline] if headline else []
                continue
            if "@" in s or PHONE.search(s) or seg_name:
                # "Jane Doe | Boston, MA | 617.555.0182 | jane@x.com" or "Jane Doe (617) 555-0182 jane@x.com":
                # the first segment of the name line is the name; the rest (a city) stays
                segs = [x for x in _clean_contact_line(s).split(" | ") if x]
                if seg_name:
                    out[0] = "Candidate" + seg_cred
                if segs and not CITY_STATE.match(segs[0]) and (seg_name or _name_cred(segs[0])[0]):
                    segs = [seg_headline] if seg_headline else []
                    segs += [x for x in _clean_contact_line(s).split(" | ") if x][1:]
                if segs:
                    out.append(" | ".join(segs))
                continue
        if not _contactish(s) or _scrub_line(s) == s and not CONTACT_WORD.search(s):
            out.append(s)  # a headline, a city or a summary line: unchanged
            continue
        kept = _clean_contact_line(s)
        if kept:
            out.append(kept)

    # the writer's own location line can repeat the city a fake street line leaves behind ('Boston, MA' twice)
    out = [x for i, x in enumerate(out) if i == 0 or x.strip().lower() != out[i - 1].strip().lower()]

    def body_line(x: str) -> str:
        if EMAIL.search(x) or URL.search(x) or DOMAIN_BODY.search(x) or STREET_BODY.search(x) or PHONE.search(x):
            return _scrub_line(x, header=False)
        return x
    body = "\n".join(body_line(x) for x in rest)
    text = "\n".join(out) + ("\n" + body if body.strip() else "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"^[ \t]*(?:[-*]\s*)?(?:email|phone|linkedin|github|website|portfolio|address)\s*:?\s*$", "", text, flags=re.I | re.M)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def has_contact(text: str) -> bool:
    """An email, phone, URL or domain anywhere ('.io', '.ai' and '.tech' domains only in the header block:
    in the body they are projects), or a street address: with a city and state or a ZIP after it anywhere,
    or in the top 8 lines with a comma and a place after it ('22 Lake Boulevard, Springfield'). A bare
    '5 Hospital Place' in a summary line is resume content."""
    lines = text.split("\n")
    head = "\n".join(lines[:8])
    block = "\n".join(lines[:_header_len(lines)])
    return (any(rx.search(text) for rx in (EMAIL, PHONE, URL, DOMAIN_BODY, STREET_BODY))
            or bool(STREET_HEAD.search(head) or DOMAIN.search(block)))


# ------------------------------------------------------------------ work authorization / requirements

# Sentences that state a work-authorization rule. EEO lists ("...citizenship status...") only count
# when a strong term is also present.
_STRONG_WA = re.compile(
    r"sponsor\w*|\bvisas?\b(?!\s+(?:card|inc|debit|credit|payment))|\bH-?1-?B\b|\bSTEM[ -]OPT\b|\bI-?983\b|\bF-?1\b|\bJ-?1\b|"
    # upper case only ('opt out', 'ear protection'); TN only as a visa ('TN is the state' is a place)
    r"(?-i:\b(?:OPT|CPT|EAR)\b)|(?-i:\bTN)[ -](?:visas?|status|classification|holders?)|green card|work(?:ing)? authori[sz]\w*|(?:legally|lawfully)\s+(?:authori[sz]ed|eligible|permitted|able)|"
    r"authori[sz]ed to work|authori[sz]ation to work|employment authori[sz]ation|work eligibility|work status|"
    r"eligib\w+ to work|right to work|employment eligibility|work permit|e-?verify|international students?|"
    r"U\.?S\.? citizen\w*|U\.?S\.? nationals?\b(?! security)|citizens? (?:of|or)|permanent resident\w*|U\.?S\.? persons?|"
    r"\bITAR\b|export[- ]control\w*|security clearance|\bclearance\b|immigration|naturali[sz]ed|stateless|"
    # residence and nationality rules ('Applications from outside the United States will not be considered',
    # 'a UK Eyes Only programme with nationality restrictions')
    r"\bnationality (?:restrictions?|requirements?)\b|(?-i:\bEyes Only\b)|"
    r"\b(?:applications?|applicants?|candidates?)\b[^.]{0,40}\b(?:outside|from outside) (?:of )?the (?:U\.?S\.?|US\b|United States)|"
    r"\b(?:reside|residing|based) (?:in|within) the (?:U\.?S\.?|US\b|United States)\b", re.I)
_EEO = re.compile(r"without regard to|equal (?:employment )?opportunity|protected (?:veteran|class|characteristic)|"
                  r"affirmative action|discriminat|all qualified applicants", re.I)


def is_work_auth(sentence: str) -> bool:
    if not _STRONG_WA.search(sentence):
        return False
    if _EEO.search(sentence) and not re.search(
            r"sponsor|visa|authori[sz]ed to work|eligib\w+ to work|e-?verify|work authori|H-?1B|(?-i:\bOPT\b)|clearance|ITAR|U\.?S\.? citizen", sentence, re.I):
        return False
    return True


_REQ = re.compile(
    r"\b(?:requires?d?|requirements?|must|minimum|qualif\w+|preferred|bonus|nice to have|proficien\w+|experience|knowledge|familiar\w*|"
    r"degree|bachelor\w*|master\w*|ph\.?d|doctorate|pursuing|enrolled|students?|graduat\w+|years?|gpa|ability|skills?|certif\w+|"
    r"licen[sc]\w+|eligible|availab\w+|hours|on-?site|hybrid|remote|salary|compensation|pay|rate|per hour|internship|duration)\b|\$\s?\d",
    re.I)
# Headings of the sections whose lines are requirements. Not 'role', 'position', 'duties', 'responsibilities',
# 'what we' or 'additional': an "About the Role" section is mostly duties and benefits, and as 'req' it filled
# the budget before the lines under an unrecognised requirement heading ("We're excited about you because").
_REQ_HEAD = re.compile(r"qualif|requirement|what you(?:['’]ll| will)? (?:need|bring|have)|skills|must|minimum|basic|preferred|"
                       r"eligib|who you|about you|you have|you bring|you.ll (?:need|bring|have)|looking for|ideal|bonus|plus|"
                       r"education|experience|compensation|pay|salary|location|schedule|logistics|work authori|internship", re.I)
# Duty sections: below the requirement lines, above intros and company text (a tool named only among the duties
# is still evidence of what the job uses).
_DUTY_HEAD = re.compile(r"responsibilit|duties|what you(?:['’]ll| will)? (?:do|work on|be doing)|in this role|the role|"
                        r"about the (?:role|job|position)|role|position|day[- ]to[- ]day|your impact|job description", re.I)
_BOILER_HEAD = re.compile(r"equal|eeo|benefit|perks|accommodat|privacy|about (?:us|the company)|why |who we are|our (?:company|mission|values|culture)|"
                          r"pay transparency|how to apply|application (?:instructions|process)|disclaimer|notice|diversity|"
                          r"inclusion|follow us|stay connected|life at", re.I)
_BOILER = re.compile(
    r"equal (?:employment )?opportunity|without regard to|protected (?:veteran|class)|affirmative action|reasonable accommodation|"
    r"accommodations?|pay transparency|nondiscrimination|\b401\(?k\)?|health(?:care)?,? dental|medical, dental|paid (?:time off|holidays)|"
    r"parental leave|employee discount|privacy (?:notice|policy)|data protection|consent to receive|text messag\w+|msg (?:&|and) data|"
    r"e-?mail address(?:es)? (?:ending|from)|we are an? .{0,40}employer|committed to (?:diversity|building an inclusive)|"
    r"background check|drug[- ]free|equal opportunity", re.I)


# A period that ends one of these is not a sentence end: "To conform to U.S. Government export regulations",
# "Varda Space Industries, Inc. participates in the U.S. Department of ...", "22 C.F.R. §120.62", "John F. Smith".
_ABBREV = re.compile(
    r"(?:^|[\s(\"'“‘])(?:(?:[A-Za-z]\.){2,}|[A-Z]\.|Ph\.D\.|Inc\.|Ltd\.|Corp\.|Co\.|Dr\.|Mr\.|Ms\.|Mrs\.|St\.|No\.|Nos\.|"
    r"Jr\.|Sr\.|vs\.|Mt\.|Ft\.|Gov\.|Govt\.|Dept\.|Prof\.|Sec\.|Rev\.|Assn\.|Bros\.|approx\.|cf\.)$")
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'“‘])")


def split_sentences(line: str) -> list[str]:
    """Sentences of one line. No split after an abbreviation (U.S., U.K., Inc., Dr., C.F.R., e.g., a single
    initial): a work-authorization sentence cut at 'U.S.' lost its head ('To conform to U.S.') or its tail."""
    line = line.strip()
    parts, start = [], 0
    for m in _BOUNDARY.finditer(line):
        if _ABBREV.search(line[start:m.start()]):
            continue
        parts.append(line[start:m.start()])
        start = m.end()
    parts.append(line[start:])
    return [p for p in parts if p]


# ------------------------------------------------------------------ one posting format

HEADER_KEYS = ("Title", "Company", "Locations")
_ATS_STYLE = {
    "new york, ny": "NYC", "new york city, ny": "NYC", "san francisco, ca": "SF", "los angeles, ca": "LA",
    "boston, ma": "Boston, MA", "seattle, wa": "Seattle, WA",
}


def h32(key: str) -> int:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)


def format_locations(locations: list[str] | tuple[str, ...], work_mode: str = "onsite", key: str = "", ats_share: float = 0.35) -> str:
    """Header locations, every one of them (a cut list hid 3 of SpaceX's 8 sites); with ATS style in some
    rows ("NYC", "SF", "Remote in USA"; Washington, DC keeps its name: real headers never write a bare 'DC')."""
    locs = [x for x in locations if x]
    ats = (h32(key + "|loc") % 1000) / 1000 < ats_share
    if not locs:
        return "Remote in USA" if work_mode == "remote" else "United States"
    if ats:
        locs = [_ATS_STYLE.get(x.lower(), x) for x in locs]
        if work_mode == "remote":
            locs = ["Remote in USA", *locs] if h32(key + "|r") % 2 else locs
    elif work_mode == "remote":
        locs = [f"{x} (Remote)" if h32(key + "|r") % 3 == 0 else x for x in locs]
    return "; ".join(locs)


def split_synthetic(text: str) -> tuple[str, str, str]:
    """(title, company, body) from a rendered posting whose first line is the title and second the
    company name. Missing or long second lines, and second lines that read as a sentence, give company ''."""
    lines = strip_markdown(text).split("\n")
    nz = [i for i, x in enumerate(lines) if x.strip()]
    if not nz:
        return "", "", ""
    title = re.sub(r"^(?:title|position|job title)\s*:\s*", "", lines[nz[0]].strip(), flags=re.I)
    company, start = "", nz[0] + 1
    if len(nz) > 1:
        c = re.sub(r"^(?:company|employer|organization)\s*:\s*", "", lines[nz[1]].strip(), flags=re.I)
        # a line ending in '.', '!' or '?' is a sentence, not a name, unless the '.' ends a company suffix ('Willowmere Inc.')
        if len(c.split()) <= 7 and len(c) <= 60 and not re.search(r"[!?]$|(?<!\bInc)(?<!\bCo)(?<!\bCorp)(?<!\bLtd)(?<!\bLLC)\.$", c) \
                and not c.startswith("-"):
            company, start = c, nz[1] + 1
    return title, company, "\n".join(lines[start:]).strip()


def header(title: str, company: str, locations: str) -> str:
    return f"Title: {title.strip()}\nCompany: {company.strip() or 'Not listed'}\nLocations: {locations.strip()}"


def compose_posting(title: str, company: str, locations: str, body: str, boiler: list[str] | None = None, key: str = "") -> str:
    """Header + plain-text body (+ 0-3 harvested boilerplate paragraphs for about 60% of rows when
    `boiler` is given: pass it for training postings, not for real ones)."""
    body = strip_markdown(body)
    if boiler:
        body = append_boilerplate(body, boiler, key)
    return f"{header(title, company, locations)}\n\n{body}".strip()


def training_posting(card: dict, rendered: str, rq_id: str, bank: list[str] | None = None) -> str:
    """The posting text for a training row: the rendered text (line 1 title, line 2 company) under a
    Title/Company/Locations header from the card, ATS-style locations in some rows, and 0-3 boilerplate
    paragraphs in about 60% of rows. `rq_id` is the row key; the same id always gives the same text."""
    title, company, body = split_synthetic(rendered)
    locs = format_locations(card.get("locations") or [], card.get("work_mode", "onsite"), rq_id)
    return compose_posting(title or card.get("title", ""), company, locs, body, boiler=bank, key=rq_id)


# ------------------------------------------------------------------ boilerplate bank
#
# The bank is appended to synthetic postings, so a paragraph must be true of every job: only equal-
# opportunity, accommodation and applicant-privacy text. A paragraph that names a place, a work mode,
# pay, a job type, a check, a number, a brand, a link or a requirement could contradict the card it
# lands under, so it is dropped (BANK_BANNED). BANK_VERSION and the patterns below are hashed into the
# bank file; load_bank rebuilds it when they (or the exclusion set or the postings file) change.

_REQ_PARA = re.compile(
    r"\b(?:requires?d?|must|minimum|qualifications?|years? of|bachelor|degree|pursuing|enrolled|proficien\w+|responsibilit\w+|"
    r"you will|you.ll|we are looking|candidate must|gpa)\b", re.I)

BANK_VERSION = 4
_BANK_TOPIC = re.compile(
    r"equal (?:employment )?opportunity|without regard to|protected (?:veteran|class|characteristic|status)|affirmative action|"
    r"nondiscriminat\w*|non-discriminat\w*|discriminat\w*|reasonable accommodation|accommodations?\b|disabilit\w+|"
    r"privacy (?:notice|policy|statement)|personal (?:data|information)|applicant data|data protection", re.I)
BANK_BANNED = re.compile(
    r"https?://|www\.|\.(?:com|org|net|io|gov|edu)\b|@|\d|\$|"
    r"\b(?:salar\w*|pay|paid|payment|wages?|compensation|bonus\w*|stipends?|overtime|commission|equity|stock|401|"
    r"interns?|internships?|co-?ops?|full[- ]time|part[- ]time|temporary|seasonal|contract(?:or)?s?|"
    r"hybrid|remote\w*|in[- ]person|on-?site|offices?|headquarter\w*|campus|travel\w*|relocat\w*|housing|commut\w*|shifts?|schedules?|"
    r"weekends?|locations?|cit(?:y|ies)|states? of|county|"
    r"background|drug|screen\w*|references?|e-?verify|verif\w*|criminal|convictions?|arrest\w*|fair chance|"
    r"benefits? (?:package|program|plan)|medical|dental|vision|insurance|retirement|vacation|holidays?|leave|"
    r"phone|call|telephone|e-?mail|text|recruiters?|deadline|"
    r"require\w*|qualif\w*|meet every|every (?:single )?requirement|all (?:of )?the requirements|degree|years?|experience|"
    r"students?|graduat\w*|apply even|encourage you to apply|"
    r"sponsor\w*|visa|citizen\w*|immigration|authori[sz]\w*|"
    # text that points at something a synthetic posting does not have, or is not US wording
    r"physical demands|headings? above|this clause|this form|home page|gender reassignment|civil partnership\w*|"
    # a pointer to a page or a link the synthetic posting does not have ("the notice at the bottom of this page")
    r"this page|bottom of|websites?|web ?site|click|links?|below|here\.|posters?)(?!\w)", re.I)
# capitalised words that are not a brand or a place
_BANK_CAPS_OK = {
    "We", "Our", "All", "The", "This", "It", "If", "Any", "In", "As", "To", "For", "At", "Please", "Applicants", "Applicant",
    "Candidates", "Individuals", "Employment", "Equal", "Opportunity", "Employer", "Employers", "Affirmative", "Action",
    "Americans", "Disabilities", "Act", "Title", "VII", "Civil", "Rights", "Privacy", "Policy", "Notice", "Statement",
    "Personal", "Data", "Information", "Accommodation", "Accommodations", "Reasonable", "EEO", "EOE", "ADA", "Federal",
    "Law", "Laws", "Know", "Your", "You", "Disability", "Veterans", "Veteran", "Protected", "Diversity", "Inclusion",
    "Equity", "Nondiscrimination", "Non-Discrimination", "Discrimination", "Human", "Resources", "Team", "Talent",
    "Acquisition", "People", "Recruiting", "Applicants", "Company", "Organization", "Employee", "Employees", "Statement:",
    "Additionally", "Further", "Furthermore", "Also", "When", "By", "Should", "Where", "Upon", "Such", "These", "Those",
    "Each", "Every", "Everyone", "No", "Not", "And", "Or", "Of", "On", "With", "From", "Who", "That", "Which", "Is", "Are",
    "Be", "Being", "May", "Will", "Can", "Information.", "Notice.", "Policy.", "EEO.", "Act.", "Employer.", "LGBTQ",
    "LGBTQ+", "LGBTQIA", "LGBTQIA+", "U.S.", "US", "USA", "United", "States", "Genetic", "Nondiscrimination", "GINA",
    "Pregnant", "Workers", "Fairness", "Rehabilitation", "Section", "Vietnam", "Era", "Readjustment", "Assistance",
    "VEVRAA", "OFCCP", "EEOC", "Commission", "Programs", "Contract", "Compliance", "Department", "Labor",
}


def _bank_caps_ok(p: str) -> bool:
    """Every capitalised word is generic (no company, product, place or person name)."""
    for m in re.finditer(r"(?<![\w'’-])[A-Z][\w'’.+-]*", p):
        w = m.group().rstrip(".,;:’'")
        if w in _BANK_CAPS_OK or w.rstrip("s") in _BANK_CAPS_OK:
            continue
        before = p[:m.start()].rstrip()
        if not before or before.endswith((".", "!", "?", ":", "(", "\"", "“")):  # sentence start: a common word
            if w.lower() in _COMMON_START:
                continue
        return False
    return True


_COMMON_START = {"we", "our", "all", "the", "this", "it", "if", "any", "in", "as", "to", "for", "at", "please", "applicants",
                 "candidates", "individuals", "employment", "equal", "additionally", "further", "furthermore", "also", "when",
                 "by", "should", "where", "upon", "such", "these", "those", "each", "every", "everyone", "no", "qualified",
                 "persons", "people", "anyone", "information", "personal", "decisions", "hiring", "consideration", "diversity",
                 "accommodations", "accommodation", "discrimination", "harassment", "inclusion", "being", "here", "there",
                 "with", "without", "you", "your", "they", "that", "which", "what", "how", "learn", "read", "requests",
                 "request", "reasonable", "accessibility", "an", "a", "is", "are", "as", "on", "of", "from", "until", "because",
                 "consistent", "pursuant", "under", "applicable", "federal", "local", "state", "together", "different", "everyone's",
                 "fostering", "creating", "building", "respect", "respecting", "committed", "proud", "dedicated", "regardless"}

_COMMON = {"the", "inc", "llc", "corp", "group", "global", "american", "national", "united", "first", "new", "digital", "general",
           "advanced", "applied", "university", "college", "technologies", "systems", "health", "labs", "company"}


def _company_names(rows: list[dict]) -> list[re.Pattern]:
    names = set()
    for r in rows:
        c = (r.get("company") or "").strip(" ,.")
        if len(c) >= 3:
            names.add(c)
            first = re.split(r"[\s,]", c)[0]
            if len(first) >= 3 and first.lower() not in _COMMON:
                names.add(first)
    return [re.compile(rf"(?<![\w]){re.escape(n)}(?![\w])") for n in sorted(names, key=len, reverse=True)]


def bank_paragraph_ok(p: str) -> bool:
    """A paragraph is safe under any synthetic card: EEO, accommodation or privacy text that names no
    brand, place, link, number, pay, job type, work mode, check or requirement."""
    return (80 <= len(p) <= 900 and not p.startswith(("-", "*", "•")) and bool(_BANK_TOPIC.search(p))
            and not BANK_BANNED.search(p) and not is_work_auth(p) and _bank_caps_ok(p))


def _is_boiler_para(p: str, company: str, names: list[re.Pattern] | None = None) -> bool:
    if not bank_paragraph_ok(p):
        return False
    if names and any(rx.search(p) for rx in names):
        return False
    first = (company or "").split()
    if first and len(first[0]) >= 4 and first[0].lower().strip(",.") in p.lower():
        return False
    return True


def harvest_boilerplate(postings: list[dict], min_count: int = 1, exclude_companies: set[str] = frozenset()) -> list[str]:
    """EEO, accommodation and applicant-privacy paragraphs from real postings (rows with "description"
    and "company"); see bank_paragraph_ok. Sorted, so the bank is the same on every run."""
    seen: dict[str, int] = {}
    names = _company_names(postings)
    for row in postings:
        if (row.get("company") or "").strip().lower() in exclude_companies:
            continue
        text = strip_markdown(row.get("description", ""))
        for para in re.split(r"\n\s*\n", text):
            if para.strip().count("\n") > 2:
                continue
            p = " ".join(para.split())
            if _is_boiler_para(p, row.get("company", ""), names):
                seen[p] = seen.get(p, 0) + 1
    return sorted(p for p, n in seen.items() if n >= min_count)


def real_test_companies(test_cards_path: Path, postings_path: Path) -> set[str]:
    """Lower-cased companies of the real-test postings: their text never goes into training."""
    ids = {json.loads(x)["id"].removeprefix("ats_") for x in test_cards_path.read_text().splitlines() if x.strip()}
    rows = (json.loads(x) for x in postings_path.read_text().splitlines() if x.strip())
    return {(r.get("company") or "").strip().lower() for r in rows if r.get("simplify_id") in ids}


def bank_fingerprint(postings_path: Path, exclude_companies: set[str] = frozenset()) -> str:
    """Hash of what the bank is built from: the harvest rules, the exclusion set and the postings file."""
    parts = [str(BANK_VERSION), _BANK_TOPIC.pattern, BANK_BANNED.pattern, " ".join(sorted(_BANK_CAPS_OK)),
             " ".join(sorted(_COMMON_START)), "\n".join(sorted(exclude_companies)),
             hashlib.sha1(postings_path.read_bytes()).hexdigest() if postings_path.exists() else ""]
    return hashlib.sha1("\x00".join(parts).encode()).hexdigest()[:16]


def load_bank(path: Path, postings_path: Path | None = None, exclude_companies: set[str] = frozenset()) -> list[str]:
    """The boilerplate bank at `path`. Rebuilt from the real postings when the file is missing, is in the
    old list format, or its fingerprint (harvest rules + exclusion set + postings file) differs. Pass the
    real-test companies (`real_test_companies`) as `exclude_companies`. Every paragraph returned passes
    bank_paragraph_ok, so a stale or hand-edited file can not put a bad paragraph into training."""
    fp = bank_fingerprint(postings_path, exclude_companies) if postings_path is not None else None
    if path.exists():
        data = json.loads(path.read_text())
        if isinstance(data, dict) and (fp is None or data.get("fingerprint") == fp):
            return [p for p in data["paragraphs"] if bank_paragraph_ok(p)]
    if postings_path is None or not postings_path.exists():
        return []
    rows = [json.loads(x) for x in postings_path.read_text().splitlines() if x.strip()]
    bank = harvest_boilerplate(rows, exclude_companies=exclude_companies)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fingerprint": fp, "n": len(bank), "paragraphs": bank}, ensure_ascii=False, indent=0))
    return bank


def training_bank(root: Path) -> list[str]:
    """The checked bank for training postings under the project `root`: built from the real postings
    with the real-test companies left out, rebuilt when stale (load_bank), every paragraph passing
    bank_paragraph_ok. Raises if the real postings are there but the bank came out empty."""
    raw, test = root / "data/raw/ats/postings.jsonl", root / "data/built/audit/test_cards.jsonl"
    exclude = real_test_companies(test, raw) if raw.exists() and test.exists() else set()
    if raw.exists() and not test.exists():
        raise FileNotFoundError(f"{test} is missing: the bank must leave out the real-test companies")
    bank = load_bank(root / "data/built/training/boilerplate.json", raw, exclude)
    if raw.exists() and not bank:
        raise ValueError("boilerplate bank is empty")
    return bank


def boilerplate_count(key: str) -> int:
    """0 for 40% of rows, 1 for 25%, 2 for 20%, 3 for 15%, by row hash (append_boilerplate may add
    fewer: one paragraph per kind)."""
    r = h32(key + "|nb") % 100
    return 0 if r < 40 else 1 if r < 65 else 2 if r < 85 else 3


def bank_kind(p: str) -> str:
    """privacy, accommodation or eeo: a posting carries at most one paragraph of each."""
    if re.search(r"privacy|personal (?:data|information)|applicant data|data protection", p, re.I):
        return "privacy"
    if re.search(r"accommodat|accessib", p, re.I) and not re.search(r"equal (?:employment )?opportunity|without regard", p, re.I):
        return "accommodation"
    return "eeo"


def append_boilerplate(body: str, bank: list[str], key: str) -> str:
    """0-3 bank paragraphs by row hash, at most one of each kind, and no second EEO statement when
    the body already has one (real postings carry one)."""
    n = min(boilerplate_count(key), len(bank))
    if not n:
        return body
    used = {"eeo"} if re.search(r"equal (?:employment )?opportunity|without regard to", body, re.I) else set()
    picks, i = [], 0
    while len(picks) < n and i < 60:
        p = bank[h32(f"{key}|bp{i}") % len(bank)]
        if p not in picks and bank_kind(p) not in used:
            picks.append(p)
            used.add(bank_kind(p))
        i += 1
    return body.rstrip() + ("\n\n" + "\n\n".join(picks) if picks else "")


# ------------------------------------------------------------------ fit_posting

_HEADER_LINE = re.compile(r"^(?:Title|Company|Locations?):", re.I)


# A line that carries pay, years, a degree, enrollment or the work mode is content even when it is
# short and title-case ("Base Pay Range: $24.00 - $32.00 Per Hour", "Workplace Category: Hybrid").
_KEY = re.compile(
    r"\$\s?\d|\b(?:pay|salary|salaries|compensation|wages?|per hour|hourly|stipend|hybrid|remote|on-?site|in[- ]person|in[- ]office|"
    r"workplace|work (?:mode|arrangement|location)|years?|degree|bachelor\w*|master\w*|ph\.?\s?d|doctora\w+|associate['’]?s?|"
    r"enrolled|enrollment|pursuing|gpa|graduat\w+|in the office|work from home|wfh|days? (?:a|per|each) week)\b", re.I)


# A requirement sentence that names what the applicant has ("You have a solid understanding of algorithms",
# "Proficient with SQL and Python (required)"); these are kept before the softer requirement lines.
_STRONG_REQ = re.compile(r"\b(?:must|required|requires?|minimum|need to|needs to|necessary|essential|you have|you bring|"
                         r"you are (?:a|an|proficient|familiar|comfortable|skilled|experienced|fluent)|understanding of|"
                         r"proficien\w*|experience (?:with|in|using)|knowledge of)\b|\(required\)", re.I)


def _is_heading(s: str) -> bool:
    s = s.strip()
    if not s or len(s) > 70 or s.startswith("-") or len(s.split()) > 9:
        return False
    if re.search(r"\d|\$", s) or _KEY.search(s) and not s.endswith(":"):
        return False
    if ":" in s and s.split(":", 1)[1].strip():  # "Label: value" is content
        return False
    if s.endswith(":"):
        return True
    if s.endswith((".", ",", ";")):
        return False
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and (s.isupper() or s.istitle() or s == s.title() or len(s.split()) <= 4 and s[0].isupper())


def _units(body: str) -> list[dict]:
    """Sentences (or short lines) with the section they sit in. kinds: wa (work authorization), key
    (pay, years, degree, enrollment, work mode), req, other, boiler, head, blank."""
    units: list[dict] = []
    sec_kind, sec = "other", -1
    value_next = False  # the first short line under a "Remote:" / "Compensation:" heading is its value
    for raw in body.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            units.append({"text": "", "kind": "blank", "sec": sec, "line": len(units)})
            continue
        as_value = value_next and len(line.strip()) <= 80 and not line.rstrip().endswith(":")
        if _is_heading(line) and not is_work_auth(line) and not as_value:
            sec += 1
            sec_kind = "boiler" if _BOILER_HEAD.search(line) and not _REQ.search(line.split(":")[0]) else \
                "req" if _REQ_HEAD.search(line) else "duty" if _DUTY_HEAD.search(line) else "other"
            units.append({"text": line, "kind": "head", "sec": sec, "line": len(units), "sec_kind": sec_kind})
            value_next = bool(_KEY.search(line))
            continue
        pieces = split_sentences(line) if len(line) > 260 else [line]
        value_next = False
        for p in pieces:
            if is_work_auth(p):
                kind = "wa"
            elif _KEY.search(p) or as_value:
                kind = "key"
            elif _BOILER.search(p) and not _REQ_PARA.search(p):
                kind = "boiler"
            elif sec_kind == "boiler":
                kind = "boiler"
            elif sec_kind == "req" or _STRONG_REQ.search(p) or _REQ.search(p) and sec_kind != "duty":
                kind = "req"
            elif sec_kind == "duty":
                kind = "duty"
            else:
                kind = "other"
            units.append({"text": p, "kind": kind, "sec": sec, "line": len(units), "sec_kind": sec_kind})
        # pieces of one line stay glued: they share the line id
        for u in units[-len(pieces):]:
            u["line"] = len(units)
    return units


FIT_ORDER = ("wa", "key", "strong", "req", "duty", "other", "boiler")


def fit_posting(text: str, budget: int) -> str:
    """The posting cut to `budget` characters. Keeps the Title/Company/Locations header and fills by
    FIT_ORDER: every work-authorization sentence, every pay / years / degree / enrollment / work-mode line
    (office days too), the strong requirement lines ("must", "(required)", "you have", "proficient with",
    "understanding of"), the other requirement lines, duty lines, other text, boilerplate. When the
    assembled text is still over budget it sheds in reverse order, each rank from the end of the posting.
    Only a posting whose key text alone is over budget is cut hard (marked)."""
    text = text.strip()
    if len(text) <= budget:
        return text
    lines = text.split("\n")
    n = 0
    while n < len(lines) and _HEADER_LINE.match(lines[n]):
        n += 1
    head = "\n".join(lines[:n])
    units = _units("\n".join(lines[n:]).strip())
    heads = {u["sec"]: u for u in units if u["kind"] == "head"}
    keep: set[int] = set()
    used = len(head) + 2
    sec_paid: set[int] = set()

    def cost(i: int) -> int:
        u = units[i]
        c = len(u["text"]) + 1
        if u["sec"] in heads and u["sec"] not in sec_paid and u["kind"] != "head":
            c += len(heads[u["sec"]]["text"]) + 1
        return c

    def take(kind_set: tuple[str, ...], order: list[int]) -> None:
        nonlocal used
        for i in order:
            u = units[i]
            if u["kind"] not in kind_set or i in keep:
                continue
            c = cost(i)
            if used + c > budget:
                continue
            keep.add(i)
            used += c
            sec_paid.add(u["sec"])

    idx = list(range(len(units)))
    strong = {i for i in idx if units[i]["kind"] == "req" and _STRONG_REQ.search(units[i]["text"])}

    def rank(i: int) -> int:
        k = units[i]["kind"]
        return FIT_ORDER.index("strong" if k == "req" and i in strong else k)

    content = [i for i in idx if units[i]["kind"] not in ("head", "blank")]
    for r, name in enumerate(FIT_ORDER):  # fill by rank, each in document order
        take(("req",) if name == "strong" else (name,), [i for i in content if rank(i) == r])
    # shed order (the last is dropped first): by rank, then from the end of the posting, so a late "(required)"
    # line outlasts an early "nice to have" one
    order = sorted(keep, key=lambda i: (rank(i), i))

    def assemble(sel: set[int]) -> str:
        out: list[str] = []
        last_line, last_blank = None, True
        shown: set[int] = set()  # headings printed: with kept content, or right above a printed heading
        for i in range(len(units) - 1, -1, -1):
            u = units[i]
            if u["kind"] != "head":
                continue
            if any(j in sel for j in range(i + 1, len(units)) if units[j]["sec"] == u["sec"] and units[j]["kind"] != "head"):
                shown.add(i)
                continue
            nxt = next((j for j in range(i + 1, len(units)) if units[j]["kind"] != "blank"), None)
            if nxt is not None and nxt in shown:
                shown.add(i)
        for i, u in enumerate(units):
            if u["kind"] == "head":
                if i in shown:
                    if not last_blank:
                        out.append("")
                    out.append(u["text"])
                    last_blank, last_line = False, None
                continue
            if u["kind"] == "blank":
                if not last_blank and out:
                    out.append("")
                    last_blank = True
                last_line = None
                continue
            if i in sel:
                if last_line == u["line"] and out:
                    out[-1] += " " + u["text"]
                else:
                    out.append(u["text"])
                last_line, last_blank = u["line"], False
        body = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
        return f"{head}\n\n{body}".strip() if head else body

    sel = set(keep)
    if not sel:  # one huge unit and nothing fits: plain cut
        sel_text = text
        cut = sel_text[:budget]
        return (cut[: cut.rfind(" ")] if " " in cut else cut).rstrip() + " [truncated]"
    res = assemble(sel)
    while len(res) > budget and order:  # separators were not in the cost: drop the least important unit
        sel.discard(order.pop())
        res = assemble(sel)
    if len(res) > budget:  # key text alone is over budget
        cut = res[:budget]
        res = (cut[: cut.rfind(" ")] if " " in cut else cut).rstrip() + " [truncated]"
    return res


def cut_posting(text: str, budget: int) -> str:
    """The old plain cut, kept for comparison reports."""
    text = text.strip()
    return text if len(text) <= budget else text[:budget]


# ------------------------------------------------------------------ report on real postings


def real_posting_text(row: dict, key: str | None = None) -> str:
    locs = format_locations(row.get("locations") or [], "onsite", key or row.get("simplify_id", ""), ats_share=0.0)
    return compose_posting(row.get("title", ""), (row.get("company") or "").strip(), locs, row.get("description", ""))


# The survival gate's own splitter and detector, different from the ones fit_posting uses, so a sentence
# fit_posting mangles is counted as lost (with fit_posting's splitter the gate split 'To conform to U.S.' off
# just as fit_posting did and reported 100%). It splits only where a lower-case word, a digit or a ')' ends
# the sentence, so it never splits at an abbreviation; its pieces are whole sentences or runs of them.
_GATE_SPLIT = re.compile(r"(?<=(?<![A-Za-z.])[a-z]{2}[.!?])\s+(?=[A-Z“\"'(])|(?<=[a-z]{3}[.!?])\s+(?=[A-Z“\"'(])|(?<=[\d)][.!?])\s+(?=[A-Z“\"'])")
_GATE_WA = re.compile(
    r"sponsor|\bvisas?\b(?! card)|\bH-?1-?B\b|authori[sz]ed to work|authori[sz]ation to work|employment authori[sz]ation|"
    r"work eligibility|work[- ]authori[sz]|authori[sz]ed for employment|U\.?S\.? national\b(?! security)|international students|"
    r"work status|eligib\w+ to work|right to work|employment eligibility|U\.?S\.? citizen|citizens? (?:of|or)|"
    r"citizenship (?:is |may be )?required|permanent resident|green card|export[- ]control|export regulation|security clearance|"
    r"U\.?S\.? persons?|e-?verify|lawful(?:ly)? (?:admitted|permanent)|(?-i:\b(?:OPT|CPT|ITAR|EAR|F-1|J-1)\b)", re.I)


def _gate_pieces(full: str) -> list[str]:
    """Work-authorization pieces of a posting as the gate sees them: whole lines up to 260 characters,
    sentence runs (by _GATE_SPLIT) of longer lines, kept when either detector fires."""
    out = []
    for ln in full.split("\n"):
        ln = ln.strip()
        for piece in ([ln] if len(ln) <= 260 else _GATE_SPLIT.split(ln)):
            if piece and (is_work_auth(piece) or _GATE_WA.search(piece) and not _EEO.search(piece)):
                out.append(_flat(piece))
    return out


def survival(rows: list[dict], budget: int) -> dict:
    """Share of real postings (that have a work-auth sentence) whose work-auth sentences all survive, each
    as written in the posting (see _gate_pieces: not fit_posting's own splitter or detector)."""
    n_with = n_all = n_any = sent_tot = sent_kept = 0
    lens, losing = [], []
    for r in rows:
        full = real_posting_text(r)
        was = _gate_pieces(full)
        fitted = fit_posting(full, budget)
        lens.append(len(fitted))
        if not was:
            continue
        n_with += 1
        flat = _flat(fitted)
        ok = [s in flat for s in was]
        n_all += all(ok)
        n_any += any(ok)
        sent_tot += len(ok)
        sent_kept += sum(ok)
        if not all(ok):
            losing.append(r.get("simplify_id", "")[:8])
    lens.sort()
    return dict(budget_chars=budget, postings=len(rows), with_work_auth=n_with,
                all_survive=n_all / max(n_with, 1), any_survives=n_any / max(n_with, 1),
                sentences=sent_kept / max(sent_tot, 1), mean_chars=sum(lens) / len(lens), p90_chars=lens[int(.9 * len(lens))],
                losing_ids=losing[:20])


def survival_plain_cut(rows: list[dict], budget: int) -> float:
    n_with = n_all = 0
    for r in rows:
        full = real_posting_text(r)
        was = _gate_pieces(full)
        if not was:
            continue
        n_with += 1
        flat = _flat(cut_posting(full, budget))
        n_all += all(s in flat for s in was)
    return n_all / max(n_with, 1)


# Measured against the card facts extracted from each real posting, not against fit_posting's own word lists
# (the old dollar / work-mode / degree gates used the classifier's words, so they passed by construction).
_GATE_FIG = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
_GATE_MODE = re.compile(r"in the office|work from home|days? (?:a|per|each) week|\bhybrid\b|\bremote\b|on-?site|in[- ]person", re.I)


def card_fact_survival(postings: list[dict], cards: list[dict], budget: int) -> dict:
    """On the real postings over budget (rows of data/raw/ats/postings.jsonl, joined to the extracted real
    cards by source_posting_id = 'ats_' + simplify_id): the share of each card fact the full text states
    that the fitted text still states. must_group: a must-skill group with any of its names; license;
    min_years ("3+ years"); pay: a dollar figure, for a card with pay; mode_sentences: the sentences that
    say where or how often the work is done. Also the postings that lose a must-skill group."""
    from .visibility import has_phrase, has_years  # visibility does not import textclean
    by_id = {r["simplify_id"]: r for r in postings if r.get("simplify_id")}
    tot = {k: 0 for k in ("must_group", "license", "min_years", "pay", "mode_sentences")}
    kept = dict.fromkeys(tot, 0)
    over, losing = 0, []
    for row in cards:
        c = row.get("card", row)
        post = by_id.get(str(c.get("source_posting_id", "")).removeprefix("ats_"))
        if post is None:
            continue
        full = real_posting_text(post)
        if len(full) <= budget:
            continue
        over += 1
        fit = fit_posting(full, budget)
        lost = 0
        for g in c.get("must_skills") or []:
            names = [g] if isinstance(g, str) else list(g)
            if any(has_phrase(full, n) for n in names):
                tot["must_group"] += 1
                ok = any(has_phrase(fit, n) for n in names)
                kept["must_group"] += ok
                lost += not ok
        if lost:
            losing.append(post["simplify_id"][:8])
        if c.get("license") and has_phrase(full, c["license"]):
            tot["license"] += 1
            kept["license"] += has_phrase(fit, c["license"])
        if c.get("min_years") and has_years(full, c["min_years"]):
            tot["min_years"] += 1
            kept["min_years"] += has_years(fit, c["min_years"])
        figs = _GATE_FIG.findall(full)
        if c.get("salary_min") is not None and figs:
            tot["pay"] += 1
            kept["pay"] += any(f in fit for f in figs)
        flat = _flat(fit)
        mode = [_flat(x) for ln in full.split("\n") for x in split_sentences(ln) if _GATE_MODE.search(x) and len(x) < 400]
        tot["mode_sentences"] += len(mode)
        kept["mode_sentences"] += sum(x in flat for x in mode)
    return dict(budget_chars=budget, over_budget=over, **{f"{k}_n": tot[k] for k in tot},
                **{f"{k}_kept": round(kept[k] / tot[k], 4) if tot[k] else None for k in tot},
                postings_losing_a_must_group=len(losing), losing_ids=losing[:20])


def _flat(t: str) -> str:
    return " ".join(t.split())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "REPORT":
        p = Path(sys.argv[2] if len(sys.argv) > 2 else "data/raw/ats/postings.jsonl")
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        for b in (3200, 4000, 4800, 6000, 8000):
            s = survival(rows, b)
            print(json.dumps(s | {"plain_cut_all_survive": survival_plain_cut(rows, b)}))
        cards_p = p.parents[2] / "built/cards/extracted_real_cards_v2.jsonl"
        if cards_p.exists():
            cards = [json.loads(x) for x in cards_p.read_text().splitlines() if x.strip()]
            for b in (6000, 8000):
                print(json.dumps(card_fact_survival(rows, cards, b)))
