"""Non-model parts of the demo: resume text in, profile out, posting pool, run B's prompt, the decider readout, the H''
rules, three buckets.

Standard library only (pypdf is imported lazily for PDF uploads). Imports from src/matcher only the
prompt, profile, rule, hybrid and decider code, which hold no model code.

The shipped system (eval/selection_cycle3.json pick, scored once on fresh test 4 in eval/fresh_test4_result.json):
run B's three float32 label scores on matcher.prompt_b's prompt -> the decider (matcher.decider: s = P(Keep) +
0.5 P(Explore) at temperature 1, s >= 0.58904 Keep, s < 0.27310 Skip, else Explore) -> H'' without work mode
(matcher.hybrid.HPP_CATS and the salary cap) overrides with Skip, or caps a Keep at Explore.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from matcher import textclean
from matcher.decider import T_HI, T_LO, decide, decide_dict, keep_score
from matcher.hybrid import HPP_CATS, SALARY_PREFIX, hybrid, rule_verdict
from matcher.prompt_b import prompt_b
from matcher.profile import render_profile
from matcher.rules import mark_thin
from matcher.schema import (
    DEGREE_LEVELS, EMPLOYMENT_TYPES, REMOTE_PREFS, WORK_AUTH, CandidateCard, RequirementCard,
)

ANYWHERE = "anywhere in the US"
BUCKET_OF = {"Keep": "For you", "Explore": "Explore", "Skip": "Excluded"}
BUCKETS = ("For you", "Explore", "Excluded")


# ---------------------------------------------------------------- resume text in

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"(?:https?://|www\.)\S+|\b(?:linkedin|github|gitlab)\.com/\S+", re.I)
# (617) 555-0142, 617.555.0142, 1-617-555-0142, +1 617 555 0142, 6175550142, with an optional x12 / ext. 12
_PHONE = re.compile(r"(?<![\w.])(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)"
                    r"(?:\s*(?:x|ext\.?)\s*\d{1,5}\b)?", re.I)
_STREET = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][\w.'-]*\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl)\b\.?"
    r"(?:,?\s*(?:Apt|Unit|Suite|Ste|#)\.?\s*\w+)?")
_HEADING = re.compile(
    r"^(?:resume|résumé|curriculum vitae|cv|contact|summary|objective|profile|about me|education|experience|"
    r"work experience|professional experience|employment|skills|technical skills|projects|coursework|"
    r"certifications?|awards|honors|activities|leadership|publications|languages|interests|references)\b", re.I)
_PARTICLES = {"de", "la", "del", "della", "der", "den", "van", "von", "da", "di", "du", "do", "dos", "das", "le",
              "bin", "binti", "ibn", "al", "el", "y", "e"}
_TITLE = re.compile(r"^(?:Dr|Mr|Mrs|Ms|Mx|Prof)\.?\s+")
_CREDENTIAL = re.compile(r"(?:,\s*|\s+)(?:B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?|MBA|M\.?Eng\.?|Ph\.?\s?D\.?|M\.?D\.?|"
                         r"Pharm\.?\s?D\.?|CPA|RN|PE|Jr\.?|Sr\.?|II|III|IV)\.?\s*$")
# Words that mark a line as an institution, a job title or a section, never a person's name.
_NOT_NAME = re.compile(
    r"\b(?:university|college|institute|school|academy|polytechnic|department|hospital|center|centre|laboratory|"
    r"lab|inc|llc|ltd|corp|corporation|company|group|engineer|engineering|intern|internship|developer|designer|"
    r"student|analyst|scientist|science|sciences|manager|assistant|associate|technician|specialist|consultant|"
    r"researcher|technology|technologies|biology|chemistry|biochemistry|physics|mathematics|computer|data|"
    r"software|candidate|fellow|tutor|volunteer|president|member|coordinator|officer|director|leader|list|"
    r"club|society)\b", re.I)
_CONTACT_WORDS = {"or", "and", "email", "e-mail", "mail", "phone", "tel", "mobile", "cell", "linkedin", "github",
                  "website", "web", "portfolio", "contact", "address"}


def _name_of(line: str) -> str | None:
    """The line itself when it reads as a person's name (2 to 4 capitalised words in any script, lowercase
    particles such as 'de la' allowed, an optional title or trailing degree), else None."""
    s = _CREDENTIAL.sub("", _TITLE.sub("", line.strip())).strip(" ,")
    if not s or any(ch.isdigit() for ch in s) or _NOT_NAME.search(s) or _HEADING.match(s):
        return None
    words = s.split()
    named = [w for w in words if w.lower() not in _PARTICLES]
    if not 2 <= len(named) <= 4 or len(words) > 6:
        return None
    for w in named:
        core = re.sub(r"[.'’-]", "", w)
        if not core or not core.isalpha() or not w[0].isupper():
            return None
    return s


def _scrub(line: str) -> str:
    """Remove contact details from one line, then the separators and labels they leave behind."""
    new = _STREET.sub("", _PHONE.sub("", _URL.sub("", _EMAIL.sub("", line))))
    if new == line:
        return line
    words = re.findall(r"[^\W\d_][\w'’-]*", new)
    if all(w.lower() in _CONTACT_WORDS for w in words):
        return ""
    new = re.sub(r"(?i)\b(?:e-?mail|phone|tel|mobile|cell|linkedin|github|website|portfolio)\s*:\s*(?=[|•·/,;]|$)", "", new)
    new = re.sub(r"(?:\s*[|•·/;,]\s*){2,}", " | ", new)
    return re.sub(r"^\s*[|•·/;,]\s*|\s*[|•·/;,]\s*$", "", new)


# Separators between the parts of a resume header line: "Name | email | phone", "Name — City", "Name · City".
_SEP = re.compile(r"\s*(?:[|•·—–]|\s-\s|\s/\s)\s*")
_PRONOUNS = re.compile(r"\s*\(?\b(?:he|she|they|ze|xe)\s*/\s*(?:him|his|her|hers|them|theirs|zir|hir|xem)\b\)?", re.I)
_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA",
    "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR",
    "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "PR",
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware", "florida",
    "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
    "maryland", "massachusetts", "michigan", "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina", "north dakota", "ohio", "oklahoma",
    "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah",
    "vermont", "virginia", "washington", "west virginia", "wisconsin", "wyoming", "puerto rico"}


def _is_state(s: str) -> bool:
    s = s.strip(" .")
    return s in _STATES or s.lower() in _STATES


def _has_place(s: str) -> bool:
    """True when s contains a "City, ST" or "City, State" place."""
    return any(_is_state(part) for part in re.split(r"[,|•·—–]", s)[1:])


def _same_case(words: list[str]) -> bool:
    return all(w.isupper() for w in words) or all(w[:1].isupper() and w[1:] == w[1:].lower() for w in words)


def _header_name(line: str, had_contact: bool, top: bool) -> tuple[list[str], str] | None:
    """The name in a resume header line and the line without it, or None. Covers:
    - the name alone on its line;
    - the name as the first part of "Name | email | City" or "Name — City" (only on the first line, or on a
      line that held contact details or a place, so "Machine Learning | Python" further down is not a name);
    - "Name, City, ST";
    - "Surname, Given" (only on the first line, so "Java, Python" under a heading is not a name);
    - a pronoun such as "(he/him)" after any of these.
    Returns the forms to remove elsewhere ("Lee, Jordan" also gives "Jordan Lee")."""
    line = _PRONOUNS.sub("", line).strip()
    if name := _name_of(line):
        return [name], ""
    first, *rest = _SEP.split(line, maxsplit=1)
    if rest and (top or had_contact or _has_place(rest[0])) and (name := _name_of(first)):
        return [name], rest[0]
    parts = [x.strip() for x in first.split(",")]
    # "Jordan Lee, Boston, MA": a name, then a "City, ST" place.
    if len(parts) >= 3 and _is_state(parts[2]) and (name := _name_of(parts[0])):
        rest_line = ", ".join(parts[1:]) + (" | " + rest[0] if rest else "")
        return [name], rest_line
    # "Lee, Jordan" (surname first): one word, then one or two given names, and nothing else on the line.
    if top and len(parts) == 2 and not rest and " " not in parts[0] and not _is_state(parts[1]) \
            and _same_case(re.sub(r"[.'’-]", "", " ".join(parts)).split()):
        if name := _name_of(f"{parts[1]} {parts[0]}"):
            return [f"{parts[0]}, {parts[1]}", name], ""
    return None


def redact(text: str) -> str:
    """The resume as the model sees it in training and in the real test, from a real upload: the demo's own
    pass for real resumes (`redact_real`: the name anywhere, a surname-first header, contact details), then
    textclean.redact, the pass every training and test resume gets (build_prompt runs it again), so the
    prompt opens with "Candidate" as training prompts do."""
    return textclean.redact(redact_real(text))


def redact_real(text: str) -> str:
    """Remove contact details from resume text: email, phone, links and street address anywhere, and the
    name: the first name found among the first three non-heading lines (alone on its line, or sharing it
    with the city and contact details), and that name wherever else it appears. The name line becomes
    "Candidate" (with what else it held, such as the city, on the next line)."""
    raw = text.splitlines()
    lines = [_scrub(line) for line in raw]
    names: list[str] = []
    seen, top = 0, True
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or _HEADING.match(s) and len(s) < 40:
            if s and not re.match(r"(?i)^(?:resume|résumé|curriculum vitae|cv|contact)\b", s):
                top = False
            continue
        found = _header_name(s, had_contact=lines[i] != raw[i], top=top and seen == 0)
        if found:
            names, rest = found
            lines[i] = "Candidate" + (f"\n{rest}" if rest.strip() else "")
            break
        seen += 1
        if seen >= 3:
            break
    out = "\n".join(re.sub(r"[ \t]{2,}", " ", line).rstrip() for line in lines)
    for name in names:
        out = re.sub(re.escape(name), "", out, flags=re.I)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"


def _docx_block(el, paras: list[str]) -> None:
    """Paragraphs under el, in order. mc:Fallback is skipped (it repeats the mc:Choice content), and a
    text box inside a paragraph becomes its own paragraphs after it instead of being read twice."""
    if el.tag == _MC_FALLBACK:
        return
    if el.tag != _W + "p":
        for child in el:
            _docx_block(child, paras)
        return
    parts: list[str] = []
    nested: list = []

    def run(node) -> None:
        for child in node:
            if child.tag == _MC_FALLBACK:
                continue
            if child.tag in (_W + "txbxContent", _W + "p"):
                nested.append(child)
            elif child.tag == _W + "t":
                parts.append(child.text or "")
            elif child.tag == _W + "tab":
                parts.append("\t")
            elif child.tag in (_W + "br", _W + "cr"):
                parts.append("\n")
            else:
                run(child)

    run(el)
    paras.append("".join(parts))
    for box in nested:
        _docx_block(box, paras)


def docx_text(data: bytes) -> str:
    """Text of a .docx: its word/document.xml, one line per paragraph, text boxes read once."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = ElementTree.fromstring(z.read("word/document.xml"))
    paras: list[str] = []
    _docx_block(root, paras)
    return "\n".join(paras).strip()


def pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ValueError("PDF upload needs pypdf: run with `uv run --group demo python demo/server.py`") from e
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def extract_text(filename: str, data: bytes) -> str:
    """Resume text from an uploaded .pdf, .docx, .txt or .md file."""
    ext = Path(filename).suffix.lower()
    if ext not in (".docx", ".pdf", ".txt", ".md", ""):
        raise ValueError(f"unsupported file type {ext!r}: use .pdf, .docx or .txt")
    try:
        if ext == ".docx":
            text = docx_text(data)
        elif ext == ".pdf":
            text = pdf_text(data)
        else:
            text = data.decode("utf-8", errors="replace").strip()
    except ValueError:
        raise
    except Exception as e:  # BadZipFile, KeyError, ParseError, pypdf errors, an encrypted PDF ...
        raise ValueError(f"could not read this {ext or 'text'} file ({type(e).__name__}: {e}); "
                         "try saving it again as PDF or paste the text") from e
    if not text:
        raise ValueError("no text found in the file (a scanned PDF has no text layer)")
    return text


# ---------------------------------------------------------------- profile form

def _list(v: Any) -> tuple[str, ...]:
    if v is None:
        return ()
    items = re.split(r"[;\n]|,(?!\s*[A-Z]{2}\b)", v) if isinstance(v, str) else v
    return tuple(s.strip() for s in items if isinstance(s, str) and s.strip())


def _check(name: str, value: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}")
    return value


def candidate_from_form(form: dict) -> CandidateCard:
    """The profile form -> CandidateCard. Relocation is explicit: anywhere_us | list | nowhere
    (an empty list would render 'nowhere else' and silently exclude every other city)."""
    reloc = form.get("relocate", "anywhere_us")
    if reloc == "anywhere_us":
        willing: tuple[str, ...] = (ANYWHERE,)
    elif reloc == "list":
        willing = _list(form.get("willing_locations"))
        if not willing:
            raise ValueError("relocation is 'a list of places' but no place is listed")
    elif reloc == "nowhere":
        willing = ()
    else:
        raise ValueError("relocate must be anywhere_us, list or nowhere")
    floor = form.get("salary_floor")
    floor = int(floor) if floor not in (None, "") else None
    grad = form.get("graduation_year")
    grad = int(grad) if grad not in (None, "") else None
    years = form.get("years")
    return CandidateCard(
        field=str(form.get("degree_field", "")).strip(),
        degree_level=_check("degree_level", form.get("degree_level", "bachelor"), DEGREE_LEVELS),
        degree_field=str(form.get("degree_field", "")).strip(),
        graduation_year=grad,
        years=int(years) if years not in (None, "") else 0,
        current_location=str(form.get("current_location", "")).strip(),
        willing_locations=willing,
        remote_preference=_check("remote_preference", form.get("remote_preference", "any"), REMOTE_PREFS),
        work_auth=_check("work_auth", form.get("work_auth", "citizen_or_pr"), WORK_AUTH),
        salary_floor=floor,
        employment_types_wanted=tuple(_check("employment type", t, EMPLOYMENT_TYPES)
                                      for t in _list(form.get("employment_types"))),
        industries_wanted=_list(form.get("industries_wanted")),
        industries_avoided=_list(form.get("industries_avoided")),
        roles_avoided=_list(form.get("roles_avoided")),
    )


def profile_text(form: dict) -> str:
    """Rendered by the training code itself, so the format cannot drift."""
    return render_profile(candidate_from_form(form))


# ---------------------------------------------------------------- posting pool

def load_pool(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def pool_summary(row: dict) -> dict:
    c = row["card"]
    return {"id": row["id"], "title": row["title"], "company": row["company"], "location": location_of(row),
            "url": row.get("url", ""), "employment_type": c.get("employment_type"),
            "citizens_only": c.get("citizens_only", False)}


def location_of(row: dict) -> str:
    locs = row.get("locations") or row["card"].get("locations") or []
    mode = row["card"].get("work_mode", "")
    head = ", ".join(locs[:2]) + (f" +{len(locs) - 2}" if len(locs) > 2 else "")
    if mode == "remote":
        return f"{head} (remote)" if head else "Remote"
    return head


# ---------------------------------------------------------------- H'': the rules on Adaption's extracted card

ELIG_CATS, ELIG_SALARY_CAP = HPP_CATS, True  # H'' without work mode (eval/selection_cycle3.json), from matcher.hybrid
ELIG_NAME = "H''"
RULE_MARK = "decided by the rule on Adaption's extracted card"


def elig_cats(form: dict) -> frozenset:
    """The H'' override categories this form can decide. A category whose profile field the form leaves empty
    is left out, so an empty field never fires an override: the rules would read a blank graduation year as
    "not a student" and a blank field of study as matching no field. Empty degree level (never sent by the
    page) drops all three degree categories; empty graduation year drops enrollment unless the degree level is
    "none" (then nobody is enrolled anyway); empty field of study drops degree field; empty "Lives in" drops
    location unless relocation is anywhere in the US. Work authorization and salary floor need no rule here:
    their empty values (citizen or PR, no floor) never fire an override."""
    cats = set(ELIG_CATS)
    level = form.get("degree_level")
    if level in (None, ""):
        cats -= {"degree_level", "degree_field", "enrollment"}
    elif form.get("graduation_year") in (None, "") and level != "none":
        cats.discard("enrollment")
    if not str(form.get("degree_field") or "").strip():
        cats.discard("degree_field")
    if not str(form.get("current_location") or "").strip() and form.get("relocate", "anywhere_us") != "anywhere_us":
        cats.discard("location")
    return frozenset(cats)


def posting_body(row: dict) -> str:
    """The posting's own text, the body rules.mark_thin reads: the pool row's prompt text after its
    Title/Company/Locations header (textclean.header ends at the first blank line). It is the markdown-free body;
    on every posting of the ATS pool source and fresh tests 1-4 it falls on the same side of rules.THIN_CHARS as
    the raw description the eval reads."""
    text = row.get("text") or ""
    return text.split("\n\n", 1)[1] if text.startswith("Title: ") and "\n\n" in text else text


def elig_verdict(cand: CandidateCard, row: dict) -> dict:
    """matcher.hybrid.rule_verdict of the form's candidate on the pool row's extracted card, marked thin or not
    from the full card and the posting body first (rules.mark_thin, as in training and in fit_eval_score). The form
    has no skills or licenses (the model reads them in the resume), so the card's skill and license asks are left out
    after that: they are never an H'' category, and left in they would make nearly every rule verdict a skills Skip,
    which would switch off the salary cap (it needs the rules' verdict to be Explore)."""
    card = mark_thin(RequirementCard.from_dict(row["card"]), posting_body(row))
    return rule_verdict(cand, replace(card, must_skills=(), nice_skills=(), license=None))


def elig_skip(verdict: dict, cats: frozenset) -> str | None:
    """The rule's reasons when H'' gives Skip whatever the model says, else None. Such a posting is decided
    before the model runs: the same label, no GPU time."""
    if all(hybrid(w, verdict, cats, ELIG_SALARY_CAP) == "Skip" for w in _LABEL_ORDER):
        return "; ".join(x for x, c in verdict["reason_cats"].items() if c in cats)
    return None


def elig_final(model_label: str, verdict: dict, cats: frozenset) -> str:
    """H'' after the model: matcher.hybrid.hybrid with the H'' categories and salary cap."""
    return hybrid(model_label, verdict, cats, ELIG_SALARY_CAP)


def model_label(scores: dict[str, float]) -> str:
    """Run B's label before the rules: the decider (matcher.decider) on the three unrounded float32 label scores."""
    return decide_dict(scores)


# ---------------------------------------------------------------- prompts and buckets

def split_prompt(resume: str, profile: str, posting_text: str) -> tuple[str, str]:
    """(prefix, suffix) of run B's prompt (matcher.prompt_b), so a scorer can cache the part every posting shares:
    the rubric, the resume and the profile."""
    prefix_full = prompt_b(resume, profile, "")
    tail = "\n\nAnswer:"
    assert prefix_full.endswith(tail)
    prefix = prefix_full[: -len(tail)]
    full = prompt_b(resume, profile, posting_text)
    assert full.startswith(prefix), "prompt_b changed shape; update split_prompt"
    return prefix, full[len(prefix):]


POSTING_MARK = "\n\nJOB POSTING:\n"


def split_row(prompt: str) -> tuple[str, str]:
    """(prefix, suffix) of a finished prompt (a dev or real-test row), split where split_prompt splits."""
    i = prompt.index(POSTING_MARK) + len(POSTING_MARK)
    return prompt[:i], prompt[i:]


def resume_in_prompt(prefix: str) -> str:
    """The resume exactly as the prompt carries it (after any redaction build_prompt itself does)."""
    head, sep, rest = prefix.partition("RESUME:\n")
    return rest.partition("\n\nCANDIDATE PROFILE:")[0] if sep else ""


def prefix_sha(prefix: str) -> str:
    return hashlib.sha1(prefix.encode()).hexdigest()[:16]


def text_sha(text: str) -> str:
    """Ties a precomputed score to the posting text it was computed on (a rebuilt pool can reuse an id)."""
    return hashlib.sha1(text.encode()).hexdigest()[:12]



def make_card(row: dict, *, label_word: str, p: dict, reason: str | None, source: str,
              scores: dict | None = None) -> dict:
    return {**pool_summary(row), "label": label_word, "bucket": BUCKET_OF[label_word],
            "p_keep": p["Keep"], "p": p, "reason": reason or "", "source": source, "scores": scores}


def rule_card(row: dict, reason: str, verdict: dict, cats: frozenset) -> dict:
    """A posting H'' set to Skip before the model ran, with the rule's reasons and the marker."""
    card = make_card(row, label_word="Skip", p={"Keep": 0.0, "Explore": 0.0, "Skip": 1.0},
                     reason=reason, source="rules")
    hit = sorted({c for c in verdict["reason_cats"].values() if c in cats})
    return {**card, "rule": {"hybrid": ELIG_NAME, "kind": "skip", "categories": hit, "mark": RULE_MARK}}


def scored_card(row: dict, *, label_word: str, p: dict, reason: str | None, source: str, scores: dict | None,
                verdict: dict, cats: frozenset) -> dict:
    """A posting the model scored (label_word: the decider's label), with H'' applied after it: the salary cap can
    turn its Keep into Explore, and then the rule's reason replaces the model's and the card carries the marker."""
    final = elig_final(label_word, verdict, cats)
    card = make_card(row, label_word=final, p=p, reason=reason, source=source, scores=scores)
    if final == label_word:
        return card
    why = "; ".join(x for x in verdict["reasons"] if x.startswith(SALARY_PREFIX))
    return {**card, "reason": why, "rule": {"hybrid": ELIG_NAME, "kind": "salary_cap", "categories": ["salary_cap"],
                                            "model_label": label_word, "model_reason": reason or "",
                                            "mark": RULE_MARK}}


def to_buckets(cards: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {b: [] for b in BUCKETS}
    for c in cards:
        out[c["bucket"]].append(c)
    for b in BUCKETS:
        out[b].sort(key=lambda c: (-c["p_keep"], c["title"]))
    return out


# ---------------------------------------------------------------- label parity with the eval

NEAR_CUT = 0.02  # eval's s this close to a decider cut-off: a flip there is dtype or kernel noise, not a bug
_LABEL_ORDER = ("Keep", "Explore", "Skip")  # evalstats.LABELS (this module stays numpy-free)


def label_parity(pairs: list[tuple[str, str, dict, list]], near_cut: float = NEAR_CUT) -> dict:
    """A demo scorer's labels and scores against a local_eval.py (colab_eval) result for the same rows.
    pairs: (id, demo label, demo scores {Keep, Explore, Skip}, eval `scores` in Keep/Explore/Skip order). The eval's
    label is the decider on its stored scores (the eval file's `pred` is the argmax, not the shipped readout). A
    disagreement where the eval's s is within near_cut of a cut-off is listed as a near-cut and not counted against
    agreement; ok = at least 98% agree once near-cuts are set aside."""
    agree, worst, flips, ties = 0, 0.0, [], []
    for pid, label, scores, ref in pairs:
        diff = max(abs(scores[w] - r) for w, r in zip(_LABEL_ORDER, ref))
        worst = max(worst, diff)
        want = decide(ref)
        if label == want:
            agree += 1
            continue
        s = keep_score(ref)
        margin = min(abs(s - T_LO), abs(s - T_HI))
        (ties if margin < near_cut else flips).append((pid, want, label, round(diff, 3), round(margin, 4)))
    n = len(pairs)
    return {"rows": n, "agree": agree, "near_cuts": ties, "flips": flips, "max_score_diff": round(worst, 4),
            "ok": bool(n) and (agree + len(ties)) / n >= 0.98}


def print_parity(res: dict, who: str) -> None:
    n = res["rows"]
    print(f"label agreement {res['agree']}/{n} ({res['agree'] / max(n, 1):.1%}) between {who} and the eval's decider; "
          f"{len(res['near_cuts'])} near-cut flips (eval s within {NEAR_CUT} of a cut-off), {len(res['flips'])} real "
          f"flips; max label-score difference {res['max_score_diff']:.4f}")
    for f in res["near_cuts"]:
        print("  near-cut (id, eval, demo, max diff, eval s to nearest cut-off):", f)
    for f in res["flips"]:
        print("  differs  (id, eval, demo, max diff, eval s to nearest cut-off):", f)
