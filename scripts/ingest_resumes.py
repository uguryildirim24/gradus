"""Real PDF resumes -> redacted, prompt-ready text, for the live-demo smoke test.

  ingest_resumes.py fetch  [--n 20] [--seed 0] [--dir DIR]
      Downloads n PDFs from the Hugging Face dataset opensporks/resumes (card says license
      cc0-1.0; it is a copy of the Kaggle "resume-dataset", itself scraped from livecareer.com,
      so the CC0 label is the uploader's claim). The files are spread over the dataset's
      categories. This only downloads; nothing is parsed or run on a model.
  ingest_resumes.py extract [--dir DIR] [--out DIR]
      pdf -> text (pypdf) -> redact() -> prompt_ready(); writes resumes.jsonl (id, category, text,
      redaction counts) and prompts.jsonl: each resume joined to one real test posting with a neutral
      profile through build_prompt, so the rows are ready to feed the model. No model runs.
      Run with `uv run --with pypdf python scripts/ingest_resumes.py extract` (pypdf is not a project
      dependency).

redact() makes a real resume look like the synthetic ones the model trained on: the header is the
single word "Candidate", and emails, phone numbers, URLs, social handles, street addresses and ZIP
codes are removed. It cannot find a person's name in the body; the CC0 files carry none (they use
"Company Name" and "City, State" placeholders), and the demo form should tell users to leave their
name out.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from matcher.pairs import build_prompt  # noqa: E402
from matcher.profile import render_profile  # noqa: E402
from matcher.schema import CandidateCard  # noqa: E402

HF = "https://huggingface.co"
DATASET = "opensporks/resumes"
DEFAULT_DIR = ROOT / "data/raw/open/resumes_cc0"
OUT = ROOT / "data/built/real_resumes"

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
URL = re.compile(r"(https?://|www\.)\S+|\b(linkedin|github|gitlab|twitter|instagram|facebook|behance)\.com\S*", re.I)
HANDLE = re.compile(r"(?<![\w.])@[A-Za-z][\w.]{2,}")
PHONE = re.compile(r"(\+?\d{1,2}[\s.-]?)?(\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b")
STREET = re.compile(r"\b\d{1,6}\s+(?:[A-Z][\w.]*\s+){1,4}(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Ct|Court|Pl|Place|Pkwy|Parkway|Hwy|Highway)\b\.?(?:\s*(?:#|Apt\.?|Suite|Ste\.?|Unit)\s*\w+)?", re.I)
ZIP = re.compile(r"(?<=\b[A-Z]{2}\s)\d{5}(-\d{4})?\b|(?<=,\s[A-Z]{2},\s)\d{5}\b")
NAME_WORD = re.compile(r"[A-Za-z][A-Za-z.'-]*")
JOB_WORD = re.compile(r"\b(manager|engineer|accountant|analyst|assistant|associate|administrator|specialist|director|"
                      r"consultant|developer|intern|nurse|teacher|coordinator|technician|representative|officer|designer|"
                      r"supervisor|clerk|executive|lead|senior|junior|chef|cook|driver|agent|advisor|adviser|architect|"
                      r"scientist|attorney|paralegal|auditor|banker|cashier|trainer|instructor|therapist|pharmacist|"
                      r"mechanic|operator|planner|recruiter|writer|editor|sales|marketing|finance|hr|it)\b", re.I)
SECTION = re.compile(r"^(summary|objective|profile|experience|work history|education|skills|highlights|qualifications|"
                     r"professional|career|accomplishments|core|certifications|affiliations|additional)\b", re.I)


def redact(text: str) -> tuple[str, dict]:
    """(text, counts of what was removed). The first line becomes "Candidate" when it looks like a
    name or a contact line; otherwise a "Candidate" header is put in front of it."""
    counts = {}

    def sub(name, pat, s, repl=""):
        s, n = pat.subn(repl, s)
        counts[name] = n
        return s

    text = text.replace("\r", "\n")
    text = sub("email", EMAIL, text)
    text = sub("url", URL, text)
    text = sub("handle", HANDLE, text)
    text = sub("phone", PHONE, text)
    text = sub("street", STREET, text)
    text = sub("zip", ZIP, text)
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    header_removed = False
    if lines:
        first = lines[0].strip()
        words = first.split()
        # A name is 2-4 plain words ("Jane Q. Doe"). Livecareer resumes open with the job title
        # instead ("ACCOUNTANT", "HR ADMINISTRATOR/MARKETING ASSOCIATE"); that line stays.
        looks_like_name = (2 <= len(words) <= 4 and all(NAME_WORD.fullmatch(w) for w in words)
                           and not SECTION.match(first) and not JOB_WORD.search(first)
                           and (first.istitle() or first.isupper()))
        contact_only = not re.search(r"[A-Za-z]{3}", first)
        if looks_like_name or contact_only:
            lines.pop(0)
            header_removed = True
    counts["name_line_removed"] = int(header_removed)
    # contact separators left behind (" | ", " - ") on the first lines
    lines = [re.sub(r"(\s*[|•·]\s*){2,}", " | ", ln) for ln in lines]
    return "Candidate\n" + "\n".join(lines).strip(), counts


def prompt_ready(text: str, max_chars: int = 5000) -> str:
    """Plain text the way the synthetic resumes read: no control characters, no runs of blank
    lines or spaces, bullets as '* ', capped at max_chars on a line boundary."""
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
    text = re.sub(r"[•●▪◦‣∙]", "*", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)
    text = "\n".join(ln.strip() for ln in text.split("\n")).strip()
    if len(text) > max_chars:
        cut = text[:max_chars]
        text = cut[: cut.rfind("\n")] if "\n" in cut else cut
    return text


def pdf_text(path: Path) -> str:
    from pypdf import PdfReader
    return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)


# ---------------------------------------------------------------- fetch / extract

def listing() -> list[str]:
    with urllib.request.urlopen(f"{HF}/api/datasets/{DATASET}") as r:
        info = json.load(r)
    if "cc0-1.0" not in info.get("cardData", {}).get("license", []):
        sys.exit(f"{DATASET} is not marked cc0-1.0 any more: {info.get('cardData')}")
    return [s["rfilename"] for s in info["siblings"] if s["rfilename"].endswith(".pdf")]


def pick_files(files: list[str], n: int, seed: int) -> list[str]:
    """n files spread round-robin over the categories (the folder under data/data/)."""
    by: dict[str, list[str]] = {}
    for f in sorted(files):
        by.setdefault(f.split("/")[-2], []).append(f)
    rng = random.Random(seed)
    cats = sorted(by)
    rng.shuffle(cats)
    out: list[str] = []
    i = 0
    while len(out) < n and any(by.values()):
        c = cats[i % len(cats)]
        if by[c]:
            out.append(by[c].pop(rng.randrange(len(by[c]))))
        i += 1
    return out


def fetch(a: argparse.Namespace) -> None:
    dest = Path(a.dir)
    dest.mkdir(parents=True, exist_ok=True)
    for f in pick_files(listing(), a.n, a.seed):
        target = dest / f"{f.split('/')[-2]}__{Path(f).name}"
        if not target.exists():
            urllib.request.urlretrieve(f"{HF}/datasets/{DATASET}/resolve/main/{f}", target)
        print(target.name, target.stat().st_size, "bytes")


def extract(a: argparse.Namespace) -> None:
    pdfs = sorted(Path(a.dir).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"no PDFs in {a.dir}; run fetch first")
    postings = [json.loads(line) for line in (ROOT / "data/built/real_test/rows.jsonl").open()] \
        if (ROOT / "data/built/real_test/rows.jsonl").exists() else []
    posting_ids = sorted({r["rq_id"] for r in postings})
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_real_test import load_postings  # noqa: E402
    texts = load_postings() if posting_ids else {}
    out, prompts = [], []
    profile = render_profile(CandidateCard())
    for i, p in enumerate(pdfs):
        raw = pdf_text(p)
        text, counts = redact(raw)
        text = prompt_ready(text)
        rec = {"id": p.stem, "category": p.stem.split("__")[0], "chars": len(text), "redacted": counts, "text": text}
        out.append(rec)
        if posting_ids:
            rq = posting_ids[i % len(posting_ids)]
            prompts.append({"id": p.stem, "rq_id": rq, "prompt": build_prompt(text, profile, texts[rq])})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("resumes.jsonl", out), ("prompts.jsonl", prompts)):
        with open(OUT / name, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    left = [r["id"] for r in out if EMAIL.search(r["text"]) or PHONE.search(r["text"]) or URL.search(r["text"])]
    print(f"{len(out)} resumes -> {OUT}/resumes.jsonl ({len(prompts)} prompts); contact data left in: {left}")
    print("median chars", sorted(r["chars"] for r in out)[len(out) // 2])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--n", type=int, default=20)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--dir", default=str(DEFAULT_DIR))
    e = sub.add_parser("extract")
    e.add_argument("--dir", default=str(DEFAULT_DIR))
    a = ap.parse_args()
    {"fetch": fetch, "extract": extract}[a.cmd](a)


if __name__ == "__main__":
    main()
