"""Cycle-3 training label batches (eval/protocol_amendment_5.json): blind batches in fresh test 3's format (row_id,
posting (whole), resume, profile) for the rubric teacher, labeler A's setup alone. Nothing here labels a row.

  cycle3_label_batches.py           -> data/built/fit/cycle3/label_batches/batch_NNN.jsonl,
                                       data/built/fit/cycle3/label_index.jsonl (row_id -> source and origin; never shown
                                       to the labeler), eval/cycle3_train_batches.json (counts, seeds, sha256)

Sources:
  new      data/built/fit/cycle3/rows.jsonl (build_fresh_test.py --set c3train: train pool, fixed generator)
  venator  train_fit.csv rows built on a real venator posting (data/built/fit/build/leak_rows.jsonl, split train)
  teaser   train_fit.csv teaser rows
  twins    a seeded random TWIN_N of the train_fit.csv twin rows

Posting text: new rows carry their whole posting; venator and teaser rows get theirs back from the cycle-1 sample
(build_fit_train.posting_text_of, the format training used, before fit_posting's 6,000-character cut, which
leak_rows.jsonl holds); twin postings are under 6,000 characters, so leak_rows holds them whole (checked). Resume and
profile are the texts training saw (leak_rows: resume redacted to "Candidate", profile form).

Old rows whose resume lists the posting's exact title as a held job title (title_copy: a resume line that is the
posting's title, as written or cleaned by build_fresh_test.clean_title) stay out: their resumes are not regenerated.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import build_fresh_test as bft  # noqa: E402

FIT = ROOT / "data/built/fit"
OUT = FIT / "cycle3"
BATCHES = OUT / "label_batches"
INDEX = OUT / "label_index.jsonl"
REPORT = ROOT / "eval/cycle3_train_batches.json"
SEED = 20261021
TWIN_N = 600
BATCH_SIZE = 75


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []


def title_copy(posting_title: str, resume: str) -> bool:
    """A resume line (its part before ' | ', ' - ', ', ' or ' (') that is the posting's title, as written or cleaned;
    a one-word cleaned title ('Director' from 'Director/Senior Director Finance') is too generic to count."""
    ban = {t for t in bft.posting_titles({"title": posting_title}) if len(t.split()) >= 2}
    for line in resume.split("\n"):
        head = re.split(r"\s[|–—-]\s|,\s|\s\(", line.strip(" -•\t"))[0].strip().casefold()
        if head and head in ban:
            return True
    return False


def title_of(posting: str) -> str:
    m = re.match(r"Title:\s*(.+)", posting)
    return m.group(1).strip() if m else ""


def rid(source: str, origin: str) -> str:
    return "c3_" + hashlib.sha1(f"{source}|{origin}".encode()).hexdigest()[:10]


def main() -> None:
    import build_fit_train as bfit
    sample = {r["key"]: r for r in jsonl(FIT / "build/sample.jsonl")}
    leak = [r for r in jsonl(FIT / "build/leak_rows.jsonl") if r["split"] == "train"]
    rows, dropped = [], Counter()
    for r in leak:
        if r["source"] not in ("venator", "teaser", "twins"):
            continue
        if r["source"] == "twins":
            posting = r["posting"]
            if len(posting) >= 6000 or "[truncated]" in posting:
                raise SystemExit(f"twin row {r['id']}: posting may be cut")
            title = title_of(posting)
        else:
            p = sample[r["posting_key"]]
            posting, title = bfit.posting_text_of(p), p["title"]
        copy = title_copy(title, r["resume"])
        rows.append({"row_id": rid(r["source"], r["id"]), "source": r["source"], "origin": r["id"],
                     "posting_key": r["posting_key"], "title_copy": copy, "rules_label": r["label"],
                     "posting": posting, "resume": r["resume"], "profile": r["profile"]})
    new = jsonl(OUT / "rows.jsonl")
    for r in new:
        rows.append({"row_id": r["row_id"], "source": "new", "origin": r["row_id"], "posting_key": r["posting_id"],
                     "title_copy": title_copy(r["title"], r["resume_text"]), "rules_label": None,
                     "posting": r["posting_text"], "resume": r["resume_text"], "profile": r["profile_text"]})
    if len({r["row_id"] for r in rows}) != len(rows):
        raise SystemExit("row id collision")
    avail = Counter(r["source"] for r in rows)
    copies = Counter(r["source"] for r in rows if r["title_copy"])
    keep = [r for r in rows if not r["title_copy"]]
    twins = sorted((r for r in keep if r["source"] == "twins"), key=lambda r: r["origin"])
    rng = random.Random(SEED)
    twin_pick = {r["row_id"] for r in rng.sample(twins, min(TWIN_N, len(twins)))}
    keep = [r for r in keep if r["source"] != "twins" or r["row_id"] in twin_pick]
    order = sorted(keep, key=lambda r: r["row_id"])
    random.Random(SEED + 1).shuffle(order)
    BATCHES.mkdir(parents=True, exist_ok=True)
    for old in BATCHES.glob("batch_*.jsonl"):
        old.unlink()
    for k in range(0, len(order), BATCH_SIZE):
        bft.write_jsonl(BATCHES / f"batch_{k // BATCH_SIZE + 1:03d}.jsonl",
                        [{x: r[x] for x in ("row_id", "posting", "resume", "profile")} for r in order[k:k + BATCH_SIZE]])
    bft.write_jsonl(INDEX, [{x: r[x] for x in ("row_id", "source", "origin", "posting_key", "title_copy", "rules_label")}
                            for r in order])
    batches = sorted(BATCHES.glob("batch_*.jsonl"))
    rel = lambda p: str(p.relative_to(ROOT))  # noqa: E731
    rep = {
        "what": "cycle-3 training label batches (eval/protocol_amendment_5.json), written by scripts/cycle3_label_batches.py; "
                "not labeled here",
        "rows": len(order), "batches": len(batches), "batch_size": BATCH_SIZE,
        "by_source": dict(Counter(r["source"] for r in order)),
        "available_by_source": dict(avail),
        "title_copy_excluded_by_source": dict(copies),
        "twin_draw": {"seed": SEED, "n": len(twin_pick), "from": sum(1 for r in rows if r["source"] == "twins" and not r["title_copy"])},
        "shuffle_seed": SEED + 1,
        "rules_label_mix_old_rows": dict(Counter(r["rules_label"] for r in order if r["rules_label"])),
        "posting_chars": {s: {"max": max(len(r["posting"]) for r in order if r["source"] == s),
                              "over_6000": sum(len(r["posting"]) > 6000 for r in order if r["source"] == s)}
                          for s in sorted({r["source"] for r in order})},
        "dir": rel(BATCHES),
        "index": rel(INDEX),
        "files": {rel(p): bft.sha256(p) for p in [*batches, INDEX]},
    }
    REPORT.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("rows", "batches", "by_source", "available_by_source",
                                          "title_copy_excluded_by_source", "posting_chars")}, indent=1))


if __name__ == "__main__":
    main()
