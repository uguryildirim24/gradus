"""The sealed v4 fresh test: 150 postings of the frozen test pool, ~600 candidate rows, blind labeling batches.

  build_fresh_test.py sample                 pick 150 postings of data/built/fit/pools/test.jsonl (no credits)
  build_fresh_test.py extract estimate|run   Adaption extracts their requirement cards with the v2 extractor as it is
                                             (extract_postings.BLUEPRINT / UNIVERSAL / adapt_kwargs / parse_card, imported,
                                             never edited) -> data/built/fresh_test/cards.jsonl
  build_fresh_test.py candidates             ~4 candidates per posting (strong / near / gap / off fits, eligibility mixed)
                                             -> data/built/fresh_test/candidates.jsonl (no credits)
  build_fresh_test.py render estimate|run    Adaption writes each candidate's resume from their card; failed gates are
                                             retried with feedback (up to MAX_TRIES) -> data/built/fresh_test/resumes.jsonl
  build_fresh_test.py assemble               rows.jsonl, label_batches/batch_NN.jsonl, blind_sheet.jsonl
  build_fresh_test.py seal                   eval/fresh_test_manifest.json (sha256 of every data file)

  --set 2 (before the command) builds fresh test 2 the same way into data/built/fresh_test2 with its own seed, row ids
  f2_..., a 10-credit cap and eval/fresh_test2_manifest.json. Its sample is not drawn: it is every English test-pool
  posting fresh test 1 did not use (no employer, teaser or country caps), so it shares employers with fresh test 1.

  --set 3 builds fresh test 3 the same way from the dev pool (data/built/fit/pools/dev.jsonl: employers disjoint from
  the test pool and from training) into data/built/fresh_test3, seed 20261223, row ids f3_..., a 12-credit cap and
  eval/fresh_test3_manifest.json. Its sample is set 1's sampler (same teaser, adzuna, empty and non-US caps, field
  quotas) with a hard employer cap of 4 (no +2 fallback), so it stops short of 150 when the cap runs out.

Sets 1 and 2 build from the test pool only, set 3 from the dev pool only: no train posting, no train-builder output,
no training resume prompt. Set 3's seal alone reads the train pool's employer keys, the training csvs and cycle 1's
dev_fit posting keys, to check and record disjointness and earlier use; nothing it builds comes from them.

Not gold. Nothing here labels a row. rules.label is never called; hybrid_rules.rule_verdict is read for one thing,
the design metadata "eligibility pass on the extracted card" (no hard Skip in H' categories and no salary cap: the
rows whose label hinges on the resume against the posting's asks). Gold comes later from blind labelers who see only
the label batches (row_id, posting text, resume text, the profile as a form).

Candidate fit is seeded from the posting TEXT as a reader sees it (skill mentions found with a vocabulary drawn from
the ATS training-side v2 cards and per-field banks here, sorted by where they sit: required, preferred, a "such as /
e.g. / one of" list, or a duty line), never from the extracted card's must_skills, so the full-rules baseline gets no
free ride. Eligibility is steered on the extracted card, since that is what the eligibility layer reads.

Since cycle 3 (eval/protocol_amendment_5.json; sets 1-3 were built before it): no job a candidate holds carries the
target posting's title (posting_titles), and the skills list is the person's own: the posting mentions their archetype
holds (requirements, list picks, half the preferred items, at most one duty mention) mixed into their field's skills,
none of which the posting names, in a shuffled order.

--set 5 builds fresh test 5 (eval/protocol_amendment_6.json, the fallback design) from the round-2 real postings: the
Simplify ATS postings behind data/built/real_test and data/built/real_train, minus the amendment's 32 dropped ids, one
posting per employer key and title, English. Each is adapted to the pool-posting fields (round2_adapter; field from the
title by fit_pools.field_of, employer key by matcher.nearcopy.company_key) and keeps its existing Adaption v2 card
(data/built/cards/extracted_real_cards_v2.jsonl): nothing is extracted. The draw is the amendment's seeded walk with an
employer cap of 7 (cmd_sample_round2); the whole walk order is written to sample_walk.json for the extension rule. The
old rows' resumes, profiles and labels are never read. Candidates use the c3train lean (pass_rate 0.75, near/gap/strong
0.3/0.3/0.4), seed 20261022, row ids f5_...
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import extract_postings as ext  # noqa: E402  the v2 extractor, used as it is
from fit_pools import load_pool  # noqa: E402
from matcher.blueprint import parse_card, posting_meta  # noqa: E402

# One entry per sealed test. Everything below that names a path, the seed, the row-id prefix or the credit cap is
# rebound by use_set() before a command runs; the build itself is the same for every set.
SETS = {
    "1": dict(out="data/built/fresh_test", manifest="eval/fresh_test_manifest.json", seed=20261021, prefix="ft_",
              cap=25.0, dataset="fresh-test", card_source="extracted_fresh_test"),
    "2": dict(out="data/built/fresh_test2", manifest="eval/fresh_test2_manifest.json", seed=20261122, prefix="f2_",
              cap=10.0, dataset="fresh-test2", card_source="extracted_fresh_test2"),
    "3": dict(out="data/built/fresh_test3", manifest="eval/fresh_test3_manifest.json", seed=20261223, prefix="f3_",
              cap=12.0, dataset="fresh-test3", card_source="extracted_fresh_test3", pool="dev", employer_cap=4,
              cap_slack=0),
    "4": dict(out="data/built/fresh_test4", manifest="eval/fresh_test4_manifest.json", seed=20261031, prefix="f4_",
              cap=20.0, dataset="fresh-test4", card_source="extracted_fresh_test4", pool="reserved", employer_cap=7,
              cap_slack=0, n_postings=240, n_teaser=27, n_empty=5, max_adzuna=14, non_us_cap=48),
    # amendment 4b: about 50 more postings for fresh test 4 from the same reserved set and the same design, none of
    # set 4's postings, set 4's postings counted toward the employer cap; merged into set 4 by `extend`
    "4b": dict(out="data/built/fresh_test4b", manifest="eval/fresh_test4_manifest.json", seed=20261032, prefix="f4_",
               cap=6.0, dataset="fresh-test4b", card_source="extracted_fresh_test4", pool="reserved", employer_cap=7,
               cap_slack=0, n_postings=50, n_teaser=0, n_empty=1, max_adzuna=0, non_us_cap=10, extends="4"),
    # cycle-3 training rows (eval/protocol_amendment_5.json): the same pipeline on the train pool, never sealed or
    # scored. Five rows a posting and a lean toward passing, stronger candidates, so the rubric labels land nearer the
    # amendment's mix band (fresh tests 1-3 run about 7% Keep) and fewer paid rows are downsampled away.
    "c3train": dict(out="data/built/fit/cycle3", manifest="data/built/fit/cycle3/build_manifest.json", seed=20261131,
                    prefix="c3_", cap=40.0, dataset="cycle3-train", card_source="extracted_cycle3_train", pool="train3",
                    employer_cap=8, cap_slack=0, n_postings=520, n_teaser=40, n_empty=10, max_adzuna=20,
                    non_us_cap=100, per_posting=5, pass_rate=0.75, arch_w=(0.3, 0.3, 0.4)),
    # amendment 6: sealed fresh test 5 on the round-2 real postings and their existing v2 cards; no Adaption credits
    "5": dict(out="data/built/fresh_test5", manifest="eval/fresh_test5_manifest.json", seed=20261022, prefix="f5_",
              cap=0.0, dataset="fresh-test5", card_source="extracted_real", pool="round2", employer_cap=7,
              cap_slack=0, n_postings=300, per_posting=4, pass_rate=0.75, arch_w=(0.3, 0.3, 0.4), writer="sonnet"),
}
SET = "1"
OUT = ROOT / "data/built/fresh_test"
STATE = OUT / "adaption"
POSTINGS = OUT / "postings.jsonl"
CARDS = OUT / "cards.jsonl"
CANDS = OUT / "candidates.jsonl"
RESUMES = OUT / "resumes.jsonl"
TRIES = OUT / "render_tries.jsonl"
ROWS = OUT / "rows.jsonl"
BATCHES = OUT / "label_batches"
SHEET = OUT / "blind_sheet.jsonl"
CREDITS = OUT / "credits.json"
MANIFEST = ROOT / "eval/fresh_test_manifest.json"
CREDIT_CAP = 25.0
ROW_PREFIX = "ft_"
DATASET = "fresh-test"
CARD_SOURCE = "extracted_fresh_test"
FRESH1_MANIFEST = ROOT / "eval/fresh_test_manifest.json"
FRESH2_MANIFEST = ROOT / "eval/fresh_test2_manifest.json"
POOL = "test"

SEED = 20261021
N_POSTINGS = 150
N_TEASER, N_EMPTY = 17, 3          # short postings: 1-699 characters, and bodies with no text at all
MAX_ADZUNA_TEASERS = 9             # all 32 adzuna rows are 502-character Boston-area truncations
EMPLOYER_CAP = 7
CAP_SLACK = 2                      # set 1: when no capped candidate is left, an employer may go 2 over the cap
NON_US_CAP = 30                    # clearly foreign (posting_meta country CA, UK or other; "unknown" is mostly bare US
                                   # city names): a fifth, or location becomes the test's dominant filter
PER_POSTING = 4
PASS_RATE = 0.58                   # share of candidates steered to pass eligibility
ARCH_W = (0.4, 0.4, 0.2)           # near, gap, strong weights of the slots after one strong and one off
BATCH_SIZE = 75
SHEET_N = 100
WRITER = "adaption"                # who writes the resumes: Adaption (sets 1-4, c3train) or Claude Sonnet (set 5)

csv.field_size_limit(10**8)


# ---------------------------------------------------------------- io


def use_set(name: str) -> None:
    """Point every path, the seed, the row-id prefix and the credit cap at one sealed test."""
    global SET, OUT, STATE, POSTINGS, CARDS, CANDS, RESUMES, TRIES, ROWS, BATCHES, SHEET, CREDITS, MANIFEST, CREDIT_CAP
    global SEED, ROW_PREFIX, DATASET, CARD_SOURCE, POOL, EMPLOYER_CAP, CAP_SLACK
    c = SETS[name]
    SET = name
    OUT = ROOT / c["out"]
    STATE, POSTINGS, CARDS, CANDS = OUT / "adaption", OUT / "postings.jsonl", OUT / "cards.jsonl", OUT / "candidates.jsonl"
    RESUMES, TRIES, ROWS = OUT / "resumes.jsonl", OUT / "render_tries.jsonl", OUT / "rows.jsonl"
    BATCHES, SHEET, CREDITS = OUT / "label_batches", OUT / "blind_sheet.jsonl", OUT / "credits.json"
    MANIFEST, CREDIT_CAP, SEED, ROW_PREFIX = ROOT / c["manifest"], c["cap"], c["seed"], c["prefix"]
    DATASET, CARD_SOURCE = c["dataset"], c["card_source"]
    POOL, EMPLOYER_CAP, CAP_SLACK = c.get("pool", "test"), c.get("employer_cap", 7), c.get("cap_slack", 2)
    global N_POSTINGS, N_TEASER, N_EMPTY, MAX_ADZUNA_TEASERS, NON_US_CAP
    N_POSTINGS, N_TEASER, N_EMPTY = c.get("n_postings", 150), c.get("n_teaser", 17), c.get("n_empty", 3)
    MAX_ADZUNA_TEASERS, NON_US_CAP = c.get("max_adzuna", 9), c.get("non_us_cap", 30)
    global PER_POSTING, PASS_RATE, ARCH_W
    PER_POSTING, PASS_RATE, ARCH_W = c.get("per_posting", 4), c.get("pass_rate", 0.58), c.get("arch_w", (0.4, 0.4, 0.2))
    global WRITER
    WRITER = c.get("writer", "adaption")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def h(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


# ---------------------------------------------------------------- sample

_GERMAN = re.compile(r"\b(?:und|der|die|das|wir|sie|mit|für|bei|eine?n?|ist|zur|zum|Ihre?|Kenntnisse|Erfahrung)\b")


def language(text: str) -> str:
    """'en', 'fr' (posting_meta's own French check) or 'de' (German stopwords)."""
    if posting_meta([], text)["language"] == "fr":
        return "fr"
    return "de" if len(_GERMAN.findall(text[:4000])) >= 12 else "en"


def adapter(p: dict) -> dict:
    """A venator pool posting in the field names extract_postings.posting_text reads."""
    locs = [x.strip() for x in re.split(r";", p.get("locations") or "") if x.strip()]
    return {"title": p["title"], "company": p["employer"], "locations": locs, "description": p["text"] or ""}


def posting_text(p: dict) -> str:
    """The whole posting as everyone downstream sees it: extract_postings.posting_text without the 12,000 cut."""
    return ext.posting_text(adapter(p), None)


def fresh1_postings() -> set[str]:
    return set(json.loads(FRESH1_MANIFEST.read_text())["posting_ids"])


def fresh1_employers() -> set[str]:
    used = fresh1_postings()
    return {p["emp_key"] for p in load_pool("test") if p["key"] in used}


def fresh_test_keys(manifest: Path) -> set[str]:
    return set(json.loads(manifest.read_text())["posting_ids"])


def dev_fit_postings() -> tuple[set[str], set[str]]:
    """Posting keys of the cycle-1 dev_fit selection set (data/built/fit/dev_fit.jsonl, the rows that selected h in
    eval/selection_v4.json) and of every dev row build_fit_train made before its filter (build/dev_fit_all.jsonl)."""
    fit = ROOT / "data/built/fit"
    return ({r["posting_key"] for r in read_jsonl(fit / "dev_fit.jsonl")},
            {r["posting_key"] for r in read_jsonl(fit / "build/dev_fit_all.jsonl")})


TRAIN_FILES = ("data/built/fit/train_fit.csv", "data/built/fit/train_fit2_bal.csv", "data/built/fit/train_fit2_nat.csv",
               "data/built/fit/dpo_pairs.csv")


def training_text_hits(posts: dict[str, dict], files: tuple[str, ...] = TRAIN_FILES) -> dict:
    """Postings whose text sits in a training file: three 12-word windows (at 25%, 50% and 75% of the normalized body)
    searched in every column of every training csv, normalized the same way; two of three found is a hit. Bodies under
    60 words are listed as too short to check."""
    from matcher.nearcopy import norm_text
    probes: dict[str, list[str]] = {}
    short = []
    for k, p in posts.items():
        w = norm_text(p["text"] or "").split()
        if len(w) < 60:
            short.append(k)
            continue
        probes[k] = [" ".join(w[int(len(w) * f):int(len(w) * f) + 12]) for f in (0.25, 0.5, 0.75)]
    found: Counter = Counter()
    for f in files:
        blob = norm_text((ROOT / f).read_text())
        for k, ws in probes.items():
            found[k] = max(found[k], sum(x in blob for x in ws))
        del blob
    return {"rule": training_text_hits.__doc__.split("\n")[0].strip() + " ...", "files": list(files),
            "checked": len(probes), "too_short_to_check": len(short),
            "hits": sorted(k for k, n in found.items() if n >= 2),
            "one_window_found": sorted(k for k, n in found.items() if n == 1)}


FRESH3_MANIFEST = ROOT / "eval/fresh_test3_manifest.json"
RESERVED_KEY = "reserved (test/dev employers' postings outside the sealed sample; in no pool, never trained on)"


def load_reserved() -> tuple[list[dict], dict]:
    """Fresh test 4's source: the postings eval/fit_freeze.json reserved (data/built/fit/split.json; every other
    posting of a test or dev employer, in no pool and no test), with the pool fields (chars, field, length, emp_key:
    the frozen employer cluster, found through the cluster names' employer tokens). Dropped, with the reason in the
    returned notes: near copies (matcher.nearcopy rules) of any fresh test 1-3 posting or of any train-pool posting,
    and every posting but the first (by key) of one employer and title."""
    from fit_pools import RareIndex, as_near, employer_tokens, field_of, length_of, load_pool
    split = json.loads((ROOT / "data/built/fit/split.json").read_text())[RESERVED_KEY]
    side = {k: name for name, ks in split.items() for k in ks}
    raw = {}
    with (ROOT / "data/built/fit/venator_postings.jsonl").open() as f:
        for line in f:
            p = json.loads(line)
            if p["key"] in side:
                raw[p["key"]] = p
    if len(raw) != len(side):
        raise SystemExit(f"{len(side) - len(raw)} reserved postings missing from the snapshot")
    pools = {n: load_pool(n) for n in ("test", "dev", "train")}
    tok: dict[str, set[str]] = defaultdict(set)
    for n in ("test", "dev"):
        for q in pools[n]:
            for t in employer_tokens(q["employer"]):
                tok[t].add(q["emp_key"])
    used = fresh_test_keys(FRESH1_MANIFEST) | fresh_test_keys(FRESH2_MANIFEST) | fresh_test_keys(FRESH3_MANIFEST)
    fresh = {q["key"]: q for n in ("test", "dev") for q in pools[n] if q["key"] in used}
    near_fresh = RareIndex({k: as_near(q) for k, q in fresh.items()})
    near_train = RareIndex({q["key"]: as_near(q) for q in pools["train"]})
    out, drop = [], Counter()
    seen_title: set[tuple[str, str]] = set()
    from matcher.nearcopy import norm_text
    for k in sorted(raw):
        p = raw[k]
        es = set().union(*[tok.get(t, set()) for t in employer_tokens(p["employer"] or p["board"])] or [set()])
        if len(es) != 1:
            raise SystemExit(f"{k}: reserved posting maps to {len(es)} frozen employers")
        q = {f: p[f] for f in ("key", "source", "board", "employer", "title", "locations", "url", "posted_at",
                               "workplace_type", "opportunity_type", "description_kind", "text")}
        q.update(chars=len(p["text"]), field=field_of(p["title"]), length=length_of(len(p["text"])), emp_key=es.pop(),
                 reserved_side=side[k])
        if near_fresh.near(as_near(q)):
            drop["near copy of a fresh test 1-3 posting"] += 1
            continue
        if near_train.near(as_near(q), title_rule=False):
            drop["near copy of a train-pool posting"] += 1
            continue
        t = (q["emp_key"], norm_text(q["title"]))
        if t in seen_title:
            drop["same employer and title as an earlier reserved posting"] += 1
            continue
        seen_title.add(t)
        out.append(q)
    notes = {"reserved": len(raw), "by_side": dict(Counter(side.values())), "dropped": dict(drop), "kept": len(out),
             "kept_by_length": dict(Counter(q["length"] for q in out))}
    return out, notes


AMENDMENT6 = ROOT / "eval/protocol_amendment_6.json"
ATS_POSTINGS = ROOT / "data/raw/ats/postings.jsonl"
ATS_CARDS = ROOT / "data/built/cards/extracted_real_cards_v2.jsonl"
ROUND2_SETS = (("real_test", "data/built/real_test/test.jsonl"), ("real_train train", "data/built/real_train/train.jsonl"),
               ("real_train dev", "data/built/real_train/dev.jsonl"))


def round2_adapter(d: dict, source_set: str) -> dict:
    """One Simplify ATS posting (data/raw/ats/postings.jsonl) in the pool-posting field names the sampler and
    candidate builder read. adapter() of the result gives back the ATS posting's own title, company, locations and
    description, so posting_text() equals extract_postings.posting_text(ats posting, None)."""
    from fit_pools import field_of, length_of
    from matcher.nearcopy import company_key
    locs = d.get("locations") or []
    if any(not x or ";" in x or x != x.strip() for x in locs):
        raise ValueError(f"ats_{d['simplify_id']}: a location that '; '-joining would not round-trip: {locs}")
    text = d.get("description") or ""
    return {"key": "ats_" + d["simplify_id"], "source": "simplify_ats", "board": d.get("ats"),
            "employer": d["company"], "title": d["title"], "locations": "; ".join(locs), "url": d.get("url"),
            "text": text, "chars": len(text), "field": field_of(d["title"]), "length": length_of(len(text)),
            "emp_key": company_key(d["company"]), "source_set": source_set, "category": d.get("category"),
            "terms": d.get("terms"), "country": posting_meta(locs, text)["country"]}


def load_round2() -> tuple[list[dict], dict]:
    """Fresh test 5's pool (amendment 6 free_pool_finding.disjoint_pool and fresh_test_5.postings.filter): every
    posting behind a round-2 real-pair row (base_rq_id of data/built/real_test/test.jsonl and real_train's train and
    dev), minus the amendment's 32 dropped ids, then one posting per (employer key, normalized title), the first by id,
    then English only. Adapted by round2_adapter, sorted by id."""
    from matcher.nearcopy import norm_text
    side: dict[str, set[str]] = defaultdict(set)
    for name, f in ROUND2_SETS:
        for r in read_jsonl(ROOT / f):
            side[r["base_rq_id"]].add(name)
    if any(len(v) > 1 for v in side.values()):
        raise SystemExit("a round-2 posting sits in two source sets")
    dropped = set(json.loads(AMENDMENT6.read_text())["free_pool_finding"]["overlap_counts_round2_real"]["dropped_ids"])
    ats = {"ats_" + d["simplify_id"]: d for d in read_jsonl(ATS_POSTINGS)}
    missing = sorted(set(side) - set(ats))
    if missing:
        raise SystemExit(f"{len(missing)} round-2 postings missing from {ATS_POSTINGS}")
    seen: set[tuple[str, str]] = set()
    out, dup, lang_out = [], [], []
    for k in sorted(set(side) - dropped):
        p = round2_adapter(ats[k], next(iter(side[k])))
        t = (p["emp_key"], norm_text(p["title"]))
        if t in seen:
            dup.append(k)
            continue
        seen.add(t)
        lang = language(p["text"])
        if lang != "en":
            lang_out.append({"posting_id": k, "why": f"language {lang}"})
            continue
        out.append(p)
    notes = {"round2_postings": len(side), "by_source_set": dict(Counter(next(iter(v)) for v in side.values())),
             "dropped_by_amendment6": sorted(set(side) & dropped), "dropped_ids_not_in_pool": sorted(dropped - set(side)),
             "after_drop": len(set(side) - dropped), "same_employer_and_title_after_the_first": len(dup),
             "same_employer_and_title_ids": dup, "not_english": lang_out, "kept": len(out),
             "kept_employers": len({p["emp_key"] for p in out})}
    return out, notes


def cmd_sample_round2() -> None:
    """Amendment 6's draw: sort the kept ids, shuffle with random.Random(SEED), walk the order and take a posting
    unless its employer key already has EMPLOYER_CAP postings taken; the test is the first N_POSTINGS taken. The whole
    walk (every posting it would take, in order) goes to sample_walk.json for the extension rule. Writes postings.jsonl
    and cards.jsonl (each posting's existing v2 card, copied as it is)."""
    pool, notes = load_round2()
    by = {p["key"]: p for p in pool}
    order = sorted(by)
    random.Random(SEED).shuffle(order)
    emp: Counter = Counter()
    walk, capped = [], []
    for k in order:
        if emp[by[k]["emp_key"]] >= EMPLOYER_CAP:
            capped.append(k)
            continue
        emp[by[k]["emp_key"]] += 1
        walk.append(k)
    picked = walk[:N_POSTINGS]
    cards = {r["id"]: r for r in read_jsonl(ATS_CARDS)}
    if any(k not in cards for k in picked):
        raise SystemExit(f"{sum(k not in cards for k in picked)} drawn postings have no v2 card in {ATS_CARDS}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "round2_notes.json").write_text(json.dumps(notes, indent=1))
    (OUT / "sample_walk.json").write_text(json.dumps({
        "rule": cmd_sample_round2.__doc__.split("\n")[0].strip() + " ...",
        "seed": SEED, "employer_cap": EMPLOYER_CAP, "n_postings": N_POSTINGS, "pool_kept": len(order),
        "walk": walk, "skipped_at_employer_cap": capped}, indent=1))
    write_jsonl(POSTINGS, sorted((by[k] for k in picked), key=lambda p: p["key"]))
    write_jsonl(CARDS, [{**cards[k], "copied_from": os.path.relpath(ATS_CARDS, ROOT)} for k in sorted(picked)])
    (OUT / "sample_excluded.json").write_text(json.dumps(
        [{"posting_id": k, "why": "dropped by amendment 6 (overlap_rule)"} for k in notes["dropped_by_amendment6"]] +
        [{"posting_id": k, "why": "same employer key and title as an earlier id"} for k in notes["same_employer_and_title_ids"]] +
        notes["not_english"], indent=1))
    taken = [by[k] for k in picked]
    print(f"{len(picked)} postings drawn of {len(order)} kept (walk length {len(walk)}); {len(emp)} employers in the "
          f"walk, {len({p['emp_key'] for p in taken})} in the test, max {max(Counter(p['emp_key'] for p in taken).values())}")
    for k in ("field", "category", "length", "source_set", "country"):
        print(k, dict(Counter(p[k] for p in taken)))


def load_train3() -> tuple[list[dict], dict]:
    """Cycle-3 training rows' source: the train pool minus every posting the cycle-1 train builder sampled
    (data/built/fit/build/sample.jsonl) and minus the employers it never sampled (kept unused, so a test built from
    them later would share no employer with any training file). The train pool shares no employer with the test and
    dev pools, so none with fresh tests 1-4; checked here."""
    from fit_pools import load_pool
    pool = load_pool("train")
    cyc1 = read_jsonl(ROOT / "data/built/fit/build/sample.jsonl")
    used = {r["key"] for r in cyc1}
    touched = {p["emp_key"] for p in pool if p["key"] in used}
    fresh_emp = {p["emp_key"] for n in ("test", "dev") for p in load_pool(n)}
    if fresh_emp & {p["emp_key"] for p in pool}:
        raise SystemExit("train pool shares an employer with the test or dev pool")
    out = [p for p in pool if p["key"] not in used and p["emp_key"] in touched]
    notes = {"train_pool": len(pool), "cycle1_sampled_postings_dropped": sum(p["key"] in used for p in pool),
             "untouched_employers_held_out": sorted({p["emp_key"] for p in pool} - touched),
             "untouched_employer_postings_held_out": sum(p["emp_key"] not in touched for p in pool),
             "kept": len(out), "employers_kept": len({p["emp_key"] for p in out}),
             "employers_shared_with_test_or_dev_pool": 0}
    return out, notes


def cmd_sample_rest() -> None:
    """Fresh test 2: every English test-pool posting fresh test 1 did not use. Nothing is drawn or capped."""
    used = fresh1_postings()
    excluded, out = [], []
    for p in sorted(load_pool("test"), key=lambda p: p["key"]):
        if p["key"] in used:
            continue
        lang = language(p["text"] or "")
        if lang != "en":
            excluded.append({"posting_id": p["key"], "why": f"language {lang}"})
            continue
        meta = posting_meta(adapter(p)["locations"], p["text"] or "")
        out.append({**p, "country": meta["country"]})
    write_jsonl(POSTINGS, out)
    (OUT / "sample_excluded.json").write_text(json.dumps(excluded, indent=1))
    e1 = fresh1_employers()
    print(f"{len(out)} postings (every test-pool posting fresh test 1 did not use); excluded {len(excluded)} not in "
          f"English; {sum(p['emp_key'] in e1 for p in out)} from employers fresh test 1 also has")
    for k in ("field", "length", "source", "country"):
        print(k, dict(Counter(p[k] for p in out)))
    print("employers", len({p['emp_key'] for p in out}), "max", max(Counter(p['emp_key'] for p in out).values()))


def cmd_sample(_a) -> None:
    if SET == "2":
        return cmd_sample_rest()
    if POOL == "round2":
        return cmd_sample_round2()
    if POOL in ("reserved", "train3"):
        pool, notes = load_reserved() if POOL == "reserved" else load_train3()
        (OUT / f"{POOL}_notes.json").parent.mkdir(parents=True, exist_ok=True)
        (OUT / f"{POOL}_notes.json").write_text(json.dumps(notes, indent=1))
        print(POOL, notes)
    else:
        pool = load_pool(POOL)
    rng = random.Random(SEED)
    excluded = []
    ok = []
    for p in sorted(pool, key=lambda p: p["key"]):
        lang = language(p["text"] or "")
        if lang != "en":
            excluded.append({"posting_id": p["key"], "why": f"language {lang}"})
            continue
        meta = posting_meta(adapter(p)["locations"], p["text"] or "")
        ok.append({**p, "_us": meta["country"] in ("US", "unknown"), "_country": meta["country"]})
    rng.shuffle(ok)
    picked: list[dict] = []
    emp = Counter()
    nonus = 0
    base = SETS[SET].get("extends")
    if base:  # an extension: none of the base set's postings, and they count toward the employer cap
        first = read_jsonl(ROOT / SETS[base]["out"] / "postings.jsonl")
        taken = {q["key"] for q in first}
        ok = [q for q in ok if q["key"] not in taken]
        emp.update(q["emp_key"] for q in first)

    def take(p: dict, cap: int = EMPLOYER_CAP) -> bool:
        nonlocal nonus
        if emp[p["emp_key"]] >= cap or (not p["_us"] and nonus >= NON_US_CAP):
            return False
        picked.append(p)
        emp[p["emp_key"]] += 1
        nonus += not p["_us"]
        return True

    empties = [p for p in ok if p["length"] == "empty"]
    teasers = [p for p in ok if p["length"] == "teaser 1-699"]
    # short postings first: every non-adzuna teaser (catch-alls, near-empty Workday bodies), then adzuna up to its cap
    n = 0
    for p in sorted(teasers, key=lambda p: p["source"] == "adzuna"):
        if n >= N_TEASER:
            break
        if p["source"] == "adzuna" and sum(q["source"] == "adzuna" for q in picked) >= MAX_ADZUNA_TEASERS:
            continue
        n += take(p)
    n = 0
    for p in empties:
        if n >= N_EMPTY:
            break
        n += take(p)
    # the rest: long postings, field quotas as even as the pool allows, rarer sources first in ties
    fields = sorted({p["field"] for p in ok})
    quota = {f: N_POSTINGS // len(fields) + (i < N_POSTINGS % len(fields)) for i, f in enumerate(fields)}
    longs = [p for p in ok if p["chars"] >= 700]
    src_n = Counter(p["source"] for p in longs)
    while len(picked) < N_POSTINGS:
        have = Counter(p["field"] for p in picked)
        open_f = [f for f in fields if have[f] < quota[f]] or fields
        cands = [p for p in longs if p not in picked and p["field"] in open_f
                 and emp[p["emp_key"]] < EMPLOYER_CAP and (p["_us"] or nonus < NON_US_CAP)]
        cap = EMPLOYER_CAP
        if not cands:  # any field; set 1 also lets an employer go CAP_SLACK over the cap here, set 3 does not
            cap = EMPLOYER_CAP + CAP_SLACK
            cands = [p for p in longs if p not in picked and emp[p["emp_key"]] < cap
                     and (p["_us"] or nonus < NON_US_CAP)]
        if not cands:
            break
        # lowest-filled field first, then the employer with fewest picks, then a rarer source
        cands.sort(key=lambda p: (have[p["field"]] / quota[p["field"]], emp[p["emp_key"]],
                                  src_n[p["source"]] * 0.001 + rng.random() * 0.01))
        if not take(cands[0], cap):
            break
    out = [{k: v for k, v in p.items() if not k.startswith("_")} | {"country": p["_country"]}
           for p in picked]
    out.sort(key=lambda p: p["key"])
    write_jsonl(POSTINGS, out)
    (OUT / "sample_excluded.json").write_text(json.dumps(excluded, indent=1))
    print(f"{len(out)} postings; excluded {len(excluded)} not in English")
    for k in ("field", "length", "source", "country"):
        print(k, dict(Counter(p[k] for p in out)))
    print("employers", len({p['emp_key'] for p in out}), "max", max(Counter(p['emp_key'] for p in out).values()))


# ---------------------------------------------------------------- Adaption plumbing


def credits_log() -> dict:
    return json.loads(CREDITS.read_text()) if CREDITS.exists() else {"runs": []}


def spent() -> float:
    return round(sum(r["estimate"] for r in credits_log()["runs"] if r.get("launched")), 3)


def log_run(entry: dict) -> None:
    log = credits_log()
    log["runs"] = [r for r in log["runs"] if r["name"] != entry["name"]] + [entry]
    log["total_launched_estimate"] = round(sum(r["estimate"] for r in log["runs"] if r.get("launched")), 3)
    CREDITS.parent.mkdir(parents=True, exist_ok=True)
    CREDITS.write_text(json.dumps(log, indent=1))


def next_batch(prefix: str) -> str:
    """The batch to work on: the last one under this prefix if it never finished (resume it), else a new one."""
    done = sorted(STATE.glob(f"{prefix}_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[1])) if STATE.exists() else []
    if done and not json.loads(done[-1].read_text()).get("finished"):
        return done[-1].stem
    return f"{prefix}_{len(done)}"


def upload_csv(c, name: str, header: list[str], rows: list[list[str]]) -> str:
    """Upload once per batch name (the state file remembers the dataset); wait until the import has a row count."""
    st_path = STATE / f"{name}.json"
    state = json.loads(st_path.read_text()) if st_path.exists() else {}
    if state.get("dataset_id"):
        return state["dataset_id"]
    path = STATE / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    did = c.datasets.upload_file(str(path), name=f"{DATASET}-{name}").dataset_id
    st_path.write_text(json.dumps({"dataset_id": did, "n": len(rows)}, indent=1))
    for _ in range(120):
        s = c.datasets.get_status(did)
        if s.row_count or s.status == "failed":
            break
        time.sleep(5)
    return did


def run_batch(c, name: str, did: str, kwargs: dict, mode: str, max_credits: float, key: str) -> dict[str, str] | None:
    """Estimate (free); with mode 'run', launch unless that would pass the credit cap, wait, download.
    Returns {id column value: completion} or None when only estimating."""
    st_path = STATE / f"{name}.json"
    state = json.loads(st_path.read_text())
    if "estimate" not in state:
        est = c.datasets.adapt(did, estimate=True, **kwargs)
        state["estimate"] = float(est.estimated_credits_consumed)
        st_path.write_text(json.dumps(state, indent=1))
    print(f"{name}: estimate {state['estimate']} credits ({state['n']} rows); spent so far {spent()}")
    if mode == "estimate":
        return None
    if "run_id" not in state:
        if state["estimate"] > max_credits or spent() + state["estimate"] > CREDIT_CAP:
            sys.exit(f"{name}: estimate {state['estimate']} over --max-credits {max_credits} or the {CREDIT_CAP} cap")
        r = c.datasets.adapt(did, estimate=False, **kwargs)
        state.update(run_id=r.run_id, started=time.time())
        st_path.write_text(json.dumps(state, indent=1))
        log_run({"name": name, "estimate": state["estimate"], "rows": state["n"], "launched": True, "run_id": r.run_id})
        time.sleep(20)
    st, n, tot = ext.wait_done(c, did, interval=20.0)
    if st.status != "succeeded":
        sys.exit(f"{name}: adapt ended {st.status}")
    res = c.datasets.download(did, file_format="csv")
    data = res.read() if hasattr(res, "read") else res.content
    (STATE / f"{name}_download.csv").write_bytes(data)
    state.update(finished=time.time(), processed=n, total=tot)
    st_path.write_text(json.dumps(state, indent=1))
    return completions(name, key)


def completions(name: str, key: str) -> dict[str, str]:
    """{id: completion} from a batch's saved download."""
    data = (STATE / f"{name}_download.csv").read_text()
    return {r[key]: (r.get("enhanced_completion") or "").strip() for r in csv.DictReader(io.StringIO(data, newline=""))}


# ---------------------------------------------------------------- extract


def cmd_extract(a) -> None:
    if POOL == "round2":
        raise SystemExit(f"set {SET} uses the existing v2 cards: nothing to extract")
    posts = {p["key"]: p for p in read_jsonl(POSTINGS)}
    have = {r["id"] for r in read_jsonl(CARDS)}
    todo = [i for i in sorted(posts) if i not in have]
    print(f"extractor blueprint {ext.BP_HASH}; {len(posts)} postings, {len(todo)} without a card")
    if not todo:
        return
    c = ext.client()
    name = next_batch(f"extract_{ext.BP_HASH}")
    did = upload_csv(c, name, ["posting_id", "posting"], [[i, ext.posting_text(adapter(posts[i]))] for i in todo])
    got = run_batch(c, name, did, ext.adapt_kwargs(), a.mode, a.max_credits, "posting_id")
    if got is None:
        return
    raw = read_jsonl(OUT / "extract_raw.jsonl")
    raw += [{"posting_id": i, "batch": name, "completion": t} for i, t in got.items() if t and i in posts]
    write_jsonl(OUT / "extract_raw.jsonl", raw)
    parse_cards()


def parse_cards() -> None:
    """Raw completions -> cards.jsonl the way extract_postings.parse_all cleans them (latest completion per id)."""
    if POOL == "round2":
        raise SystemExit(f"set {SET} uses the existing v2 cards: nothing to parse")
    posts = {p["key"]: p for p in read_jsonl(POSTINGS)}
    latest = {}
    for r in read_jsonl(OUT / "extract_raw.jsonl"):
        latest[r["posting_id"]] = r["completion"]
    cards, fails = [], []
    for i, text in sorted(latest.items()):
        ad = adapter(posts[i])
        try:
            body = text.strip()
            if body.startswith("```"):
                body = body.strip("`").removeprefix("json").strip()
            card = parse_card(i, body, ext.posting_text(ad, None), ad["locations"])
        except Exception as e:  # noqa: BLE001
            fails.append({"posting_id": i, "error": f"{type(e).__name__}: {e}", "completion": text[:600]})
            continue
        meta = posting_meta(ad["locations"], ad["description"], card.locations)
        cards.append({"id": i, "source": CARD_SOURCE, "extractor": ext.BP_HASH, "card": card.to_dict(),
                      "meta": meta})
    write_jsonl(CARDS, cards)
    write_jsonl(OUT / "extract_failures.jsonl", fails)
    print(f"{len(cards)} cards, {len(fails)} failed to parse, {len(posts) - len(latest)} postings with no completion")


# ---------------------------------------------------------------- candidate banks
# Per-field banks for realistic people. Written for this test; the train builder's banks and prompts are not read.

KITS: dict[str, dict] = {
    "advanced_practice": dict(
        label="Advanced practice clinician",
        titles=dict(entry=["Registered Nurse", "Physician Assistant"], mid=["Nurse Practitioner", "Physician Assistant"],
                    senior=["Lead Nurse Practitioner", "Hospitalist Physician"]),
        fields=["Nursing", "Physician Assistant Studies", "Medicine"], levels=dict(master=0.7, doctoral=0.3),
        skills=["Epic", "Patient assessment", "Differential diagnosis", "Prescribing", "Chronic disease management",
                "Procedural sedation", "Telehealth", "Care planning", "Patient education", "Clinical documentation"],
        licenses=["Board-certified Family Nurse Practitioner (FNP-BC)", "PA-C", "DEA registration", "BLS", "ACLS"],
        duties=["managed a panel of adult outpatients", "ordered and interpreted labs and imaging",
                "coordinated care with specialists", "precepted NP students", "ran walk-in urgent care visits"],
        employers=["a community hospital", "a multispecialty physician group", "an urgent care network",
                   "an academic medical center"], salary=(110, 130, 160)),
    "nursing": dict(
        label="Nursing",
        titles=dict(entry=["Registered Nurse", "Staff Nurse", "Licensed Nursing Assistant"],
                    mid=["Registered Nurse", "Charge Nurse", "Clinical Nurse"], senior=["Nurse Manager", "Clinical Nurse Educator"]),
        fields=["Nursing"], levels=dict(associate=0.35, bachelor=0.55, master=0.1),
        skills=["Epic", "Medication administration", "IV therapy", "Patient assessment", "Wound care", "Telemetry",
                "Patient education", "Care planning", "Phlebotomy", "Infection control"],
        licenses=["RN license", "BLS", "ACLS", "PALS"],
        duties=["cared for 5-6 patients per shift on a med-surg unit", "precepted new graduate nurses",
                "led discharge teaching", "coordinated with physicians on care plans", "managed admissions and transfers"],
        employers=["a regional hospital", "a skilled nursing facility", "an academic medical center", "a home health agency"],
        salary=(65, 80, 100)),
    "allied_health": dict(
        label="Allied health",
        titles=dict(entry=["Medical Assistant", "Patient Care Technician", "Phlebotomist", "Dialysis Technician"],
                    mid=["Certified Medical Assistant", "Lead Phlebotomist", "Senior Dialysis Technician", "Radiologic Technologist"],
                    senior=["Clinical Coordinator", "Lead Medical Assistant"]),
        fields=["Medical Assisting", "Radiologic Technology", "Health Sciences", "Diagnostic Medical Sonography"],
        levels=dict(none=0.45, associate=0.4, bachelor=0.15),
        skills=["Vital signs", "Venipuncture", "Specimen collection", "EKG", "Epic", "Patient intake", "Scheduling",
                "Infection control", "Dialysis machine setup", "Medical terminology", "Point-of-care testing", "HIPAA"],
        licenses=["CCMA", "Certified Phlebotomy Technician", "CCHT", "BLS", "ARRT (R)"],
        duties=["roomed 25-30 patients a day", "drew blood and processed specimens", "set up and monitored dialysis treatments",
                "scheduled follow-up visits", "kept exam rooms stocked and sterile"],
        employers=["a family medicine practice", "an outpatient dialysis clinic", "a hospital lab", "an orthopedic clinic"],
        salary=(36, 45, 55)),
    "social_care": dict(
        label="Social work and care coordination",
        titles=dict(entry=["Care Coordinator", "Case Management Assistant"], mid=["Social Worker", "Care Coordinator"],
                    senior=["Clinical Social Worker", "Care Management Supervisor"]),
        fields=["Social Work", "Psychology", "Public Health", "Sociology"], levels=dict(bachelor=0.5, master=0.5),
        skills=["Case management", "Motivational interviewing", "Care coordination", "Crisis intervention",
                "Psychosocial assessment", "Discharge planning", "Epic", "Community resources", "HIPAA"],
        licenses=["LCSW", "LSW", "LICSW"],
        duties=["managed a caseload of 60 patients", "ran psychosocial assessments", "connected families to community resources",
                "coordinated discharge plans", "followed up with high-risk patients by phone"],
        employers=["a community health center", "a hospital social work department", "a behavioral health clinic",
                   "a home dialysis program"], salary=(45, 60, 75)),
    "clinical_research": dict(
        label="Clinical research",
        titles=dict(entry=["Clinical Research Assistant", "Clinical Trial Assistant", "Clinical Research Coordinator"],
                    mid=["Clinical Research Coordinator", "Clinical Research Associate", "Clinical Data Coordinator"],
                    senior=["Senior Clinical Research Associate", "Clinical Operations Manager"]),
        fields=["Biology", "Health Sciences", "Public Health", "Nursing", "Psychology"], levels=dict(bachelor=0.75, master=0.25),
        skills=["Good Clinical Practice (GCP)", "ICH guidelines", "Informed consent", "Medidata Rave", "CTMS",
                "Source document verification", "IRB submissions", "Site monitoring", "EDC", "Veeva Vault", "Phlebotomy"],
        licenses=["CCRP", "CCRC", "RN license"],
        duties=["screened and enrolled study participants", "kept regulatory binders inspection-ready",
                "entered and cleaned visit data", "ran monitoring visits at 8 sites", "prepared IRB amendments"],
        employers=["an academic medical center", "a contract research organization", "a GI research network",
                   "a site management organization"], salary=(50, 75, 110)),
    "medical_affairs": dict(
        label="Medical affairs",
        titles=dict(entry=["Medical Information Specialist", "Associate Medical Writer"],
                    mid=["Medical Science Liaison", "Medical Writer", "Drug Safety Scientist"],
                    senior=["Senior Medical Science Liaison", "Director, Medical Writing", "Clinical Pharmacologist"]),
        fields=["Pharmacy", "Pharmacology", "Biology", "Immunology", "Pharmaceutical Sciences"],
        levels=dict(master=0.3, doctoral=0.7),
        skills=["Medical writing", "Literature review", "KOL engagement", "Clinical study reports", "Pharmacovigilance",
                "Scientific presentations", "Protocol development", "Argus Safety", "PK/PD analysis", "Phoenix WinNonlin"],
        licenses=["RPh license", "Board Certified Medical Affairs Specialist"],
        duties=["built relationships with key opinion leaders in immunology", "wrote clinical study reports and protocols",
                "reviewed adverse event cases", "presented data at advisory boards", "answered unsolicited medical questions"],
        employers=["a mid-size biopharma company", "a medical communications agency", "a global CRO", "a rare-disease biotech"],
        salary=(90, 150, 210)),
    "science": dict(
        label="Laboratory science",
        titles=dict(entry=["Research Associate", "Laboratory Technician", "QC Analyst"],
                    mid=["Scientist", "Senior Research Associate", "Associate Scientist"],
                    senior=["Senior Scientist", "Principal Scientist", "Associate Director, Research"]),
        fields=["Biology", "Biochemistry", "Chemistry", "Molecular Biology", "Bioengineering", "Pharmaceutical Sciences",
                "Chemical Engineering", "Materials Science", "Physics"],
        levels=dict(bachelor=0.45, master=0.2, doctoral=0.35),
        skills=["PCR", "qPCR", "ELISA", "Cell culture", "Flow cytometry", "HPLC", "LC-MS/MS", "Western blot", "GLP", "GMP",
                "Assay development", "CRISPR", "Aseptic technique", "GraphPad Prism", "Electronic lab notebook",
                "Mass spectrometry", "Mouse handling", "Microsoft Excel"],
        licenses=[],
        duties=["ran plate-based and cell-based assays for a preclinical program", "kept SOP-compliant lab records",
                "optimized a sample prep workflow", "trained two new associates", "presented results at weekly group meetings"],
        projects=["a thesis on protein aggregation", "a summer research project on antibody binding",
                  "a capstone on assay miniaturization"],
        employers=["a university research lab", "a biotech startup", "a bioanalytical CRO", "a pharmaceutical QC lab"],
        salary=(55, 95, 150)),
    "physical_science": dict(
        label="Physical and computational science",
        titles=dict(entry=["Research Assistant", "Research Associate"], mid=["Research Scientist", "Postdoctoral Researcher",
                                                                               "Computational Scientist"],
                    senior=["Senior Research Scientist", "Staff Scientist"]),
        fields=["Physics", "Materials Science", "Chemistry", "Chemical Engineering", "Applied Mathematics"],
        levels=dict(bachelor=0.15, master=0.2, doctoral=0.65),
        skills=["Python", "C++", "DFT", "Molecular dynamics", "Thin film deposition", "X-ray diffraction", "SEM",
                "Monte Carlo methods", "PyTorch", "High-performance computing", "LAMMPS", "VASP", "Sputtering", "MATLAB"],
        licenses=[],
        duties=["ran simulations on a 2,000-core cluster", "grew and characterized thin-film samples",
                "published two first-author papers", "built analysis scripts for beamline data", "mentored undergraduate researchers"],
        projects=["a thesis on phonon transport in oxides", "a code for automated phase-diagram mapping"],
        employers=["a national laboratory", "a university physics department", "a materials startup"],
        salary=(70, 130, 190)),
    "quality_reg": dict(
        label="Quality and regulatory",
        titles=dict(entry=["QA Associate", "Regulatory Affairs Associate", "Document Control Specialist"],
                    mid=["QA Specialist", "Regulatory Affairs Specialist", "Compliance Specialist"],
                    senior=["QA Manager", "Associate Director, Regulatory Affairs", "Director, Compliance"]),
        fields=["Chemistry", "Biology", "Regulatory Affairs", "Pharmaceutical Sciences", "Business Administration"],
        levels=dict(bachelor=0.65, master=0.3, doctoral=0.05),
        skills=["GMP", "CAPA", "Deviation management", "Change control", "Veeva Vault", "21 CFR Part 11", "Audits",
                "ISO 13485", "Regulatory submissions", "eCTD", "TrackWise", "Risk assessment"],
        licenses=["ASQ Certified Quality Auditor", "RAC"],
        duties=["owned deviations and CAPAs for a fill-finish line", "hosted supplier audits",
                "compiled regulatory submission modules", "reviewed batch records", "rolled out a document control system"],
        employers=["a biologics manufacturer", "a medical device company", "a generic drug maker", "a CDMO"],
        salary=(65, 100, 170)),
    "finance": dict(
        label="Finance and accounting",
        titles=dict(entry=["Staff Accountant", "Financial Analyst", "Accounts Payable Specialist"],
                    mid=["Senior Accountant", "Financial Analyst II", "Revenue Analyst"],
                    senior=["Accounting Manager", "Director, Financial Reporting", "Controller"]),
        fields=["Accounting", "Finance", "Economics", "Business Administration"], levels=dict(bachelor=0.75, master=0.25),
        skills=["GAAP", "NetSuite", "SAP", "Microsoft Excel", "Month-end close", "Account reconciliation",
                "Financial reporting", "Budgeting", "Oracle", "QuickBooks", "SEC reporting", "Variance analysis"],
        licenses=["CPA"],
        duties=["ran month-end close for three entities", "prepared variance analyses for leadership",
                "reconciled balance sheet accounts", "supported the annual audit", "built the budget model"],
        employers=["a public biotech company", "a regional accounting firm", "a medical device maker", "a SaaS company"],
        salary=(60, 90, 160)),
    "legal": dict(
        label="Legal",
        titles=dict(entry=["Associate Attorney"], mid=["Corporate Counsel", "Senior Associate"],
                    senior=["Senior Corporate Counsel", "Associate General Counsel"]),
        fields=["Law"], levels=dict(doctoral=1.0),
        skills=["Contract negotiation", "Commercial agreements", "Healthcare compliance", "Anti-Kickback Statute",
                "Advertising and promotion review", "Legal research", "Westlaw"],
        licenses=["Bar admission (Massachusetts)", "Bar admission (New York)"],
        duties=["negotiated distribution and services agreements", "reviewed promotional materials",
                "advised commercial teams on healthcare law", "managed outside counsel"],
        employers=["a large law firm", "a pharmaceutical company", "a medical device company"], salary=(150, 220, 300)),
    "data": dict(
        label="Data and analytics",
        titles=dict(entry=["Data Analyst", "Junior Data Scientist", "Reporting Analyst"],
                    mid=["Data Scientist", "Senior Data Analyst", "Data Engineer", "Statistical Programmer"],
                    senior=["Senior Data Scientist", "Analytics Manager", "Associate Director, Statistical Programming"]),
        fields=["Statistics", "Computer Science", "Mathematics", "Economics", "Data Science", "Biostatistics"],
        levels=dict(bachelor=0.5, master=0.4, doctoral=0.1),
        skills=["SQL", "Python", "R", "Tableau", "Power BI", "SAS", "Snowflake", "dbt", "Airflow", "Spark",
                "Machine learning", "Microsoft Excel", "CDISC", "Looker", "AWS", "pandas", "scikit-learn"],
        licenses=[],
        duties=["built dashboards used by the commercial team", "built data pipelines for claims data",
                "ran A/B tests and reported lift", "programmed analysis datasets for two Phase 2 studies", "cut a weekly report from 6 hours to 20 minutes"],
        projects=["a churn model on public telecom data", "a COVID hospitalization dashboard", "an NLP classifier for support tickets"],
        employers=["a health insurer", "a pharma analytics consultancy", "a hospital system", "an e-commerce startup", "a CRO"],
        salary=(70, 115, 170)),
    "software": dict(
        label="Software engineering",
        titles=dict(entry=["Software Engineer", "Junior Developer", "Software Developer"],
                    mid=["Software Engineer II", "Backend Engineer", "Full Stack Engineer", "Site Reliability Engineer"],
                    senior=["Senior Software Engineer", "Staff Engineer", "Engineering Manager"]),
        fields=["Computer Science", "Computer Engineering", "Information Systems", "Electrical Engineering"],
        levels=dict(bachelor=0.7, master=0.3),
        skills=["Python", "Java", "TypeScript", "React", "Node.js", "AWS", "Docker", "Kubernetes", "Terraform",
                "PostgreSQL", "Git", "CI/CD", "Linux", "Go", "REST APIs", "C++", "Salesforce Apex", "ServiceNow", "GraphQL"],
        licenses=["AWS Certified Solutions Architect"],
        duties=["shipped features for a patient-facing web app", "moved services onto a container platform", "owned on-call for the API tier",
                "cut p95 latency by a third", "mentored two junior engineers"],
        projects=["an open-source CLI for log search", "a hackathon scheduling app", "a personal finance tracker"],
        employers=["a health-tech startup", "a fintech company", "an enterprise SaaS company", "a consulting firm"],
        salary=(90, 140, 200)),
    "it_support": dict(
        label="IT and security",
        titles=dict(entry=["IT Help Desk Technician", "Desktop Support Technician"],
                    mid=["Systems Administrator", "IT Support Specialist II", "Security Analyst"],
                    senior=["IT Manager", "Director of IT", "Information Security Manager"]),
        fields=["Information Technology", "Computer Science", "Information Systems"], levels=dict(none=0.2, associate=0.3, bachelor=0.5),
        skills=["Active Directory", "Microsoft 365", "Jamf", "Okta", "ServiceNow", "Windows", "macOS", "Networking",
                "Intune", "PowerShell", "Vulnerability management", "SIEM"],
        licenses=["CompTIA A+", "CompTIA Security+", "CISSP"],
        duties=["resolved 40 tickets a week", "imaged and deployed laptops", "ran onboarding and offboarding for 300 staff",
                "managed device policies for the laptop fleet", "ran a security awareness program"],
        employers=["a hospital IT department", "a managed service provider", "a university", "a biotech company"],
        salary=(50, 85, 160)),
    "sales": dict(
        label="Sales and business development",
        titles=dict(entry=["Sales Development Representative", "Inside Sales Representative"],
                    mid=["Account Executive", "Territory Sales Representative", "Key Account Manager"],
                    senior=["Regional Sales Director", "Director, Business Development", "VP, Sales"]),
        fields=["Business Administration", "Marketing", "Communications", "Biology"], levels=dict(bachelor=0.85, master=0.15),
        skills=["Salesforce", "Territory management", "Pipeline management", "Negotiation", "Forecasting", "HubSpot",
                "Key account management", "Veeva CRM", "Contract negotiation", "Market access"],
        licenses=[],
        duties=["grew territory revenue 22% year over year", "managed 40 accounts across two states",
                "closed partnership deals with pharma clients", "built pipeline through conferences and cold outreach",
                "trained new reps on the product line"],
        employers=["a diagnostics company", "a medical device company", "a CRO", "a specialty pharma company", "a SaaS vendor"],
        salary=(55, 100, 190)),
    "marketing": dict(
        label="Marketing and communications",
        titles=dict(entry=["Marketing Coordinator", "Graphic Designer", "Events Coordinator"],
                    mid=["Marketing Manager", "Product Designer", "Communications Specialist"],
                    senior=["Senior Marketing Manager", "Associate Director, Marketing", "Lead UX Designer"]),
        fields=["Marketing", "Communications", "Graphic Design", "Business Administration"], levels=dict(bachelor=0.85, master=0.15),
        skills=["HubSpot", "Google Analytics", "Adobe Creative Cloud", "Figma", "SEO", "Content marketing", "Event planning",
                "Marketo", "Copywriting", "User research"],
        licenses=[],
        duties=["ran a 6-city conference program", "launched an email campaign aimed at physicians", "redesigned the onboarding flow",
                "managed a $400K agency budget", "wrote product launch copy"],
        employers=["a pharma marketing agency", "a consumer health brand", "a SaaS company", "a nonprofit"],
        salary=(50, 90, 150)),
    "engineering": dict(
        label="Engineering and manufacturing",
        titles=dict(entry=["Engineering Intern", "Manufacturing Technician", "Production Operator"],
                    mid=["Manufacturing Engineer", "Reliability Engineer", "CNC Programmer"],
                    senior=["Senior Process Engineer", "Engineering Manager"]),
        fields=["Mechanical Engineering", "Electrical Engineering", "Chemical Engineering", "Industrial Engineering"],
        levels=dict(none=0.25, associate=0.15, bachelor=0.55, master=0.05),
        skills=["SolidWorks", "AutoCAD", "MATLAB", "Lean manufacturing", "Six Sigma", "CNC programming", "Mastercam",
                "Root cause analysis", "PLC programming", "GD&T", "Clean room protocols"],
        licenses=["Six Sigma Green Belt", "OSHA 10"],
        duties=["ran preventive maintenance on packaging lines", "programmed machining jobs for aluminum parts",
                "cut scrap 15% with a Kaizen project", "followed gowning and clean room procedures", "designed fixtures for a new assembly line"],
        projects=["a senior design project on a heat exchanger", "a Formula SAE suspension design"],
        employers=["an aerospace parts supplier", "a medical device manufacturer", "a chemical plant", "a machine shop"],
        salary=(45, 85, 140)),
    "ops": dict(
        label="Operations and supply chain",
        titles=dict(entry=["Supply Chain Coordinator", "Operations Associate", "Procurement Assistant"],
                    mid=["Operations Manager", "Supply Chain Analyst", "Buyer"],
                    senior=["Director, Operations", "Head of Supply Chain", "Senior Director, Business Operations"]),
        fields=["Supply Chain Management", "Business Administration", "Industrial Engineering", "Economics"],
        levels=dict(bachelor=0.8, master=0.2),
        skills=["SAP", "Inventory management", "Procurement", "Vendor management", "Demand planning", "ERP", "Logistics",
                "Microsoft Excel", "Oracle", "S&OP"],
        licenses=["APICS CSCP"],
        duties=["managed 120 suppliers", "led S&OP meetings", "cut stockouts by 30%", "negotiated freight contracts",
                "built the operating plan with finance"],
        employers=["a pharmaceutical distributor", "a biotech startup", "a contract manufacturer", "a retailer"],
        salary=(50, 90, 180)),
    "hr_admin": dict(
        label="People and administration",
        titles=dict(entry=["Administrative Assistant", "HR Coordinator", "Program Coordinator", "Office Assistant"],
                    mid=["HR Generalist", "Executive Assistant", "Office Manager", "Program Administrator"],
                    senior=["HR Manager", "Chief of Staff", "Director, People Operations"]),
        fields=["Human Resources", "Psychology", "Business Administration", "Communications", "English"],
        levels=dict(none=0.1, associate=0.1, bachelor=0.7, master=0.1),
        skills=["Workday", "ADP", "Microsoft Office", "Calendar management", "Onboarding", "HRIS", "Concur",
                "Event coordination", "Employee relations", "Recruiting", "Benefits administration"],
        licenses=["SHRM-CP", "PHR"],
        duties=["managed calendars for three executives", "ran onboarding for 50 hires a quarter", "coordinated travel and expenses",
                "answered employee benefits questions", "planned the annual offsite"],
        employers=["a university department", "a biotech company", "a hospital administration office", "a staffing agency"],
        salary=(42, 65, 140)),
    "support_services": dict(
        label="Service and support",
        titles=dict(entry=["Customer Service Associate", "Patient Transporter", "Nutrition Services Aide", "Retail Associate"],
                    mid=["Shift Lead", "Customer Service Lead"], senior=["Store Manager"]),
        fields=[], levels=dict(none=0.85, associate=0.15),
        skills=["Customer service", "Cash handling", "POS systems", "Patient transport", "Food safety", "Inventory",
                "Scheduling", "Conflict resolution"],
        licenses=["ServSafe Food Handler", "BLS"],
        duties=["helped 100+ customers a shift", "moved patients safely between units", "prepared and delivered patient meals",
                "kept the floor stocked", "trained new hires"],
        employers=["a grocery store", "a hospital", "a coffee shop", "a thrift store", "a senior living community"],
        salary=(30, 38, 50)),
}

KIT_RULES = [
    (r"counsel|legal|attorney", "legal"),
    (r"sales representative|account executive", "sales"),
    (r"program coordinator|program administrator|\bhr\b|human resources", "hr_admin"),
    (r"generative models|biomolecule", "data"),
    (r"experimental engine|models of life|r&d consultant", "science"),
    (r"combination products|device development", "quality_reg"),
    (r"nurse practitioner|\bnp\b|aprn|physician assistant|\bphysicians?\b|\bapp\b|sub investigator|intensivist|"
     r"\bresidents?\b|\bresidency\b|fellowship", "advanced_practice"),
    (r"clinical research|\bcra\b|clinical trial|site identification|sample management|clinical data|research nurse|"
     r"clinical operations", "clinical_research"),
    (r"nurse|\brn\b|nursing|\blna\b|\blpn\b|\bcna\b", "nursing"),
    (r"medical assistant|phlebotom|dialysis|therapy assistant|\bpta\b|cota|sonograph|ultrasound|imaging tech|radiolog|"
     r"patient care associate|respiratory care|medical lab assistant", "allied_health"),
    (r"social worker|care coordinator|behavioral health|integrated care|absence case", "social_care"),
    (r"data scien|machine learning|\bml\b|\bai\b|data engineer|data architect|bioinformatic|statistical programm|biostatist|"
     r"elluminate|clinical programmer", "data"),
    (r"medical science liaison|\bmsl\b|medical writing|drug safety|pharmacovigil|pharmacology|clinical scientist|"
     r"clinical science|biomarker development", "medical_affairs"),
    (r"condensed matter|materials|thin films|physics|simulation|computational scientist|midtraining", "physical_science"),
    (r"scientist|research associate|research intern|\blab\b|laborator|assay|formulation|mass spec|in vivo|microbiolog|"
     r"biomarker|postdoc|thin films|drug product|oligonucleotide|analytical development|lims", "science"),
    (r"\bqa\b|quality|regulatory|compliance|\bgmp\b", "quality_reg"),
    (r"account(?:ant|ing)|financ|payment|pricing|contract analyst|investor relations|proposal analyst|\btax\b|payroll|"
     r"aid awarding", "finance"),
    (r"counsel|legal|attorney", "legal"),
    (r"engineering intern|reliability engineer|\bcnc\b|clean room|trade apprentice|mechanical|electrical|manufactur", "engineering"),
    (r"analytics|analyst|insights|reporting|report writer|\bdata\b", "data"),
    (r"software|developer|programmer|\bsre\b|devops|servicenow|salesforce engineer|engineering manager|technical consultant|"
     r"systems administrator|business systems|agentic|research engineer|forward deployed|embedded|engineer", "software"),
    (r"\bit\b|help ?desk|desktop support|technical support|technical services|cyber|ciso|security", "it_support"),
    (r"sales|account (?:executive|manager|partner)|business development|\bbd\b|key account|commercial|territory|"
     r"partnership|business director", "sales"),
    (r"marketing|brand|communication|grafik|design|\bux\b|events|congress", "marketing"),
    (r"supply chain|operations|procurement|sourcing|production|planning|logistics|operator|apprentice", "ops"),
    (r"\bhr\b|human resources|recruit|talent|administrative|assistant|office manager|coordinator|administrator|program|"
     r"training manager|chief of staff|office of the ceo|strategy|trainer", "hr_admin"),
    (r"transporter|thrift|customer service|nutrition|food|student observer", "support_services"),
]
FIELD_KIT = {"health_nursing": "allied_health", "lab_bio": "science", "data": "data", "software": "software",
             "sales_ops": "sales", "internship": "hr_admin", "other": "hr_admin"}
# a different field someone might still apply from (career changers, a mass applier)
OFF_KITS = {
    "advanced_practice": ["nursing", "medical_affairs", "allied_health"], "nursing": ["allied_health", "sales", "hr_admin"],
    "allied_health": ["support_services", "hr_admin", "nursing"], "social_care": ["hr_admin", "allied_health", "support_services"],
    "clinical_research": ["science", "sales", "hr_admin"], "medical_affairs": ["science", "sales", "clinical_research"],
    "science": ["sales", "quality_reg", "data", "support_services"],
    "physical_science": ["software", "data", "science"], "quality_reg": ["science", "ops", "hr_admin"],
    "finance": ["data", "hr_admin", "ops"], "legal": ["hr_admin", "finance", "quality_reg"],
    "data": ["finance", "science", "software"], "software": ["data", "it_support", "marketing"],
    "it_support": ["software", "support_services", "hr_admin"], "sales": ["marketing", "support_services", "science"],
    "marketing": ["sales", "hr_admin", "data"], "engineering": ["ops", "support_services", "software"],
    "ops": ["finance", "engineering", "hr_admin"], "hr_admin": ["support_services", "marketing", "finance"],
    "support_services": ["allied_health", "hr_admin", "ops"],
}

US_CITIES = ["Boston, MA", "Cambridge, MA", "Worcester, MA", "Waltham, MA", "Lowell, MA", "New York, NY", "Brooklyn, NY",
             "Albany, NY", "Philadelphia, PA", "Pittsburgh, PA", "Chicago, IL", "St. Louis, MO", "Kansas City, MO",
             "Dallas, TX", "Houston, TX", "Austin, TX", "Phoenix, AZ", "Denver, CO", "Seattle, WA", "San Francisco, CA",
             "San Jose, CA", "Los Angeles, CA", "San Diego, CA", "Atlanta, GA", "Raleigh, NC", "Minneapolis, MN",
             "Columbus, OH", "Detroit, MI", "Portland, ME", "Augusta, ME", "Manchester, NH", "Nashua, NH", "Washington, DC",
             "Baltimore, MD", "Miami, FL", "Tampa, FL", "Salt Lake City, UT", "Indianapolis, IN", "Madison, WI",
             "Providence, RI", "Hartford, CT", "New Haven, CT", "Nashville, TN", "Charlotte, NC", "Sacramento, CA"]
FOREIGN_CITIES = ["Toronto, ON", "Vancouver, BC", "London, UK", "Berlin, Germany", "Bengaluru, India"]
SCHOOLS = [("University of Massachusetts Amherst", "Amherst, MA"), ("Northeastern University", "Boston, MA"),
           ("Boston University", "Boston, MA"), ("Worcester Polytechnic Institute", "Worcester, MA"),
           ("University of New Hampshire", "Durham, NH"), ("University of Maine", "Orono, ME"),
           ("Rutgers University", "New Brunswick, NJ"), ("Temple University", "Philadelphia, PA"),
           ("Penn State University", "University Park, PA"), ("Ohio State University", "Columbus, OH"),
           ("University of Michigan", "Ann Arbor, MI"), ("Purdue University", "West Lafayette, IN"),
           ("University of Illinois Chicago", "Chicago, IL"), ("Saint Louis University", "St. Louis, MO"),
           ("University of Missouri-Kansas City", "Kansas City, MO"), ("University of Texas at Austin", "Austin, TX"),
           ("Texas A&M University", "College Station, TX"), ("Arizona State University", "Tempe, AZ"),
           ("University of Colorado Boulder", "Boulder, CO"), ("University of Washington", "Seattle, WA"),
           ("San Jose State University", "San Jose, CA"), ("UC San Diego", "La Jolla, CA"), ("UCLA", "Los Angeles, CA"),
           ("Georgia State University", "Atlanta, GA"), ("NC State University", "Raleigh, NC"),
           ("University of Minnesota", "Minneapolis, MN"), ("University of Florida", "Gainesville, FL"),
           ("Howard University", "Washington, DC"), ("University of Maryland", "College Park, MD"),
           ("University of Utah", "Salt Lake City, UT"), ("Lasell University", "Newton, MA"),
           ("Simmons University", "Boston, MA"), ("MCPHS University", "Boston, MA"), ("Brandeis University", "Waltham, MA"),
           ("Vanderbilt University", "Nashville, TN"), ("University of Connecticut", "Storrs, CT")]
COMMUNITY = [("Bunker Hill Community College", "Boston, MA"), ("Quinsigamond Community College", "Worcester, MA"),
             ("Manchester Community College", "Manchester, NH"), ("Kennebec Valley Community College", "Fairfield, ME"),
             ("St. Louis Community College", "St. Louis, MO"), ("Houston Community College", "Houston, TX"),
             ("Maricopa Community Colleges", "Phoenix, AZ"), ("Community College of Philadelphia", "Philadelphia, PA"),
             ("Austin Community College", "Austin, TX"), ("Portland Community College", "Portland, OR")]
FOREIGN_SCHOOLS = [("University of Mumbai", "Mumbai, India"), ("Peking University", "Beijing, China"),
                   ("University of Lagos", "Lagos, Nigeria"), ("Universidad de los Andes", "Bogota, Colombia"),
                   ("Seoul National University", "Seoul, South Korea"), ("Middle East Technical University", "Ankara, Turkey")]
FIRST = ["Maya", "Jordan", "Ananya", "Diego", "Grace", "Kwame", "Lena", "Mohammed", "Sara", "Tomas", "Yuki", "Chloe",
         "Ibrahim", "Elena", "Marcus", "Nadia", "Owen", "Rosa", "Samuel", "Tara", "Victor", "Wen", "Zainab", "Ethan",
         "Fatima", "Gabriel", "Hana", "Isaac", "Julia", "Kevin", "Leila", "Mateo", "Noah", "Olivia", "Ravi", "Sofia"]
LAST = ["Alvarez", "Brennan", "Chowdhury", "Dubois", "Eze", "Fischer", "Garcia", "Huang", "Iyer", "Jackson", "Kowalski",
        "Lopez", "Mensah", "Nakamura", "O'Brien", "Petrov", "Quinn", "Rahman", "Santos", "Thompson", "Uddin", "Vargas",
        "Williams", "Xu", "Yilmaz", "Zhang", "Carter", "Doyle", "Ferreira", "Gupta", "Hoang", "Kaur", "Moreau", "Novak"]
FORMATS = [
    "reverse-chronological: a two-line summary, then Experience, Education, Skills",
    "skills-first: a Skills section right under the header, then Experience and Education",
    "education-first: Education at the top (as a recent graduate or student would), then Experience, Projects, Skills",
    "no summary: straight into Professional Experience, then Education, then a one-line Skills list",
    "compact: section headings in capitals, skills as a comma-separated line, short bullets",
    "a headline line under the name, Core Competencies as a short list, then Experience and Education",
]
FILLER_JOBS = [
    {"title": "Barista", "employer": "a coffee shop", "focus": ["opened the store three mornings a week",
                                                                "trained new baristas"]},
    {"title": "Retail Associate", "employer": "a clothing store", "focus": ["handled the register and returns",
                                                                            "set up seasonal displays"]},
    {"title": "Resident Assistant", "employer": "the university's residence life office",
     "focus": ["supported 40 first-year students", "planned floor events"]},
    {"title": "Peer Tutor", "employer": "the campus tutoring center", "focus": ["tutored intro courses",
                                                                                 "ran weekly review sessions"]},
    {"title": "Server", "employer": "a family restaurant", "focus": ["served a busy dinner section",
                                                                     "trained new servers"]},
    {"title": "Front Desk Assistant", "employer": "the campus recreation center",
     "focus": ["checked in members", "answered phones and emails"]},
]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December"]
NOW = 2026 * 12 + 9  # October 2026, months since year 0

_ROLE_NOUN = re.compile(r"\b(?:engineer|scientist|manager|director|analyst|associate|assistant|specialist|coordinator|"
                        r"technician|tech|nurse|developer|lead|consultant|representative|executive|officer|intern|counsel|"
                        r"writer|designer|administrator|programmer|accountant|advisor|liaison|operator|trainee|apprentice|"
                        r"student|worker|sonographer|phlebotomist|practitioner|transporter|head|partner|vp|president|"
                        r"fellow|resident|controller|planner|owner|architect|investigator|leader|rep|ciso|generalist|"
                        r"contractor|programmer|np|aprn|physician|counsel|werkstudent|graduate|cra|msl|sme|aide)\b", re.I)


def kit_of(p: dict) -> str:
    t = p["title"].lower()
    for rx, kit in KIT_RULES:
        if re.search(rx, t):
            return kit
    return FIELD_KIT.get(p["field"], "hr_admin")


def level_of(title: str) -> str:
    t = title.lower()
    if re.search(r"director|\bhead\b|\bvp\b|vice president|chief|principal|ciso|\bsr\.? manager|senior manager|partner,", t):
        return "senior"
    if re.search(r"intern|trainee|apprentice|student|werkstudent|graduate|junior|\bi\b(?!i)|assistant|technician|aide|"
                 r"observer|transporter|associate(?! director)", t):
        return "entry"
    return "mid"


def is_student_role(p: dict, card: dict | None) -> bool:
    t = p["title"].lower()
    return bool(re.search(r"intern|co-op|student|werkstudent|working student|trainee|apprentice|graduate \(intern\)", t)) or \
        bool(card and (card["employment_type"] == "internship" or card["enrolled_in"] != "none"))


def clean_title(title: str) -> str | None:
    """The posting's title as a job title someone could have held, or None for a catch-all or a sentence."""
    t = re.split(r"\s[-–—|]\s|\(|:", title)[0].strip(" ,-")
    t = re.sub(r"\b(?:I{1,3}|IV|1|2|3)\b$", "", t).strip(" ,-")
    t = re.sub(r"\s*/\s*Sr\.?\s.*$|\s*/\s*Senior\s.*$", "", t)
    if not _ROLE_NOUN.search(t) or len(t.split()) > 8 or re.search(r"apply|pitch|see your role|good fit|dream", t, re.I):
        return None
    return t


# ---------------------------------------------------------------- skill mentions in the posting text

_SKILL_STOP = {"research", "programming", "debugging", "analysis", "design", "documentation", "coding", "presentation",
               "reporting", "modeling", "validation", "simulation", "optimization", "controls", "synthesis", "innovation",
               "agents", "autonomy", "publications", "visualization", "databases", "electronics", "AI", "ML", "CAN", "C", "R",
               "Go", "Word", "Claude", "Cursor", "Codex", "Node", "Shell", "REST", "Mentor", "Transformers", "Assembly"}
_PREF = re.compile(r"prefer|nice[- ]to[- ]have|bonus|\bplus\b|desired|ideal(?:ly)?|advantage|not required|a strong plus|"
                   r"what sets you apart|would be great|helpful", re.I)
_LISTCUE = re.compile(r"such as|e\.g\.|for example|including|one of|or similar|or equivalent|\blike\b|any of", re.I)
_REQ_HEAD = re.compile(r"requir|qualification|what you bring|what you(?:'|’)ll bring|you have|must have|minimum|basic|"
                       r"about you|who you are|skills|experience|education|competenc", re.I)
_DUTY_HEAD = re.compile(r"responsibilit|what you(?:'|’)ll do|duties|the role|day to day|you will|essential functions|"
                        r"key accountabilities|job description|summary", re.I)
_PREF_HEAD = re.compile(r"prefer|nice[- ]to[- ]have|bonus|desired|plus|additional|ideal", re.I)
_GENERIC = re.compile(r"skill|learn|solver|thinking|experience|environment|collaborat|communicat|centric|attitude|detail|"
                      r"motivat|abilit|\bteams?\b|self-|curiosity|passion|ownership|interpersonal|leadership|mindset|"
                      r"\bwork\b|written|verbal|fast-paced|independen|organiz|multitask|\bknowledge\b|understanding|"
                      r"background|degree|proficien|familiarity|\bdriven\b|initiative|fluency|english", re.I)
_VOCAB: list[str] | None = None


def skill_vocab() -> list[str]:
    """Tool and technique names a posting can ask for: the per-field banks here plus the names on the ATS-side v2 cards
    (training-side postings, never test ones) that read like proper tools or multiword techniques."""
    global _VOCAB
    if _VOCAB is not None:
        return _VOCAB
    from matcher.schema import is_soft_skill
    n = Counter()
    src = ROOT / "data/built/cards/extracted_real_cards_v2.jsonl"
    for r in read_jsonl(src):
        c = r["card"]
        for g in c["must_skills"]:
            n.update(g)
        n.update(c["nice_skills"])
    from matcher.onet import load_occupations
    from matcher.schema import SKILL_SYNONYMS
    try:
        onet = {x for o in load_occupations() for x in o.hot_tech}
    except (FileNotFoundError, OSError):
        onet = set()
    known = onet | set(SKILL_SYNONYMS) | {a for v in SKILL_SYNONYMS.values() for a in v} | \
        {x for kit in KITS.values() for x in kit["skills"]}
    majors = {f.casefold() for kit in KITS.values() for f in kit["fields"]} | {"software engineering", "data science"}
    names = set()
    for k, v in n.items():
        if v < 3 or len(k) > 40 or len(k.split()) > 4 or is_soft_skill(k) or k in _SKILL_STOP or _GENERIC.search(k) \
                or k.casefold() in majors:
            continue
        if len(k.split()) >= 2 or k in known or re.search(r"[A-Z0-9+#./-]", k[1:]):
            names.add(k)
    names = {x for x in names if x.casefold() not in {w.casefold() for w in _SKILL_STOP}}
    # longest first so 'Microsoft Excel' wins over 'Excel'
    _VOCAB = sorted(names, key=lambda x: (-len(x), x))
    return _VOCAB


def name_rx(name: str) -> re.Pattern:
    """Multiword names in any case; a single word in its own capitals ('Spark' the tool, not 'spark' the verb)."""
    flags = 0 if len(name.split()) == 1 else re.I
    return re.compile(rf"(?<![\w+#/]){re.escape(name)}(?![\w+#])", flags)


_RX_CACHE: dict[str, re.Pattern] = {}


def found(name: str, text: str) -> bool:
    rx = _RX_CACHE.get(name) or _RX_CACHE.setdefault(name, name_rx(name))
    return rx.search(text) is not None


def mentions(text: str) -> list[dict]:
    """Every vocabulary skill the posting names, once, with where it sits: required, preferred, list (an item of a
    'such as / e.g. / one of' list, with its place in the list and the list's id) or duty."""
    from matcher.schema import skill_canon
    out: list[dict] = []
    seen: set[str] = set()
    head = "duty"
    lists = 0
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if len(s) <= 80 and (s.endswith(":") or len(s.split()) <= 6) and not s.startswith(("-", "•", "*")):
            head = "preferred" if _PREF_HEAD.search(s) else "duty" if _DUTY_HEAD.search(s) else \
                "required" if _REQ_HEAD.search(s) else head
        for sent in re.split(r"(?<=[.;!?])\s+", s):
            covered: list[tuple[int, int]] = []
            hits = []
            for name in skill_vocab():
                rx = _RX_CACHE.get(name) or _RX_CACHE.setdefault(name, name_rx(name))
                for m in rx.finditer(sent):
                    if any(a <= m.start() < b for a, b in covered):
                        continue
                    covered.append((m.start(), m.end()))
                    hits.append((m.start(), name))
                    break
            if not hits:
                continue
            hits.sort()
            pref = bool(_PREF.search(sent)) or head == "preferred"
            listed = bool(_LISTCUE.search(sent)) and len(hits) >= 2 and not pref
            if listed:
                lists += 1
            for pos, (_, name) in enumerate(hits):
                k = skill_canon(name)
                if k in seen:
                    continue
                seen.add(k)
                where = "preferred" if pref else "list" if listed else ("required" if head == "required" else "duty")
                out.append({"name": name, "where": where, **({"list": lists, "pos": pos} if listed else {})})
    return out[:24]


_YEARS = re.compile(r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:or more\s+)?(?:years?|yrs?)", re.I)


def text_min_years(text: str) -> int | None:
    ys = [int(m.group(1)) for m in _YEARS.finditer(text)
          if re.search(r"experience|exp\b|background|working", text[m.end():m.end() + 60], re.I) and 0 < int(m.group(1)) <= 15]
    return min(ys) if ys else None


# ---------------------------------------------------------------- candidates


def _month_text(m: int) -> str:
    return f"{MONTHS[m % 12]} {m // 12}"


def _weighted(rng: random.Random, d: dict[str, float]) -> str:
    keys = list(d)
    return rng.choices(keys, [d[k] for k in keys])[0]


def held_skills_for(arch: str, ms: list[dict], kit: dict, rng: random.Random) -> tuple[list[str], list[str]]:
    """(skills held, posting mentions the candidate lacks) for one archetype."""
    req = [m["name"] for m in ms if m["where"] == "required"]
    duty = [m["name"] for m in ms if m["where"] == "duty"]
    pref = [m["name"] for m in ms if m["where"] == "preferred"]
    lists: dict[int, list[str]] = defaultdict(list)
    for m in ms:
        if m["where"] == "list":
            lists[m["list"]].append(m["name"])
    held: list[str] = []
    # duties describe the work, not what a candidate must hold: at most one duty mention, so the skills list is the
    # person's own and not the posting's keyword list (the rows the cycle-3 row read flagged)
    if arch == "strong":
        held += req + duty[:1] + rng.sample(pref, len(pref) // 2 + (len(pref) % 2 and rng.random() < 0.5))
        for items in lists.values():  # the alternative, not the first item
            held.append(rng.choice(items[1:]) if len(items) > 1 else items[0])
    elif arch == "near":
        held += req
        for items in lists.values():
            held.append(rng.choice(items))
    elif arch == "gap":
        drop = max(1, round(len(req) * 0.4)) if req else 0
        keep = rng.sample(req, len(req) - drop) if req else []
        held += keep + rng.sample(duty, min(1, len(duty)))
        if not req and lists:  # nothing listed as required: miss the lists instead
            pass
        elif lists and rng.random() < 0.5:
            items = rng.choice(list(lists.values()))
            held.append(rng.choice(items))
    # the field's own skills, none the posting names: a posting mention is held only where the archetype puts it
    # (a gap candidate's dropped requirement never comes back through the kit)
    named = {m["name"].casefold() for m in ms}
    own = [x for x in kit["skills"] if x not in held and x.casefold() not in named]
    held += rng.sample(own, min(len(own), rng.randint(4, 7) if arch != "off" else rng.randint(5, 8)))
    if arch == "off":
        posting_names = {m["name"].casefold() for m in ms}
        held = [x for x in held if x.casefold() not in posting_names] or held[:3]
    seen, out = set(), []
    for x in held:
        if x.casefold() not in seen:
            seen.add(x.casefold())
            out.append(x)
    out = out[:16]
    rng.shuffle(out)  # the person's order, never the posting's
    lacks = [m["name"] for m in ms if not any(m["name"].casefold() == x.casefold() for x in out)]
    return out, lacks


def education_for(level: str, field: str, grad: int | None, enrolled: bool, rng: random.Random, foreign: bool) -> list[dict]:
    """Education entries, highest first."""
    def school(kind: str):
        if foreign and kind != "community":
            return rng.choice(FOREIGN_SCHOOLS)
        return rng.choice(COMMUNITY if kind == "community" else SCHOOLS)
    name = {"associate": "Associate of Applied Science", "bachelor": rng.choice(["Bachelor of Science", "Bachelor of Arts"]),
            "master": "Master of Science", "doctoral": "Ph.D."}
    if level == "bachelor" and field in ("Nursing",):
        name["bachelor"] = "Bachelor of Science in Nursing"
    if level == "master" and field in ("Business Administration",):
        name["master"] = "MBA"
    if level == "master" and field == "Social Work":
        name["master"] = "Master of Social Work"
    if level == "master" and field == "Public Health":
        name["master"] = "Master of Public Health"
    if level == "doctoral" and field == "Law":
        name["doctoral"] = "Juris Doctor"
    if level == "doctoral" and field == "Pharmacy":
        name["doctoral"] = "Doctor of Pharmacy"
    if level == "doctoral" and field == "Medicine":
        name["doctoral"] = "Doctor of Medicine"
    if level == "master" and field == "Physician Assistant Studies":
        name["master"] = "Master of Physician Assistant Studies"
    if level == "master" and field == "Nursing":
        name["master"] = "Master of Science in Nursing"
    out = []
    if level == "none":
        hs_year = (grad or (2026 - rng.randint(2, 15)))
        return [{"degree": "High school diploma", "school": f"a public high school in {rng.choice(US_CITIES)}",
                 "date": str(hs_year), "status": "completed"}]
    s, city = school("community" if level == "associate" else "uni")
    month = rng.choice(["May", "May", "June", "December", "August"])
    out.append({"degree": name[level], **({"field": field} if field and field not in name[level] else {}),
                "school": s, "school_city": city, "date": f"{month} {grad}" if grad else "",
                "status": "in progress, expected" if enrolled else "completed"})
    if level in ("master", "doctoral") and grad:
        s2, c2 = school("uni")
        out.append({"degree": "Bachelor of Science", "field": field if field not in ("Law", "Medicine", "Pharmacy") else
                    rng.choice(["Biology", "Political Science", "Chemistry", "Economics"]),
                    "school": s2, "school_city": c2, "status": "completed",
                    "date": f"May {grad - (rng.randint(4, 6) if level == 'doctoral' else rng.randint(1, 3))}"})
    return out


def posting_titles(p: dict) -> set[str]:
    """The target posting's title as written and cleaned (clean_title), casefolded: no job a candidate holds may carry it."""
    ct = clean_title(p["title"])
    return {t.casefold().strip() for t in (p["title"], ct) if t}


def not_posting(titles: list[str], banned: set[str]) -> list[str]:
    return [t for t in titles if t.casefold().strip() not in banned and
            re.sub(r"\s*\(.*?\)|\s+Intern$", "", t).casefold().strip() not in banned]


def jobs_for(kit: dict, cur_title: str, years: int, student: bool, city: str, held: list[str],
             rng: random.Random, level: str, banned: set[str] = frozenset()) -> list[dict]:
    """Dated paid jobs, newest first, summing to about `years` (plus a few months). No title is in `banned` (the
    target posting's own title)."""
    months = years * 12 + rng.randint(0, 5)
    if student:
        jobs = []
        if rng.random() < 0.8:  # an internship or a research job in the field: it gets the field's duties
            t = rng.choice(not_posting(["Research Assistant (part-time)", "Summer Intern"], banned)) \
                if rng.random() < 0.4 else rng.choice(not_posting([x + " Intern" for x in kit["titles"]["entry"]], banned)
                                                      or not_posting(["Summer Intern", "Research Assistant (part-time)"],
                                                                     banned))
            end = NOW - rng.choice([1, 2, 13, 14])
            jobs.append({"title": t, "start": end - rng.randint(2, 4), "end": end, "field_job": True})
        if months >= 12:  # a job to pay for school
            fill = [j for j in FILLER_JOBS if not_posting([j["title"]], banned)]
            jobs.insert(0, {**rng.choice(fill), "start": NOW - months, "end": None, "field_job": False})
    else:
        n = 1 if months < 30 else 2 if months < 72 else 3 if months < 140 else 4
        cuts = sorted(rng.sample(range(8, max(9, months - 8)), n - 1)) if n > 1 else []
        bounds = [0, *cuts, months]
        jobs = []
        end = None
        for i in range(n):
            span = bounds[i + 1] - bounds[i]
            start = (NOW if end is None else end) - span
            # a career goes up: the job before the current one is a mid-level title, older ones entry-level
            older = kit["titles"]["mid"] if i == 1 and level != "entry" else kit["titles"]["entry"]
            older = not_posting(older, banned) or not_posting(kit["titles"]["mid"] + kit["titles"]["senior"], banned)
            title = cur_title if i == 0 else rng.choice([x for x in older if x != cur_title] or older)
            jobs.append({"title": title, "start": start, "end": end, "field_job": True})
            end = start - (rng.randint(1, 3) if rng.random() < 0.3 else 0)
    out = []
    pool = list(held)
    rng.shuffle(pool)
    duties = list(kit["duties"])
    rng.shuffle(duties)
    field_jobs = [j for j in jobs if j["field_job"]] or jobs
    for i, j in enumerate(jobs):
        dates = f"{_month_text(j['start'])} – {'Present' if j['end'] is None else _month_text(j['end'])}"
        city_j = city if i == 0 or rng.random() < 0.5 else rng.choice(US_CITIES)
        if not j["field_job"]:
            out.append({"title": j["title"], "employer": j["employer"], "city": city_j, "dates": dates,
                        "focus": j["focus"], "skills_used": []})
            continue
        k = max(1, len(pool) // len(field_jobs)) if pool else 0
        use, pool = pool[:k + 1], pool[k + 1:]
        out.append({"title": j["title"], "employer": rng.choice(kit["employers"]), "city": city_j, "dates": dates,
                    "focus": [duties[(2 * i + m) % len(duties)] for m in range(2)], "skills_used": use})
    target = next((j for j in out if j["employer"] not in {f["employer"] for f in FILLER_JOBS}), None)
    if pool and target:
        target["skills_used"] += pool
    return out


def profile_of(rng: random.Random, kit_name: str, kit: dict, level: str, student: bool, years: int) -> dict:
    """A CandidateCard dict (the profile form plus the resume facts the rules read)."""
    deg = _weighted(rng, kit["levels"])
    if student and deg in ("none",):
        deg = "associate" if kit_name in ("allied_health", "support_services", "it_support") else "bachelor"
    if student and deg == "doctoral" and rng.random() < 0.6:
        deg = "bachelor"
    field = rng.choice(kit["fields"]) if kit["fields"] and deg != "none" else ""
    if student:
        grad = 2026 + rng.choice([1, 1, 2, 2, 3]) if deg != "associate" else 2027
    elif deg == "none":
        grad = None
    else:
        grad = max(1990, 2026 - years - rng.choice([0, 0, 1, 2]))
    city = rng.choice(US_CITIES)
    auth = rng.choices(["citizen_or_pr", "needs_h1b_now", "other_visa"], [0.8, 0.12, 0.08])[0]
    grad_opt = None
    if student:
        auth = rng.choices(["citizen_or_pr", "f1_cpt"], [0.7, 0.3])[0]
    elif deg in ("bachelor", "master", "doctoral") and years <= 3 and rng.random() < 0.35:
        # an international graduate: OPT the year after graduating, STEM OPT one to three years after
        if years <= 1 or rng.random() < 0.5:
            auth, years = "f1_opt", min(years, 1)
            grad_opt = 2026 - rng.randint(0, 1)
        else:
            auth = "stem_opt"
            years = max(1, min(years, 3))
            grad_opt = 2026 - years
    if grad_opt is not None:
        grad = grad_opt
        if auth == "stem_opt" and kit_name not in ("science", "data", "software", "engineering", "it_support",
                                                   "quality_reg", "medical_affairs"):
            auth = "f1_opt" if grad >= 2025 else "needs_h1b_now"
    floor_k = kit["salary"][{"entry": 0, "mid": 1, "senior": 2}[level]]
    if student:  # an intern's floor is an hourly wage, annualized
        floor_k = min(45, round(floor_k * 0.6))
    c = {
        "field": kit["label"], "degree_level": deg, "degree_field": field, "graduation_year": grad, "years": years,
        "skills": [], "licenses": [], "current_location": city, "willing_locations": [],
        "remote_preference": rng.choices(["any", "hybrid_or_remote", "remote_only"], [0.7, 0.2, 0.1])[0],
        "work_auth": auth,
        "salary_floor": None if rng.random() < 0.4 else int(round(floor_k * rng.uniform(0.85, 1.15) / 5) * 5 * 1000),
        "employment_types_wanted": [] if rng.random() < 0.35 else (
            ["internship"] if student else rng.choice([["full_time"], ["full_time"], ["full_time", "contract"],
                                                       ["full_time", "part_time"]])),
        "industries_wanted": [], "industries_avoided": [], "roles_avoided": [],
        "education_country": "", "language": "",
    }
    if rng.random() < 0.35:
        c["willing_locations"] = rng.sample([x for x in US_CITIES if x != city], rng.randint(1, 3))
    if rng.random() < 0.05:
        c["current_location"] = ""
    return c


def consistent(c: dict) -> list[str]:
    from matcher.rules import license_problems, status_problems
    from matcher.schema import CandidateCard
    cc = CandidateCard.from_dict(c)
    return status_problems(cc) + license_problems(cc)


def fix_auth(c: dict, rng: random.Random) -> None:
    """Keep work authorization consistent with education (rules.status_problems): else a U.S. citizen."""
    if consistent({**c, "licenses": []}):
        c["work_auth"] = "citizen_or_pr"
    if consistent({**c, "licenses": []}) and c["degree_level"] == "none":
        c["graduation_year"] = None
        c["degree_field"] = ""


def verdict(c: dict, card: dict | None) -> dict | None:
    """rule_verdict for eligibility metadata only, never for labels."""
    if card is None:
        return None
    from hybrid_rules import HP_CATS, rule_verdict
    from matcher.schema import CandidateCard, RequirementCard
    v = rule_verdict(CandidateCard.from_dict(c), RequirementCard.from_dict(card))
    cats = sorted(set(v["hard_cats"]) & HP_CATS) if v["rule"] == "Skip" else []
    return {"pass": not cats and not v["salary_soft"], "hard_cats": cats, "salary_cap": v["salary_soft"]}


def place_for(card: dict) -> str | None:
    locs = [x for x in card.get("locations") or [] if x and not re.search(r"remote", x, re.I)]
    return locs[0] if locs else None


def steer_pass(c: dict, card: dict, rng: random.Random, student: bool) -> bool:
    """Change the profile until no H' filter on the extracted card fires; False if it cannot."""
    from matcher.degrees import card_fields, constrains, field_match
    from matcher.schema import RequirementCard
    rc = RequirementCard.from_dict(card)
    for _ in range(6):
        v = verdict(c, card)
        if v["pass"]:
            return True
        cats = set(v["hard_cats"])
        if "work_auth" in cats:
            c["work_auth"] = "citizen_or_pr"
        if "employment_type" in cats:
            c["employment_types_wanted"] = [] if rng.random() < 0.5 else sorted({card["employment_type"],
                                                                                  *c["employment_types_wanted"]})
        if "location" in cats and (loc := place_for(card)):
            if rng.random() < 0.6:
                c["current_location"] = loc
            else:
                c["willing_locations"] = [*c["willing_locations"][:2], loc]
        if "enrollment" in cats:
            need = card["enrolled_in"]
            c["degree_level"] = need if need != "none" else "bachelor"
            c["graduation_year"] = 2027 + rng.randint(0, 1)
            c["years"] = min(c["years"], 2)
        if "degree_level" in cats:
            order = ["none", "associate", "bachelor", "master", "doctoral"]
            if order.index(c["degree_level"]) < order.index(card["degree_level"]):
                c["degree_level"] = card["degree_level"]
                if c["graduation_year"] is None:
                    c["graduation_year"] = 2026 - max(0, c["years"] - rng.randint(0, 2))
        if "degree_field" in cats or (c["degree_level"] != "none" and not c["degree_field"]):
            fields = [f for f in card_fields(rc) if not re.search(r"related|equivalent|stem|similar|any", f, re.I)]
            ok = [f for f in fields if field_match(f, card_fields(rc)) == "exact"]
            if ok:
                c["degree_field"] = rng.choice(ok)
            elif not constrains(card_fields(rc)) and not c["degree_field"]:
                c["degree_field"] = "Biology"
        if v["salary_cap"]:
            c["salary_floor"] = None if rng.random() < 0.5 else int(card["salary_max"] * rng.uniform(0.7, 0.95) // 5000 * 5000)
        fix_auth(c, rng)
    return verdict(c, card)["pass"]


def steer_fail(c: dict, card: dict, rng: random.Random, student: bool) -> str | None:
    """Make exactly one H' filter fire (the candidate is otherwise as built). Returns the category, or None."""
    from matcher.degrees import card_fields, constrains
    from matcher.schema import RequirementCard
    rc = RequirementCard.from_dict(card)
    options = []
    if card["citizens_only"] or card["sponsorship"] in ("will_not_sponsor", "will_not_sponsor_or_opt"):
        options += ["work_auth"] * 3
    if card["work_mode"] != "remote" and place_for(card):
        options += ["location"] * 3
    options += ["employment_type"] * 2
    if card["degree_level"] not in ("none", "associate"):
        options += ["degree_level"] * 2
    if constrains(card_fields(rc)) and not card.get("related_ok"):
        options += ["degree_field"]
    if card["enrolled_in"] != "none":
        options += ["enrollment"] * 2
    if card.get("salary_max"):
        options += ["salary"] * 2
    rng.shuffle(options)
    for cat in dict.fromkeys(options):
        t = json.loads(json.dumps(c))
        if cat == "work_auth":
            if t["degree_level"] in ("none", "associate"):
                t["degree_level"], t["degree_field"] = "bachelor", t["degree_field"] or "Biology"
                t["graduation_year"] = t["graduation_year"] or 2026 - max(0, t["years"] - 1)
            enrolled = (t["graduation_year"] or 0) > 2026
            if card["citizens_only"] or card["sponsorship"] == "will_not_sponsor_or_opt":
                if enrolled:
                    t["work_auth"] = "f1_cpt"
                elif t["graduation_year"] in (2025, 2026):
                    t["work_auth"] = "f1_opt"
                else:
                    t["work_auth"] = "needs_h1b_now"
            else:
                if enrolled:
                    t["graduation_year"], t["years"] = 2025, max(t["years"], 1)
                t["work_auth"] = "needs_h1b_now"
        elif cat == "location":
            here = place_for(card) or ""
            far = [x for x in US_CITIES if x.split(",")[-1] != here.split(",")[-1]]
            t["current_location"] = rng.choice(far)
            t["willing_locations"] = rng.sample([x for x in far if x != t["current_location"]], rng.randint(0, 2))
        elif cat == "employment_type":
            et = card["employment_type"]
            t["employment_types_wanted"] = ["full_time"] if et == "internship" else \
                ["internship"] if student else rng.choice([["part_time"], ["contract"], ["full_time"]]) \
                if et != "full_time" else rng.choice([["part_time"], ["contract"], ["part_time", "contract"]])
        elif cat == "degree_level":
            order = ["none", "associate", "bachelor", "master", "doctoral"]
            lower = order[:order.index(card["degree_level"])]
            t["degree_level"] = rng.choice(lower[-2:])
            if t["degree_level"] == "none":
                t["degree_field"], t["graduation_year"] = "", None
        elif cat == "degree_field":
            t["degree_field"] = rng.choice(["Communications", "History", "Psychology", "Business Administration",
                                            "Graphic Design", "English", "Sociology"])
        elif cat == "enrollment":
            t["graduation_year"] = 2026 - rng.randint(0, 3)
            if t["work_auth"] == "f1_cpt":
                t["work_auth"] = "f1_opt" if t["graduation_year"] >= 2025 else "citizen_or_pr"
        elif cat == "salary":
            t["salary_floor"] = int(card["salary_max"] * rng.uniform(1.15, 1.5) // 5000 * 5000)
        fix_auth(t, rng)
        v = verdict(t, card)
        hit = cat if cat != "salary" else "salary"
        if not v["pass"] and (hit in v["hard_cats"] or (hit == "salary" and v["salary_cap"])):
            c.clear()
            c.update(t)
            return cat
    return None


def build_candidate(p: dict, card: dict | None, slot: int, arch: str, want_pass: bool, ms: list[dict]) -> dict:
    rng = random.Random(h(SEED, p["key"], slot))
    base_kit = kit_of(p)
    kit_name = rng.choice(OFF_KITS[base_kit]) if arch == "off" else base_kit
    kit = KITS[kit_name]
    plevel = level_of(p["title"])
    student = is_student_role(p, card) and arch != "off" and rng.random() < 0.8
    # years: the posting's stated minimum when it states one, else what its level suggests
    tmin = text_min_years(p["text"] or "")
    lo, hi = {"entry": (0, 3), "mid": (3, 8), "senior": (9, 18)}[plevel]
    if student:
        years = rng.choice([0, 0, 0, 1])
    elif arch == "strong":
        years = max(tmin or lo, lo) + rng.randint(0, 3)
    elif arch == "near":
        years = max(tmin or lo, lo) + rng.randint(0, 2)
    elif arch == "gap":
        years = max(0, (tmin or lo) - rng.randint(0, 2)) if rng.random() < 0.6 else max(tmin or lo, lo)
    else:
        years = rng.randint(*{"entry": (0, 5), "mid": (2, 10), "senior": (5, 16)}[plevel])
    c = profile_of(rng, kit_name, kit, "entry" if student else plevel, student, years)
    held, lacks = held_skills_for(arch, ms, kit, rng)
    lic_mentions = [m["name"] for m in ms if any(m["name"] == x for k in KITS.values() for x in k["licenses"])]
    lic = [x for x in lic_mentions if x in held] or ([] if arch == "off" else [x for x in kit["licenses"][:1]
                                                                              if rng.random() < 0.5])
    held = [x for x in held if x not in lic]
    if kit_name in ("nursing", "advanced_practice") and not student and arch != "off":
        need = {"nursing": "RN license", "advanced_practice": "PA-C" if "Physician Assistant" in str(kit["fields"][:0])
                else "Board-certified Family Nurse Practitioner (FNP-BC)"}[kit_name]
        lic = [need, *[x for x in lic if x != need]]
    c["skills"], c["licenses"] = held, lic
    fix_auth(c, rng)
    if consistent(c):  # a license the education could not lead to: drop it
        c["licenses"] = [x for x in c["licenses"] if not consistent({**c, "licenses": [x]})]
    elig, barrier = None, None
    if card is not None:
        if want_pass:
            steer_pass(c, card, rng, student)
        else:
            steer_pass(c, card, rng, student)  # one filter wrong, everything else right
            barrier = steer_fail(c, card, rng, student)
        elig = verdict(c, card)
    if consistent(c):
        c["licenses"] = [x for x in c["licenses"] if not consistent({**c, "licenses": [x]})]
    enrolled = c["graduation_year"] is not None and c["graduation_year"] > 2026
    # the current job is a title of the candidate's own field at their level, never the target posting's title
    # (cycle-3 row read: labelers leaned on "already holds the same title", which real resumes rarely show)
    banned = posting_titles(p)
    lvl = "entry" if c["years"] < 3 else "senior" if c["years"] >= 10 else "mid"
    cur = rng.choice(not_posting(kit["titles"][lvl], banned) or
                     not_posting(kit["titles"]["entry"] + kit["titles"]["mid"] + kit["titles"]["senior"], banned))
    jobs = jobs_for(kit, cur, c["years"], enrolled, c["current_location"] or rng.choice(US_CITIES), c["skills"], rng,
                    "entry" if c["years"] < 3 else plevel, banned)
    foreign = c["work_auth"] in ("f1_cpt", "f1_opt", "stem_opt", "needs_h1b_now") and rng.random() < 0.5
    edu = education_for(c["degree_level"], c["degree_field"], c["graduation_year"], enrolled, rng, foreign)
    projects = rng.sample(kit.get("projects", []), min(len(kit.get("projects", [])), rng.randint(1, 2))) \
        if (enrolled or c["years"] < 3 or rng.random() < 0.3) and kit.get("projects") else []
    contact = None
    if rng.random() < 0.6:
        f, l = rng.choice(FIRST), rng.choice(LAST)
        contact = {"name": f"{f} {l}", "city": c["current_location"] or "",
                   "phone": f"({rng.randint(201, 989)}) 555-01{rng.randint(10, 99)}",
                   "email": f"{f.lower()}.{l.lower().replace(chr(39), '')}@example.com"}
    rid = ROW_PREFIX + hashlib.sha1(f"{SEED}|{p['key']}|{slot}".encode()).hexdigest()[:10]
    return {"row_id": rid, "posting_id": p["key"], "slot": slot, "kit": kit_name, "posting_kit": base_kit,
            "archetype": arch, "target_pass": want_pass, "barrier": barrier, "eligibility": elig,
            "lacks": lacks, "mentions": ms, "candidate_card": c,
            "writer": {"header": contact, "format": rng.choice(FORMATS), "jobs": jobs, "education": edu,
                       "projects": projects, "enrolled": enrolled}}


def cmd_candidates(_a) -> None:
    posts = read_jsonl(POSTINGS)
    cards = {r["id"]: r["card"] for r in read_jsonl(CARDS)}
    out = []
    for p in posts:
        rng = random.Random(h(SEED, p["key"], "arch"))
        others = rng.choices(["near", "gap", "strong"], list(ARCH_W), k=PER_POSTING - 2)
        archs = ["strong", "off", *others]
        rng.shuffle(archs)
        passes = [rng.random() < PASS_RATE for _ in archs]
        if not any(passes):
            passes[rng.randrange(len(passes))] = True
        ms = mentions(p["text"] or "")
        for slot, (arch, want) in enumerate(zip(archs, passes)):
            out.append(build_candidate(p, cards.get(p["key"]), slot, arch, want, ms))
    old = {r["row_id"]: r for r in read_jsonl(CANDS)}
    write_jsonl(CANDS, out)
    # a candidate whose resume facts changed needs a new resume: its kept text and tries move to replaced_*.jsonl
    changed = {r["row_id"] for r in out if r["row_id"] in old and
               any(old[r["row_id"]][k] != r[k] for k in ("writer", "lacks")) or
               (r["row_id"] in old and old[r["row_id"]]["candidate_card"]["skills"] != r["candidate_card"]["skills"])}
    if changed:
        for path in (RESUMES, TRIES):
            rows = read_jsonl(path)
            gone = [x for x in rows if x["row_id"] in changed]
            write_jsonl(path, [x for x in rows if x["row_id"] not in changed])
            prev = read_jsonl(OUT / f"replaced_{path.name}")
            write_jsonl(OUT / f"replaced_{path.name}", prev + gone)
        print(f"{len(changed)} candidates changed: their resumes and tries moved aside for a new render")
    el = [r["eligibility"]["pass"] for r in out if r["eligibility"]]
    print(f"{len(out)} candidates for {len(posts)} postings; eligibility pass {sum(el)}/{len(el)} = {sum(el)/max(1,len(el)):.3f}; "
          f"no card {sum(r['eligibility'] is None for r in out)}")
    print("archetypes", dict(Counter(r["archetype"] for r in out)))
    print("barriers", dict(Counter(r["barrier"] for r in out)))
    print("fail cats", dict(Counter(c for r in out if r["eligibility"] for c in r["eligibility"]["hard_cats"])),
          "salary cap", sum(bool(r["eligibility"] and r["eligibility"]["salary_cap"]) for r in out))
    print("work_auth", dict(Counter(r["candidate_card"]["work_auth"] for r in out)))
    print("degree", dict(Counter(r["candidate_card"]["degree_level"] for r in out)))
    print("problems", sum(bool(consistent(r["candidate_card"])) for r in out))


# ---------------------------------------------------------------- resumes
# The writer prompt for this test only: distinct from render_cards.RESUME_* (what d3 trained on) and from the
# train builder's. The writer sees the candidate's card, never the posting.

MAX_TRIES = 3
WRITER_UNIVERSAL = (
    "Write the resume of the person described in the context, as plain text they could paste into an application form. "
    "Use the facts given and nothing that contradicts them."
)
WRITER_BLUEPRINT = (
    "Today is October 2026. Write a one-page resume of about 300 to 550 words in plain text: no markdown, no asterisks, "
    "no '#'. Bullets may start with '-' or '•'.\n"
    "Header: if 'header' is given, open with the name on its own line, then the city, phone and email on one line exactly "
    "as given. If 'header' is null, open with a one-line professional headline and give no name or contact details.\n"
    "Follow 'layout' for the section order and style.\n"
    "Experience: write every entry of 'jobs', in the given order, with its exact title and exact dates. Invent a realistic, "
    "fictional employer name that fits 'employer' and give the city as written. Turn 'focus' into two to four specific "
    "bullets with modest, believable detail, and work every item of 'skills_used' into that job's bullets. If 'jobs' is "
    "empty, write no work history at all.\n"
    "Education: write each entry with its exact degree, field, school, school city and date. When status says 'in progress, "
    "expected', write 'Expected' before the date; otherwise the date is a past graduation date and the word 'expected' never "
    "appears near it. Write no other degree.\n"
    "Projects: if 'projects' is given, add a Projects section with one or two bullets for each.\n"
    "Skills: every item of 'skills' appears in the resume word for word. Name no other software, programming language, "
    "instrument, lab technique, framework, platform or methodology anywhere.\n"
    "Licenses and certifications: list exactly the ones given, under their own heading; when none are given write none at "
    "all (no CPR, BLS, first aid or safety certificates).\n"
    "Experience length: if years_of_experience is above 0 the summary may state it as a whole number of years, never more.\n"
    "Never write about visas, citizenship, work authorization, sponsorship, salary, relocation or remote-work preferences. "
    "Write in English. Output only the resume.\n"
    "fix_from_last_try, when present: what your previous version got wrong. Write the whole resume again from the facts, "
    "fix every point, and never mention this key or a previous version."
)


def writer_kwargs() -> dict:
    return dict(
        column_mapping={"context": ["card"], "universal_prompt": WRITER_UNIVERSAL},
        recipe_specification={"recipes": {"prompt_rephrase": False, "deduplication": False, "reasoning_traces": False}},
        brand_controls={"blueprint": WRITER_BLUEPRINT, "length": "detailed", "hallucination_mitigation": False},
        training_type="instruction_dataset",
    )


def writer_context(r: dict, feedback: list[str] | None = None) -> dict:
    c, w = r["candidate_card"], r["writer"]
    ctx = {"header": w["header"], "layout": w["format"], "years_of_experience": c["years"],
           "jobs": [{k: v for k, v in j.items()} for j in w["jobs"]], "education": w["education"],
           "skills": c["skills"]}
    if w["projects"]:
        ctx["projects"] = w["projects"]
    if c["licenses"]:
        ctx["licenses_and_certifications"] = c["licenses"]
    if feedback:
        ctx["fix_from_last_try"] = feedback
    return ctx


_STOPWORDS = re.compile(r"\b(?:the|and|of|to|in|for|with|a|on|at|as|by|from|an|is|or)\b", re.I)
_LEAK = re.compile(r"\b(?:visas?|sponsor(?:ship|ed)?|H-?1B|green card|citizen(?:ship)?|work authori[sz]ation|salary|"
                   r"OPT|CPT|relocat\w*)\b", re.I)
_HIGHER = {"associate": r"\bBachelor|\bB\.?S\.?\b|\bB\.?A\.?\b|\bBSN\b|\bMaster(?:'s|\s+of|\b)|\bM\.?S\.?\b(?!\s*Office)|\bMBA\b|\bPh\.?D|\bDoctor",
           "bachelor": r"\bMaster(?:'s| of)|\bM\.S\.|\bMBA\b|\bMPH\b|\bMSN\b|\bPh\.?D|\bDoctor(?:ate| of)|\bJuris",
           "master": r"\bPh\.?D|\bDoctor(?:ate| of)|\bJuris|\bPharm\.?D|\bM\.D\.",
           "none": r"\bBachelor|\bB\.S\.|\bB\.A\.|\bAssociate of|\bAssociate's|\bMaster(?:'s| of)|\bMBA\b|\bPh\.?D"}
_LICENSE_WORDS = sorted({x for k in KITS.values() for x in k["licenses"]} |
                        {"CPR", "BLS", "ACLS", "PALS", "First Aid", "CNA", "LPN", "CPA", "PMP", "OSHA"}, key=len, reverse=True)


def gate(r: dict, text: str) -> list[str]:
    """Fatal problems of a rendered resume, each worded as feedback for the next try."""
    c, w = r["candidate_card"], r["writer"]
    t = text.strip()
    out: list[str] = []
    if len(t) < 600:
        return ["The resume was empty or far too short. Write the whole one-page resume."]
    if len(t) > 7000:
        out.append("The resume ran past one page. Keep it to about 300-550 words.")
    words = re.findall(r"[A-Za-z']+", t)
    nonascii = sum(ch.isalpha() and ord(ch) > 127 for ch in t) / max(1, len(t))
    if not words or len(_STOPWORDS.findall(t)) / len(words) < 0.06 or nonascii > 0.03:
        out.append("The resume was not in English. Write it in English.")
    if re.search(r"^\s*(?:None|N/A|None listed|Not applicable)\s*$", t, re.M | re.I):
        out.append("Your last version wrote a section with 'None' or 'N/A'. Leave out any section with nothing in it.")
    if re.search(r"\*\*|^\s*#|^\s*\*\s", t, re.M):
        out.append("The resume used markdown (asterisks or '#'). Use plain text only.")
    lvl = c["degree_level"]
    if lvl in _HIGHER and (m := re.search(_HIGHER[lvl], t)) and not any(m.group(0).lower() in e["degree"].lower()
                                                                      for e in w["education"]):
        line = next((l for l in t.split("\n") if m.group(0) in l), m.group(0)).strip()
        out.append(f'Your last version wrote "{line[:120]}". The person holds no such degree: write only the degrees in '
                   f'education.')
    for e in w["education"]:
        if e.get("date") and e["date"][-4:] not in t:
            out.append(f"The {e['degree']} date {e['date']} was missing. Write it exactly.")
        yr = e.get("date", "")[-4:]
        if not yr:
            continue
        near = [t[max(0, m.start() - 60):m.end() + 20] for m in re.finditer(yr, t)]
        exp = any(re.search(r"expected|anticipated|in progress|candidate", x, re.I) for x in near)
        if e["status"].startswith("in progress") and not exp:
            out.append(f"The {e['degree']} is still in progress: write 'Expected {e['date']}'.")
        if e["status"] == "completed" and exp and r["writer"]["enrolled"] is False:
            out.append(f"The {e['degree']} was finished in {e['date']}; never write 'expected' near it.")
    low = t.casefold()
    for j in w["jobs"]:
        title = re.sub(r"\s*\(.*?\)", "", j["title"]).strip().casefold()
        if title not in low:
            out.append(f"The job title '{j['title']}' ({j['dates']}) was missing. Write every job with its exact title.")
        start_year = re.search(r"\d{4}", j["dates"]).group(0)
        if start_year not in t:
            out.append(f"The dates '{j['dates']}' for '{j['title']}' were missing. Write them exactly.")
    literal = [j["employer"] for j in w["jobs"]
               if re.search(r"(?:^|[,|–—-]\s*)" + re.escape(j["employer"]) + r"\b", t, re.I | re.M)]
    if literal:
        out.append("Your last version used the descriptions " + "; ".join(f"'{x}'" for x in literal) + " as employer names. "
                   "Give each job a realistic, fictional employer name of that kind.")
    missing = [x for x in c["skills"] if not found(x, t) and x.casefold() not in low]
    if len(missing) > max(1, len(c["skills"]) * 0.15):
        out.append("These skills were missing; write each one word for word: " + "; ".join(missing))
    held = " | ".join([*c["skills"], *c["licenses"], *(j["title"] for j in w["jobs"]),
                       *(e["degree"] + " " + e.get("field", "") for e in w["education"])]).casefold()
    named = [x for x in r["lacks"] if found(x, t) and x.casefold() not in held]
    if named:
        out.append("Your last version named " + "; ".join(f"'{x}'" for x in named) + ", which this person does not have. "
                   "Name no skill, tool or credential that is not listed.")
    lic_held = " | ".join(c["licenses"]).casefold()
    extra_lic = [x for x in _LICENSE_WORDS if found(x, t) and x.casefold() not in lic_held and x.casefold() not in held]
    if extra_lic:
        out.append("Your last version listed " + "; ".join(f"'{x}'" for x in extra_lic[:4]) + ", a license or "
                   "certification this person does not hold. List only the ones given.")
    leaks = {m.group(0) for m in _LEAK.finditer(t)}
    if "Certified Phlebotomy Technician" in c["licenses"]:
        leaks.discard("CPT")
    if leaks:
        out.append("Your last version mentioned " + ", ".join(sorted(leaks)) + ". Never write about work authorization, "
                   "visas, salary or relocation.")
    return out


# ---------------------------------------------------------------- the Sonnet writer (set 5, amendment 6 FALLBACK)
# Claude Sonnet takes Adaption's place as the writer: the same instruction text (WRITER_UNIVERSAL, then
# WRITER_BLUEPRINT), the same per-candidate context (writer_context, fix_from_last_try on a retry), the same gates,
# MAX_TRIES, regate and repair rules. `render prepare` writes one batch of chunk files for writer agents (Agent tool,
# model sonnet, effort medium, one agent per chunk, started by the author with the chunk's agent_prompt text);
# `render absorb` reads the files they saved and gates them exactly as absorb() gates Adaption's completions. A row
# with no saved file is not a try: it goes out again with the next batch.

SONNET_CHUNK = 24
WRITER_PROMPT = "writer_prompt.txt"
WRITER_STAGE = ROOT / "data/writer_stage"  # the writers' folder: chunk files and their outputs, nothing else
WRITER_HARNESS = (
    "You write resumes for a test set. Read {prompt} first: it is your whole instruction for every resume. Then read "
    "{chunk}: one JSON object per line, each with a row_id and a context (the facts of one person). For each line, write "
    "that person's resume by following the instruction with that line's context as the context, and save it with the "
    "Write tool as {out}/<row_id>.txt, holding the resume text only. Write each resume yourself, one at a time, from its "
    "own context alone; don't write or run code to produce them. Write each once and move on: don't check it against a "
    "list of rules afterwards and don't rewrite it after saving. Don't open, search or list any other file. When every "
    "row has its file, reply with the number of files written."
)


def writer_instruction() -> str:
    """Adaption's two writer texts in the order Adaption is given them: the universal prompt, then the blueprint."""
    return WRITER_UNIVERSAL + "\n\n" + WRITER_BLUEPRINT


def sonnet_batches() -> list[Path]:
    return sorted(STATE.glob("sonnet_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[1])) if STATE.exists() else []


def stage_dir(name: str) -> Path:
    """Where one batch's writers read and write: outside the set's folder, so no candidate, posting or design file
    sits next to them."""
    return WRITER_STAGE / OUT.name / name


def deal(todo: list[str], posting_of: dict[str, str], size: int) -> list[list[str]]:
    """Split the shuffled rows into chunks of at most `size` so that no chunk holds two candidates of one posting:
    each row goes to the least-filled chunk without its posting (ties to the lower index)."""
    per = Counter(posting_of[i] for i in todo)
    n = max(-(-len(todo) // size), max(per.values(), default=1))
    while True:
        chunks: list[list[str]] = [[] for _ in range(n)]
        seen: list[set[str]] = [set() for _ in range(n)]
        ok = True
        for i in todo:
            free = [k for k in range(n) if len(chunks[k]) < size and posting_of[i] not in seen[k]]
            if not free:
                ok = False
                break
            k = min(free, key=lambda k: (len(chunks[k]), k))
            chunks[k].append(i)
            seen[k].add(posting_of[i])
        if ok:
            return [c for c in chunks if c]
        n += 1


def render_sonnet_prepare(a) -> None:
    """One batch of chunk files for the writer agents: every candidate without a kept resume and with tries left,
    shuffled with the seed and dealt (deal) so no chunk holds two candidates of one posting, SONNET_CHUNK at most."""
    open_ = [p for p in sonnet_batches() if not json.loads(p.read_text()).get("finished")]
    if open_:
        raise SystemExit(f"{open_[-1].stem} is not absorbed yet: run `render absorb` first")
    cands = {r["row_id"]: r for r in read_jsonl(CANDS)}
    regate()
    tries = read_jsonl(TRIES)
    kept = {r["row_id"] for r in read_jsonl(RESUMES)}
    by: dict[str, list[dict]] = defaultdict(list)
    for t in tries:
        by[t["row_id"]].append(t)
    todo = [i for i in cands if i not in kept and len(by[i]) < MAX_TRIES]
    if a.limit:
        todo = todo[::max(1, len(todo) // a.limit)][:a.limit]
    print(f"{len(cands)} candidates, {len(kept)} with a kept resume, {len(todo)} to render "
          f"({sum(bool(by[i]) for i in todo)} of them retries)")
    if not todo:
        repair_exhausted()
        return
    STATE.mkdir(parents=True, exist_ok=True)
    instr = writer_instruction()
    name = f"sonnet_{len(sonnet_batches())}"
    d = stage_dir(name)
    if d.exists():
        raise SystemExit(f"{d} exists")
    (d / "out").mkdir(parents=True)
    pp = d.parent / WRITER_PROMPT
    if pp.exists() and pp.read_text() != instr:
        raise SystemExit(f"{pp} differs from the writer instruction in this script")
    pp.write_text(instr)
    random.Random(h(SEED, name)).shuffle(todo)
    chunks = []
    for k, rows in enumerate(deal(todo, {i: cands[i]["posting_id"] for i in todo}, SONNET_CHUNK)):
        cp = d / f"chunk_{k + 1:02d}.jsonl"
        write_jsonl(cp, [{"row_id": i, "context": writer_context(cands[i], by[i][-1]["problems"] if by[i] else None)}
                         for i in rows])
        ap = cp.with_name(cp.stem + ".agent_prompt.txt")
        ap.write_text(WRITER_HARNESS.format(prompt=pp, chunk=cp, out=d / "out"))
        chunks.append(cp.name)
    (STATE / f"{name}.json").write_text(json.dumps({
        "n": len(todo), "retries": sum(bool(by[i]) for i in todo), "stage": os.path.relpath(d, ROOT),
        "chunks": chunks, "writer": "claude sonnet",
        "agent": {"tool": "Agent", "model": "sonnet", "effort": "medium"},
        "instruction_sha256": hashlib.sha256(instr.encode()).hexdigest(),
        "harness_sha256": hashlib.sha256(WRITER_HARNESS.encode()).hexdigest(), "created": time.time()}, indent=1))
    print(f"{name}: {len(todo)} candidates in {len(chunks)} chunks under {d}")


def render_sonnet_absorb(_a) -> None:
    """Gate the open batch's saved resumes (absorb(), unchanged) and close the batch."""
    open_ = [p for p in sonnet_batches() if not json.loads(p.read_text()).get("finished")]
    if not open_:
        raise SystemExit("no open Sonnet batch")
    st_path = open_[-1]
    name, state = st_path.stem, json.loads(st_path.read_text())
    d = ROOT / state["stage"]
    if any(t["batch"] == name for t in read_jsonl(TRIES)):
        raise SystemExit(f"{name} already has tries in {TRIES}: absorbed once and not marked finished; check by hand")
    sent = [r["row_id"] for c in state["chunks"] for r in read_jsonl(d / c)]
    got = {i: (d / "out" / f"{i}.txt").read_text().strip() for i in sent if (d / "out" / f"{i}.txt").exists()}
    stray = sorted({p.stem for p in (d / "out").glob("*.txt")} - set(sent))
    absorb(name, got)
    missing = [i for i in sent if i not in got]
    state.update(finished=time.time(), returned=len(got), missing=missing, stray_files=stray)
    st_path.write_text(json.dumps(state, indent=1))
    print(f"{name}: {len(got)} of {len(sent)} saved; {len(missing)} missing (sent again next batch, no try counted)"
          + (f"; {len(stray)} stray files ignored" if stray else ""))


def cmd_render(a) -> None:
    if WRITER == "sonnet":
        if a.mode not in ("prepare", "absorb"):
            raise SystemExit(f"set {SET} is written by Claude Sonnet: use `render prepare` or `render absorb`")
        return render_sonnet_prepare(a) if a.mode == "prepare" else render_sonnet_absorb(a)
    if a.mode not in ("estimate", "run"):
        raise SystemExit(f"set {SET} is written by Adaption: use `render estimate` or `render run`")
    cands = {r["row_id"]: r for r in read_jsonl(CANDS)}
    regate()
    tries = read_jsonl(TRIES)
    kept = {r["row_id"] for r in read_jsonl(RESUMES)}
    by: dict[str, list[dict]] = defaultdict(list)
    for t in tries:
        by[t["row_id"]].append(t)
    todo = [i for i in cands if i not in kept and len(by[i]) < MAX_TRIES]
    if a.limit:  # a pilot spread over the postings
        todo = todo[::max(1, len(todo) // a.limit)][:a.limit]
    print(f"{len(cands)} candidates, {len(kept)} with a kept resume, {len(todo)} to render "
          f"({sum(bool(by[i]) for i in todo)} of them retries)")
    if not todo:
        repair_exhausted()
        return
    c = ext.client()
    name = next_batch("resumes")
    rows = [[i, json.dumps(writer_context(cands[i], by[i][-1]["problems"] if by[i] else None), ensure_ascii=False)]
            for i in todo]
    did = upload_csv(c, name, ["row_id", "card"], rows)
    got = run_batch(c, name, did, writer_kwargs(), a.mode, a.max_credits, "row_id")
    if got is None:
        return
    absorb(name, got)


_EMPTY_LINE = re.compile(r"^\s*(?:None|N/A|None listed|Not applicable)\s*$", re.I)


def drop_empty_sections(text: str) -> str:
    """Remove a 'None' / 'N/A' line and the heading right above it (a section with nothing in it). Mechanical: no
    fact changes."""
    lines = text.split("\n")
    out: list[str] = []
    for ln in lines:
        if _EMPTY_LINE.match(ln):
            while out and not out[-1].strip():
                out.pop()
            if out and len(out[-1].split()) <= 5 and not out[-1].strip().startswith(("-", "•")):
                out.pop()
            continue
        out.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def repair_exhausted() -> None:
    """A candidate whose last allowed try failed only on an empty 'None' section keeps that try with the section
    removed (resume row marked repaired). Every other exhausted candidate has no resume and leaves the test."""
    cands = {r["row_id"]: r for r in read_jsonl(CANDS)}
    tries = read_jsonl(TRIES)
    resumes = read_jsonl(RESUMES)
    kept = {r["row_id"] for r in resumes}
    last: dict[str, dict] = {}
    for t in tries:
        last[t["row_id"]] = t
    n = 0
    for i, t in last.items():
        if i in kept or t["try"] < MAX_TRIES or not t["problems"]:
            continue
        fixed = drop_empty_sections(t["text"])
        if fixed != t["text"].strip() and not gate(cands[i], fixed):
            resumes.append({"row_id": i, "text": fixed, "tries": t["try"], "batch": t["batch"], "repaired": "empty section"})
            n += 1
    if n:
        write_jsonl(RESUMES, resumes)
    print(f"repaired {n} exhausted candidates (empty section removed); "
          f"{sum(1 for i in cands if i not in kept) - n} candidates have no resume")


def regate() -> None:
    """Kept resumes that fail a gate added after they were rendered become failed tries (counted toward MAX_TRIES)."""
    cands = {r["row_id"]: r for r in read_jsonl(CANDS)}
    tries = read_jsonl(TRIES)
    keep, moved = [], 0
    for r in read_jsonl(RESUMES):
        probs = gate(cands[r["row_id"]], r["text"])
        if probs:
            moved += 1
            for t in tries:
                if t["row_id"] == r["row_id"] and t["batch"] == r["batch"]:
                    t["problems"] = probs
        else:
            keep.append(r)
    if moved:
        write_jsonl(TRIES, tries)
        write_jsonl(RESUMES, keep)
        print(f"re-gated: {moved} kept resumes now fail a gate and go back for a retry")


def absorb(name: str, got: dict[str, str]) -> None:
    cands = {r["row_id"]: r for r in read_jsonl(CANDS)}
    sent = json.loads((STATE / f"{name}.json").read_text()).get("n")
    tries = read_jsonl(TRIES)
    resumes = read_jsonl(RESUMES)
    n_try = Counter(t["row_id"] for t in tries)
    ok = bad = 0
    for i, text in got.items():
        if i not in cands:
            continue
        probs = gate(cands[i], text)
        tries.append({"row_id": i, "batch": name, "try": n_try[i] + 1, "problems": probs, "text": text})
        if probs:
            bad += 1
        else:
            ok += 1
            resumes.append({"row_id": i, "text": text.strip(), "tries": n_try[i] + 1, "batch": name})
    write_jsonl(TRIES, tries)
    write_jsonl(RESUMES, resumes)
    print(f"{name}: {len(got)} returned of {sent} sent; {ok} pass the gates, {bad} fail")
    print("problems", Counter(p.split(".")[0][:60] for t in tries if t["batch"] == name for p in t["problems"]).most_common(12))


# ---------------------------------------------------------------- assemble and seal


def profile_form(c: dict) -> str:
    from matcher.profile import render_profile
    from matcher.schema import CandidateCard
    return render_profile(CandidateCard.from_dict(c))


def blind(r: dict) -> dict:
    """What a labeler sees: nothing but the row id, the posting, the resume and the profile form."""
    return {"row_id": r["row_id"], "posting": r["posting_text"], "resume": r["resume_text"], "profile": r["profile_text"]}


def cmd_assemble(_a) -> None:
    posts = {p["key"]: p for p in read_jsonl(POSTINGS)}
    cards = {r["id"]: r for r in read_jsonl(CARDS)}
    resumes = {r["row_id"]: r for r in read_jsonl(RESUMES)}
    e1 = fresh1_employers() if SET == "2" else set()
    df, dfa = dev_fit_postings() if SET == "3" else (set(), set())
    e123 = fresh123_employers() if SET in ("4", "4b") else set()
    rows = []
    dropped = []
    for cnd in read_jsonl(CANDS):
        res = resumes.get(cnd["row_id"])
        if res is None:
            dropped.append(cnd["row_id"])
            continue
        p = posts[cnd["posting_id"]]
        el = cnd["eligibility"]
        rows.append({
            "row_id": cnd["row_id"], "posting_id": p["key"], "employer": p["employer"], "field": p["field"],
            "title": p["title"], "posting_text": posting_text(p), "resume_text": res["text"],
            "profile": cnd["candidate_card"], "profile_text": profile_form(cnd["candidate_card"]),
            "design": {
                "eligibility_pass": el["pass"] if el else None, "eligibility_fails": el["hard_cats"] if el else None,
                "salary_cap": el["salary_cap"] if el else None, "teaser": p["chars"] < 700, "empty_body": p["chars"] == 0,
                "posting_length": p["length"], "posting_chars": p["chars"], "source": p["source"], "country": p["country"],
                "archetype": cnd["archetype"], "candidate_kit": cnd["kit"], "posting_kit": cnd["posting_kit"],
                "target_pass": cnd["target_pass"], "barrier": cnd["barrier"], "posting_mentions": cnd["mentions"],
                "lacks_mentions": cnd["lacks"], "render_tries": res["tries"], "resume_repaired": res.get("repaired"),
                "has_card": p["key"] in cards,
                "card_country": cards[p["key"]]["meta"]["country"] if p["key"] in cards else None,
                **({"employer_in_fresh_test1": p["emp_key"] in e1} if SET == "2" else {}),
                **({"posting_in_dev_fit": p["key"] in df, "posting_in_dev_fit_all": p["key"] in dfa}
                   if SET == "3" else {}),
                **({"employer_in_fresh_test1_3": p["emp_key"] in e123, "reserved_side": p["reserved_side"]}
                   if SET in ("4", "4b") else {}),
                **({"extension": "4b"} if SET == "4b" else {}),
                **({"source_set": p["source_set"], "employer_key": p["emp_key"], "category": p["category"]}
                   if SET == "5" else {})},
            "resume_facts": cnd["writer"]})
    write_jsonl(ROWS, rows)
    rng = random.Random(SEED + 1)
    order = list(rows)
    rng.shuffle(order)
    BATCHES.mkdir(parents=True, exist_ok=True)
    for old in BATCHES.glob("batch_*.jsonl"):
        old.unlink()
    for k in range(0, len(order), BATCH_SIZE):
        write_jsonl(BATCHES / f"batch_{k // BATCH_SIZE + 1:02d}.jsonl", [blind(r) for r in order[k:k + BATCH_SIZE]])
    # the blind human-read sheet: 100 rows stratified by field x eligibility pass, at most one row per posting while the cell allows
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in order:
        cells[(r["field"], bool(r["design"]["eligibility_pass"]))].append(r)
    keys = sorted(cells)
    quota = {k: max(1, round(SHEET_N * len(cells[k]) / len(rows))) for k in keys}
    while sum(quota.values()) > SHEET_N:
        quota[max(keys, key=lambda k: quota[k])] -= 1
    while sum(quota.values()) < SHEET_N:
        quota[max(keys, key=lambda k: len(cells[k]) - quota[k])] += 1
    sheet = []
    for k in keys:
        used: set[str] = set()
        first = [r for r in cells[k] if not (r["posting_id"] in used or used.add(r["posting_id"]))]
        rest = [r for r in cells[k] if r not in first]
        sheet += (first + rest)[:quota[k]]
    rng.shuffle(sheet)
    write_jsonl(SHEET, [blind(r) for r in sheet])
    el = [bool(r["design"]["eligibility_pass"]) for r in rows]
    print(f"{len(rows)} rows ({len(dropped)} candidates without a resume dropped) for "
          f"{len({r['posting_id'] for r in rows})} postings; eligibility pass {sum(el) / len(el):.3f}; "
          f"teaser rows {sum(r['design']['teaser'] for r in rows)}; {len(list(BATCHES.glob('batch_*.jsonl')))} batches; "
          f"sheet {len(sheet)}")
    (OUT / "assemble_dropped.json").write_text(json.dumps(dropped, indent=1))


def foreign(r: dict) -> bool:
    from matcher.metros import parse_place
    loc = r["profile"]["current_location"]
    return bool(loc) and not parse_place(loc).in_us


def cmd_seal(_a) -> None:
    rows = read_jsonl(ROWS)
    posts = {p["key"]: p for p in read_jsonl(POSTINGS)}
    cards = {r["id"]: r for r in read_jsonl(CARDS)}
    # the eligibility flag is reproducible from the stored profile and card, the way the eval will read it
    for r in rows:
        if r["posting_id"] not in cards:
            if r["design"]["has_card"] or r["design"]["eligibility_pass"] is not None:
                raise SystemExit(f"{r['row_id']}: design says it has a card, cards.jsonl has none")
            continue
        v = verdict(r["profile"], cards[r["posting_id"]]["card"])
        d = r["design"]
        if (v["pass"], v["hard_cats"], v["salary_cap"]) != (d["eligibility_pass"], d["eligibility_fails"], d["salary_cap"]):
            raise SystemExit(f"{r['row_id']}: stored eligibility does not reproduce from rows.jsonl and cards.jsonl")
    ids = sorted({r["posting_id"] for r in rows})
    batches = sorted(BATCHES.glob("batch_*.jsonl"))
    el = [bool(r["design"]["eligibility_pass"]) for r in rows]
    by_field = defaultdict(list)
    for r in rows:
        by_field[r["field"]].append(bool(r["design"]["eligibility_pass"]))
    rel = lambda p: str(p.relative_to(ROOT))  # noqa: E731
    what = ("v4 sealed fresh test (scripts/build_fresh_test.py): 150 postings of the frozen test pool, candidates "
            "designed per posting, resumes written by Adaption from the candidates' cards. Not gold: no row carries a "
            "label; gold comes from blind labelers reading the label batches.")
    extra: dict = {}
    if SET == "2":
        what = (f"Sealed fresh test {SET} (scripts/build_fresh_test.py --set {SET}): every English posting of the frozen "
                "test pool that fresh test 1 did not use, built exactly as fresh test 1 (same v2 extractor imported "
                "unchanged, same resume writer prompt and gates, same candidate design, new seed, row ids "
                f"{ROW_PREFIX}...). Not gold: no row carries a label; gold comes from blind labelers reading the label "
                "batches.")
        e1 = fresh1_employers()
        shared = sorted({posts[i]["emp_key"] for i in ids} & e1)
        extra = {
            "sample": {"rule": "the complement: every data/built/fit/pools/test.jsonl key not in "
                               "eval/fresh_test_manifest.json posting_ids, minus the non-English ones (language()); no "
                               "employer, teaser, adzuna or country caps, nothing drawn",
                       "fresh_test1_manifest_sha256": sha256(FRESH1_MANIFEST)},
            "employers_vs_fresh_test1": {
                "what": "fresh test 2 is new at the posting level, not the employer level: the test pool's employers "
                        "were already in fresh test 1, which this cycle is open for error diagnosis",
                "employers": len({posts[i]["emp_key"] for i in ids}), "shared_with_fresh_test1": len(shared),
                "postings_from_shared_employers": sum(posts[i]["emp_key"] in e1 for i in ids),
                "rows_from_shared_employers": sum(bool(r["design"].get("employer_in_fresh_test1")) for r in rows),
                "new_employers": sorted({posts[i]["emp_key"] for i in ids} - e1)},
            "no_card_postings": sorted(i for i in ids if i not in cards)}
    elig_rule = ("design metadata only: hybrid_rules.rule_verdict on the profile and Adaption's extracted card; pass = "
                 "no hard Skip in H' categories (work_auth, enrollment, degree_level, degree_field, employment_type, "
                 "location) and no salary cap")
    if SET == "3":
        extra = set3_seal_extra(rows, posts, cards, ids)
        what = extra.pop("what")
        from hybrid_rules import HP_CATS
        elig_rule = ("design metadata only: hybrid_rules.rule_verdict on the profile and Adaption's extracted card, with "
                     "the rules at this commit; pass = no hard Skip in H' categories (" + ", ".join(sorted(HP_CATS)) +
                     ") and no salary cap. work_mode became an H' hard Skip in c6c4790, after the candidate design was "
                     "written: steer_pass does not steer it, so some rows designed to pass fail on work_mode alone "
                     "(target_pass_but_fail_work_mode_only)")
        extra["target_pass_but_fail_work_mode_only"] = sum(
            bool(r["design"]["target_pass"]) and r["design"]["eligibility_fails"] == ["work_mode"] for r in rows)
    if SET == "4":
        extra = set4_seal_extra(rows, posts, cards, ids)
        what = extra.pop("what")
        from hybrid_rules import HP_CATS
        elig_rule = ("design metadata only: hybrid_rules.rule_verdict on the profile and Adaption's extracted card, with "
                     "the rules at this commit; pass = no hard Skip in H' categories (" + ", ".join(sorted(HP_CATS)) +
                     ") and no salary cap (amendment 5 keeps this slice definition)")
    if SET == "5":
        extra = set5_seal_extra(rows, posts, cards, ids)
        what = extra.pop("what")
        from hybrid_rules import HP_CATS
        elig_rule = ("design metadata only: hybrid_rules.rule_verdict on the profile and the posting's v2 card (without "
                     "rules.mark_thin, as every earlier test's flag), with the rules at this commit; pass = no hard Skip "
                     "in H' categories (" + ", ".join(sorted(HP_CATS)) + ") and no salary cap. steer_pass does not steer "
                     "work_mode, so some rows designed to pass fail on work_mode alone "
                     "(target_pass_but_fail_work_mode_only), as in tests 3 and 4")
    if POOL == "round2":
        kept = json.loads((OUT / "sample_walk.json").read_text())
        pool_doc = {"file": "data/raw/ats/postings.jsonl (the postings behind data/built/real_test/test.jsonl and "
                            "data/built/real_train/{train,dev}.jsonl), filtered by load_round2",
                    "kept": kept["pool_kept"], "walk_length": len(kept["walk"]),
                    "sha256 sorted walk ids": hashlib.sha256("\n".join(sorted(kept["walk"])).encode()).hexdigest()}
    elif POOL == "reserved":
        res = json.loads((ROOT / "data/built/fit/split.json").read_text())[RESERVED_KEY]
        pool_doc = {"file": "data/built/fit/split.json (reserved test and dev postings)",
                    "sha256 sorted ids": hashlib.sha256("\n".join(sorted(k for v in res.values() for k in v)).encode()).hexdigest(),
                    "fit_freeze_reserved_counts": json.loads((ROOT / "eval/fit_freeze.json").read_text())["reserved"]}
    else:
        pool_doc = {"file": f"data/built/fit/pools/{POOL}.jsonl",
                    "sha256 sorted ids": json.loads((ROOT / "eval/fit_freeze.json").read_text())["pools"][POOL]["sha256 sorted ids"]}
    man = {
        "what": what,
        "seed": SEED,
        "row_id_prefix": ROW_PREFIX,
        "credit_cap": CREDIT_CAP,
        **extra,
        "pool": pool_doc,
        "n_rows": len(rows), "n_postings": len(ids),
        "dropped_candidates_without_resume": json.loads((OUT / "assemble_dropped.json").read_text()),
        "sample_excluded": json.loads((OUT / "sample_excluded.json").read_text()),
        "posting_ids": ids,
        "employers": dict(Counter(posts[i]["employer"] for i in ids).most_common()),
        "field_mix": {"postings": dict(Counter(posts[i]["field"] for i in ids)), "rows": dict(Counter(r["field"] for r in rows))},
        "length_mix": dict(Counter(posts[i]["length"] for i in ids)),
        "source_mix": dict(Counter(posts[i]["source"] for i in ids)),
        "country_mix": {"by extracted card meta": dict(Counter(cards[i]["meta"]["country"] for i in ids if i in cards)),
                        "by pool locations before extraction": dict(Counter(posts[i]["country"] for i in ids))},
        "foreign_resident_rows": {
            "what": "rows whose profile lives outside the US (matcher.metros.parse_place not in_us): steer_pass put some "
                    "candidates in a foreign posting's city so it could pass the location filter",
            "rows": sum(foreign(r) for r in rows),
            "eligibility_pass_rows": sum(foreign(r) and bool(r["design"]["eligibility_pass"]) for r in rows)},
        "repaired_resumes": [r["row_id"] for r in rows if r["design"]["resume_repaired"]],
        "teaser_rows": sum(r["design"]["teaser"] for r in rows),
        "eligibility": {
            "rule": elig_rule,
            "pass_frac": round(sum(el) / len(el), 4),
            "by_field": {f: round(sum(v) / len(v), 3) for f, v in sorted(by_field.items())},
            "fails": dict(Counter(c for r in rows for c in (r["design"]["eligibility_fails"] or []))),
            "salary_cap_rows": sum(bool(r["design"]["salary_cap"]) for r in rows)},
        "archetypes": dict(Counter(r["design"]["archetype"] for r in rows)),
        "work_auth": dict(Counter(r["profile"]["work_auth"] for r in rows)),
        "degree_level": dict(Counter(r["profile"]["degree_level"] for r in rows)),
        "extractor": {"script": "scripts/extract_postings.py (imported, unchanged)", "blueprint_hash": ext.BP_HASH},
        "writer": {"prompt_sha1": hashlib.sha1((WRITER_UNIVERSAL + WRITER_BLUEPRINT).encode()).hexdigest()[:12],
                   "max_tries": MAX_TRIES,
                   "tries": dict(Counter(r["design"]["render_tries"] for r in rows))},
        "credits": credits_log(),
        "files": {rel(p): sha256(p) for p in [ROWS, CARDS, POSTINGS, CANDS, RESUMES, SHEET, *batches]},
        "label_batches": [rel(p) for p in batches],
        "blind_sheet": rel(SHEET),
    }
    MANIFEST.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    if SET == "5":
        man["files"].update({rel(p): sha256(p) for p in [OUT / "sample_walk.json", TRIES]})
        MANIFEST.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {write_data_hashes()}")
    print(f"wrote {MANIFEST}: {len(rows)} rows, {len(ids)} postings, {len(batches)} batches, "
          f"credits {man['credits'].get('total_launched_estimate')}")


def cmd_extend(_a) -> None:
    """Amendment 4b: append this extension's assembled set to its base set (postings, cards, candidates, resumes, tries,
    rows; every base line left as it is) and write blind label batches for the new rows only, numbered after the base
    set's batches. Seal the base set afterwards (--set 4 seal)."""
    base = SETS[SET].get("extends")
    if not base:
        raise SystemExit(f"set {SET} extends nothing")
    bout = ROOT / SETS[base]["out"]
    new_rows = read_jsonl(ROWS)
    if any(r["design"].get("extension") != SET for r in new_rows):
        raise SystemExit("an extension row without its extension flag")
    have = {r["row_id"] for r in read_jsonl(bout / "rows.jsonl")}
    if have & {r["row_id"] for r in new_rows}:
        raise SystemExit(f"{len(have & {r['row_id'] for r in new_rows})} extension rows already in set {base}")
    for name in ("postings.jsonl", "cards.jsonl", "candidates.jsonl", "resumes.jsonl", "render_tries.jsonl", "rows.jsonl"):
        add = (OUT / name).read_text()
        with (bout / name).open("a") as f:
            f.write(add)
    bb = bout / "label_batches"
    start = len(list(bb.glob("batch_*.jsonl"))) + 1
    order = list(new_rows)
    random.Random(SEED + 1).shuffle(order)
    made = []
    for k in range(0, len(order), BATCH_SIZE):
        path = bb / f"batch_{start + k // BATCH_SIZE:02d}.jsonl"
        if path.exists():
            raise SystemExit(f"{path} exists")
        write_jsonl(path, [blind(r) for r in order[k:k + BATCH_SIZE]])
        made.append(path.name)
    print(f"appended {len(new_rows)} rows for {len({r['posting_id'] for r in new_rows})} postings to set {base}; "
          f"batches {made}")


def fresh123_employers() -> set[str]:
    from fit_pools import load_pool
    keys = fresh_test_keys(FRESH1_MANIFEST) | fresh_test_keys(FRESH2_MANIFEST) | fresh_test_keys(FRESH3_MANIFEST)
    return {p["emp_key"] for n in ("test", "dev") for p in load_pool(n) if p["key"] in keys}


def set4_seal_extra(rows: list[dict], posts: dict, cards: dict, ids: list[str]) -> dict:
    """Fresh test 4's manifest fields: where its postings come from, that none was used before, its employer overlap
    with fresh tests 1-3, the missing teasers, and disjointness from training (checked here, not assumed)."""
    from fit_pools import load_pool
    split = json.loads((ROOT / "data/built/fit/split.json").read_text())
    pool_ids = {k for v in split["pools"].values() for k in v["ids"]}
    reserved = {k for v in split[RESERVED_KEY].values() for k in v}
    k123 = {i: fresh_test_keys(m) for i, m in (("1", FRESH1_MANIFEST), ("2", FRESH2_MANIFEST), ("3", FRESH3_MANIFEST))}
    used_any = set().union(*k123.values())
    e123 = fresh123_employers()
    emps = {posts[i]["emp_key"] for i in ids}
    e_train = {p["emp_key"] for p in load_pool("train")}
    notes = json.loads((OUT / "reserved_notes.json").read_text())
    cyc3 = {r["posting_id"] for r in read_jsonl(ROOT / "data/built/fit/cycle3/rows.jsonl")}
    by_len = Counter(posts[i]["length"] for i in ids)
    kp = {"fresh_test1": (52, 592), "fresh_test2": (19, 371), "fresh_test3": (18, 437)}
    pooled = sum(a for a, _ in kp.values()) / sum(b for _, b in kp.values())
    late = (19 + 18) / (371 + 437)
    ext = [r for r in rows if r["design"].get("extension") == "4b"]
    ext_doc = {}
    if ext:
        e_ids = sorted({r["posting_id"] for r in ext})
        eo = ROOT / SETS["4b"]["out"]
        first_ids = set(ids) - set(e_ids)
        ext_doc = {"extension_4b": {
            "what": "eval/protocol_amendment_4b.json: about 200 more rows on about 50 new reserved postings, the same "
                    "unsteered design, decided from the first 940 rows' gold Keep count (34 on the eligibility-pass "
                    "slice) before any system was scored on fresh test 4; its rows carry design.extension = '4b' and "
                    "were appended to the files of the first 940 rows, which are unchanged",
            "seed": SETS["4b"]["seed"], "credit_cap": SETS["4b"]["cap"],
            "sample": "cmd_sample on load_reserved() minus the first 240 postings, which count toward the employer cap "
                      f"of {SETS['4b']['employer_cap']}; n_postings {SETS['4b']['n_postings']}, n_empty "
                      f"{SETS['4b']['n_empty']}, no teasers (the reserved set holds none)",
            "n_rows": len(ext), "n_postings": len(e_ids),
            "employers": len({posts[i]["emp_key"] for i in e_ids}),
            "employers_new_to_test4": sorted({posts[i]["emp_key"] for i in e_ids} -
                                             {posts[i]["emp_key"] for i in first_ids}),
            "employers_shared_with_fresh_tests_1_3": len({posts[i]["emp_key"] for i in e_ids} & e123),
            "postings_overlapping_first_240": sorted(set(e_ids) & first_ids),
            "field_mix_postings": dict(Counter(posts[i]["field"] for i in e_ids)),
            "field_mix_note": "the first 240 postings had used most of the software, internship and health employers' "
                              "room under the cap of 7, so the extension leans to data, lab, sales/ops and other",
            "length_mix": dict(Counter(posts[i]["length"] for i in e_ids)),
            "eligibility_pass_rows": sum(bool(r["design"]["eligibility_pass"]) for r in ext),
            "archetypes": dict(Counter(r["design"]["archetype"] for r in ext)),
            "dropped_candidates_without_resume": json.loads((eo / "assemble_dropped.json").read_text()),
            "label_batches": sorted(p.name for p in (OUT / "label_batches").glob("batch_*.jsonl")
                                    if int(p.stem.split("_")[1]) > 13),
            "credits": json.loads((eo / "credits.json").read_text())}}
    return {**ext_doc,
        "what": (f"Sealed fresh test 4 (scripts/build_fresh_test.py --set 4, eval/protocol_amendment_5.json): {len(ids)} "
                 "English postings the freeze reserved (other postings of the test and dev employers, in no pool and no "
                 "test), built as fresh tests 1-3 with the cycle-3 candidate generator (8c6bac5: no held job title is the "
                 "posting's title, the skills list is the candidate's own), seed "
                 f"{SEED}, row ids {ROW_PREFIX}... Not gold: no row carries a label; gold comes from blind labelers A and "
                 "B and the adjudicator reading the label batches."),
        "decided_by": "the author, 2026-10-21: build test 4 on the employer-sharing reading",
        "sample": {
            "rule": "cmd_sample (fresh test 1's sampler) on load_reserved(): the reserved postings minus near copies of "
                    "fresh test 1-3 postings and of train-pool postings and minus repeats of one employer and title, "
                    f"English only; N_POSTINGS={N_POSTINGS}, N_TEASER={N_TEASER}, N_EMPTY={N_EMPTY}, "
                    f"MAX_ADZUNA_TEASERS={MAX_ADZUNA_TEASERS}, NON_US_CAP={NON_US_CAP}, EMPLOYER_CAP={EMPLOYER_CAP} "
                    f"with no over-cap fallback, {PER_POSTING} candidates a posting",
            "reserved_filter": notes,
            "taken": len(ids),
            "sizing": {
                "rule": "about 4 rows a posting, sized so the expected gold Keeps on the eligibility-pass slice are at "
                        "least 50 at the pooled tests 1-3 rate",
                "pass_slice_keeps_per_row": {k: f"{a}/{b}" for k, (a, b) in kp.items()},
                "expected_pass_slice_keeps": {"pooled_tests_1_3": round(pooled * len(rows), 1),
                                              "tests_2_3_rate": round(late * len(rows), 1)},
                "caveat": "the cycle-3 generator removes the held posting title the labelers leaned on for Keep "
                          "(eval/diag_cycle3_rowread.md), so the yield may sit nearer the lower figure"}},
        "freshness": {
            "postings_unused": "every posting is reserved, in no pool (split.json pools) and in no fresh test 1-3 "
                               "manifest; checked below",
            "postings_in_a_pool": sorted(set(ids) & pool_ids),
            "postings_not_reserved": sorted(set(ids) - reserved),
            "postings_in_fresh_tests_1_3": sorted(set(ids) & used_any),
            "postings_in_cycle3_training_rows": sorted(set(ids) & cyc3),
            "training_text": training_text_hits({i: posts[i] for i in ids})},
        "employers_vs_fresh_tests_1_3": {
            "what": "fresh test 4 is new at the posting level, not the employer level: the reserved set is by "
                    "construction the other postings of the test and dev employers, and all 33 test employers and 46 of "
                    "the 70 dev employers are in fresh tests 1-3 (the only unused dev employer with reserved postings is "
                    "akebia therapeutics). This is the same overlap fresh test 2 had with fresh test 1. Tests 1-3 are "
                    "practice sets this cycle (amendment 4), so the pick is selected on postings of the same employers.",
            "employers": len(emps), "shared_with_fresh_tests_1_3": len(emps & e123),
            "new_employers": sorted(emps - e123),
            "postings_from_shared_employers": sum(posts[i]["emp_key"] in e123 for i in ids),
            "rows_from_shared_employers": sum(bool(r["design"]["employer_in_fresh_test1_3"]) for r in rows),
            "by_reserved_side": dict(Counter(posts[i]["reserved_side"] for i in ids)),
            "design_flag": "design.employer_in_fresh_test1_3 on every row",
            "employers_in_train_pool": sorted(emps & e_train)},
        "teasers": {
            "what": "the reserved set holds no teaser postings (1-699 characters): fresh tests 1-3's teasers were adzuna "
                    "truncations in the test and dev pools, and none was reserved. Fresh test 4 has no teasers; its short "
                    "postings are the empty bodies (title, company and place only).",
            "reserved_kept_by_length": notes.get("kept_by_length"),
            "test4_by_length": dict(by_len)},
    }


def set5_seal_extra(rows: list[dict], posts: dict, cards: dict, ids: list[str]) -> dict:
    """Fresh test 5's manifest fields (eval/protocol_amendment_6.json): the draw as executed, the writer, the checks
    that no posting was used before, and what departs from tests 1-4."""
    from fit_eval_score import title_in_resume
    walk = json.loads((OUT / "sample_walk.json").read_text())
    notes = json.loads((OUT / "round2_notes.json").read_text())
    used = set().union(*(fresh_test_keys(ROOT / f"eval/{m}") for m in (
        "fresh_test_manifest.json", "fresh_test2_manifest.json", "fresh_test3_manifest.json", "fresh_test4_manifest.json")))
    cyc3 = {r["posting_id"] for r in read_jsonl(ROOT / "data/built/fit/cycle3/rows.jsonl")}
    tries = read_jsonl(TRIES)
    batches = [json.loads(p.read_text()) | {"name": p.stem} for p in sonnet_batches()]
    first = Counter(t["try"] for t in tries)
    passed = Counter(t["try"] for t in tries if not t["problems"])
    emp = Counter(posts[i]["emp_key"] for i in ids)
    return {
        "what": (f"Sealed fresh test 5 (scripts/build_fresh_test.py --set 5, eval/protocol_amendment_6.json, the FALLBACK "
                 f"design): {len(ids)} round-2 Simplify ATS postings with their existing Adaption v2 cards, the "
                 "post-8c6bac5 candidate builder with the Keep-leaning design the amendment declares, resumes written "
                 "by Claude Sonnet through the same writer instruction, gates and retries as test 4, seed "
                 f"{SEED}, row ids {ROW_PREFIX}... Not gold: no row carries a label; gold comes from blind labelers A and "
                 "B and the adjudicator reading the label batches."),
        "decided_by": "eval/protocol_amendment_6.json (709da4a + a8e6dfd) and the author's decisions of "
                      "2026-10-22: zero Adaption credits, Sonnet writer, 300 postings at most 7 per employer, 4 "
                      "candidates a posting, refit-B as the bar's comparator",
        "sample": {
            "rule": "load_round2 (amendment 6 fresh_test_5.postings.filter) then cmd_sample_round2 (its draw): sorted "
                    f"ids, random.Random({SEED}).shuffle, walk taking a posting unless its company_key already has "
                    f"{EMPLOYER_CAP}; the first {N_POSTINGS} taken",
            "pool_filter": {k: v for k, v in notes.items() if k not in ("same_employer_and_title_ids",)},
            "walk_file": "data/built/fresh_test5/sample_walk.json (the whole walk, for amendment 6 size.extension: the "
                         "next 50 postings are walk[300:350], then walk[350:400], ...)",
            "walk_length": len(walk["walk"]), "next_extension_block": walk["walk"][N_POSTINGS:N_POSTINGS + 50],
            "employers": len(emp), "max_per_employer": max(emp.values()),
            "employers_at_cap": sorted(k for k, v in emp.items() if v >= EMPLOYER_CAP),
            "source_set_mix": {"postings": dict(Counter(posts[i]["source_set"] for i in ids)),
                               "rows": dict(Counter(r["design"]["source_set"] for r in rows))},
            "category_mix": dict(Counter(posts[i]["category"] for i in ids)),
            "field_note": "every posting is a tech internship, co-op or new-grad listing (Simplify categories above); "
                          "fit_pools.field_of puts most in 'internship' and kit_of picks the candidate field from the "
                          "title, so the field mix is narrower than test 4's, as the amendment expected"},
        "candidate_design": {
            "per_posting": PER_POSTING, "pass_rate": PASS_RATE, "arch_w_near_gap_strong": list(ARCH_W),
            "why": "amendment 6 candidates.design: Keep-leaning by design, declared before the build (the c3train "
                   "setting); test 4 used pass_rate 0.58 and 0.4/0.4/0.2",
            "target_pass_but_fail_work_mode_only": sum(bool(r["design"]["target_pass"]) and
                                                       r["design"]["eligibility_fails"] == ["work_mode"] for r in rows),
            "skill_vocab_coupling": {
                "what": "skill_vocab() reads every v2 card in data/built/cards/extracted_real_cards_v2.jsonl, whose "
                        "docstring says training-side postings only; test 5's postings are in that file, so for the first "
                        "time the vocabulary that finds skill mentions in a test posting includes that posting's own "
                        "card. Kept unchanged (amendment 6: the sampler is the existing one; dropping the test cards "
                        "would also change the first 300 postings' candidates whenever an extension block is added).",
                "measured_before_render": "with the 300 test postings' cards left out of the vocabulary, 172 of 300 "
                                          "postings get a different mention list (3,050 -> 2,835 mentions), and the "
                                          "candidates meet 34.7% of their card's must_skills groups instead of 36.0% "
                                          "(strong 51.8% vs 53.9%, near 47.8% both, gap 29.2% vs 31.2%, off 1.6% both): "
                                          "a small free ride for rules_full, which reads the card"}},
        "writer_sonnet": {
            "who": "Claude Sonnet via the Agent tool (model sonnet, effort medium), one agent per chunk of at most "
                   f"{SONNET_CHUNK} candidates, no two candidates of one posting in a chunk; Adaption wrote every earlier "
                   "test's resumes, so this is the amendment's one known style shift",
            "instruction": "WRITER_UNIVERSAL + blank line + WRITER_BLUEPRINT, unchanged since test 4 "
                           "(sha1[:12] of the two concatenated, as test 4's manifest records it: "
                           f"{hashlib.sha1((WRITER_UNIVERSAL + WRITER_BLUEPRINT).encode()).hexdigest()[:12]})",
            "instruction_file_sha256": hashlib.sha256(writer_instruction().encode()).hexdigest(),
            "harness": WRITER_HARNESS, "harness_sha256": hashlib.sha256(WRITER_HARNESS.encode()).hexdigest(),
            "context": "writer_context(r) per candidate, with fix_from_last_try (the gate's feedback) on a retry; the "
                       "writer sees no posting text and no label",
            "batches": [{k: b[k] for k in ("name", "n", "retries", "stage", "returned", "missing")} for b in batches],
            "tries_by_try_number": {"sent": dict(first), "passed": dict(passed)},
            "pass_counts_vs_test4": {
                "test5": {"first_try": passed[1], "second_try": passed[2], "third_try": passed[3],
                          "no_resume": len(json.loads((OUT / "assemble_dropped.json").read_text())),
                          "candidates": len(read_jsonl(CANDS))},
                "test4": {"first_try": 910, "second_try": 216, "third_try": 13,
                          "source": "eval/fresh_test4_manifest.json writer.tries (kept resumes by tries)"}},
            "staging": "data/writer_stage/fresh_test5/ (chunks, agent prompts and every saved file; hashed in "
                       "eval/fresh_test5_data_hashes.json)"},
        "freshness": {
            "postings_in_fresh_tests_1_4": sorted(set(ids) & used),
            "postings_in_cycle3_rows": sorted(set(ids) & cyc3),
            "postings_dropped_by_amendment6_overlap_rule": sorted(set(ids) & set(notes["dropped_by_amendment6"])),
            "training_text": training_text_hits({i: posts[i] for i in ids},
                                                TRAIN_FILES + ("data/built/fit/cycle3/train_B.csv",)),
            "training_text_hits_in_B_training": training_text_hits({i: posts[i] for i in ids},
                                                                   ("data/built/fit/cycle3/train_B.csv",))["hits"],
            "training_text_note": "train_fit.csv, train_fit2_*.csv and dpo_pairs.csv hold round-2 real_train rows "
                                  "(cycle 1-2 runs h and d3; amendment 6: never B, C or D), so their hits are these "
                                  "postings' own round-2 rows. In B's train_B.csv the two hits (Moderna, Regeneron) "
                                  "match on employer boilerplate (benefits and equal-opportunity paragraphs) shared "
                                  "with other postings of the same employers; the amendment's overlap rule found no "
                                  "near copy of either, and employer overlap with B's training is allowed and "
                                  "reported (amendment 6 employer_overlap_kept)"},
        "title_copies": {"rule": "fit_eval_score.title_in_resume (the posting's exact title in the resume)",
                         "rows": sum(title_in_resume(r["title"], r["resume_text"]) for r in rows)},
    }


def write_data_hashes() -> Path:
    """eval/fresh_test5_data_hashes.json: sha256 of every file of the sealed set (data/ is gitignored) and of the
    writers' staging folder, with the manifest's own hash."""
    import subprocess
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rel = lambda p: str(p.relative_to(ROOT))  # noqa: E731
    files = {rel(p): sha256(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    stage = WRITER_STAGE / OUT.name
    st = {rel(p): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    digest = hashlib.sha256("\n".join(f"{k} {v}" for k, v in sorted(st.items())).encode()).hexdigest()
    out = ROOT / "eval" / f"{OUT.name}_data_hashes.json"
    out.write_text(json.dumps({
        "what": f"sha256 of {OUT.name}'s data files (data/ is gitignored) and of the Sonnet writers' staging folder, "
                "computed at sealing, before any label batch was read by a labeler and before any model, rules or "
                "baseline output on this test",
        "computed_at_commit": head, "manifest": rel(MANIFEST), "manifest_sha256": sha256(MANIFEST),
        "sha256": files,
        "writer_stage": {"folder": rel(stage), "files": len(st),
                         "digest": "sha256 of the sorted 'path sha256' lines of every file in the folder",
                         "sha256": digest,
                         "prompts_and_chunks": {k: v for k, v in st.items() if "/out/" not in k}},
    }, indent=1) + "\n")
    return out


def set3_seal_extra(rows: list[dict], posts: dict, cards: dict, ids: list[str]) -> dict:
    """Fresh test 3's manifest fields: the sample rule as executed, disjointness from fresh tests 1 and 2 and from
    training (checked here, not assumed), and every earlier use of dev-pool postings."""
    from matcher.nearcopy import NearIndex
    from fit_pools import load_pool
    pool = [p for p in load_pool("dev") if language(p["text"] or "") == "en"]
    per_emp = Counter(p["emp_key"] for p in pool)
    k1, k2 = fresh_test_keys(FRESH1_MANIFEST), fresh_test_keys(FRESH2_MANIFEST)
    test_pool = {p["key"]: p for p in load_pool("test")}
    e12 = {test_pool[k]["emp_key"] for k in k1 | k2}
    e_train = {p["emp_key"] for p in load_pool("train")}
    emps = {posts[i]["emp_key"] for i in ids}
    raw = lambda p: {"title": p["title"], "company": p["employer"], "description": p["text"] or ""}  # noqa: E731
    idx = NearIndex({k: raw(test_pool[k]) for k in k1 | k2})
    # an empty or near-empty body "matches" every other one on description; only a same company and title hit, or a
    # description hit on a body of 60+ words, is a near copy
    near = {i: idx.near(raw(posts[i])) for i in ids}
    near = {i: v for i, v in near.items() if v and (v[1] == "same company and title" or
                                                    len((posts[i]["text"] or "").split()) >= 60)}
    df, dfa = dev_fit_postings()
    v2_cycle1 = {r["key"] for r in read_jsonl(ROOT / "data/built/fit/build/extract_cards.jsonl")}
    return {
        "what": (f"Sealed fresh test 3 (scripts/build_fresh_test.py --set 3): {len(ids)} English postings of the dev pool "
                 "(employers disjoint from the test pool and from training), sampled with fresh test 1's sampler and a "
                 "hard cap of 4 postings per employer, built exactly as fresh tests 1 and 2 (same v2 extractor imported "
                 "unchanged, same resume writer prompt and gates, same candidate design, new seed, row ids "
                 f"{ROW_PREFIX}...). Not gold: no row carries a label; gold comes from blind labelers reading the label "
                 "batches."),
        "sample": {
            "rule": "cmd_sample (fresh test 1's sampler) on data/built/fit/pools/dev.jsonl minus the non-English postings "
                    f"(language()): up to N_TEASER={N_TEASER} teasers with at most MAX_ADZUNA_TEASERS={MAX_ADZUNA_TEASERS} "
                    f"from adzuna, N_EMPTY={N_EMPTY} empty bodies, NON_US_CAP={NON_US_CAP}, then long postings by even "
                    f"field quotas; EMPLOYER_CAP={EMPLOYER_CAP} with no over-cap fallback (CAP_SLACK={CAP_SLACK})",
            "requested_postings": N_POSTINGS,
            "reachable_at_employer_cap": sum(min(EMPLOYER_CAP, v) for v in per_emp.values()),
            "taken": len(ids),
            "why_fewer": f"the dev pool has {len(pool)} English postings from {len(per_emp)} employers, {sum(v <= EMPLOYER_CAP for v in per_emp.values())} "
                         f"of them with {EMPLOYER_CAP} or fewer; at a cap of {EMPLOYER_CAP} at most "
                         f"{sum(min(EMPLOYER_CAP, v) for v in per_emp.values())} postings exist, and fresh test 1's teaser "
                         "and adzuna caps leave out most of the single-posting adzuna teasers. The employer cap was kept "
                         "over the posting count: raising it only adds postings from the same big employers.",
            "pool_employers_not_sampled": len(set(per_emp) - emps)},
        "disjointness": {
            "employers": len(emps),
            "employers_shared_with_fresh_test1_or_2": sorted(emps & e12),
            "employers_in_train_pool": sorted(emps & e_train),
            "postings_in_fresh_test1_or_2": sorted(set(ids) & (k1 | k2)),
            "near_copies_of_fresh_test1_or_2_postings": {i: list(v) for i, v in near.items() if v},
            "near_copy_rule": "matcher.nearcopy.NearIndex against every fresh test 1 and 2 posting: same company and "
                              "title, or description 6-word-window Jaccard >= 0.8 on a body of 60+ words (empty bodies "
                              "all match each other on description and are not counted)",
            "training_text": training_text_hits({i: posts[i] for i in ids}),
            "fresh_manifests_sha256": {"eval/fresh_test_manifest.json": sha256(FRESH1_MANIFEST),
                                       "eval/fresh_test2_manifest.json": sha256(FRESH2_MANIFEST)}},
        "earlier_use_of_these_postings": {
            "dev_fit": {
                "what": "cycle 1's dev_fit (data/built/fit/dev_fit.jsonl, rules-made gold) had rows built on dev-pool "
                        "postings, and dev_fit is the set that selected h over d3 (eval/selection_v4.json): model-"
                        "selection exposure, not training. Nothing was trained on them.",
                "postings": sum(i in df for i in ids), "rows": sum(bool(r["design"]["posting_in_dev_fit"]) for r in rows),
                "postings_in_dev_fit_all": sum(i in dfa for i in ids),
                "design_flags": "design.posting_in_dev_fit and design.posting_in_dev_fit_all on every row"},
            "cycle1_v2_cards": {
                "what": "build_fit_train extracted v2 cards (data/built/fit/build/extract_cards.jsonl) for dev-pool "
                        "postings to build dev_fit; those cards are not read here (test 3's cards are a new extraction)",
                "postings": sum(i in v2_cycle1 for i in ids)},
            "extractor_v3": "extract_v3.py's 'fit' set includes every dev-pool posting; its chunks were uploaded for a "
                            "free estimate (data/adaption/fixcycle2_fit_0e7b4209_*.json carry no run_id) and never run; "
                            "v3 is not live either way",
            "extractor_v2": "the v2 instruction text (blueprint b2b2ed8e) is the text the pools were frozen under "
                            "(c6c814b) and fresh tests 1 and 2 were built with; no extractor run on dev-pool postings "
                            "fed an extractor change (cycle 1's v2 cards built dev_fit; v3's fit chunks were never run)"},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), default="1", help="which sealed test to build (1: the v4 fresh test)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample")
    e = sub.add_parser("extract")
    e.add_argument("mode", choices=("estimate", "run"))
    e.add_argument("--max-credits", type=float, default=4.0)
    sub.add_parser("parse")
    sub.add_parser("candidates")
    r = sub.add_parser("render")
    r.add_argument("mode", choices=("estimate", "run", "prepare", "absorb"),
                   help="estimate|run: Adaption writes (sets 1-4); prepare|absorb: Claude Sonnet writes (set 5)")
    r.add_argument("--max-credits", type=float, default=15.0)
    sub.add_parser("assemble")
    sub.add_parser("seal")
    sub.add_parser("extend")
    r.add_argument("--limit", type=int, default=0, help="render only the first N rows still to do (a pilot)")
    a = ap.parse_args()
    use_set(a.set)
    {"sample": cmd_sample, "extract": cmd_extract, "parse": lambda _a: parse_cards(), "candidates": cmd_candidates, "render": cmd_render, "assemble": cmd_assemble,
     "seal": cmd_seal, "extend": cmd_extend}[a.cmd](a)


if __name__ == "__main__":
    main()
