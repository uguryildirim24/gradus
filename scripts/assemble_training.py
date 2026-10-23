"""Smoke-run training set: pick pairs, then join rendered text into prompt/completion rows.

  assemble_training.py pick [--n 0] [--per-posting 7] [--dev 750] [--dev-per-posting 7] [--enroll-max-share 0.10]
                           [--dry-run]
      Seeded, bucket-balanced pick of train-split pairs (--n rows; 0, the default, is the largest balanced set the
      rendered rows allow) and --dev held-out dev-split pairs, from the pairs whose resume and posting both have the
      text build() puts in the prompt (prompt_texts). Rows are taken round-robin over the base postings, at most
      --per-posting (--dev-per-posting) per base, so a bucket that fills early does not leave the last bases with
      one bucket only. Then, in each split, enrollment re-pairs (enrollment_repairs: a graduated candidate on a
      student-only posting, an enrolled one on a student role for a higher level, variant kinds
      repair_enrollment_graduated and repair_enrollment_level) take the place of picked Skip rows, as many as keep
      every stance, variant kind and other Skip reason at 80% of its count and the enrollment slice under
      --enroll-max-share (0 turns them off). Writes data/built/training/smoke_pairs.jsonl, dev_pairs.jsonl and their
      card ids (resume_ids.txt, posting_ids.txt); --dry-run prints the counts and writes nothing. Refuses (exit 1,
      before writing) when the texts the gates dropped skew a split: among the split's rows with both texts, any
      work-authorization stance or candidate variant kind under SKEW_MIN (80%) of its share of all the split's rows
      (skew_problems). The bucket balance then cuts the larger buckets by design, as it would on a perfect render.
  assemble_training.py build
      Looks up each card's rendered text (render_cards.rendered_texts: text for the card's
      current context only; postings through render_cards.training_postings, which puts the
      Title/Company/Locations header and 0-3 boilerplate bank paragraphs around the rendered text,
      the format real test and demo postings get from textclean.real_posting_text), renders the
      candidate profile from the card, builds prompts with build_prompt(resume, profile, posting)
      (the same call build_real_test and the demo make), and checks label words, bucket counts,
      split leaks, posting truncation against build_prompt's budget and fact visibility
      (matcher.visibility) on the texts as the prompt holds them (pairs.prompt_parts: resume
      redacted, posting fitted). Writes data/built/training/smoke.csv (prompt, completion),
      train.jsonl and dev.jsonl (ids, bucket, prompt, completion, missing facts) and
      leakage_rows.jsonl (the prompt's own texts) for leakage_report.py.
  assemble_training.py all-ids
      Every card id the pair manifest uses, all splits (the full render): data/built/training/all_resume_ids.txt
      and all_posting_ids.txt, for render_cards.py resumes|postings run.
  assemble_training.py cross [--swap-profile 225] [--swap-posting 75]
      Re-paired dev set, data/built/training/dev_cross.jsonl, labeled by rules.label from dev text
      only: true resume + posting with another dev candidate's profile (swap_profile, buckets
      steered toward equal; no profile whose work authorization the resume's education contradicts),
      and a dev candidate with another base's posting (swap_posting). One-edit
      variants put their edit in the profile, so on true pairs the profile alone hints the label; on
      these rows it does not, so a model that only read the profile falls apart here.
"""
from __future__ import annotations

import argparse
import math
import csv
import dataclasses
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from render_cards import POSTING_BUDGET, load_cards, rendered_texts, training_postings  # noqa: E402

from matcher.pairs import build_prompt, prompt_parts  # noqa: E402
from matcher.profile import render_profile  # noqa: E402
from matcher.rules import CURRENT_YEAR, DEGREE_LEVELS, LABEL_WORDS, format_completion, is_enrolled, label, reasons_by_check, \
    status_problems  # noqa: E402
from matcher.schema import CandidateCard, RequirementCard  # noqa: E402
from matcher.textclean import strip_markdown  # noqa: E402
from matcher.visibility import missing_facts, stance_conflict  # noqa: E402

PAIRS = ROOT / "data/built/pairs"
OUT = ROOT / "data/built/training"
TARGET = ("For you", "Explore", "Excluded")
WORDING = "skill_wording"  # a skill named differently in the text; every other miss is an unseen fact


def read_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def base_of_cd(cd_id: str) -> str | None:
    m = re.match(r"cd_v_(\d+)_", cd_id)
    return f"rq_{m.group(1)}" if m else None


SKEW_MIN = 0.8  # a stance or variant kind among a split's rows with text keeps at least this share of its split share
SKEW_MIN_EXPECTED = 10  # ... judged when its split share gives it at least this many of those rows


def stance_of(card: dict) -> str:
    """A posting card's work-authorization stance: citizens_only, or its sponsorship stance."""
    return "citizens_only" if card.get("citizens_only") else card.get("sponsorship") or "unstated"


def skew_problems(picked: list[dict], train: list[dict], have_r, have_p, posting_cards: dict[str, dict]) -> tuple[dict, list[str]]:
    """Rows of `picked` whose resume and posting both have rendered text, against all of `picked`: every
    work-authorization stance (of the posting card) and every candidate variant kind keeps at least SKEW_MIN of its
    share. pick() passes a whole split as both `picked` and `train`. A value whose share gives it fewer than
    SKEW_MIN_EXPECTED of those rows is reported, not judged (too few to tell). No row with text yet: nothing to
    judge. The share in `train` is reported next to it. Returns (shares, problems)."""
    got = [r for r in picked if r["cd_id"] in have_r and r["rq_id"] in have_p]
    shares: dict = {"picked_with_text": len(got), "picked": len(picked)}
    if not got or not picked:
        return shares, []
    out = []
    for name, f in (("work-authorization stance", lambda r: stance_of(posting_cards.get(r["rq_id"], {}))),
                    ("candidate variant kind", lambda r: r["variant_kind"])):
        t, p, g = Counter(f(r) for r in train), Counter(f(r) for r in picked), Counter(f(r) for r in got)
        shares[name] = {}
        for k, n in sorted(p.items(), key=lambda kv: -kv[1]):
            sp, sg = n / len(picked), g[k] / len(got)
            shares[name][k] = {"train_split": round(t[k] / len(train), 4) if train else None, "picked": round(sp, 4),
                               "picked_with_text": round(sg, 4)}
            if sp * len(got) >= SKEW_MIN_EXPECTED and sg < SKEW_MIN * sp:
                out.append(f"{name} {k}: {sg:.1%} of the {len(got)} rows with text, under {SKEW_MIN:.0%} of its "
                           f"{sp:.1%} share of all the rows ({t[k] / max(len(train), 1):.1%} of the reference)")
    return shares, out


def prompt_texts(cd_cards: dict[str, dict], rq_cards: dict[str, dict]) -> tuple[dict[str, str], dict[str, str]]:
    """(resume texts, posting texts) a prompt can use: gated rendered resumes, and training postings
    (render_cards.training_postings) less those whose text contradicts the card's stance (stance_conflict)."""
    resumes, postings = rendered_texts("resumes", cd_cards), training_postings(rq_cards)
    conflicts = {i: why for i, t in postings.items() if (why := stance_conflict(rq_cards[i], t))}
    for i, why in sorted(conflicts.items()):
        print(f"dropping posting {i}: {why}")
    return resumes, {i: t for i, t in postings.items() if i not in conflicts}


def pick_balanced(rows: list[dict], n: int, per_base: int, rng: random.Random) -> list[dict]:
    """Equal buckets, round-robin over shuffled base postings: each round every base adds the row of the bucket
    furthest under its target, at most per_base rows a base. n = 0: as many as the smallest bucket allows. The
    result is cut to equal buckets when per_base keeps a bucket under its target."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["base_rq_id"]].append(r)
    bases = sorted(by)
    rng.shuffle(bases)
    for b in bases:
        rng.shuffle(by[b])
    have = Counter(r["bucket"] for r in rows)
    n = n or len(TARGET) * min(have[b] for b in TARGET)
    per = n // len(TARGET)
    tgt = {b: per + (1 if i < n - per * len(TARGET) else 0) for i, b in enumerate(TARGET)}
    got: Counter = Counter()
    taken: Counter = Counter()
    sel: list[dict] = []
    while len(sel) < n:
        added = False
        for b in bases:
            if taken[b] >= per_base:
                continue
            open_ = [r for r in by[b] if got[r["bucket"]] < tgt[r["bucket"]]]
            if not open_:
                continue
            r = min(open_, key=lambda r: got[r["bucket"]] - tgt[r["bucket"]])
            by[b].remove(r)
            sel.append(r)
            got[r["bucket"]] += 1
            taken[b] += 1
            added = True
            if len(sel) >= n:
                break
        if not added:
            break
    low = min(got[b] for b in TARGET)
    if any(got[b] != low for b in TARGET):  # keep the earliest rows of each bucket: later rounds are the extras
        kept: Counter = Counter()
        out = []
        for r in sel:
            if kept[r["bucket"]] < low:
                kept[r["bucket"]] += 1
                out.append(r)
        sel = out
    return sel


# ---------------------------------------------------------------- enrollment re-pairs
# A graduate on a student-only role ('Skip — for students currently pursuing ...; you have graduated') and a student on
# a role for a higher level ('...; you are pursuing a bachelor's degree') were 2.2% of train and about 30% of the real
# test, and the split's own variants cannot give more. Re-pairing gives more at no cost: a graduated candidate of the
# split with a student-only posting of the split, an enrolled one with a student-only posting of a higher level, both
# texts already rendered, the label from rules.label. A re-pair is kept only when the label's hard reasons are the
# enrollment one (and a work-authorization one, as on the real test) and every fact behind them shows in the prompt.
# Each one takes the place of a picked Skip row, so the buckets stay equal.
REPAIR_KINDS = {"graduated": "repair_enrollment_graduated", "level": "repair_enrollment_level"}
REPAIR_RATIO = 2  # graduated re-pairs per wrong-level one, as far as the pools allow
REPAIR_MAX_SHARE = 0.10  # the enrollment slice (old rows and re-pairs) is at most this share of the set
REPAIR_ALLOWED = frozenset({"enrollment", "work_auth"})
# a candidate in at most this many re-pairs: few graduates want an internship in the posting's place, so the valid
# pairs of a posting share a handful of candidates
REPAIR_MAX_USES = 3


def _rank(level: str) -> int:
    return DEGREE_LEVELS.index(level)


def hard_checks(c: CandidateCard, q: RequirementCard, reasons: list[str]) -> list[str]:
    own = reasons_by_check(c, q)
    return [own.get(x, "?") for x in reasons]


def repair_row(cd_id: str, c: CandidateCard, rq_id: str, q: RequirementCard, resumes: dict[str, str],
               postings: dict[str, str], split: str, kind: str) -> dict | None:
    """The re-paired row, or None when its label is not an enrollment Skip of that kind or a fact behind it is unseen."""
    if status_problems(c):
        return None
    if kind == "graduated" and not (c.degree_level != "none" and c.graduation_year is not None
                                     and c.graduation_year <= CURRENT_YEAR):
        return None
    if kind == "level" and not (is_enrolled(c) and _rank(c.degree_level) < _rank(q.enrolled_in)):
        return None
    b, reasons, flags = label(c, q)
    checks = hard_checks(c, q, reasons)
    if b != "Excluded" or "enrollment" not in checks or not set(checks) <= REPAIR_ALLOWED:
        return None
    resume, profile, posting = prompt_parts(build_prompt(resumes[cd_id], render_profile(c), postings[rq_id]))
    if missing_facts(c, q, reasons, resume, profile, posting):
        return None
    return {"pair_id": "pe_" + hashlib.sha1(f"enroll|{rq_id}|{cd_id}".encode()).hexdigest()[:12], "rq_id": rq_id,
            "cd_id": cd_id, "base_rq_id": rq_id.split("~")[0], "bucket": b, "reasons": reasons, "flags": flags,
            "variant_kind": REPAIR_KINDS[kind], "candidate_kind": "enrollment_repair", "repaired_from": None,
            "source": None, "family": None, "split": split, "completion": format_completion(b, reasons)}


def _attrs(r: dict, checks: list[str], rq_cards: dict[str, dict]) -> list[tuple[str, str]]:
    """What a row counts toward under the floor: its stance, its variant kind and (a Skip) each hard reason's check."""
    out = [("stance", stance_of(rq_cards.get(r["rq_id"], {}))), ("kind", str(r["variant_kind"]))]
    if r["bucket"] == "Excluded":
        out += [("skip reason", ch) for ch in dict.fromkeys(checks)]
    return out


def enrollment_repairs(sel: list[dict], split_rows: list[dict], split: str, cd_cards: dict[str, dict],
                       rq_cards: dict[str, dict], resumes: dict[str, str], postings: dict[str, str], per_base: int,
                       rng: random.Random, max_share: float = REPAIR_MAX_SHARE) -> tuple[list[dict], dict]:
    """`sel` with picked Skip rows swapped for enrollment re-pairs (repair_row) from the split's own cards, as many as
    keep every stance, every variant kind and every other Skip reason at SKEW_MIN (80%) or more of its count in `sel`
    and the enrollment slice at most `max_share` of the set. A re-pair goes on a student-only base with room under
    `per_base` rows; when none has room, the Skip row it replaces comes from its own base. A candidate in at most
    REPAIR_MAX_USES re-pairs, never a pair the manifest has. Returns (rows, report)."""
    C = {i: CandidateCard.from_dict(cd_cards[i]) for i in dict.fromkeys(r["cd_id"] for r in split_rows) if i in resumes}
    Q = {i: RequirementCard.from_dict(rq_cards[i]) for i in dict.fromkeys(r["rq_id"] for r in split_rows)
         if i in postings and rq_cards[i].get("enrolled_in", "none") != "none"}
    assert all(r["split"] == split for r in split_rows), "re-pairs come from one split"
    seen = {(r["cd_id"], r["rq_id"]) for r in split_rows}
    checks_of = {}
    for r in sel:
        if r["bucket"] == "Excluded":
            c, q = CandidateCard.from_dict(cd_cards[r["cd_id"]]), RequirementCard.from_dict(rq_cards[r["rq_id"]])
            checks_of[r["pair_id"]] = hard_checks(c, q, r["reasons"])
    before: Counter = Counter(a for r in sel for a in _attrs(r, checks_of.get(r["pair_id"], []), rq_cards))
    now = before.copy()
    slice_old = sum("enrollment" in checks_of.get(r["pair_id"], ()) for r in sel)
    target = max(0, math.floor(max_share * len(sel)) - slice_old)
    rows = list(sel)
    per: Counter = Counter(r["base_rq_id"] for r in rows)
    order = {k: sorted(C) for k in REPAIR_KINDS}
    for k in order:
        rng.shuffle(order[k])
    by_base: dict[str, list[str]] = defaultdict(list)
    for i in sorted(Q):
        by_base[i.split("~")[0]].append(i)
    bases = sorted(by_base)
    rng.shuffle(bases)
    ptr: dict[tuple[str, str], int] = {}
    taken: Counter = Counter()  # re-pairs (made or reserved) per candidate

    def next_pair(base: str, kind: str) -> dict | None:
        """The next valid re-pair of this kind on this base (pointer per base and posting, so a miss is not tried again)."""
        for rq in by_base[base]:
            lst, j = order[kind], ptr.get((rq, kind), 0)
            while j < len(lst):
                cd = lst[j]
                j += 1
                if taken[cd] >= REPAIR_MAX_USES or (cd, rq) in seen or (cd, rq) in made:
                    continue
                row = repair_row(cd, C[cd], rq, Q[rq], resumes, postings, split, kind)
                if row:
                    ptr[(rq, kind)] = j
                    return row
            ptr[(rq, kind)] = j
        return None

    def slack(a) -> float:
        return now[a] - 1 - SKEW_MIN * before[a]

    # the Skip rows a re-pair may replace: not an enrollment Skip already
    swap = {r["pair_id"]: r for r in sel if r["pair_id"] in checks_of and "enrollment" not in checks_of[r["pair_id"]]}
    attrs = {i: _attrs(r, checks_of[i], rq_cards) for i, r in swap.items()}
    swap_by_base: dict[str, set[str]] = defaultdict(set)
    for i, r in swap.items():
        swap_by_base[r["base_rq_id"]].add(i)

    def removable(base: str | None) -> dict | None:
        """The swappable Skip row (of `base`, if given) whose stance, kind and reasons keep the most room over the
        floor, so the swaps spread over them; None when every one would take something under it."""
        best, score = None, None
        for i in sorted(swap if base is None else swap_by_base[base]):
            if any(slack(a) < -1e-9 for a in attrs[i]):
                continue
            sc = min(slack(a) / before[a] for a in attrs[i])
            if score is None or sc > score:
                best, score = swap[i], sc
        return best

    made: set[tuple[str, str]] = set()
    added: Counter = Counter()
    pending: dict[tuple[str, str], dict | None] = {}
    dead: set[tuple[str, str]] = set()
    stop = "target"
    while sum(added.values()) < target:
        kind = "level" if added["graduated"] >= REPAIR_RATIO * (added["level"] + 1) else "graduated"
        progressed = False
        for k in (kind, *(x for x in REPAIR_KINDS if x != kind)):
            for b in bases:
                if (b, k) in dead:
                    continue
                if (b, k) not in pending:
                    pending[(b, k)] = next_pair(b, k)
                    if pending[(b, k)]:
                        taken[pending[(b, k)]["cd_id"]] += 1  # reserved
                        made.add((pending[(b, k)]["cd_id"], pending[(b, k)]["rq_id"]))
                row = pending[(b, k)]
                if row is None:
                    dead.add((b, k))
                    continue
                out = removable(None if per[b] < per_base else b)
                if out is None:
                    continue
                rows[rows.index(out)] = row  # in the Skip row's place, so re-pairs spread through the file
                per[out["base_rq_id"]] -= 1
                for a in attrs[out["pair_id"]]:
                    now[a] -= 1
                del swap[out["pair_id"]]
                swap_by_base[out["base_rq_id"]].discard(out["pair_id"])
                per[b] += 1
                for a in _attrs(row, hard_checks(C[row["cd_id"]], Q[row["rq_id"]], row["reasons"]), rq_cards):
                    now[a] += 1
                added[k] += 1
                del pending[(b, k)]
                bases.remove(b)
                bases.append(b)  # round-robin: the next re-pair goes to another base
                progressed = True
                break
            if progressed:
                break
        if not progressed:
            waiting = {b for (b, _), v in pending.items() if v}
            no_swap = sum(not swap_by_base[b] for b in waiting)
            stop = (f"no more valid re-pairs; {len(waiting)} bases still hold one but are at {per_base} rows, "
                    f"{no_swap} of them with no Skip row left to swap and the rest with only Skip rows the floor keeps"
                    if waiting else "no more valid re-pairs")
            break
    lowest = sorted(((now[a] / before[a], a) for a in before if before[a]), key=lambda x: x[0])[:5]
    rep = {"split": split, "rows": len(rows), "slice_before": slice_old, "added": dict(added),
           "slice_after": slice_old + sum(added.values()), "slice_share": round((slice_old + sum(added.values())) / len(rows), 4),
           "target_added": target, "stopped_by": stop, "student_bases": len(bases),
           "repair_candidates": len({r["cd_id"] for r in rows if r["candidate_kind"] == "enrollment_repair"}),
           "repair_postings": len({r["rq_id"] for r in rows if r["candidate_kind"] == "enrollment_repair"}),
           "repair_bases": len({r["base_rq_id"] for r in rows if r["candidate_kind"] == "enrollment_repair"}),
           "repair_stances": dict(Counter(stance_of(rq_cards[r["rq_id"]]) for r in rows
                                          if r["candidate_kind"] == "enrollment_repair")),
           "repair_work_auth": dict(Counter(C[r["cd_id"]].work_auth for r in rows if r["candidate_kind"] == "enrollment_repair")),
           "lowest_count_ratios": [(f"{a[0]} {a[1]}", round(x, 3)) for x, a in lowest],
           "buckets": dict(Counter(r["bucket"] for r in rows))}
    assert Counter(r["bucket"] for r in rows) == Counter(r["bucket"] for r in sel)
    assert max(Counter(r["base_rq_id"] for r in rows).values()) <= max(per_base, max(Counter(r["base_rq_id"] for r in sel).values()))
    return rows, rep


def pick(a: argparse.Namespace) -> None:
    rows = read_jsonl(PAIRS / "manifest.jsonl")
    rng = random.Random(a.seed)
    cd_cards, rq_cards = load_cards("resumes"), load_cards("postings")
    have_r, have_p = prompt_texts(cd_cards, rq_cards)
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "dev")}
    skew = []
    for s, rs in split.items():
        shares, why = skew_problems(rs, rs, have_r, have_p, rq_cards)
        print(f"skew guard {s}: {shares['picked_with_text']} of {shares['picked']} {s}-split rows have both texts")
        skew += [f"{s} split: {w}" for w in why]
    if skew:  # before anything is written
        sys.exit("refusing: the rows with rendered text skew against their split (texts the gates dropped fall "
                 "unevenly):\n  " + "\n  ".join(skew))
    pool = {s: [r for r in rs if r["cd_id"] in have_r and r["rq_id"] in have_p] for s, rs in split.items()}
    for s, rs in pool.items():
        print(f"{s} rows with both texts by bucket", dict(Counter(r["bucket"] for r in rs)))
    sel = pick_balanced(pool["train"], a.n, a.per_posting, rng)
    dsel = pick_balanced(pool["dev"], a.dev, a.dev_per_posting, rng)
    if a.dev and len(dsel) < a.dev:
        sys.exit(f"refusing: only {len(dsel)} balanced dev rows (asked {a.dev}); raise --dev-per-posting")
    if a.enroll_max_share > 0:
        for name, per_base in (("train", a.per_posting), ("dev", a.dev_per_posting)):
            got = sel if name == "train" else dsel
            new, rep = enrollment_repairs(got, split[name], name, cd_cards, rq_cards, have_r, have_p, per_base, rng,
                                          a.enroll_max_share)
            print(f"enrollment re-pairs {name}: {json.dumps(rep)}")
            if name == "train":
                sel = new
            else:
                dsel = new
    if a.dry_run:
        for name, rs in (("train", sel), ("dev", dsel)):
            print(f"dry run {name}: {len(rs)} rows {dict(Counter(r['bucket'] for r in rs))}, "
                  f"max {max(Counter(r['base_rq_id'] for r in rs).values())} a base; nothing written")
        return
    write_jsonl(OUT / "smoke_pairs.jsonl", sel)
    write_jsonl(OUT / "dev_pairs.jsonl", dsel)
    res_ids = list(dict.fromkeys(r["cd_id"] for r in sel + dsel))
    post_ids = list(dict.fromkeys(r["rq_id"] for r in sel + dsel))
    (OUT / "resume_ids.txt").write_text("\n".join(res_ids) + "\n")
    (OUT / "posting_ids.txt").write_text("\n".join(post_ids) + "\n")
    for name, rs in (("train", sel), ("dev", dsel)):
        print(f"{name} pairs {len(rs)} {dict(Counter(r['bucket'] for r in rs))} | kinds "
              f"{dict(Counter(r['candidate_kind'] for r in rs))} | base postings {len({r['base_rq_id'] for r in rs})}, "
              f"posting texts {len({r['rq_id'] for r in rs})}, resumes {len({r['cd_id'] for r in rs})}")


def build(a: argparse.Namespace) -> None:
    splits = json.loads((PAIRS / "splits.json").read_text())
    rq_split, cd_split = splits["requirement_split"], splits["candidate_split"]
    cd_cards, rq_cards = load_cards("resumes"), load_cards("postings")
    resumes, postings = prompt_texts(cd_cards, rq_cards)

    def join(rows: list[dict]) -> tuple[list[dict], int, int]:
        out, missing, trunc = [], 0, 0
        for r in rows:
            if r["cd_id"] not in resumes or r["rq_id"] not in postings:
                missing += 1
                continue
            c = CandidateCard.from_dict(cd_cards[r["cd_id"]])
            q = RequirementCard.from_dict(rq_cards[r["rq_id"]])
            prompt = build_prompt(resumes[r["cd_id"]], render_profile(c), postings[r["rq_id"]])
            resume, profile, posting = prompt_parts(prompt)  # as the prompt holds them
            trunc += len(strip_markdown(postings[r["rq_id"]]).strip()) > POSTING_BUDGET
            miss = missing_facts(c, q, r["reasons"], resume, profile, posting)
            out.append(r | {"resume": resume, "profile": profile, "posting": posting, "prompt": prompt, "missing": miss})
        return out, missing, trunc

    train, miss_t, trunc_t = join(read_jsonl(OUT / "smoke_pairs.jsonl"))
    dev, miss_d, trunc_d = join(read_jsonl(OUT / "dev_pairs.jsonl"))
    print(f"train rows {len(train)} (missing text {miss_t}, posting truncated {trunc_t}); "
          f"dev rows {len(dev)} (missing {miss_d}, truncated {trunc_d})")
    if not train or not dev:  # nothing rendered for these cards yet: keep the last outputs, write nothing
        raise SystemExit("no text for the picked pairs yet: render data/built/training/resume_ids.txt and "
                         "posting_ids.txt (render_cards.py resumes|postings run) first")

    words = tuple(LABEL_WORDS.values())
    bad = [r["pair_id"] for r in train + dev if not r["completion"].startswith(words)]
    wrong = [r["pair_id"] for r in train + dev if r["completion"].split()[0] != LABEL_WORDS[r["bucket"]]]
    assert not bad and not wrong, (bad[:5], wrong[:5])
    print("label words OK; train buckets", dict(Counter(r["completion"].split()[0] for r in train)),
          "| dev buckets", dict(Counter(r["completion"].split()[0] for r in dev)))

    leaks = []
    for r in train:
        cd_ok = (cd_split.get(r["cd_id"]) == "train") if not r["cd_id"].startswith("cd_v_") \
            else rq_split.get(base_of_cd(r["cd_id"])) == "train"
        if r["split"] != "train" or rq_split.get(r["base_rq_id"]) != "train" or not cd_ok:
            leaks.append(r["pair_id"])
    train_ids = {r["cd_id"] for r in train} | {r["rq_id"] for r in train} | {r["base_rq_id"] for r in train}
    dev_overlap = [r["pair_id"] for r in dev if {r["cd_id"], r["rq_id"], r["base_rq_id"]} & train_ids]
    assert not leaks and not dev_overlap, (leaks[:5], dev_overlap[:5])
    print("leak check OK: every posting and candidate in train is train-split; dev shares no card id with train")

    for name, rows in (("train", train), ("dev", dev)):
        cats: Counter = Counter()
        for r in rows:
            cats.update({k for k, _ in r["missing"]})
        unseen = sum(any(k != WORDING for k, _ in r["missing"]) for r in rows)
        print(f"visibility {name}: {unseen}/{len(rows)} rows ({unseen / max(len(rows), 1):.1%}) rely on a fact the "
              f"prompt does not show; {cats[WORDING]} rows with skill wording only or as well; rows by category {dict(cats)}")

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
        n = sorted(len(tok(r["prompt"] + " " + r["completion"])["input_ids"]) for r in train)
        print(f"tokens per row: median {n[len(n) // 2]}, max {n[-1]}, total {sum(n)}")
    except Exception as e:  # tokenizer is optional
        print("token count skipped:", e)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "smoke.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "completion"])
        for r in train:
            w.writerow([r["prompt"], r["completion"]])
    keep = ("pair_id", "rq_id", "cd_id", "bucket", "candidate_kind", "variant_kind", "prompt", "completion", "missing")
    write_jsonl(OUT / "train.jsonl", [{k: r[k] for k in keep} for r in train])
    write_jsonl(OUT / "dev.jsonl", [{k: r[k] for k in keep} for r in dev])
    write_jsonl(OUT / "leakage_rows.jsonl", [
        {"id": r["pair_id"], "split": s, "label": r["bucket"], "resume": r["resume"], "profile": r["profile"],
         "posting": r["posting"]} for s, rows in (("train", train), ("dev", dev)) for r in rows])
    print("wrote", ", ".join(str((OUT / x).relative_to(ROOT)) for x in ("smoke.csv", "train.jsonl", "dev.jsonl",
                                                                         "leakage_rows.jsonl")))


# Candidate facts that live only in the profile (current_location is also the resume's header city).
PROFILE_FIELDS = ("work_auth", "willing_locations", "remote_preference", "salary_floor", "employment_types_wanted",
                  "industries_wanted", "industries_avoided", "roles_avoided")


def profile_swaps(c: CandidateCard, q: RequirementCard, donors: list[CandidateCard]) -> dict[str, list[CandidateCard]]:
    """`c` with each donor's profile fields, by the bucket rules.label gives the mix. A mix whose work
    authorization the resume's education contradicts (rules.status_problems: CPT on a graduate, OPT on a
    student) is left out: the profile would contradict the resume."""
    opts: dict[str, list[CandidateCard]] = defaultdict(list)
    for donor in donors:
        mixed = dataclasses.replace(c, **{f: getattr(donor, f) for f in PROFILE_FIELDS})
        if not status_problems(mixed):
            opts[label(mixed, q)[0]].append(mixed)
    return opts


def cross(a: argparse.Namespace) -> None:
    """Two re-paired sets from dev text only, labels from rules.label:
    swap_profile: the true resume + posting with the profile fields of another dev candidate
                  (another base), buckets steered toward equal;
    swap_posting: a dev candidate with a dev posting from another base (nearly always Skip)."""
    dev = read_jsonl(OUT / "dev.jsonl")
    cd_cards, rq_cards = load_cards("resumes"), load_cards("postings")
    resumes, postings = rendered_texts("resumes", cd_cards), training_postings(rq_cards)
    rng = random.Random(a.seed)
    base = {r["pair_id"]: r["rq_id"].split("~")[0] for r in dev}
    out: list[dict] = []

    def add(kind: str, cd_id: str, c: CandidateCard, rq_id: str, q: RequirementCard) -> None:
        b, reasons, _ = label(c, q)
        prompt = build_prompt(resumes[cd_id], render_profile(c), postings[rq_id])
        resume, profile, posting = prompt_parts(prompt)
        out.append({"pair_id": f"{kind[5:9]}_" + hashlib.sha1(f"{rq_id}|{cd_id}|{profile}".encode()).hexdigest()[:12],
                    "rq_id": rq_id, "cd_id": cd_id, "bucket": b, "candidate_kind": kind, "variant_kind": None,
                    "prompt": prompt, "completion": format_completion(b, reasons),
                    "missing": missing_facts(c, q, reasons, resume, profile, posting)})

    order = dev[:]
    rng.shuffle(order)
    target = a.swap_profile / len(TARGET)
    got: Counter = Counter()
    for r in order:
        if len([x for x in out if x["candidate_kind"] == "swap_profile"]) >= a.swap_profile:
            break
        c, q = CandidateCard.from_dict(cd_cards[r["cd_id"]]), RequirementCard.from_dict(rq_cards[r["rq_id"]])
        opts = profile_swaps(c, q, [CandidateCard.from_dict(cd_cards[o["cd_id"]]) for o in dev
                                    if base[o["pair_id"]] != base[r["pair_id"]]])
        need = sorted((b for b in TARGET if opts[b]), key=lambda b: got[b] - target)
        if need:
            got[need[0]] += 1
            add("swap_profile", r["cd_id"], rng.choice(opts[need[0]]), r["rq_id"], q)
    for r in order[: a.swap_posting]:
        others = [o for o in dev if base[o["pair_id"]] != base[r["pair_id"]]]
        o = rng.choice(others)
        add("swap_posting", r["cd_id"], CandidateCard.from_dict(cd_cards[r["cd_id"]]), o["rq_id"],
            RequirementCard.from_dict(rq_cards[o["rq_id"]]))
    write_jsonl(OUT / "dev_cross.jsonl", out)
    for kind in ("swap_profile", "swap_posting"):
        rows = [x for x in out if x["candidate_kind"] == kind]
        unseen = sum(any(k != WORDING for k, _ in x["missing"]) for x in rows)
        print(f"{kind}: {len(rows)} rows {dict(Counter(x['bucket'] for x in rows))}; {unseen} rely on a fact the prompt "
              f"does not show")
    print("wrote", (OUT / "dev_cross.jsonl").relative_to(ROOT))


def all_ids(a: argparse.Namespace) -> None:
    rows = read_jsonl(PAIRS / "manifest.jsonl")
    OUT.mkdir(parents=True, exist_ok=True)
    for kind, key in (("resume", "cd_id"), ("posting", "rq_id")):
        ids = list(dict.fromkeys(r[key] for r in rows))
        (OUT / f"all_{kind}_ids.txt").write_text("\n".join(ids) + "\n")
        print(f"wrote {len(ids)} {kind} ids to {(OUT / f'all_{kind}_ids.txt').relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--per-posting", type=int, default=7)
    p.add_argument("--dev", type=int, default=750)
    p.add_argument("--dev-per-posting", type=int, default=7)
    p.add_argument("--enroll-max-share", type=float, default=REPAIR_MAX_SHARE)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    sub.add_parser("build")
    sub.add_parser("all-ids")
    x = sub.add_parser("cross")
    x.add_argument("--swap-profile", type=int, default=225)
    x.add_argument("--swap-posting", type=int, default=75)
    x.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"pick": pick, "build": build, "cross": cross, "all-ids": all_ids}[a.cmd](a)


if __name__ == "__main__":
    main()
