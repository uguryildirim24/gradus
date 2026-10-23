"""Cycle-3 training sets for runs A and B (eval/protocol_amendment_5.json, 5b): the rubric teacher's labels on the
4,102 rows of data/built/fit/cycle3/label_batches/.

  build_fit_train3.py            the real build: refuses unless every batch row has a label line
                                 -> data/built/fit/cycle3/train_A.csv, train_B.csv (prompt, completion; same rows, same
                                    order), train3_report.json, train3_rows.jsonl beside them, and a copy of the report
                                    at eval/cycle3_train_csvs.json
  build_fit_train3.py --partial  dry run on whatever labels exist: rows without a label line are dropped as
                                 no_label_yet; everything goes to data/built/fit/cycle3/dryrun/, eval/ is not written

Labels: every line of data/built/fit/cycle3/labels/A/*.jsonl ({row_id, label, reason, confidence}), pooled by row_id
across files (a labeler may write a row of another batch into its file; recorded, and the row's reason is taken from its own
batch's file). A row's label is valid
when its word is Keep, Explore or Skip and its reason is not empty; a row with two lines that disagree on the label is
invalid. Invalid rows are dropped (recorded). Confidence is recorded, never filtered on: amendment 5's teacher has one
label per row and names no confidence cut.

Disjointness ("no row, posting or employer of fresh tests 1, 2, 3 or 4"): a row is dropped when its posting key is a
posting of fresh tests 1-3 (manifests' posting_ids) or fresh test 4 (data/built/fresh_test4/postings.jsonl, read only,
and eval/fresh_test4_manifest.json once it exists), when its employer (the posting's employer field where the row has
one, and the posting's Company line) is an employer of those tests (nearcopy.company_key), or when its posting is a near
copy of a fresh-test posting (6-word shingle containment >= 0.5, as build_fit_train2.fresh_overlap). Fresh test 2's
texts come from the frozen test pool by its manifest's ids.

Label mix (amendment 5 training_label_mix): after the drops, the largest set whose every label share is inside the
band (Keep 12-30%, Explore 22-40%, Skip 40-65%), reached only by dropping rows of labels over their share in that set
(seeded random, SEED, recorded by row id); no row is relabeled. See band_counts.

Prompts:
  A  matcher.pairs.build_prompt(resume, profile, posting): h's INSTRUCTION, the resume redacted, the posting fitted to
     6,000 characters by textclean.fit_posting. For the venator, teaser and twin rows this is byte for byte the prompt
     train_fit.csv holds for the row (checked: origin tN is train_fit.csv row N).
  B  the rubric as the opening of the prompt, then h's layout without INSTRUCTION and with the whole posting:
     SYSTEM_B + "\\n\\n" + "RESUME:\\n...\\n\\nCANDIDATE PROFILE:\\n...\\n\\nJOB POSTING:\\n<whole posting>\\n\\nAnswer:".
     SYSTEM_B is labeler_prompt_test3 from its start through the end of the three rubric bullets (its opening paragraph
     with rubric_bullets under it, both verbatim; sha256 of the bullets checked against amendment 5), nothing after.
     Adaption's raw upload has no system column (SourceColumnMapping: prompt, completion, chosen, rejected, context;
     context is folded into the prompt), so the rubric opens the prompt column and training sees it at the top of the
     single user turn. B must be scored with the same framing (local_eval --framing chat on prompt_b).
  Both: completion = build_fit_train2.careful_completion(label, reason): "Label — reason." on one line, the teacher's
  reason as written. Only prompt and completion are written: train_main's raw upload sends nothing else.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from build_fit_train2 import careful_completion  # noqa: E402
from matcher.nearcopy import company_key, shingles  # noqa: E402
from matcher.pairs import INSTRUCTION, build_prompt  # noqa: E402

FIT = ROOT / "data/built/fit"
C3 = FIT / "cycle3"
BATCHES = C3 / "label_batches"
LABELS = C3 / "labels/A"
INDEX = C3 / "label_index.jsonl"
C3_POSTINGS = C3 / "postings.jsonl"
LEAK_ROWS = FIT / "build/leak_rows.jsonl"
TRAIN_FIT = FIT / "train_fit.csv"
AMEND5 = ROOT / "eval/protocol_amendment_5.json"
AMEND5B = ROOT / "eval/protocol_amendment_5b.json"
FT_MANIFESTS = {"fresh_test1": ROOT / "eval/fresh_test_manifest.json",
                "fresh_test2": ROOT / "eval/fresh_test2_manifest.json",
                "fresh_test3": ROOT / "eval/fresh_test3_manifest.json",
                "fresh_test4": ROOT / "eval/fresh_test4_manifest.json"}
FT_POSTINGS = {"fresh_test1": ROOT / "data/built/fresh_test/postings.jsonl",
               "fresh_test3": ROOT / "data/built/fresh_test3/postings.jsonl",
               "fresh_test4": ROOT / "data/built/fresh_test4/postings.jsonl"}
EVAL_COPY = ROOT / "eval/cycle3_train_csvs.json"
SEED = 20261023  # the band downsample (the batches used 20261021 for the twin draw and 20261022 for the shuffle)
WORDS = ("Keep", "Explore", "Skip")
NEAR = 0.5
A_BUDGET = 6000  # build_prompt's default max_posting_tokens 1500 x 4 characters
WHOLE = 10 ** 12  # a posting budget no posting reaches: fit_posting returns the posting whole


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- the rubric and the two prompt formats

def system_b(amend: dict | None = None) -> str:
    """labeler_prompt_test3 from its start through the end of the rubric bullets, verbatim, checked against the
    amendment's sha256s."""
    t = (amend or json.loads(AMEND5.read_text()))["teacher"]
    bullets, lp = t["rubric_bullets"], t["labeler_prompt_test3"]
    if hashlib.sha256(bullets.encode()).hexdigest() != t["rubric_bullets_sha256"]:
        raise SystemExit("rubric_bullets do not match their sha256 in amendment 5")
    if hashlib.sha256(lp.encode()).hexdigest() != t["labeler_prompt_test3_sha256"]:
        raise SystemExit("labeler_prompt_test3 does not match its sha256 in amendment 5")
    i = lp.find(bullets)
    if i <= 0 or lp.count(bullets) != 1 or not lp[:i].endswith("\n\n") or "\n\n" in lp[:i].rstrip("\n"):
        raise SystemExit("labeler_prompt_test3 is not one opening paragraph with the rubric bullets under it")
    return lp[:i + len(bullets)]


def prompt_a(resume: str, profile: str, posting: str) -> str:
    return build_prompt(resume, profile, posting)


def prompt_b(resume: str, profile: str, posting: str, system: str) -> str:
    """h's layout without INSTRUCTION and with the whole posting, under the rubric."""
    p = build_prompt(resume, profile, posting, max_posting_tokens=WHOLE)
    head = INSTRUCTION + "\n"
    if not p.startswith(head + "RESUME:\n"):
        raise SystemExit("build_prompt no longer opens with INSTRUCTION then RESUME:")
    return system + "\n\n" + p[len(head):]


# ---------------------------------------------------------------- labels

def read_labels(files: list[Path]) -> tuple[dict[str, list[dict]], list[dict]]:
    """Every label line by row_id (with the file it came from). A line that does not parse is recorded; in a
    partial run it may be a line still being written."""
    by_id: dict[str, list[dict]] = defaultdict(list)
    bad = []
    for f in files:
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if not isinstance(r, dict) or not isinstance(r.get("row_id"), str):
                    raise ValueError("no row_id")
            except (ValueError, json.JSONDecodeError) as e:
                bad.append({"file": f.name, "line": n, "error": str(e)[:120]})
                continue
            by_id[r["row_id"]].append(r | {"_file": f.name})
    return by_id, bad


def judge(lines: list[dict]) -> tuple[dict | None, str | None]:
    """The row's label line, or None and why it is invalid."""
    labs = {x.get("label") for x in lines}
    if len(labs) > 1:
        return None, "conflicting_labels"
    x = lines[0]
    if x.get("label") not in WORDS:
        return None, "bad_label_word"
    if not isinstance(x.get("reason"), str) or not " ".join(x["reason"].split()).rstrip(".").strip():
        return None, "empty_reason"
    return x, None


# ---------------------------------------------------------------- disjointness from fresh tests 1-4

def company_line(posting: str) -> str:
    m = re.search(r"^Company: (.+)$", posting, re.M)
    return m.group(1).strip() if m else ""


def fresh_sets() -> dict:
    """Posting ids, employer keys and posting texts of fresh tests 1-4 (test 4 from its postings file, read only,
    and its manifest once written)."""
    ids: dict[str, str] = {}
    emps: dict[str, str] = {}
    texts: dict[str, str] = {}
    for t, p in FT_MANIFESTS.items():
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        ids |= {i: t for i in m.get("posting_ids", [])}
        emps |= {company_key(e): t for e in m.get("employers", []) if company_key(e)}
    for t, p in FT_POSTINGS.items():
        for x in jsonl(p):
            ids.setdefault(x["key"], t)
            for e in (x.get("employer"), x.get("emp_key")):
                if company_key(e or ""):
                    emps.setdefault(company_key(e), t)
            texts[x["key"]] = f"{x.get('title', '')}\n{x.get('text', '')}"
    ft2 = json.loads(FT_MANIFESTS["fresh_test2"].read_text())["posting_ids"]
    from fit_pools import load_pool
    pool = {p["key"]: p for p in load_pool("test")}
    texts |= {i: f"{pool[i].get('title', '')}\n{pool[i].get('text', '')}" for i in ft2}
    return {"ids": ids, "emps": emps, "texts": texts,
            "fresh_test4_manifest": FT_MANIFESTS["fresh_test4"].exists(),
            "counts": {"posting_ids": len(ids), "employers": len(emps), "texts": len(texts)}}


def overlap(rows: list[dict], fs: dict) -> dict[str, dict]:
    """row_id -> what ties it to a fresh test (rows absent are clear)."""
    inv: dict[str, set[str]] = defaultdict(set)
    for k, t in fs["texts"].items():
        for s in shingles(t):
            inv[s].add(k)
    near_of: dict[str, dict | None] = {}
    out = {}
    for r in rows:
        why = {}
        if r["posting_key"] in fs["ids"]:
            why["posting_key"] = fs["ids"][r["posting_key"]]
        hits = sorted({fs["emps"][k] + ":" + k for e in r["employers"] if (k := company_key(e)) in fs["emps"]})
        if hits:
            why["employer"] = hits
        pk = r["posting_key"]
        if pk not in near_of:
            sa = shingles(r["posting"])
            near_of[pk] = None
            if len(sa) >= 5:
                for k, n in Counter(k for s in sa for k in inv.get(s, ())).most_common(1):
                    if n / len(sa) >= NEAR:
                        near_of[pk] = {"fresh_posting": k, "containment": round(n / len(sa), 3)}
        if near_of[pk]:
            why["near_copy"] = near_of[pk]
        if why:
            out[r["row_id"]] = why
    return out


# ---------------------------------------------------------------- the label-mix band

def in_band(counts: dict[str, int], band: dict[str, list[float]]) -> bool:
    n = sum(counts.values())
    return n > 0 and all(band[w][0] <= counts.get(w, 0) / n <= band[w][1] for w in band)


def band_counts(have: dict[str, int], band: dict[str, list[float]]) -> dict[str, int]:
    """How many rows of each label to keep: the largest total N for which some keep[w] <= have[w] puts every share
    inside the band. Each label starts at min(have, its upper bound at N); the rows over N then come off the labels
    most over their band's upper share in the whole set first, each no lower than its lower bound at N. The scarcest
    label (the one that sets N) keeps every row."""
    total = sum(have.values())
    for n in range(total, 0, -1):
        lo = {w: math.ceil(band[w][0] * n - 1e-9) for w in band}
        hi = {w: math.floor(band[w][1] * n + 1e-9) for w in band}
        if any(have.get(w, 0) < lo[w] for w in band) or sum(lo.values()) > n:
            continue
        keep = {w: min(have.get(w, 0), hi[w]) for w in band}
        if sum(keep.values()) < n:
            continue
        extra = sum(keep.values()) - n
        for w in sorted(band, key=lambda w: (have.get(w, 0) / total - band[w][1], w), reverse=True):
            cut = min(extra, keep[w] - lo[w])
            keep[w] -= cut
            extra -= cut
        if extra == 0 and in_band(keep, band):
            return keep
    raise SystemExit(f"no subset of {have} fits the band {band}")


def downsample(rows: list[dict], band: dict[str, list[float]], seed: int) -> tuple[list[dict], dict]:
    have = Counter(r["label"] for r in rows)
    if in_band(have, band):
        return rows, {"needed": False, "seed": seed, "keep": dict(have), "dropped": {}}
    keep = band_counts(have, band)
    rng = random.Random(seed)
    kept_ids: set[str] = set()
    dropped: dict[str, list[str]] = {}
    for w in WORDS:
        ids = sorted(r["row_id"] for r in rows if r["label"] == w)
        k = set(rng.sample(ids, keep[w])) if keep[w] < len(ids) else set(ids)
        kept_ids |= k
        if len(k) < len(ids):
            dropped[w] = sorted(set(ids) - k)
    return [r for r in rows if r["row_id"] in kept_ids], {"needed": True, "seed": seed, "keep": keep,
                                                          "dropped": dropped}


def mix(rows: list[dict]) -> dict:
    c = Counter(r["label"] for r in rows)
    n = sum(c.values())
    return {"n": n, "counts": {w: c.get(w, 0) for w in WORDS},
            "shares": {w: round(c.get(w, 0) / n, 4) if n else None for w in WORDS}}


# ---------------------------------------------------------------- build

def load_rows() -> tuple[list[dict], dict[str, str]]:
    """Batch rows in batch order with their index entry and employers; row_id -> batch file."""
    idx = {r["row_id"]: r for r in jsonl(INDEX)}
    emp_new = {p["key"]: p.get("employer", "") for p in jsonl(C3_POSTINGS)}
    emp_old = {r["id"]: r.get("employer", "") for r in jsonl(LEAK_ROWS)}
    rows, where = [], {}
    for f in sorted(BATCHES.glob("batch_*.jsonl")):
        for b in jsonl(f):
            if b["row_id"] in where:
                raise SystemExit(f"{b['row_id']} in two batches")
            where[b["row_id"]] = f.name
            i = idx[b["row_id"]]
            if i.get("title_copy"):
                raise SystemExit(f"{b['row_id']}: a title-copy row is in the batches")
            emp = emp_new.get(i["posting_key"], "") if i["source"] == "new" else emp_old.get(i["origin"], "")
            rows.append(b | {"source": i["source"], "origin": i["origin"], "posting_key": i["posting_key"],
                             "employers": [e for e in (emp, company_line(b["posting"])) if e]})
    if set(idx) != set(where):
        raise SystemExit("label_index.jsonl and the batches hold different rows")
    return rows, where


def train_fit_prompts() -> list[str]:
    csv.field_size_limit(10 ** 9)
    with TRAIN_FIT.open(newline="") as f:
        return [r["prompt"] for r in csv.DictReader(f)]


def write_csv(p: Path, rows: list[tuple[str, str]]) -> None:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["prompt", "completion"])
    w.writerows(rows)
    p.write_bytes(buf.getvalue().encode())


def stats(xs: list[int]) -> dict:
    xs = sorted(xs)
    return {"min": xs[0], "median": xs[len(xs) // 2], "p99": xs[min(len(xs) - 1, int(0.99 * len(xs)))],
            "max": xs[-1]} if xs else {}


def build(partial: bool, out_dir: Path = C3, eval_copy: Path | None = EVAL_COPY, seed: int = SEED,
          fresh: dict | None = None) -> dict:
    amend = json.loads(AMEND5.read_text())
    band = amend["training_label_mix"]["band"]
    system = system_b(amend)
    rows, where = load_rows()
    labels, unparsed = read_labels(sorted(LABELS.glob("*.jsonl")))
    missing = [r["row_id"] for r in rows if r["row_id"] not in labels]
    if missing and not partial:
        raise SystemExit(f"{len(missing)} batch rows have no label line yet (first {missing[:3]}); "
                         "use --partial for a dry run")
    if unparsed and not partial:
        raise SystemExit(f"{len(unparsed)} label lines do not parse: {unparsed[:3]}")
    unknown = sorted(set(labels) - set(where))
    cross = sorted({x["row_id"] for xs in labels.values() for x in xs
                    if x["row_id"] in where and x["_file"] != where[x["row_id"]]})
    dupes = sorted(k for k, xs in labels.items() if len(xs) > 1)

    fate: dict[str, str] = {}
    invalid: dict[str, list[str]] = defaultdict(list)
    labeled = []
    for r in rows:
        if r["row_id"] not in labels:
            fate[r["row_id"]] = "no_label_yet"
            continue
        # the line from the row's own batch file first: its reason is the one written
        x, why = judge(sorted(labels[r["row_id"]], key=lambda x: x["_file"] != where[r["row_id"]]))
        if why:
            fate[r["row_id"]] = why
            invalid[why].append(r["row_id"])
            continue
        labeled.append(r | {"label": x["label"], "reason": x["reason"], "confidence": x.get("confidence")})

    fs = fresh if fresh is not None else fresh_sets()
    ties = overlap(labeled, fs)
    for rid in ties:
        fate[rid] = "fresh_test_overlap"
    clear = [r for r in labeled if r["row_id"] not in ties]
    kept, ds = downsample(clear, band, seed)
    for w, ids in ds["dropped"].items():
        for rid in ids:
            fate[rid] = "band_downsample"
    if not in_band(Counter(r["label"] for r in kept), band):
        raise SystemExit("the kept set is outside the band")

    tf = train_fit_prompts()
    a_rows, b_rows, side = [], [], []
    a_match = Counter()
    a_cut = 0
    for n, r in enumerate(kept):
        pa = prompt_a(r["resume"], r["profile"], r["posting"])
        pb = prompt_b(r["resume"], r["profile"], r["posting"], system)
        comp = careful_completion(r["label"], r["reason"])
        if r["source"] != "new":
            ok = r["origin"].startswith("t") and tf[int(r["origin"][1:])] == pa
            a_match[f"{r['source']}:{'same' if ok else 'differs'}"] += 1
        a_cut += len(r["posting"].strip()) > A_BUDGET
        if not pb.startswith(system + "\n\nRESUME:\n") or not pb.endswith("\n\nAnswer:"):
            raise SystemExit(f"{r['row_id']}: B prompt layout")
        if comp.split()[0] != r["label"] or "\n" in comp:
            raise SystemExit(f"{r['row_id']}: completion {comp!r}")
        a_rows.append((pa, comp))
        b_rows.append((pb, comp))
        side.append({"row_id": r["row_id"], "csv_row": n, "source": r["source"], "origin": r["origin"],
                     "posting_key": r["posting_key"], "label": r["label"], "confidence": r["confidence"],
                     "completion": comp})
    if any(k.endswith(":differs") for k in a_match):
        raise SystemExit(f"A prompts differ from train_fit.csv's for some old rows: {dict(a_match)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pa_csv, pb_csv = out_dir / "train_A.csv", out_dir / "train_B.csv"
    write_csv(pa_csv, a_rows)
    write_csv(pb_csv, b_rows)
    kept_ids = {s["row_id"] for s in side}
    with (out_dir / "train3_rows.jsonl").open("w") as f:
        for s in side:
            f.write(json.dumps(s | {"kept": True}, ensure_ascii=False) + "\n")
        for r in rows:
            if r["row_id"] not in kept_ids:
                f.write(json.dumps({"row_id": r["row_id"], "source": r["source"], "origin": r["origin"],
                                    "posting_key": r["posting_key"], "kept": False, "why": fate[r["row_id"]]})
                        + "\n")

    conf = Counter(r["confidence"] for r in labeled)
    rep = {
        "what": "cycle-3 training sets for runs A and B (eval/protocol_amendment_5.json, 5b), scripts/build_fit_train3.py"
                + (" -- DRY RUN on partial labels, not for upload" if partial else ""),
        "partial": partial,
        "amendments_sha256": {"eval/protocol_amendment_5.json": sha256(AMEND5),
                              "eval/protocol_amendment_5b.json": sha256(AMEND5B)},
        "inputs_sha256": {rel(p): sha256(p) for p in [INDEX, *sorted(BATCHES.glob("*.jsonl"))]},
        "labels_sha256": {rel(p): sha256(p) for p in sorted(LABELS.glob("*.jsonl"))},
        "batch_rows": len(rows),
        "label_lines": {"row_ids": len(labels), "unparsed_lines": unparsed, "row_ids_in_no_batch": unknown,
                        "rows_labeled_in_another_batch_file": cross, "rows_with_several_lines": dupes},
        "labeled_valid": len(labeled),
        "dropped": {
            "no_label_yet": sum(v == "no_label_yet" for v in fate.values()),
            "invalid": {k: v for k, v in invalid.items()},
            "fresh_test_overlap": ties,
            "band_downsample": {w: len(v) for w, v in ds["dropped"].items()},
        },
        "confidence_of_valid_labels": dict(conf),
        "confidence_note": "no confidence filter: amendment 5's teacher is one label per row, no cut named",
        "by_source_valid": {s: dict(Counter(r["label"] for r in labeled if r["source"] == s))
                            for s in sorted({r["source"] for r in labeled})},
        "fresh_tests_checked": fs["counts"] | {"fresh_test4_manifest_read": fs["fresh_test4_manifest"]},
        "band": band,
        "mix_before_band": mix(clear),
        "mix_before_band_in_band": in_band(Counter(r["label"] for r in clear), band),
        "downsample": {"needed": ds["needed"], "seed": seed, "keep": ds["keep"],
                       "rule": "largest set inside the band; only labels over their band cut, seeded random by row id",
                       "dropped_row_ids": ds["dropped"]},
        "mix_after_band": mix(kept),
        "by_source_kept": {s: dict(Counter(r["label"] for r in kept if r["source"] == s))
                           for s in sorted({r["source"] for r in kept})},
        "rows": len(kept),
        "row_parity": "train_A.csv and train_B.csv hold the same rows in the same order with the same completions "
                      "(train3_rows.jsonl gives row_id by csv_row)",
        "prompt_a": "matcher.pairs.build_prompt(resume, profile, posting): INSTRUCTION, resume redacted, posting fitted "
                    "to 6,000 characters (textclean.fit_posting)",
        "prompt_a_vs_train_fit": dict(a_match),
        "prompt_a_postings_cut": a_cut,
        "prompt_b": "SYSTEM_B + '\\n\\n' + build_prompt's text after INSTRUCTION with the whole posting "
                    "(RESUME:/CANDIDATE PROFILE:/JOB POSTING:/Answer:)",
        "prompt_b_system": system,
        "prompt_b_system_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "prompt_b_system_note": "labeler_prompt_test3's opening paragraph with rubric_bullets under it, both verbatim, "
                                "nothing after the bullets; the opening sentence read as the opening paragraph",
        "prompt_b_where": "Adaption's raw upload has no system column, so the rubric opens the prompt column (the top "
                          "of the single user turn in chat format); score B with local_eval --framing chat",
        "completion": "build_fit_train2.careful_completion: 'Label — reason.' on one line",
        "chars": {"prompt_a": stats([len(p) for p, _ in a_rows]), "prompt_b": stats([len(p) for p, _ in b_rows]),
                  "completion": stats([len(c) for _, c in a_rows])},
        "files": {rel(p): {"sha256": sha256(p), "md5": md5(p), "rows": len(kept)}
                  for p in (pa_csv, pb_csv)},
        "rows_file": rel(out_dir / "train3_rows.jsonl"),
        "rows_file_sha256": sha256(out_dir / "train3_rows.jsonl"),
    }
    (out_dir / "train3_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    if eval_copy and not partial:
        eval_copy.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partial", action="store_true", help="dry run on the labels so far -> data/built/fit/cycle3/dryrun/")
    a = ap.parse_args()
    rep = build(a.partial, out_dir=C3 / "dryrun" if a.partial else C3, eval_copy=None if a.partial else EVAL_COPY)
    print(json.dumps({k: rep[k] for k in ("batch_rows", "labeled_valid", "rows", "mix_before_band", "mix_after_band",
                                         "prompt_a_vs_train_fit", "files")}
                     | {"dropped": {k: (len(v) if isinstance(v, dict) and k == "fresh_test_overlap" else v)
                                    for k, v in rep["dropped"].items()}}, indent=1))


if __name__ == "__main__":
    main()
