"""Blind relabeling chunks for the fit-dependent training rows of train_fit.csv.

  relabel_prep.py build     data/built/fit/relabel/{chunks/chunk_NNN.jsonl, index.jsonl, eligibility_decided.jsonl,
                            report.json}

train_fit.csv's labels are rules.label on cards. Where a hard eligibility check decides the row (rules Skip with a
reason from work authorization, enrollment, degree level, degree field, employment type, location or license), the
rules label stays. Every other row's label depends on reading fit (skills, hedged wording, years, seniority, domain,
preferences, thin postings), and careful-reading labels are to replace the rules labels there.

Rows come back with their cards by rebuilding the training pool the way build_fit_train.assemble does (mix3_rows,
twin_rows, venator_rows; train split only) and joining each train_fit.csv row by (prompt, completion); every row must
join by prompt exactly once. The eligibility test is hybrid_rules.rule_verdict on the row's own card, categories
ELIG_CATS. Run it on the rules that built train_fit.csv (rules.py, hybrid_rules.py, build_fit_train.py at the commit
that assembled it): changed rules change the rebuilt pool's gates and labels. The report counts any disagreement.

Selection (seeded): every fit-dependent venator and teaser row; TWINS_N fit-dependent twin rows stratified by
(variant, label); every fit-dependent real mix3 row and MIX3_N fit-dependent synthetic mix3 rows stratified by
label, MIX3_N moved by at most 37 so the total fills whole chunks of CHUNK. The selected rows are shuffled before chunking, so a chunk mixes sources and
labels.

A chunk row holds row_id and the three texts exactly as the prompt shows them to the model (pairs.prompt_parts: the
resume as redacted, the profile as render_profile wrote it, the posting as fitted to the budget). Nothing else: no
label, source, card or reason. row_id = "rl_" + sha1(prompt)[:12], stable across rebuilds and joinable to
train_fit.csv by hashing its prompt column; index.jsonl also carries csv_row (0-based data row of train_fit.csv).

index.jsonl has one row per train_fit.csv row: row_id, csv_row, source, sub, rules_label, rules_completion,
eligibility_decided, hard_cats, selected, chunk. eligibility_decided.jsonl lists the eligibility-decided rows alone.
Reads local files only; calls nothing. Never reads data/built/fresh_test2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

FIT = ROOT / "data/built/fit"
TRAIN_CSV = FIT / "train_fit.csv"
OUT = FIT / "relabel"
CHUNKS = OUT / "chunks"
SEED = 20261021
CHUNK = 75
TWINS_N = 1200
MIX3_N = 1500
# the hard eligibility categories (hybrid_rules.category names), spelled out: HP_CATS is moving (work mode)
ELIG_CATS = frozenset({"work_auth", "enrollment", "degree_level", "degree_field", "employment_type", "location", "license"})

csv.field_size_limit(10**9)


def row_id(prompt: str) -> str:
    return "rl_" + hashlib.sha1(prompt.encode()).hexdigest()[:12]


def write_jsonl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def joined_rows() -> list[dict]:
    """train_fit.csv rows in file order, each with its pool row (source, sub, cand, card)."""
    import build_fit_train as bft
    mix, tw, vr = bft.mix3_rows(), bft.twin_rows(), bft.venator_rows()
    pool = [*mix, *[r for r in tw if r["split"] == "train"], *[r for r in vr if r["split"] == "train"]]
    idx: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(pool):
        idx[r["prompt"]].append(i)
    out = []
    for n, t in enumerate(csv.DictReader(TRAIN_CSV.open(newline=""))):
        hits = idx.get(t["prompt"], [])
        if len(hits) != 1:
            sys.exit(f"train_fit.csv row {n}: {len(hits)} pool rows match (want 1)")
        p = pool[hits[0]]
        out.append({"csv_row": n, "prompt": t["prompt"], "completion": t["completion"], "source": p["source"],
                    "sub": p["sub"], "cand": p["cand"], "card": p["card"], "pool_completion": p["completion"]})
    return out


def classify(rows: list[dict]) -> dict:
    """The rules label is the CSV's; a row is eligibility-decided when that label is Skip and the rules on the row's
    card give a hard reason in ELIG_CATS. Disagreements between the CSV completion, the rebuilt pool's completion and
    rule_verdict are counted (all zero when this runs on the rules that built train_fit.csv)."""
    import hybrid_rules as hr
    mismatch = Counter()
    for r in rows:
        v = hr.rule_verdict(r["cand"], r["card"])
        word = r["completion"].split()[0]
        if v["rule"] != word:
            mismatch[(r["source"], word, v["rule"])] += 1
        if r["pool_completion"] != r["completion"]:
            mismatch[(r["source"], "csv_completion", "pool_completion")] += 1
        hit = sorted(set(v["hard_cats"]) & ELIG_CATS) if v["rule"] == "Skip" and word == "Skip" else []
        r["rules_label"] = word
        r["hard_cats"] = v["hard_cats"]
        r["eligibility_decided"] = bool(hit)
    return {f"{s}:{w}->{v}": n for (s, w, v), n in mismatch.items()}


def stratified(rows: list[dict], n: int, key, rng: random.Random) -> list[dict]:
    """n rows, each stratum's share proportional to its size (largest remainder), seeded."""
    groups: dict = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    n = min(n, len(rows))
    quota = {k: n * len(v) / len(rows) for k, v in groups.items()}
    take = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (-(quota[k] - take[k]), str(k)))[: n - sum(take.values())]:
        take[k] += 1
    out = []
    for k in sorted(groups, key=str):
        g = sorted(groups[k], key=lambda r: r["csv_row"])
        rng.shuffle(g)
        out += g[: take[k]]
    return out


def build(_: argparse.Namespace) -> None:
    from matcher.pairs import prompt_parts
    rows = joined_rows()
    mism = classify(rows)
    rng = random.Random(SEED)
    fit = [r for r in rows if not r["eligibility_decided"]]
    by_src = defaultdict(list)
    for r in fit:
        by_src[r["source"]].append(r)
    picked = [*by_src["venator"], *by_src["teaser"]]
    picked += stratified(by_src["twins"], TWINS_N, lambda r: (r["sub"], r["rules_label"]), rng)
    # every real mix3 row (real postings on extracted cards: where duty phrases become must-skills and seniority and
    # domain fit show), then MIX3_N synthetic ones, moved by at most CHUNK // 2 so the total fills whole chunks
    picked += [r for r in by_src["mix3"] if r["sub"] == "real"]
    rest = (len(picked) + MIX3_N) % CHUNK
    mix_n = MIX3_N - rest if rest <= CHUNK // 2 else MIX3_N + CHUNK - rest
    picked += stratified([r for r in by_src["mix3"] if r["sub"] != "real"], mix_n, lambda r: r["rules_label"], rng)
    order = sorted(picked, key=lambda r: r["csv_row"])
    rng.shuffle(order)
    chunk_of = {}
    for f in CHUNKS.glob("chunk_*.jsonl"):
        f.unlink()
    for c in range(0, len(order), CHUNK):
        name = f"chunk_{c // CHUNK:03d}"
        out = []
        for r in order[c: c + CHUNK]:
            resume, profile, posting = prompt_parts(r["prompt"])
            rid = row_id(r["prompt"])
            chunk_of[rid] = name
            out.append({"row_id": rid, "posting": posting, "resume": resume, "profile": profile})
        write_jsonl(CHUNKS / f"{name}.jsonl", out)
    ids = [row_id(r["prompt"]) for r in rows]
    if len(set(ids)) != len(ids):
        sys.exit("row_id collision")
    index = [{"row_id": row_id(r["prompt"]), "csv_row": r["csv_row"], "source": r["source"], "sub": r["sub"],
              "rules_label": r["rules_label"], "rules_completion": r["completion"],
              "eligibility_decided": r["eligibility_decided"], "hard_cats": r["hard_cats"],
              "selected": row_id(r["prompt"]) in chunk_of, "chunk": chunk_of.get(row_id(r["prompt"]))} for r in rows]
    write_jsonl(OUT / "index.jsonl", index)
    write_jsonl(OUT / "eligibility_decided.jsonl", [x for x in index if x["eligibility_decided"]])
    sel = [x for x in index if x["selected"]]
    rep = {
        "train_rows": len(rows), "rules_vs_csv_label_mismatch": mism,
        "eligibility_decided": len(rows) - len(fit),
        "eligibility_by_source": dict(Counter(x["source"] for x in index if x["eligibility_decided"])),
        "eligibility_by_cat": dict(Counter(c for x in index if x["eligibility_decided"] for c in x["hard_cats"])),
        "fit_dependent_by_source_label": {s: dict(Counter(r["rules_label"] for r in v)) for s, v in by_src.items()},
        "selected": len(sel), "chunks": len(set(chunk_of.values())), "chunk_size": CHUNK, "mix3_synthetic_n": mix_n,
        "selected_by_source_label": {s: dict(Counter(x["rules_label"] for x in sel if x["source"] == s))
                                     for s in ("mix3", "twins", "venator", "teaser")},
        "selected_twins_by_variant_label": dict(Counter(f"{x['sub']}:{x['rules_label']}" for x in sel if x["source"] == "twins")),
        "selected_mix3_by_sub": dict(Counter(x["sub"] for x in sel if x["source"] == "mix3")),
        "selected_label_counts": dict(Counter(x["rules_label"] for x in sel)),
    }
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(fn=build)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
