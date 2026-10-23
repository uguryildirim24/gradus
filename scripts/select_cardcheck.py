"""Choose the real postings whose extracted cards get checked by hand, and write them as 4 batch files.

Train side: the real_train train postings whose rows best teach the ask gap (data/models/main_4b/diagnosis_c):
the most Keep or Explore rows with an ask gap above 4, ties by posting id. The ask gap is the diagnosis's measure
(quant/askgap.py: its lexicon minus names in more than 25% of shown postings, gap_req_hard + gap_req_soft), on the
rendered rows in train.jsonl. train.jsonl, not screened_train.jsonl, because the screen drops rows whose extracted
card is wrong, and those are the rows a checked card can relabel.

Dev side: real dev postings (real_dev.jsonl) without a checked card (data/built/audit/fresh_checked_cards.jsonl),
enough to bring checked dev postings to about 70, picked greedily so the checked set covers the variant kinds
evenly: each step takes the posting whose rows add most to the kinds the checked set has fewest rows of
(sum of 1 / (1 + rows of that kind already checked)), ties by posting id.

Each batch entry: posting_id, split, the raw posting's file and key (data/raw/ats/postings.jsonl, simplify_id),
title, company, the extracted card (data/built/cards/extracted_real_cards_v2.jsonl) and the selection numbers.
Train and dev postings are dealt round-robin over the batches, so each batch gets an even share of both.

Deterministic; reads local files only, loads no model, calls nothing.
Run: .venv/bin/python scripts/select_cardcheck.py [--train-n 120] [--dev-target 70] [--batches 4]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RT = ROOT / "data/built/real_train"
OUT = RT / "cardcheck"
V2 = ROOT / "data/built/cards/extracted_real_cards_v2.jsonl"
CHECKED = ROOT / "data/built/audit/fresh_checked_cards.jsonl"
RAW = ROOT / "data/raw/ats/postings.jsonl"
QUANT = ROOT / "data/models/main_4b/diagnosis_c/quant"
GAP_OVER = 4


def read_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def gold(r: dict) -> str:
    return r["completion"].split()[0]


# ---------------------------------------------------------------- ask gap (the diagnosis's measure)
def ask_gaps(rows: list[dict]) -> dict[str, int]:
    """pair_id -> ask gap, with askgap.py's lexicon and generic list (as real_train_check/gap_real_train.py)."""
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(QUANT))
    import askgap as A  # noqa: E402
    from matcher.pairs import prompt_parts  # noqa: E402

    lex = A.lexicon()
    base = {s: A.rj(p) for s, p in (("test", A.RT / "test.jsonl"), ("dev", A.TB / "dev.jsonl"),
                                    ("dev_cross", A.TB / "dev_cross.jsonl"), ("train", A.TB / "train.jsonl"))}
    posts = {}
    for rs in base.values():
        for r in rs:
            posts.setdefault(r["rq_id"], prompt_parts(r["prompt"])[2])
    df = Counter(t for p in posts.values() for t in A.grams(p) & lex)
    lex -= {t for t, c in df.items() if c > 0.25 * len(posts)} | {"history"}
    out = {}
    for r in rows:
        m = re.search(r"Years of work experience: (\d+)", r["prompt"])
        c = re.search(r"Lives in: ([^\n]+)", r["prompt"])
        g = A.row_feats(r["prompt"], lex, int(m.group(1)) if m else None, c.group(1) if c else None)
        out[r["pair_id"]] = g["gap_req_hard"] + g["gap_req_soft"]
    return out


# ---------------------------------------------------------------- selection
def train_scores(rows: list[dict], gaps: dict[str, int], over: int = GAP_OVER) -> dict[str, dict]:
    """posting -> rows, Keep/Explore rows, Keep/Explore rows with ask gap > over."""
    s: dict[str, dict] = {}
    for r in rows:
        d = s.setdefault(r["base_rq_id"], {"rows": 0, "keep_explore": 0, "keep_explore_gap": 0})
        d["rows"] += 1
        if gold(r) in ("Keep", "Explore"):
            d["keep_explore"] += 1
            d["keep_explore_gap"] += gaps[r["pair_id"]] > over
    return s


def pick_train(scores: dict[str, dict], n: int) -> list[str]:
    ranked = sorted(scores, key=lambda i: (-scores[i]["keep_explore_gap"], i))
    return [i for i in ranked[:n] if scores[i]["keep_explore_gap"] > 0]


def pick_dev(rows: list[dict], checked: set[str], target: int) -> list[str]:
    """Greedy: add the unchecked dev posting whose rows most fill the variant kinds the checked set has least of."""
    kinds: dict[str, Counter] = {}
    for r in rows:
        kinds.setdefault(r["base_rq_id"], Counter())[r["variant_kind"]] += 1
    cov = Counter()
    for i in checked & kinds.keys():
        cov.update(kinds[i])
    have = len(checked & kinds.keys())
    left = sorted(kinds.keys() - checked)
    picked = []
    while have + len(picked) < target and left:
        best = max(left, key=lambda i: sum(c / (1 + cov[k]) for k, c in kinds[i].items()))  # first max: lowest id
        picked.append(best)
        left.remove(best)
        cov.update(kinds[best])
    return picked


def deal(train: list[str], dev: list[str], k: int) -> list[list[tuple[str, str]]]:
    """Round-robin each side over k batches; each batch sorted by posting id so train and dev interleave."""
    out = [[] for _ in range(k)]
    seq = [(i, "train") for i in train] + [(i, "dev") for i in dev]  # dev picks up where train left off
    for j, it in enumerate(seq):
        out[j % k].append(it)
    return [sorted(b) for b in out]


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=120)
    ap.add_argument("--dev-target", type=int, default=70)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)

    train_rows = read_jsonl(RT / "train.jsonl")
    dev_rows = read_jsonl(RT / "real_dev.jsonl")
    screened = {r["pair_id"] for r in read_jsonl(RT / "screened_train.jsonl")}
    checked = {r["id"] for r in read_jsonl(CHECKED)}
    gaps = ask_gaps(train_rows + dev_rows)

    tscore = train_scores(train_rows, gaps)
    sscore = train_scores([r for r in train_rows if r["pair_id"] in screened], gaps)
    dscore = train_scores(dev_rows, gaps)
    train = pick_train(tscore, a.train_n)
    dev = pick_dev(dev_rows, checked, a.dev_target)

    cards = {r["id"]: r["card"] for r in read_jsonl(V2)}
    raw = {"ats_" + p["simplify_id"]: p for p in read_jsonl(RAW)}
    dkinds: dict[str, Counter] = {}
    for r in dev_rows:
        dkinds.setdefault(r["base_rq_id"], Counter())[r["variant_kind"]] += 1

    def entry(i: str, side: str) -> dict:
        p = raw[i]
        sc = (tscore if side == "train" else dscore)[i]
        e = {"posting_id": i, "split": side, "raw_path": str(RAW.relative_to(ROOT)), "raw_key": p["simplify_id"],
             "title": p.get("title"), "company": p.get("company"), "extracted_card": cards[i],
             "rows": sc["rows"], "keep_explore_rows": sc["keep_explore"],
             f"keep_explore_rows_gap_over_{GAP_OVER}": sc["keep_explore_gap"]}
        if side == "train":
            e[f"screened_keep_explore_rows_gap_over_{GAP_OVER}"] = sscore.get(i, {}).get("keep_explore_gap", 0)
        else:
            e["variant_kinds"] = dict(sorted(dkinds[i].items()))
        return e

    a.out.mkdir(parents=True, exist_ok=True)
    batches = deal(train, dev, a.batches)
    for b, items in enumerate(batches, 1):
        with open(a.out / f"batch_{b}.json", "w") as f:
            json.dump([entry(i, s) for i, s in items], f, indent=1, ensure_ascii=False)
            f.write("\n")

    cut = tscore[train[-1]]["keep_explore_gap"] if train else None
    cov_before, cov_after = Counter(), Counter()
    for i in checked & dkinds.keys():
        cov_before.update(dkinds[i])
    cov_after.update(cov_before)
    for i in dev:
        cov_after.update(dkinds[i])
    rep = {
        "train_postings": len(train), "train_pool": len(tscore),
        "train_cut_keep_explore_gap_rows": cut,
        "train_postings_at_cut_left_out": sum(1 for i, s in tscore.items()
                                              if s["keep_explore_gap"] == cut and i not in train),
        "train_rows_selected": sum(tscore[i]["rows"] for i in train),
        "train_keep_explore_gap_rows_selected": sum(tscore[i]["keep_explore_gap"] for i in train),
        "train_keep_explore_gap_rows_pool": sum(s["keep_explore_gap"] for s in tscore.values()),
        "dev_checked_before": len(checked & dkinds.keys()), "dev_added": len(dev),
        "dev_checked_after": len(checked & dkinds.keys()) + len(dev), "dev_pool": len(dkinds),
        "dev_rows_checked_before": sum(cov_before.values()), "dev_rows_checked_after": sum(cov_after.values()),
        "dev_kinds": len({k for c in dkinds.values() for k in c}),
        "dev_kinds_covered_before": len(cov_before), "dev_kinds_covered_after": len(cov_after),
        "dev_min_rows_per_kind_after": min(cov_after.values()) if cov_after else 0,
        "batches": [{"file": f"batch_{b}.json", "train": sum(s == "train" for _, s in it),
                     "dev": sum(s == "dev" for _, s in it)} for b, it in enumerate(batches, 1)],
    }
    with open(a.out / "selection_report.json", "w") as f:
        json.dump(rep, f, indent=1)
        f.write("\n")
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    main()
