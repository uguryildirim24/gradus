"""Run c's own reasons on the rows it got wrong: pick the rows, then join the Colab outputs into one file.

  diagnosis_reasons.py subset --result RESULT.json --rows SOURCE.jsonl --set NAME [--correct 150] [--seed 0] --out SUB.jsonl
      The decidable rows RESULT.json got wrong, then (with --correct N) N rows it got right, sampled within variant kind
      in proportion to the errors' kind mix (so each error kind has right answers of the same kind beside it; a kind
      with fewer right rows than its share gives all it has and the rest goes to the other kinds by the same mix).
      Writes the source rows (prompt and all) in that order, each with diag_set and diag_sample (error or correct),
      ready for colab_eval.py --dev SUB.jsonl --gen <at least its rows>.
  diagnosis_reasons.py assemble --pair SUB.jsonl=RESULT.json=COLAB_OUT.json [--pair ...] --out c_reasons.jsonl
      One line per subset row: pair_id, set, sample, gold, pred (from RESULT.json, the run that defined the errors),
      scores (Keep, Explore, Skip from that run), generation (the greedy "Label — reason" from COLAB_OUT.json),
      gen_label (its first word), pred_now and scores_now (the Colab rerun), max_score_diff, gold_completion,
      variant_kind, base_rq_id, prompt_tokens. Refuses if a subset row has no generation.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def allocate(want: Counter, have: Counter, total: int) -> dict:
    """Spread `total` over kinds in proportion to `want`, never more than `have` per kind (largest remainder),
    handing what a capped kind cannot take to the others by the same proportions."""
    got = {k: 0 for k in want}
    left = min(total, sum(have[k] for k in want if want[k]))
    while left > 0:
        open_ = [k for k in want if want[k] and got[k] < have[k]]
        if not open_:
            break
        w = sum(want[k] for k in open_)
        share = {k: left * want[k] / w for k in open_}
        add = {k: min(int(share[k]), have[k] - got[k]) for k in open_}
        rest = left - sum(add.values())
        for k in sorted(open_, key=lambda k: (-(share[k] - int(share[k])), str(k))):
            if rest <= 0:
                break
            if got[k] + add[k] < have[k]:
                add[k] += 1
                rest -= 1
        if not sum(add.values()):
            break
        for k in open_:
            got[k] += add[k]
        left = total - sum(got.values())
    return got


def subset_rows(result_rows: list[dict], n_correct: int, seed: int) -> list[tuple[str, str]]:
    """(pair_id, error|correct) in output order: the decidable errors in file order, then the sampled right rows."""
    rows = [r for r in result_rows if r.get("visible", True)]
    err = [r for r in rows if r["gold"] != r["pred"]]
    ok = [r for r in rows if r["gold"] == r["pred"]]
    out = [(r["pair_id"], "error") for r in err]
    if n_correct:
        by_kind = defaultdict(list)
        for r in ok:
            by_kind[r.get("variant_kind")].append(r)
        want = Counter(r.get("variant_kind") for r in err)
        take = allocate(want, Counter({k: len(v) for k, v in by_kind.items()}), n_correct)
        rng = random.Random(seed)
        picked = set()
        for k in sorted(take, key=str):
            picked |= {r["pair_id"] for r in rng.sample(by_kind[k], take[k])}
        out += [(r["pair_id"], "correct") for r in ok if r["pair_id"] in picked]
    return out


def subset(a: argparse.Namespace) -> None:
    res = json.loads(Path(a.result).read_text())
    src = {}
    for line in open(a.rows):
        if line.strip():
            r = json.loads(line)
            assert r["pair_id"] not in src, f"pair_id {r['pair_id']} twice in {a.rows}"
            src[r["pair_id"]] = r
    picked = subset_rows(res["rows"], a.correct, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for pid, kind in picked:
            f.write(json.dumps(src[pid] | {"diag_set": a.set, "diag_sample": kind}) + "\n")
    c = Counter(k for _, k in picked)
    print(f"wrote {a.out}: {len(picked)} rows ({dict(c)})")


def assemble_rows(sub: list[dict], ref: dict, now: dict) -> list[dict]:
    ref_rows = {r["pair_id"]: r for r in ref["rows"]}
    now_rows = {r["pair_id"]: r for r in now["rows"]}
    gens = {g["pair_id"]: g["text"] for g in now["generations"]}
    missing = [r["pair_id"] for r in sub if r["pair_id"] not in gens or r["pair_id"] not in now_rows]
    if missing:
        raise SystemExit(f"{len(missing)} subset rows have no generation or score in the rerun, e.g. {missing[:3]}")
    out = []
    for s in sub:
        pid = s["pair_id"]
        r, n, g = ref_rows[pid], now_rows[pid], gens[pid].strip()
        out.append({"pair_id": pid, "set": s["diag_set"], "sample": s["diag_sample"], "gold": r["gold"],
                    "pred": r["pred"], "scores": r["scores"], "generation": g,
                    "gen_label": g.split()[0].strip("—-:,.") if g else "",
                    "pred_now": n["pred"], "scores_now": n["scores"],
                    "max_score_diff": round(max(abs(x - y) for x, y in zip(r["scores"], n["scores"])), 4),
                    "gold_completion": s.get("completion"), "variant_kind": s.get("variant_kind"),
                    "base_rq_id": r.get("base_rq_id"), "prompt_tokens": r.get("prompt_tokens")})
    return out


def assemble(a: argparse.Namespace) -> None:
    rows = []
    for spec in a.pair:
        sub_p, ref_p, now_p = spec.split("=")
        sub = [json.loads(line) for line in open(sub_p) if line.strip()]
        rows += assemble_rows(sub, json.loads(Path(ref_p).read_text()), json.loads(Path(now_p).read_text()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    c = Counter((r["set"], r["sample"]) for r in rows)
    print(f"wrote {a.out}: {len(rows)} rows {dict(c)}")
    print("pred_now differs from pred:", sum(r["pred_now"] != r["pred"] for r in rows),
          "| gen_label differs from pred_now:", sum(r["gen_label"] != r["pred_now"] for r in rows),
          "| max score diff:", max((r["max_score_diff"] for r in rows), default=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subset")
    s.add_argument("--result", required=True)
    s.add_argument("--rows", required=True)
    s.add_argument("--set", required=True)
    s.add_argument("--correct", type=int, default=0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    m = sub.add_parser("assemble")
    m.add_argument("--pair", action="append", required=True, help="SUBSET.jsonl=REFERENCE_RESULT.json=RERUN.json")
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    {"subset": subset, "assemble": assemble}[a.cmd](a)


if __name__ == "__main__":
    main()
