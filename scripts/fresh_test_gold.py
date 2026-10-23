"""Assemble the fresh-test gold from two labelers (A, B) and the adjudicator.

Agreed rows take the shared label; disagreements take the adjudicator's gold.
Writes data/built/fresh_test/gold.jsonl, updates agreement.json with adjudicator
side counts and gold label mix, and writes eval/<name>_gold_manifest.json (arg: fresh_test, fresh_test2, ... or a path to the set's directory).

Flags (off by default, so tests 1-3 assemble exactly as before):
  --own-batch    a labeler's row counts only from the labels/<X>/batch_NN.jsonl file whose number matches the row's
                 label_batches/batch_NN.jsonl. Set 4's labeler A has 8 stray rows in its batch_09 file that belong to
                 batch_05 and are already labeled there; the stray copies are ignored (one of them flips Skip to
                 Explore) and listed in the manifest.
  --hash-inputs  add sha256 of every label and adjudication file to the manifest.
"""
import hashlib, json, glob, os, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# usage: fresh_test_gold.py [fresh_test|fresh_test2]; default fresh_test
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
OWN_BATCH = "--own-batch" in sys.argv
HASH_INPUTS = "--hash-inputs" in sys.argv
_target = _args[0] if _args else "fresh_test"
if os.sep in _target:
    FT = os.path.abspath(_target)
    NAME = os.path.basename(FT)
else:
    NAME = _target
    FT = os.path.join(ROOT, "data/built", NAME)


def jl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def by_id(paths, key="row_id"):
    out = {}
    for p in sorted(paths):
        for r in jl(p):
            out[r[key]] = r
    return out


def mix(labels):
    c = Counter(labels)
    return {k: c.get(k, 0) for k in ("Keep", "Explore", "Skip")}


def by_id_own_batch(side):
    """Labels for `side`, keeping a row only from the batch file that owns it; returns (labels, ignored_strays)."""
    out, strays = {}, []
    for p in sorted(glob.glob(f"{FT}/labels/{side}/batch_*.jsonl")):
        owned = {r["row_id"] for r in jl(os.path.join(FT, "label_batches", os.path.basename(p)))}
        for r in jl(p):
            if r["row_id"] not in owned:
                strays.append({"side": side, "file": os.path.basename(p), "row_id": r["row_id"], "label": r["label"]})
                continue
            assert r["row_id"] not in out, f"{side}: {r['row_id']} labeled twice in its own batch"
            out[r["row_id"]] = r
    return out, strays


rows = {r["row_id"]: r for r in jl(f"{FT}/rows.jsonl")}
strays = []
if OWN_BATCH:
    A, sa = by_id_own_batch("A")
    B, sb = by_id_own_batch("B")
    strays = sa + sb
else:
    A = by_id(glob.glob(f"{FT}/labels/A/*.jsonl"))
    B = by_id(glob.glob(f"{FT}/labels/B/*.jsonl"))
adj = by_id(glob.glob(f"{FT}/adjudicate/chunk_*.gold.jsonl"))

gold, missing = [], []
for rid in rows:
    a, b = A.get(rid), B.get(rid)
    if not a or not b:
        missing.append({"row_id": rid, "why": "no label from " + ("A and B" if not a and not b else "A" if not a else "B")})
        continue
    la, lb = a["label"], b["label"]
    if la == lb:
        gold.append({"row_id": rid, "gold": la, "source": "agree", "label_A": la, "label_B": lb, "adjudicator_side": None})
    elif rid in adj:
        g = adj[rid]
        gold.append({"row_id": rid, "gold": g["gold"], "source": "adjudicated", "label_A": la, "label_B": lb,
                     "adjudicator_side": g.get("sided_with")})
    else:
        missing.append({"row_id": rid, "why": "disagreement without adjudication"})

with open(f"{FT}/gold.jsonl", "w") as f:
    for g in gold:
        f.write(json.dumps(g) + "\n")
sha = hashlib.sha256(open(f"{FT}/gold.jsonl", "rb").read()).hexdigest()

n_both = [r for r in rows if r in A and r in B]
raw = sum(A[r]["label"] == B[r]["label"] for r in n_both) / len(n_both)
adj_rows = [g for g in gold if g["source"] == "adjudicated"]
sides = Counter(g["adjudicator_side"] for g in adj_rows)
passing = {r for r, v in rows.items() if v["design"].get("eligibility_pass")}
gold_mix = mix(g["gold"] for g in gold)
gold_mix_pass = mix(g["gold"] for g in gold if g["row_id"] in passing)

ag_path = f"{FT}/agreement.json"
ag = json.load(open(ag_path))
ag["adjudicator_sided_with"] = {"A": sides.get("A", 0), "B": sides.get("B", 0), "neither": sides.get("neither", 0)}
ag["n_adjudicated"] = len(adj_rows)
ag["gold_label_mix"] = gold_mix
ag["gold_label_mix_eligibility_pass"] = gold_mix_pass
ag["n_gold"] = len(gold)
ag["missing_rows"] = missing
json.dump(ag, open(ag_path, "w"), indent=1)

man = {"n_gold": len(gold), "gold_sha256": sha, "raw_agreement": ag["raw_agreement"], "kappa": ag["kappa"],
       "n_rows_both": ag["n_rows_both"], "adjudicated_n": len(adj_rows),
       "adjudicator_sided_with": ag["adjudicator_sided_with"], "label_mix": gold_mix,
       "label_mix_eligibility_pass": gold_mix_pass, "n_eligibility_pass": sum(g["row_id"] in passing for g in gold),
       "missing_rows": [m["row_id"] for m in missing]}
if OWN_BATCH:
    man["ignored_stray_label_rows"] = strays
if HASH_INPUTS:
    def h(path):
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
    rel = lambda p: os.path.relpath(p, ROOT)
    man["input_sha256"] = {rel(p): h(p) for p in sorted(glob.glob(f"{FT}/labels/A/*.jsonl") + glob.glob(f"{FT}/labels/B/*.jsonl")
                                                         + glob.glob(f"{FT}/adjudicate/*.jsonl") + [f"{FT}/rows.jsonl", f"{FT}/agreement.json"])}
os.makedirs(f"{ROOT}/eval", exist_ok=True)
json.dump(man, open(f"{ROOT}/eval/{NAME}_gold_manifest.json", "w"), indent=1)
print(json.dumps(man))
