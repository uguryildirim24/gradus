"""Blind relabel batches for candidate C (eval/protocol_amendment_6b.json C_labels).

  relabel_c_prep.py   the 1,292 Keep/Explore rows of train_B.csv (train3_rows.jsonl kept rows), joined to their
                      cycle-3 label batch rows, in random.Random(20261024) order, 72 a batch
                      -> data/built/fit/cycle3/relabel_C/batches/batch_NNN.jsonl (row_id, posting, resume, profile
                         only), relabel_C/prompt.txt (the clarified labeler prompt) and relabel_C/prep.json
Reads local files only.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C3 = ROOT / "data/built/fit/cycle3"
OUT = C3 / "relabel_C"
SEED = 20261024
SIZE = 72
CLAR = "- A required item the resume names but doesn't show doing is not met."
PROMPT_SHA = "cc1ab478caaeb7bd34dea18325901fb7cb24c0aeb3a754e1f3fcb59008eafab8"


def clarified_prompt() -> str:
    t = json.loads((ROOT / "eval/protocol_amendment_5.json").read_text())["teacher"]
    lp, b = t["labeler_prompt_test3"], t["rubric_bullets"]
    assert hashlib.sha256(lp.encode()).hexdigest() == t["labeler_prompt_test3_sha256"]
    i = lp.find(b) + len(b)
    p = lp[:i] + "\n" + CLAR + lp[i:]
    assert hashlib.sha256(p.encode()).hexdigest() == PROMPT_SHA, "clarified prompt differs from amendment 6b's"
    return p


def main() -> None:
    kept = [json.loads(l) for l in (C3 / "train3_rows.jsonl").open() if l.strip()]
    kept = [r for r in kept if r["kept"]]
    assert len(kept) == 2686
    ids = sorted(r["row_id"] for r in kept if r["label"] in ("Keep", "Explore"))
    assert len(ids) == 1292, len(ids)
    src = {}
    for f in sorted((C3 / "label_batches").glob("batch_*.jsonl")):
        for l in f.open():
            if l.strip():
                b = json.loads(l)
                src[b["row_id"]] = {k: b[k] for k in ("row_id", "posting", "resume", "profile")}
    random.Random(SEED).shuffle(ids)
    (OUT / "batches").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    names = []
    for n, i in enumerate(range(0, len(ids), SIZE), 1):
        p = OUT / "batches" / f"batch_{n:03d}.jsonl"
        p.write_text("".join(json.dumps(src[r], ensure_ascii=False) + "\n" for r in ids[i:i + SIZE]))
        names.append(p.name)
    (OUT / "prompt.txt").write_text(clarified_prompt())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / "prep.json").write_text(json.dumps({
        "rows": len(ids), "seed": SEED, "batch_size": SIZE, "batches": len(names),
        "prompt_sha256": PROMPT_SHA,
        "batch_sha256": {f: sha(OUT / "batches" / f) for f in names}}, indent=1))
    print(len(ids), "rows in", len(names), "batches")


if __name__ == "__main__":
    main()
