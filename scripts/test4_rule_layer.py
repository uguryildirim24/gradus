"""What the eligibility layer adds on sealed fresh test 4: run B read with the decider, alone, against the shipped
B + H'' without work mode. Reported, not part of any pass bar. CPU only: it reads the stored per-row file.

    uv run python scripts/test4_rule_layer.py [--rows data/models/c3_eval/test4/test4_rows.jsonl]
        -> eval/test4_rule_layer.json

"B alone" is matcher.decider on B's stored float32 label scores for every row. "B + H''" is the committed
label (systems["B+H''_without_work_mode|decider"]). Intervals use fit_eval_score's two-way bootstrap (postings,
then rows; 2,000 draws, seed 20261021), the draws the committed result used. The per-row file must match the
sha256 eval/fresh_test4_result.json pins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import fit_eval_score as fe  # noqa: E402
from matcher import decider  # noqa: E402

SHIPPED = "B+H''_without_work_mode|decider"
RESULT = ROOT / "eval/fresh_test4_result.json"
OUT = ROOT / "eval/test4_rule_layer.json"


def keeps(gold: list[str], pred: list[str]) -> dict:
    return {"keeps_found": sum(g == p == "Keep" for g, p in zip(gold, pred)),
            "wrong_keeps": sum(p == "Keep" and g != "Keep" for g, p in zip(gold, pred)),
            "predicted_keeps": sum(p == "Keep" for p in pred)}


def main() -> None:
    ap = argparse.ArgumentParser()
    pinned = json.loads(RESULT.read_text())["per_row_file"]
    ap.add_argument("--rows", default=str(ROOT / pinned["path"]))
    a = ap.parse_args()
    path = Path(a.rows)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != pinned["sha256"]:
        sys.exit(f"{path}: sha256 {digest} is not the one eval/fresh_test4_result.json pins ({pinned['sha256']})")
    rows = [json.loads(line) for line in path.open()]
    gold = [r["gold"] for r in rows]
    both = [r["systems"][SHIPPED] for r in rows]
    alone = [decider.decide(r["scores"]["B"]) for r in rows]
    out = {"what": __doc__.split("\n\n")[0].replace("\n", " "),
           "script": "scripts/test4_rule_layer.py", "per_row_file": {"path": pinned["path"], "sha256": digest},
           "draws": fe.DRAWS, "seed": fe.SEED, "slices": {}}
    for name, keep in (("all rows", lambda r: True), ("eligibility pass", lambda r: r["eligibility_pass"])):
        idx = [i for i, r in enumerate(rows) if keep(r)]
        g, b, h = ([x[i] for i in idx] for x in (gold, alone, both))
        G, B, H = fe.as_idx(g), fe.as_idx(b), fe.as_idx(h)
        dr = fe.two_way_draws([rows[i]["posting_id"] for i in idx])
        changed = [j for j in range(len(idx)) if b[j] != h[j]]
        out["slices"][name] = {
            "rows": len(idx),
            "B alone": {"macro_f1": fe.metrics(G, B, dr)["macro_f1"], **keeps(g, b)},
            "B + H''": {"macro_f1": fe.metrics(G, H, dr)["macro_f1"], **keeps(g, h)},
            "B + H'' minus B alone": fe.paired(G, H, B, dr),
            "labels_changed_by_H''": len(changed),
            "changes": {f"{b[j]} -> {h[j]}": sum(1 for k in changed if (b[k], h[k]) == (b[j], h[j])) for j in changed}}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["slices"], indent=1))


if __name__ == "__main__":
    main()
