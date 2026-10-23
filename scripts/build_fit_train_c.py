"""Candidate C's training set (eval/protocol_amendment_6b.json C_labels, train_C) and the Colab eval-loss rows.

  build_fit_train_c.py   -> data/built/fit/cycle3/relabel_C/train_C.csv (prompt, completion), train_C_rows.jsonl,
                            eval_loss_B.csv / eval_loss_C.csv (the 64 eval-loss rows in each run's format),
                            train_C_report.json and its copy eval/cycleC_train_csv.json

Rows: train_B.csv's 2,686 rows in their order (train3_rows.jsonl gives row_id by csv_row). A Skip row keeps its label.
A Keep or Explore row takes the relabel's label (relabel_C/labels/*.jsonl, pooled by row_id; one valid line a row);
where the label changed the completion is careful_completion(new label, new reason), where it did not the train_B
completion stays byte for byte. A row without a valid relabel line keeps its train_B label (listed).

Before anything is written every train_B.csv row is rebuilt with build_fit_train3.prompt_b and B's system from the
batch rows and must equal its CSV prompt byte for byte, and the CSV completion must equal train3_rows.jsonl's.

Band: amendment 5's training_label_mix band with build_fit_train3.downsample unchanged (downsample only, never
relabel), seed 20261024. Prompt: build_fit_train3.prompt_b with C's system = B's system + "\\n" + the clarification
line (sha256 checked against amendment 6b).

Eval-loss rows: 64 drawn with random.Random(20261024) from the band-dropped rows of train3_rows.jsonl (all Skip),
completion careful_completion from the cycle-3 teacher's line (labels/A), prompt_b with B's and with C's system.
Loads no model and calls nothing.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import build_fit_train3 as T3  # noqa: E402
from build_fit_train2 import careful_completion  # noqa: E402

C3 = ROOT / "data/built/fit/cycle3"
RC = C3 / "relabel_C"
SEED = 20261024
CLAR = "- A required item the resume names but doesn't show doing is not met."
SYSTEM_C_SHA = "eb304433e80f6a71d43d5404cec86131f5094a32487472f4494ae8361c22cc9f"
SYSTEM_B_SHA = "1d12b0d0ec70e3dcab5e7523a3477c04fc75714564b5e33b94c31f44e7cb2e0d"
EVAL_N = 64
WORDS = ("Keep", "Explore", "Skip")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def systems() -> tuple[str, str]:
    sb = T3.system_b()
    sc = sb + "\n" + CLAR
    assert hashlib.sha256(sb.encode()).hexdigest() == SYSTEM_B_SHA
    assert hashlib.sha256(sc.encode()).hexdigest() == SYSTEM_C_SHA, "C's system differs from amendment 6b's"
    return sb, sc


def read_csv(p: Path) -> list[dict]:
    csv.field_size_limit(10 ** 9)
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    sb, sc = systems()
    side = [json.loads(l) for l in (C3 / "train3_rows.jsonl").open() if l.strip()]
    kept = sorted((r for r in side if r["kept"]), key=lambda r: r["csv_row"])
    dropped = [r for r in side if not r["kept"]]
    src = {}
    for f in sorted((C3 / "label_batches").glob("batch_*.jsonl")):
        for l in f.open():
            if l.strip():
                b = json.loads(l)
                src[b["row_id"]] = b
    csv_b = read_csv(C3 / "train_B.csv")
    assert len(csv_b) == len(kept) == 2686
    assert sha(C3 / "train_B.csv") == json.loads((ROOT / "eval/cycle3_train_csvs.json").read_text())[
        "files"]["data/built/fit/cycle3/train_B.csv"]["sha256"]
    for n, r in enumerate(kept):
        b = src[r["row_id"]]
        if T3.prompt_b(b["resume"], b["profile"], b["posting"], sb) != csv_b[n]["prompt"]:
            raise SystemExit(f"csv row {n} ({r['row_id']}): prompt_b rebuild differs from train_B.csv")
        if csv_b[n]["completion"] != r["completion"]:
            raise SystemExit(f"csv row {n}: completion differs from train3_rows.jsonl")

    lab_files = sorted((RC / "labels").glob("batch_*.jsonl"))
    lines, bad = T3.read_labels(lab_files)
    if bad:
        raise SystemExit(f"unparsed relabel lines: {bad[:3]}")
    relabel_ids = {json.loads(l)["row_id"] for f in sorted((RC / "batches").glob("*.jsonl")) for l in f.open()}
    assert len(relabel_ids) == 1292
    rows, no_label, trans = [], [], Counter()
    for r in kept:
        old, comp = r["label"], r["completion"]
        new, reason = old, None
        if r["row_id"] in relabel_ids:
            x, why = T3.judge(lines[r["row_id"]]) if r["row_id"] in lines else (None, "missing")
            if x is None:
                no_label.append({"row_id": r["row_id"], "why": why})
            else:
                new, reason = x["label"], x["reason"]
        if new != old:
            comp = careful_completion(new, reason)
        trans[f"{old}->{new}"] += 1
        rows.append({"row_id": r["row_id"], "old": old, "label": new, "completion": comp, "changed": new != old})

    band = json.loads(T3.AMEND5.read_text())["training_label_mix"]["band"]
    before = T3.mix(rows)
    kept_c, ds = T3.downsample(rows, band, SEED)
    if not T3.in_band(Counter(r["label"] for r in kept_c), band):
        raise SystemExit("C's set is outside the band")
    out_rows = []
    for r in kept_c:
        b = src[r["row_id"]]
        p = T3.prompt_b(b["resume"], b["profile"], b["posting"], sc)
        if not p.startswith(sc + "\n\nRESUME:\n") or not p.endswith("\n\nAnswer:"):
            raise SystemExit(f"{r['row_id']}: C prompt layout")
        if r["completion"].split()[0] != r["label"] or "\n" in r["completion"]:
            raise SystemExit(f"{r['row_id']}: completion {r['completion']!r}")
        out_rows.append((p, r["completion"]))
    T3.write_csv(RC / "train_C.csv", out_rows)
    with (RC / "train_C_rows.jsonl").open("w") as f:
        for n, r in enumerate(kept_c):
            f.write(json.dumps(r | {"csv_row": n}, ensure_ascii=False) + "\n")

    # eval-loss rows: band-dropped Skips with the teacher's own line
    a_lines, _ = T3.read_labels(sorted(T3.LABELS.glob("*.jsonl")))
    pool = sorted(r["row_id"] for r in dropped)
    ev_ids = random.Random(SEED).sample(pool, EVAL_N)
    ev = {"B": [], "C": []}
    for i in ev_ids:
        x, why = T3.judge(a_lines[i])
        assert x and x["label"] == "Skip", (i, why)
        b = src[i]
        comp = careful_completion("Skip", x["reason"])
        ev["B"].append((T3.prompt_b(b["resume"], b["profile"], b["posting"], sb), comp))
        ev["C"].append((T3.prompt_b(b["resume"], b["profile"], b["posting"], sc), comp))
    for k, v in ev.items():
        T3.write_csv(RC / f"eval_loss_{k}.csv", v)

    rep = {
        "what": "candidate C's training set and the Colab eval-loss rows (eval/protocol_amendment_6b.json), "
                "scripts/build_fit_train_c.py",
        "amendment_sha256": {"eval/protocol_amendment_6b.json": sha(ROOT / "eval/protocol_amendment_6b.json")},
        "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                          [C3 / "train_B.csv", C3 / "train3_rows.jsonl", RC / "prep.json", RC / "prompt.txt"]},
        "relabel_labels_sha256": {str(p.relative_to(ROOT)): sha(p) for p in lab_files},
        "train_B_parity": f"{len(kept)}/{len(kept)} prompts rebuilt byte for byte with B's system; completions equal",
        "relabeled_rows": len(relabel_ids),
        "rows_without_valid_relabel": no_label,
        "transitions": dict(sorted(trans.items())),
        "changed": sum(r["changed"] for r in rows),
        "changed_stricter": sum(r["changed"] and WORDS.index(r["label"]) > WORDS.index(r["old"]) for r in rows),
        "changed_looser": sum(r["changed"] and WORDS.index(r["label"]) < WORDS.index(r["old"]) for r in rows),
        "train_B_mix": T3.mix([{"label": r["old"]} for r in rows]),
        "band": band,
        "mix_before_band": before,
        "mix_before_band_in_band": T3.in_band(Counter(r["label"] for r in rows), band),
        "downsample": {"needed": ds["needed"], "seed": SEED, "keep": ds["keep"],
                       "dropped_counts": {w: len(v) for w, v in ds["dropped"].items()},
                       "dropped_row_ids": ds["dropped"]},
        "mix_after_band": T3.mix(kept_c),
        "changed_kept": sum(r["changed"] for r in kept_c),
        "prompt_c_system_sha256": SYSTEM_C_SHA,
        "prompt_b_system_sha256": SYSTEM_B_SHA,
        "completion_rule": "changed label: careful_completion(new label, relabel reason); unchanged: train_B's completion",
        "files": {str(p.relative_to(ROOT)): {"sha256": sha(p), "rows": n} for p, n in
                  [(RC / "train_C.csv", len(out_rows)), (RC / "eval_loss_B.csv", EVAL_N), (RC / "eval_loss_C.csv", EVAL_N)]},
        "rows_file_sha256": sha(RC / "train_C_rows.jsonl"),
        "eval_loss_rows": {"seed": SEED, "from": "band-dropped Skip rows of train3_rows.jsonl", "row_ids": ev_ids},
    }
    for p in (RC / "train_C_report.json", ROOT / "eval/cycleC_train_csv.json"):
        p.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("transitions", "changed", "changed_stricter", "changed_looser",
                                         "mix_before_band", "mix_after_band", "rows_without_valid_relabel")}
                     | {"dropped": rep["downsample"]["dropped_counts"]}, indent=1))


if __name__ == "__main__":
    main()
