"""Score a matcher model's Keep / Explore / Skip and report accuracy, calibration and safety.

Runs on oci-pi (CPU) or a Colab L4 (colab_eval.py passes --device cuda), never the Mac (the Mac only
runs the final demo).

  local_eval.py score MODEL_DIR --dev rows.jsonl [--device auto|cpu|cuda] [--shard K/N] [--batch B]
                      [--dtype float32|bfloat16] [--framing chat|raw] [--limit N] [--gen 1] [--threads 1] --out FILE
  local_eval.py combine SHARD.json ... --out FILE [--temperature-from DEV_RESULT.json | --temperature T]
  local_eval.py analyze RESULT.json [--temperature-from DEV_RESULT.json | --temperature T] [--draws 1000]
  local_eval.py batchcheck BATCH1.json BATCHN.json [--tol 1e-3]
  local_eval.py decide --b1 PROBE1.json [--b4 PROBE4.json] --forwards N --jobs J --deadline EPOCH
      (colab_eval's jobs.sh: the batch, and whether every job fits before the session's deadline)
      (score, combine and analyze take --include-undecidable: see "Rows" below)

First run of `score` on a new machine or model (it has never run before this round): score 5 rows
with --batch 1 and check that agreement_with_old_argmax is about 100% and exact_ties is 0; score the
same 5 rows with --batch 4 and run `batchcheck` on the two result files (left padding on Qwen3.5's
linear-attention layers must not change a score). Only then start a full run. --batch 1 is the default.

How a label is scored (evalstats.LabelHead and evalstats.pick; the demo scorers use the same two).
The three labels (Keep, Explore, Skip) are one token each. For each row the final hidden state (the
input of the output head, after the last norm) is cast to float32 and multiplied by the head's rows
for the three label tokens (the embedding rows when the head is tied, as in Qwen3.5), so the three
scores are float32 and almost never tie even when the weights are bfloat16. A tie (exactly equal
scores) goes to the safer label: Skip, then Explore, then Keep. The
probabilities are a softmax over the three labels at a temperature fitted on a dev run
(`--temperature-from`), not on the rows being reported, unless you pass nothing (then it is fitted
on these rows and the report says so).

Every report line carries a 95% bootstrap interval that resamples postings (base_rq_id), because
rows of one posting share text: accuracy, macro-F1, ECE (15 bins), Brier, and the hard-filter
violation rate (gold Skip because of a work-authorization hard rule, predicted Keep).

Rows: JSONL with prompt, bucket (For you / Explore / Excluded), pair_id and, when present,
base_rq_id (the cluster; derived from rq_id otherwise), flags or completion (for the hard-filter
rate), variant_kind (accuracy per kind; the headline again without the added Keep kinds and on the
resume-dependent kinds alone) and decidable/missing. A row marked not decidable (the prompt does not
show a fact its label rests on) is left out of every number, as build_real_test.py score and
baselines.py leave it out; --include-undecidable keeps it. The report prints how many were left out
and the all-rows accuracy and macro-F1 beside the headline.

On CPU, the reference kernels for Qwen3.5's linear-attention layers (no flash-linear-attention)
spend 80% of a forward in aten::bmm: 25-40 s per 1,200-token prompt on oci-pi whatever the thread
count, so on CPU run N one-thread shards in parallel (`--shard K/N`) and `combine` their outputs. On a GPU use --batch.
torch and transformers are imported only by `score`; `combine` and `analyze` need numpy alone.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    from matcher.evalstats import (  # noqa: E402,F401
        LABELS, WORD, LabelHead, analyze, base_of, cluster_bootstrap, cluster_ci, fit_temperature, macro_f1_of, pick,
        report,
    )
except ModuleNotFoundError:  # on the Colab VM: colab_eval.py uploads evalstats.py beside this script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evalstats import (  # noqa: E402,F401
        LABELS, WORD, LabelHead, analyze, base_of, cluster_bootstrap, cluster_ci, fit_temperature, macro_f1_of, pick,
        report,
    )

# ---------------------------------------------------------------- the model (torch, lazy)

def frame(tok, prompt: str, framing: str) -> str:
    if framing == "raw":
        return prompt
    return tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=False)


def row_visible(r: dict) -> bool:
    if "decidable" in r:
        return bool(r["decidable"])
    return not any(c != "skill_wording" for c, _ in r.get("missing", []))


def score(a: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(a.threads)
    k, n = (int(x) for x in a.shard.split("/"))
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[a.dtype]
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).eval()
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dev = next(model.parameters()).device
    head = LabelHead(model, tok)  # the label read the demo scorers share (evalstats.py)
    print(f"loaded {a.model} ({type(model).__name__}, {a.dtype}, tied head {head.tied}) on {dev} in {time.time() - t0:.1f}s")
    lid = head.ids

    all_rows = [json.loads(line) for line in open(a.dev) if line.strip()]
    if a.limit:
        all_rows = all_rows[: a.limit]
    rows = [r for i, r in enumerate(all_rows) if i % n == k]
    out = []
    with torch.inference_mode():
        for start in range(0, len(rows), a.batch):
            chunk = rows[start:start + a.batch]
            enc = tok([frame(tok, r["prompt"], a.framing) for r in chunk], add_special_tokens=False,
                      return_tensors="pt", padding=True).to(dev)
            t = time.time()
            o = model(**enc, logits_to_keep=1)
            dt = time.time() - t
            sc = head.scores().cpu()  # (B, 3) float32 label scores
            lp = torch.log_softmax(o.logits[:, -1].float(), -1)
            for j, r in enumerate(chunk):
                s = [float(x) for x in sc[j]]
                old = LABELS[max(range(3), key=lambda i: lp[j, lid[i]].item())]  # the old rule: argmax, ties to Keep
                out.append(dict(pair_id=r["pair_id"], gold=WORD[r["bucket"]], pred=LABELS[pick(s)], pred_old=old,
                                scores=s, visible=row_visible(r), top1=tok.decode([int(lp[j].argmax())]),
                                prompt_tokens=int(enc["attention_mask"][j].sum()), forward_s=round(dt / len(chunk), 2),
                                base_rq_id=base_of(r), variant_kind=r.get("variant_kind"),
                                flags=r.get("flags"), completion=r.get("completion", "")))
                print(json.dumps({k2: out[-1][k2] for k2 in ("pair_id", "gold", "pred", "pred_old", "scores", "forward_s")}),
                      flush=True)
        gens = []
        for r in rows[: a.gen]:
            enc = tok(frame(tok, r["prompt"], a.framing), add_special_tokens=False, return_tensors="pt").to(dev)
            t = time.time()
            g = model.generate(**enc, max_new_tokens=a.max_tokens, do_sample=False)
            text = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            gens.append({"pair_id": r["pair_id"], "gold": r.get("completion"), "text": text,
                         "tok_per_s": round((g.shape[1] - enc["input_ids"].shape[1]) / (time.time() - t), 1)})
            print("GEN", json.dumps(gens[-1]), flush=True)
    secs = [o["forward_s"] for o in out]
    timing = {"rows": len(out), "seconds_per_row": round(sum(secs) / max(len(secs), 1), 2),
              "mean_prompt_tokens": round(sum(o["prompt_tokens"] for o in out) / max(len(out), 1)),
              "threads": a.threads, "batch": a.batch, "load_s": round(time.time() - t0 - sum(secs), 1)}
    Path(a.out).write_text(json.dumps({"model": a.model, "dev": a.dev, "framing": a.framing, "dtype": a.dtype,
                                       "shard": a.shard, "timing": timing, "generations": gens, "rows": out}, indent=1))
    print("timing", json.dumps(timing))
    if n == 1:
        finish(Path(a.out), a)


def temperature_arg(a: argparse.Namespace) -> tuple[float | None, str | None]:
    if getattr(a, "temperature", None):
        return a.temperature, "given"
    src = getattr(a, "temperature_from", None)
    if src:
        d = json.loads(Path(src).read_text())
        res = d.get("summary") or analyze(d["rows"])
        return float(res["temperature"]), f"fitted on {Path(src).name}"
    return None, None


def finish(path: Path, a: argparse.Namespace) -> dict:
    d = json.loads(path.read_text())
    t, src = temperature_arg(a)
    res = analyze(d["rows"], t, src, getattr(a, "draws", 1000), getattr(a, "include_undecidable", False))
    d["summary"] = res
    path.write_text(json.dumps(d, indent=1))
    report(res)
    return res


def combine(a: argparse.Namespace) -> None:
    parts = [json.loads(Path(f).read_text()) for f in a.files]
    assert len({(p["model"], p["dev"], p["framing"], p["dtype"]) for p in parts}) == 1, "shards of different runs"
    rows = [r for p in parts for r in p["rows"]]
    assert len({r["pair_id"] for r in rows}) == len(rows), "a row is in two shards"
    secs = [r["forward_s"] for r in rows]
    timing = {"rows": len(rows), "seconds_per_row_per_shard": round(sum(secs) / len(secs), 2), "shards": len(parts),
              "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in rows) / len(rows))}
    Path(a.out).write_text(json.dumps({k: parts[0][k] for k in ("model", "dev", "framing", "dtype")} | {
        "shards": len(parts), "timing": timing, "generations": [g for p in parts for g in p["generations"]],
        "rows": rows}, indent=1))
    finish(Path(a.out), a)


def analyze_cmd(a: argparse.Namespace) -> None:
    finish(Path(a.result), a)


def batch_diff(rows_a: list[dict], rows_b: list[dict], tol: float = 1e-3) -> dict:
    """Score differences between two runs of the same rows (batch 1 against batch N), by pair_id."""
    b = {r["pair_id"]: r for r in rows_b}
    common = [r for r in rows_a if r["pair_id"] in b]
    diffs = [max(abs(x - y) for x, y in zip(r["scores"], b[r["pair_id"]]["scores"])) for r in common]
    flips = [r["pair_id"] for r in common if r["pred"] != b[r["pair_id"]]["pred"]]
    return {"rows": len(common), "only_in_one": len(rows_a) + len(rows_b) - 2 * len(common),
            "max_abs_diff": max(diffs, default=0.0), "over_tol": sum(d > tol for d in diffs), "pred_flips": flips,
            "ok": bool(common) and not flips and all(d <= tol for d in diffs)}


def decide_run(b1: dict | None, b4: dict | None, forwards: int, jobs: int, seconds_left: float,
               load_s: float = 90.0) -> dict:
    """From two probe runs of the same rows (--batch 1 and --batch 4), the batch to use and whether all `forwards`
    rows over `jobs` model loads fit in `seconds_left`. Batch 4 only when batchcheck passes. The rate is the median
    per-row forward time (the first row carries the CUDA warm-up); a missing probe (out of memory) does not fit."""
    if not b1 or not b1.get("rows"):
        return {"fits": False, "batch": 1, "why": "batch-1 probe has no rows (out of memory or failed)"}
    four = bool(b4 and b4.get("rows")) and batch_diff(b1["rows"], b4["rows"])["ok"]
    rows = (b4 if four else b1)["rows"]
    rate = float(np.median([r["forward_s"] for r in rows]))
    need = rate * forwards + load_s * jobs
    return {"fits": need <= seconds_left, "batch": 4 if four else 1, "seconds_per_row": round(rate, 3),
            "need_s": round(need), "have_s": round(seconds_left), "batch4_ok": four,
            "why": f"{forwards} rows at {rate:.2f} s + {jobs} loads at {load_s:.0f} s = {need / 60:.1f} min, "
                   f"{seconds_left / 60:.1f} min left"}


def decide(a: argparse.Namespace) -> None:
    def load(p):
        return json.loads(Path(p).read_text()) if p and Path(p).exists() else None

    res = decide_run(load(a.b1), load(a.b4), a.forwards, a.jobs, a.deadline - time.time(), a.load_s)
    print(json.dumps(res))


def batchcheck(a: argparse.Namespace) -> None:
    res = batch_diff(json.loads(Path(a.first).read_text())["rows"], json.loads(Path(a.second).read_text())["rows"], a.tol)
    print(json.dumps(res))
    if not res["ok"]:
        sys.exit("batch runs disagree: do not use --batch > 1 for this model")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def temp_args(p):
        p.add_argument("--temperature-from", help="a dev result file whose fitted temperature is used")
        p.add_argument("--temperature", type=float, help="use this temperature")
        p.add_argument("--draws", type=int, default=1000)
        p.add_argument("--include-undecidable", action="store_true",
                       help="score rows marked not decidable too (left out by default)")

    s = sub.add_parser("score")
    s.add_argument("model")
    s.add_argument("--dev", required=True)
    s.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    s.add_argument("--shard", default="0/1")
    s.add_argument("--batch", type=int, default=1)
    s.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    s.add_argument("--framing", choices=("chat", "raw"), default="chat")
    s.add_argument("--limit", type=int, default=0, help="only the first N rows of the file")
    s.add_argument("--gen", type=int, default=0)
    s.add_argument("--max-tokens", type=int, default=48)
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--out", required=True)
    temp_args(s)
    c = sub.add_parser("combine")
    c.add_argument("files", nargs="+")
    c.add_argument("--out", required=True)
    temp_args(c)
    n = sub.add_parser("analyze")
    n.add_argument("result")
    temp_args(n)
    bc = sub.add_parser("batchcheck")
    bc.add_argument("first")
    bc.add_argument("second")
    bc.add_argument("--tol", type=float, default=1e-3)
    dc = sub.add_parser("decide", help="pick dtype fit and batch from two 5-row probes (colab_eval's jobs.sh)")
    dc.add_argument("--b1", required=True, help="probe result at --batch 1")
    dc.add_argument("--b4", help="probe result at --batch 4, same rows")
    dc.add_argument("--forwards", type=int, required=True, help="rows every job together will score")
    dc.add_argument("--jobs", type=int, required=True, help="model loads (one per job)")
    dc.add_argument("--deadline", type=float, required=True, help="epoch seconds by which every job must be done")
    dc.add_argument("--load-s", type=float, default=90.0)
    a = ap.parse_args()
    {"score": score, "combine": combine, "analyze": analyze_cmd, "batchcheck": batchcheck, "decide": decide}[a.cmd](a)


if __name__ == "__main__":
    main()
