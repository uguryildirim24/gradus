"""Live checks of the deployed Modal demo, driven from the Mac (no model runs here; it only sends HTTP
requests and calls the app's own Modal functions).

    ~/.local/share/uv/tools/modal/bin/python scripts/modal_demo_check.py URL --pool demo/local/pool.jsonl [--compare 4]
    ~/.local/share/uv/tools/modal/bin/python scripts/modal_demo_check.py URL --capped     # expect a 402
    ~/.local/share/uv/tools/modal/bin/python scripts/modal_demo_check.py URL --eval RESULT.json --rows ROWS.jsonl [--n 40]

Scores the saved persona against the pool through the public URL, timing a cold start (202 until the
model is up), a request with reasons and one without. With --compare N it sends N of the scored
postings through the `reference` function (demo_backend_torch on CPU float32, the oci-pi path) and
prints both label probabilities side by side. Ledger spend before and after comes from `ledger_report`.

With --eval (the 4B check) it scores N rows of ROWS.jsonl (pair_id, prompt: the dev or real-test file a
colab_eval / local_eval.py run scored) on the deployed GPU scorer and compares labels and float32 label
scores with that run's RESULT.json, row by row (demo_lib.label_parity: the eval's label is run B's decider on its
stored scores; 98% agreement once near-cuts, eval s within 0.02 of a cut-off, are set aside). Use a result of the same
adapter and dtype (B's evals ran bfloat16). Both read the scores with the same code (evalstats.LabelHead) and the
label with the same decider, so what is left is kernels and batching, which can flip a row near a cut-off only.
Rows that share a resume go in one call, through the scorer's budget check like any visitor's request.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "demo"))

import demo_lib as lib  # noqa: E402


def call(url: str, body: dict | None = None, timeout: float = 200) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def ledger() -> dict:
    import modal
    return modal.Function.from_name("gradus", "ledger_report").remote()


def score(base: str, body: dict, wait_s: float = 600) -> tuple[dict, float, float]:
    """(result, seconds until the model was warm, seconds of the scoring request itself)."""
    t0 = time.time()
    warm_s = 0.0
    while True:
        code, r = call(base + "/score", body)
        if code != 202:
            break
        while time.time() - t0 < wait_s:
            time.sleep(3)
            h = call(base + "/api/health")[1]
            if h["scorer"]["state"] in ("ready", "error", "capped"):
                break
        warm_s = time.time() - t0
    took = time.time() - t0 - warm_s
    if code != 200:
        raise SystemExit(f"score answered {code}: {r}")
    return r, round(warm_s, 1), round(took, 1)


def eval_compare(eval_path: str, rows_path: str, n: int, chunk: int = 20) -> dict:
    """The deployed GPU scorer against a local_eval.py result for the same rows (see the module docstring)."""
    import modal

    ev = json.loads(Path(eval_path).read_text())
    if ev.get("framing", "chat") != "chat":
        raise SystemExit(f"{eval_path} was scored with --framing {ev['framing']}; the demo frames prompts as chat")
    ref = {r["pair_id"]: r for r in ev["rows"]}
    rows = [r for r in (json.loads(line) for line in open(rows_path) if line.strip()) if r["pair_id"] in ref][:n]
    if not rows:
        raise SystemExit(f"no row of {rows_path} is in {eval_path}")
    groups: dict[str, list] = {}
    for r in rows:
        prefix, suffix = lib.split_row(r["prompt"])
        groups.setdefault(prefix, []).append({"id": r["pair_id"], "suffix": suffix})
    scorer = modal.Cls.from_name("gradus", "Scorer")()
    got: dict[str, dict] = {}
    model = ""
    t = time.time()
    for prefix, items in groups.items():
        for k in range(0, len(items), chunk):
            out = scorer.score.remote(prefix, items[k:k + chunk], False, 1)
            if out.get("capped"):
                raise SystemExit("the GPU refused the call: the month's budget is used (raise the cap to run this check)")
            if "error" in out:
                raise SystemExit(f"scorer error: {out['error']}")
            model = out.get("model", model)
            got.update({x["id"]: x for x in out["results"]})
    print(f"scored {len(got)} eval rows on the deployed scorer ({model}) in {time.time() - t:.0f} s, "
          f"{len(groups)} resumes; eval {Path(eval_path).name}: model {ev.get('model')}, dtype {ev.get('dtype')}")
    pairs = [(r["pair_id"], got[r["pair_id"]]["label"], got[r["pair_id"]]["scores"], ref[r["pair_id"]]["scores"])
             for r in rows]
    res = lib.label_parity(pairs)
    lib.print_parity(res, f"the deployed demo ({model})")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--pool", default=str(ROOT / "demo/local/pool.jsonl"))
    ap.add_argument("--persona", default=str(ROOT / "demo/personas/alex_cs_f1.json"))
    ap.add_argument("--live", type=int, default=20)
    ap.add_argument("--compare", type=int, default=0)
    ap.add_argument("--capped", action="store_true", help="only check that a score is refused with 402")
    ap.add_argument("--eval", help="a colab_eval / local_eval.py result file: compare the deployed labels with it")
    ap.add_argument("--rows", help="with --eval: the JSONL (pair_id, prompt) that result scored")
    ap.add_argument("--n", type=int, default=40, help="with --eval: how many of its rows to score")
    a = ap.parse_args()
    if a.eval and not a.rows:
        ap.error("--eval needs --rows")
    base = a.url.rstrip("/")
    persona = json.loads(Path(a.persona).read_text())
    body = {"resume": persona["resume"], "profile": persona["profile"], "live": a.live, "redact": True}
    t = time.time()
    code, h = call(base + "/api/health")
    print(f"health {code} in {time.time() - t:.1f} s: scorer {h['scorer']['state']}, budget {h['budget']}, pool {h['pool']}")
    if a.capped:
        code, r = call(base + "/score", {**body, "reasons": True})
        print("score answered", code, r)
        sys.exit(0 if code == 402 else 1)
    if a.eval:
        before = ledger()
        res = eval_compare(a.eval, a.rows, a.n)
        after = ledger()
        print(f"counted spend for the check: ${after['spent_usd'] - before['spent_usd']:.4f}")
        sys.exit(0 if res["ok"] else 1)
    before = ledger()
    print("ledger before:", json.dumps({k: before[k] for k in ("month", "cap_usd", "spent_usd", "by_kind")}))
    r1, warm_s, took1 = score(base, {**body, "reasons": True})
    s1 = r1["stats"]["scorer_stats"]
    print(f"cold start {warm_s} s; with reasons: {s1['items']} postings, request {took1} s, GPU {s1['seconds']} s, "
          f"{s1['seconds'] / max(s1['items'], 1):.2f} s per posting")
    r2, _, took2 = score(base, {**body, "reasons": False})
    s2 = r2["stats"]["scorer_stats"]
    print(f"no reasons: {s2['items']} postings, request {took2} s, GPU {s2['seconds']} s, "
          f"{s2['seconds'] / max(s2['items'], 1):.2f} s per posting")
    for b in ("For you", "Explore", "Excluded"):
        cards = r1["buckets"][b]
        print(f"  {b}: {len(cards)}", "; ".join(f"{c['title'][:40]} p_keep={c['p_keep']} [{c['source']}] {c['reason'][:60]!r}"
                                               for c in cards[:3]))
    after = ledger()
    print("ledger after:", json.dumps({k: after[k] for k in ("month", "cap_usd", "spent_usd", "by_kind")}))
    print(f"counted spend for these two requests and the cold start: ${after['spent_usd'] - before['spent_usd']:.4f} "
          "(the idle scaledown that follows is counted as it happens)")
    if a.compare:
        import modal
        pool = {r["id"]: r for r in lib.load_pool(Path(a.pool))}
        by_bucket = [[c for c in b if c["source"] == "model"] for b in r1["buckets"].values()]
        model_cards = [c for k in range(a.compare) for b in by_bucket if k < len(b) for c in [b[k]]][: a.compare]
        resume = lib.redact(persona["resume"])
        prof = lib.profile_text(persona["profile"])
        prefix, _ = lib.split_prompt(resume, prof, "")
        items = [{"id": c["id"], "suffix": lib.split_prompt(resume, prof, pool[c["id"]]["text"])[1]} for c in model_cards]
        t = time.time()
        ref = modal.Function.from_name("gradus", "reference").remote(prefix, items, False)
        print(f"reference (demo_backend_torch, CPU float32) on {len(items)} postings in {time.time() - t:.0f} s")
        worst = 0.0
        for c, rr in zip(model_cards, ref):
            d = max(abs(c["p"][w] - rr["p"][w]) for w in ("Keep", "Explore", "Skip"))
            worst = max(worst, d)
            print(f"  {c['id'][:28]:28} live {c['label']:7} {c['p']}  reference {rr['label']:7} {rr['p']}  max|dp|={d:.4f}")
        same = sum(c["label"] == rr["label"] for c, rr in zip(model_cards, ref))
        print(f"labels agree {same}/{len(ref)}; largest probability difference {worst:.4f}")


if __name__ == "__main__":
    main()
