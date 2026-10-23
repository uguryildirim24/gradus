"""Demo scorer with mlx: the bounded, prefix-cached scorer for the final demo on the Mac.

Same JSON API as demo_backend_torch.py (see demo_common.py), so demo/server.py does not care which
one it talks to. Runs on the Mac only inside the Wed/Thu switch window (docs/DEMO.md).

    uv run python scripts/demo_score.py serve --model data/models/<run>/merged --port 8765
    uv run python scripts/demo_score.py check --model <dir> --dev data/built/training/dev.jsonl --n 20 --peak-gate-gb 4
    uv run python scripts/demo_score.py parity --model <dir> --dev <dev.jsonl> --torch <local_eval out.json>

What keeps it small (the 17 GB benchmark scored every position over a 248,320-token vocabulary):
  - logits for the LAST token only: the decoder runs to hidden states, and only the final position goes
    through the output head (no [length x 248,320] logits, about 1 GB per call at 2,000 tokens);
  - the prompt's shared prefix (rubric + resume + profile) is prefilled once;
    each posting gets a deep copy (Qwen3.5's linear-attention state cannot be trimmed back);
  - the posting suffix goes through in chunks with only the cache state evaluated, so nothing but the
    cache survives a step; the last token alone produces logits;
  - mx.set_cache_limit(2 GB), mx.clear_cache() every 20 pairs, optional mx.set_memory_limit, and
    mx.get_peak_memory() logged per request plus the highest peak since load; --peak-gate-gb aborts a run
    that exceeds the budget.

The scores are read as local_eval.py reads them (evalstats.LabelHead, here in mlx): the last position's final
hidden state in float32 times the float32 head rows of Keep, Explore and Skip; the label is run B's decider on them
(demo_common.decider_label). The full-vocabulary logits are only used to generate the reason. `parity` compares these
scores with a local_eval.py result file for the same rows.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "demo")]
import demo_lib as lib  # noqa: E402
from demo_common import LABELS, clean_reason, decider_label, label_probs, label_token_ids, plan_prefix, serve  # noqa: E402

GB = 1 << 30
CHUNK = 512  # tokens per forward step while prefilling


def head_label_rows(inner, ids: list[int]) -> mx.array:
    """The output head's rows for the label tokens in float32 (the embedding rows when the head is tied)."""
    layer = inner.model.embed_tokens if getattr(inner.args, "tie_word_embeddings", False) else inner.lm_head
    assert not hasattr(layer, "scales"), "the output head is quantized; serve the bf16 merged checkpoint"
    assert "bias" not in layer, "output head has a bias: the float32 label scores would skip it"
    return layer.weight[mx.array(ids)].astype(mx.float32)


def label_scores(hidden: mx.array, rows32: mx.array) -> dict[str, float]:
    """evalstats.LabelHead.scores in mlx: the last position's hidden state in float32 times the label rows."""
    s = hidden.reshape(-1, hidden.shape[-1])[-1].astype(mx.float32) @ rows32.T
    return dict(zip(LABELS, s.tolist()))


class MlxScorer:
    def __init__(self, model_dir: str, cache_limit_gb: float = 2.0, memory_limit_gb: float | None = None,
                 clear_every: int = 20, peak_gate_gb: float | None = None) -> None:
        self.name = "mlx"
        self.model_dir = model_dir
        self.clear_every = clear_every
        self.peak_gate = int(peak_gate_gb * GB) if peak_gate_gb else None
        mx.set_cache_limit(int(cache_limit_gb * GB))
        if memory_limit_gb:
            mx.set_memory_limit(int(memory_limit_gb * GB))
        t0 = time.time()
        self.model, self.tok = load(model_dir)
        self.load_s = round(time.time() - t0, 1)
        self.label_ids = label_token_ids(self.tok)
        self.stop = {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id}
        self.inner = getattr(self.model, "language_model", self.model)
        self.label_rows = head_label_rows(self.inner, self.label_ids)
        self.done = 0
        self.max_peak = 0  # highest per-request peak since load, in bytes (each request resets mlx's own peak)

    # -- model plumbing -----------------------------------------------------
    def _hidden_last(self, ids: list[int], cache: list) -> mx.array:
        """The final hidden state (after the last norm) at the last token; the decoder sees every token."""
        return self.inner.model(mx.array([ids]), cache)[:, -1:, :]

    def _logits_last(self, ids: list[int], cache: list) -> mx.array:
        """Output logits for the last token only (for generating the reason)."""
        hidden = self._hidden_last(ids, cache)
        if getattr(self.inner.args, "tie_word_embeddings", False):
            out = self.inner.model.embed_tokens.as_linear(hidden)
        else:
            out = self.inner.lm_head(hidden)
        return out[0, -1].astype(mx.float32)

    def _feed(self, ids: list[int], cache: list) -> None:
        """Run tokens through the model for their cache effect. Only the cache state is evaluated, so the
        (lazy) hidden states and logits are never computed."""
        for k in range(0, len(ids), CHUNK):
            self.inner.model(mx.array([ids[k:k + CHUNK]]), cache)
            mx.eval([c.state for c in cache])

    def _check_peak(self) -> None:
        self.max_peak = max(self.max_peak, mx.get_peak_memory())
        if self.peak_gate is not None and mx.get_peak_memory() > self.peak_gate:
            raise MemoryError(f"peak memory {mx.get_peak_memory() / GB:.2f} GB is over the "
                              f"{self.peak_gate / GB:.1f} GB gate; stopping")

    # -- scoring ------------------------------------------------------------
    def score_one(self, ids: list[int], pre_n: int, base: list | None, reasons: bool, max_tokens: int) -> dict:
        cache = copy.deepcopy(base) if base is not None else make_prompt_cache(self.inner)
        n = pre_n if base is not None else 0
        self._feed(ids[n:-1], cache)
        lab = label_scores(self._hidden_last(ids[-1:], cache), self.label_rows)  # float32, unrounded
        word = decider_label(lab)
        reason = None
        if reasons:
            nxt, gen, finished = self.label_ids[LABELS.index(word)], [], False
            for _ in range(max_tokens):
                step = self._logits_last([nxt], cache)
                nxt = int(mx.argmax(step))
                if nxt in self.stop:
                    finished = True
                    break
                gen.append(nxt)
            reason = clean_reason(word + self.tok.decode(gen), finished)
        return {"scores": lab, "p": label_probs(lab), "label": word, "reason": reason, "prompt_tokens": len(ids),
                "cached_tokens": n}

    def score(self, prefix: str, items: list[dict], reasons: bool, max_reason_tokens: int) -> dict:
        t0 = time.time()
        mx.reset_peak_memory()
        pre_ids, all_ids = plan_prefix(self.tok, prefix, items)
        base = None
        if pre_ids:
            base = make_prompt_cache(self.inner)
            self._feed(pre_ids, base)
        results = []
        for it, ids in zip(items, all_ids):
            t = time.time()
            r = self.score_one(ids, len(pre_ids), base, reasons, max_reason_tokens)
            r.update(id=it["id"], seconds=round(time.time() - t, 2))
            results.append(r)
            self.done += 1
            self._check_peak()
            if self.done % self.clear_every == 0:
                mx.clear_cache()
        mx.clear_cache()
        self.max_peak = max(self.max_peak, mx.get_peak_memory())
        stats = {"items": len(items), "seconds": round(time.time() - t0, 2), "cached_tokens": len(pre_ids),
                 "peak_gb": round(mx.get_peak_memory() / GB, 3), "max_peak_gb": round(self.max_peak / GB, 3),
                 "active_gb": round(mx.get_active_memory() / GB, 3)}
        print(f"[mlx] {stats}", flush=True)
        return {"backend": self.name, "model": Path(self.model_dir).parent.name + "/" + Path(self.model_dir).name,
                "results": results, "stats": stats}

    def health(self) -> dict:
        return {"backend": self.name, "model": Path(self.model_dir).parent.name + "/" + Path(self.model_dir).name,
                "load_seconds": self.load_s, "max_peak_gb": round(self.max_peak / GB, 3)}


# ---------------------------------------------------------------- checks

def load_rows(path: str, n: int | None) -> list[dict]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return rows[:n] if n else rows


def _gate(sc: MlxScorer, a: argparse.Namespace, what: str) -> float:
    peak = sc.max_peak / GB
    if a.peak_gate_gb and peak > a.peak_gate_gb:
        raise SystemExit(f"FAIL: {what} peak {peak:.2f} GB over the {a.peak_gate_gb} GB gate")
    return peak


def check(a: argparse.Namespace) -> None:
    """Score N pairs one request each, then (with --shared-prefix K) K postings against one shared prefix
    in a single request, the way the live page calls it. Prints time per pair and the highest peak of all
    requests, and fails past --peak-gate-gb."""
    sc = MlxScorer(a.model, a.cache_limit_gb, a.memory_limit_gb, a.clear_every, a.peak_gate_gb)
    rows = load_rows(a.dev, a.n)
    print(f"loaded {a.model} in {sc.load_s} s; scoring {len(rows)} pairs")
    out = []
    for r in rows:
        prefix, suffix = lib.split_row(r["prompt"])
        res = sc.score(prefix, [{"id": r["pair_id"], "suffix": suffix}], a.reasons, 40)["results"][0]
        out.append(res)
        print(json.dumps({k: res[k] for k in ("id", "label", "scores", "prompt_tokens", "seconds")}), flush=True)
    peak = _gate(sc, a, "single-pair")
    print(f"single pairs: max peak {peak:.2f} GB, mean {sum(o['seconds'] for o in out) / max(len(out), 1):.2f} s per pair")
    if a.shared_prefix:
        many = load_rows(a.dev, max(a.shared_prefix, a.n))[: a.shared_prefix]
        prefix = lib.split_row(many[0]["prompt"])[0]
        items = [{"id": r["pair_id"], "suffix": lib.split_row(r["prompt"])[1]} for r in many]
        t = time.time()
        res = sc.score(prefix, items, a.reasons, 40)
        peak = _gate(sc, a, "shared-prefix")
        print(f"shared prefix: {len(items)} postings in one request in {time.time() - t:.1f} s "
              f"({res['stats']['cached_tokens']} cached tokens); max peak {peak:.2f} GB")


def parity(a: argparse.Namespace) -> None:
    """Compare label and label scores with a local_eval.py --out file for the same rows (its rows carry `scores`; the
    decider reads them). Gate (demo_lib.label_parity): label agreement of 98% or better, where a flip with the eval's
    s within NEAR_CUT of a cut-off is reported but not counted."""
    torch_rows = {r["pair_id"]: r for r in json.loads(Path(a.torch).read_text())["rows"]}
    sc = MlxScorer(a.model, a.cache_limit_gb, a.memory_limit_gb, a.clear_every, a.peak_gate_gb)
    rows = [r for r in load_rows(a.dev, None) if r["pair_id"] in torch_rows][: a.n]
    pairs = []
    for r in rows:
        prefix, suffix = lib.split_row(r["prompt"])
        res = sc.score(prefix, [{"id": r["pair_id"], "suffix": suffix}], False, 0)["results"][0]
        ref = torch_rows[r["pair_id"]]
        pairs.append((r["pair_id"], res["label"], res["scores"], ref["scores"]))
    res = lib.label_parity(pairs)
    lib.print_parity(res, "mlx")
    print(f"max peak {sc.max_peak / GB:.2f} GB")
    _gate(sc, a, "parity")
    if not res["ok"]:
        raise SystemExit(f"FAIL: {len(res['flips'])} real flips in {res['rows']} rows; agreement without near-cuts under 98%")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("serve", "check", "parity"):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True, help="merged HF-layout checkpoint (merge_lora.py)")
        s.add_argument("--cache-limit-gb", type=float, default=2.0)
        s.add_argument("--memory-limit-gb", type=float, default=None)
        s.add_argument("--clear-every", type=int, default=20)
        s.add_argument("--peak-gate-gb", type=float, default=None, help="abort when the peak passes this")
        if name == "serve":
            s.add_argument("--host", default="127.0.0.1")
            s.add_argument("--port", type=int, default=8765)
        else:
            s.add_argument("--dev", required=True, help="jsonl with pair_id and prompt")
            s.add_argument("--n", type=int, default=20)
            s.add_argument("--reasons", action="store_true")
        if name == "check":
            s.add_argument("--shared-prefix", type=int, default=0,
                           help="also score this many postings against one shared prefix in one request (the page's path)")
        if name == "parity":
            s.add_argument("--torch", required=True)
    a = ap.parse_args()
    if a.cmd == "serve":
        serve(MlxScorer(a.model, a.cache_limit_gb, a.memory_limit_gb, a.clear_every, a.peak_gate_gb),
              a.host, a.port)
    else:
        {"check": check, "parity": parity}[a.cmd](a)


if __name__ == "__main__":
    main()
