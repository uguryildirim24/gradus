"""Demo scorer on oci-pi: transformers on CPU, same JSON API as demo_score.py (see demo_common.py).

Runs on oci-pi, never the Mac. The Qwen3.5 linear-attention reference kernels on CPU are loop-bound
and do not speed up with more threads (local_eval.py measured ~26 s per 1,200-token prompt), so this
service runs WORKERS single-threaded processes and splits each request across them. WORKERS is the
total thread budget (default 2).

    ~/matcher-demo/.venv/bin/python demo_backend_torch.py --model ~/matcher-demo/model --port 8765

The Mac reaches it through `ssh -L 8765:127.0.0.1:8765 oci-pi`. Copy src/matcher/evalstats.py beside
this file and demo_common.py, and src/matcher/decider.py beside it too. Each worker prefills the request's shared
prefix (rubric + resume + profile) once, deep-copies that cache per posting, and reads the three label scores at the
last position the way local_eval.py does (evalstats.LabelHead: float32 hidden state times the float32 label rows of
the head); the label is run B's decider on them (demo_common.decider_label). With --reasons that label is forced as
the first token and a short reason is generated greedily from the same cache.
"""
from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

from demo_common import LABELS, LabelHead, clean_reason, decider_label, label_probs, plan_prefix, serve

_STATE: dict = {}


def _init(model_dir: str, device: str = "cpu", dtype: str = "float32") -> None:
    """Load once per process. CPU float32 is the oci-pi service; the Modal demo calls the same function
    with device="cuda" (demo/modal_app.py), so both score through _score_chunk below."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cpu":
        torch.set_num_threads(1)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype)).to(device).eval()
    _attach(torch, tok, model, device)


def _attach(torch, tok, model, device: str) -> None:
    """Hold a loaded model for _score_chunk (the unit tests attach a tiny fake one here)."""
    head = LabelHead(model, tok)
    _STATE.update(torch=torch, tok=tok, model=model, device=device, head=head, label_ids=head.ids,
                  stop={tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})


def _score_chunk(prefix: str, items: list[dict], reasons: bool, max_tokens: int) -> list[dict]:
    torch, tok, model = _STATE["torch"], _STATE["tok"], _STATE["model"]
    dev = _STATE.get("device", "cpu")

    def t(x: list) -> "torch.Tensor":
        return torch.tensor(x, device=dev)

    out = []
    with torch.inference_mode():
        pre_ids, all_ids = plan_prefix(tok, prefix, items)
        cache = None
        if pre_ids:
            cache = model(input_ids=t([pre_ids]), use_cache=True, logits_to_keep=1).past_key_values
        for it, ids in zip(items, all_ids):
            t0 = time.time()
            n = len(pre_ids) if cache is not None else 0
            c = copy.deepcopy(cache) if cache is not None else None
            res = model(input_ids=t([ids[n:]]), past_key_values=c, use_cache=True, logits_to_keep=1)
            lab = dict(zip(LABELS, _STATE["head"].scores()[0].cpu().tolist()))  # float32, unrounded
            word = decider_label(lab)
            reason = None
            if reasons:
                nxt = _STATE["label_ids"][LABELS.index(word)]
                gen, done = [], False
                cache_g = res.past_key_values
                for _ in range(max_tokens):
                    r = model(input_ids=t([[nxt]]), past_key_values=cache_g, use_cache=True)
                    cache_g = r.past_key_values
                    nxt = int(r.logits[0, -1].argmax())
                    if nxt in _STATE["stop"]:
                        done = True
                        break
                    gen.append(nxt)
                reason = clean_reason(word + tok.decode(gen, skip_special_tokens=True), done)
            out.append({"id": it["id"], "scores": lab, "p": label_probs(lab), "label": word, "reason": reason,
                        "prompt_tokens": len(ids), "cached_tokens": n, "seconds": round(time.time() - t0, 2)})
    return out


class TorchScorer:
    def __init__(self, model_dir: str, workers: int) -> None:
        self.name = "torch-cpu"
        self.model_dir = model_dir
        self.workers = workers
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                        initializer=_init, initargs=(model_dir, "cpu", "float32"))
        t0 = time.time()
        list(self.pool.map(int, range(workers)))  # forces the workers up (loads happen in the initializer)
        for f in [self.pool.submit(_warm) for _ in range(workers)]:
            f.result()
        self.load_s = round(time.time() - t0, 1)

    def health(self) -> dict:
        return {"backend": self.name, "model": os.path.basename(os.path.dirname(self.model_dir.rstrip("/")) or self.model_dir)
                + "/" + os.path.basename(self.model_dir.rstrip("/")), "workers": self.workers, "threads_total": self.workers,
                "load_seconds": self.load_s}

    def score(self, prefix: str, items: list[dict], reasons: bool, max_reason_tokens: int) -> dict:
        t0 = time.time()
        chunks = [items[k::self.workers] for k in range(self.workers) if items[k::self.workers]]
        parts = [f.result() for f in [self.pool.submit(_score_chunk, prefix, ch, reasons, max_reason_tokens) for ch in chunks]]
        by_id = {r["id"]: r for p in parts for r in p}
        results = [by_id[it["id"]] for it in items]
        return {"backend": self.name, "model": self.health()["model"], "results": results,
                "stats": {"items": len(items), "seconds": round(time.time() - t0, 2), "workers": len(chunks),
                          "cached_tokens": results[0]["cached_tokens"] if results else 0}}


def _warm() -> int:
    _STATE["torch"].inference_mode()
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--workers", type=int, default=2,
                    help="processes of one thread each (about 4-5 GB RAM apiece); oci-pi went unreachable 15 minutes "
                         "after a 6-worker start, so start at 2 and raise only once the box stays up")
    a = ap.parse_args()
    serve(TorchScorer(os.path.expanduser(a.model), a.workers), a.host, a.port)


if __name__ == "__main__":
    main()
