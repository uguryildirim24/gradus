"""Shared pieces of the two demo scorers (demo_score.py on the Mac, demo_backend_torch.py on oci-pi).

No model library (numpy through evalstats), so it is unit-tested on the Mac. It holds
  - the prompt framing (the Qwen3.5 chat template with thinking off, as local_eval.py and training use it),
  - the shared-prefix planner (which leading tokens every posting in a request has in common),
  - label maths (three float32 label scores -> probabilities at temperature 1 and the label word through run B's
    decider, matcher.decider, so a demo label is the shipped readout's label before the eligibility rules) and reason
    cleanup,
  - the small JSON HTTP API both scorers speak:

    GET  /health -> {"ok": true, "backend": ..., "model": ..., ...}
    POST /score  {"prefix": str, "items": [{"id": str, "suffix": str}], "reasons": bool, "max_reason_tokens": int}
              -> {"backend", "model", "results": [{"id", "scores": {"Keep","Explore","Skip"}, "p": {...},
                  "label": "Keep"|"Explore"|"Skip", "reason": str|None, "prompt_tokens": int,
                  "cached_tokens": int, "seconds": float}], "stats": {...}}

`prefix` + `suffix` is the exact matcher.prompt_b text (rubric, resume and profile are the prefix, the posting and
"Answer:" the suffix), so the same prefix is reused across postings. `scores` are the three float32 label
scores local_eval.py writes (hidden state times the label rows of the output head, evalstats.LabelHead).

On oci-pi copy src/matcher/evalstats.py and src/matcher/decider.py beside this file (as colab_eval.py does for
local_eval.py).
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

try:
    from matcher.decider import decide_dict
    from matcher.evalstats import LABELS, LabelHead, label_token_ids, softmax  # noqa: F401 (for the scorers)
except ModuleNotFoundError:  # oci-pi and Modal's flat copies: evalstats.py and decider.py beside this file or under src/
    import sys

    _here = Path(__file__).resolve().parent
    sys.path[:0] = [str(_here.parent / "src"), str(_here)]
    try:
        from matcher.decider import decide_dict
        from matcher.evalstats import LABELS, LabelHead, label_token_ids, softmax  # noqa: F401 (for the scorers)
    except ModuleNotFoundError:
        from decider import decide_dict
        from evalstats import LABELS, LabelHead, label_token_ids, softmax  # noqa: F401 (for the scorers)

LABEL_BUCKET = {"Keep": "For you", "Explore": "Explore", "Skip": "Excluded"}
MAX_BODY = 8 * 1024 * 1024


def frame(tok: Any, prompt: str) -> str:
    """Prompt as one user turn, assistant turn open, thinking off (how a chat-format SFT row is rendered)."""
    return tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=False)


def label_probs(scores: dict[str, float]) -> dict[str, float]:
    """Softmax over the three label scores only at temperature 1 (evalstats.softmax, the decider's), rounded for
    display. The label comes from the unrounded scores (decider_label), never from these."""
    p = softmax([scores[w] for w in LABELS], 1.0)
    return {w: round(float(v), 4) for w, v in zip(LABELS, p)}


def decider_label(scores: dict[str, float]) -> str:
    """Run B's decider (matcher.decider): s = P(Keep) + 0.5 P(Explore) at temperature 1, s >= T_HI Keep, s < T_LO
    Skip, else Explore. Pass the unrounded float32 scores. The scorers force this word as the reason's first token,
    so a reason explains the label the page shows before the eligibility rules."""
    return decide_dict(scores)


def clean_reason(text: str, complete: bool = True) -> str:
    """The generated completion is 'Keep — reasons; reasons'. Return just the reasons, one line:
    label word and separator dropped, repeated clauses dropped, at most 4 clauses (what training
    writes), and a clause cut off by the token limit dropped when others remain."""
    line = text.strip().split("\n")[0].strip()
    line = re.sub(r"^(Keep|Explore|Skip)\b\s*[—–:\-]*\s*", "", line)
    clauses = [c.strip() for c in line.rstrip(".").split(";") if c.strip()]
    if not complete and len(clauses) > 1:
        clauses = clauses[:-1]
    seen: list[str] = []
    for c in clauses:
        if c not in seen:
            seen.append(c)
    return "; ".join(seen[:4])


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def plan_prefix(tok: Any, prefix: str, items: list[dict]) -> tuple[list[int], list[list[int]]]:
    """Tokenize every full prompt; return (shared_prefix_ids, per-item full ids).

    The shared part is the longest token run all prompts agree on, held back by one token from the
    end of the prefix text, because BPE can merge the last prefix token with the first posting token.
    Every item's ids start with the returned prefix ids, by construction."""
    fulls = [frame(tok, prefix + it["suffix"]) for it in items]
    ids = [tok.encode(f, add_special_tokens=False) for f in fulls]
    if not ids:
        return [], []
    at = fulls[0].find(prefix)
    if at < 0:
        return [], ids
    ptext = fulls[0][: at + len(prefix)]
    limit = len(tok.encode(ptext, add_special_tokens=False)) - 1
    n = limit
    for other in ids[1:]:
        n = min(n, common_prefix_len(ids[0], other))
    n = max(n, 0)
    return ids[0][:n], ids


class Scorer(Protocol):
    name: str

    def health(self) -> dict: ...

    def score(self, prefix: str, items: list[dict], reasons: bool, max_reason_tokens: int) -> dict: ...


def serve(scorer: Scorer, host: str, port: int) -> None:
    lock = threading.Lock()  # one request at a time: the model and its memory are the bottleneck

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True, **scorer.health()})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/score":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > MAX_BODY:
                    return self._send(413, {"error": "body too large"})
                req = json.loads(self.rfile.read(n))
                prefix, items = req["prefix"], req["items"]
                assert isinstance(prefix, str) and all("id" in i and "suffix" in i for i in items)
            except Exception as e:  # malformed request
                return self._send(400, {"error": f"bad request: {e}"})
            try:
                with lock:
                    out = scorer.score(prefix, items, bool(req.get("reasons", False)),
                                       int(req.get("max_reason_tokens", 48)))
                self._send(200, out)
            except Exception as e:
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt: str, *args: Any) -> None:
            print("[scorer]", fmt % args, flush=True)

    print(f"scorer {scorer.name} listening on http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()

