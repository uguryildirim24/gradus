"""The web side of the Modal demo (demo/modal_app.py), kept free of Modal so it is tested on the Mac.

It wraps demo/server.py's page and API and adds what a public link needs:
  - /api/health reads the GPU's state from the ledger Dict and never wakes the GPU;
  - POST /score is refused with 402 once the month's budget cannot cover the request, answered with 202
    ("warming up") while the GPU starts, limited per IP (429), bounded in queue (429) and in size (400/413);
  - nothing from a request is logged: log lines carry the method, the path without query and the status.
"""
from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import server as srv
import spend


class Capped(Exception):
    """The GPU refused the call: the month's budget cannot cover it."""


@dataclass
class Limits:
    max_live: int = 25          # postings scored per request (citizens-only extras come on top, up to hard_items)
    hard_items: int = 40        # absolute ceiling on postings sent to the GPU in one call
    max_resume: int = 20_000    # characters
    max_field: int = 400        # characters per profile field
    max_body: int = 256 * 1024  # bytes of JSON per /score request
    reason_tokens: int = 40
    rate: tuple[int, float] = (8, 600)  # scoring requests per client per window (s)
    queue: tuple[int, int] = (1, 3)     # calls running at once, calls waiting
    queue_wait_s: float = 30
    cold_start_s: int = 240


@dataclass
class Hooks:
    """What the Modal app provides. gpu_state() -> {"state": ready|loading|asleep|capped|error, ...};
    budget(warm) -> {"admit": bool, "month": str, ...}; score(prefix, items, reasons, max_tokens) -> scorer
    JSON (or {"capped": True} / {"error": str}); wake() starts the GPU without sending it any request data."""
    gpu_state: Callable[[], dict]
    budget: Callable[[bool], dict]
    score: Callable[[str, list, bool, int], dict]
    wake: Callable[[], None]
    say: Callable[..., None] = print
    model: str = ""
    links: dict = field(default_factory=dict)
    timeout_errors: tuple = ()


def check_request(req: Any, lim: Limits) -> dict:
    if not isinstance(req, dict):
        raise ValueError("bad request")
    if len(str(req.get("resume", ""))) > lim.max_resume:
        raise ValueError(f"resume too long for the demo (max {lim.max_resume:,} characters)")
    form = req.get("profile") or {}
    if not isinstance(form, dict) or any(len(str(v)) > lim.max_field for v in form.values()):
        raise ValueError(f"a profile field is too long (max {lim.max_field} characters)")
    ids = req.get("posting_ids")
    if ids is not None and (not isinstance(ids, list) or len(ids) > lim.hard_items):
        raise ValueError(f"too many postings (max {lim.hard_items})")
    try:
        live = int(req.get("live", 20) or 20)
    except (TypeError, ValueError):
        live = 20
    req["live"] = max(1, min(live, lim.max_live))
    return req


def make_server(hooks: Hooks, lim: Limits, pool: Path, code: Path, host: str = "0.0.0.0",
                port: int = 8000) -> tuple[ThreadingHTTPServer, srv.Demo]:
    # server.py prints a traceback on unexpected errors; here only the exception type reaches the logs.
    srv.traceback = types.SimpleNamespace(print_exc=lambda: hooks.say("web error:", sys.exc_info()[0].__name__))
    limiter = spend.RateLimiter(lim.rate[0], lim.rate[1], os.urandom(32))
    gate = spend.Gate(*lim.queue)
    capped_body = {"error": "This month's demo budget is used up.", "capped": True, "links": hooks.links}

    class ModalDemo(srv.Demo):
        def _call(self, path: str, body: dict | None = None, timeout: float = 0) -> dict:
            if path == "/health":  # never wakes the GPU
                g = hooks.gpu_state()
                return {"ok": g["state"] == "ready", "backend": "torch-cuda", "model": hooks.model, **g}
            if path != "/score" or body is None:
                raise RuntimeError("unknown scorer call")
            if len(body["items"]) > lim.hard_items:
                raise ValueError(f"too many postings in one request (max {lim.hard_items})")
            try:
                out = hooks.score(body["prefix"], body["items"], bool(body.get("reasons")),
                                  int(body.get("max_reason_tokens", lim.reason_tokens)))
            except hooks.timeout_errors as e:
                raise RuntimeError("the model took too long on this request; try fewer postings") from e
            if out.get("capped"):
                raise Capped()
            if out.get("error"):
                raise RuntimeError(out["error"])
            return out

    demo = ModalDemo("modal", pool, code / "demo" / "results", code / "demo" / "personas")
    Base = srv.make_handler(demo, code / "demo" / "index.html")

    def health() -> dict:
        g = hooks.gpu_state()
        warm = g["state"] == "ready"
        b = hooks.budget(warm)
        return {"ok": True, "pool": len(demo.pool), "scorer_url": "modal",
                "scorer": {"ok": warm, "backend": "torch-cuda (Modal L4)", "model": hooks.model, **g,
                           "cold_start_s": lim.cold_start_s},
                "budget": {"capped": g["state"] == "capped" or not b["admit"], "month": b.get("month")},
                "links": hooks.links, "limits": {"max_live": lim.max_live, "max_resume_chars": lim.max_resume}}

    class Handler(Base):
        def _client(self) -> str:
            fwd = self.headers.get("X-Forwarded-For", "")
            # the right-most entry is the one Modal's proxy added; a visitor can write anything to its left
            return fwd.split(",")[-1].strip() or self.client_address[0]

        def do_GET(self) -> None:
            if urlparse(self.path).path == "/api/health":
                try:
                    return self._json(200, health())
                except Exception as e:
                    hooks.say("health failed:", type(e).__name__)
                    return self._json(500, {"error": "health check failed"})
            return super().do_GET()

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/score":
                return super().do_POST()
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > lim.max_body:
                    return self._json(413, {"error": "request too large"})
                req = check_request(json.loads(self.rfile.read(n) or b"{}"), lim)
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                return self._json(400, {"error": str(e) or "bad request"})
            if not limiter.allow(self._client()):
                return self._json(429, {"error": "too many requests from here; try again in a few minutes"})
            g = hooks.gpu_state()
            warm = g["state"] == "ready"
            if g["state"] == "capped" or not hooks.budget(warm)["admit"]:
                return self._json(402, capped_body)
            if not warm:
                if g["state"] != "loading":
                    hooks.wake()
                    hooks.say("web: waking the gpu")
                return self._json(202, {"warming": True, "state": g["state"], "cold_start_s": lim.cold_start_s})
            if not gate.enter(lim.queue_wait_s):
                return self._json(429, {"error": "the demo is busy; try again in a minute"})
            try:
                return self._json(200, demo.score(req))
            except Capped:
                return self._json(402, capped_body)
            except (ValueError, KeyError, TypeError) as e:
                return self._json(400, {"error": str(e) or type(e).__name__})
            except RuntimeError as e:
                return self._json(502, {"error": str(e)})
            except Exception as e:
                hooks.say("web: score failed:", type(e).__name__)
                return self._json(500, {"error": f"server error: {type(e).__name__}"})
            finally:
                gate.leave()

        def log_message(self, fmt: str, *args) -> None:
            """Method, path without its query, and status. Nothing else from the request."""
            parts = str(args[0]).split(" ") if args else []
            if len(parts) >= 2:
                hooks.say("web:", parts[0], parts[1].split("?")[0], *(str(a) for a in args[1:2]))

    return ThreadingHTTPServer((host, port), Handler), demo
