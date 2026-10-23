"""Local demo server for Gradus. Standard library only; it never imports model code.

    uv run python demo/server.py [--port 8077] [--scorer http://127.0.0.1:8765]
    uv run python demo/server.py precompute --persona demo/personas/alex_cs_f1.json --out demo/results/alex.json
    uv run python demo/server.py precompute --persona demo/personas/alex_cs_f1.json --work-auth citizen_or_pr --out demo/results/alex_citizen.json

It serves demo/index.html and a small JSON API, and asks a scorer (scripts/demo_backend_torch.py on
oci-pi, or scripts/demo_score.py on the Mac) for the three label scores of each posting over HTTP. The label shown is
decided here from those scores, live or precomputed: run B's decider (demo_lib.model_label), then the H'' rules on the
posting's Adaption-extracted card (demo_lib.elig_final), as eval/fresh_test4_result.json scored the shipped system.

    GET  /                 the page
    GET  /api/health       server + scorer status
    GET  /api/pool         the posting pool (id, title, company, location)
    GET  /api/personas     saved personas (resume + profile)      GET /api/persona?name=...
    GET  /api/results      precomputed result files
    POST /api/extract      raw .pdf/.docx/.txt body, header X-Filename (URI-encoded) -> {"text"}
    POST /api/profile      profile form -> {"profile": rendered text}
    POST /score            {"resume", "profile": {form}, "live": 20, "precomputed": name|null,
                            "redact": true, "reasons": true} -> {"buckets": {...}, "stats": {...}}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import demo_lib as lib  # noqa: E402

MAX_UPLOAD = 10 * 1024 * 1024


class Demo:
    def __init__(self, scorer_url: str, pool_path: Path, results_dir: Path, personas_dir: Path) -> None:
        self.scorer_url = scorer_url.rstrip("/")
        self.pool_path, self.results_dir, self.personas_dir = pool_path, results_dir, personas_dir
        self.pool = lib.load_pool(pool_path) if pool_path.exists() else []
        self.by_id = {r["id"]: r for r in self.pool}

    # -- scorer ---------------------------------------------------------
    def _call(self, path: str, body: dict | None = None, timeout: float = 1800) -> dict:
        req = urllib.request.Request(self.scorer_url + path, json.dumps(body).encode() if body is not None else None,
                                     {"Content-Type": "application/json"})
        try:
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"scorer said {e.code}: {e.read()[:300].decode(errors='replace')}") from e
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"cannot reach the scorer at {self.scorer_url}: {e}") from e

    def health(self) -> dict:
        try:
            scorer = self._call("/health", timeout=5)
        except RuntimeError as e:
            scorer = {"ok": False, "error": str(e)}
        return {"ok": True, "pool": len(self.pool), "scorer_url": self.scorer_url, "scorer": scorer}

    # -- files ----------------------------------------------------------
    def _json_files(self, folder: Path) -> list[str]:
        return sorted(p.stem for p in folder.glob("*.json")) if folder.exists() else []

    def _results_file(self, name: str) -> dict:
        path = self.results_dir / f"{Path(name).name}.json"
        if not path.exists():
            raise ValueError(f"no precomputed results named {name!r}")
        return json.loads(path.read_text())

    # Shown when a chosen precomputed file does not apply, so a silent fall to live scoring is visible.
    PREFIX_MISMATCH = ("precomputed file {name!r} was made for a different resume or profile (or redaction setting), "
                       "so none of its scores were used")

    def results(self, name: str, prefix: str) -> tuple[dict, str, str]:
        """Precomputed scores, the model that made them, and a warning when the file does not apply. Valid
        only for this exact resume + profile (same prompt prefix), and a score is used only while its
        posting's text is unchanged."""
        data = self._results_file(name)
        if data.get("prefix_sha") != lib.prefix_sha(prefix):
            return {}, data.get("model", ""), self.PREFIX_MISMATCH.format(name=name)
        hits = {}
        for pid, hit in data["results"].items():
            row = self.by_id.get(pid)
            if row is not None and hit["text_sha"] == lib.text_sha(row["text"]):
                hits[pid] = hit
        return hits, data.get("model", ""), ""

    def results_index(self) -> list[dict]:
        """Each precomputed file with its persona, model, and how many of its scores still match the pool."""
        out = []
        for name in self._json_files(self.results_dir):
            try:
                data = self._results_file(name)
            except (ValueError, json.JSONDecodeError):
                continue
            res = data.get("results", {})
            fresh = sum(1 for pid, h in res.items() if pid in self.by_id
                        and h["text_sha"] == lib.text_sha(self.by_id[pid]["text"]))
            out.append({"name": name, "persona": data.get("persona"), "model": data.get("model", ""),
                        "scores": len(res), "match_pool": fresh,
                        "warning": "" if fresh == len(res) else f"only {fresh} of {len(res)} scores match the current pool"})
        return out

    # -- scoring --------------------------------------------------------
    def prepare(self, req: dict) -> dict:
        resume = req.get("resume", "").strip()
        if len(resume) < 40:
            raise ValueError("paste or upload a resume first")
        if req.get("redact", True):
            resume = lib.redact(resume)
        form = req.get("profile") or {}
        prof = lib.profile_text(form)
        ids = req.get("posting_ids")
        rows = [self.by_id[i] for i in ids if i in self.by_id] if ids else self.pool
        # H'' (eval/selection_cycle3.json): the rules on each posting's Adaption-extracted card. A posting they set to
        # Skip whatever the model says is decided here and never sent to the model; the salary cap runs after it.
        cand, cats = lib.candidate_from_form(form), lib.elig_cats(form)
        survivors, cards, verdicts = [], [], {}
        for row in rows:
            v = lib.elig_verdict(cand, row)
            why = lib.elig_skip(v, cats)
            if why:
                cards.append(lib.rule_card(row, why, v, cats))
            else:
                survivors.append(row)
                verdicts[row["id"]] = v
        prefix, _ = lib.split_prompt(resume, prof, "")
        return {"resume": resume, "profile": prof, "prefix": prefix, "survivors": survivors, "cards": cards,
                "verdicts": verdicts, "cats": cats, "skipped": len(cards), "total": len(rows),
                "resume_seen": lib.resume_in_prompt(prefix)}

    def score(self, req: dict) -> dict:
        t0 = time.time()
        p = self.prepare(req)
        cards, survivors = p["cards"], p["survivors"]
        pre, pre_model, pre_warning = self.results(req["precomputed"], p["prefix"]) if req.get("precomputed") \
            else ({}, "", "")
        from_file, todo = 0, []
        for row in survivors:
            hit = pre.get(row["id"])
            if hit and hit.get("scores"):  # the label is read from the scores; a hit without them is not used
                cards.append(lib.scored_card(row, label_word=lib.model_label(hit["scores"]), p=hit["p"],
                                             reason=hit.get("reason"), source="precomputed", scores=hit["scores"],
                                             verdict=p["verdicts"][row["id"]], cats=p["cats"]))
                from_file += 1
            else:
                todo.append(row)
        live = max(0, int(req.get("live", 20)))
        # Citizens-only postings that survive the rules (the candidate is a citizen or PR) are scored on top
        # of the live budget, so flipping work authorization adds them without pushing other postings out.
        extra = [r for r in todo if r["card"].get("citizens_only")]
        rest = [r for r in todo if not r["card"].get("citizens_only")]
        batch = extra + rest[:live]
        skipped = rest[live:]
        scorer_info, scorer_stats = {}, {}
        if batch:
            items = [{"id": r["id"], "suffix": lib.split_prompt(p["resume"], p["profile"], r["text"])[1]} for r in batch]
            try:
                out = self._call("/score", {"prefix": p["prefix"], "items": items,
                                            "reasons": bool(req.get("reasons", True)), "max_reason_tokens": 40})
            except RuntimeError as e:
                raise RuntimeError(f"{pre_warning}; {e}" if pre_warning else str(e)) from e
            scorer_info, scorer_stats = {"backend": out.get("backend"), "model": out.get("model")}, out.get("stats", {})
            by_id = {r["id"]: r for r in batch}
            for res in out["results"]:
                cards.append(lib.scored_card(by_id[res["id"]], label_word=lib.model_label(res["scores"]), p=res["p"],
                                             reason=res.get("reason"), source="model", scores=res["scores"],
                                             verdict=p["verdicts"][res["id"]], cats=p["cats"]))
        warnings = [pre_warning] if pre_warning else []
        if from_file and scorer_info.get("model") and pre_model and pre_model != scorer_info["model"]:
            warnings.append(f"precomputed scores come from {pre_model}, live scores from {scorer_info['model']}")
        warning = "; ".join(warnings)
        return {"buckets": lib.to_buckets(cards), "profile": p["profile"], "resume": p["resume_seen"],
                "not_scored": [lib.pool_summary(r) for r in skipped],
                "stats": {"pool": p["total"], "rule_skipped": p["skipped"],
                          "rule_capped": sum(c.get("rule", {}).get("kind") == "salary_cap" for c in cards),
                          "precomputed": from_file,
                          "precomputed_model": pre_model if from_file else "", "scored_live": len(batch),
                          "citizens_only_extra": len(extra), "not_scored": len(skipped), "warning": warning,
                          "live_seconds": round(time.time() - t0, 2), "scorer": scorer_info, "scorer_stats": scorer_stats}}

    def precompute(self, persona_path: Path, out_path: Path, chunk: int = 20, reasons_per_label: int = 8,
                   work_auth: str | None = None) -> None:
        """Score every posting the H'' rules let through, then add reasons to the most confident
        reasons_per_label postings of each label, so all three columns show reasons. The file holds the
        model's scores and its decider labels; the page reads the label from the scores again and applies the H''
        salary cap after it."""
        persona = json.loads(persona_path.read_text())
        form = dict(persona["profile"])
        if work_auth:
            form["work_auth"] = work_auth
        p = self.prepare({"resume": persona["resume"], "profile": form})
        results: dict[str, dict] = {}
        rows = p["survivors"]
        by_id = {r["id"]: r for r in rows}
        print(f"scoring {len(rows)} postings ({p['skipped']} set to Skip by the H'' rules before the model)")
        model = backend = ""
        for k in range(0, len(rows), chunk):
            part = rows[k:k + chunk]
            items = [{"id": r["id"], "suffix": lib.split_prompt(p["resume"], p["profile"], r["text"])[1]} for r in part]
            out = self._call("/score", {"prefix": p["prefix"], "items": items, "reasons": False})
            model, backend = out.get("model", ""), out.get("backend", "")
            results.update({r["id"]: {"label": r["label"], "p": r["p"], "scores": r["scores"], "reason": "",
                                      "text_sha": lib.text_sha(by_id[r["id"]]["text"])} for r in out["results"]})
            print(f"  {min(k + chunk, len(rows))}/{len(rows)}", flush=True)
        top: list[str] = []
        for word in ("Keep", "Explore", "Skip"):
            mine = sorted((i for i in results if results[i]["label"] == word), key=lambda i: -results[i]["p"][word])
            top += mine[:reasons_per_label]
        print(f"generating {len(top)} reasons, up to {reasons_per_label} per label")
        for k in range(0, len(top), chunk):
            items = [{"id": i, "suffix": lib.split_prompt(p["resume"], p["profile"], by_id[i]["text"])[1]} for i in top[k:k + chunk]]
            for r in self._call("/score", {"prefix": p["prefix"], "items": items, "reasons": True})["results"]:
                results[r["id"]]["reason"] = r["reason"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"name": out_path.stem, "persona": persona.get("name"), "work_auth": form.get("work_auth"),
                                        "model": model, "backend": backend, "prefix_sha": lib.prefix_sha(p["prefix"]),
                                        "pool": str(self.pool_path), "profile": p["profile"], "results": results}, indent=1))
        counts = {w: sum(1 for i in results if results[i]["label"] == w) for w in ("Keep", "Explore", "Skip")}
        with_reason = {w: sum(1 for i in results if results[i]["label"] == w and results[i]["reason"]) for w in counts}
        print(f"wrote {out_path}: {len(results)} scores " + ", ".join(f"{w} {counts[w]} ({with_reason[w]} with a reason)" for w in counts))

def make_handler(demo: Demo, index: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj).encode())

        def do_GET(self) -> None:
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    self._send(200, index.read_bytes(), "text/html; charset=utf-8")
                elif u.path == "/og.jpg" and (index.parent / "og.jpg").exists():  # the link preview card
                    self._send(200, (index.parent / "og.jpg").read_bytes(), "image/jpeg")
                elif u.path == "/api/health":
                    self._json(200, demo.health())
                elif u.path == "/api/pool":
                    self._json(200, {"postings": [lib.pool_summary(r) for r in demo.pool]})
                elif u.path == "/api/personas":
                    self._json(200, {"personas": demo._json_files(demo.personas_dir)})
                elif u.path == "/api/persona":
                    self._json(200, json.loads((demo.personas_dir / f"{Path(q.get('name', '')).name}.json").read_text()))
                elif u.path == "/api/results":
                    self._json(200, {"results": demo._json_files(demo.results_dir), "files": demo.results_index()})
                else:
                    self._json(404, {"error": "not found"})
            except (FileNotFoundError, ValueError) as e:
                self._json(404, {"error": str(e)})
            except Exception as e:  # never drop the connection without an answer
                traceback.print_exc()
                self._json(500, {"error": f"server error: {type(e).__name__}: {e}"})

        def do_POST(self) -> None:
            u = urlparse(self.path)
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > MAX_UPLOAD:
                    return self._json(413, {"error": "file too large (10 MB max)"})
                raw = self.rfile.read(n)
                if u.path == "/api/extract":
                    name = unquote(self.headers.get("X-Filename", "resume.txt"))  # the page sends it URI-encoded
                    return self._json(200, {"text": lib.extract_text(name, raw)})
                req = json.loads(raw or b"{}")
                if u.path == "/api/profile":
                    return self._json(200, {"profile": lib.profile_text(req)})
                if u.path == "/score":
                    return self._json(200, demo.score(req))
                self._json(404, {"error": "not found"})
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                self._json(400, {"error": str(e) or type(e).__name__})
            except RuntimeError as e:  # scorer unreachable or failed
                self._json(502, {"error": str(e)})
            except Exception as e:  # never drop the connection without an answer
                traceback.print_exc()
                self._json(500, {"error": f"server error: {type(e).__name__}: {e}"})

        def log_message(self, fmt: str, *args) -> None:
            print("[demo]", fmt % args, flush=True)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("serve", "precompute"), default="serve")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--scorer", default="http://127.0.0.1:8765")
    ap.add_argument("--pool", default=str(HERE / "local" / "pool.jsonl"))
    ap.add_argument("--persona")
    ap.add_argument("--out")
    ap.add_argument("--reasons-per-label", type=int, default=8,
                    help="precompute: reasons for the N most confident postings of each label (about 3N generations)")
    ap.add_argument("--work-auth", help="precompute: score the persona with this work authorization instead")
    a = ap.parse_args()
    demo = Demo(a.scorer, Path(a.pool), HERE / "results", HERE / "personas")
    if a.cmd == "precompute":
        if not (a.persona and a.out):
            ap.error("precompute needs --persona and --out")
        return demo.precompute(Path(a.persona), Path(a.out), reasons_per_label=a.reasons_per_label, work_auth=a.work_auth)
    if not demo.pool:
        print(f"warning: no posting pool at {a.pool}; run `uv run python demo/build_pool.py`", file=sys.stderr)
    print(f"demo on http://{a.host}:{a.port}  (pool {len(demo.pool)}, scorer {a.scorer})", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(demo, HERE / "index.html")).serve_forever()


if __name__ == "__main__":
    main()
