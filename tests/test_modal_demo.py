"""Tests for the Modal demo's spend cap (demo/spend.py) and web wrapper (demo/modal_web.py). No Modal, no model."""
import ast
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(ROOT / "scripts"))

import demo_common as dc  # noqa: E402
import modal_web as mw  # noqa: E402
import server as srv  # noqa: E402
import spend  # noqa: E402

from matcher.decider import keep_score  # noqa: E402

from test_demo import FORM, RESUME, posting  # noqa: E402


# ---------------------------------------------------------------- spend.py

def test_rates_follow_modal_prices_and_minimums():
    assert spend.rate_per_s("L4", 2, 12) * 3600 == pytest.approx(0.80 + 2 * 0.0473 + 12 * 0.008)
    assert spend.rate_per_s(None, 0.01, 0.01) * 3600 == pytest.approx(0.125 * 0.0473 + 0.125 * 0.008)


def test_ledger_sums_one_key_per_container_by_month():
    store: dict = {}
    led = spend.Ledger(store)
    oct_ = 1_791_000_000.0  # 2026-10-03 UTC
    assert spend.month_of(oct_) == "2026-10"
    led.record("gpu", "ta-1", oct_, oct_ + 100, 0.001, 20)  # (100 + 20) s * 0.001
    led.record("gpu", "ta-1", oct_, oct_ + 200, 0.001, 20)  # same container: overwrites, never adds twice
    led.record("web", "ta-2", oct_, oct_ + 1000, 0.00001, 20)
    led.record("gpu", "ta-3", oct_ + 40 * 86400, oct_ + 40 * 86400 + 50, 0.001, 0)  # next month
    store["gpu_state"] = {"state": "ready"}  # not a ledger key
    assert led.spent("2026-10") == pytest.approx(0.22 + 0.0102)
    assert led.spent("2026-10", exclude=spend.Ledger.key("2026-10", "gpu", "ta-1")) == pytest.approx(0.0102)
    assert led.by_kind("2026-10") == {"gpu": pytest.approx(0.22), "web": pytest.approx(0.0102)}
    assert led.spent("2026-11") == pytest.approx(0.05)
    assert led.refresh(["2026-10"]) == 2


def test_cap_per_month():
    assert spend.cap_for("2026-10", {"2026-10": 0}, 15) == 0
    assert spend.cap_for("2026-11", {"2026-10": 0}, 15) == 15


def test_admission_counts_cold_start_request_and_idle_scaledown():
    rate = spend.rate_per_s("L4", 2, 12)
    reserve = spend.gpu_reserve_usd(rate, 110, 120)
    assert reserve == pytest.approx((110 + 120 + spend.MARGIN_S) * rate)
    kw = dict(rate=rate, request_s=110, scaledown_s=120)
    assert spend.admit_gpu(cap=1.0, spent_others=0.5, own_usd=0.1, **kw)
    assert not spend.admit_gpu(cap=1.0, spent_others=0.9, own_usd=0.1, **kw)
    assert not spend.admit_gpu(cap=0.0, spent_others=0.0, own_usd=0.0, **kw)  # a zero cap never starts a GPU
    cold_extra = (240 + spend.MARGIN_S) * rate
    cap = reserve + cold_extra - 1e-6
    assert spend.admit_web(cap=cap, spent=0, warm=True, cold_s=240, **kw)
    assert not spend.admit_web(cap=cap, spent=0, warm=False, cold_s=240, **kw)


def test_serialized_calls_cannot_spend_past_the_cap():
    """Many requests reach the single GPU container one at a time; each re-checks with its exact cost so far.
    Simulate the worst case (every admitted call runs to its timeout) and check the total never passes the cap."""
    rate, request_s, scaledown_s, cap = spend.rate_per_s("L4", 2, 12), 110, 120, 0.25
    own, admitted = 0.0, 0
    for _ in range(50):
        if not spend.admit_gpu(cap=cap, spent_others=0, own_usd=own, rate=rate, request_s=request_s,
                               scaledown_s=scaledown_s):
            break
        own += request_s * rate
        admitted += 1
    final = own + scaledown_s * rate  # the idle window after the last call
    assert admitted >= 1 and final <= cap


def test_rate_limiter_and_gate():
    rl = spend.RateLimiter(2, 60, b"salt")
    assert rl.allow("1.2.3.4", 0) and rl.allow("1.2.3.4", 1) and not rl.allow("1.2.3.4", 2)
    assert rl.allow("5.6.7.8", 2)
    assert rl.allow("1.2.3.4", 62)
    assert all("1.2.3.4" not in k for k in rl.events)  # only salted hashes are kept
    g = spend.Gate(1, 1)
    assert g.enter(0.1)
    hold = threading.Thread(target=lambda: (g.enter(5), g.leave()))
    hold.start()
    time.sleep(0.05)
    assert not g.enter(0.01)  # one running, one waiting: the third is refused
    g.leave()
    hold.join()
    assert g.enter(0.1)
    g.leave()


# ---------------------------------------------------------------- modal_web.py

class Fake:
    def __init__(self):
        self.state, self.admit, self.calls, self.wakes, self.logs, self.capped_gpu = "ready", True, [], 0, [], False

    def hooks(self):
        return mw.Hooks(gpu_state=lambda: {"state": self.state, "load_s": 1.0, "up_s": 5, "error": ""},
                        budget=lambda warm: {"admit": self.admit, "month": "2026-10"}, score=self.score,
                        wake=self.wake, say=lambda *a: self.logs.append(" ".join(map(str, a))), model="smoke_v2",
                        links={"video": "https://example.org/v", "repo": "https://example.org/r"})

    def wake(self):
        self.wakes += 1

    def score(self, prefix, items, reasons, n):
        self.calls.append((prefix, items, reasons, n))
        if self.capped_gpu:
            return {"capped": True}
        lab = {"Keep": -0.2, "Explore": -2.0, "Skip": -3.0}
        return {"backend": "torch-cuda", "model": "smoke_v2", "stats": {},
                "results": [{"id": it["id"], "scores": lab, "p": dc.label_probs(lab), "label": "Keep",
                             "reason": "fits", "prompt_tokens": 1, "cached_tokens": 0, "seconds": 0.0} for it in items]}


@pytest.fixture()
def web(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "traceback", srv.traceback)  # make_server swaps it; restore after the test
    code = tmp_path / "code"
    (code / "demo" / "personas").mkdir(parents=True)
    (code / "demo" / "results").mkdir()
    (code / "demo" / "index.html").write_text((ROOT / "demo" / "index.html").read_text())
    pool = tmp_path / "pool.jsonl"
    pool.write_text("\n".join(json.dumps(posting(f"p{i}")) for i in range(30)))
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    fake = Fake()
    lim = mw.Limits(max_live=5, hard_items=8, max_resume=2000, rate=(3, 600), cold_start_s=90)
    httpd, _ = mw.make_server(fake.hooks(), lim, pool, code, host="127.0.0.1", port=port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}", fake
    httpd.shutdown()


def call(url, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {"Content-Type": "application/json", **(headers or {})})
    try:
        r = urllib.request.urlopen(req)
        return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_health_never_calls_the_model(web):
    base, fake = web
    fake.state = "asleep"
    code, h = call(base + "/api/health")
    assert code == 200 and h["scorer"]["state"] == "asleep" and h["scorer"]["cold_start_s"] == 90
    assert h["budget"] == {"capped": False, "month": "2026-10"} and h["limits"]["max_live"] == 5
    assert fake.calls == [] and fake.wakes == 0


def test_cold_start_answers_202_and_wakes_once(web):
    base, fake = web
    fake.state = "asleep"
    code, r = call(base + "/score", {"resume": RESUME, "profile": FORM}, {"X-Forwarded-For": "9.9.9.1"})
    assert code == 202 and r["warming"] and fake.wakes == 1 and fake.calls == []
    fake.state = "loading"
    code, r = call(base + "/score", {"resume": RESUME, "profile": FORM}, {"X-Forwarded-For": "9.9.9.1"})
    assert code == 202 and fake.wakes == 1  # already loading: no second wake


def test_warm_score_is_clamped_to_the_live_limit(web):
    base, fake = web
    code, r = call(base + "/score", {"resume": RESUME, "profile": FORM, "live": 999})
    assert code == 200 and r["stats"]["scored_live"] == 5 and r["stats"]["not_scored"] == 25
    assert len(fake.calls) == 1 and len(fake.calls[0][1]) == 5


def test_cap_refuses_before_any_model_call(web):
    base, fake = web
    fake.admit = False
    code, r = call(base + "/score", {"resume": RESUME, "profile": FORM})
    assert code == 402 and r["capped"] and r["links"]["video"] == "https://example.org/v"
    assert fake.calls == [] and fake.wakes == 0
    fake.state = "asleep"
    assert call(base + "/api/health")[1]["budget"]["capped"] is True


def test_gpu_side_refusal_is_a_402_too(web):
    base, fake = web
    fake.capped_gpu = True
    code, r = call(base + "/score", {"resume": RESUME, "profile": FORM})
    assert code == 402 and r["capped"]


def test_size_limits_and_rate_limit(web):
    base, fake = web
    assert call(base + "/score", {"resume": "x" * 2001, "profile": FORM})[0] == 400
    assert call(base + "/score", {"resume": RESUME, "profile": {**FORM, "roles_avoided": "y" * 401}})[0] == 400
    assert call(base + "/score", {"resume": RESUME, "profile": FORM, "posting_ids": [f"p{i}" for i in range(9)]})[0] == 400
    ip = {"X-Forwarded-For": "7.7.7.7, 10.0.0.1"}
    codes = [call(base + "/score", {"resume": RESUME, "profile": FORM}, ip)[0] for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    assert call(base + "/score", {"resume": RESUME, "profile": FORM}, {"X-Forwarded-For": "8.8.8.8"})[0] == 200
    # a visitor writing their own X-Forwarded-For only changes the entries left of the proxy's
    assert call(base + "/score", {"resume": RESUME, "profile": FORM}, {"X-Forwarded-For": "1.1.1.1, 10.0.0.1"})[0] == 429


def test_nothing_from_the_request_is_logged(web):
    base, fake = web
    secret = "Zanzibar-Quokka-4417"
    call(base + "/score", {"resume": RESUME + " " + secret, "profile": {**FORM, "degree_field": secret}})
    call(base + "/api/persona?name=" + secret)
    call(base + "/score", {"resume": "x" * 2001 + secret, "profile": FORM})
    assert fake.logs and not any(secret in line or "Harbor" in line for line in fake.logs)


def test_page_handles_warming_and_cap():
    page = (ROOT / "demo" / "index.html").read_text()
    assert 'id="capbox"' in page and "r.status !== 202" in page and "err.status === 402" in page


def test_settings_file_keeps_the_cap_inside_the_free_credit():
    s = json.loads((ROOT / "demo" / "modal_settings.json").read_text())
    assert s["cap_default"] <= 20 and all(0 <= v <= 20 for v in s["caps"].values())
    assert all(len(m) == 7 and m[4] == "-" for m in s["caps"])  # YYYY-MM
    assert s["dtype"] in ("float32", "bfloat16") and int(s["rate"].split("/")[0]) <= 20


def test_settings_say_what_is_live_and_have_no_temperature():
    """The file describes the deployed app: run B in bfloat16, as its evals ran. The decider reads P at temperature 1,
    so there is no temperature setting left to get wrong."""
    s = json.loads((ROOT / "demo" / "modal_settings.json").read_text())
    assert s["model"] == "main_4b_B" and s["dtype"] == "bfloat16" and "temperature" not in s
    src = (ROOT / "demo" / "modal_app.py").read_text()
    assert "TEMPERATURE" not in src and "temperature" not in src


def test_probabilities_are_the_deciders_and_the_label_is_its_readout():
    s = {"Keep": 24.1, "Explore": 23.4, "Skip": 20.0}
    p = dc.label_probs(s)
    assert abs(p["Keep"] + 0.5 * p["Explore"] - keep_score([24.1, 23.4, 20.0])) < 1e-4
    assert dc.decider_label(s) == "Keep"  # s = 0.80
    s2 = {"Keep": 24.1, "Explore": 24.0, "Skip": 23.9}  # Keep the largest, s = 0.53: the decider says Explore
    assert max(s2, key=s2.get) == "Keep" and dc.decider_label(s2) == "Explore"


def _tar(tmp_path, flag):
    import subprocess
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    for f in ("adapter_config.json", "adapter_model.safetensors", "config.json"):
        (src / f).write_text("{}")
    out = tmp_path / f"run{flag}.tar.gz"  # the name says gzip whatever the content is, as Adaption's do
    r = subprocess.run(["tar", flag, "-cf", str(out), "-C", str(src), "."], capture_output=True)
    if r.returncode:
        pytest.skip(f"tar {flag} not available here")
    return out


@pytest.mark.parametrize("flag", ["--zstd", "-z"])
def test_deploy_unpacks_a_run_archive_by_its_bytes(tmp_path, flag):
    import modal_demo_deploy as dep
    dest = tmp_path / "out"
    dest.mkdir()
    adir = dep.unpack_adapter(_tar(tmp_path, flag), dest)
    assert adir.resolve() == dest.resolve()
    dep.check_adapter(adir)
    bad = tmp_path / "bad.tar.gz"
    bad.write_bytes(b"not an archive")
    with pytest.raises(SystemExit, match="neither zstd nor gzip"):
        dep.unpack_adapter(bad, dest)


def test_modal_app_settings_bound_the_spend():
    """modal_app.py imports modal, which the project venv does not have: read its settings from the source."""
    tree = ast.parse((ROOT / "demo" / "modal_app.py").read_text())
    consts = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
              if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    consts = {k: v.value for k, v in consts.items()}
    assert consts["GPU"] == "L4" and consts["SCALEDOWN_S"] <= 120 and consts["REQUEST_S"] < 150
    cold_s = json.loads((ROOT / "demo" / "modal_settings.json").read_text())["cold_s"]
    assert cold_s < consts["STARTUP_S"] <= 600  # a hung load bills at most 10 minutes
    src = (ROOT / "demo" / "modal_app.py").read_text()
    assert src.count("max_containers=1") >= 2  # one GPU container, one web container
    assert "startup_timeout=STARTUP_S" in src
    assert "print(" not in src.replace('print("[gradus]"', "")  # one logger, fixed words only


# ---------------------------------------------------------------- the 4B check against an eval file

def test_eval_compare_scores_eval_rows_on_the_scorer_and_compares_labels(tmp_path, monkeypatch, capsys):
    import types

    import modal_demo_check as chk

    rows = [{"pair_id": f"r{i}", "prompt": f"INSTR resume {i // 2}\n\nJOB POSTING:\nposting {i}\n\nAnswer:"}
            for i in range(4)]
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    # `pred` is the argmax the eval file stores; the check reads the decider on `scores` instead
    ev_rows = [{"pair_id": "r0", "pred": "Keep", "scores": [3.0, 1.0, 0.0]},  # s 0.90: Keep
               {"pair_id": "r1", "pred": "Keep", "scores": [0.47, 0.0, 0.0]},  # s 0.583, just under T_HI: Explore
               {"pair_id": "r2", "pred": "Skip", "scores": [0.0, 0.0, 4.0]}]
    (tmp_path / "ev.json").write_text(json.dumps({"model": "m", "dtype": "float32", "framing": "chat", "rows": ev_rows}))
    calls = []

    class Scorer:
        class score:
            @staticmethod
            def remote(prefix, items, reasons, n):
                calls.append((prefix, [i["id"] for i in items]))
                scores = {"r0": [3.1, 1.0, 0.0], "r1": [0.52, 0.0, 0.0], "r2": [0.0, 0.0, 4.2]}  # r1 s 0.593: Keep
                return {"model": "v3_4b", "results": [
                    {"id": i["id"], "scores": dict(zip(dc.LABELS, scores[i["id"]])),
                     "label": dc.decider_label(dict(zip(dc.LABELS, scores[i["id"]])))} for i in items]}

    fake = types.SimpleNamespace(Cls=types.SimpleNamespace(from_name=lambda app, name: Scorer))
    monkeypatch.setitem(sys.modules, "modal", fake)
    res = chk.eval_compare(str(tmp_path / "ev.json"), str(tmp_path / "rows.jsonl"), 10)
    assert calls == [("INSTR resume 0\n\nJOB POSTING:\n", ["r0", "r1"]), ("INSTR resume 1\n\nJOB POSTING:\n", ["r2"])]
    assert res["rows"] == 3 and res["agree"] == 2 and [t[0] for t in res["near_cuts"]] == ["r1"] and res["ok"]
    assert res["near_cuts"][0][1:3] == ("Explore", "Keep")
    assert "near-cut" in capsys.readouterr().out

    Scorer.score.remote = staticmethod(lambda *a: {"capped": True})
    with pytest.raises(SystemExit, match="budget"):
        chk.eval_compare(str(tmp_path / "ev.json"), str(tmp_path / "rows.jsonl"), 10)
