"""The public demo on Modal: the demo page and its API on a small CPU container, the model on one L4.

No model ever runs on the Mac: merging, loading and scoring all happen in Modal containers.

    # deploy, or swap the model (adapter into the Volume, merged there, deployed, old containers stopped)
    ~/.local/share/uv/tools/modal/bin/python scripts/modal_demo_deploy.py [--adapter DIR] [--set key=value ...]
    # once per pool rebuild (employer text: it lives in the Volume, never in git)
    ~/.local/bin/modal volume put -f matcher-demo demo/local/pool.jsonl pool.jsonl

Settings live in demo/modal_settings.json; MATCHER_<KEY> in the environment overrides one for a single
deploy. They are read on the Mac at deploy time and baked into the containers' environment:
    model        merged model under /vol/models/ (smoke_v2 is the 0.8B stand-in; main_4b_B is run B, the shipped 4B)
    dtype        float32 (what oci-pi runs) or bfloat16 (for 4B, as its eval ran)
    gpu_mem_gib  GPU container memory limit in GiB
    caps         {"YYYY-MM": usd} per calendar month (UTC); cap_default for any other month
    cold_s       seconds a cold GPU start is assumed to cost when admitting a request
    rate         scoring requests per IP per window, "8/600"
    video_url, repo_url   links on the page shown once the month's budget is used

Spend cap (demo/spend.py): every container writes its own uptime cost into the modal.Dict
matcher-demo-ledger (the Volume and the ledger kept their names when the app became gradus) every 15 s and at exit. The web container refuses a score when the month's total
plus the most the request can add (cold start if the GPU is asleep, the request timeout, the 2-minute
idle scaledown) would pass the cap; the GPU container repeats that check with its own exact cost at the
start of every call, and calls are serialized there (one container, one input at a time), so concurrent
requests cannot race past the cap. Resumes and profiles are never written to disk, the Dict or the logs.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import modal

APP_NAME = "gradus"  # the app and its web label: https://<workspace>--gradus.modal.run
CODE = Path("/root/app")  # where the repo's code sits inside every container
ROOT = Path(__file__).resolve().parent.parent  # the repo, when this file is read on the Mac at deploy time

# ---------------------------------------------------------------- deploy-time settings
_FILE = Path(__file__).with_name("modal_settings.json")  # next to this file on the Mac; absent in a container
_SETTINGS = json.loads(_FILE.read_text()) if _FILE.exists() else {}


def _setting(key: str, default):
    raw = os.environ.get("MATCHER_" + key.upper())
    if raw is None:
        return _SETTINGS.get(key, default)
    return json.loads(raw) if isinstance(default, (dict, int, float)) else raw


MODEL = str(_setting("model", "smoke_v2"))
DTYPE = str(_setting("dtype", "float32"))
GPU = "L4"
GPU_CORES = 2.0
GPU_MEM_GIB = float(_setting("gpu_mem_gib", 12.0))
CAPS = {k: float(v) for k, v in _setting("caps", {}).items()}
CAP_DEFAULT = float(_setting("cap_default", 15.0))
COLD_S = float(_setting("cold_s", 420.0))  # budget reserve: covers a preempted start (347 s measured)
WARM_ETA_S = 110  # what the page tells a visitor: run B's cold start measured 107 s (first request to ready model)
VIDEO_URL = str(_setting("video_url", ""))
REPO_URL = str(_setting("repo_url", ""))
_rate = str(_setting("rate", "8/600")).split("/")
RATE_LIMIT = (int(_rate[0]), float(_rate[1]))  # scoring requests per IP per window (s)

SCALEDOWN_S = 120      # GPU idle time before it shuts down (billed)
IDLE_BILLED_S = SCALEDOWN_S + 60  # measured 2026-10-19: the container exited 181 s after its last call
REQUEST_S = 110        # GPU time limit per scoring call; Modal's web requests end at 150 s
BOOT_S = 20            # Modal bills from container boot, a little before our first timestamp
STARTUP_S = 600        # longest model load before Modal stops the container (347 s measured on a preempted L4)
WEB_CORES, WEB_MEM_GIB, WEB_SCALEDOWN_S = 0.5, 0.5, 60
MAX_LIVE = 25          # postings scored per request (plus citizens-only extras, see HARD_ITEMS)
HARD_ITEMS = 40        # absolute ceiling on postings sent to the GPU in one call
MAX_RESUME = 20_000    # characters
MAX_FIELD = 400        # characters per profile field
MAX_BODY = 256 * 1024  # bytes of JSON per /score request
REASON_TOKENS = 40
QUEUE = (1, 3)         # one call on the GPU at a time, three more may wait
QUEUE_WAIT_S = 30

CONFIG = {"MATCHER_MODEL": MODEL, "MATCHER_DTYPE": DTYPE, "MATCHER_GPU_MEM_GIB": json.dumps(GPU_MEM_GIB),
          "MATCHER_CAPS": json.dumps(CAPS), "MATCHER_CAP_DEFAULT": json.dumps(CAP_DEFAULT),
          "MATCHER_COLD_S": json.dumps(COLD_S), "MATCHER_RATE": f"{RATE_LIMIT[0]}/{int(RATE_LIMIT[1])}",
          "MATCHER_VIDEO_URL": VIDEO_URL, "MATCHER_REPO_URL": REPO_URL,
          "PYTHONPATH": f"{CODE}/src:{CODE}/demo:{CODE}/scripts", "PYTHONDONTWRITEBYTECODE": "1"}

app = modal.App(APP_NAME)
# The Volume and the spend ledger keep the names they had before the app was renamed: the Volume holds the pool,
# the models and the compiled kernels, and a new ledger would start the month's spend at zero under the cap.
vol = modal.Volume.from_name("matcher-demo", create_if_missing=True)
ledger_dict = modal.Dict.from_name("matcher-demo-ledger", create_if_missing=True)

_IGNORE = ["**/__pycache__/**", "**/*.pyc", "local/**", "results/*.json"]


def _code(img: modal.Image) -> modal.Image:
    return (img.env(CONFIG)
            .add_local_dir(ROOT / "src" / "matcher", f"{CODE}/src/matcher", ignore=_IGNORE)
            .add_local_dir(ROOT / "demo", f"{CODE}/demo", ignore=_IGNORE)
            .add_local_file(ROOT / "scripts" / "demo_common.py", f"{CODE}/scripts/demo_common.py")
            .add_local_file(ROOT / "scripts" / "demo_backend_torch.py", f"{CODE}/scripts/demo_backend_torch.py")
            .add_local_file(ROOT / "scripts" / "merge_lora.py", f"{CODE}/scripts/merge_lora.py"))


web_image = _code(modal.Image.debian_slim(python_version="3.12").uv_pip_install("pypdf==6.19.0"))
_TORCH = ("torch==2.14.0", "transformers==5.17.0", "safetensors==0.8.0", "huggingface_hub==1.33.0")
# The GPU scorer uses flash-linear-attention's Triton kernels for Qwen3.5's linear-attention layers. The CPU
# image leaves them out: transformers would pick them on CPU too and fail without a GPU driver, and the
# reference must be the oci-pi path (the reference kernels on CPU).
gpu_image = _code(modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    *_TORCH, "flash-linear-attention==0.5.2").env({"TRITON_CACHE_DIR": "/vol/triton-cache"}))  # compiled kernels survive a restart
cpu_image = _code(modal.Image.debian_slim(python_version="3.12").uv_pip_install(*_TORCH))


def _spend():
    import spend
    return spend


def _caps() -> dict:
    return CAPS


def _cid() -> str:
    return os.environ.get("MODAL_TASK_ID", f"pid{os.getpid()}")


def _say(*parts) -> None:
    """The only logging on the request path: fixed words and numbers, never request content."""
    print("[gradus]", *parts, flush=True)


class _Meter:
    """This container's uptime cost, written to the ledger every HEARTBEAT_S seconds and at exit."""

    def __init__(self, kind: str, rate: float, on_beat=None) -> None:
        sp = _spend()
        self.kind, self.rate, self.start, self.cid = kind, rate, time.time(), _cid()
        self.ledger = sp.Ledger(ledger_dict)
        self.key = sp.Ledger.key(sp.month_of(self.start), kind, self.cid)
        self.on_beat, self.stopped = on_beat, threading.Event()
        self.beat()
        threading.Thread(target=self._loop, daemon=True, name="meter").start()

    def own(self) -> float:
        return (time.time() - self.start + BOOT_S) * self.rate

    def beat(self) -> None:
        try:
            self.ledger.record(self.kind, self.cid, self.start, time.time(), self.rate, BOOT_S)
            if self.on_beat:
                self.on_beat()
        except Exception as e:  # a missed beat is covered by MARGIN_S; never kill the container over it
            _say("meter write failed:", type(e).__name__)

    def _loop(self) -> None:
        while not self.stopped.wait(_spend().HEARTBEAT_S):
            self.beat()

    def close(self) -> None:
        self.stopped.set()
        self.beat()


def _budget(warm: bool) -> dict:
    """The month's counted spend against its cap, and whether a request can be admitted right now."""
    sp = _spend()
    now = time.time()
    month = sp.month_of(now)
    cap = sp.cap_for(month, _caps(), CAP_DEFAULT)
    led = sp.Ledger(ledger_dict)
    spent = led.spent(month)
    rate = sp.rate_per_s(GPU, GPU_CORES, GPU_MEM_GIB)
    ok = sp.admit_web(cap=cap, spent=spent, warm=warm, rate=rate, cold_s=COLD_S, request_s=REQUEST_S,
                      scaledown_s=IDLE_BILLED_S)
    return {"month": month, "cap_usd": cap, "spent_usd": round(spent, 4), "admit": ok}


# ---------------------------------------------------------------- the model, on one L4

@app.cls(image=gpu_image, gpu=GPU, cpu=(GPU_CORES, GPU_CORES), memory=(int(GPU_MEM_GIB * 1024),) * 2,
         volumes={"/vol": vol}, max_containers=1, scaledown_window=SCALEDOWN_S, timeout=REQUEST_S,
         startup_timeout=STARTUP_S, retries=0)
class Scorer:
    @modal.enter()
    def load(self) -> None:
        sp = _spend()
        self.rate = sp.rate_per_s(GPU, GPU_CORES, GPU_MEM_GIB)
        self.state, self.load_s, self.error = "loading", None, ""
        self.meter = _Meter("gpu", self.rate, on_beat=self._publish)
        if not self._admit(COLD_S):
            self.state = "capped"
            self._publish()
            _say("gpu: month budget reached, not loading the model")
            return
        t0 = time.time()
        try:
            import demo_backend_torch as bt
            path = f"/vol/models/{MODEL}"
            if not Path(path, "config.json").exists():
                raise FileNotFoundError(f"no merged model at {path}")
            bt._init(path, device="cuda", dtype=DTYPE)
            # compile the kernels once on a throwaway prompt, so the first real request is not slower
            bt._score_chunk("warm up " * 40, [{"id": "a", "suffix": " one"}, {"id": "b", "suffix": " two"}], True, 4)
            self.bt = bt
            self.state, self.load_s = "ready", round(time.time() - t0, 1)
        except Exception as e:
            self.state, self.error = "error", f"{type(e).__name__}: {e}"
            _say("gpu: load failed:", self.error)
        self._publish()
        _say("gpu:", self.state, "model", MODEL, "load_s", self.load_s, "since boot",
             round(time.time() - self.meter.start, 1))

    def _publish(self) -> None:
        ledger_dict["gpu_state"] = {"state": self.state, "t": time.time(), "cid": self.meter.cid if hasattr(self, "meter") else "",
                                    "model": MODEL, "load_s": self.load_s, "error": self.error,
                                    "up_s": round(time.time() - self.meter.start, 1) if hasattr(self, "meter") else 0}

    def _admit(self, extra_s: float) -> bool:
        sp = _spend()
        month = sp.month_of(time.time())
        others = self.meter.ledger.spent(month, exclude=self.meter.key)
        return sp.admit_gpu(cap=sp.cap_for(month, _caps(), CAP_DEFAULT), spent_others=others, own_usd=self.meter.own(),
                            rate=self.rate, request_s=extra_s, scaledown_s=IDLE_BILLED_S)

    def _stop(self) -> None:
        try:
            modal.experimental.stop_fetching_inputs()  # leave now instead of idling through scaledown
        except Exception:
            pass

    @modal.method()
    def warm(self) -> dict:
        if self.state == "capped":
            self._stop()
        return {"state": self.state, "load_s": self.load_s, "error": self.error}

    @modal.method()
    def score(self, prefix: str, items: list[dict], reasons: bool, max_reason_tokens: int) -> dict:
        if self.state == "capped" or not self._admit(REQUEST_S):
            self.state = "capped"
            self._publish()
            self._stop()
            _say("gpu: refused a call, month budget reached; up", round(time.time() - self.meter.start, 1), "s")
            return {"capped": True}
        if self.state != "ready":
            return {"error": f"model not loaded ({self.error or self.state})"}
        if not isinstance(prefix, str) or len(prefix) > MAX_RESUME + 20 * MAX_FIELD + 4000 or len(items) > HARD_ITEMS:
            return {"error": "request too large"}
        t0 = time.time()
        res = self.bt._score_chunk(prefix, items, bool(reasons), max(1, min(int(max_reason_tokens), 48)))
        secs = round(time.time() - t0, 2)
        _say("gpu: scored", len(res), "postings, reasons", bool(reasons), "in", secs, "s")
        return {"backend": "torch-cuda", "model": MODEL, "results": res,
                "stats": {"items": len(res), "seconds": secs, "workers": 1,
                          "cached_tokens": res[0]["cached_tokens"] if res else 0}}

    @modal.exit()
    def bye(self) -> None:
        self.state = "gone"
        self.meter.close()
        _say("gpu: exit after", round(time.time() - self.meter.start, 1), "s")


# ---------------------------------------------------------------- the page and its API, on a small CPU container

def _gpu_state() -> dict:
    """The GPU container's last published state; 'asleep' when it has not written for a while."""
    s = ledger_dict.get("gpu_state") or {}
    fresh = time.time() - s.get("t", 0) < 2.5 * _spend().HEARTBEAT_S
    state = s.get("state") if fresh else "asleep"
    if state in (None, "gone"):
        state = "asleep"
    return {"state": state, "load_s": s.get("load_s"), "up_s": s.get("up_s"), "error": s.get("error", "")}


def _make_server():
    import modal_web as mw

    hooks = mw.Hooks(
        gpu_state=_gpu_state, budget=lambda warm: _budget(warm),
        score=lambda prefix, items, reasons, n: Scorer().score.remote(prefix, items, reasons, n),
        wake=lambda: Scorer().warm.spawn(), say=_say, model=MODEL, links={"video": VIDEO_URL, "repo": REPO_URL},
        timeout_errors=(modal.exception.FunctionTimeoutError,))
    lim = mw.Limits(max_live=MAX_LIVE, hard_items=HARD_ITEMS, max_resume=MAX_RESUME, max_field=MAX_FIELD,
                    max_body=MAX_BODY, reason_tokens=REASON_TOKENS, rate=RATE_LIMIT, queue=QUEUE,
                    queue_wait_s=QUEUE_WAIT_S, cold_start_s=WARM_ETA_S)
    return mw.make_server(hooks, lim, Path("/vol/pool.jsonl"), CODE)


@app.cls(image=web_image, cpu=(0.125, WEB_CORES), memory=(256, int(WEB_MEM_GIB * 1024)), volumes={"/vol": vol},
         max_containers=1, scaledown_window=WEB_SCALEDOWN_S, timeout=150, retries=0)
@modal.concurrent(max_inputs=32)
class Web:
    @modal.enter()
    def start(self) -> None:
        self.meter = _Meter("web", _spend().rate_per_s(None, WEB_CORES, WEB_MEM_GIB))
        httpd, demo = _make_server()
        threading.Thread(target=httpd.serve_forever, daemon=True, name="http").start()
        _say("web: up, pool", len(demo.pool), "model", MODEL)

    @modal.web_server(8000, startup_timeout=60, label=APP_NAME)
    def page(self) -> None:
        pass

    @modal.exit()
    def bye(self) -> None:
        self.meter.close()


# ---------------------------------------------------------------- housekeeping

@app.function(image=web_image, schedule=modal.Period(days=1), cpu=0.125, memory=128, timeout=120)
def keepalive() -> dict:
    """Re-writes this and last month's ledger entries (a Dict entry expires after 7 idle days)."""
    sp = _spend()
    meter = _Meter("cron", sp.rate_per_s(None, 0.125, 0.125))
    now = time.time()
    months = [sp.month_of(now), sp.month_of(now - 32 * 86400)]
    n = meter.ledger.refresh(months)
    meter.close()
    return {"refreshed": n}


BUILD_CORES, BUILD_MEM_GIB = 4.0, 32.0  # a 4B merge or a float32 CPU reference of 4B needs ~18 GB


def _public_base(repo_id: str) -> str:
    """togethercomputer/X -> Qwen/X (Adaption trains on Together's copies of the public Qwen checkpoints)."""
    return "Qwen/" + repo_id.split("/", 1)[1] if repo_id.startswith("togethercomputer/") else repo_id


@app.function(image=cpu_image, cpu=BUILD_CORES, memory=int(BUILD_MEM_GIB * 1024), volumes={"/vol": vol}, timeout=3600)
def prepare(name: str) -> dict:
    """Merge /vol/adapters/NAME into its base model, writing /vol/models/NAME. The base comes from the
    adapter's own adapter_config.json; a togethercomputer/ id is mapped to the public Qwen/ repo only when
    the architecture fields of the adapter's config.json match the public config.json exactly."""
    import hashlib
    import subprocess

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    sp = _spend()
    meter = _Meter("build", sp.rate_per_s(None, BUILD_CORES, BUILD_MEM_GIB))
    try:
        adir = Path(f"/vol/adapters/{name}")
        acfg = json.loads((adir / "adapter_config.json").read_text())
        declared = acfg["base_model_name_or_path"]
        base = _public_base(declared)
        report: dict = {"declared_base": declared, "base_used": base}
        try:
            info = HfApi().model_info(declared)
            report["declared_on_hub"] = f"yes ({info.id}, sha {info.sha})"
        except Exception as e:
            report["declared_on_hub"] = f"no ({type(e).__name__})"
        if base != declared:
            own = json.loads((adir / "config.json").read_text())
            pub = json.loads(Path(hf_hub_download(base, "config.json")).read_text())
            keys = ("architectures", "model_type", "tie_word_embeddings")
            tkeys = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                     "head_dim", "vocab_size", "layer_types", "linear_num_key_heads", "linear_num_value_heads",
                     "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim", "rope_parameters",
                     "rms_norm_eps", "full_attention_interval", "attn_output_gate", "tie_word_embeddings", "eos_token_id")
            ot, pt = own.get("text_config", own), pub.get("text_config", pub)
            diff = {k: (own.get(k), pub.get(k)) for k in keys if own.get(k) != pub.get(k)}
            diff.update({f"text_config.{k}": (ot.get(k), pt.get(k)) for k in tkeys if ot.get(k) != pt.get(k)})
            report["config_fields_compared"] = len(keys) + len(tkeys)
            report["config_diff"] = {k: [str(a), str(b)] for k, a, b in ((k, *v) for k, v in diff.items())}
            if diff:
                raise SystemExit(f"{base} config differs from the adapter's config.json: {diff}")
        bdir = Path(f"/vol/base/{base}")
        snapshot_download(base, local_dir=str(bdir), allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
        for f in ("tokenizer.json", "chat_template.jinja"):
            if (adir / f).exists() and (bdir / f).exists():
                report[f"same_{f}"] = hashlib.sha256((adir / f).read_bytes()).hexdigest() == \
                    hashlib.sha256((bdir / f).read_bytes()).hexdigest()
        out = Path(f"/vol/models/{name}")
        r = subprocess.run([sys.executable, str(CODE / "scripts" / "merge_lora.py"), str(adir), str(out), "--base", str(bdir)],
                           capture_output=True, text=True)
        report["merge_log"] = (r.stdout + r.stderr)[-2000:]
        if r.returncode:
            raise SystemExit(f"merge failed:\n{report['merge_log']}")
        (out / "MERGED_FROM.json").write_text(json.dumps(report, indent=1))
        vol.commit()
        return report
    finally:
        meter.close()


@app.function(image=cpu_image, cpu=BUILD_CORES, memory=int(BUILD_MEM_GIB * 1024), volumes={"/vol": vol}, timeout=3600)
def reference(prefix: str, items: list[dict], reasons: bool) -> list[dict]:
    """The oci-pi code path exactly (demo_backend_torch on CPU, float32), for comparing against the deployed GPU scorer."""
    import demo_backend_torch as bt

    meter = _Meter("test", _spend().rate_per_s(None, BUILD_CORES, BUILD_MEM_GIB))
    try:
        bt._init(f"/vol/models/{MODEL}")
        return bt._score_chunk(prefix, items, reasons, REASON_TOKENS)
    finally:
        meter.close()


@app.function(image=web_image, cpu=0.125, memory=128, timeout=60)
def ledger_report() -> dict:
    """The month's counted spend by kind, and the GPU state (for checks from the Mac; no request content is stored)."""
    sp = _spend()
    month = sp.month_of(time.time())
    led = sp.Ledger(ledger_dict)
    return {"month": month, "cap_usd": sp.cap_for(month, _caps(), CAP_DEFAULT), "spent_usd": led.spent(month),
            "by_kind": led.by_kind(month), "entries": len(led.entries(month)), "gpu_state": ledger_dict.get("gpu_state")}
