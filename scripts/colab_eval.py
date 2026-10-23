"""Merge AutoScientist adapters and score dev sets on a Colab L4, driven from the Mac (no model runs here).

Usage: colab_eval.py SESSION --base-model HF_ID --model TAG=RUN.tar.gz [--model TAG=...] [--base] --dev FILE
                     [--dev FILE ...] --out DIR [--dtype auto|float32|bfloat16] [--merge-float32] [--attach] [--keep]
                     [--max-min 70] [--gen 2] [--max-tokens N] [--batch auto|1]
       colab_eval.py SESSION --check --base-model HF_ID [--dev FILE] [--max-min 15]

--base-model is required: the model the adapters were trained on (Qwen/Qwen3.5-0.8B for the v2 smoke run,
Qwen/Qwen3.5-4B for train_main.py). Each base downloads to its own directory on the VM (base_<id>), and a merged model
records its base (merged_TAG/BASE_MODEL), so a session that holds another model's files never merges onto them. Before
a merge the VM must have free RAM for about 2.5 times the base's weights (merge_lora holds the base, the merged copy and
the save buffer); otherwise it stops and says so (use a high-RAM runtime, or merge on oci-pi with merge_lora.py).
merge_lora runs with --expect-base: the adapter's togethercomputer/ base must map to --base-model and its config.json
architecture fields must equal the base's, and every LoRA pair must merge, or nothing is scored. --merge-float32 keeps the
merged weights in float32 (merge_lora --keep-float32: no bfloat16 rounding of the LoRA update; needs about 4 times the
base's weights in free RAM); pair it with --dtype float32.

Job order is priority order, so a session cut at --max-min loses the last jobs: for each --dev file in the order
given, every --model in the order given, then the base (--base). For the 4B: --dev test.jsonl, dev_cross.jsonl,
dev.jsonl gives 4B test, base test, 4B dev_cross, base dev_cross, 4B dev, base dev.

--gen N (default 2) generates a greedy completion for the first N rows of each --dev file; --max-tokens caps its new
tokens (local_eval's default, 48, when unset). A subset file with --gen at least its row count gets a reason per row.

--dtype auto (the default): jobs.sh first scores the 5 longest-prompt rows of the first --dev file with the first job's
model at --batch 1 and --batch 4 in float32 and runs
`local_eval.py decide` (batch 4 only when batchcheck passes; the median per-row time times every job's rows, plus a
load per job, must fit before the deadline: --max-min from the session start less 8 minutes). If float32 does not
fit (or runs out of memory), it probes bfloat16 the same way; every job then runs in that one dtype and batch
(RUN_CHOICE). bfloat16 keeps the label math in float32 (local_eval casts the last hidden state). A job that dies
without its result at batch 4 (memory on a prompt longer than the probe's) runs once more at batch 1. --dtype float32 or
bfloat16 skips the choice of dtype but still probes the batch. --batch 1 pins batch 1: the probe runs at batch 1 only
and every job runs at batch 1 (a protocol that fixes batch 1, e.g. eval/protocol_amendment_1.json).

Starts the session only through the Colab watchdog (an external colab_guard.sh, L4, stopped at
--max-min or after 20 idle minutes); --attach reuses a session already started that way. Once the session is up the
VM installs transformers and downloads --base-model in a detached script while the Mac uploads local_eval.py,
evalstats.py, merge_lora.py, the dev files and each adapter archive (`colab upload` in 8 MB parts, since a 90 MB
file failed in one call, joined on the VM and checked by size and md5; base64 parts 4 at a time as the fallback; a
file whose remote size already matches is skipped). Then it merges each adapter into
/content/m/merged_TAG and runs every (model, dev file) job in one detached shell script on the VM, so no exec has to
stay connected while the GPU works. Polls, downloads DIR/<base|TAG>_<dev stem>.json as each lands (and the probes,
decide_*.json and RUN_CHOICE at the end), and stops the session unless --keep. Every colab call runs under `timeout`;
a kernel thread touches /content/heartbeat.

--check loads no model: it starts the session through the guard, prints the GPU, free RAM and disk, times an upload
(the first --dev file; says whether `colab upload` worked or base64 parts were needed), times the transformers install, fetches --base-model's
config.json and safetensors index (no weights) and prints the weight size, then stops the session.

The driver runs on the Mac, as does the watchdog: keep the Mac awake and on power for the whole run
(`caffeinate -is`); a sleeping Mac stops both the polls and the watchdog while the VM keeps running.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# a watchdog script that stops the Colab session after its time budget: COLAB_GUARD SESSION GPU MAX_MIN IDLE_MIN
GUARD = os.environ.get("COLAB_GUARD", "colab_guard.sh")
COLAB = str(Path.home() / ".local/bin/colab")
M = "/content/m"
CHUNK = 500_000  # base64 characters per exec call; a 2 MB call lost the kernel connection
BEAT = '''
import pathlib, threading, time
def _beat():
    while True:
        pathlib.Path("/content/heartbeat").touch(); time.sleep(60)
if not any(t.name == "beat" for t in threading.enumerate()):
    threading.Thread(target=_beat, name="beat", daemon=True).start()
'''


def run(cmd: list[str], limit: int, tries: int = 1) -> str:
    for attempt in range(tries):
        r = subprocess.run(["timeout", str(limit), *cmd], capture_output=True, text=True)
        out = r.stdout + r.stderr
        if not r.returncode:
            return out
        print(f"{' '.join(cmd[1:3])} failed ({r.returncode}), attempt {attempt + 1}/{tries}", flush=True)
    raise SystemExit(f"{' '.join(cmd[:4])} failed ({r.returncode}):\n{out[-3000:]}")


def exec_code(session: str, code: str, limit: int = 300, tries: int = 3) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(BEAT + code)
    try:
        return run([COLAB, "exec", "-s", session, "-f", f.name, "--timeout", str(limit)], limit + 60, tries)
    finally:
        Path(f.name).unlink()


def sh(session: str, cmd: str, limit: int = 900, tries: int = 1) -> str:
    """A short shell command on the VM; long work goes through a detached script instead. tries > 1 only for
    commands that are safe to run twice."""
    out = exec_code(session, f"import subprocess\nr = subprocess.run({cmd!r}, shell=True, capture_output=True, text=True)\n"
                             "print(r.stdout[-4000:]); print(r.stderr[-3000:]); print('EXIT', r.returncode)", limit, tries)
    print(out[-1500:], flush=True)
    if "EXIT 0" not in out:
        raise SystemExit(f"remote command failed: {cmd}")
    return out


def remote_size(session: str, path: str) -> int:
    out = exec_code(session, f"import os\nprint('SIZE', os.path.getsize({path!r}) if os.path.exists({path!r}) else -1)")
    return int(out.split("SIZE", 1)[1].split()[0])


CLI_PART = 8_000_000  # bytes per `colab upload` call: 8.9 MB went up in 7 s, 90 MB failed (2026-10-20)


def quiet(text: str) -> str:
    """CLI error text without the runtime proxy token it can echo in a URL."""
    return re.sub(r"(token=)[^&\s]+", r"\1...", text)


def join_parts(session: str, remote: str, n: int, size: int, digest: str) -> bool:
    """Join remote.part0000.. into remote (removing the parts) and check its size and md5."""
    out = exec_code(session, f"import hashlib, os\nh = hashlib.md5()\nwith open({remote!r}, 'wb') as f:\n"
                             f"    for n in range({n}):\n"
                             f"        p = {remote!r} + f'.part{{n:04d}}'\n"
                             f"        b = open(p, 'rb').read(); h.update(b); f.write(b); os.remove(p)\n"
                             f"print('SIZE', os.path.getsize({remote!r}), 'MD5', h.hexdigest())")
    return f"SIZE {size} MD5 {digest}" in out


def upload(session: str, local: Path, remote: str, how: str = "auto") -> str:
    """`colab upload` in parts of CLI_PART bytes, joined on the VM and checked by size and md5 (how auto or cli),
    else base64 parts through exec. Parts go to separate files, so a retried part never doubles. Returns the way it
    went up."""
    size = local.stat().st_size
    if remote_size(session, remote) == size:
        print("already there", local.name, flush=True)
        return "there"
    raw = local.read_bytes()
    digest = hashlib.md5(raw).hexdigest()
    if how in ("auto", "cli"):
        t = time.time()
        ok = True
        with tempfile.TemporaryDirectory() as tmp:
            starts = list(range(0, size, CLI_PART)) or [0]
            for n, i in enumerate(starts):
                part = Path(tmp) / f"{local.name}.part{n:04d}"
                part.write_bytes(raw[i:i + CLI_PART])
                for attempt in range(3):
                    r = subprocess.run(["timeout", "300", COLAB, "upload", "-s", session, str(part),
                                        f"{remote}.part{n:04d}"], capture_output=True, text=True)
                    if not r.returncode:
                        break
                    print(f"colab upload of part {n} failed ({r.returncode}), attempt {attempt + 1}/3: "
                          f"{quiet(r.stdout + r.stderr)[-200:]}", flush=True)
                else:
                    ok = False
                    break
        if ok and join_parts(session, remote, len(starts), size, digest):
            print(f"uploaded {local.name} {size} bytes by colab upload ({len(starts)} parts) in {time.time() - t:.0f} s",
                  flush=True)
            return "cli"
        print(f"colab upload of {local.name} did not complete", flush=True)
        if how == "cli":
            raise SystemExit("colab upload failed")
    t = time.time()
    data = base64.b64encode(raw).decode()
    starts = list(range(0, len(data), CHUNK))

    def part_b64(n: int) -> None:
        i = starts[n]
        exec_code(session, f"import base64\nopen({remote + f'.part{n:04d}'!r}, 'wb').write("
                           f"base64.b64decode({data[i:i + CHUNK]!r}))", 120)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(part_b64, range(len(starts))))
    if not join_parts(session, remote, len(starts), size, digest):
        raise SystemExit(f"upload of {local} came out wrong")
    print(f"uploaded {local.name} {size} bytes by base64 parts in {time.time() - t:.0f} s", flush=True)
    return "base64"


def base_dir(model_id: str) -> str:
    """The VM directory a base model downloads to: one per model, so two bases never share one."""
    return "base_" + model_id.replace("/", "--")


def ram_check(base_path: str, factor: float = 2.5) -> str:
    """A shell command that fails unless MemAvailable covers `factor` times the base's safetensors."""
    code = ("import glob, os, sys\n"
            f"need = {factor} * sum(os.path.getsize(f) for f in glob.glob({base_path!r} + '/*.safetensors'))\n"
            "have = next(int(l.split()[1]) * 1024 for l in open('/proc/meminfo') if l.startswith('MemAvailable'))\n"
            "print(f'RAM free {have / 1e9:.1f} GB, merge needs about {need / 1e9:.1f} GB')\n"
            "sys.exit(0 if have >= need else 'not enough RAM to merge here: use a high-RAM runtime or merge on oci-pi')")
    return f"python -c {shlex.quote(code)}"


SETUP = """#!/bin/bash
cd {M}
python -c 'import transformers; assert int(transformers.__version__.split(".")[0]) >= 5' 2>/dev/null || \\
  pip install -q 'transformers>=5.13' safetensors huggingface_hub 2>&1 | tail -2
(test -f {base}/config.json || hf download {model} --local-dir {base} > /dev/null) || exit 1
(which zstd > /dev/null || apt-get -qq install -y zstd > /dev/null) || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv
"""


def longest_rows(dev: Path, n: int = 5) -> list[dict]:
    """The n rows of a dev file with the longest prompts: the probe's worst case for memory and time."""
    rows = [json.loads(line) for line in open(dev) if line.strip()]
    return sorted(rows, key=lambda r: len(r["prompt"]), reverse=True)[:n]


def jobs_script(jobs: list[tuple[str, str, str]], rows: dict[str, int], deadline: float, dtype: str,
                probe_dev: str, gen: int = 2, max_tokens: int | None = None, batch: str = "auto") -> str:
    """The detached VM script: probe (on probe_dev, the longest rows) and decide, then every (name, model dir, dev
    file) job in order. A job that dies without its .json at batch > 1 (out of memory on a long prompt) runs once more
    at batch 1. Each job generates a greedy completion for its first `gen` rows (local_eval --gen), up to
    `max_tokens` new tokens when given (local_eval's own default otherwise). batch "1" probes and runs at batch 1 only."""
    first_dir = jobs[0][1]
    forwards = sum(rows[d] for _, _, d in jobs)
    lines = ["#!/bin/bash", f"cd {M}",
             "probe() {",
             f"  for b in {'1' if batch == '1' else '1 4'}; do python local_eval.py score {first_dir} --dev {probe_dev} --device cuda --dtype $1 "
             "--limit 5 --batch $b --out probe_$1_b$b.json > probe_$1_b$b.log 2>&1; done",
             f"  python local_eval.py decide --b1 probe_$1_b1.json{'' if batch == '1' else ' --b4 probe_$1_b4.json'} --forwards {forwards} "
             f"--jobs {len(jobs)} --deadline {deadline:.0f} > decide_$1.json 2>> decide.log || echo '{{\"fits\": false, \"batch\": 1}}' > decide_$1.json",
             "}",
             "fits() { python -c \"import json, sys; sys.exit(0 if json.load(open('decide_$1.json'))['fits'] else 1)\"; }"]
    if dtype == "auto":
        lines += ["probe float32", "if fits float32; then DTYPE=float32; else probe bfloat16; DTYPE=bfloat16; fi"]
    else:
        lines += [f"probe {dtype}", f"DTYPE={dtype}"]
    lines += ["BATCH=$(python -c \"import json; print(json.load(open('decide_$DTYPE.json'))['batch'])\")",
              "echo \"$DTYPE $BATCH\" > RUN_CHOICE"]
    for name, mdir, dev in jobs:
        cap = f" --max-tokens {max_tokens}" if max_tokens else ""
        cmd = f"python local_eval.py score {mdir} --dev {dev} --device cuda --dtype $DTYPE --gen {gen}{cap} --out {name}.json"
        lines.append(f"test -f {name}.json || {{ {cmd} --batch $BATCH > {name}.log 2>&1; test -f {name}.json || "
                     f"[ \"$BATCH\" = 1 ] || {cmd} --batch 1 >> {name}.log 2>&1; }}")
    lines.append("touch ALL_DONE")
    return "\n".join(lines) + "\n"


def job_list(models: list[str], base: bool, devs: list[str], base_path: str) -> list[tuple[str, str, str]]:
    """(result name, model dir, dev file) in priority order: per dev file, the trained models, then the base."""
    tags = [*models, *(["base"] if base else [])]
    return [(f"{t}_{Path(d).stem}", base_path if t == "base" else f"merged_{t}", Path(d).name) for d in devs for t in tags]


def merge_cmd(tag: str, archive: str, base_path: str, base_model: str, float32: bool = False) -> str:
    """Unpack an adapter archive on the VM and merge it into merged_TAG, stamped with its base (and float32)."""
    keep = " --keep-float32" if float32 else ""
    stamp = base_model + (" float32" if float32 else "")
    return (f"cd {M} && rm -rf adapter_{tag} merged_{tag} && mkdir adapter_{tag} && tar -xf {archive} -C adapter_{tag} && "
            f"python merge_lora.py adapter_{tag} merged_{tag} --base {base_path} --expect-base {base_model!r}{keep} && "
            f"echo {stamp!r} > merged_{tag}/BASE_MODEL")


def wait_for(session: str, marker: str, fail: str, limit_s: int, what: str) -> None:
    t = time.time()
    while time.time() - t < limit_s:
        ls = run([COLAB, "ls", "-s", session, M], 120, 3)
        if marker in ls:
            return
        if fail in ls:
            sh(session, f"tail -n 20 {M}/setup.log")
            raise SystemExit(f"{what} failed")
        time.sleep(20)
    raise SystemExit(f"{what} not done after {limit_s} s")


def start_detached(session: str, name: str, script: str, done: str, fail: str = "") -> None:
    tail = f" && touch {M}/{done}" + (f" || touch {M}/{fail}" if fail else "")
    exec_code(session, f"import subprocess\nopen('{M}/{name}.sh', 'w').write({script!r})\n"
                       f"subprocess.Popen('rm -f {M}/{done} {M}/{fail or done}; nohup bash -c \"bash {M}/{name}.sh > {M}/{name}.log 2>&1{tail}\" "
                       f"> /dev/null 2>&1 &', shell=True)")


def check(a: argparse.Namespace) -> None:
    """The Colab path without a model: guard, GPU, RAM, upload speed, install time, base size."""
    sh(a.session, "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv; "
                  "grep -E 'MemTotal|MemAvailable' /proc/meminfo; df -h /content | tail -1; python --version", 120, 3)
    exec_code(a.session, f"import os\nos.makedirs({M!r}, exist_ok=True)")
    f = Path(a.dev[0]) if a.dev else ROOT / "data/built/real_test/test.jsonl"
    how = upload(a.session, f, f"{M}/{f.name}")
    print("upload went by", how, flush=True)
    sh(a.session, f"md5sum {M}/{f.name}", 120, 3)
    t = time.time()
    sh(a.session, "pip install -q 'transformers>=5.13' safetensors huggingface_hub 2>&1 | tail -2; "
                  "python -c 'import transformers, torch; print(\"transformers\", transformers.__version__, "
                  "\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())'", 900)
    print(f"install took {time.time() - t:.0f} s", flush=True)
    base = base_dir(a.base_model)
    sh(a.session, f"cd {M} && hf download {a.base_model} config.json model.safetensors.index.json --local-dir {base} "
                  f"> /dev/null && python -c \"import json; i = json.load(open('{base}/model.safetensors.index.json')); "
                  f"print('weights GB', round(i['metadata']['total_size'] / 1e9, 2)); "
                  f"c = json.load(open('{base}/config.json')); print('arch', c['architectures'], 'dtype', "
                  f"c.get('torch_dtype') or c.get('dtype') or c.get('text_config', {{}}).get('dtype'))\"", 300, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--model", action="append", default=[], help="TAG=archive.tar.gz")
    ap.add_argument("--base", action="store_true", help="also score the base model")
    ap.add_argument("--dev", action="append", default=[])
    ap.add_argument("--out")
    ap.add_argument("--attach", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--max-min", type=int, default=70)
    ap.add_argument("--base-model", required=True, help="HF id the adapters were trained on, e.g. Qwen/Qwen3.5-4B")
    ap.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    ap.add_argument("--merge-float32", action="store_true",
                    help="keep the merged weights in float32 (merge_lora --keep-float32); score them with --dtype float32")
    ap.add_argument("--check", action="store_true", help="check the Colab path without loading a model, then stop")
    ap.add_argument("--gen", type=int, default=2,
                    help="greedy completions per job: the first N rows of each --dev file (local_eval --gen)")
    ap.add_argument("--batch", choices=("auto", "1"), default="auto",
                    help="auto: batch 4 when the probe's batchcheck passes, else 1; 1: batch 1 only")
    ap.add_argument("--max-tokens", type=int, help="new tokens per completion (local_eval --max-tokens; its default if unset)")
    a = ap.parse_args()
    if not a.check and (not a.dev or not a.out):
        ap.error("--dev and --out are required unless --check")
    base = base_dir(a.base_model)
    started = time.time()
    if not a.attach:
        print(run([str(GUARD), a.session, "L4", str(a.max_min), "20"], 1000), flush=True)
    deadline = started + a.max_min * 60 - 8 * 60
    try:
        if a.check:
            check(a)
            return
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        models = dict(m.split("=", 1) for m in a.model)
        probe = Path(tempfile.mkdtemp()) / f"probe_{Path(a.dev[0]).stem}.jsonl"
        probe.write_text("".join(json.dumps(r) + "\n" for r in longest_rows(Path(a.dev[0]))))
        exec_code(a.session, f"import os\nos.makedirs({M!r}, exist_ok=True)")
        start_detached(a.session, "setup", SETUP.format(M=M, base=base, model=a.base_model), "SETUP_DONE", "SETUP_FAIL")
        for f in (ROOT / "scripts/local_eval.py", ROOT / "src/matcher/evalstats.py", ROOT / "scripts/merge_lora.py",
                  probe, *map(Path, a.dev), *map(Path, models.values())):
            upload(a.session, f, f"{M}/{f.name}")
        wait_for(a.session, "SETUP_DONE", "SETUP_FAIL", 1800, "install and base download")
        sh(a.session, f"tail -n 3 {M}/setup.log")
        stamp = a.base_model + (" float32" if a.merge_float32 else "")
        for tag, arc in models.items():
            if "MERGED" in sh(a.session, f"test -f {M}/merged_{tag}/model.safetensors && "
                                         f"grep -qxF {stamp!r} {M}/merged_{tag}/BASE_MODEL && echo MERGED || true"):
                continue
            sh(a.session, ram_check(f"{M}/{base}", 4.0 if a.merge_float32 else 2.5))
            sh(a.session, merge_cmd(tag, Path(arc).name, base, a.base_model, a.merge_float32), 1800)
        jobs = job_list(list(models), a.base, a.dev, base)
        rows = {Path(d).name: sum(1 for line in open(d) if line.strip()) for d in a.dev}
        script = jobs_script(jobs, rows, deadline, a.dtype, probe.name, a.gen, a.max_tokens, a.batch)
        exec_code(a.session, f"import subprocess\nopen('{M}/jobs.sh', 'w').write({script!r})\n"
                             f"subprocess.Popen('rm -f {M}/ALL_DONE; nohup bash {M}/jobs.sh > {M}/jobs.log 2>&1 &', shell=True)")
        names = [n for n, _, _ in jobs]
        print("started", len(names), "scoring jobs in this order:", ", ".join(names), flush=True)
        got: set[str] = set()
        while len(got) < len(names):
            time.sleep(30)
            ls = run([COLAB, "ls", "-s", a.session, M], 120, 3)
            for name in names:
                # the whole file name: tag e's e_X.json is a substring of probe_X.jsonl
                if name not in got and re.search(rf"(?<![\w.-]){re.escape(name)}\.json(?![\w.])", ls):
                    run([COLAB, "download", "-s", a.session, f"{M}/{name}.json", str(out / f"{name}.json")], 300, 3)
                    got.add(name)
                    print("downloaded", out / f"{name}.json", flush=True)
            if "ALL_DONE" in ls and len(got) < len(names):
                sh(a.session, f"tail -n 5 {M}/*.log")
                raise SystemExit(f"jobs ended without {sorted(set(names) - got)}")
    finally:
        if not a.check and a.out:
            for f in ("RUN_CHOICE", "decide_float32.json", "decide_bfloat16.json", "probe_float32_b1.json",
                      "probe_bfloat16_b1.json", "jobs.log"):
                subprocess.run(["timeout", "120", COLAB, "download", "-s", a.session, f"{M}/{f}", str(Path(a.out) / f)],
                               capture_output=True)
        if not a.keep:
            print(run([COLAB, "stop", "-s", a.session], 180), flush=True)


if __name__ == "__main__":
    sys.exit(main())
