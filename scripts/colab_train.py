"""Train B-colab or C on a Colab H100/A100 with B's recipe (eval/protocol_amendment_6b.json colab_training), driven
from the Mac (no model runs here). One model per session; the session is stopped at the end, whatever happens.

  colab_train.py TAG [--gpu H100,A100] [--max-min 300] [--probe-rows N]
      TAG Bcolab: data/built/fit/cycle3/train_B.csv (sha256 checked against eval/cycle3_train_csvs.json)
      TAG C:      data/built/fit/cycle3/relabel_C/train_C.csv (sha256 checked against eval/cycleC_train_csv.json)
      eval-loss rows: relabel_C/eval_loss_B.csv (Bcolab) or eval_loss_C.csv (C), checked the same way
  -> data/models/main_4b_c3/TAG_colab_<UTC stamp>.tar.gz (adapter_config.json, adapter_model.safetensors,
     train_log.json) and data/models/c3_colab/TAG/{train.log, train_log.json, archive.json}

The session starts through scripts/colab_guard_fast.sh (an external Colab watchdog; GPUs tried in --gpu order). The VM
installs peft and flash-linear-attention, downloads the base, and runs scripts/colab_train_vm.py detached; the Mac
polls the log every minute, stops the session if the log has not grown for 30 minutes, and downloads the adapter
archive (colab download, else 8 MB parts) checked by md5. --probe-rows N trains N rows only and reports the time
(nothing is saved). Every colab call runs under `timeout`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from colab_eval import COLAB, M, exec_code, run, sh, start_detached, upload, wait_for  # noqa: E402

GUARD = ROOT / "scripts/colab_guard_fast.sh"
C3 = ROOT / "data/built/fit/cycle3"
RC = C3 / "relabel_C"
BASE = "Qwen/Qwen3.5-4B"
RUNS = {
    "Bcolab": {"train": C3 / "train_B.csv", "eval": RC / "eval_loss_B.csv",
               "pin": ("eval/cycle3_train_csvs.json", "data/built/fit/cycle3/train_B.csv")},
    "C": {"train": RC / "train_C.csv", "eval": RC / "eval_loss_C.csv",
          "pin": ("eval/cycleC_train_csv.json", "data/built/fit/cycle3/relabel_C/train_C.csv")},
}
SETUP = """#!/bin/bash
cd {M}
pip install -q 'transformers>=5.13' peft safetensors huggingface_hub 2>&1 | tail -2
pip install -q flash-linear-attention 2>&1 | tail -2 || true
pip uninstall -q -y torchao 2>&1 | tail -1 || true  # Colab's torchao 0.10 breaks peft's LoRA dispatch
(test -f base/config.json || hf download {base} --local-dir base > /dev/null) || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c 'import torch, transformers, peft; print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__)'
python -c 'import fla; print("fla", fla.__version__)' || echo 'fla not importable'
"""


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_pins(tag: str) -> None:
    r = RUNS[tag]
    rep = json.loads((ROOT / r["pin"][0]).read_text())
    if sha(r["train"]) != rep["files"][r["pin"][1]]["sha256"]:
        raise SystemExit(f"{r['train']} is not the file {r['pin'][0]} pins")
    crep = json.loads((ROOT / "eval/cycleC_train_csv.json").read_text())
    key = str(r["eval"].relative_to(ROOT))
    if sha(r["eval"]) != crep["files"][key]["sha256"]:
        raise SystemExit(f"{r['eval']} is not the file eval/cycleC_train_csv.json pins")
    for f in ("eval/cycle3_train_csvs.json", "eval/cycleC_train_csv.json", "eval/protocol_amendment_6b.json"):
        if subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", f]).returncode or \
                subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", f], capture_output=True).returncode:
            raise SystemExit(f"{f} is not committed unchanged")


def start(session: str, gpus: list[str], max_min: int) -> str:
    for gpu in gpus:
        r = subprocess.run(["timeout", "1000", str(GUARD), session, gpu, str(max_min), "20"], capture_output=True,
                           text=True)
        print(r.stdout + r.stderr, flush=True)
        if r.returncode == 0:
            return gpu
    raise SystemExit(f"no session on any of {gpus}")


def download(session: str, remote: str, local: Path, size: int, md5: str) -> str:
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["timeout", "900", COLAB, "download", "-s", session, remote, str(local)], capture_output=True,
                       text=True)
    if not r.returncode and local.exists() and hashlib.md5(local.read_bytes()).hexdigest() == md5:
        return "cli"
    print("whole-file download failed or came out wrong; trying 8 MB parts", flush=True)
    part = 8_000_000
    n = (size + part - 1) // part
    sh(session, f"cd {M} && split -b {part} -d -a 4 {remote} {remote}.p", 300)
    with local.open("wb") as f:
        for i in range(n):
            tmp = local.with_suffix(f".p{i:04d}")
            run([COLAB, "download", "-s", session, f"{remote}.p{i:04d}", str(tmp)], 300, 3)
            f.write(tmp.read_bytes())
            tmp.unlink()
    if hashlib.md5(local.read_bytes()).hexdigest() != md5:
        raise SystemExit("adapter archive md5 differs after the parts download")
    return "parts"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag", choices=sorted(RUNS))
    ap.add_argument("--gpu", default="H100,A100")
    ap.add_argument("--max-min", type=int, default=300)
    ap.add_argument("--probe-rows", type=int, default=0)
    a = ap.parse_args()
    check_pins(a.tag)
    r = RUNS[a.tag]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session = f"matcher-train-{a.tag.lower()}"
    local = ROOT / "data/models/c3_colab" / a.tag
    local.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    gpu = start(session, a.gpu.split(","), a.max_min)
    rec = {"tag": a.tag, "session": session, "gpu_asked": gpu, "started_utc": stamp,
           "train_csv": str(r["train"].relative_to(ROOT)), "train_sha256": sha(r["train"]),
           "eval_csv": str(r["eval"].relative_to(ROOT)), "eval_sha256": sha(r["eval"])}
    try:
        exec_code(session, f"import os\nos.makedirs({M!r}, exist_ok=True)")
        start_detached(session, "setup", SETUP.format(M=M, base=BASE), "SETUP_DONE", "SETUP_FAIL")
        for f in (ROOT / "scripts/colab_train_vm.py", r["train"], r["eval"]):
            upload(session, f, f"{M}/{f.name}")
        wait_for(session, "SETUP_DONE", "SETUP_FAIL", 1800, "install and base download")
        print(sh(session, f"tail -n 8 {M}/setup.log"), flush=True)
        out = f"{M}/out_{a.tag}"
        probe = f" --max-rows {a.probe_rows}" if a.probe_rows else ""
        cmd = (f"cd {M} && rm -rf {out} && python colab_train_vm.py --train {r['train'].name} --eval {r['eval'].name} "
               f"--out {out} --base base{probe}")
        start_detached(session, "train", cmd + "\n", "TRAIN_EXIT", "TRAIN_FAIL")
        done = "PROBE_DONE" if a.probe_rows else "TRAIN_DONE"
        last_size, last_change = -1, time.time()
        while True:
            time.sleep(60)
            st = exec_code(session, f"import os\nprint('ST', os.path.exists('{out}/{done}'), "
                                    f"os.path.exists('{M}/TRAIN_FAIL'))", 120, 3).split("ST", 1)[1].split()
            subprocess.run(["timeout", "120", COLAB, "download", "-s", session, f"{M}/train.log", str(local / "train.log")],
                           capture_output=True)
            size = (local / "train.log").stat().st_size if (local / "train.log").exists() else 0
            if size != last_size:
                last_size, last_change = size, time.time()
                tail = (local / "train.log").read_text(errors="replace").strip().splitlines()[-1:] if size else []
                print(f"{(time.time() - t0) / 60:.0f} min:", *(t[:300] for t in tail), flush=True)
            if st[0] == "True":
                break
            if st[1] == "True":
                print((local / "train.log").read_text(errors="replace")[-3000:], flush=True)
                raise SystemExit("training exited without finishing")
            if time.time() - last_change > 30 * 60:
                raise SystemExit("the training log has not grown for 30 minutes")
        if a.probe_rows:
            print("probe done", flush=True)
            return
        info = sh(session, f"cd {out} && tar -czf {M}/{a.tag}_adapter.tar.gz adapter_config.json adapter_model.safetensors "
                           f"train_log.json && python -c \"import hashlib, os; p = '{M}/{a.tag}_adapter.tar.gz'; "
                           f"print('SIZE', os.path.getsize(p), 'MD5', hashlib.md5(open(p, 'rb').read()).hexdigest())\"",
                  600)
        size = int(info.split("SIZE", 1)[1].split()[0])
        md5 = info.split("MD5", 1)[1].split()[0]
        arc = ROOT / f"data/models/main_4b_c3/{a.tag}_colab_{stamp}.tar.gz"
        how = download(session, f"{M}/{a.tag}_adapter.tar.gz", arc, size, md5)
        subprocess.run(["tar", "-xzf", str(arc), "-C", str(local), "train_log.json"], check=True)
        log = json.loads((local / "train_log.json").read_text())
        rec |= {"archive": str(arc.relative_to(ROOT)), "archive_sha256": sha(arc), "archive_md5": md5,
                "archive_bytes": size, "download": how, "gpu": log["gpu"], "wall_seconds": log["wall_seconds"],
                "session_minutes": round((time.time() - t0) / 60, 1), "final_train_loss": log["final_train_loss"],
                "final_eval_loss": log["final_eval_loss"], "eval": log["eval"], "train_rows": log["train_rows"],
                "recipe": log["recipe"], "versions": log["versions"]}
        (local / "archive.json").write_text(json.dumps(rec, indent=1))
        print("DONE", json.dumps({k: rec[k] for k in ("archive", "gpu", "wall_seconds", "final_train_loss",
                                                       "final_eval_loss")}), flush=True)
    finally:
        print(run([COLAB, "stop", "-s", session], 180), flush=True)


if __name__ == "__main__":
    sys.exit(main())
