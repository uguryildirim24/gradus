"""Deploy the Modal demo in one command, optionally swapping in a new adapter first. Run with the Python that
runs the Modal CLI (it needs the modal package):

    PY=~/.local/share/uv/tools/modal/bin/python
    $PY scripts/modal_demo_deploy.py                                  # redeploy with demo/modal_settings.json
    $PY scripts/modal_demo_deploy.py --set 'caps={"2026-11": 20}'     # change a setting, then deploy
    $PY scripts/modal_demo_deploy.py --adapter ADAPTER_DIR --set model=v3_4b --set dtype=bfloat16 --set gpu_mem_gib=24

With --adapter: uploads ADAPTER_DIR (adapter_config.json + adapter_model.safetensors + config.json) to the
matcher-demo Volume as adapters/<model> (ADAPTER_DIR may also be the run's archive as Adaption gives it, zstd
or gzip whatever its name says: it is unpacked into a fresh temporary folder first, never into the repo), merges it on Modal (`prepare`: base from the adapter's own
adapter_config.json, checked against the public Qwen config) and prints the merge report. Then it writes
the settings, deploys, and gracefully stops containers that started before this deploy, so the new model
and cap apply at once (an old warm container would otherwise keep its old settings until it scales down).
Nothing runs a model on the Mac.

Swapping in run B, the shipped 4B (one command, from the repo root; docs/DEMO.md "Swap to run B + H''" has the
whole sequence and what each step costs):

    ~/.local/share/uv/tools/modal/bin/python scripts/modal_demo_deploy.py \\
        --adapter data/models/main_4b_c3/B_8dc3d64a-cd0f-4807-a6e9-4eadb542ed3d.tar.gz \\
        --set model=main_4b_B --set dtype=bfloat16 --set gpu_mem_gib=16 \\
        --set 'caps={"2026-10": CAP}'

The archive is Adaption's run 8dc3d64a-cd0f-4807-a6e9-4eadb542ed3d; eval/selection_cycle3.json pins its sha256
(e0895cbe...eb35). bfloat16 is the dtype every B eval ran. There is no
temperature setting: the decider reads P at temperature 1 (matcher.decider), and the page shows those probabilities.

The demo reads the three label scores with the eval's own code (evalstats.LabelHead, as local_eval.py and so
colab_eval do: float32 hidden state times the float32 label rows of the head) and the label with run B's decider
(matcher.decider, the cut-offs of eval/selection_cycle3.json). What bfloat16 changes is the hidden state itself, a
little; B's evals ran bfloat16 too.

Then check it against the eval (needs a month cap above zero; about $0.10 with a cold start and the idle tail):

    ~/.local/share/uv/tools/modal/bin/python scripts/modal_demo_check.py URL \
        --eval <colab_eval out>/<TAG>_<rows stem>.json --rows ROWS.jsonl --n 40

(for B: --eval data/models/c3_eval/B/B_fresh_test.json --rows data/built/fit/eval_inputs/B/fresh_test.jsonl)

with a colab_eval result of the same adapter (colab_eval writes DIR/<TAG>_<rows stem>.json). It scores those
rows on the deployed GPU and compares label and scores row by row; the eval's label is the decider on its stored
scores. Gate: 98% of labels agree once near-cuts (the eval's s within 0.02 of a cut-off) are set aside. B's evals ran
bfloat16, as the demo does, so flips should all but vanish. If real flips show up, deploy float32 instead
(`--set dtype=float32 --set gpu_mem_gib=32`: 4B float32 is about 18 GB of the L4's 24, and the bigger container costs
about $0.13/h more) and check again.
`--compare 6` still compares a persona's postings with the CPU float32 `reference` function.

Measured with the 0.8B stand-in (float32, L4, 2026-10-19): cold start 21-43 s once the Triton kernels are
cached in the Volume (128-140 s the first time, 347 s when an L4 was preempted mid-start); 1.3 s per
posting with a reason, 0.19 s without; a cold visit with one 20-posting request costs about $0.07 of
container time, a warm request about $0.008.
Measured for run B (4B, bfloat16, L4, first deploy 2026-10-22): the first cold start took 107 s from the
first request to a ready model (model load 100 s); 0.27 s per posting without a
reason and 2.7 s with one; a 20-posting request with reasons 57 s, under the 110 s call limit. That cold start,
one request with reasons and one without cost $0.055 of counted spend before the idle tail; 40 eval rows on a
warm GPU, $0.009.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = ROOT / "demo" / "modal_settings.json"
MODAL = str(Path.home() / ".local/bin/modal")
APP = "gradus"


def run(*args: str, capture: bool = False) -> str:
    print("$ modal", " ".join(args), flush=True)
    r = subprocess.run([MODAL, *args], cwd=ROOT, text=True, capture_output=capture)
    if r.returncode:
        raise SystemExit(f"modal {args[0]} failed ({r.returncode})\n{(r.stdout or '') + (r.stderr or '')}")
    return r.stdout or ""


ARCHIVE_MAGIC = {b"\x28\xb5\x2f\xfd": "--zstd", b"\x1f\x8b": "-z"}  # zstd, gzip: read from the bytes, not the name


def unpack_adapter(archive: Path, dest: Path) -> Path:
    """Unpack a run archive into dest (an empty folder) and return the folder holding adapter_config.json."""
    with archive.open("rb") as f:
        head = f.read(4)
    flag = next((f for m, f in ARCHIVE_MAGIC.items() if head.startswith(m)), None)
    if flag is None:
        raise SystemExit(f"{archive}: neither zstd nor gzip")
    r = subprocess.run(["tar", flag, "-xf", str(archive), "-C", str(dest)], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"could not unpack {archive}: {r.stderr.strip()}")
    found = sorted(p.parent for p in dest.rglob("adapter_config.json"))
    if len(found) != 1:
        raise SystemExit(f"{archive}: expected one adapter_config.json, found {len(found)}")
    return found[0]


def check_adapter(adir: Path) -> None:
    for f in ("adapter_config.json", "adapter_model.safetensors", "config.json"):
        if not (adir / f).exists():
            raise SystemExit(f"{adir} has no {f}")


def parse_set(items: list[str]) -> dict:
    out = {}
    for it in items:
        key, _, raw = it.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="LoRA adapter directory, or the run's archive, to merge and serve "
                                      "(needs --set model=NAME)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="change a setting before deploying")
    a = ap.parse_args()
    settings = json.loads(SETTINGS.read_text())
    changes = parse_set(a.set)
    unknown = set(changes) - set(settings)
    if unknown:
        raise SystemExit(f"unknown settings {sorted(unknown)}; known: {sorted(settings)}")
    settings.update(changes)
    if a.adapter:
        if "model" not in changes:
            raise SystemExit("--adapter needs --set model=NAME (the name it is stored and served under)")
        src = Path(a.adapter).expanduser()
        with tempfile.TemporaryDirectory(prefix="matcher-adapter-") as tmp:
            adir = unpack_adapter(src, Path(tmp)) if src.is_file() else src
            check_adapter(adir)
            name = settings["model"]
            run("volume", "put", "--force", "matcher-demo", str(adir), f"/adapters/{name}")
        run("run", "demo/modal_app.py::prepare", "--name", name)
        report = run("volume", "get", "--force", "matcher-demo", f"/models/{name}/MERGED_FROM.json", "-", capture=True)
        print(report)
    SETTINGS.write_text(json.dumps(settings, indent=1) + "\n")
    started = datetime.now(timezone.utc)
    run("deploy", "demo/modal_app.py")
    rows = json.loads(run("container", "list", "--json", capture=True) or "[]")
    for row in rows:
        if row.get("app_name") != APP:
            continue
        if datetime.fromisoformat(row["start_time"]).astimezone(timezone.utc) < started:
            try:
                run("container", "stop", "--graceful", "--yes", row["container_id"], capture=True)
            except SystemExit:
                run("container", "stop", "--yes", row["container_id"], capture=True)
    print("deployed with", json.dumps(settings))


if __name__ == "__main__":
    sys.exit(main())
