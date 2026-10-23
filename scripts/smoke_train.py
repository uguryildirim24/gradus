"""AutoScientist smoke run on data/built/training/smoke.csv. State lives in data/adaption/smoke_run_v2.json.
The v2 smoke run only (Qwen3.5-0.8B, 1,344 rows): upload, hparams and create refuse when smoke.csv no longer has the
stored dataset's row count. The main set trains through scripts/train_main.py.

  smoke_train.py upload     raw upload (prompt/completion), wait until the dataset is ready
  smoke_train.py hparams    recommend_hyperparams for the model (free)
  smoke_train.py create     read credits, launch the run with the recommended recipe plus the
                            OVERRIDES below, read credits again
  smoke_train.py wait       poll with backoff until terminal; record win rate, iterations, run time, credits
  smoke_train.py download   save the best checkpoint archive to data/models/smoke_v2/<run id>.tar.gz

autoscientist.create has no estimate, so credits are read before and after (datasets.invent with
estimate=True returns available_credits and costs nothing).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from adaption import Adaption  # noqa: E402

STATE = ROOT / "data/adaption/smoke_run_v2.json"
CSV = ROOT / "data/built/training/smoke.csv"
MODELS = ROOT / "data/models/smoke_v2"
MODEL = "Qwen/Qwen3.5-0.8B"
# v1 trained 1 epoch (24 steps) and eval loss was still falling. Only n_epochs is overridden, so
# AutoScientist still derives every other value (and can retune them in iteration 2).
OVERRIDES = {"n_epochs": 3}
MAX_ITERATIONS = 2
# v2 training draws credits as it runs (v1 drew none): cancel past this, polling every 2 minutes,
# so rendering (19) + training stays inside the task's 45.
MAX_CREDITS = 24.0
MAPPING = {"prompt": "prompt", "completion": "completion"}
# a raw upload canonicalizes the mapped columns to original_prompt / original_completion
TRAIN_MAPPING = {"prompt": "original_prompt", "completion": "original_completion"}


def client() -> Adaption:
    os.environ.setdefault("ADAPTION_API_KEY", Path("~/.config/adaption/api_key").expanduser().read_text().strip())
    return Adaption()


def credits(c: Adaption) -> float:
    return c.datasets.invent(rows=10, estimate=True, prompt="balance check").available_credits


def load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=1, default=str))


def dump(obj) -> dict:
    return json.loads(obj.model_dump_json()) if hasattr(obj, "model_dump_json") else obj


def save_archive(c: Adaption, s: dict, stem: str) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    res = c.autoscientist.download(s["run_id"], timeout=1800)
    path = MODELS / f"{stem}.tar.gz"
    res.write_to_file(path)
    s["archive"] = str(path.relative_to(ROOT))
    s["archive_bytes"] = path.stat().st_size
    save(s)
    print("saved", path, path.stat().st_size, "bytes", flush=True)


def csv_rows() -> int:
    import csv
    with open(CSV, newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def main() -> None:
    cmd = sys.argv[1]
    s = load()
    stored = (s.get("dataset_status") or {}).get("row_count")
    if cmd in ("upload", "hparams", "create") and stored is not None and CSV.exists() and csv_rows() != stored:
        # the v2 state holds the 1,344-row smoke dataset; smoke.csv now holds the main set: train_main.py is its path
        sys.exit(f"refusing: {STATE.name} holds a {stored}-row dataset but {CSV.name} has {csv_rows()} rows; "
                 "the main set trains through scripts/train_main.py")
    c = client()
    if cmd == "upload":
        if "dataset_id" not in s:
            up = c.datasets.upload_file(str(CSV), name="smoke-train-v2", processing_mode="raw", column_mapping=MAPPING)
            print("UPLOAD", up)
            s.update(dataset_id=up.dataset_id, uploaded_at=datetime.now(timezone.utc).isoformat())
            save(s)
        st = c.datasets.wait_for_completion(s["dataset_id"], timeout=3600)
        print("STATUS", st)
        s["dataset_status"] = dump(st)
        save(s)
    elif cmd == "hparams":
        r = c.autoscientist.recommend_hyperparams(dataset_id=s["dataset_id"], model=MODEL)
        print("RECOMMEND", r)
        s["recommended"] = dump(r)
        save(s)
    elif cmd == "create":
        if "run_id" in s:
            sys.exit(f"run already created: {s['run_id']}")
        before = credits(c)
        print("credits before", before)
        if "recommended" not in s:
            sys.exit("run hparams first")
        run = c.autoscientist.create(dataset_id=s["dataset_id"], model=MODEL, training_method="instruction",
                                     data_format="chat", column_mapping=TRAIN_MAPPING, max_iterations=MAX_ITERATIONS,
                                     hyperparams=OVERRIDES, idempotency_key=f"smoke-v2-{s['dataset_id']}")
        print("CREATE", run)
        s.update(run_id=run.id, credits_before=before, overrides=OVERRIDES, created=dump(run),
                 created_at=datetime.now(timezone.utc).isoformat())
        save(s)
        time.sleep(20)
        s["credits_after_create"] = credits(c)
        print("credits after create", s["credits_after_create"])
        save(s)
    elif cmd == "wait":
        delay = 30.0
        while True:
            run = c.autoscientist.get(s["run_id"])
            now = credits(c)
            print(datetime.now().strftime("%H:%M:%S"), run.status, "iter", run.iterations_completed,
                  "win", run.best_win_rate, "dl", run.download_available, "credits", now, flush=True)
            s["last"] = dump(run)
            s.setdefault("credit_trace", []).append([datetime.now(timezone.utc).isoformat(), now, run.status])
            save(s)
            if run.download_available and not s.get("archive") and run.status == "running":
                # keep the best checkpoint so far, in case the budget cancels a later iteration
                save_archive(c, s, f"{s['run_id']}_iter{int(run.iterations_completed)}")
            if run.status in ("succeeded", "failed", "cancelled"):
                break
            if s["credits_before"] - now > MAX_CREDITS:
                print(f"spent {s['credits_before'] - now} > {MAX_CREDITS}; cancelling", flush=True)
                print("CANCEL", c.autoscientist.cancel(s["run_id"]), flush=True)
                s["cancelled_for_budget"] = True
                save(s)
            time.sleep(delay)
            delay = min(delay * 1.5, 120)
        s["credits_after"] = credits(c)
        start = datetime.fromisoformat(str(run.created_at))
        end = datetime.fromisoformat(str(run.completed_at)) if run.completed_at else datetime.now(timezone.utc)
        s["run_minutes"] = round((end - start).total_seconds() / 60, 1)
        s["credits_spent"] = round(s["credits_before"] - s["credits_after"], 2)
        save(s)
        print("FINAL", run)
        print(f"run {s['run_minutes']} min, credits {s['credits_before']} -> {s['credits_after']} ({s['credits_spent']})")
    elif cmd == "download":
        save_archive(c, s, s["run_id"])
    elif cmd == "status":
        print(c.autoscientist.get(s["run_id"]))
        print("credits now", credits(c))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
