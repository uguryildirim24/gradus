import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import colab_eval as ce  # noqa: E402


def test_jobs_run_in_priority_order_set_first_trained_model_before_base():
    jobs = ce.job_list(["4b"], True, ["x/test.jsonl", "y/dev_cross.jsonl", "y/dev.jsonl"], "base_Qwen--Qwen3.5-4B")
    assert [n for n, _, _ in jobs] == ["4b_test", "base_test", "4b_dev_cross", "base_dev_cross", "4b_dev", "base_dev"]
    assert jobs[1] == ("base_test", "base_Qwen--Qwen3.5-4B", "test.jsonl")


def test_jobs_script_probes_then_runs_every_job_in_one_dtype():
    jobs = ce.job_list(["4b"], True, ["test.jsonl"], "base_q")
    auto = ce.jobs_script(jobs, {"test.jsonl": 1183}, 1.8e9, "auto", "probe_test.jsonl")
    assert "--forwards 2366 --jobs 2" in auto and "probe float32" in auto and "probe bfloat16" in auto
    assert auto.index("RUN_CHOICE") < auto.index("4b_test.json") < auto.index("base_test.json")
    assert auto.count("--dtype $DTYPE --gen 2") == 4 and auto.count("--batch $BATCH >") == 2
    assert "--dev probe_test.jsonl" in auto and "--dev test.jsonl" in auto
    # a job lost at batch 4 runs once more at batch 1
    assert '[ "$BATCH" = 1 ] || python local_eval.py score merged_4b --dev test.jsonl' in auto
    fixed = ce.jobs_script(jobs, {"test.jsonl": 1183}, 1.8e9, "bfloat16", "probe_test.jsonl")
    assert "probe bfloat16" in fixed and "probe float32" not in fixed
    assert "for b in 1 4;" in fixed and "--b4 probe_$1_b4.json" in fixed
    pinned = ce.jobs_script(jobs, {"test.jsonl": 1183}, 1.8e9, "bfloat16", "probe_test.jsonl", batch="1")
    # --batch 1: the probe runs at batch 1 only and decide gets no batch-4 probe, so it picks batch 1
    assert "for b in 1;" in pinned and "--b4" not in pinned and "probe_$1_b1.json --forwards 2366" in pinned


def test_the_probe_takes_the_longest_prompts(tmp_path):
    dev = tmp_path / "test.jsonl"
    dev.write_text("".join(json.dumps({"pair_id": f"p{i}", "prompt": "x" * n}) + "\n" for i, n in enumerate((5, 50, 9, 70, 1))))
    assert [r["pair_id"] for r in ce.longest_rows(dev, 2)] == ["p3", "p1"]


def test_cli_errors_are_printed_without_the_proxy_token():
    msg = "v/api/contents/content/m/x.bin?authuser=0&colab-runtime-proxy-token=FAKE-TOKEN-123 failed"
    assert ce.quiet(msg) == "v/api/contents/content/m/x.bin?authuser=0&colab-runtime-proxy-token=... failed"


def test_merge_float32_passes_keep_float32_and_stamps_the_merge():
    plain = ce.merge_cmd("4b", "a.tar.gz", "base_q", "Qwen/Qwen3.5-4B")
    f32 = ce.merge_cmd("4b", "a.tar.gz", "base_q", "Qwen/Qwen3.5-4B", True)
    assert "--keep-float32" not in plain and plain.endswith("echo 'Qwen/Qwen3.5-4B' > merged_4b/BASE_MODEL")
    assert "--expect-base 'Qwen/Qwen3.5-4B' --keep-float32 &&" in f32
    assert f32.endswith("echo 'Qwen/Qwen3.5-4B float32' > merged_4b/BASE_MODEL")
