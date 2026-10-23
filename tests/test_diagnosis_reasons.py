import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import colab_eval as ce  # noqa: E402
import diagnosis_reasons as dr  # noqa: E402


def test_allocate_follows_the_error_mix_and_hands_on_what_a_capped_kind_cannot_take():
    got = dr.allocate(Counter(a=30, b=10, c=10), Counter(a=6, b=40, c=40), 25)
    assert got["a"] == 6 and sum(got.values()) == 25 and abs(got["b"] - got["c"]) <= 1
    assert dr.allocate(Counter(a=1), Counter(a=3), 10) == {"a": 3}


def test_subset_takes_every_decidable_error_then_right_rows_of_the_error_kinds():
    rows = ([dict(pair_id=f"e{i}", gold="Keep", pred="Skip", variant_kind="k1") for i in range(4)]
            + [dict(pair_id="hidden", gold="Keep", pred="Skip", variant_kind="k1", visible=False)]
            + [dict(pair_id=f"r{i}", gold="Skip", pred="Skip", variant_kind="k1" if i < 5 else "k2") for i in range(10)])
    out = dr.subset_rows(rows, 3, 0)
    assert [p for p, k in out if k == "error"] == ["e0", "e1", "e2", "e3"]
    right = [p for p, k in out if k == "correct"]
    assert len(right) == 3 and all(int(p[1:]) < 5 for p in right)  # only kind k1 had errors
    assert dr.subset_rows(rows, 3, 0) == out  # seeded


def test_assemble_keeps_the_reference_scores_and_refuses_a_row_without_a_generation():
    sub = [dict(pair_id="p1", diag_set="real_test", diag_sample="error", completion="Keep — x", variant_kind="k")]
    ref = {"rows": [dict(pair_id="p1", gold="Keep", pred="Skip", scores=[1.0, 2.0, 3.0])]}
    now = {"rows": [dict(pair_id="p1", pred="Skip", scores=[1.0, 2.0, 3.5])],
           "generations": [dict(pair_id="p1", text="Skip — not in the city.\n")]}
    (r,) = dr.assemble_rows(sub, ref, now)
    assert (r["gold"], r["pred"], r["scores"], r["gen_label"], r["max_score_diff"]) == ("Keep", "Skip", [1.0, 2.0, 3.0], "Skip", 0.5)
    assert r["generation"] == "Skip — not in the city."
    with pytest.raises(SystemExit):
        dr.assemble_rows(sub, ref, {"rows": now["rows"], "generations": []})


def test_colab_eval_gen_and_max_tokens_default_to_the_old_jobs():
    jobs = ce.job_list(["c"], False, ["reasons_test.jsonl"], "base_q")
    old = ce.jobs_script(jobs, {"reasons_test.jsonl": 395}, 1.8e9, "bfloat16", "probe.jsonl")
    assert "--gen 2 --out c_reasons_test.json" in old and "--max-tokens" not in old
    new = ce.jobs_script(jobs, {"reasons_test.jsonl": 395}, 1.8e9, "bfloat16", "probe.jsonl", 1000, 80)
    assert new.count("--gen 1000 --max-tokens 80 --out c_reasons_test.json") == 2
