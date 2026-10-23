"""Card-check selection: train postings ranked by high-gap Keep/Explore rows, dev picked to spread variant kinds,
batches dealt evenly."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import select_cardcheck as sc  # noqa: E402


def row(pid, post, label, kind="k"):
    return {"pair_id": pid, "base_rq_id": post, "completion": f"{label} because", "variant_kind": kind}


def test_train_scores_count_only_keep_explore_over_gap():
    rows = [row("a", "p1", "Keep"), row("b", "p1", "Skip"), row("c", "p1", "Explore"), row("d", "p1", "Keep")]
    gaps = {"a": 5, "b": 9, "c": 7, "d": 4}
    s = sc.train_scores(rows, gaps)["p1"]
    assert s == {"rows": 4, "keep_explore": 3, "keep_explore_gap": 2}  # Skip at 9 and Keep at exactly 4 don't count


def test_pick_train_ranks_then_ties_by_id_and_drops_zero():
    scores = {"p3": {"keep_explore_gap": 2}, "p1": {"keep_explore_gap": 2}, "p2": {"keep_explore_gap": 5},
              "p0": {"keep_explore_gap": 0}}
    assert sc.pick_train(scores, 2) == ["p2", "p1"]
    assert sc.pick_train(scores, 10) == ["p2", "p1", "p3"]


def test_pick_dev_skips_checked_fills_rare_kinds_and_stops_at_target():
    rows = [row("1", "c1", "Keep", "common"), row("2", "c1", "Skip", "common"),
            row("3", "d1", "Skip", "common"), row("4", "d2", "Skip", "rare"), row("5", "d3", "Skip", "common")]
    picked = sc.pick_dev(rows, checked={"c1"}, target=2)
    assert picked == ["d2"]  # the uncovered kind wins; checked c1 counts toward the target
    assert sc.pick_dev(rows, checked={"c1"}, target=10) == ["d2", "d1", "d3"]  # ties by id


def test_deal_is_even_and_keeps_every_posting_once():
    train = [f"t{i:02d}" for i in range(10)]
    dev = [f"d{i:02d}" for i in range(5)]
    b = sc.deal(train, dev, 4)
    flat = [x for it in b for x in it]
    assert sorted(flat) == sorted([(i, "train") for i in train] + [(i, "dev") for i in dev])
    assert max(len(x) for x in b) - min(len(x) for x in b) <= 1
    for side in ("train", "dev"):
        n = [sum(s == side for _, s in it) for it in b]
        assert max(n) - min(n) <= 1


OUT = sc.OUT


@pytest.mark.skipif(not (OUT / "batch_1.json").exists(), reason="batches not built")
def test_written_batches():
    checked = {r["id"] for r in sc.read_jsonl(sc.CHECKED)}
    splits = json.load(open(sc.RT / "splits.json"))
    seen = []
    for b in range(1, 5):
        for e in json.load(open(OUT / f"batch_{b}.json")):
            assert {"posting_id", "split", "raw_path", "raw_key", "extracted_card"} <= e.keys()
            assert e["extracted_card"]["source_posting_id"] == e["posting_id"]
            assert splits[e["posting_id"]] == e["split"]
            assert e["posting_id"] not in checked
            seen.append(e["posting_id"])
    assert len(seen) == len(set(seen))
