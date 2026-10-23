"""Cycle-3 training sets for runs A and B (scripts/build_fit_train3.py): labels pooled by row id, invalid labels
dropped, fresh-test postings and employers dropped, the label-mix band met by downsampling only, and the two prompt
formats on the same rows."""
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import build_fit_train3 as b3  # noqa: E402
from matcher.pairs import INSTRUCTION, build_prompt  # noqa: E402

BAND = json.loads((ROOT / "eval/protocol_amendment_5.json").read_text())["training_label_mix"]["band"]


def test_the_band_is_amendment_5s():
    assert BAND == {"Keep": [0.12, 0.3], "Explore": [0.22, 0.4], "Skip": [0.4, 0.65]}


def test_system_b_is_the_labeler_prompts_opening_and_the_rubric_verbatim():
    t = json.loads((ROOT / "eval/protocol_amendment_5.json").read_text())["teacher"]
    s = b3.system_b()
    assert t["labeler_prompt_test3"].startswith(s)
    assert s.endswith(t["rubric_bullets"])
    assert hashlib.sha256(s[-len(t["rubric_bullets"]):].encode()).hexdigest() == t["rubric_bullets_sha256"]
    opening = s[:-len(t["rubric_bullets"])]
    assert opening.startswith("You are labeling job-match rows") and opening.endswith("Today is October 2026.\n\n")
    assert "Judge only" not in s and "batch file" not in s


def test_system_b_refuses_an_edited_rubric():
    a = json.loads((ROOT / "eval/protocol_amendment_5.json").read_text())
    a["teacher"]["rubric_bullets"] = a["teacher"]["rubric_bullets"].replace("Skip:", "Skip -")
    with pytest.raises(SystemExit):
        b3.system_b(a)


def test_prompt_b_is_prompt_a_with_the_rubric_for_instruction_and_the_whole_posting():
    s = b3.system_b()
    short = "Title: Analyst\nCompany: Acme\n\nYou need SQL."
    pa, pb = b3.prompt_a("Jo Doe\nSQL", "Years: 2", short), b3.prompt_b("Jo Doe\nSQL", "Years: 2", short, s)
    assert pa.startswith(INSTRUCTION + "\nRESUME:\n")
    assert pb == s + "\n\n" + pa[len(INSTRUCTION) + 1:]
    long = "Title: Analyst\nCompany: Acme\n\n" + "\n".join(f"Duty line {i} about the work we do here." for i in range(400))
    pa, pb = b3.prompt_a("r", "p", long), b3.prompt_b("r", "p", long, s)
    assert len(long) > 6000 and long.strip() in pb and long.strip() not in pa
    assert pa == build_prompt("r", "p", long)


def test_judge():
    assert b3.judge([{"label": "Keep", "reason": "meets it."}])[1] is None
    assert b3.judge([{"label": "keep", "reason": "x"}])[1] == "bad_label_word"
    assert b3.judge([{"label": "Skip", "reason": "  . "}])[1] == "empty_reason"
    assert b3.judge([{"label": "Skip", "reason": "a"}, {"label": "Keep", "reason": "b"}])[1] == "conflicting_labels"
    assert b3.judge([{"label": "Skip", "reason": "a"}, {"label": "Skip", "reason": "b"}])[1] is None


def test_read_labels_pools_files_and_records_unparsed_lines(tmp_path):
    (tmp_path / "batch_001.jsonl").write_text('{"row_id": "r1", "label": "Keep", "reason": "x"}\n{"row_id": "r2"\n')
    (tmp_path / "batch_002.jsonl").write_text('{"row_id": "r1", "label": "Keep", "reason": "y"}\n\n')
    by, bad = b3.read_labels(sorted(tmp_path.glob("*.jsonl")))
    assert [x["_file"] for x in by["r1"]] == ["batch_001.jsonl", "batch_002.jsonl"] and "r2" not in by
    assert bad == [{"file": "batch_001.jsonl", "line": 2, "error": bad[0]["error"]}]


@pytest.mark.parametrize("have", [{"Keep": 701, "Explore": 591, "Skip": 2810}, {"Keep": 100, "Explore": 100, "Skip": 100},
                                  {"Keep": 50, "Explore": 300, "Skip": 400}, {"Keep": 2000, "Explore": 300, "Skip": 900},
                                  {"Keep": 300, "Explore": 300, "Skip": 400}])
def test_band_counts_is_the_largest_in_band_set_and_cuts_only_labels_over_their_band(have):
    keep = b3.band_counts(have, BAND)
    n = sum(keep.values())
    assert b3.in_band(keep, BAND) and all(keep[w] <= have[w] for w in have)
    total = sum(have.values())
    scarce = min(have, key=lambda w: have[w] / BAND[w][0])  # the label that sets the size keeps every row
    assert keep[scarce] == have[scarce] or b3.in_band(have, BAND)
    # no bigger set fits: brute force one row more
    for m in range(n + 1, min(total, n + 30) + 1):
        assert not any(b3.in_band({"Keep": k, "Explore": e, "Skip": m - k - e}, BAND)
                       for k in range(0, min(have["Keep"], m) + 1) for e in range(0, min(have["Explore"], m - k) + 1)
                       if m - k - e <= have["Skip"])


def test_band_counts_today():
    assert b3.band_counts({"Keep": 701, "Explore": 591, "Skip": 2810}, BAND) == {"Keep": 701, "Explore": 591,
                                                                                 "Skip": 1394}


def test_downsample_is_seeded_never_relabels_and_leaves_an_in_band_set_alone():
    rows = [{"row_id": f"r{i:04d}", "label": w} for i, w in enumerate(["Keep"] * 20 + ["Explore"] * 15 + ["Skip"] * 90)]
    random.Random(1).shuffle(rows)
    k1, d1 = b3.downsample(rows, BAND, 7)
    k2, d2 = b3.downsample(list(reversed(rows)), BAND, 7)
    assert {r["row_id"] for r in k1} == {r["row_id"] for r in k2} and d1["dropped"] == d2["dropped"]
    assert list(d1["dropped"]) == ["Skip"] and b3.in_band(Counter(r["label"] for r in k1), BAND)
    assert [r for r in rows if r in k1] == k1  # order kept
    lab = {r["row_id"]: r["label"] for r in rows}
    assert all(lab[r["row_id"]] == r["label"] for r in k1)
    k3, _ = b3.downsample(rows, BAND, 8)
    assert {r["row_id"] for r in k3} != {r["row_id"] for r in k1}
    ok = [{"row_id": str(i), "label": w} for i, w in enumerate(["Keep"] * 2 + ["Explore"] * 3 + ["Skip"] * 5)]
    assert b3.downsample(ok, BAND, 7) == (ok, {"needed": False, "seed": 7, "keep": {"Keep": 2, "Explore": 3, "Skip": 5},
                                              "dropped": {}})


def _fixture(tmp_path, monkeypatch, labels: dict[str, dict | None], extra_lines: str = ""):
    """Ten new rows in two batches; labels maps row_id -> label line (None: no line)."""
    bd, ld = tmp_path / "label_batches", tmp_path / "labels"
    bd.mkdir()
    ld.mkdir()
    rows = []
    for i in range(10):
        emp = "Fresh Corp" if i == 9 else f"Emp {i}"
        rows.append({"row_id": f"c3_{i}", "posting": f"Title: Role {i}\nCompany: {emp}\n\nYou need skill {i} and "
                     f"{'some more words here ' * 3}{i}.", "resume": f"Candidate\nskill {i}", "profile": "Years: 3"})
    for n, part in enumerate((rows[:5], rows[5:]), 1):
        (bd / f"batch_00{n}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part))
        (ld / f"batch_00{n}.jsonl").write_text("".join(json.dumps(labels[r["row_id"]]) + "\n" for r in part
                                                       if labels.get(r["row_id"])) + (extra_lines if n == 2 else ""))
    idx = tmp_path / "label_index.jsonl"
    idx.write_text("".join(json.dumps({"row_id": r["row_id"], "source": "new", "origin": r["row_id"],
                                       "posting_key": "ft:1" if r["row_id"] == "c3_8" else f"k:{r['row_id']}",
                                       "title_copy": False}) + "\n" for r in rows))
    posts = tmp_path / "postings.jsonl"
    posts.write_text("".join(json.dumps({"key": f"k:c3_{i}", "employer": f"Emp {i}"}) + "\n" for i in range(10)))
    tf = tmp_path / "train_fit.csv"
    tf.write_text("prompt,completion\n")
    for k, v in {"BATCHES": bd, "LABELS": ld, "INDEX": idx, "C3_POSTINGS": posts, "LEAK_ROWS": tmp_path / "none",
                 "TRAIN_FIT": tf}.items():
        monkeypatch.setattr(b3, k, v)
    return {"ids": {"ft:1": "fresh_test4"}, "emps": {"fresh": "fresh_test3"}, "texts": {}, "fresh_test4_manifest": False,
            "counts": {}}


WORD = ["Keep", "Keep", "Explore", "Explore", "Skip", "Skip", "Skip", "Skip", "Skip", "Skip"]


def test_build_writes_a_and_b_on_the_same_rows_and_records_every_drop(tmp_path, monkeypatch):
    labels = {f"c3_{i}": {"row_id": f"c3_{i}", "label": w, "reason": f"Reason {i}.\n second line", "confidence": "low"}
              for i, w in enumerate(WORD)}
    labels["c3_7"]["label"] = "Maybe"
    fresh = _fixture(tmp_path, monkeypatch, labels)
    out = tmp_path / "out"
    rep = b3.build(False, out_dir=out, eval_copy=tmp_path / "eval.json", fresh=fresh)
    assert rep["dropped"]["invalid"] == {"bad_label_word": ["c3_7"]}
    assert set(rep["dropped"]["fresh_test_overlap"]) == {"c3_8", "c3_9"}  # posting key; Company line employer
    assert rep["confidence_of_valid_labels"] == {"low": 9}  # low confidence is kept
    with open(out / "train_A.csv", newline="") as fa, open(out / "train_B.csv", newline="") as fb:
        a, b = list(csv.DictReader(fa)), list(csv.DictReader(fb))
    assert list(a[0]) == ["prompt", "completion"] == list(b[0])
    assert len(a) == len(b) == rep["rows"] and [r["completion"] for r in a] == [r["completion"] for r in b]
    assert b3.in_band(Counter(r["completion"].split()[0] for r in a), BAND)
    assert all(r["completion"].startswith(("Keep — Reason", "Explore — Reason", "Skip — Reason"))
               and r["completion"].endswith(" second line.") for r in a)
    assert all(r["prompt"].startswith(INSTRUCTION) for r in a)
    assert all(r["prompt"].startswith(b3.system_b() + "\n\nRESUME:\n") for r in b)
    side = [json.loads(l) for l in (out / "train3_rows.jsonl").open()]
    assert [s["csv_row"] for s in side if s["kept"]] == list(range(len(a)))
    assert {s["row_id"]: s["why"] for s in side if not s["kept"]}["c3_7"] == "bad_label_word"
    assert json.loads((tmp_path / "eval.json").read_text())["files"] == rep["files"]


def test_the_real_build_refuses_missing_labels_and_the_dry_run_drops_them(tmp_path, monkeypatch):
    labels = {f"c3_{i}": {"row_id": f"c3_{i}", "label": w, "reason": "r", "confidence": "high"}
              for i, w in enumerate(WORD) if i != 3}
    fresh = _fixture(tmp_path, monkeypatch, labels, extra_lines='{"row_id": "c3_3", "lab\n')
    with pytest.raises(SystemExit):
        b3.build(False, out_dir=tmp_path / "out", eval_copy=None, fresh=fresh)
    rep = b3.build(True, out_dir=tmp_path / "dry", eval_copy=tmp_path / "eval.json", fresh=fresh)
    assert rep["dropped"]["no_label_yet"] == 1 and len(rep["label_lines"]["unparsed_lines"]) == 1
    assert not (tmp_path / "eval.json").exists()


def test_a_row_labeled_in_another_batchs_file_counts_once_and_conflicts_drop_it(tmp_path, monkeypatch):
    labels = {f"c3_{i}": {"row_id": f"c3_{i}", "label": w, "reason": "r", "confidence": "high"}
              for i, w in enumerate(WORD)}
    fresh = _fixture(tmp_path, monkeypatch, labels,
                     extra_lines=json.dumps(labels["c3_0"]) + "\n" + json.dumps(labels["c3_1"] | {"label": "Skip"}) + "\n")
    rep = b3.build(False, out_dir=tmp_path / "out", eval_copy=None, fresh=fresh)
    assert rep["label_lines"]["rows_labeled_in_another_batch_file"] == ["c3_0", "c3_1"]
    assert rep["dropped"]["invalid"] == {"conflicting_labels": ["c3_1"]}
