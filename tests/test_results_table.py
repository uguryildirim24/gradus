import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import results_table as rt  # noqa: E402

BUCKETS = ("For you", "Explore", "Excluded")


def _src(n=12, kinds=("enrollment_graduated", "full_match", "location")):
    return [{"pair_id": f"p{i}", "rq_id": f"rq{i % 4}", "base_rq_id": f"rq{i % 4}", "bucket": BUCKETS[i % 3],
             "variant_kind": kinds[i % len(kinds)], "decidable": i != 0, "prompt": "x"} for i in range(n)]


def _result(path: Path, dev: str, rows: list[dict], pred=lambda r: rt.WORD[r["bucket"]], visible=lambda r: True):
    path.write_text(json.dumps({"model": "m", "dev": dev, "dtype": "float32",
                                "rows": [{"pair_id": r["pair_id"], "pred": pred(r), "visible": visible(r)} for r in rows]}))
    return path


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_test_set_takes_decidable_from_the_current_file_not_the_result(tmp_path, monkeypatch):
    src = _src()
    monkeypatch.setattr(rt, "rules_entries", lambda rows: {"rules": {r["pair_id"]: "Skip" for r in rows}})
    # the result still says p0 is visible (scored before p0 was marked): the table leaves it out anyway
    res = _result(tmp_path / "4b_test.json", "test.jsonl", src)
    out = rt.build([("", res)], 20, {"test": _write(tmp_path / "test.jsonl", src)})["test"]
    assert out["rows"] == 11 and out["not_decidable_left_out"] == 1
    assert set(out["entries"]) == {"4b", "rules"}
    assert out["entries"]["4b"]["accuracy"][0] == 1.0
    assert out["entries"]["4b"]["slices"]["enrollment_graduated"]["n"] == 3


def test_rows_every_entry_scored_and_a_short_result_is_named(tmp_path):
    src = _src(kinds=("swap_profile",))
    for r in src:
        r.pop("decidable")
        r["candidate_kind"] = r.pop("variant_kind")
    path = _write(tmp_path / "dev_cross.jsonl", src)
    full = _result(tmp_path / "base_dev_cross.json", "dev_cross.jsonl", src, pred=lambda r: "Skip")
    short = _result(tmp_path / "4b_dev_cross.json", "dev_cross.jsonl", src[:9])
    out = rt.build([("", full), ("", short)], 20, {"dev_cross": path})["dev_cross"]
    assert out["rows"] == 9 and out["notes"] == ["4b is missing 3 decidable rows"]
    assert out["entries"]["base"]["recall_by_bucket"]["Skip"] == 1.0
    assert list(out["entries"]["4b"]["by_kind"]) == ["swap_profile"]
    assert "Re-paired dev" in rt.markdown({"dev_cross": out})


def test_dev_uses_the_evaluators_visible_flag(tmp_path):
    src = _src()
    for r in src:
        r.pop("decidable")
    res = _result(tmp_path / "4b_dev.json", "dev.jsonl", src, visible=lambda r: r["pair_id"] != "p5")
    out = rt.build([("", res)], 20, {"dev": _write(tmp_path / "dev.jsonl", src)})["dev"]
    assert out["rows"] == 11 and out["not_decidable_left_out"] == 1


def test_paired_difference_is_a_minus_b_on_the_same_rows(tmp_path, monkeypatch):
    src = _src()
    monkeypatch.setattr(rt, "rules_entries", lambda rows: {"rules": {r["pair_id"]: "Skip" for r in rows}})
    res = _result(tmp_path / "4b_test.json", "test.jsonl", src)
    out = rt.build([("", res)], 50, {"test": _write(tmp_path / "test.jsonl", src)},
                   pairs=[("4b", "rules"), ("4b", "nobody")])["test"]
    assert list(out["paired"]) == ["4b vs rules"]
    acc = out["paired"]["4b vs rules"]["accuracy"]
    e = out["entries"]
    assert abs(acc[0] - (e["4b"]["accuracy"][0] - e["rules"]["accuracy"][0])) < 1e-3
    assert acc[1] <= acc[0] <= acc[2]
    assert "Paired differences" in rt.markdown({"test": out})
