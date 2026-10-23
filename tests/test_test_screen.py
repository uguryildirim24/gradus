import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import test_screen as ts  # noqa: E402


def _rows():
    out = []
    for p in range(4):
        for i, b in enumerate(("For you", "Explore", "Excluded", "Excluded")):
            out.append({"pair_id": f"p{p}r{i}", "rq_id": f"q{p}", "base_rq_id": f"q{p}", "bucket": b,
                        "variant_kind": "full_match"})
    return out


def test_score_and_paired_on_the_same_rows():
    rows = _rows()
    gold = {r["pair_id"]: ts.WORD[r["bucket"]] for r in rows}
    preds = {"right": dict(gold), "skip": {k: "Skip" for k in gold}}
    res = ts.score(rows, preds, [("right", "skip")], draws=50)
    assert res["rows"] == 16 and res["postings"] == 4
    assert res["entries"]["right"]["accuracy"][0] == 1.0 and res["entries"]["skip"]["accuracy"][0] == 0.5
    d = res["paired"]["right - skip"]
    assert d["accuracy_diff"][0] == 0.5 and d["rows_fixed"] == 8 and d["rows_broken"] == 0


def test_markdown_lists_flagged_postings():
    rows = _rows()
    gold = {r["pair_id"]: ts.WORD[r["bucket"]] for r in rows}
    s = ts.score(rows, {"m": gold}, [], draws=20)
    res = {"with": s, "without": s, "notes": [], "draws": 20, "postings_read": 4, "postings_flagged": 1,
           "postings_removed": 1, "removed_by_kind": {"grad_window": 1},
           "rows_removed": {"all_rows": 4, "all_rows_by_label": {"Keep": 1}, "decidable_scored": 4,
                            "decidable_scored_by_label": {"Keep": 1}},
           "flagged": [{"posting_id": "q0", "company": "Co", "title": "Intern", "kind": "grad_window",
                        "quote": "graduating in 2028 | 2029", "limit": None, "remove": True,
                        "rows": {"Keep": 1}, "decidable_rows": {"Keep": 1}}]}
    text = ts.markdown(res)
    assert "| q0 | Co, Intern | grad_window | \"graduating in 2028 / 2029\" | yes | 1 (1) | 0 (0) | 0 (0) |" in text
