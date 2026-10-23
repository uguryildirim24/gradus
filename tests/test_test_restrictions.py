import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import clean_real_checked as ccr  # noqa: E402
import test_restrictions as tr  # noqa: E402


def test_same_rule_as_dev_v2():
    assert tr.REMOVE_KINDS == ccr.REMOVE_KINDS and tr.CHECK_KINDS == ccr.CHECK_KINDS


def test_table_is_consistent():
    assert not set(tr.TEST_RESTRICTIONS) & tr.TEST_READ_CLEAR
    assert set(tr.TEST_NOTES) <= set(tr.TEST_RESTRICTIONS) | tr.TEST_READ_CLEAR
    for i, (kind, source, quote, limit) in tr.TEST_RESTRICTIONS.items():
        assert kind in ccr.REMOVE_KINDS + ccr.CHECK_KINDS and source == "read" and quote, i
        assert (limit is not None) == (kind in ccr.CHECK_KINDS), i


def test_every_quote_is_in_its_posting():
    import assemble_real_train as art
    try:
        raw = art.brt_raw()
    except (FileNotFoundError, OSError):
        pytest.skip("raw postings not here")
    if not raw:
        pytest.skip("raw postings not here")
    assert tr.quote_problems(raw) == []
    assert tr.quote_problems(raw, shown=True) == []  # the model sees every quote too


def test_every_test_posting_was_read():
    if not tr.TEST_FILE.exists() or not tr.BUILDER_ROWS.exists():
        pytest.skip("real test files not here")
    rows = tr.read_jsonl(tr.TEST_FILE)
    ids = {r["rq_id"] for r in rows}
    assert ids == set(tr.TEST_RESTRICTIONS) | tr.TEST_READ_CLEAR
    dec = tr.test_decisions(rows)  # stops on an unread posting
    assert len(dec) == len(ids)
