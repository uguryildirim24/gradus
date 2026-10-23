"""The demo's decision path on fresh tests 1-4, offline: run B's stored label scores stand in for the model.

The shipped system is run B, read with the decider, behind H'' without work mode (eval/selection_cycle3.json pick;
eval/fresh_test4_result.json scores it once on sealed fresh test 4). For every row of each fresh test, the row goes
through demo/server.py's own scoring path, one request per row: the row's profile as the page's form, its posting as
a pool row carrying Adaption's extracted card, and a scorer that answers with B's stored float32 label scores for that
row (data/models/c3_eval/B/B_fresh_test*.json, data/models/c3_eval/test4/B/B_fresh_test4.json) and a label word the
page must not read. The page's final label must equal the committed per-row label of B+H''_without_work_mode|decider
(data/models/c3_eval/selection_rows.jsonl for tests 1-3, data/models/c3_eval/test4/test4_rows.jsonl for test 4) on
every row, the demo's s must equal the stored s, and the prompt the demo sends (or would send, for a row H'' decides
before the model) must equal, byte for byte, the one B was scored on (data/built/fit/eval_inputs/B/). The posting
text comes from B's prompt itself, so this checks the rubric, resume, profile and split; that the live pool's text
(demo/build_pool.py, textclean.real_posting_text) is the posting as the labelers saw it is build_fresh_test's
pipeline, not checked here. No model runs.

Fresh test 4 has 4 rows whose posting has no extracted card (eval/fresh_test4_eval_inputs.json no_card_rows_policy: no
rule verdict, so H'' overrides nothing). The demo's pool holds only postings with a card (demo/build_pool.py), so those
rows do not go through the page: for them the committed label must be the decider's label alone.

Reads data/ from this checkout, or else from $MATCHER_DATA_DIR;
each test skips when its files are absent.

    uv run --group demo pytest -q -s tests/test_demo_b_fresh.py     # -s prints the counts
"""
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(ROOT / "scripts"))

import demo_lib as lib  # noqa: E402
import server as srv  # noqa: E402

from matcher.decider import T_HI, T_LO, decide, keep_score  # noqa: E402
from matcher.prompt_b import SYSTEM_B, SYSTEM_B_SHA256  # noqa: E402

PICK = "B+H''_without_work_mode|decider"
SETS = {
    "fresh_test": {"dir": "built/fresh_test", "scores": "models/c3_eval/B/B_fresh_test.json",
                   "ref": "models/c3_eval/selection_rows.jsonl", "rows": 592},
    "fresh_test2": {"dir": "built/fresh_test2", "scores": "models/c3_eval/B/B_fresh_test2.json",
                    "ref": "models/c3_eval/selection_rows.jsonl", "rows": 371},
    "fresh_test3": {"dir": "built/fresh_test3", "scores": "models/c3_eval/B/B_fresh_test3.json",
                    "ref": "models/c3_eval/selection_rows.jsonl", "rows": 437},
    "fresh_test4": {"dir": "built/fresh_test4", "scores": "models/c3_eval/test4/B/B_fresh_test4.json",
                    "ref": "models/c3_eval/test4/test4_rows.jsonl", "rows": 1139, "no_card": 4},
}
SET_FILES = ("rows.jsonl", "cards.jsonl")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def data_dir(name: str) -> Path | None:
    s = SETS[name]
    places = [ROOT / "data"]
    if os.environ.get("MATCHER_DATA_DIR"):
        places.append(Path(os.environ["MATCHER_DATA_DIR"]))
    need = [s["scores"], s["ref"], f"built/fit/eval_inputs/B/{name}.jsonl"] + [f"{s['dir']}/{f}" for f in SET_FILES]
    return next((d for d in places if all((d / f).exists() for f in need)), None)


def form_of(c: dict) -> dict:
    """The profile form that says what this candidate card says."""
    willing = list(c["willing_locations"])
    reloc = {"relocate": "anywhere_us"} if willing == [lib.ANYWHERE] else {"relocate": "nowhere"} if not willing \
        else {"relocate": "list", "willing_locations": willing}
    blank = lambda v: "" if v is None else str(v)  # noqa: E731  (the page sends numbers as text, empty when unset)
    return {"work_auth": c["work_auth"], "current_location": c["current_location"], **reloc,
            "remote_preference": c["remote_preference"], "salary_floor": blank(c["salary_floor"]),
            "employment_types": list(c["employment_types_wanted"]), "industries_wanted": list(c["industries_wanted"]),
            "industries_avoided": list(c["industries_avoided"]), "roles_avoided": list(c["roles_avoided"]),
            "degree_level": c["degree_level"], "degree_field": c["degree_field"],
            "graduation_year": blank(c["graduation_year"]), "years": str(c["years"])}


class StoredScorer(srv.Demo):
    """The demo, with the GPU call answered from B's stored per-row scores. The label word it sends is one the page
    has no bucket for: the page must read the label from the scores."""

    def __init__(self, pool: dict[str, dict]) -> None:
        super().__init__("stored", Path("/nonexistent"), Path("/nonexistent"), Path("/nonexistent"))
        self.by_id, self.pool, self.scores, self.sent = pool, list(pool.values()), None, []

    def _call(self, path: str, body: dict | None = None, timeout: float = 0) -> dict:
        assert path == "/score"
        self.sent += [body["prefix"] + it["suffix"] for it in body["items"]]
        res = [{"id": it["id"], "label": "not-a-label", "scores": dict(zip(lib._LABEL_ORDER, self.scores)),
                "p": {"Keep": 0.0, "Explore": 0.0, "Skip": 0.0}, "reason": "stored"} for it in body["items"]]
        return {"backend": "stored", "model": "B (stored eval scores)", "results": res, "stats": {}}


def test_rubric_is_the_one_b_was_trained_and_scored_on():
    import build_fit_train3 as b3
    assert SYSTEM_B == b3.system_b()
    assert hashlib.sha256(SYSTEM_B.encode()).hexdigest() == SYSTEM_B_SHA256
    pins = json.loads((ROOT / "eval/cycle3_eval_inputs.json").read_text())
    assert pins["system_b_sha256"] == SYSTEM_B_SHA256
    assert json.loads((ROOT / "eval/cycle3_train_csvs.json").read_text())["prompt_b_system_sha256"] == SYSTEM_B_SHA256


def demo_vs_committed(name: str, data: Path) -> dict:
    s = SETS[name]
    d = data / s["dir"]
    rows = read_jsonl(d / "rows.jsonl")
    ids = [r["row_id"] for r in rows]
    scores = {r["pair_id"]: r["scores"] for r in json.loads((data / s["scores"]).read_text())["rows"]}
    ref = {r["row_id"]: r for r in read_jsonl(data / s["ref"]) if r.get("test", name) == name}
    prompts = {r["row_id"]: r["prompt"] for r in read_jsonl(data / f"built/fit/eval_inputs/B/{name}.jsonl")}
    assert len(ids) == s["rows"] and set(ids) == set(ref) == set(scores) == set(prompts)
    assert all(ref[i]["scores"]["B"] == scores[i] for i in ids)  # the committed rows hold the same scores

    cards = {c["id"]: c["card"] for c in read_jsonl(d / "cards.jsonl")}
    pool = {}
    for pid, card in cards.items():
        pool[pid] = {"id": pid, "company": "", "title": card.get("title", ""), "url": "",
                     "locations": card.get("locations", []), "card": card, "text": ""}
    demo = StoredScorer(pool)

    got, s_off, near, before_model, prompt_same, no_card, off = {}, 0.0, (1.0, ""), 0, 0, [], []
    for r in rows:
        rid, pid = r["row_id"], r["posting_id"]
        want = ref[rid]["systems"][PICK]
        sb = keep_score(scores[rid])
        s_off = max(s_off, abs(sb - ref[rid]["s"]["B"]))
        near = min(near, (min(abs(sb - T_LO), abs(sb - T_HI)), rid))
        if pid not in cards:  # no extracted card: nothing overrides (the pinned policy); the page never sees these
            no_card.append(rid)
            assert want == decide(scores[rid]), rid
            continue
        # the posting as B's eval prompt carries it (the labelers' posting text, markdown stripped)
        pool[pid]["text"] = lib.split_row(prompts[rid])[1].removesuffix("\n\nAnswer:")
        demo.scores, demo.sent = scores[rid], []
        req = {"resume": r["resume_text"], "profile": form_of(r["profile"]), "posting_ids": [pid], "redact": False,
               "live": 20, "reasons": False}
        out = demo.score(req)
        [card] = [c for b in out["buckets"].values() for c in b]
        word = card["label"]
        assert card["id"] == pid and card["bucket"] == lib.BUCKET_OF[word]
        if not demo.sent:  # decided before the model: H'' gives Skip whatever the model says
            before_model += 1
            assert card["source"] == "rules" and card["rule"]["kind"] == "skip" and word == "Skip"
            p = demo.prepare(req)  # the prompt the page would have sent: it too must be B's
            demo.sent = [p["prefix"] + lib.split_prompt(p["resume"], p["profile"], pool[pid]["text"])[1]]
        assert demo.sent == [prompts[rid]], f"{rid}: the demo's prompt is not the one B was scored on"
        prompt_same += 1
        got[rid] = word
        if word != want:
            off.append(f"{rid} (demo {word}, committed {want}, decider {decide(scores[rid])}, "
                       f"rules {ref[rid]['verdict']})")
    agree = sum(got[i] == ref[i]["systems"][PICK] for i in got)
    print(f"\n{name}: demo B+H'' decider vs the committed per-row label {agree}/{len(got)} rows agree over "
          f"{len({r['posting_id'] for r in rows})} postings; {before_model} decided before the model; "
          f"{prompt_same}/{len(got)} demo prompts (sent, or built for the rows decided before the model) equal B's "
          f"eval prompt byte for byte; {len(no_card)} no-card rows "
          f"(committed label = decider alone on all); max |s demo - s stored| {s_off:.1e}; closest s to a cut-off "
          f"{near[0]:.2e} ({near[1]}); labels {dict(Counter(got.values()))}")
    for x in off:
        print("  differs:", x)
    assert len(no_card) == s.get("no_card", 0)
    assert s_off < 1e-12
    assert agree == len(got), off
    assert prompt_same == len(got)
    return {"rows": len(ids), "through_page": len(got), "agree": agree, "before_model": before_model,
            "prompts_equal": prompt_same, "no_card": len(no_card)}


@pytest.mark.parametrize("name", list(SETS))
def test_demo_b_decider_h_double_prime_equals_the_committed_labels(name):
    data = data_dir(name)
    if data is None:
        pytest.skip(f"{name} rows, cards, B's prompts and stored scores, or the committed per-row file not here")
    demo_vs_committed(name, data)
