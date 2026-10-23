"""fit_eval_inputs' fresh test 3 path on a synthetic fresh test dir (no data/ file is read)."""
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fit_eval_inputs as fi  # noqa: E402
from matcher import RequirementCard, full_match  # noqa: E402
from matcher.pairs import build_prompt  # noqa: E402
from matcher.profile import render_profile  # noqa: E402

CARD = RequirementCard(
    source_posting_id="p1", title="Lab Technician", field="biology", industry="biotech",
    degree_level="bachelor", degree_fields=("biochemistry", "biology"), min_years=0,
    must_skills=(("pcr",), ("cell culture",)), nice_skills=(), license=None, locations=("Boston, MA",),
    work_mode="onsite", sponsorship="will_not_sponsor", salary_min=50_000, salary_max=70_000,
    employment_type="full_time")
BODY = ("We are hiring a lab technician to run PCR and cell culture in our Boston lab. " * 12).strip()


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def synth(tmp: Path) -> dict:
    """data/built/fresh_test3 (rows, postings, gold with the real files' key names) and its two manifests."""
    d = tmp / "data/built/fresh_test3"
    d.mkdir(parents=True)
    base = full_match(CARD)
    profiles = [base, replace(base, skills=("pcr",)), replace(base, salary_floor=None, employment_types_wanted=())]
    posts, rows, gold = [], [], []
    for p in range(3):
        pid = f"greenhouse:synth:{p}"
        posts.append({"key": pid, "title": f"Lab Technician {p}", "employer": f"Synth {p}",
                      "locations": "Boston, MA; Cambridge, MA" if p == 1 else "Boston, MA", "text": BODY})
        for j, c in enumerate(profiles):
            rid = f"f3_{p}{j}"
            rows.append({"row_id": rid, "posting_id": pid, "title": f"Lab Technician {p}", "employer": f"Synth {p}",
                         "field": "lab_bio", "posting_text": f"Title: Lab Technician {p}\n{BODY}",
                         "profile": c.to_dict(), "profile_text": "", "resume_facts": {},
                         "resume_text": f"Jane Doe {j}\nBoston, MA\nLab work", "design": {}})
            gold.append({"row_id": rid, "gold": ("Keep", "Explore", "Skip")[j], "label_A": "Keep", "label_B": "Keep",
                         "adjudicator_side": None, "source": "agreed"})
    for name, xs in (("rows", rows), ("postings", posts), ("gold", gold)):
        (d / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in xs))
    ev = tmp / "eval"
    ev.mkdir()
    man = ev / "fresh_test3_manifest.json"
    man.write_text(json.dumps({"files": {f"data/built/fresh_test3/{n}.jsonl": sha(d / f"{n}.jsonl")
                                         for n in ("rows", "postings")}}))
    gm = ev / "fresh_test3_gold_manifest.json"
    gm.write_text(json.dumps({"gold_sha256": sha(d / "gold.jsonl")}))
    return {"dir": d, "rows": rows, "posts": {p["key"]: p for p in posts}, "man": man, "gm": gm}


def test_fresh_rows_on_test3_are_the_shared_builders(tmp_path):
    S = synth(tmp_path)
    for m in fi.MODELS:
        out = fi.fresh_rows(m, S["dir"])
        assert [r["row_id"] for r in out] == [r["row_id"] for r in S["rows"]]
        for r, src in zip(out, S["rows"]):
            want = build_prompt(src["resume_text"], render_profile(fi.CandidateCard.from_dict(src["profile"])),
                                fi.POSTING[m](S["posts"][src["posting_id"]]))
            assert r["prompt"] == want and r["pair_id"] == r["row_id"] and r["base_rq_id"] == r["posting_id"]
            assert r["bucket"] == fi.BUCKET[r["gold"]]
    h, base = fi.fresh_rows("h", S["dir"]), fi.fresh_rows("base", S["dir"])
    assert [r["prompt"] for r in h] == [r["prompt"] for r in base]  # base takes h's format


def point(monkeypatch, tmp: Path, S: dict) -> Path:
    out = tmp / "data/built/fit/eval_inputs"
    for k, v in {"ROOT": tmp, "OUT": out, "FRESH3": S["dir"], "FRESH3_MANIFEST": S["man"],
                 "FRESH3_GOLD_MANIFEST": S["gm"]}.items():
        monkeypatch.setattr(fi, k, v)
    monkeypatch.setattr(fi, "parity", lambda: {"stub": True})
    monkeypatch.setattr(fi, "h2_prompts_are_train_fit_rows", lambda: {})
    monkeypatch.setattr(fi, "csv_prompts", lambda p: [])
    return out


def test_build3_writes_once_and_leaves_tests_1_and_2_alone(tmp_path, monkeypatch):
    S = synth(tmp_path)
    out = point(monkeypatch, tmp_path, S)
    keep = {}
    for m in fi.MODELS:
        (out / m).mkdir(parents=True)
        for s in ("fresh_test", "fresh_test2"):
            (out / m / f"{s}.jsonl").write_text(f"{m} {s} sentinel\n")
            keep[out / m / f"{s}.jsonl"] = (out / m / f"{s}.jsonl").read_bytes()
    fi.build3(fi.argparse.Namespace())
    rep = json.loads((out / "report_fresh_test3.json").read_text())
    assert rep["format"]["rows"] == 9 and rep["inputs_sha256"]["data/built/fresh_test3/gold.jsonl"] == sha(S["dir"] / "gold.jsonl")
    for m in fi.MODELS:
        rows = [json.loads(l) for l in (out / m / "fresh_test3.jsonl").open()]
        assert [r["prompt"] for r in rows] == [r["prompt"] for r in fi.fresh_rows(m, S["dir"])]
        assert rep["files"][f"{m}/fresh_test3.jsonl"] == sha(out / m / "fresh_test3.jsonl")
    assert all(p.read_bytes() == b for p, b in keep.items())
    with pytest.raises(SystemExit, match="built once"):
        fi.build3(fi.argparse.Namespace())
    (out / "report_fresh_test3.json").unlink()  # any one target blocks a rebuild
    with pytest.raises(SystemExit, match="built once"):
        fi.build3(fi.argparse.Namespace())


def test_build3_refuses_unsealed_inputs(tmp_path, monkeypatch):
    S = synth(tmp_path)
    out = point(monkeypatch, tmp_path, S)
    for name, match in (("rows.jsonl", "fresh_test3_manifest.json sealed"), ("postings.jsonl", "manifest.json sealed"),
                        ("gold.jsonl", "gold_manifest.json pins")):
        p = S["dir"] / name
        old = p.read_bytes()
        p.write_bytes(old + b"\n")
        with pytest.raises(SystemExit, match=match):
            fi.build3(fi.argparse.Namespace())
        p.write_bytes(old)
    assert not out.exists()  # nothing written before the pins hold
    fi.build3(fi.argparse.Namespace())


# ---------------------------------------------------------------- cycle 3: run B's format (build4)

C3 = Path(__file__).resolve().parents[1] / "data/built/fit/cycle3"


def test_prompt_b_row_is_train_B_byte_for_byte_on_a_training_row():
    """A kept new row of train_B.csv, rebuilt from cycle3/rows.jsonl's raw parts by the builder the eval inputs use."""
    if not (C3 / "train_B.csv").exists():
        pytest.skip("data/built/fit/cycle3 is only in the integrate worktree")
    import build_fit_train3 as b3
    side = [x for x in fi.jsonl(C3 / "train3_rows.jsonl") if x.get("kept") and x["source"] == "new"]
    want = {x["row_id"]: x["csv_row"] for x in side[:5]}
    csv_b = fi.csv_prompts(C3 / "train_B.csv")
    system = b3.system_b()
    seen = 0
    for r in fi.jsonl(C3 / "rows.jsonl"):
        if r["row_id"] in want:
            assert fi.prompt_b_row(r, system).encode() == csv_b[want[r["row_id"]]].encode()
            seen += 1
    assert seen == len(want) == 5


def test_prompt_b_row_is_the_rubric_then_hs_layout_with_the_whole_posting():
    import build_fit_train3 as b3
    system = b3.system_b()
    body = "\n".join(f"Requirement line {i}: you will run assays in the lab." for i in range(300))
    r = {"resume_text": "Jane Doe\nBoston, MA\nLab work", "profile": full_match(CARD).to_dict(),
         "posting_text": f"Title: Lab Technician\nCompany: Synth\n\n{body}"}
    p = fi.prompt_b_row(r, system)
    prof = render_profile(fi.CandidateCard.from_dict(r["profile"]))
    h = build_prompt(r["resume_text"], prof, r["posting_text"])
    whole = build_prompt(r["resume_text"], prof, r["posting_text"], max_posting_tokens=10 ** 9)
    from matcher.pairs import INSTRUCTION
    assert p == system + "\n\n" + whole[len(INSTRUCTION) + 1:]
    assert p.startswith(system + "\n\nRESUME:\n") and p.endswith("\n\nAnswer:") and INSTRUCTION not in p
    assert len(r["posting_text"]) > 6000 and body.splitlines()[-1] in p and body.splitlines()[-1] not in h


def test_build4_writes_once(tmp_path, monkeypatch):
    sets = {}
    for s in ("fresh_test", "fresh_test2", "fresh_test3"):
        S = synth(tmp_path / s)
        sets[s] = S["dir"]
    out = tmp_path / "data/built/fit/eval_inputs"
    for k, v in {"ROOT": tmp_path, "OUT": out, "B_DIR": out / "B", "B_REPORT": out / "report_cycle3_B.json",
                 "C3_PINS": tmp_path / "eval/cycle3_eval_inputs.json", "B_SETS": sets}.items():
        monkeypatch.setattr(fi, k, v)
    monkeypatch.setattr(fi, "check_b_inputs", lambda: {"stub": True})
    monkeypatch.setattr(fi, "b_parity", lambda system: {"stub": True})
    monkeypatch.setattr(fi, "write_c3_pins", lambda rep: {})
    fi.build4(fi.argparse.Namespace())
    rep = json.loads((out / "report_cycle3_B.json").read_text())
    import build_fit_train3 as b3
    for s, d in sets.items():
        rows = [json.loads(l) for l in (out / "B" / f"{s}.jsonl").open()]
        assert [r["prompt"] for r in rows] == [fi.prompt_b_row(x, b3.system_b()) for x in fi.jsonl(d / "rows.jsonl")]
        assert rep["files"][f"B/{s}.jsonl"] == sha(out / "B" / f"{s}.jsonl") and rep["rows"][s] == 9
    with pytest.raises(SystemExit, match="built once"):
        fi.build4(fi.argparse.Namespace())
