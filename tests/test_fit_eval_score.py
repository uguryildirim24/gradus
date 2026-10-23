"""fit_eval_score on synthetic rows and synthetic score files (no data/ file is read)."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fit_eval_score as fs  # noqa: E402
from matcher import RequirementCard, full_match  # noqa: E402

CARD = RequirementCard(
    source_posting_id="p1", title="Lab Technician", field="biology", industry="biotech",
    degree_level="bachelor", degree_fields=("biochemistry", "biology"), min_years=0,
    must_skills=(("pcr",), ("cell culture",)), nice_skills=(), license=None, locations=("Boston, MA",),
    work_mode="onsite", sponsorship="will_not_sponsor", salary_min=50_000, salary_max=70_000,
    employment_type="full_time")


def rows_fixture(n_post: int = 12) -> list[dict]:
    """Per posting: a full match (gold Keep), a candidate missing a skill (gold Skip, eligibility passes), one who
    needs sponsorship (gold Skip, H' fires), one with a floor over the pay (gold Explore, salary cap)."""
    base = full_match(CARD)
    kinds = [("Keep", base), ("Skip", replace(base, skills=("pcr",))),
             ("Skip", replace(base, work_auth="needs_h1b_now")), ("Explore", replace(base, salary_floor=90_000))]
    rows = []
    for p in range(n_post):
        for j, (g, c) in enumerate(kinds):
            r = fs.base_row(f"r{p}_{j}", f"post{p}", g, c, CARD, "Lab Technician",
                            "Lab Technician at X" if j == 0 else "Assistant at Y")
            r |= {"label_A": g, "label_B": g if j else "Explore", "agreed": bool(j), "teaser": p % 3 == 0,
                  "empty_body": False, "blank_profile": j == 1, "lives_in_not_given": False, "kind": "lab_bio",
                  "pool_field": "lab_bio", "archetype": "strong"}
            rows.append(r)
    return rows


def write_scores(tmp: Path, name: str, labels: dict[str, str], tie: bool = False, drop: str | None = None,
                 bad_pred: str | None = None) -> Path:
    out = []
    for rid, lab in labels.items():
        if rid == drop:
            continue
        s = [0.0, 0.0, 0.0]
        s[fs.IDX[lab]] = 5.0
        if tie:
            s = [1.0, 1.0, 1.0]
        out.append({"pair_id": rid, "scores": s, "pred": bad_pred or (fs.LABELS[fs.pick(s)])})
    p = tmp / name
    p.write_text(json.dumps({"rows": out}))
    return p


def test_rows_fixture_rules_and_eligibility():
    rows = rows_fixture(1)
    assert [r["rules_full"] for r in rows] == ["Keep", "Skip", "Skip", "Explore"]
    # without skills, the missing-skill candidate is a Keep; the hard and salary rows do not change
    assert [r["rules_skills_free"] for r in rows] == ["Keep", "Keep", "Skip", "Explore"]
    assert [r["eligibility_pass"] for r in rows] == [True, True, False, False]
    assert [r["title_in_resume"] for r in rows] == [True, False, False, False]


def test_read_scores_rereads_labels_with_the_tie_order(tmp_path):
    ids = ["a", "b"]
    p = write_scores(tmp_path, "t.json", {"a": "Keep", "b": "Explore"}, tie=True)
    assert fs.read_scores(p, ids) == {"a": "Skip", "b": "Skip"}  # an exact tie goes to Skip
    s = [2.0, 2.0, 1.0]
    assert fs.LABELS[fs.pick(s)] == "Explore"  # then Explore, then Keep


def test_read_scores_refuses_missing_rows_and_wrong_preds(tmp_path):
    lab = {"a": "Keep", "b": "Skip"}
    with pytest.raises(SystemExit, match="missing"):
        fs.read_scores(write_scores(tmp_path, "m.json", lab, drop="b"), ["a", "b"])
    with pytest.raises(SystemExit, match="stored pred"):
        fs.read_scores(write_scores(tmp_path, "p.json", lab, bad_pred="Explore"), ["a", "b"])
    with pytest.raises(SystemExit, match="not in the set"):
        fs.read_scores(write_scores(tmp_path, "x.json", lab), ["a"])


def test_hprime_overrides_hard_skip_and_caps_salary_keep():
    rows = rows_fixture(1)
    keep_all = {r["row_id"]: "Keep" for r in rows}
    P = fs.systems(rows, {"m": keep_all})
    assert P["m raw"] == ["Keep"] * 4
    assert P["m+H'"] == ["Keep", "Keep", "Skip", "Explore"]
    assert P["title_match"] == ["Keep", "Explore", "Explore", "Explore"]


def test_two_way_draws_are_seeded_and_keep_posting_sizes():
    from collections import Counter
    groups = ["a"] * 3 + ["b"] * 5 + ["c"] * 2
    sizes = Counter(groups)
    d1, d2 = fs.two_way_draws(groups, 50, 7), fs.two_way_draws(groups, 50, 7)
    assert all((x == y).all() for x, y in zip(d1, d2))
    g = np.array(groups)
    for idx in d1:
        # every drawn posting brings as many rows as it has, all of them its own
        assert all(v % sizes[k] == 0 for k, v in Counter(g[idx]).items())
        assert set(g[idx]) <= set(sizes)
    # rows are drawn within postings: a draw can repeat a row of a posting while leaving out another
    assert any(len(set(idx.tolist())) < len(idx) for idx in d1)


def test_paired_identical_systems_is_zero():
    g = np.array([0, 1, 2, 0, 1, 2])
    dr = fs.two_way_draws(["p", "p", "q", "q", "r", "r"], 100, 0)
    d = fs.paired(g, g.copy(), g.copy(), dr)
    assert d["macro_f1_diff"] == [0.0, 0.0, 0.0] and d["accuracy_diff"] == [0.0, 0.0, 0.0]


def test_fresh_pass_rule_on_synthetic_scores(tmp_path):
    rows = rows_fixture(12)
    ids = [r["row_id"] for r in rows]
    perfect = {r["row_id"]: r["gold"] for r in rows}
    p = write_scores(tmp_path, "perfect.json", perfect)
    models = {"h": fs.read_scores(p, ids), "d3": {r["row_id"]: "Keep" for r in rows}}
    res = fs.score_fresh(rows, models, "h", draws=200)
    ps = res["pass"]
    # on the eligibility-pass slice gold is Keep, Skip per posting; skills-free rules say Keep, Keep
    assert ps["passed"] and ps["macro_f1_diff"][1] > 0
    assert res["slices"]["eligibility_pass"]["n"] == 24
    assert res["slices"]["all"]["systems"]["h+H'"]["accuracy"][0] == 1.0
    assert res["slices"]["agreed_only"]["n"] == 36
    assert res["slices"]["all"]["labeler_ceiling"]["raw_agreement"] == 0.75
    # a model no better than the baseline does not pass
    res2 = fs.score_fresh(rows, {"h": {r["row_id"]: r["rules_skills_free"] for r in rows}}, "h", draws=200)
    assert not res2["pass"]["passed"] and res2["pass"]["macro_f1_diff"] == [0.0, 0.0, 0.0]
    md = fs.markdown(res)
    assert "PASS" in md and "rules_skills_free" in md and "kind:lab_bio" in md


def test_dev_selection_tie_keeps_d3():
    rows = rows_fixture(6)
    for r in rows:
        r |= {"source": "venator", "stored_eligibility_pass": r["eligibility_pass"]}
    gold = {r["row_id"]: r["gold"] for r in rows}
    res = fs.score_dev(rows, {"h": gold, "d3": gold}, draws=50)
    assert res["selection"]["pick"] == "d3"
    worse = {k: "Keep" for k in gold}
    res = fs.score_dev(rows, {"h": gold, "d3": worse}, draws=50)
    assert res["selection"]["pick"] == "h"


def test_title_and_kind_definitions():
    assert fs.title_in_resume("Senior Data Scientist", "...worked as a senior  data scientist at...")
    assert fs.title_in_resume("QC Analyst – Micro", "QC Analyst - Micro, 2019")
    assert not fs.title_in_resume("", "anything")
    assert fs.kind_of("Senior Data Scientist", "") == "data"
    assert fs.kind_of("Research Associate Intern", "") == "internship"
    assert fs.kind_of("Territory Sales Representative - GI Physician Partnerships", "") == "sales_ops"
    assert fs.kind_of("Medical Assistant II", "") == "health_nursing"
    assert fs.kind_of("Open apply", "") == "other"
    assert fs.posting_title("Title: Lab Tech\nCompany: X") == "Lab Tech"


# ---------------------------------------------------------------- scored once (amendment 2)


def git_repo(tmp: Path) -> Path:
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp)], check=True)
    subprocess.run(["git", "-C", str(tmp), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp), "config", "user.name", "t"], check=True)
    return tmp


def git_commit(repo: Path, *files: Path) -> None:
    import subprocess
    subprocess.run(["git", "-C", str(repo), "add", *map(str, files)], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "x"], check=True)


def test_write_once_refuses_an_existing_result(tmp_path):
    rows = rows_fixture(4)
    res = fs.score_fresh(rows, {"h": {r["row_id"]: r["gold"] for r in rows}}, "h", draws=20)
    out = tmp_path / "r" / "fresh_test2_result.json"
    fs.write_once(out, res)
    assert json.loads(out.read_text())["chosen"] == "h+H'" and out.with_suffix(".md").exists()
    before = out.read_text()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        fs.write_once(out, res)
    assert out.read_text() == before
    out.unlink()  # the .md alone still blocks a second scoring
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        fs.write_once(out, res)


def test_read_pick_takes_only_a_committed_selection(tmp_path):
    repo = git_repo(tmp_path)
    sel = repo / "selection_v5.json"
    sel.write_text(json.dumps({"pick": "h2a", "amendment": "eval/protocol_amendment_2.json"}))
    with pytest.raises(SystemExit, match="not committed"):
        fs.read_pick(sel)
    git_commit(repo, sel)
    assert fs.read_pick(sel, need_amendment="eval/protocol_amendment_2.json") == "h2a"
    with pytest.raises(SystemExit, match="not a selection made under"):
        fs.read_pick(sel, need_amendment="eval/protocol_amendment_3.json")
    sel.write_text(json.dumps({"pick": "h", "amendment": "eval/protocol_amendment_2.json"}))  # edited after commit
    with pytest.raises(SystemExit, match="not committed"):
        fs.read_pick(sel)
    git_commit(repo, sel)
    old = repo / "selection_v4.json"
    old.write_text(json.dumps({"selection": {"pick": "h"}}))  # amendment-1 shape: no amendment key
    git_commit(repo, old)
    assert fs.read_pick(old) == "h"
    with pytest.raises(SystemExit, match="not a selection made under"):
        fs.read_pick(old, need_amendment="eval/protocol_amendment_2.json")


def test_cli_has_no_free_chosen_argument(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fit_eval_score.py", "fresh", "--score", "h=x.json", "--chosen", "h"])
    with pytest.raises(SystemExit):
        fs.main()


def test_fresh2_result_already_there_stops_before_anything(tmp_path, monkeypatch):
    res = tmp_path / "fresh_test2_result.json"
    res.write_text("{}")
    monkeypatch.setattr(fs, "FRESH2_RESULT", res)
    monkeypatch.setattr(fs, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="scored once"):
        fs.cmd_fresh2(fs.argparse.Namespace(selection=str(tmp_path / "none.json"), score=[]))


def fresh2_rows(n_post: int = 12) -> list[dict]:
    """rows_fixture with the missing-skill candidate's gold moved to Explore, so rules_full (Skip) is wrong there."""
    rows = rows_fixture(n_post)
    for r in rows:
        if r["row_id"].endswith("_1"):
            r["gold"] = "Explore"
    return rows


def test_fresh2_pass_needs_both_baselines_beaten():
    rows = fresh2_rows()
    gold = {r["row_id"]: r["gold"] for r in rows}
    keep = {k: "Keep" for k in gold}
    models = {"h2a": gold, "base": keep, "h": keep, "d3": keep}
    res = fs.score_fresh2(rows, models, "h2a", draws=200)
    p = res["pass"]
    assert p["passed"] and set(p["against"]) == {"rules_full", "base+H'"}
    assert all(v["lower_bound_exact"] > 0 for v in p["against"].values())
    assert set(res["reported_against"]) == {"h+H'", "d3+H'", "rules_skills_free", "rules_full", "base+H'"}
    # the untrained base as good as the pick: rules_full is beaten, base+H' is not, so no pass
    res2 = fs.score_fresh2(rows, {**models, "base": gold}, "h2a", draws=200)
    assert res2["pass"]["against"]["rules_full"]["beaten"] and not res2["pass"]["against"]["base+H'"]["beaten"]
    assert not res2["pass"]["passed"]
    with pytest.raises(SystemExit, match="needs score files"):
        fs.score_fresh2(rows, {"h2a": gold, "base": keep}, "h2a", draws=20)
    md = fs.markdown(res)
    assert "amendment 2" in md and "PASS" in md


def test_select_on_fresh1_keeps_h_within_the_margin():
    rows = fresh2_rows(6)
    gold = {r["row_id"]: r["gold"] for r in rows}
    keep = {k: "Keep" for k in gold}
    res = fs.select_fresh1(rows, {"h": gold, "h2a": gold, "d3": keep, "base": keep}, draws=20)
    assert res["pick"] == "h" and res["candidates"] == ["h", "h2a"] and res["ignored_tags"] == ["base", "d3"]
    res = fs.select_fresh1(rows, {"h": keep, "h2a": keep, "h2b": gold}, draws=20)
    assert res["pick"] == "h2b"
    with pytest.raises(SystemExit, match="incumbent"):
        fs.select_fresh1(rows, {"h2a": gold}, draws=20)


def test_design_flag_may_move_only_by_a_work_mode_hard_skip():
    v = lambda cats: {"rule": "Skip", "hard_cats": cats, "salary_soft": False}  # noqa: E731
    fs.check_design_flag("a", {"eligibility_pass": False, "verdict": v(["work_mode"])}, {"eligibility_pass": True})
    fs.check_design_flag("b", {"eligibility_pass": True, "verdict": v([])}, {"eligibility_pass": True})
    with pytest.raises(SystemExit, match="differs from the sealed design flag"):
        fs.check_design_flag("c", {"eligibility_pass": False, "verdict": v(["location"])}, {"eligibility_pass": True})
    with pytest.raises(SystemExit, match="differs from the sealed design flag"):
        fs.check_design_flag("d", {"eligibility_pass": True, "verdict": v([])}, {"eligibility_pass": False})
    with pytest.raises(SystemExit, match="differs from the sealed design flag"):
        fs.check_design_flag("e", {"eligibility_pass": False, "verdict": {"rule": "Explore", "hard_cats": [],
                                                                           "salary_soft": True}},
                             {"eligibility_pass": True})


# ---------------------------------------------------------------- fresh test 3 (amendment 3), synthetic only


def fresh3_rows(n_post: int = 12) -> list[dict]:
    """fresh2_rows with test 3's design flags: half the postings in dev_fit, two lengths, two pool fields, the sealed
    design flag equal to the recomputed one except one moved-out row per posting set below by the test."""
    rows = fresh2_rows(n_post)
    for r in rows:
        p = int(r["posting_id"][4:])
        r |= {"eligibility_pass_design": r["eligibility_pass"], "posting_in_dev_fit": p % 2 == 0,
              "posting_in_dev_fit_all": p % 2 == 0 or p == 1, "posting_length": "3000+" if p % 3 else "teaser 1-699",
              "pool_field": "lab_bio" if p < n_post // 2 else "data"}
    return rows


def test_fresh3_pass_needs_both_baselines_and_reports_every_amendment3_slice():
    rows = fresh3_rows()
    gold = {r["row_id"]: r["gold"] for r in rows}
    keep = {k: "Keep" for k in gold}
    models = {"h": gold, "base": keep, "d3": keep}
    res = fs.score_fresh3(rows, models, "h", draws=200)
    p = res["pass"]
    assert res["set"] == "fresh_test3" and res["amendment"] == "eval/protocol_amendment_3.json"
    assert p["passed"] and set(p["against"]) == {"rules_full", "base+H'"} and p["slice"] == "eligibility_pass"
    assert all(v["lower_bound_exact"] > 0 for v in p["against"].values())
    sens = res["sealed_design_slice (sensitivity, not for pass)"]
    assert set(sens["against"]) == {"rules_full", "base+H'"} and all(v["would_be_beaten"] for v in sens["against"].values())
    S = res["slices"]
    for name in ("all", "eligibility_pass", "eligibility_pass_design", "posting_in_dev_fit", "posting_not_in_dev_fit",
                 "posting_in_dev_fit_all", "posting_not_in_dev_fit_all", "teaser", "field:lab_bio", "field:data",
                 "length:3000+", "length:teaser 1-699", "kind:lab_bio"):
        assert S[name] is not None, name
    assert S["posting_in_dev_fit"]["n"] + S["posting_not_in_dev_fit"]["n"] == len(rows)
    assert S["posting_in_dev_fit"]["postings"] == 6
    # every system the report needs is scored on both eligibility slices and all rows
    for name in ("all", "eligibility_pass", "eligibility_pass_design"):
        assert {"h+H'", "d3+H'", "base+H'", "rules_full", "rules_skills_free"} <= set(S[name]["systems"])
        assert S[name]["labeler_ceiling"]["kappa"] is not None or S[name]["labeler_ceiling"]["raw_agreement"] == 1.0
    assert set(res["reported_against"]) == {"d3+H'", "rules_skills_free", "rules_full", "base+H'"}
    assert set(res["reported_against"]["d3+H'"]) == {"all", "eligibility_pass", "eligibility_pass_design"}
    # base as good as the pick: rules_full beaten, base+H' not, no pass
    res2 = fs.score_fresh3(rows, {**models, "base": gold}, "h", draws=200)
    assert res2["pass"]["against"]["rules_full"]["beaten"] and not res2["pass"]["against"]["base+H'"]["beaten"]
    assert not res2["pass"]["passed"]
    with pytest.raises(SystemExit, match="needs score files"):
        fs.score_fresh3(rows, {"h": gold, "base": keep}, "h", draws=20)
    with pytest.raises(SystemExit, match="needs score files"):
        fs.score_fresh3(rows, {"h3a": gold, "base": keep, "d3": keep}, "h3a", draws=20)  # h is reported too


def test_fresh3_sealed_design_slice_is_scored_on_its_own_rows():
    rows = fresh3_rows()
    for r in rows:
        if r["row_id"].endswith("_1"):
            r["eligibility_pass_design"] = False  # moved in: fails the sealed design, passes now
        if r["row_id"].endswith("_2"):
            r["eligibility_pass_design"] = True  # moved out: passed the sealed design, a hard Skip now
    gold = {r["row_id"]: r["gold"] for r in rows}
    keep = {k: "Keep" for k in gold}
    res = fs.score_fresh3(rows, {"h": gold, "base": keep, "d3": keep}, "h", draws=50)
    S = res["slices"]
    assert S["eligibility_pass"]["n"] == 24 and S["eligibility_pass_design"]["n"] == 24
    assert S["eligibility_pass_design"]["gold"] == {"Keep": 12, "Skip": 12}
    ev = res["eligibility_vs_sealed_design"]
    assert ev["rows_moved_in"] == 12 and ev["rows_moved_out"] == 12
    assert ev["moved_out_by"] == {"work_auth": 12} and ev["gold_of_moved_in_rows"] == {"Explore": 12}


def test_fresh3_markdown_uses_the_two_baseline_table():
    rows = fresh3_rows(6)
    gold = {r["row_id"]: r["gold"] for r in rows}
    keep = {k: "Keep" for k in gold}
    md = fs.markdown(fs.score_fresh3(rows, {"h": gold, "base": keep, "d3": keep}, "h", draws=20))
    assert "Pass rule (amendment 3)" in md and "## eligibility_pass_design" in md
    assert "Sealed-design eligibility-pass slice" in md and "| posting_not_in_dev_fit |" in md
    assert "h+H' - base+H'" in md and "rules_skills_free macro-F1" not in md
    # fresh test 2's markdown is unchanged by the fresh-3 branch
    md2 = fs.markdown(fs.score_fresh2(fresh2_rows(6), {"h": gold, "base": keep, "d3": keep}, "h", draws=20))
    assert "Pass rule (amendment 2)" in md2 and "eligibility_pass_design" not in md2 and "Sealed-design" not in md2


# synthetic fresh test 3 on disk: rows, postings, cards, gold; a tmp git repo stands in for the project

LONG_BODY = ("We are hiring a lab technician to run PCR and cell culture in our Boston lab. " * 20).strip()


def synth_fresh_dir(d: Path, n_post: int = 6, moved: bool = False) -> None:
    """rows/postings/cards/gold.jsonl as build_fresh_test writes them (the key names of the real files)."""
    base = full_match(CARD)
    kinds = [("Keep", base), ("Explore", replace(base, skills=("pcr",))),
             ("Skip", replace(base, work_auth="needs_h1b_now")), ("Explore", replace(base, salary_floor=90_000))]
    design_pass = [True, not moved, moved, False]  # moved: row 1 moves in, row 2 moves out
    d.mkdir(parents=True, exist_ok=True)
    rows, posts, cards, gold = [], [], [], []
    for p in range(n_post):
        pid = f"greenhouse:synth:{p}"
        posts.append({"key": pid, "title": "Lab Technician", "employer": f"Synth {p}", "locations": "Boston, MA",
                      "text": LONG_BODY})
        cards.append({"id": pid, "card": replace(CARD, source_posting_id=pid).to_dict()})
        for j, (g, c) in enumerate(kinds):
            rid = f"f3_{p:02d}{j}"
            rows.append({"row_id": rid, "posting_id": pid, "title": "Lab Technician", "employer": f"Synth {p}",
                         "field": "lab_bio" if p % 2 else "data", "posting_text": LONG_BODY,
                         "profile": c.to_dict(), "profile_text": "", "resume_facts": {},
                         "resume_text": "Jane Doe\nBoston, MA\nLab Technician at X" if j == 0 else "Jane Doe\nBoston, MA\nAssistant",
                         "design": {"eligibility_pass": design_pass[j], "teaser": False, "empty_body": False,
                                    "archetype": "strong", "posting_in_dev_fit": p % 2 == 0,
                                    "posting_in_dev_fit_all": p % 2 == 0, "posting_length": "3000+"}})
            gold.append({"row_id": rid, "gold": g, "label_A": g, "label_B": g if j else "Explore",
                         "adjudicator_side": None, "source": "agreed"})
    for name, xs in (("rows", rows), ("postings", posts), ("cards", cards), ("gold", gold)):
        (d / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in xs))
    (d / "label_batches").mkdir(exist_ok=True)
    (d / "label_batches" / "batch_01.jsonl").write_text(json.dumps({"row_id": "f3_000"}) + "\n")
    (d / "labels" / "A").mkdir(parents=True, exist_ok=True)
    (d / "labels" / "A" / "batch_01.jsonl").write_text(json.dumps({"row_id": "f3_000", "label": "Keep"}) + "\n")


def test_load_fresh3_reports_moves_either_way_without_raising(tmp_path):
    d = tmp_path / "fresh_test3"
    synth_fresh_dir(d, moved=True)
    with pytest.raises(SystemExit, match="differs from the sealed design flag"):
        fs.load_fresh(d)  # tests 1 and 2's strict check would stop here
    rows = fs.load_fresh(d, strict_design=False)
    assert len(rows) == 24 and all(r["posting_length"] == "3000+" for r in rows)
    assert sum(r["posting_in_dev_fit"] for r in rows) == 12
    s = fs.design_flag_summary3(rows)
    assert s["recomputed_pass_rows"] == 12 and s["sealed_design_pass_rows"] == 12
    assert s["rows_moved_in"] == 6 and s["rows_moved_out"] == 6 and s["moved_out_by"] == {"work_auth": 6}
    # no moves: the strict loader and the fresh-3 loader agree
    d2 = tmp_path / "same"
    synth_fresh_dir(d2)
    assert fs.load_fresh(d2) == fs.load_fresh(d2, strict_design=False)


def synth_project(tmp: Path) -> dict:
    """A git repo laid out like the project with a synthetic sealed fresh test 3 and its committed pins."""
    import hashlib
    repo = git_repo(tmp / "proj")
    d = repo / "data/built/fresh_test3"
    synth_fresh_dir(d)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    ev = repo / "eval"
    ev.mkdir()
    for n in ("protocol.json", "protocol_amendment_1.json", "protocol_amendment_2.json"):
        (ev / n).write_text(json.dumps({"what": n}))
    core = ["rows", "cards", "postings"]
    man = {"n_rows": 24, "files": {f"data/built/fresh_test3/{n}.jsonl": sha(d / f"{n}.jsonl") for n in core}
           | {"data/built/fresh_test3/label_batches/batch_01.jsonl": sha(d / "label_batches/batch_01.jsonl")}}
    (ev / "fresh_test3_manifest.json").write_text(json.dumps(man))
    (ev / "protocol_amendment_3.json").write_text(json.dumps({
        "amends": {"files": {f"eval/{n}": sha(ev / n) for n in ("protocol.json", "protocol_amendment_1.json",
                                                                 "protocol_amendment_2.json")}},
        "fresh_test_3": {"manifest_sha256": sha(ev / "fresh_test3_manifest.json")}}))
    data = {"sha256": {f"data/built/fresh_test3/{n}": sha(d / n) for n in
                       ("rows.jsonl", "cards.jsonl", "postings.jsonl", "gold.jsonl", "label_batches/batch_01.jsonl",
                        "labels/A/batch_01.jsonl")}}
    (ev / "fresh_test3_data_hashes.json").write_text(json.dumps(data))
    (ev / "fresh_test3_gold_manifest.json").write_text(json.dumps({"n_gold": 24, "gold_sha256": sha(d / "gold.jsonl"),
                                                                   "missing_rows": []}))
    sel = ev / "selection_cycle3.json"
    sel.write_text(json.dumps({"pick": "h", "amendment": "eval/protocol_amendment_3.json", "decision": "stop"}))
    git_commit(repo, *sorted(ev.iterdir()))
    rows = [json.loads(l) for l in (d / "gold.jsonl").open()]
    return {"repo": repo, "dir": d, "sel": sel, "gold": {r["row_id"]: r["gold"] for r in rows}}


def point_at(monkeypatch, repo: Path) -> None:
    ev = repo / "eval"
    for k, v in {"ROOT": repo, "FRESH3": repo / "data/built/fresh_test3",
                 "AMENDMENT2": ev / "protocol_amendment_2.json", "AMENDMENT3": ev / "protocol_amendment_3.json",
                 "FRESH3_MANIFEST": ev / "fresh_test3_manifest.json",
                 "FRESH3_DATA_HASHES": ev / "fresh_test3_data_hashes.json",
                 "FRESH3_GOLD_MANIFEST": ev / "fresh_test3_gold_manifest.json",
                 "FRESH3_RESULT": ev / "fresh_test3_result.json"}.items():
        monkeypatch.setattr(fs, k, v)


def test_cmd_fresh3_scores_once_on_a_synthetic_project(tmp_path, monkeypatch):
    P = synth_project(tmp_path)
    point_at(monkeypatch, P["repo"])
    keep = {k: "Keep" for k in P["gold"]}
    sc = [f"h={write_scores(tmp_path, 'h.json', P['gold'])}", f"base={write_scores(tmp_path, 'b.json', keep)}",
          f"d3={write_scores(tmp_path, 'd.json', keep)}"]
    a = fs.argparse.Namespace(selection=str(P["sel"]), score=sc)
    fs.cmd_fresh3(a)
    out = json.loads((P["repo"] / "eval/fresh_test3_result.json").read_text())
    assert out["set"] == "fresh_test3" and out["chosen"] == "h+H'" and out["selection_file"]["pick"] == "h"
    assert out["selection_file"]["path"] == "eval/selection_cycle3.json"
    assert out["eligibility_vs_sealed_design"]["rows_moved_in"] == 0
    assert (P["repo"] / "eval/fresh_test3_result.md").exists()
    before = (P["repo"] / "eval/fresh_test3_result.json").read_text()
    with pytest.raises(SystemExit, match="scored once"):
        fs.cmd_fresh3(a)
    (P["repo"] / "eval/fresh_test3_result.json").unlink()  # the .md alone still blocks a second scoring
    with pytest.raises(SystemExit, match="scored once"):
        fs.cmd_fresh3(a)
    assert before


def test_check_fresh3_catches_every_pin(tmp_path, monkeypatch):
    P = synth_project(tmp_path)
    point_at(monkeypatch, P["repo"])
    fs.check_fresh3()
    d, ev = P["dir"], P["repo"] / "eval"

    def broken(path: Path, match: str) -> None:
        old = path.read_bytes()
        path.write_bytes(old + b"\n")
        with pytest.raises(SystemExit, match=match):
            fs.check_fresh3()
        path.write_bytes(old)
        fs.check_fresh3()

    broken(d / "rows.jsonl", "fresh_test3_manifest.json pinned")  # a manifest pin
    broken(d / "labels/A/batch_01.jsonl", "fresh_test3_data_hashes.json pinned")  # pinned by the data-hash file only
    broken(d / "gold.jsonl", "gold.jsonl is not the file eval/fresh_test3_data_hashes.json pinned")
    broken(ev / "protocol_amendment_2.json", "not committed unchanged")
    broken(ev / "fresh_test3_data_hashes.json", "not committed unchanged")
    (d / "labels/A/batch_01.jsonl").unlink()
    with pytest.raises(SystemExit, match="is missing"):
        fs.check_fresh3()


def test_check_fresh3_reads_amendment3s_own_pins(tmp_path, monkeypatch):
    import subprocess
    P = synth_project(tmp_path)
    point_at(monkeypatch, P["repo"])
    ev = P["repo"] / "eval"
    (ev / "protocol.json").write_text(json.dumps({"what": "edited"}))
    git_commit(P["repo"], ev / "protocol.json")  # committed, but not the file amendment 3 amends
    with pytest.raises(SystemExit, match="is not the file eval/protocol_amendment_3.json amends"):
        fs.check_fresh3()
    subprocess.run(["git", "-C", str(P["repo"]), "reset", "-q", "--hard", "HEAD~1"], check=True)
    fs.check_fresh3()
    m = json.loads((ev / "fresh_test3_manifest.json").read_text())
    (ev / "fresh_test3_manifest.json").write_text(json.dumps(m | {"note": "x"}))
    git_commit(P["repo"], ev / "fresh_test3_manifest.json")
    with pytest.raises(SystemExit, match="not the manifest eval/protocol_amendment_3.json pins"):
        fs.check_fresh3()


def test_read_pick3_needs_a_committed_selection_naming_amendment3(tmp_path):
    repo = git_repo(tmp_path)
    s2 = repo / "selection_cycle2.json"
    s2.write_text(json.dumps({"pick": "h", "amendment": "eval/protocol_amendment_2.json"}))
    s3 = repo / "selection_cycle3.json"
    s3.write_text(json.dumps({"pick": "h3a", "amendments": ["eval/protocol_amendment_3.json",
                                                            "eval/protocol_amendment_4.json"]}))
    with pytest.raises(SystemExit, match="not committed"):
        fs.read_pick3(s3)
    git_commit(repo, s2, s3)
    assert fs.read_pick3(s3) == "h3a"
    with pytest.raises(SystemExit, match="does not name eval/protocol_amendment_3.json"):
        fs.read_pick3(s2)


def test_fresh3_cli_takes_no_free_chosen_argument(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fit_eval_score.py", "fresh3", "--score", "h=x.json", "--chosen", "h"])
    with pytest.raises(SystemExit):
        fs.main()


# ---------------------------------------------------------------- fresh test 4 (amendments 4c, 5, 5b), synthetic only

PICK4 = "B+H''_without_work_mode|decider"
BASE4 = "base+H''_without_work_mode|decider"
CUT = {"t_lo": 0.27, "t_hi": 0.59}
S_OF = {"Keep": [5.0, 0.0, 0.0], "Explore": [0.0, 5.0, 0.0], "Skip": [0.0, 0.0, 5.0]}  # s: 0.99 / 0.50 / 0.007


def spec4() -> dict:
    return {"pick_system": PICK4, "base_system": BASE4,
            "pick": {"model": "B", "variant": "H''_without_work_mode", "readout": "decider", **CUT},
            "base": dict(CUT), "selection": {"path": "eval/selection_cycle3.json", "sha256": "x"}}


def fresh4_rows(n_post: int = 12) -> list[dict]:
    """fresh2_rows with test 4's fields: one posting without a card (its rows gold Skip, no verdict), the last posting's
    employer new to tests 1-3, the last third of postings amendment 4b's extension, labeler B off on row 0."""
    rows = fresh2_rows(n_post)
    for r in rows:
        p = int(r["posting_id"][4:])
        r |= {"has_card": True, "eligibility_pass_design": r["eligibility_pass"], "posting_length": "3000+",
              "employer": f"emp{p}", "employer_in_fresh_test1_3": p != n_post - 1, "extension_4b": p >= 2 * n_post // 3}
        if p == 0:
            r |= {"has_card": False, "verdict": dict(fs.NO_CARD_VERDICT), "eligibility_pass": False,
                  "eligibility_pass_design": False, "rules_full": "Explore", "rules_skills_free": "Explore", "gold": "Skip",
                  "label_A": "Skip", "label_B": "Skip", "agreed": True}
    return rows


def scores_of(rows: list[dict], labels: list[str]) -> np.ndarray:
    return np.array([S_OF[x] for x in labels])


def test_fresh4_pass_needs_both_baselines_and_reports_what_the_brief_asks():
    rows = fresh4_rows()
    gold = [r["gold"] for r in rows]
    S = {"B": scores_of(rows, gold), "h": scores_of(rows, ["Keep"] * len(rows)), "base": scores_of(rows, ["Keep"] * len(rows))}
    res, P, sv = fs.score_fresh4(rows, S, spec4(), draws=200)
    p = res["pass"]
    assert res["set"] == "fresh_test4" and res["chosen"] == PICK4 and res["base_comparator"] == BASE4
    assert set(res["systems"]) == {PICK4, "h+H'", "rules_full", BASE4}
    assert p["passed"] and set(p["against"]) == {"rules_full", BASE4} and p["slice"] == "eligibility_pass"
    assert all(v["lower_bound_exact"] > 0 for v in p["against"].values())
    # the pick's decider reads gold back; H'' (without work_mode) Skips the work-auth rows, caps nothing that is gold
    assert P[PICK4] == gold and sv["B"].shape == (len(rows),) and set(sv) == {"B", "base"}
    # base all-Keep: H'' turns the hard rows to Skip and the salary rows to Explore; no-card rows stay Keep
    assert [P[BASE4][i] for i in range(4)] == ["Keep"] * 4  # posting 0 has no card: no override
    assert [P[BASE4][i] for i in range(4, 8)] == ["Keep", "Keep", "Skip", "Explore"]
    assert P["h+H'"] == P[BASE4]  # h is argmax + H' (HP_CATS + salary cap); the fixture moves nothing between them
    S_ = res["slices"]
    for name in ("all", "all_with_card", "eligibility_pass", "agreed_only", "agreed_eligibility_pass",
                 "eligibility_pass_design", "employer_new_vs_fresh_tests_1_3", "employer_shared_with_fresh_tests_1_3",
                 "extension_4b", "first_940", "field:lab_bio", "length:3000+"):
        assert S_[name] is not None, name
    assert S_["all"]["n"] == 48 and S_["all_with_card"]["n"] == 44 and S_["eligibility_pass"]["n"] == 22
    assert S_["employer_new_vs_fresh_tests_1_3"]["n"] == 4 and S_["extension_4b"]["n"] == 16
    assert S_["agreed_eligibility_pass"]["n"] == 11  # row 0 of each posting has A != B
    h = res["headline"]
    assert set(h) == {"all", "eligibility_pass", "agreed_only", "agreed_eligibility_pass"}
    k = h["eligibility_pass"]["systems"][PICK4]["keep"]
    assert k == {"gold_keeps": 11, "found": 11, "recall": 1.0, "predicted_keeps": 11, "precision": 1.0,
                 "recall_step": round(1 / 11, 4)}
    assert h["all"]["systems"][BASE4]["unsafe_keep"] == 4 + 11  # the no-card Skips and the Explore rows
    ag = res["agreement_subset (reported, not for pass)"]
    assert ag["agreed_eligibility_pass"]["would_pass"] and set(ag["agreed_eligibility_pass"]["against"]) == {"rules_full", BASE4}
    ne = res["new_employer_slice"]
    assert ne["n"] == 4 and ne["employers"] == ["emp11"] and "too small" in ne["flag"]
    assert res["no_card_rows"]["rows"] == ["r0_0", "r0_1", "r0_2", "r0_3"] and res["no_card_rows"]["gold"] == {"Skip": 4}
    # base as good as the pick: rules_full beaten, base not, no pass
    res2, _, _ = fs.score_fresh4(rows, {**S, "base": S["B"]}, spec4(), draws=200)
    assert res2["pass"]["against"]["rules_full"]["beaten"] and not res2["pass"]["against"][BASE4]["beaten"]
    assert not res2["pass"]["passed"]
    with pytest.raises(SystemExit, match="needs score files"):
        fs.score_fresh4(rows, {"B": S["B"], "base": S["base"]}, spec4(), draws=20)
    md = fs.markdown(res)
    assert "Pass bar (amendment 2" in md and "(11/11)" in md and "## agreed_eligibility_pass" in md
    assert "New-employer slice" in md and "| extension_4b |" in md and "Sealed-design eligibility-pass slice" in md
    sd = res["sealed_design_slice (sensitivity, not for pass)"]
    assert sd["n"] == 22 and set(sd["against"]) == {"rules_full", BASE4} and all(v["would_be_beaten"] for v in sd["against"].values())


def test_fresh4_uses_the_committed_cut_offs_not_a_refit():
    rows = fresh4_rows(6)
    s_mid = [0.0, 2.0, 0.0]  # s about 0.5 + ...: P = (0.106, 0.787, 0.106) -> s = 0.50
    S = {"B": np.array([s_mid] * len(rows)), "h": scores_of(rows, ["Skip"] * len(rows)),
         "base": np.array([s_mid] * len(rows))}
    sp = spec4()
    sp["pick"] |= {"t_lo": 0.2, "t_hi": 0.45}  # s 0.50 >= t_hi: Keep
    sp["base"] = {"t_lo": 0.55, "t_hi": 0.9}  # s 0.50 < t_lo: Skip
    _, P, _ = fs.score_fresh4(rows, S, sp, draws=10)
    ok = [i for i, r in enumerate(rows) if r["eligibility_pass"]]
    assert {P[PICK4][i] for i in ok} == {"Keep"} and {P[BASE4][i] for i in ok} == {"Skip"}


# synthetic fresh test 4 project on disk: the real file layout, a tmp git repo stands in for the project

def synth_fresh4_project(tmp: Path) -> dict:
    import hashlib
    repo = git_repo(tmp / "proj")
    d = repo / "data/built/fresh_test4"
    synth_fresh_dir(d)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    # test 4's design keys; posting 0 loses its card (its rows gold Skip)
    rows = [json.loads(l) for l in (d / "rows.jsonl").open()]
    nocard = "greenhouse:synth:0"
    for r in rows:
        p = int(r["posting_id"].rsplit(":", 1)[1])
        r["design"] |= {"has_card": p != 0, "employer_in_fresh_test1_3": p != 5}
        if p >= 4:
            r["design"]["extension"] = "4b"
        if p == 0:
            r["design"]["eligibility_pass"] = None
    (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cards = [json.loads(l) for l in (d / "cards.jsonl").open()]
    (d / "cards.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cards if c["id"] != nocard))
    gold = [json.loads(l) for l in (d / "gold.jsonl").open()]
    for g in gold:
        if g["row_id"].startswith("f3_00"):
            g |= {"gold": "Skip", "label_A": "Skip", "label_B": "Skip"}
    (d / "gold.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gold))
    ev = repo / "eval"
    ev.mkdir()
    (ev / "protocol_amendment_2.json").write_text(json.dumps({"what": "2"}))
    (ev / "protocol_amendment_4.json").write_text(json.dumps({"what": "4"}))
    (ev / "protocol_amendment_4b.json").write_text(json.dumps({"amends": {"files": {"eval/protocol_amendment_4.json": "old"}}}))
    (ev / "protocol_amendment_5.json").write_text(json.dumps({"amends": {"files": {
        f"eval/protocol_amendment_{k}.json": sha(ev / f"protocol_amendment_{k}.json") for k in ("2", "4")}}}))
    (ev / "protocol_amendment_5b.json").write_text(json.dumps({"amends": {"files": {
        "eval/protocol_amendment_5.json": sha(ev / "protocol_amendment_5.json")}}}))
    core = ["rows", "cards", "postings"]
    man = {"n_rows": 24, "files": {f"data/built/fresh_test4/{n}.jsonl": sha(d / f"{n}.jsonl") for n in core}}
    (ev / "fresh_test4_manifest.json").write_text(json.dumps(man))
    data = {"manifest_sha256": sha(ev / "fresh_test4_manifest.json"),
            "sha256": {f"data/built/fresh_test4/{n}": sha(d / n) for n in
                       ("rows.jsonl", "label_batches/batch_01.jsonl", "labels/A/batch_01.jsonl")}}
    (ev / "fresh_test4_data_hashes.json").write_text(json.dumps(data))
    (ev / "fresh_test4_gold_manifest.json").write_text(json.dumps({"n_gold": 24, "gold_sha256": sha(d / "gold.jsonl"),
                                                                   "missing_rows": [], "input_sha256": {}}))
    (ev / "protocol_amendment_4c.json").write_text(json.dumps({"amends": {"files": {
        "eval/fresh_test4_gold_manifest.json": sha(ev / "fresh_test4_gold_manifest.json")}}}))
    inp = repo / "data/built/fit/eval_inputs"
    pins = {}
    for m in ("B", "h", "base"):
        (inp / m).mkdir(parents=True)
        (inp / m / "fresh_test4.jsonl").write_text(f"{m}\n")
        pins[m] = {f"data/built/fit/eval_inputs/{m}/fresh_test4.jsonl": sha(inp / m / "fresh_test4.jsonl")}
    (ev / "fresh_test4_eval_inputs.json").write_text(json.dumps(pins))
    amend = {f"eval/protocol_amendment_{k}.json": sha(ev / f"protocol_amendment_{k}.json") for k in ("5", "5b")}
    sel = {"amendments": list(amend), "amendments_sha256": amend,
           "definitions": {"H''_without_work_mode": ["degree_field", "degree_level", "enrollment", "location",
                                                    "work_auth", "salary cap"]},
           "pick": {"system": PICK4, "model": "B", "eligibility_layer": "H''_without_work_mode", "readout": "decider",
                    "cut_offs": {**CUT, "floors_met": "0.60/0.25/0.25"}, "no_system_cleared_floors": False},
           "decider_cut_offs": {"B+H''_without_work_mode": dict(CUT)},
           "base_decider_cut_offs": {"base+H''_without_work_mode": {"t_lo": 0.6, "t_hi": 0.9}},
           "base_comparator_for_test_4": {"system": BASE4, "cut_offs": {"t_lo": 0.6, "t_hi": 0.9,
                                                                         "floors_met": "0.60/0.25/0.25"}}}
    (ev / "selection_cycle3.json").write_text(json.dumps(sel))
    git_commit(repo, *sorted(ev.iterdir()))
    gold = {json.loads(l)["row_id"]: json.loads(l)["gold"] for l in (d / "gold.jsonl").open()}
    return {"repo": repo, "dir": d, "sel": ev / "selection_cycle3.json", "gold": gold}


def point_at4(monkeypatch, repo: Path) -> None:
    ev = repo / "eval"
    for k, v in {"ROOT": repo, "FRESH4": repo / "data/built/fresh_test4",
                 "FRESH4_MANIFEST": ev / "fresh_test4_manifest.json",
                 "FRESH4_DATA_HASHES": ev / "fresh_test4_data_hashes.json",
                 "FRESH4_GOLD_MANIFEST": ev / "fresh_test4_gold_manifest.json",
                 "FRESH4_INPUTS": ev / "fresh_test4_eval_inputs.json",
                 "FRESH4_RESULT": ev / "fresh_test4_result.json",
                 "FRESH4_ROWS_OUT": repo / "data/models/c3_eval/test4/test4_rows.jsonl",
                 "SELECTION_C3": ev / "selection_cycle3.json",
                 "FRESH4_AMENDMENTS": tuple(ev / f"protocol_amendment_{k}.json" for k in ("2", "4", "4b", "4c", "5", "5b"))
                 }.items():
        monkeypatch.setattr(fs, k, v)


def write_scores4(tmp: Path, tag: str, labels: dict[str, str], **hdr) -> Path:
    model = "base_Qwen--Qwen3.5-4B" if tag == "base" else f"merged_{tag}"
    head = {"model": model, "dev": "fresh_test4.jsonl", "framing": "chat", "dtype": "bfloat16",
            "timing": {"batch": 1}} | hdr
    out = [{"pair_id": rid, "scores": S_OF[lab], "pred": lab} for rid, lab in labels.items()]
    p = tmp / f"{tag}_fresh_test4.json"
    p.write_text(json.dumps(head | {"rows": out}))
    return p


def test_load_fresh4_no_card_rows_get_no_verdict_and_explore_rules(tmp_path, monkeypatch):
    P = synth_fresh4_project(tmp_path)
    rows = fs.load_fresh4(P["dir"])
    nc = [r for r in rows if not r["has_card"]]
    assert len(nc) == 4 and all(r["verdict"]["rule"] is None and not r["eligibility_pass"] for r in nc)
    assert {r["rules_full"] for r in nc} == {"Explore"} and {r["gold"] for r in nc} == {"Skip"}
    assert sum(r["extension_4b"] for r in rows) == 8 and sum(not r["employer_in_fresh_test1_3"] for r in rows) == 4
    s = fs.design_flag_summary3(rows)
    assert s["rows_moved_in"] == 0 and s["rows_moved_out"] == 0


def test_cmd_fresh4_scores_once_on_a_synthetic_project(tmp_path, monkeypatch):
    P = synth_fresh4_project(tmp_path)
    point_at4(monkeypatch, P["repo"])
    keep = {k: "Keep" for k in P["gold"]}
    sc = [f"B={write_scores4(tmp_path, 'B', P['gold'])}", f"h={write_scores4(tmp_path, 'h', keep)}",
          f"base={write_scores4(tmp_path, 'base', keep)}"]
    a = fs.argparse.Namespace(selection=str(P["sel"]), score=sc, session=["matcher-t4-b", "matcher-t4-h"])
    fs.cmd_fresh4(a)
    res = P["repo"] / "eval/fresh_test4_result.json"
    out = json.loads(res.read_text())
    assert out["set"] == "fresh_test4" and out["chosen"] == PICK4 and out["spec"]["base"]["t_lo"] == 0.6
    assert out["selection_file"]["path"] == "eval/selection_cycle3.json" and out["colab_sessions"][0] == "matcher-t4-b"
    assert set(out["score_files"]) == {"B", "h", "base"} and len(out["prompt_files"]) == 3
    rows_file = P["repo"] / "data/models/c3_eval/test4/test4_rows.jsonl"
    lines = [json.loads(l) for l in rows_file.open()]
    assert len(lines) == 24 and set(lines[0]["systems"]) == {PICK4, "h+H'", "rules_full", BASE4}
    assert set(lines[0]["scores"]) == {"B", "h", "base"} and set(lines[0]["s"]) == {"B", "base"}
    assert out["per_row_file"]["path"] == "data/models/c3_eval/test4/test4_rows.jsonl"
    with pytest.raises(SystemExit, match="scored once"):
        fs.cmd_fresh4(a)
    res.unlink()
    res.with_suffix(".md").unlink()  # the per-row file alone still blocks a second scoring
    with pytest.raises(SystemExit, match="scored once"):
        fs.cmd_fresh4(a)


def test_cmd_fresh4_refuses_wrong_score_files(tmp_path, monkeypatch):
    P = synth_fresh4_project(tmp_path)
    point_at4(monkeypatch, P["repo"])
    keep = {k: "Keep" for k in P["gold"]}
    good = {t: write_scores4(tmp_path, t, P["gold"] if t == "B" else keep) for t in ("B", "h", "base")}

    def run(scores: dict, match: str) -> None:
        a = fs.argparse.Namespace(selection=str(P["sel"]), score=[f"{t}={p}" for t, p in scores.items()], session=None)
        with pytest.raises(SystemExit, match=match):
            fs.cmd_fresh4(a)
        assert not (P["repo"] / "eval/fresh_test4_result.json").exists()

    run({"B": good["B"], "h": good["h"]}, "exactly")
    run({**good, "d3": good["h"]}, "exactly")
    sub = tmp_path / "x"
    sub.mkdir()
    run({**good, "B": write_scores4(sub, "B", P["gold"], dtype="float32")}, "dtype")
    run({**good, "h": write_scores4(sub, "h", keep, timing={"batch": 4})}, "batch")
    run({**good, "base": write_scores4(sub, "base", keep, model="merged_h")}, "model")
    run({**good, "B": write_scores4(sub, "B", P["gold"], dev="fresh_test3.jsonl")}, "dev")
    run({**good, "B": write_scores4(sub, "B", dict(list(P["gold"].items())[1:]))}, "rows missing")


def test_check_fresh4_catches_every_pin(tmp_path, monkeypatch):
    P = synth_fresh4_project(tmp_path)
    point_at4(monkeypatch, P["repo"])
    fs.check_fresh4()
    d, ev, repo = P["dir"], P["repo"] / "eval", P["repo"]

    def broken(path: Path, match: str) -> None:
        old = path.read_bytes()
        path.write_bytes(old + b"\n")
        with pytest.raises(SystemExit, match=match):
            fs.check_fresh4()
        path.write_bytes(old)
        fs.check_fresh4()

    broken(d / "rows.jsonl", "fresh_test4_manifest.json pinned")
    broken(d / "labels/A/batch_01.jsonl", "fresh_test4_data_hashes.json pinned")
    broken(d / "gold.jsonl", "gold eval/fresh_test4_gold_manifest.json pins")
    broken(repo / "data/built/fit/eval_inputs/B/fresh_test4.jsonl", "prompt file")
    for n in ("protocol_amendment_2.json", "protocol_amendment_4c.json", "protocol_amendment_5b.json",
              "fresh_test4_eval_inputs.json", "fresh_test4_gold_manifest.json"):
        broken(ev / n, "not committed unchanged")
    (ev / "protocol_amendment_2.json").write_text(json.dumps({"what": "edited"}))
    git_commit(repo, ev / "protocol_amendment_2.json")  # committed, but not the file amendment 5 amends
    with pytest.raises(SystemExit, match="is not the file eval/protocol_amendment_5.json amends"):
        fs.check_fresh4()


def test_read_spec4_reads_only_the_committed_pick_and_its_fitted_cut_offs(tmp_path, monkeypatch):
    P = synth_fresh4_project(tmp_path)
    point_at4(monkeypatch, P["repo"])
    sp = fs.read_spec4(P["sel"])
    assert sp["pick_system"] == PICK4 and sp["base_system"] == BASE4 and sp["pick"]["t_lo"] == 0.27
    assert sp["base"] == {"t_lo": 0.6, "t_hi": 0.9, "floors_met": "0.60/0.25/0.25"}
    sel = json.loads(P["sel"].read_text())

    def bad(edit, match: str) -> None:
        x = json.loads(json.dumps(sel))
        edit(x)
        P["sel"].write_text(json.dumps(x))
        with pytest.raises(SystemExit, match="not committed"):
            fs.read_spec4(P["sel"])
        git_commit(P["repo"], P["sel"])
        with pytest.raises(SystemExit, match=match):
            fs.read_spec4(P["sel"])
        P["sel"].write_text(json.dumps(sel))
        git_commit(P["repo"], P["sel"])
        fs.read_spec4(P["sel"])

    bad(lambda x: x["pick"]["cut_offs"].update(t_lo=0.3), "cut-offs are not the fitted")
    bad(lambda x: x["base_comparator_for_test_4"]["cut_offs"].update(t_hi=0.8), "cut-offs are not the fitted")
    bad(lambda x: x["base_comparator_for_test_4"].update(system="base+H''_with_work_mode|decider"), "base comparator")
    bad(lambda x: x["pick"].update(model="A"), "disagree")
    bad(lambda x: x["amendments_sha256"].update({"eval/protocol_amendment_5.json": "0" * 64}), "selected under")
    bad(lambda x: x["pick"].update(no_system_cleared_floors=True), "fallback")


def test_fresh4_cli_takes_no_free_chosen_argument(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fit_eval_score.py", "fresh4", "--score", "B=x.json", "--chosen", "B"])
    with pytest.raises(SystemExit):
        fs.main()
