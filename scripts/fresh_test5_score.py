"""Amendment 6b scoring: B-colab's and C's decider cut-offs on dev (fresh tests 1-4 pooled), then fresh test 5 once.

  fresh_test5_score.py dev     -> eval/cutoffs_Bcolab_C_t1to4.json
      B-colab and C + H''_without_work_mode + decider: cut-offs fit on tests 1-4 pooled (2,539 rows, all rows) by
      fit_select_cycle3.fit_cutoffs, exactly as refit_cutoffs_t1to4.py fits refit-B; dev metrics beside refit-B's.
      Reads the stored test 1-4 row files (selection_rows.jsonl, test4_rows.jsonl; sha256 checked against
      selection_cycle3.json and fresh_test4_result.json) and the score files of B-colab and C on tests 1-4.
      Never reads a test-5 score file.
  fresh_test5_score.py test5   -> eval/fresh_test5_result.json (+ .md), data/models/c3_eval/test5/test5_rows.jsonl
      write-once; refuses unless eval/fresh_test5_eval_inputs.json, eval/cutoffs_B_t1to4.json,
      eval/cutoffs_Bcolab_C_t1to4.json, eval/protocol_amendment_6b.json and the test-5 manifests are committed
      unchanged. Systems: B (shipped cut-offs, eval/selection_cycle3.json), refit-B (eval/cutoffs_B_t1to4.json),
      B-colab and C (their dev cut-offs), each + H''_without_work_mode + decider; rules_full beside them.

Definitions (amendments 5b, 6, 6b, nothing added):
  label      decider: s = P(Keep) + 0.5 P(Explore) (softmax of the float32 first-token scores, temperature 1);
             s >= t_hi Keep, s < t_lo Skip, else Explore; then hybrid(label, rule verdict, HP_CATS - {employment_type,
             work_mode}, salary cap).
  rows       fresh test 5's 1,187 rows; rule verdict and the eligibility-pass slice recomputed from profile and card
             with the rules in effect (fit_eval_score.base_row on the card after rules.mark_thin on the posting body),
             as load_fresh4 does.
  Keep-F1    F1 of Keep against not-Keep (2 tp / (2 tp + fp + fn)).
  macro-F1   evalstats.macro_f1_of.
  bootstrap  fit_eval_score.two_way_draws: postings with replacement, then rows within each drawn posting; 2,000
             draws, numpy default_rng(20261022); one draw set per slice, shared by every system and difference.
  bar        on the eligibility-pass slice: C minus refit-B on Keep-F1 with a 2.5th-percentile lower bound (unrounded)
             above 0, and C minus refit-B on macro-F1 with a lower bound above -0.03. B-colab is reported beside it
             with the same differences, no pass claim.
Loads no model and calls nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import fit_eval_score as F  # noqa: E402
import fit_select_cycle3 as S  # noqa: E402
from matcher.evalstats import LABELS, macro_f1_of  # noqa: E402
from matcher.nearcopy import company_key  # noqa: E402
from matcher.rules import mark_thin  # noqa: E402
from matcher.schema import CandidateCard, RequirementCard  # noqa: E402

VARIANT = "H''_without_work_mode"
CATS = S.VARIANTS[VARIANT]
TESTS = ("fresh_test", "fresh_test2", "fresh_test3", "fresh_test4")
SEL = ROOT / "eval/selection_cycle3.json"
T4R = ROOT / "eval/fresh_test4_result.json"
REFIT_B = ROOT / "eval/cutoffs_B_t1to4.json"
DEV_OUT = ROOT / "eval/cutoffs_Bcolab_C_t1to4.json"
PINS = ROOT / "eval/fresh_test5_eval_inputs.json"
AM6, AM6B = ROOT / "eval/protocol_amendment_6.json", ROOT / "eval/protocol_amendment_6b.json"
ROWS123 = ROOT / "data/models/c3_eval/selection_rows.jsonl"
ROWS4 = ROOT / "data/models/c3_eval/test4/test4_rows.jsonl"
SCORES = ROOT / "data/models/c3_eval/test5"
FRESH5 = ROOT / "data/built/fresh_test5"
RESULT = ROOT / "eval/fresh_test5_result.json"
ROWS5_OUT = SCORES / "test5_rows.jsonl"
SEED5, DRAWS = 20261022, 2000
MARGIN = -0.03
INFERENCE_HISTORY = {
    "scored": "this command is the only time any test-5 score file is read against gold; it ran once",
    "B": "L4 session 1 (2026-10-22 22:49 UTC) lost its connection at the merge, before any inference; session 2 "
         "(22:59-23:25) wrote B_fresh_test5.json, the file used",
    "Bcolab and C": "L4 session 1 (00:15 / 00:18) stopped at the merge (adapter_config named base 'base'; fixed in "
                    "3d2349f, weights unchanged). Session 2 (00:38-00:41 start) scored tests 1-4; the Mac lost DNS "
                    "about 00:50, the watchdogs read that as 'session gone' and exited, and both sessions were gone "
                    "by 01:35 before any test-5 file reached the Mac, so test-5 inference may have run on those VMs "
                    "and was lost unread. Session 3 (01:35-02:02) ran test 5 only and wrote the files used. The dev "
                    "cut-offs (498c1ec) were committed before any test-5 file of B-colab or C existed on the Mac.",
}
CANDS = ("Bcolab", "C")
FMT = {"B": "B", "Bcolab": "B", "C": "C"}
SYS = {"B|shipped": "B", "B|refit": "B", "Bcolab": "Bcolab", "C": "C"}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT.resolve()))


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def need_committed(*fs: Path) -> None:
    for f in fs:
        if not F.committed(f):
            raise SystemExit(f"{rel(f)} is not committed unchanged")


def score_file(tag: str, test: str) -> Path:
    return SCORES / tag / f"{tag}_{test}.json"


def read_scores(tag: str, test: str, ids: list[str]) -> np.ndarray:
    """rows x 3 float32 first-token scores in ids order, from a colab_eval run of merged_TAG on the pinned prompt file
    (bf16, batch 1, chat framing), after read_raw's checks."""
    p = score_file(tag, test)
    d = json.loads(p.read_text())
    prompt = f"data/built/fit/eval_inputs/{FMT[tag]}/{test}.jsonl"
    pins = json.loads(PINS.read_text())[tag]
    if prompt not in pins:
        raise SystemExit(f"{prompt} is not pinned for {tag}")
    want = {"framing": "chat", "dtype": "bfloat16", "model": f"merged_{tag}", "dev": Path(prompt).name}
    for k, v in want.items():
        if d.get(k) != v:
            raise SystemExit(f"{rel(p)}: {k} is {d.get(k)!r}, not {v!r}")
    if (d.get("timing") or {}).get("batch") != 1:
        raise SystemExit(f"{rel(p)}: not batch 1")
    return S.read_raw(p, ids)


def keep_f1(g: np.ndarray, p: np.ndarray) -> float:
    tp = int(((g == 0) & (p == 0)).sum())
    fp = int(((g != 0) & (p == 0)).sum())
    fn = int(((g == 0) & (p != 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def counts(g: np.ndarray, p: np.ndarray) -> dict:
    tp = int(((g == 0) & (p == 0)).sum())
    pk, gk = int((p == 0).sum()), int((g == 0).sum())
    rec = {LABELS[w]: (float(((g == w) & (p == w)).sum() / (g == w).sum()) if (g == w).sum() else None) for w in range(3)}
    return {"keep_gold": gk, "keep_found": tp, "keep_predicted": pk, "wrong_keeps": pk - tp,
            "keep_recall": tp / gk if gk else None, "keep_precision": tp / pk if pk else None,
            "keep_f1": keep_f1(g, p), "macro_f1": float(macro_f1_of(g, p)),
            "explore_recall": rec["Explore"], "skip_recall": rec["Skip"], "accuracy": float((g == p).mean())}


# ---------------------------------------------------------------- dev: cut-offs on tests 1-4 pooled


def dev_rows() -> list[dict]:
    sel, t4 = json.loads(SEL.read_text()), json.loads(T4R.read_text())
    if sha(ROWS123) != sel["per_row_file"]["sha256"] or sha(ROWS4) != t4["per_row_file"]["sha256"]:
        raise SystemExit("the stored test 1-4 row files differ from their recorded sha256")
    rows = load(ROWS123) + [r | {"test": "fresh_test4"} for r in load(ROWS4)]
    assert len(rows) == 2539
    return rows


def cmd_dev(_: argparse.Namespace) -> None:
    need_committed(PINS, AM6B, REFIT_B)
    rows = dev_rows()
    g = F.as_idx([r["gold"] for r in rows])
    hard, sal = S.override_masks(rows, CATS)
    elig = np.array([bool(r["eligibility_pass"]) for r in rows])
    tid = np.array([TESTS.index(r["test"]) for r in rows])
    # refit-B on the same rows, recomputed from the stored B scores (must equal cutoffs_B_t1to4.json's numbers)
    rb = json.loads(REFIT_B.read_text())["new_cutoffs"]
    sB = S.decider_score(np.array([r["scores"]["B"] for r in rows], dtype=np.float64))
    labB = S.apply_h(S.decider_idx(sB, rb["t_lo"], rb["t_hi"]), hard, sal)
    refB = json.loads(REFIT_B.read_text())["metrics"]["all_rows_tests_1to4_pooled (the fit set)"]["new_cutoffs"]
    if abs(float(macro_f1_of(g, labB)) - refB["macro_f1"]) > 1e-12:
        raise SystemExit("refit-B recomputed here differs from eval/cutoffs_B_t1to4.json")
    out = {"refit-B": {"cutoffs": {"t_lo": rb["t_lo"], "t_hi": rb["t_hi"]}, "source": rel(REFIT_B),
                       "all_rows": counts(g, labB), "eligibility_pass": counts(g[elig], labB[elig]),
                       **{f"{t}_eligibility_pass": counts(g[elig & (tid == k)], labB[elig & (tid == k)])
                          for k, t in enumerate(TESTS)}}}
    files = {}
    for tag in CANDS:
        Sx = np.zeros((len(rows), 3))
        for k, t in enumerate(TESTS):
            m = tid == k
            Sx[m] = read_scores(tag, t, [r["row_id"] for r in np.array(rows, dtype=object)[m]])
            files[rel(score_file(tag, t))] = sha(score_file(tag, t))
        s = S.decider_score(Sx)
        fit = S.fit_cutoffs(s, g, hard, sal)
        lab = S.apply_h(S.decider_idx(s, fit["t_lo"], fit["t_hi"]), hard, sal)
        lab_rows = S.apply_h_rows(S.decider_idx(s, fit["t_lo"], fit["t_hi"]), rows, CATS)
        assert (lab == lab_rows).all(), "vectorised H'' differs from hybrid()"
        out[tag] = {"cutoffs": {k: fit[k] for k in ("t_lo", "t_hi", "quantile_lo", "quantile_hi", "grid_index_lo",
                                                     "grid_index_hi", "floors_met", "pairs_tried",
                                                     "pairs_clearing_floors")},
                    "fit_macro_f1": fit["macro_f1"], "fit_recall": fit["recall"],
                    "all_rows": counts(g, lab), "eligibility_pass": counts(g[elig], lab[elig]),
                    **{f"{t}_eligibility_pass": counts(g[elig & (tid == k)], lab[elig & (tid == k)])
                       for k, t in enumerate(TESTS)},
                    "argmax_eligibility_pass": counts(g[elig], S.apply_h(S.argmax_idx(Sx), hard, sal)[elig])}
    res = {"what": "B-colab's and C's decider cut-offs fit on fresh tests 1-4 pooled (amendment 6b dev), refit-B beside",
           "rule": "fit_select_cycle3.fit_cutoffs on all 2,539 rows of tests 1-4, B-colab / C + H''_without_work_mode, "
                   "decider; the same call refit_cutoffs_t1to4.py makes for refit-B",
           "rows": {"pooled": len(rows), "eligibility_pass": int(elig.sum()),
                    "gold_pooled": dict(Counter(r["gold"] for r in rows)),
                    "gold_eligibility_pass": dict(Counter(r["gold"] for r, e in zip(rows, elig) if e))},
           "systems": out,
           "inputs_sha256": {rel(p): sha(p) for p in (ROWS123, ROWS4, SEL, T4R, REFIT_B, PINS, AM6B,
                                                     ROOT / "scripts/fresh_test5_score.py",
                                                     ROOT / "scripts/fit_select_cycle3.py")} | files,
           "fresh_test5": "not read"}
    DEV_OUT.write_text(json.dumps(res, indent=1) + "\n")
    for k, v in out.items():
        e, a = v["eligibility_pass"], v["all_rows"]
        print(f"{k:8s} cut {v['cutoffs']['t_lo']:.4f}/{v['cutoffs']['t_hi']:.4f}  pass: mF1 {e['macro_f1']:.4f} "
              f"KF1 {e['keep_f1']:.4f} rK {e['keep_found']}/{e['keep_gold']} pK {e['keep_precision']:.3f} "
              f"wrongK {e['wrong_keeps']} | all: mF1 {a['macro_f1']:.4f} KF1 {a['keep_f1']:.4f}")
    print("wrote", rel(DEV_OUT))


# ---------------------------------------------------------------- test 5


def b_train_employers() -> set[str]:
    import build_fit_train3 as T3
    kept = {r["row_id"] for r in load(ROOT / "data/built/fit/cycle3/train3_rows.jsonl") if r["kept"]}
    rows, _ = T3.load_rows()
    return {company_key(e) for r in rows if r["row_id"] in kept for e in r["employers"] if company_key(e)}


def load_fresh5() -> list[dict]:
    rows = load(FRESH5 / "rows.jsonl")
    posts = {p["key"]: p for p in load(FRESH5 / "postings.jsonl")}
    cards = {c["id"]: c["card"] for c in load(FRESH5 / "cards.jsonl")}
    gold = {g["row_id"]: g for g in load(FRESH5 / "gold.jsonl")}
    if set(gold) != {r["row_id"] for r in rows}:
        raise SystemExit("gold.jsonl and rows.jsonl hold different rows")
    emps = b_train_employers()
    out = []
    for r in rows:
        p, g, d = posts[r["posting_id"]], gold[r["row_id"]], r["design"]
        if r["posting_id"] not in cards:
            raise SystemExit(f"{r['row_id']}: no card")
        card = mark_thin(RequirementCard.from_dict(cards[r["posting_id"]]), p.get("text") or "")
        x = F.base_row(r["row_id"], r["posting_id"], g["gold"], CandidateCard.from_dict(r["profile"]), card,
                       r["title"], r["resume_text"])
        x |= {"label_A": g["label_A"], "label_B": g["label_B"], "agreed": g["label_A"] == g["label_B"],
              "eligibility_pass_design": bool(d["eligibility_pass"]), "archetype": d["archetype"],
              "source_set": d["source_set"], "employer": r["employer"],
              "employer_in_B_training": company_key(r["employer"]) in emps}
        out.append(x)
    return out


def slices5(rows: list[dict]) -> dict[str, np.ndarray]:
    e = np.array([r["eligibility_pass"] for r in rows])
    a = np.array([r["agreed"] for r in rows])
    s = {"eligibility_pass": e, "agreed_eligibility_pass": e & a, "all_rows": np.ones(len(rows), bool),
         "agreed_only": a,
         "new_employer_eligibility_pass": e & ~np.array([r["employer_in_B_training"] for r in rows])}
    for k in sorted({r["archetype"] for r in rows}):
        s[f"archetype:{k}_eligibility_pass"] = e & np.array([r["archetype"] == k for r in rows])
    for k in sorted({r["source_set"] for r in rows}):
        s[f"source:{k}_eligibility_pass"] = e & np.array([r["source_set"] == k for r in rows])
    return s


def ci(vals: list[float], point: float) -> dict:
    return {"point": point, "low": float(np.percentile(vals, 2.5)), "high": float(np.percentile(vals, 97.5))}


def score_slice(rows: list[dict], g: np.ndarray, P: dict[str, np.ndarray], mask: np.ndarray,
                pairs: list[tuple[str, str]]) -> dict | None:
    idx = np.flatnonzero(mask)
    if not len(idx):
        return None
    gs = g[idx]
    groups = [rows[i]["posting_id"] for i in idx]
    draws = F.two_way_draws(groups, DRAWS, SEED5)
    out = {"n": int(len(idx)), "postings": len(set(groups)), "gold": dict(Counter(LABELS[i] for i in gs)),
           "systems": {}, "paired": {}}
    for k, p in P.items():
        ps = p[idx]
        c = counts(gs, ps)
        c["macro_f1_ci"] = ci([macro_f1_of(gs[d], ps[d]) for d in draws], c["macro_f1"])
        c["keep_f1_ci"] = ci([keep_f1(gs[d], ps[d]) for d in draws], c["keep_f1"])
        out["systems"][k] = c
    for a, b in pairs:
        pa, pb = P[a][idx], P[b][idx]
        kd = [keep_f1(gs[d], pa[d]) - keep_f1(gs[d], pb[d]) for d in draws]
        md = [macro_f1_of(gs[d], pa[d]) - macro_f1_of(gs[d], pb[d]) for d in draws]
        out["paired"][f"{a} - {b}"] = {
            "keep_f1_diff": ci(kd, keep_f1(gs, pa) - keep_f1(gs, pb)),
            "macro_f1_diff": ci(md, float(macro_f1_of(gs, pa) - macro_f1_of(gs, pb))),
            "rows_fixed": int(((pa == gs) & (pb != gs)).sum()), "rows_broken": int(((pa != gs) & (pb == gs)).sum())}
    return out


def verdict(sl: dict, me: str, ref: str) -> dict:
    d = sl["paired"][f"{me} - {ref}"]
    k_ok, m_ok = d["keep_f1_diff"]["low"] > 0, d["macro_f1_diff"]["low"] > MARGIN
    return {"keep_f1_diff": d["keep_f1_diff"], "keep_f1_lower_bound_above_0": bool(k_ok),
            "macro_f1_diff": d["macro_f1_diff"], "macro_f1_lower_bound_above_-0.03": bool(m_ok),
            "passed": bool(k_ok and m_ok)}


def cmd_test5(_: argparse.Namespace) -> None:
    from fit_eval_inputs import check_fresh5_inputs
    for f in (RESULT, RESULT.with_suffix(".md"), ROWS5_OUT):
        if f.exists():
            raise SystemExit(f"{rel(f)} exists: fresh test 5 is scored once")
    need_committed(PINS, AM6, AM6B, REFIT_B, DEV_OUT, SEL)
    inputs = check_fresh5_inputs()
    rows = load_fresh5()
    ids = [r["row_id"] for r in rows]
    g = F.as_idx([r["gold"] for r in rows])
    raw = {t: read_scores(t, "fresh_test5", ids) for t in ("B", "Bcolab", "C")}
    sel = json.loads(SEL.read_text())["decider_cut_offs"][f"B+{VARIANT}"]
    dev = json.loads(DEV_OUT.read_text())["systems"]
    cut = {"B|shipped": {"t_lo": sel["t_lo"], "t_hi": sel["t_hi"], "source": rel(SEL)},
           "B|refit": {**{k: json.loads(REFIT_B.read_text())["new_cutoffs"][k] for k in ("t_lo", "t_hi")},
                       "source": rel(REFIT_B)},
           **{t: {**{k: dev[t]["cutoffs"][k] for k in ("t_lo", "t_hi")}, "source": rel(DEV_OUT)} for t in CANDS}}
    svals = {t: S.decider_score(raw[t]) for t in raw}
    P = {k: S.apply_h_rows(S.decider_idx(svals[SYS[k]], c["t_lo"], c["t_hi"]), rows, CATS) for k, c in cut.items()}
    P["rules_full"] = F.as_idx([r["rules_full"] for r in rows])
    pairs = [("C", "B|refit"), ("Bcolab", "B|refit"), ("C", "Bcolab"), ("B|refit", "B|shipped"),
             ("C", "B|shipped"), ("Bcolab", "B|shipped")]
    sl = slices5(rows)
    res_sl = {k: score_slice(rows, g, P, m, pairs) for k, m in sl.items()}
    ep = res_sl["eligibility_pass"]
    bar = verdict(ep, "C", "B|refit")
    res = {
        "set": "fresh_test5", "amendments": [rel(AM6), rel(AM6B)],
        "systems": {k: f"{SYS.get(k, k)} + {VARIANT} + decider, cut-offs {c['t_lo']:.6f} / {c['t_hi']:.6f} ({c['source']})"
                    for k, c in cut.items()} | {"rules_full": "rules.label on the card (mark_thin)"},
        "cutoffs": cut,
        "bar": {"rule": "eligibility-pass slice: C minus refit-B on Keep-F1, 95% two-way bootstrap lower bound "
                        "(2.5th percentile, unrounded) above 0; C minus refit-B on macro-F1, lower bound above -0.03",
                "n": ep["n"], "postings": ep["postings"], "gold": ep["gold"], **bar},
        "B_colab_beside_bar (no pass claim)": verdict(ep, "Bcolab", "B|refit"),
        "trainer_effect (B-colab - refit-B, B-colab - B shipped)": {
            k: ep["paired"][k] for k in ("Bcolab - B|refit", "Bcolab - B|shipped")},
        "relabel_effect (C - B-colab)": ep["paired"]["C - Bcolab"],
        "agreement_subset (reported, not for pass)": {
            "slice": "agreed_eligibility_pass", "n": res_sl["agreed_eligibility_pass"]["n"],
            "gold": res_sl["agreed_eligibility_pass"]["gold"],
            "C_vs_refit_B": verdict(res_sl["agreed_eligibility_pass"], "C", "B|refit"),
            "Bcolab_vs_refit_B": verdict(res_sl["agreed_eligibility_pass"], "Bcolab", "B|refit")},
        "eligibility_pass_recomputed_vs_design": {
            "recomputed": int(sum(r["eligibility_pass"] for r in rows)),
            "design": int(sum(r["eligibility_pass_design"] for r in rows)),
            "differ": [r["row_id"] for r in rows if r["eligibility_pass"] != r["eligibility_pass_design"]]},
        "inference_history": INFERENCE_HISTORY,
        "draws": DRAWS, "seed": SEED5, "slices": res_sl,
        "cannot_ship": "C and B-colab were trained on Colab; neither can ship as the contest model without an Adaption "
                       "retrain (amendment 6b cannot_ship). B ships as is.",
        "inputs_sha256": {**inputs, **{rel(p): sha(p) for p in (PINS, AM6, AM6B, REFIT_B, DEV_OUT, SEL,
                                                                 ROOT / "scripts/fresh_test5_score.py")}},
        "score_files": {t: {"path": rel(score_file(t, "fresh_test5")), "sha256": sha(score_file(t, "fresh_test5"))}
                        for t in raw},
    }
    ROWS5_OUT.parent.mkdir(parents=True, exist_ok=True)
    with ROWS5_OUT.open("w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"row_id": r["row_id"], "posting_id": r["posting_id"], "gold": r["gold"],
                                "label_A": r["label_A"], "label_B": r["label_B"], "agreed": r["agreed"],
                                "eligibility_pass": r["eligibility_pass"],
                                "eligibility_pass_design": r["eligibility_pass_design"], "archetype": r["archetype"],
                                "source_set": r["source_set"], "employer": r["employer"],
                                "employer_in_B_training": r["employer_in_B_training"],
                                "verdict": {k: r["verdict"][k] for k in ("rule", "hard_cats", "salary_soft")},
                                "rules_full": r["rules_full"],
                                "scores": {t: [float(x) for x in raw[t][i]] for t in raw},
                                "s": {t: float(svals[t][i]) for t in svals},
                                "systems": {k: LABELS[int(v[i])] for k, v in P.items()}}) + "\n")
    res["per_row_file"] = {"path": rel(ROWS5_OUT), "sha256": sha(ROWS5_OUT)}
    RESULT.write_text(json.dumps(res, indent=1) + "\n")
    RESULT.with_suffix(".md").write_text(markdown(res))
    print(markdown(res))


def markdown(res: dict) -> str:
    def f(x, d=3):
        return "-" if x is None else f"{x:.{d}f}"

    def row(name: str, c: dict) -> str:
        return (f"| {name} | {f(c['keep_f1'])} [{f(c['keep_f1_ci']['low'])}, {f(c['keep_f1_ci']['high'])}] | "
                f"{f(c['macro_f1'])} [{f(c['macro_f1_ci']['low'])}, {f(c['macro_f1_ci']['high'])}] | "
                f"{f(c['keep_recall'])} ({c['keep_found']}/{c['keep_gold']}) | {f(c['keep_precision'])} | "
                f"{c['wrong_keeps']} | {f(c['explore_recall'])} | {f(c['skip_recall'])} |")

    b = res["bar"]
    out = ["# Fresh test 5 (amendments 6 and 6b), scored once", "",
           f"**Bar (C vs refit-B, eligibility-pass slice, n {b['n']}, {b['postings']} postings, gold {b['gold']}):** "
           f"Keep-F1 diff {f(b['keep_f1_diff']['point'], 4)} [{f(b['keep_f1_diff']['low'], 4)}, "
           f"{f(b['keep_f1_diff']['high'], 4)}] (lower bound above 0: {b['keep_f1_lower_bound_above_0']}); "
           f"macro-F1 diff {f(b['macro_f1_diff']['point'], 4)} [{f(b['macro_f1_diff']['low'], 4)}, "
           f"{f(b['macro_f1_diff']['high'], 4)}] (lower bound above -0.03: {b['macro_f1_lower_bound_above_-0.03']}). "
           f"**{'PASS' if b['passed'] else 'NO PASS'}.**", "",
           res["cannot_ship"], ""]
    for name in ("eligibility_pass", "agreed_eligibility_pass", "all_rows"):
        s = res["slices"][name]
        out += [f"## {name} (n {s['n']}, {s['postings']} postings, gold {s['gold']})", "",
                "| system | Keep-F1 [95%] | macro-F1 [95%] | Keep recall (found/gold) | Keep precision | wrong Keeps | "
                "Explore recall | Skip recall |", "|---|---|---|---|---|---|---|---|"]
        out += [row(k, c) for k, c in s["systems"].items()]
        out += ["", "| difference | Keep-F1 [95%] | macro-F1 [95%] |", "|---|---|---|"]
        for k, d in s["paired"].items():
            out.append(f"| {k} | {f(d['keep_f1_diff']['point'], 4)} [{f(d['keep_f1_diff']['low'], 4)}, "
                       f"{f(d['keep_f1_diff']['high'], 4)}] | {f(d['macro_f1_diff']['point'], 4)} "
                       f"[{f(d['macro_f1_diff']['low'], 4)}, {f(d['macro_f1_diff']['high'], 4)}] |")
        out.append("")
    out += ["Two-way bootstrap (postings, then rows within them), 2,000 draws, seed 20261022, one draw set per slice. "
            "Per-row file: " + res["per_row_file"]["path"], ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dev").set_defaults(fn=cmd_dev)
    sub.add_parser("test5").set_defaults(fn=cmd_test5)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
