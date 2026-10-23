"""The rules-override hybrid for a model's Keep / Explore / Skip: score it from stored per-row outputs (no model runs).

Pre-registered hybrid H (2026-10-20): when the rules on Adaption's extracted card say Skip for a hard-filter reason
(work authorization, enrollment status or level, degree level), output Skip; otherwise the model's label.
H' (extended, test-informed: the diagnosis review looked at a kind-based version of it on the real test) also
overrides on degree field, employment type, location and work mode Skips, and caps a model Keep at Explore when the
rules flag pay under the candidate's salary floor (the rules never Skip on salary: it is a soft reason, so this is a
cap, not a Skip override). Work mode (fix cycle 2, eval/fix_cycle2.json): a work mode the profile rules out is a hard
rules Skip (rules._check_location) except on a thin posting, so H' Skips it; its reasons come from the location check
and are split off as their own category, so the override report shows its hit rate apart from place mismatches.

  hybrid_rules.py test --pred RESULT.json [--out DIR]
      The real test (decidable rows of data/built/real_test/test.jsonl): the model, the rules alone, H and H'.
      Accuracy, macro-F1, posting-resampled 95% intervals, paired H - model and H' - model, side headlines
      (evalstats.side_headlines plus build_real_test.CORE_KINDS), and how often each override is right.
      Report H once; H' is printed marked test-informed. Writes DIR/test.json (summary and per-row labels).
  hybrid_rules.py checked-dev [--pred RESULT.json] [--out DIR]
      The real-dev rows (data/built/real_train/real_dev.jsonl) of the postings with a hand-checked card, labelled by
      rules.label on the checked card (as build_real_train.checked_agreement does), the only dev labels that are not
      the extracted-card rules themselves. Without --pred (no model outputs on real dev yet): each override's
      precision against the checked label, which bounds what H can lose against any model, and the checked Skips
      the extracted card misses. With --pred: H and H' scored there too. Also the rows whose checked label the shown
      posting states (build_real_test.posting_misses). Writes DIR/checked_dev.json. --checked picks the gold cards:
      raw (default) is fresh_checked_cards.jsonl as read, which is not corrected; hand_truth and clean are
      build_real_train.CHECKED_SOURCES.

  hybrid_rules.py checked-dev-v2 [--form builder|redrawn] --pred TAG=RESULT.json [--pred ...] [--ref c]
      [--drop-posting BASE_RQ_ID] [--out DIR]
      The clean checked real dev v2 (clean_real_checked dev-v2; builder form is the primary real-posting selection
      set). Gold is the file's label; the candidate is the row's own from clean_checked rows_step1.jsonl (builder)
      or rows.jsonl (redrawn); H and H' read the rules on Adaption's extracted card. Per model: raw, H and H'
      accuracy, macro-F1, Keep recall and Keep recall at ask gap over 4 (select_cardcheck.ask_gaps), each with a
      posting-resampled 95% interval, plus the rules alone; paired H - raw, H' - raw, H' - H per model and
      model - ref for each of raw, H, H'. Writes DIR/checked_dev_v2_<form>.json. --drop-posting BASE_RQ_ID leaves
      a posting's rows out (e.g. one whose labels the data review said not to select on); use its own --out.

RESULT.json is a local_eval/colab_eval output (rows with pair_id and pred). Nothing here calls Adaption or loads a model.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from matcher.evalstats import WORD, cluster_ci, macro_f1_of, side_headlines, subset_f1  # noqa: E402
# The verdict and the override live in matcher.hybrid (numpy-free, so the demo server imports the same code).
from matcher.hybrid import H_CATS, HP_CATS, SALARY_PREFIX, VARIANTS, category, hybrid, overrides, rule_verdict  # noqa: E402,F401
from matcher.rules import label  # noqa: E402
from matcher.schema import CandidateCard, RequirementCard  # noqa: E402

TEST = ROOT / "data/built/real_test/test.jsonl"
TEST_ROWS = ROOT / "data/built/real_test/rows.jsonl"
CARDS = ROOT / "data/built/cards/extracted_real_cards_v2.jsonl"
CHECKED = ROOT / "data/built/audit/fresh_checked_cards.jsonl"
REAL_DEV = ROOT / "data/built/real_train/real_dev.jsonl"
REAL_TRAIN_ROWS = ROOT / "data/built/real_train/rows.jsonl"
OUT = ROOT / "data/models/main_4b/hybrid"

def read_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- metrics


def scores(gold: list[str], pred: list[str], groups: list[str], draws: int = 1000) -> dict:
    g, p = np.array(gold), np.array(pred)
    acc = cluster_ci(groups, lambda i: float((g[i] == p[i]).mean()), draws)
    f1 = cluster_ci(groups, lambda i: macro_f1_of(g[i], p[i]), draws)
    keep = g == "Keep"
    return {"n": len(g), "postings": len(set(groups)),
            "accuracy": [round(x, 4) for x in acc], "macro_f1": [round(x, 4) for x in f1],
            "keep_recall": round(float((p[keep] == "Keep").mean()), 4) if keep.any() else None,
            "unsafe_keep": int(((g != "Keep") & (p == "Keep")).sum())}


def paired(gold: list[str], a: list[str], b: list[str], groups: list[str], draws: int = 1000) -> dict:
    """a minus b on the same rows, the interval from resampling postings (same draws for both)."""
    g, pa, pb = np.array(gold), np.array(a), np.array(b)
    acc = cluster_ci(groups, lambda i: float((g[i] == pa[i]).mean() - (g[i] == pb[i]).mean()), draws)
    f1 = cluster_ci(groups, lambda i: macro_f1_of(g[i], pa[i]) - macro_f1_of(g[i], pb[i]), draws)
    return {"accuracy_diff": [round(x, 4) for x in acc], "macro_f1_diff": [round(x, 4) for x in f1],
            "rows_fixed": int(((pa == g) & (pb != g)).sum()), "rows_broken": int(((pa != g) & (pb == g)).sum())}


def override_report(rows: list[dict], name: str) -> dict:
    """On the rows where the override fires: how many, how often the gold label is the one it outputs, by category."""
    cats, salary = VARIANTS[name]
    fired = [r for r in rows if overrides(r["model"], r["verdict"], cats, salary)]
    right = [r for r in fired if r[name] == r["gold"]]
    by_cat: dict[str, Counter] = {}
    for r in fired:
        hit = sorted(set(r["verdict"]["hard_cats"]) & cats) or ["salary_cap"]
        for c in hit:
            by_cat.setdefault(c, Counter())[r[name] == r["gold"]] += 1
    return {"fired": len(fired), "gold_agrees": len(right),
            "changed_model_label": sum(r[name] != r["model"] for r in fired),
            "by_category": {c: {"fired": sum(v.values()), "gold_agrees": v[True]} for c, v in sorted(by_cat.items())},
            "wrong": [{"pair_id": r["pair_id"], "variant_kind": r.get("variant_kind"), "gold": r["gold"],
                       "out": r[name], "rule_cats": r["verdict"]["hard_cats"]} for r in fired if r[name] != r["gold"]]}


def model_preds(path: Path) -> dict[str, str]:
    return {r["pair_id"]: r["pred"] for r in json.loads(Path(path).read_text())["rows"]}


def load_cards(path: Path) -> dict[str, RequirementCard]:
    out = {}
    for r in read_jsonl(path):
        try:
            out[r["id"]] = RequirementCard.from_dict(r["card"])
        except (ValueError, TypeError, KeyError):
            pass
    return out


def attach(rows: list[dict], model: dict[str, str] | None) -> list[dict]:
    """Each row gets the model label (the rules' own when there is no model), H and H'."""
    for r in rows:
        r["model"] = model[r["pair_id"]] if model else r["verdict"]["rule"]
        for name, (cats, salary) in VARIANTS.items():
            r[name] = hybrid(r["model"], r["verdict"], cats, salary)
    return rows


def macro_f1_present(gold: list[str], pred: list[str]) -> float:
    """Macro-F1 over the labels found in gold or pred (scikit-learn's default), the averaging the profile-only
    shortcut numbers in data/built/real_train/checks.json use; evalstats.macro_f1_of always averages all three."""
    present = sorted(set(gold) | set(pred))
    f1s = []
    for w in present:
        tp = sum(g == w and p == w for g, p in zip(gold, pred))
        fp = sum(g != w and p == w for g, p in zip(gold, pred))
        fn = sum(g == w and p != w for g, p in zip(gold, pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return round(sum(f1s) / len(f1s), 4) if f1s else float("nan")


def side(rows: list[dict], key: str, core_kinds: tuple[str, ...]) -> dict:
    from matcher.evalstats import RESUME_KINDS
    sub = [{"gold": r["gold"], "pred": r[key], "variant_kind": r["variant_kind"], "base_rq_id": r["base_rq_id"]}
           for r in rows]
    out = side_headlines(sub)
    out["core_kinds"] = subset_f1([s for s in sub if s["variant_kind"] in core_kinds])
    for name, kinds in (("resume_kinds", RESUME_KINDS), ("core_kinds", core_kinds)):
        part = [s for s in sub if s["variant_kind"] in kinds]
        if out.get(name):
            out[name]["macro_f1_present_labels"] = macro_f1_present([s["gold"] for s in part],
                                                                    [s["pred"] for s in part])
    out["all_macro_f1_present_labels"] = macro_f1_present([s["gold"] for s in sub], [s["pred"] for s in sub])
    return out


# ---------------------------------------------------------------- commands


def cmd_test(a: argparse.Namespace) -> dict:
    from build_real_test import CORE_KINDS
    held = {r["pair_id"]: r for r in read_jsonl(TEST)}
    model = model_preds(Path(a.pred))
    cards = load_cards(CARDS)
    rows = []
    for r in read_jsonl(TEST_ROWS):
        t = held.get(r["pair_id"])
        if t is None or t.get("decidable", True) is False or r.get("decidable", True) is False:
            continue
        if r["rq_id"] not in cards or r["pair_id"] not in model:
            continue
        v = rule_verdict(CandidateCard.from_dict(r["candidate"]), cards[r["rq_id"]])
        rows.append({"pair_id": r["pair_id"], "base_rq_id": r["base_rq_id"], "variant_kind": r["variant_kind"],
                     "gold": WORD[r["bucket"]], "verdict": v})
    attach(rows, model)
    gold, groups = [r["gold"] for r in rows], [r["base_rq_id"] for r in rows]
    col = {k: [r[k] for r in rows] for k in ("model", "H", "H_prime")}
    col["rules"] = [r["verdict"]["rule"] for r in rows]
    out = {"set": "real_test", "pred": str(a.pred), "rows": len(rows),
           "note": "H is pre-registered and reported once. H_prime is test-informed (a kind-based version was looked at "
                   "on this test by the diagnosis review); no choice is made from these numbers.",
           "scores": {k: scores(gold, v, groups) for k, v in col.items()},
           "paired": {"H_minus_model": paired(gold, col["H"], col["model"], groups),
                      "H_prime_minus_model": paired(gold, col["H_prime"], col["model"], groups),
                      "H_prime_minus_H": paired(gold, col["H_prime"], col["H"], groups)},
           "overrides": {k: override_report(rows, k) for k in VARIANTS},
           "side_headlines": {k: side(rows, k, CORE_KINDS) for k in ("model", "H", "H_prime")},
           "per_row": [{k: r[k] for k in ("pair_id", "base_rq_id", "variant_kind", "gold", "model", "H", "H_prime")}
                       | {"rule": r["verdict"]["rule"], "rule_cats": r["verdict"]["hard_cats"]} for r in rows]}
    out["side_headlines"]["rules"] = side([dict(r, rules=r["verdict"]["rule"]) for r in rows], "rules", CORE_KINDS)
    write(a.out, "test.json", out)
    for k, s in out["scores"].items():
        print(f"{k:8s} acc {s['accuracy'][0]:.3f} [{s['accuracy'][1]:.3f}, {s['accuracy'][2]:.3f}]  "
              f"macro-F1 {s['macro_f1'][0]:.3f} [{s['macro_f1'][1]:.3f}, {s['macro_f1'][2]:.3f}]  "
              f"Keep recall {s['keep_recall']}")
    for k, p in out["paired"].items():
        print(f"{k}: acc {p['accuracy_diff']}, macro-F1 {p['macro_f1_diff']}, fixed {p['rows_fixed']}, "
              f"broken {p['rows_broken']}")
    return out


def checked_source_cards(source: str) -> dict[str, RequirementCard]:
    """The checked cards by build_real_train.CHECKED_SOURCES name ("raw" is CHECKED, the cards as read)."""
    from build_real_train import checked_card_dicts
    out = {}
    for i, c in checked_card_dicts(source).items():
        try:
            out[i] = RequirementCard.from_dict(c)
        except (ValueError, TypeError, KeyError):
            pass
    return out


def checked_rows(dev_path: Path = REAL_DEV, train_rows: Path = REAL_TRAIN_ROWS, checked_path: Path = CHECKED,
                 cards_path: Path = CARDS, checked: dict[str, RequirementCard] | None = None) -> list[dict]:
    """Real-dev rows of postings with a checked card: gold = rules.label on the checked card, verdict on the
    extracted card (which must reproduce the row's own bucket). `checked` (cards by posting id) replaces
    checked_path's."""
    checked = load_cards(checked_path) if checked is None else checked
    cards = load_cards(cards_path)
    dev = [r for r in read_jsonl(dev_path) if r["rq_id"] in checked]
    want = {r["pair_id"] for r in dev}
    cand = {r["pair_id"]: r["candidate"] for r in read_jsonl(train_rows) if r["pair_id"] in want}
    rows = []
    for r in dev:
        c = CandidateCard.from_dict(cand[r["pair_id"]])
        v = rule_verdict(c, cards[r["rq_id"]])
        if v["rule"] != WORD[r["bucket"]]:
            raise SystemExit(f"{r['pair_id']}: extracted-card rule {v['rule']} != row bucket {r['bucket']}")
        gb, greasons, _ = label(c, checked[r["rq_id"]])
        gv = rule_verdict(c, checked[r["rq_id"]])
        rows.append({"pair_id": r["pair_id"], "base_rq_id": r["base_rq_id"], "rq_id": r["rq_id"],
                     "variant_kind": r["variant_kind"], "gold": WORD[gb], "gold_reasons": greasons,
                     "gold_cats": gv["hard_cats"], "cand": c, "verdict": v})
    return rows


def readable(rows: list[dict], checked: dict[str, RequirementCard] | None = None) -> set[str]:
    """Pair ids whose checked label the shown posting states (build_real_train.checked_agreement's test)."""
    import build_real_test as brt
    from build_real_train import load_raw
    from matcher.profile import render_profile
    checked = load_cards(CHECKED) if checked is None else checked
    raw = load_raw()
    shown = {i: brt.shown_posting(brt.posting_text(raw[i])) for i in {r["rq_id"] for r in rows}}
    return {r["pair_id"] for r in rows
            if not brt.posting_misses(r["cand"], checked[r["rq_id"]], r["gold_reasons"], render_profile(r["cand"]),
                                      shown[r["rq_id"]], hard=r["gold"] == "Skip")}


def missed(rows: list[dict], cats: frozenset) -> dict:
    """Checked Skips with a hard reason in `cats` on the checked card where the extracted-card rule does not fire
    the override: the requirement the extraction dropped, which H cannot see."""
    hit = [r for r in rows if r["gold"] == "Skip" and set(r["gold_cats"]) & cats
           and not (r["verdict"]["rule"] == "Skip" and set(r["verdict"]["hard_cats"]) & cats)]
    return {"rows": len(hit), "postings": len({r["rq_id"] for r in hit}),
            "by_checked_category": dict(Counter(c for r in hit for c in sorted(set(r["gold_cats"]) & cats))),
            "extracted_rule_said": dict(Counter(r["verdict"]["rule"] for r in hit))}


def checked_summary(rows: list[dict], with_model: bool) -> dict:
    gold, groups = [r["gold"] for r in rows], [r["base_rq_id"] for r in rows]
    out = {"rows": len(rows), "postings": len({r["rq_id"] for r in rows}),
           "gold_labels": dict(Counter(gold)),
           "extracted_rules": scores(gold, [r["verdict"]["rule"] for r in rows], groups),
           "overrides": {k: {kk: vv for kk, vv in override_report(rows, k).items() if kk != "changed_model_label"}
                         for k in VARIANTS},
           "missed_by_extraction": {k: missed(rows, VARIANTS[k][0]) for k in VARIANTS}}
    for k in VARIANTS:
        o = out["overrides"][k]
        o["most_rows_lost_vs_any_model"] = o["fired"] - o["gold_agrees"]
    # H' salary cap (Keep -> Explore) fires only on a model Keep, so without model outputs show the checked label
    # on every row the rules flag for pay under the floor: the cap can only cost a row whose checked label is Keep
    sal = [r["gold"] for r in rows if r["verdict"]["salary_soft"]]
    out["salary_floor_rows_checked_label"] = dict(Counter(sal))
    if with_model:
        col = {k: [r[k] for r in rows] for k in ("model", "H", "H_prime")}
        out["scores"] = {k: scores(gold, v, groups) for k, v in col.items()}
        out["paired"] = {"H_minus_model": paired(gold, col["H"], col["model"], groups),
                         "H_prime_minus_model": paired(gold, col["H_prime"], col["model"], groups)}
    return out


def cmd_checked_dev(a: argparse.Namespace) -> dict:
    checked = checked_source_cards(getattr(a, "checked", "raw"))
    rows = checked_rows(checked=checked)
    model = None
    if a.pred:
        model = model_preds(Path(a.pred))
        rows = [r for r in rows if r["pair_id"] in model]
    attach(rows, model)
    ok = readable(rows, checked)
    out = {"set": "real_dev, postings with a hand-checked card", "pred": a.pred,
           "checked_cards": getattr(a, "checked", "raw"),
           "model_outputs": bool(model),
           "note": None if model else "No model outputs on real dev: the model column is the extracted-card rule, so "
                                      "only the override counts mean anything. H's worst case against any model is "
                                      "most_rows_lost_vs_any_model.",
           "all": checked_summary(rows, bool(model)),
           "checked_label_readable": checked_summary([r for r in rows if r["pair_id"] in ok], bool(model))}
    write(a.out, "checked_dev.json", out)
    for part in ("all", "checked_label_readable"):
        s = out[part]
        print(f"{part}: {s['rows']} rows over {s['postings']} postings; extracted rules vs checked "
              f"acc {s['extracted_rules']['accuracy'][0]:.3f}")
        for k, o in s["overrides"].items():
            print(f"  {k}: fires {o['fired']}, checked label agrees {o['gold_agrees']}; "
                  f"missed by extraction {s['missed_by_extraction'][k]['rows']}")
    return out


# ---------------------------------------------------------------- checked real dev v2 (clean_real_checked dev-v2)

V2_FILES = {"builder": (ROOT / "data/built/real_train/checked_real_dev_v2_builder.jsonl",
                        ROOT / "data/built/real_train/clean_checked/rows_step1.jsonl"),
            "redrawn": (ROOT / "data/built/real_train/checked_real_dev_v2.jsonl",
                        ROOT / "data/built/real_train/clean_checked/rows.jsonl")}
GAP_OVER = 4


def v2_rows(form: str, cards_path: Path = CARDS) -> list[dict]:
    """The decidable rows of a checked real dev v2 file. gold = the file's label (rules.label on the clean checked
    card); the candidate is the row's own (clean_checked rows_step1.jsonl for the builder form, rows.jsonl for the
    redrawn one, dev split, kept); verdict = the rules on Adaption's extracted card, which is what H sees at run
    time; ask_gap = select_cardcheck.ask_gaps on the row's prompt (the diagnosis's measure)."""
    import select_cardcheck as sc
    dev_path, step_path = V2_FILES[form]
    dev = [r for r in read_jsonl(dev_path) if r.get("decidable", True) is not False]
    cand = {r["pair_id"]: r["candidate"] for r in read_jsonl(step_path)
            if r["split"] == "dev" and r["stage"] == "kept"}
    cards = load_cards(cards_path)
    gaps = sc.ask_gaps(dev)
    rows = []
    for r in dev:
        c = CandidateCard.from_dict(cand[r["pair_id"]])
        rows.append({"pair_id": r["pair_id"], "base_rq_id": r["base_rq_id"], "rq_id": r["rq_id"],
                     "variant_kind": r["variant_kind"], "gold": WORD[r["bucket"]], "ask_gap": gaps[r["pair_id"]],
                     "verdict": rule_verdict(c, cards[r["rq_id"]])})
    return rows


def drop_postings(rows: list[dict], ids: list[str]) -> tuple[list[dict], dict[str, int]]:
    """rows without the postings in ids (base_rq_id), and how many rows each one took; an id with no rows stops."""
    unknown = set(ids) - {r["base_rq_id"] for r in rows}
    if unknown:
        raise SystemExit(f"--drop-posting: no rows on {sorted(unknown)}")
    return [r for r in rows if r["base_rq_id"] not in set(ids)], {b: sum(r["base_rq_id"] == b for r in rows) for b in ids}


def keep_recall_of(g: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    m = (g == "Keep") & mask
    return float((p[m] == "Keep").mean()) if m.any() else float("nan")


def v2_metrics(gold: np.ndarray, gap: np.ndarray, groups: list[str]) -> dict:
    """metric name -> fn(pred, index array): accuracy, macro-F1, Keep recall, Keep recall at ask gap > GAP_OVER."""
    over = gap > GAP_OVER
    every = np.ones(len(gold), bool)
    return {"accuracy": lambda p, i: float((gold[i] == p[i]).mean()),
            "macro_f1": lambda p, i: macro_f1_of(gold[i], p[i]),
            "keep_recall": lambda p, i: keep_recall_of(gold[i], p[i], every[i]),
            f"keep_recall_gap_over_{GAP_OVER}": lambda p, i: keep_recall_of(gold[i], p[i], over[i])}


def v2_score(rows: list[dict], cols: dict[str, list[str]], ref: str | None, draws: int = 1000) -> dict:
    """Per column: each metric with a posting-resampled 95% interval. Paired (same draws): H - raw, H' - raw and
    H' - H within each model, and model - ref for each of raw, H, H'."""
    gold, gap = np.array([r["gold"] for r in rows]), np.array([r["ask_gap"] for r in rows])
    groups = [r["base_rq_id"] for r in rows]
    met = v2_metrics(gold, gap, groups)
    arr = {k: np.array(v) for k, v in cols.items()}
    rnd = lambda t: [round(x, 4) for x in t]  # noqa: E731
    out = {"scores": {k: {m: rnd(cluster_ci(groups, lambda i, f=f, p=p: f(p, i), draws)) for m, f in met.items()}
                      | {"unsafe_keep": int(((gold != "Keep") & (p == "Keep")).sum())} for k, p in arr.items()}}

    def diff(a: str, b: str) -> dict:
        pa, pb = arr[a], arr[b]
        d = {m: rnd(cluster_ci(groups, lambda i, f=f: f(pa, i) - f(pb, i), draws)) for m, f in met.items()}
        return d | {"rows_fixed": int(((pa == gold) & (pb != gold)).sum()),
                    "rows_broken": int(((pa != gold) & (pb == gold)).sum())}

    models = sorted({k.split("|")[0] for k in cols if "|" in k})
    out["paired"] = {}
    for m in models:
        out["paired"][f"{m}|H - {m}|raw"] = diff(f"{m}|H", f"{m}|raw")
        out["paired"][f"{m}|H_prime - {m}|raw"] = diff(f"{m}|H_prime", f"{m}|raw")
        out["paired"][f"{m}|H_prime - {m}|H"] = diff(f"{m}|H_prime", f"{m}|H")
        if ref and m != ref:
            for v in ("raw", "H", "H_prime"):
                out["paired"][f"{m}|{v} - {ref}|{v}"] = diff(f"{m}|{v}", f"{ref}|{v}")
    return out


def cmd_checked_dev_v2(a: argparse.Namespace) -> dict:
    preds = dict(p.split("=", 1) for p in a.pred)
    out = {"set": f"checked real dev v2, {a.form} form ({V2_FILES[a.form][0].name})",
           "gold": "the file's label (rules.label on the clean checked card)",
           "hybrid_cards": "Adaption's extracted card (extracted_real_cards_v2.jsonl), as at run time",
           "groups": "base_rq_id (posting) for every interval", "ask_gap_over": GAP_OVER, "ref": a.ref,
           "pred": preds}
    rows = v2_rows(a.form)
    if a.drop_posting:
        rows, out["dropped_postings"] = drop_postings(rows, a.drop_posting)
    cols: dict[str, list[str]] = {"rules": [r["verdict"]["rule"] for r in rows]}
    per_row = {r["pair_id"]: {k: r[k] for k in ("base_rq_id", "variant_kind", "gold", "ask_gap")}
               | {"rule": r["verdict"]["rule"], "rule_cats": r["verdict"]["hard_cats"]} for r in rows}
    overrides_by = {}
    for tag, path in preds.items():
        model = model_preds(Path(path))
        miss = [r["pair_id"] for r in rows if r["pair_id"] not in model]
        if miss:
            raise SystemExit(f"{tag}: {len(miss)} rows have no prediction in {path}: {miss[:3]}")
        attach(rows, model)
        for k in ("model", "H", "H_prime"):
            cols[f"{tag}|{'raw' if k == 'model' else k}"] = [r[k] for r in rows]
        overrides_by[tag] = {k: {kk: vv for kk, vv in override_report(rows, k).items() if kk != "wrong"}
                             for k in VARIANTS}
        for r in rows:
            per_row[r["pair_id"]] |= {f"{tag}_raw": r["model"], f"{tag}_H": r["H"], f"{tag}_H_prime": r["H_prime"]}
    gold = [r["gold"] for r in rows]
    out |= {"rows": len(rows), "postings": len({r["rq_id"] for r in rows}), "gold_labels": dict(Counter(gold)),
            "keep_rows_gap_over": sum(r["gold"] == "Keep" and r["ask_gap"] > GAP_OVER for r in rows),
            "keep_rows": gold.count("Keep")}
    out |= v2_score(rows, cols, a.ref if a.ref in preds else None, a.draws)
    out["overrides"] = overrides_by
    out["per_row"] = [{"pair_id": k} | v for k, v in per_row.items()]
    write(a.out, f"checked_dev_v2_{a.form}.json", out)
    for k, s in out["scores"].items():
        print(f"{k:14s} " + "  ".join(f"{m} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for m, v in s.items()
                                      if isinstance(v, list)))
    return out


def write(out_dir: str | None, name: str, obj: dict) -> None:
    d = Path(out_dir) if out_dir else OUT
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(obj, indent=1, default=str))
    print(f"wrote {d / name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("test")
    t.add_argument("--pred", required=True)
    t.add_argument("--out")
    c = sub.add_parser("checked-dev")
    c.add_argument("--pred")
    c.add_argument("--out")
    c.add_argument("--checked", choices=("raw", "hand_truth", "clean"), default="raw",
                   help="gold cards: raw (default) is fresh_checked_cards.jsonl as read, uncorrected; hand_truth "
                        "applies fresh_verdicts.jsonl (score_extraction.hand_truth); clean is "
                        "clean_checked/cards.jsonl (all checked dev postings)"),
    v = sub.add_parser("checked-dev-v2")
    v.add_argument("--form", choices=tuple(V2_FILES), default="builder")
    v.add_argument("--pred", action="append", default=[], help="TAG=RESULT.json, a local_eval output on that file")
    v.add_argument("--ref", default="c", help="the tag every other model is paired against")
    v.add_argument("--draws", type=int, default=1000)
    v.add_argument("--drop-posting", action="append", default=[],
                   help="base_rq_id of a posting to leave out (repeatable); the output records how many rows went")
    v.add_argument("--out")
    a = ap.parse_args()
    {"test": cmd_test, "checked-dev": cmd_checked_dev, "checked-dev-v2": cmd_checked_dev_v2}[a.cmd](a)


if __name__ == "__main__":
    main()
