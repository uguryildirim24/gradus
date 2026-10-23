"""Cycle-3 pick on fresh tests 1-3 pooled (eval/protocol_amendment_5.json selection, 5b decider readout and floors).

  fit_select_cycle3.py fetch TAG      TAG in A, B: read run TAG's state file on oci-pi (data/adaption/main_4b_TAG.json),
                                      require a terminal run with the expected run id and an archive, sha256 it there,
                                      copy it to data/models/main_4b_c3/, sha256 it again
                                      -> data/models/c3_eval/TAG/archive.json (+ the state file's copy beside it)
  fit_select_cycle3.py infer TAG      TAG on fresh tests 1-3 on one Colab L4 session (colab_eval.py under the watchdog,
                                      bf16, batch 1, the same first-token float32 label scores as every earlier run):
                                      A on h's prompt files, B on build4's; every prompt file is checked against
                                      eval/cycle3_eval_inputs.json and the archive against archive.json first; a test
                                      whose score file exists is skipped -> data/models/c3_eval/TAG/TAG_fresh_test*.json
  fit_select_cycle3.py select         the twelve systems, the decider cut-offs, the pick and base's cut-offs
                                      -> eval/selection_cycle3.json (write-once) and data/models/c3_eval/selection_rows.jsonl
  fit_select_cycle3.py select --dry-run [--models h,...]
                                      the same on whichever candidate models have score files, written to
                                      data/models/c3_eval/dryrun/ only (never eval/)

Definitions (amendments 5 and 5b, nothing added):
  rows       fresh tests 1, 2, 3 (592 + 371 + 437 = 1,400), each checked as its scoring command checked it and loaded
             with the rules in effect now (diag_cycle3_h2n.check_set: fit_eval_score.load_fresh, strict design check on
             tests 1 and 2, none on test 3); fresh test 4 is never read.
  label      argmax: fit_eval_score.read_scores (evalstats.pick on the float32 first-token scores, ties Skip > Explore
             > Keep). decider: s = P(Keep) + 0.5 P(Explore), P = evalstats.softmax(scores, temperature 1); s >= t_hi ->
             Keep, s < t_lo -> Skip, else Explore.
  H''        hybrid_rules.hybrid(label, rule_verdict, cats, salary_cap=True) with cats = HP_CATS - {employment_type}
             (with work_mode) or HP_CATS - {employment_type, work_mode} (without), applied after either readout.
  cut-offs   per model x H'' variant on the 1,400 pooled rows: grid = np.quantile(s, [0, 0.005, ..., 1]) (numpy's
             default linear method) of that model's s on those rows; every pair t_lo <= t_hi of grid values (grid
             index i <= j); the pair with the highest pooled macro-F1 (after H'') among pairs with exact recall >= 0.60
             Keep, >= 0.25 Explore, >= 0.25 Skip; ties (macro-F1 rounded to 6 decimals, as amendment 5's system ties):
             higher Keep recall, then lower t_lo, then lower t_hi. If no pair reaches those floors: the same among
             pairs with >= 0.25 recall on every class, flagged "fallback_0.25".
  pick       among {h, A, B} x {H'' without work_mode, H'' with} x {argmax, decider}: systems with exact pooled recall
             >= 0.60 Keep, 0.25 Explore, 0.25 Skip; best = the highest pooled macro-F1 rounded to 6 decimals; tied =
             those within 0.005 of best; the tied system first in the order (variant without work_mode first, then
             model h, A, B, then argmax before decider) wins. If none clears the floors: amendment 5's fallback, the same
             rule over every system, recorded as no system cleared the floors.
  base       base (the untrained Qwen3.5-4B, h's prompts) gets its decider cut-offs fitted the same way for both H''
             variants (amendment 5b: test 4's base comparator when the pick is a decider); never a candidate.
  numbers    macro-F1 = evalstats.macro_f1_of, accuracy, per-class precision and recall, unsafe Keeps (predicted Keep,
             gold Explore or Skip) through fit_eval_score.metrics; 95% intervals from fit_eval_score.two_way_draws per
             test (seed SEED + k for test k, k = 0, 1, 2), concatenated for the pooled rows (diag_cycle3_cpu.pooled_draws:
             no posting crosses tests); the pick reads the exact unrounded values, not the 4-decimal ones.

Loads no model on the Mac.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import fit_eval_score as F  # noqa: E402
from hybrid_rules import HP_CATS, hybrid  # noqa: E402
from matcher.evalstats import LABELS, pick, softmax  # noqa: E402

TESTS = ("fresh_test", "fresh_test2", "fresh_test3")
CANDIDATES = ("h", "A", "B")  # tie order among models
REF = ("base",)
VARIANTS = {"H''_without_work_mode": frozenset(HP_CATS - {"employment_type", "work_mode"}),
            "H''_with_work_mode": frozenset(HP_CATS - {"employment_type"})}  # dict order = tie order
READOUTS = ("argmax", "decider")  # tie order
FLOORS = {"Keep": 0.60, "Explore": 0.25, "Skip": 0.25}
FALLBACK_FLOORS = {"Keep": 0.25, "Explore": 0.25, "Skip": 0.25}
MARGIN = 0.005
GRID_LEVELS = np.linspace(0.0, 1.0, 201)
RUN_IDS = {"A": "cf083e47-6000-44d3-96c0-2d5a0c6613f6", "B": "8dc3d64a-cd0f-4807-a6e9-4eadb542ed3d"}
V4 = ROOT / "data/models/v4_eval"
C3E = ROOT / "data/models/c3_eval"
ARCH_DIR = ROOT / "data/models/main_4b_c3"
PINS = ROOT / "eval/cycle3_eval_inputs.json"
H2N = ROOT / "eval/diag_cycle3_h2n.json"  # records the sha256 of h's and base's score files on tests 1-3
AMEND = ("eval/protocol_amendment_5.json", "eval/protocol_amendment_5b.json")
RESULT = ROOT / "eval/selection_cycle3.json"
ROWS_OUT = C3E / "selection_rows.jsonl"
OCI = "oci-pi"
OCI_DIR = "matcher-train"
INPUT_DIR = {"A": "h", "B": "B", "h": "h", "base": "h"}


# ---------------------------------------------------------------- files


def score_path(tag: str, test: str) -> Path:
    if tag in ("h", "base"):
        return V4 / test / f"{tag}_run/{tag}_{test}.json"
    return C3E / tag / f"{tag}_{test}.json"


def input_path(tag: str, test: str) -> Path:
    return ROOT / f"data/built/fit/eval_inputs/{INPUT_DIR[tag]}/{test}.jsonl"


def check_inputs(tag: str) -> dict[str, str]:
    """TAG's prompt files are the ones eval/cycle3_eval_inputs.json pins (committed unchanged)."""
    if not F.committed(PINS):
        raise SystemExit(f"{PINS.relative_to(ROOT)} is not committed unchanged")
    pins = json.loads(PINS.read_text())["A" if INPUT_DIR[tag] == "h" else "B"]
    out = {}
    for t in TESTS:
        p = input_path(tag, t)
        k = str(p.relative_to(ROOT))
        if F.sha256(p) != pins[k]:
            raise SystemExit(f"{k} is not the prompt file {PINS.name} pins")
        out[k] = pins[k]
    return out


def check_ref_scores() -> dict[str, dict]:
    """h's and base's score files on tests 1-3 are the ones eval/diag_cycle3_h2n.json read."""
    rec = json.loads(H2N.read_text())["score_files"]
    out = {}
    for t in TESTS:
        for tag in ("h", "base"):
            p = score_path(tag, t)
            if rec[t][tag]["path"] != str(p.relative_to(ROOT)) or F.sha256(p) != rec[t][tag]["sha256"]:
                raise SystemExit(f"{p.relative_to(ROOT)} is not the {tag} score file {H2N.name} records")
            out[f"{tag}/{t}"] = {"path": rec[t][tag]["path"], "sha256": rec[t][tag]["sha256"]}
    return out


def archive_record(tag: str) -> dict:
    p = C3E / tag / "archive.json"
    if not p.exists():
        raise SystemExit(f"{p.relative_to(ROOT)} is missing: run `fit_select_cycle3.py fetch {tag}` first")
    rec = json.loads(p.read_text())
    a = ROOT / rec["local"]
    if rec["run_id"] != RUN_IDS[tag] or RUN_IDS[tag] not in a.name or F.sha256(a) != rec["sha256"]:
        raise SystemExit(f"{a.relative_to(ROOT)}: not run {tag}'s archive as fetched ({p.name})")
    return rec


# ---------------------------------------------------------------- fetch and infer


def ssh(cmd: str, limit: int = 120) -> str:
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=20", OCI, cmd], capture_output=True, text=True, timeout=limit)
    if r.returncode:
        raise SystemExit(f"ssh {OCI} {cmd!r} failed: {r.stderr[-800:]}")
    return r.stdout


def cmd_fetch(a: argparse.Namespace) -> None:
    tag = a.tag
    state = json.loads(ssh(f"cat ~/{OCI_DIR}/data/adaption/main_4b_{tag}.json"))
    r = state["runs"][tag]
    last = r.get("last") or {}
    if r.get("run_id") != RUN_IDS[tag]:
        raise SystemExit(f"run {tag}'s state holds run id {r.get('run_id')}, not {RUN_IDS[tag]}")
    if last.get("status") not in ("succeeded", "cancelled", "failed") or not r.get("archive"):
        raise SystemExit(f"run {tag}: status {last.get('status')}, archive {r.get('archive')}: not ready")
    remote = f"~/{OCI_DIR}/{r['archive']}"
    rsha = ssh(f"sha256sum {remote}").split()[0]
    ARCH_DIR.mkdir(parents=True, exist_ok=True)
    local = ARCH_DIR / Path(r["archive"]).name
    if RUN_IDS[tag] not in local.name:
        raise SystemExit(f"{local.name} does not carry run id {RUN_IDS[tag]}")
    if not local.exists() or F.sha256(local) != rsha:
        subprocess.run(["scp", "-q", f"{OCI}:{remote}", str(local)], check=True, timeout=3600)
    lsha = F.sha256(local)
    if lsha != rsha:
        raise SystemExit(f"{local.name}: sha256 on the Mac {lsha} != on {OCI} {rsha}")
    out = C3E / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / f"state_main_4b_{tag}.json").write_text(json.dumps(state, indent=1))
    rec = {"tag": tag, "run_id": r["run_id"], "status": last.get("status"), "remote": f"{OCI}:{remote}",
           "local": str(local.relative_to(ROOT)), "sha256": lsha, "bytes": local.stat().st_size,
           "sha256_on_oci_pi": rsha, "best_win_rate": last.get("best_win_rate"),
           "iterations_completed": last.get("iterations_completed"), "credits_before": r.get("credits_before"),
           "credits_after": r.get("credits_after"), "credits_spent": r.get("credits_spent"),
           "run_minutes": r.get("run_minutes"), "cancelled_for_budget": bool(r.get("cancelled_for_budget")),
           "best_hyperparams": last.get("best_hyperparams")}
    (out / "archive.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "best_hyperparams"}, indent=1))


def cmd_infer(a: argparse.Namespace) -> None:
    tag = a.tag
    check_inputs(tag)
    rec = archive_record(tag)
    out = C3E / tag
    todo = [t for t in TESTS if not score_path(tag, t).exists()]
    if not todo:
        print(f"{tag} already scored on fresh tests 1-3")
        return
    log = out / f"eval_{tag}.log"
    if log.exists():
        n = 1
        while (out / f"eval_{tag}_try{n}.log").exists():
            n += 1
        log.rename(out / f"eval_{tag}_try{n}.log")
    cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/colab_eval.py"), f"matcher-c3{tag.lower()}-eval",
           "--base-model", "Qwen/Qwen3.5-4B", "--model", f"{tag}={rec['local']}"]
    for t in todo:
        cmd += ["--dev", str(input_path(tag, t).relative_to(ROOT))]
    cmd += ["--out", str(out.relative_to(ROOT)), "--dtype", "bfloat16", "--batch", "1", "--max-min", str(a.max_min)]
    with log.open("w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"{tag} {todo} exit {rc}; missing now: {[t for t in TESTS if not score_path(tag, t).exists()]}")


# ---------------------------------------------------------------- loading


def load_rows() -> list[dict]:
    from diag_cycle3_h2n import check_set
    rows = []
    for k, t in enumerate(TESTS):
        for r in check_set(t):
            rows.append(r | {"test": t, "test_k": k})
    return rows


def read_raw(path: Path, ids: list[str]) -> np.ndarray:
    """rows x (Keep, Explore, Skip) float scores in `ids` order, after fit_eval_score.read_scores' checks; each row's
    evalstats.pick must equal read_scores' label."""
    labels = F.read_scores(path, ids)
    d = json.loads(Path(path).read_text())
    by = {r.get("pair_id", r.get("row_id")): [float(x) for x in r["scores"]] for r in d["rows"]}
    S = np.array([by[i] for i in ids], dtype=np.float64)
    if any(LABELS[pick(list(s))] != labels[i] for s, i in zip(S, ids)):
        raise SystemExit(f"{path}: scores and read_scores disagree")
    return S


def decider_score(S: np.ndarray) -> np.ndarray:
    P = softmax(S, 1.0)
    return P[:, 0] + 0.5 * P[:, 1]


# ---------------------------------------------------------------- readouts and H''


def argmax_idx(S: np.ndarray) -> np.ndarray:
    """evalstats.pick on every row, vectorised (an exact tie to Skip, then Explore, then Keep)."""
    return 2 - np.argmax(S[:, ::-1], axis=1)


def decider_idx(s: np.ndarray, t_lo: float, t_hi: float) -> np.ndarray:
    return np.where(s >= t_hi, 0, np.where(s < t_lo, 2, 1))


def override_masks(rows: list[dict], cats: frozenset) -> tuple[np.ndarray, np.ndarray]:
    """(hard Skip in cats, salary soft) per row: hybrid's two overrides."""
    hard = np.array([r["verdict"]["rule"] == "Skip" and bool(set(r["verdict"]["hard_cats"]) & cats) for r in rows])
    sal = np.array([bool(r["verdict"]["salary_soft"]) for r in rows])
    return hard, sal


def apply_h(lab: np.ndarray, hard: np.ndarray, sal: np.ndarray) -> np.ndarray:
    """hybrid(label, verdict, cats, salary_cap=True), vectorised over the last axis."""
    out = np.where(sal & (lab == 0), 1, lab)
    return np.where(hard, 2, out)


def apply_h_rows(lab: np.ndarray, rows: list[dict], cats: frozenset) -> np.ndarray:
    """The same through hybrid_rules.hybrid itself, row by row (the system labels written out use this)."""
    return F.as_idx([hybrid(LABELS[int(x)], r["verdict"], cats, True) for x, r in zip(lab, rows)])


# ---------------------------------------------------------------- the cut-off fit


def class_stats(g: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(macro-F1, recall per class) of each row of P (pairs x rows) against g, as evalstats.macro_f1_of."""
    f1s, rec = [], []
    for w in range(3):
        gw = g == w
        pw = P == w
        tp = (pw & gw).sum(axis=-1)
        fp = (pw & ~gw).sum(axis=-1)
        fn = (~pw & gw).sum(axis=-1)
        f1s.append(np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0))
        rec.append(tp / gw.sum() if gw.sum() else np.full(tp.shape, np.nan))
    return sum(f1s) / 3, np.stack(rec, axis=-1)


def fit_cutoffs(s: np.ndarray, g: np.ndarray, hard: np.ndarray, sal: np.ndarray) -> dict:
    grid = np.quantile(s, GRID_LEVELS)
    I, J = np.triu_indices(len(grid))
    lo, hi = grid[I], grid[J]
    P = np.where(s[None, :] >= hi[:, None], 0, np.where(s[None, :] < lo[:, None], 2, 1)).astype(np.int8)
    P = apply_h(P, hard[None, :], sal[None, :])
    f1, rec = class_stats(g, P)
    for name, floors in (("0.60/0.25/0.25", FLOORS), ("fallback_0.25", FALLBACK_FLOORS)):
        ok = np.all(rec >= np.array([floors[w] for w in LABELS]), axis=1)
        if ok.any():
            idx = np.flatnonzero(ok)
            k = min(idx, key=lambda i: (-round(float(f1[i]), 6), -float(rec[i, 0]), float(lo[i]), float(hi[i])))
            return {"t_lo": float(lo[k]), "t_hi": float(hi[k]), "grid_index_lo": int(I[k]), "grid_index_hi": int(J[k]),
                    "quantile_lo": float(GRID_LEVELS[I[k]]), "quantile_hi": float(GRID_LEVELS[J[k]]),
                    "floors_met": name, "macro_f1": float(f1[k]),
                    "recall": {w: float(rec[k, i]) for i, w in enumerate(LABELS)},
                    "pairs_tried": int(len(I)), "pairs_clearing_floors": int(ok.sum())}
    return {"t_lo": None, "t_hi": None, "floors_met": None, "pairs_tried": int(len(I)), "pairs_clearing_floors": 0}


# ---------------------------------------------------------------- the pick


def exact(g: np.ndarray, p: np.ndarray) -> dict:
    f1, rec = class_stats(g, p[None, :])
    return {"macro_f1": float(f1[0]), "recall": {w: float(rec[0, i]) for i, w in enumerate(LABELS)},
            "accuracy": float((g == p).mean())}


def clears(x: dict, floors: dict) -> bool:
    return all(x["recall"][w] >= floors[w] for w in LABELS)


def order_key(name: str) -> tuple[int, int, int]:
    m, v, r = parse(name)
    return list(VARIANTS).index(v), CANDIDATES.index(m), READOUTS.index(r)


def system_name(m: str, v: str, r: str) -> str:
    return f"{m}+{v}|{r}"


def parse(name: str) -> tuple[str, str, str]:
    m, rest = name.split("+", 1)
    v, r = rest.split("|")
    return m, v, r


def choose(ex: dict[str, dict]) -> dict:
    """Amendment 5b's pick from exact pooled numbers of the candidate systems."""
    qual = [k for k in ex if clears(ex[k], FLOORS)]
    pool, fallback = (qual, False) if qual else (list(ex), True)
    r6 = {k: round(ex[k]["macro_f1"], 6) for k in pool}
    best = max(r6.values())
    tied = sorted([k for k in pool if round(best - r6[k], 6) <= MARGIN], key=order_key)
    return {"pick": tied[0], "best_macro_f1_rounded": best, "tied_within_margin": tied,
            "systems_clearing_floors": sorted(qual, key=order_key), "no_system_cleared_floors": fallback}


# ---------------------------------------------------------------- select


def build(rows: list[dict], S: dict[str, np.ndarray]) -> dict:
    """Every system's labels (index arrays), the fitted cut-offs and the pick, from the rows and each model's raw
    scores (rows x 3, rows order). Candidates are the CANDIDATES present in S; base only gets cut-offs and labels."""
    g = F.as_idx([r["gold"] for r in rows])
    labels, cut, sval = {}, {}, {}
    for m in [*CANDIDATES, *REF]:
        if m not in S:
            continue
        s = decider_score(S[m])
        sval[m] = s
        am = argmax_idx(S[m])
        for v, cats in VARIANTS.items():
            hard, sal = override_masks(rows, cats)
            c = fit_cutoffs(s, g, hard, sal)
            cut[f"{m}+{v}"] = c
            labels[system_name(m, v, "argmax")] = apply_h_rows(am, rows, cats)
            if c["t_lo"] is not None:
                labels[system_name(m, v, "decider")] = apply_h_rows(decider_idx(s, c["t_lo"], c["t_hi"]), rows, cats)
    ex = {k: exact(g, p) for k, p in labels.items()}
    cands = {k: v for k, v in ex.items() if parse(k)[0] in CANDIDATES}
    return {"g": g, "labels": labels, "cutoffs": cut, "s": sval, "exact": ex, "choice": choose(cands)}


def report_system(k: str, p: np.ndarray, g: np.ndarray, rows: list[dict], draws_pooled, draws_test, ex: dict) -> dict:
    m = F.metrics(g, p, draws_pooled)
    out = {"pooled": {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "per_label": m["per_label"],
                      "unsafe_keep": m["unsafe_keep"], "exact": ex,
                      "clears_0.60/0.25/0.25": clears(ex, FLOORS), "clears_0.25_each": clears(ex, FALLBACK_FLOORS)},
           "per_test": {}}
    for t in TESTS:
        idx = np.array([i for i, r in enumerate(rows) if r["test"] == t])
        mt = F.metrics(g[idx], p[idx], draws_test[t])
        out["per_test"][t] = {"n": len(idx), "macro_f1": mt["macro_f1"], "accuracy": mt["accuracy"],
                              "recall": {w: mt["per_label"][w]["recall"] for w in LABELS},
                              "keep_precision": mt["per_label"]["Keep"]["precision"], "unsafe_keep": mt["unsafe_keep"]}
    return out


def table(res: dict) -> str:
    lines = [f"{'system':44s} {'mF1':>7s} {'acc':>7s} {'rK':>6s} {'rE':>6s} {'rS':>6s} {'pK':>6s} {'unsafeK':>7s} floors"]
    for k, x in res["systems"].items():
        p = x["pooled"]
        pl = p["per_label"]
        lines.append(f"{k:44s} {p['macro_f1'][0]:7.4f} {p['accuracy'][0]:7.4f} {pl['Keep']['recall']:6.3f} "
                     f"{pl['Explore']['recall']:6.3f} {pl['Skip']['recall']:6.3f} "
                     f"{pl['Keep']['precision'] if pl['Keep']['precision'] is not None else float('nan'):6.3f} "
                     f"{p['unsafe_keep']:7d} {'yes' if p['clears_0.60/0.25/0.25'] else 'no'}")
    lines.append(f"pick: {res['pick']['system']}  (no system cleared floors: {res['pick']['no_system_cleared_floors']})")
    return "\n".join(lines)


def cmd_select(a: argparse.Namespace) -> None:
    dry = a.dry_run
    models = [m for m in (a.models.split(",") if a.models else CANDIDATES)]
    if not dry and (RESULT.exists() or ROWS_OUT.exists()):
        raise SystemExit(f"{RESULT.relative_to(ROOT)} or {ROWS_OUT.relative_to(ROOT)} exists: the selection is written once")
    if not dry and sorted(models) != sorted(CANDIDATES):
        raise SystemExit("the real selection takes every candidate: h, A and B")
    for f in AMEND:
        if not F.committed(ROOT / f):
            raise SystemExit(f"{f} is not committed unchanged")
    ref_files = check_ref_scores()
    inputs = {m: check_inputs(m) for m in [*models, *REF]}
    archives = {}
    for m in models:
        if m in RUN_IDS:
            if dry and not (C3E / m / "archive.json").exists():
                continue
            archives[m] = archive_record(m)
    rows = load_rows()
    ids = [r["row_id"] for r in rows]
    S, score_files = {}, dict(ref_files)
    for m in [*models, *REF]:
        parts = []
        for t in TESTS:
            p = score_path(m, t)
            if not p.exists():
                if dry:
                    break
                raise SystemExit(f"{p.relative_to(ROOT)} is missing")
            parts.append(read_raw(p, [r["row_id"] for r in rows if r["test"] == t]))
            score_files[f"{m}/{t}"] = {"path": str(p.relative_to(ROOT)), "sha256": F.sha256(p)}
        else:
            S[m] = np.concatenate(parts)
    if dry:
        models = [m for m in models if m in S]
        print(f"dry run on {models} + base")
    B = build(rows, S)
    g = B["g"]
    draws_test = {t: F.two_way_draws([r["posting_id"] for r in rows if r["test"] == t], F.DRAWS, F.SEED + k)
                  for k, t in enumerate(TESTS)}
    offs = np.cumsum([0] + [sum(r["test"] == t for r in rows) for t in TESTS[:-1]])
    pooled = [np.concatenate([draws_test[t][i] + o for t, o in zip(TESTS, offs)]) for i in range(F.DRAWS)]
    order = sorted([k for k in B["labels"] if parse(k)[0] in CANDIDATES], key=order_key)
    base_names = [k for k in B["labels"] if parse(k)[0] in REF]
    systems = {k: report_system(k, B["labels"][k], g, rows, pooled, draws_test, B["exact"][k]) for k in order}
    base_sys = {k: report_system(k, B["labels"][k], g, rows, pooled, draws_test, B["exact"][k]) for k in base_names}
    ch = B["choice"]
    pm, pv, pr = parse(ch["pick"])
    pick_cut = B["cutoffs"][f"{pm}+{pv}"] if pr == "decider" else None
    out_rows = C3E / ("dryrun" if dry else "") / ROWS_OUT.name
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    with out_rows.open("w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"row_id": r["row_id"], "test": r["test"], "posting_id": r["posting_id"], "gold": r["gold"],
                                "eligibility_pass": r["eligibility_pass"],
                                "verdict": {k: r["verdict"][k] for k in ("rule", "hard_cats", "salary_soft")},
                                "scores": {m: [float(x) for x in S[m][i]] for m in S},
                                "s": {m: float(B["s"][m][i]) for m in B["s"]},
                                "argmax_raw": {m: LABELS[int(argmax_idx(S[m][i:i + 1])[0])] for m in S},
                                "systems": {k: LABELS[int(v[i])] for k, v in B["labels"].items()}}) + "\n")
    res = {
        "what": "cycle-3 selection on fresh tests 1-3 pooled (practice sets): amendment 5's six systems x amendment "
                "5b's two readouts, the decider cut-offs, the pick, and base's decider cut-offs for test 4's comparator"
                + (" -- DRY RUN, not the selection" if dry else ""),
        "amendments": list(AMEND),
        "amendments_sha256": {f: F.sha256(ROOT / f) for f in AMEND},
        "script": "scripts/fit_select_cycle3.py select",
        "rows": {"pooled": len(rows), **{t: sum(r["test"] == t for r in rows) for t in TESTS},
                 "gold_pooled": dict(Counter(r["gold"] for r in rows)),
                 "loaded": "diag_cycle3_h2n.check_set: fit_eval_score.load_fresh with the rules in effect, strict "
                           "design check on tests 1 and 2, none on test 3"},
        "definitions": {
            "H''_without_work_mode": sorted(VARIANTS["H''_without_work_mode"]) + ["salary cap"],
            "H''_with_work_mode": sorted(VARIANTS["H''_with_work_mode"]) + ["salary cap"],
            "override": "hybrid_rules.hybrid(label, rule_verdict, cats, salary_cap=True) after either readout",
            "argmax": "fit_eval_score.read_scores: evalstats.pick on the float32 first-token scores",
            "decider": "s = P(Keep) + 0.5 P(Explore), P = evalstats.softmax(scores, 1.0); s >= t_hi Keep, s < t_lo "
                       "Skip, else Explore",
            "grid": "np.quantile(s over the 1,400 pooled rows, linspace(0, 1, 201)), numpy's default linear method; "
                    "pairs t_lo = grid[i], t_hi = grid[j], i <= j (20,301 pairs)",
            "cut_off_rule": "highest pooled macro-F1 (after H'') among pairs with exact recall >= 0.60 Keep, 0.25 "
                            "Explore, 0.25 Skip; ties on macro-F1 rounded to 6 decimals: higher Keep recall, lower t_lo, "
                            "lower t_hi; else the same among pairs with >= 0.25 on every class (floors_met "
                            "'fallback_0.25')",
            "pick_rule": "exact pooled recall >= 0.60 Keep, 0.25 Explore, 0.25 Skip; best pooled macro-F1 rounded to 6 "
                         "decimals; tied if best - value (both rounded, difference rounded to 6) <= 0.005; first tied "
                         "in order H''_without_work_mode, H''_with_work_mode; h, A, B; argmax, decider. None clearing: "
                         "the same rule over all twelve (amendment 5's fallback), flagged",
            "bootstrap": f"{F.DRAWS} two-way draws per test (fit_eval_score.two_way_draws, seed {F.SEED} + k for test "
                         "k = 0, 1, 2), concatenated for pooled; intervals are reported, the pick reads exact values",
        },
        "pick": {"system": ch["pick"], "model": pm, "eligibility_layer": pv, "readout": pr,
                 "cut_offs": None if pick_cut is None else {"t_lo": pick_cut["t_lo"], "t_hi": pick_cut["t_hi"],
                                                            "floors_met": pick_cut["floors_met"]},
                 "pooled_macro_f1_exact": B["exact"][ch["pick"]]["macro_f1"],
                 "best_macro_f1_rounded": ch["best_macro_f1_rounded"], "tied_within_margin": ch["tied_within_margin"],
                 "systems_clearing_floors": ch["systems_clearing_floors"],
                 "no_system_cleared_floors": ch["no_system_cleared_floors"]},
        "decider_cut_offs": {k: v for k, v in B["cutoffs"].items() if k.split("+")[0] in CANDIDATES},
        "base_decider_cut_offs": {k: v for k, v in B["cutoffs"].items() if k.split("+")[0] in REF},
        "base_comparator_for_test_4": {
            "rule": "amendment 5b: base + the pick's H'' variant, read with the pick's readout",
            "system": system_name("base", pv, pr),
            "cut_offs": None if pr == "argmax" else {k: B["cutoffs"][f"base+{pv}"][k] for k in ("t_lo", "t_hi",
                                                                                               "floors_met")}},
        "systems": systems,
        "base_systems_reported_not_candidates": base_sys,
        "score_files": score_files,
        "input_files": inputs,
        "input_pins": {"file": str(PINS.relative_to(ROOT)), "sha256": F.sha256(PINS)},
        "archives": archives,
        "per_row_file": {"path": str(out_rows.relative_to(ROOT)), "sha256": F.sha256(out_rows)},
    }
    dest = C3E / "dryrun/selection_cycle3.json" if dry else RESULT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1) + "\n")
    print(table(res))
    print(f"wrote {dest.relative_to(ROOT)} and {out_rows.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("tag", choices=sorted(RUN_IDS))
    f.set_defaults(fn=cmd_fetch)
    i = sub.add_parser("infer")
    i.add_argument("tag", choices=sorted(RUN_IDS))
    i.add_argument("--max-min", type=int, default=70)
    i.set_defaults(fn=cmd_infer)
    s = sub.add_parser("select")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--models", help="dry run only: comma-separated candidates to include")
    s.set_defaults(fn=cmd_select)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
