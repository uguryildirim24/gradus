"""Score h, d3 and the base model as systems on dev_fit (selection) and the sealed fresh test, from stored label scores.

Every definition here is pinned by eval/protocol_amendment_1.json (written before any model output on these sets);
the files it reads are pinned by eval/fit_eval_hashes.json and checked before anything is scored.

  fit_eval_score.py dev   --score h=H_DEV.json --score d3=D3_DEV.json [--out DIR]
      dev_fit (data/built/fit/dev_fit.jsonl, rules-made gold): every system, the selection rule's pick
      -> DIR/dev_fit.json, DIR/dev_fit.md
  fit_eval_score.py fresh --score h=H_FT.json --score d3=D3_FT.json --score base=BASE_FT.json \
                          --selection eval/selection_v4.json [--out DIR]
      fresh test 1 (gold from two blind labelers and blind adjudication), amendment 1's pass rule: every system and
      baseline, paired differences and every slice -> DIR/fresh_test.json, DIR/fresh_test.md
  fit_eval_score.py select --score h=H_FT.json --score h2a=H2A_FT.json ... --out eval/selection_v5.json
      amendment 2 selection on all fresh test 1 rows: the candidate (h and h2*) whose TAG+H' has the highest macro-F1;
      within 0.01 of h, the incumbent h stays -> the selection json (commit it before scoring fresh test 2)
  fit_eval_score.py fresh2 --selection eval/selection_v5.json --score PICK=... --score base=... --score h=... \
                           --score d3=...
      fresh test 2, scored once (amendment 2): the pick is read from the committed selection json, never typed; the
      result goes to eval/fresh_test2_result.json (+ .md) and the run refuses when that file exists
  fit_eval_score.py fresh3 --selection eval/selection_<x>.json --score PICK=... --score base=... --score h=... \
                           --score d3=...
      fresh test 3, scored once (amendment 3): the selection json is committed and names the pick and
      eval/protocol_amendment_3.json ({"pick": TAG, "amendment": "eval/protocol_amendment_3.json"}, or "amendments": [...]
      holding it); every hash in eval/fresh_test3_manifest.json, eval/fresh_test3_data_hashes.json and
      eval/fresh_test3_gold_manifest.json is checked before a row is read; amendment 2's bar on the eligibility-pass
      slice recomputed with the rules in effect, the sealed-design slice beside it -> eval/fresh_test3_result.json (+ .md),
      refused when either exists
  fit_eval_score.py fresh4 --score B=... --score h=... --score base=... [--selection eval/selection_cycle3.json]
      fresh test 4, scored once (amendments 4, 4b, 4c, 5, 5b): the pick (model, H'' variant, readout, decider cut-offs)
      and base's comparator cut-offs are read from the committed selection json, never typed or refitted; every pin of
      fresh test 4 (manifest, data hashes, gold manifest, prompt files in eval/fresh_test4_eval_inputs.json) is checked
      before a row is read; the pick, h+H', rules_full and base on the same 1,139 rows; amendment 2's bar on the
      eligibility-pass slice (pick minus rules_full and minus base, two-way lower bound above 0), the same on the
      agreement subset, the new-employer slice flagged, the gold Keep count beside every Keep recall (amendment 4c)
      -> eval/fresh_test4_result.json (+ .md) and data/models/c3_eval/test4/test4_rows.jsonl, refused when any exists

No command takes the chosen system as a free argument, and none overwrites a result file that already exists.

A score file is a local_eval / colab_eval result (rows[].pair_id and rows[].scores, the float32 label scores of Keep,
Explore and Skip at the first answer token). The label is read again here from the scores with evalstats.pick (ties
to Skip, then Explore, then Keep); a stored pred that disagrees, a row missing or a row extra stops the run.

Systems: TAG+H' = hybrid_rules.hybrid(model label, rule_verdict on the Adaption-extracted card, HP_CATS, salary cap)
(H' Skips on a hard rules Skip in work authorization, enrollment, degree level, degree field, job type or location,
and caps a model Keep at Explore when the pay is under the floor; otherwise the model's label); TAG raw = the model's
label alone. Baselines: rules_full (rules.label on the extracted card), rules_skills_free (the same with must and
nice skills removed), title_match (Keep when the resume holds the posting's exact title, else Explore).
The extracted card gets rules.mark_thin on the posting body first, as in training (dev_fit's cards carry it already).

Statistics: accuracy and macro-F1 (evalstats.macro_f1_of: the mean of the three per-label F1s) with 95% percentile
intervals from a two-way bootstrap (postings drawn with replacement, then each drawn posting's rows drawn with
replacement), DRAWS draws from numpy default_rng(SEED); one set of draws per slice, shared by every system and every
paired difference on that slice. Loads no model and calls nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from hybrid_rules import HP_CATS, hybrid, rule_verdict  # noqa: E402
from matcher.evalstats import LABELS, WORD, macro_f1_of, pick  # noqa: E402
from matcher.rules import label, mark_thin  # noqa: E402
from matcher.schema import CandidateCard, RequirementCard  # noqa: E402

FIT = ROOT / "data/built/fit"
FRESH = ROOT / "data/built/fresh_test"
INPUTS = FIT / "eval_inputs"
HASHES = ROOT / "eval/fit_eval_hashes.json"
AMENDMENT = ROOT / "eval/protocol_amendment_1.json"
FRESH2 = ROOT / "data/built/fresh_test2"
AMENDMENT2 = ROOT / "eval/protocol_amendment_2.json"
FRESH2_MANIFEST = ROOT / "eval/fresh_test2_manifest.json"
FRESH2_GOLD_MANIFEST = ROOT / "eval/fresh_test2_gold_manifest.json"
FRESH2_RESULT = ROOT / "eval/fresh_test2_result.json"
INCUMBENT = "h"  # amendment 2: within TIE_MARGIN on fresh test 1 the incumbent h stays
CANDIDATE_RX = re.compile(r"^h(?:2\w*)?$")  # amendment 2 candidates: h and this cycle's h2 runs
FRESH2_REQUIRED = ("base", "h", "d3")  # scored beside the pick on fresh test 2 (amendment 2 reporting)
FRESH3 = ROOT / "data/built/fresh_test3"
AMENDMENT3 = ROOT / "eval/protocol_amendment_3.json"
AMENDMENT3_NAME = "eval/protocol_amendment_3.json"
FRESH3_MANIFEST = ROOT / "eval/fresh_test3_manifest.json"
FRESH3_DATA_HASHES = ROOT / "eval/fresh_test3_data_hashes.json"
FRESH3_GOLD_MANIFEST = ROOT / "eval/fresh_test3_gold_manifest.json"
FRESH3_RESULT = ROOT / "eval/fresh_test3_result.json"
FRESH3_REQUIRED = ("base", "h", "d3")  # amendment 3 report: h+H', d3+H' and base+H' beside the pick
OUT = ROOT / "data/models/main_4b/fit_eval"
SEED = 20261021
DRAWS = 2000
TIE_MARGIN = 0.01
SELECTION_TAGS = ("h", "d3")
TIE_PICK = "d3"  # within TIE_MARGIN on dev_fit the incumbent stays (amendment: selection.tie_break)
IDX = {w: i for i, w in enumerate(LABELS)}


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- pinned definitions


def norm(s: str) -> str:
    """Lower case, every dash as '-', every run of whitespace one space, trimmed."""
    s = re.sub(r"[‐-―−]", "-", (s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def title_in_resume(title: str, resume: str) -> bool:
    """The posting's exact title (norm) is a substring of the resume (norm)."""
    t = norm(title)
    return bool(t) and t in norm(resume)


def posting_title(posting: str) -> str:
    """The title from a posting's 'Title: ...' header line."""
    m = re.match(r"Title: (.*)", posting or "")
    return m.group(1).strip() if m else ""


KIND_ORDER = ("data", "software", "health_nursing", "lab_bio", "sales_ops")
INTERN_RX = re.compile(r"\b(intern|interns|internship|internships|co-?op|summer associate|student|apprentice|"
                       r"apprenticeship|fellow|fellowship|werkstudent|trainee|graduate program)\b")
KIND_TERMS = {
    "data": r"data scien\w*|data engineer\w*|data analy\w*|data architect\w*|analytics|machine learning|\bml\b|\bai\b|"
            r"artificial intelligence|statistic\w*|biostatistic\w*|bioinformatic\w*|computational|"
            r"business intelligence|quantitative|\bllms?\b|deep learning",
    "software": r"software|developer|devops|\bsre\b|\bcloud\b|cyber ?security|security engineer|full[- ]?stack|"
                r"front[- ]?end|back[- ]?end|\bweb\b|platform engineer|infrastructure|\bit\b|help ?desk|network|"
                r"embedded|firmware|programmer|systems administrator|\bux\b|designer",
    "health_nursing": r"\bnurs\w*|\brn\b|\blpn\b|\blvn\b|\bcna\b|nurse practitioner|pharmacist|physician|therap\w+|"
                      r"dental|dialysis|medical assistant|phlebotom\w*|sonograph\w*|radiolog\w*|respiratory|"
                      r"surgical|caregiver|patient care|home health|social worker|dietitian|paramedic|\bemt\b|"
                      r"veterinar\w*|psychiatr\w*|behavioral health|clinic\b|clinical care|care coordinator|"
                      r"patient|\baprn\b|\bnp\b|per[- ]diem|transporter",
    "lab_bio": r"scientist|research associate|laborator\w*|\blab\b|chemist\w*|assay|bioprocess|\bcell\b|molecular|"
               r"microbiolog\w*|biolog\w*|process development|formulation|toxicolog\w*|pharmacolog\w*|\bqc\b|"
               r"immunolog\w*|genomic\w*|sequencing|technologist|specimen|antibod\w*|protein|purification|"
               r"in vivo|in vitro|clean ?room|\bgmp\b|materials|physics|mass spec\w*|lc-?ms\w*|bioanaly\w*",
    "sales_ops": r"\bsales\b|account executive|account manager|business development|marketing|operations|customer|"
                 r"retail|representative|coordinator|logistics|supply chain|procurement|recruit\w*|"
                 r"talent acquisition|\bhr\b|human resources|financ\w*|accountant|accounting|administrative|"
                 r"(?<!medical )assistant|commercial|legal|counsel|compliance|consultant|partnerships?|strategy",
}
_KIND_RX = {k: re.compile(v) for k, v in KIND_TERMS.items()}
BODY_CHARS = 3000
BODY_MIN = 3


def kind_of(title: str, text: str) -> str:
    """The posting kind, recomputed from the title and the body (the pool's field tag is title-only, first match
    wins, and calls 'Senior Data Scientist' lab_bio). internship when the title names an internship, co-op,
    apprenticeship, trainee, fellowship or student role. Otherwise each kind counts its distinct terms in the title
    (norm) and in the first BODY_CHARS characters of the body (norm): the kind with the most title terms wins, ties
    broken by body terms, then by KIND_ORDER; with no title term, the kind with the most body terms if it has at least
    BODY_MIN, else other."""
    t, b = norm(title), norm(text)[:BODY_CHARS]
    if INTERN_RX.search(t):
        return "internship"
    score = {k: (len(set(rx.findall(t))), len(set(rx.findall(b)))) for k, rx in _KIND_RX.items()}
    best = max(KIND_ORDER, key=lambda k: (score[k][0], score[k][1], -KIND_ORDER.index(k)))
    if score[best][0] > 0:
        return best
    best = max(KIND_ORDER, key=lambda k: (score[k][1], -KIND_ORDER.index(k)))
    return best if score[best][1] >= BODY_MIN else "other"


def skills_free(card: RequirementCard) -> RequirementCard:
    return replace(card, must_skills=(), nice_skills=())


def eligibility(cand: CandidateCard, card: RequirementCard) -> tuple[dict, bool]:
    """(rule_verdict, eligibility pass): pass = no hard rules Skip in HP_CATS and no salary cap (H' leaves the
    model's label alone on exactly these rows)."""
    v = rule_verdict(cand, card)
    hard = v["rule"] == "Skip" and bool(set(v["hard_cats"]) & HP_CATS)
    return v, not hard and not v["salary_soft"]


def blank_profile(p: dict) -> bool:
    """A form field the fresh test left blank by draw: no salary floor, or no job type wanted."""
    return p.get("salary_floor") is None or not p.get("employment_types_wanted")


# ---------------------------------------------------------------- rows


def base_row(rid: str, pid: str, gold: str, cand: CandidateCard, card: RequirementCard, title: str, resume: str) -> dict:
    v, ok = eligibility(cand, card)
    return {"row_id": rid, "posting_id": pid, "gold": gold, "verdict": v, "eligibility_pass": ok,
            "rules_full": WORD[label(cand, card)[0]], "rules_skills_free": WORD[label(cand, skills_free(card))[0]],
            "title_in_resume": title_in_resume(title, resume)}


# Rules fixes this cycle (fix cycle 2, eval/fix_cycle2.json) that may move a row out of the eligibility-pass slice the
# sealed design flag recorded. Amendment 2 defines the slice as recomputed from profile and card with the rules in
# effect, the same for every system; a row may leave the design slice only through one of these hard categories.
DESIGN_FLAG_MOVES = {"work_mode"}


def check_design_flag(rid: str, x: dict, d: dict) -> None:
    """The recomputed eligibility pass may differ from the sealed design flag only where a fix-cycle-2 category made a
    design-pass row a hard Skip; any other difference stops the run."""
    if x["eligibility_pass"] == d["eligibility_pass"]:
        return
    cats = set(x["verdict"]["hard_cats"]) & HP_CATS if x["verdict"]["rule"] == "Skip" else set()
    if not (d["eligibility_pass"] and not x["eligibility_pass"] and cats and cats <= DESIGN_FLAG_MOVES):
        raise SystemExit(f"{rid}: eligibility pass differs from the sealed design flag other than by a "
                         f"{sorted(DESIGN_FLAG_MOVES)} hard Skip (design {d['eligibility_pass']}, now "
                         f"{x['eligibility_pass']}, hard {sorted(cats)}, salary {x['verdict']['salary_soft']})")


def design_flag_summary(rows: list[dict]) -> dict:
    """How the recomputed eligibility-pass slice (the one scored) differs from the sealed design flag."""
    design = [r.get("eligibility_pass_design", r["eligibility_pass"]) for r in rows]
    moved = [r for r, dflag in zip(rows, design) if r["eligibility_pass"] != dflag]
    return {"slice_scored": "eligibility_pass recomputed with the rules in effect (amendment 2 definition)",
            "recomputed_pass_rows": sum(r["eligibility_pass"] for r in rows),
            "sealed_design_pass_rows": sum(design),
            "rows_moved_out": len(moved), "postings_moved_out": len({r["posting_id"] for r in moved}),
            "moved_by": dict(Counter(",".join(sorted(set(r["verdict"]["hard_cats"]) & HP_CATS)) for r in moved)),
            "gold_of_moved_rows": dict(Counter(r["gold"] for r in moved))}


def design_flag_summary3(rows: list[dict]) -> dict:
    """Fresh test 3: the recomputed eligibility-pass slice (scored) against the sealed design flag (reported beside
    it), with the rows that moved out (by the H' hard categories or the salary cap now firing) and the rows that moved
    in (design fail, pass now)."""
    out_ = [r for r in rows if r["eligibility_pass_design"] and not r["eligibility_pass"]]
    in_ = [r for r in rows if r["eligibility_pass"] and not r["eligibility_pass_design"]]

    def why(r: dict) -> str:
        v = r["verdict"]
        cats = sorted(set(v["hard_cats"]) & HP_CATS) if v["rule"] == "Skip" else []
        return ",".join(cats + (["salary_cap"] if v["salary_soft"] else [])) or "none"

    return {"slice_scored": "eligibility_pass recomputed with the rules in effect (amendment 3 primary)",
            "slice_reported_beside": "eligibility_pass_design: rows.jsonl design.eligibility_pass, sealed (amendment 3 "
                                     "sensitivity)",
            "recomputed_pass_rows": sum(r["eligibility_pass"] for r in rows),
            "sealed_design_pass_rows": sum(r["eligibility_pass_design"] for r in rows),
            "rows_moved_out": len(out_), "postings_moved_out": len({r["posting_id"] for r in out_}),
            "moved_out_by": dict(Counter(why(r) for r in out_)),
            "gold_of_moved_out_rows": dict(Counter(r["gold"] for r in out_)),
            "rows_moved_in": len(in_), "postings_moved_in": len({r["posting_id"] for r in in_}),
            "gold_of_moved_in_rows": dict(Counter(r["gold"] for r in in_))}


def load_fresh(d: Path = FRESH, strict_design: bool = True) -> list[dict]:
    """The fresh test's rows with their gold, rule verdicts and slice flags. strict_design (fresh tests 1 and 2): the
    recomputed eligibility pass may leave the sealed design flag only by a DESIGN_FLAG_MOVES hard Skip. Fresh test 3
    (amendment 3) scores the recomputed slice with whatever rules are in effect and reports the sealed one beside it,
    so it loads with strict_design False and design_flag_summary3 says which rows moved, either way."""
    rows = jsonl(d / "rows.jsonl")
    posts = {p["key"]: p for p in jsonl(d / "postings.jsonl")}
    cards = {r["id"]: r["card"] for r in jsonl(d / "cards.jsonl")}
    gold = {g["row_id"]: g for g in jsonl(d / "gold.jsonl")}
    if set(gold) != {r["row_id"] for r in rows}:
        raise SystemExit("gold.jsonl and rows.jsonl hold different rows")
    out = []
    for r in rows:
        if r["posting_id"] not in cards:
            raise SystemExit(f"{r['row_id']}: no extracted card (the amendment has no policy for that: none is missing)")
        p = posts[r["posting_id"]]
        card = mark_thin(RequirementCard.from_dict(cards[r["posting_id"]]), p.get("text") or "")
        cand = CandidateCard.from_dict(r["profile"])
        g = gold[r["row_id"]]
        x = base_row(r["row_id"], r["posting_id"], g["gold"], cand, card, r["title"], r["resume_text"])
        d = r["design"]
        if strict_design:
            check_design_flag(r["row_id"], x, d)
        x["eligibility_pass_design"] = bool(d["eligibility_pass"])
        x |= {"label_A": g["label_A"], "label_B": g["label_B"], "agreed": g["label_A"] == g["label_B"],
              "teaser": bool(d["teaser"]), "empty_body": bool(d["empty_body"]), "blank_profile": blank_profile(r["profile"]),
              "lives_in_not_given": not r["profile"].get("current_location"),
              "kind": kind_of(p["title"], p.get("text") or ""), "pool_field": r["field"], "archetype": d["archetype"]}
        if "employer_in_fresh_test1" in d:
            x["employer_in_fresh_test1"] = bool(d["employer_in_fresh_test1"])
        for k in ("posting_in_dev_fit", "posting_in_dev_fit_all"):
            if k in d:
                x[k] = bool(d[k])
        if "posting_length" in d:
            x["posting_length"] = d["posting_length"]
        out.append(x)
    return out


def load_dev() -> list[dict]:
    out = []
    for r in jsonl(FIT / "dev_fit.jsonl"):
        cand, card = CandidateCard.from_dict(r["profile"]), RequirementCard.from_dict(r["card"])
        x = base_row(r["id"], r["posting_key"], r["gold"], cand, card, posting_title(r["posting"]), r["resume"])
        x |= {"source": r["source"], "stored_eligibility_pass": bool(r["eligibility_pass"])}
        out.append(x)
    return out


def read_scores(path: Path, ids: list[str]) -> dict[str, str]:
    """row id -> label from a local_eval / colab_eval result, read again from the float32 scores."""
    d = json.loads(Path(path).read_text())
    rows = d["rows"] if isinstance(d, dict) else d
    out = {}
    for r in rows:
        rid = r.get("pair_id", r.get("row_id"))
        s = [float(x) for x in r["scores"]]
        if len(s) != 3 or not all(np.isfinite(s)):
            raise SystemExit(f"{path}: {rid} has scores {r['scores']}")
        lab = LABELS[pick(s)]
        if r.get("pred") is not None and r["pred"] != lab:
            raise SystemExit(f"{path}: {rid} stored pred {r['pred']} but its scores read {lab}")
        if rid in out:
            raise SystemExit(f"{path}: {rid} twice")
        out[rid] = lab
    want = set(ids)
    if set(out) != want:
        raise SystemExit(f"{path}: {len(want - set(out))} rows missing, {len(set(out) - want)} rows not in the set")
    return out


def systems(rows: list[dict], models: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    out = {}
    for tag, lab in models.items():
        raw = [lab[r["row_id"]] for r in rows]
        out[f"{tag}+H'"] = [hybrid(m, r["verdict"], HP_CATS, True) for m, r in zip(raw, rows)]
        out[f"{tag} raw"] = raw
    out["rules_full"] = [r["rules_full"] for r in rows]
    out["rules_skills_free"] = [r["rules_skills_free"] for r in rows]
    out["title_match"] = ["Keep" if r["title_in_resume"] else "Explore" for r in rows]
    return out


# ---------------------------------------------------------------- statistics


def two_way_draws(groups: list[str], draws: int = DRAWS, seed: int = SEED) -> list[np.ndarray]:
    """Index arrays: postings drawn with replacement, then each drawn posting's rows drawn with replacement."""
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        picked = rng.integers(0, len(uniq), len(uniq))
        out.append(np.concatenate([m[rng.integers(0, len(m), len(m))] for m in (members[g] for g in picked)]))
    return out


def _f1(g: np.ndarray, p: np.ndarray) -> float:
    return macro_f1_of(g, p)


def _acc(g: np.ndarray, p: np.ndarray) -> float:
    return float((g == p).mean())


def interval(vals: list[float], point: float) -> list[float]:
    return [round(point, 4), round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def metrics(g: np.ndarray, p: np.ndarray, draws: list[np.ndarray]) -> dict:
    per = {}
    for w in range(3):
        tp, fp, fn = int(((g == w) & (p == w)).sum()), int(((g != w) & (p == w)).sum()), int(((g == w) & (p != w)).sum())
        per[LABELS[w]] = {"gold": int((g == w).sum()), "pred": int((p == w).sum()),
                          "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                          "recall": round(tp / (tp + fn), 4) if tp + fn else None}
    return {"accuracy": interval([_acc(g[i], p[i]) for i in draws], _acc(g, p)),
            "macro_f1": interval([_f1(g[i], p[i]) for i in draws], _f1(g, p)),
            "per_label": per, "unsafe_keep": int(((g != 0) & (p == 0)).sum())}


def paired(g: np.ndarray, a: np.ndarray, b: np.ndarray, draws: list[np.ndarray]) -> dict:
    """a minus b on the same rows and the same draws; macro_f1_diff_low_exact is the unrounded 2.5th percentile
    (the pass decision reads it, not the 4-decimal value)."""
    f1d = [_f1(g[i], a[i]) - _f1(g[i], b[i]) for i in draws]
    return {"macro_f1_diff_low_exact": float(np.percentile(f1d, 2.5)),
            "accuracy_diff": interval([_acc(g[i], a[i]) - _acc(g[i], b[i]) for i in draws], _acc(g, a) - _acc(g, b)),
            "macro_f1_diff": interval(f1d, _f1(g, a) - _f1(g, b)),
            "rows_fixed": int(((a == g) & (b != g)).sum()), "rows_broken": int(((a != g) & (b == g)).sum())}


def kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[w] * cb[w] for w in LABELS) / n / n
    return round((po - pe) / (1 - pe), 4) if pe < 1 else None


def as_idx(labels: list[str]) -> np.ndarray:
    return np.array([IDX[x] for x in labels])


def score_slice(rows: list[dict], preds: dict[str, list[str]], mask: list[bool], diffs: list[tuple[str, str]],
                draws: int = DRAWS, seed: int = SEED) -> dict | None:
    sel = [i for i, m in enumerate(mask) if m]
    if not sel:
        return None
    sub = [rows[i] for i in sel]
    g = as_idx([r["gold"] for r in sub])
    dr = two_way_draws([r["posting_id"] for r in sub], draws, seed)
    P = {k: as_idx([v[i] for i in sel]) for k, v in preds.items()}
    out = {"n": len(sub), "postings": len({r["posting_id"] for r in sub}),
           "gold": dict(Counter(r["gold"] for r in sub)),
           "systems": {k: metrics(g, p, dr) for k, p in P.items()},
           "paired": {f"{a} - {b}": paired(g, P[a], P[b], dr) for a, b in diffs if a in P and b in P}}
    if "label_A" in sub[0]:
        A, B = [r["label_A"] for r in sub], [r["label_B"] for r in sub]
        out["labeler_ceiling"] = {"raw_agreement": round(sum(x == y for x, y in zip(A, B)) / len(A), 4),
                                  "kappa": kappa(A, B)}
    return out


# ---------------------------------------------------------------- the two sets


def fresh_slices(rows: list[dict]) -> dict[str, list[bool]]:
    s = {"all": [True] * len(rows), "eligibility_pass": [r["eligibility_pass"] for r in rows],
         "teaser": [r["teaser"] for r in rows], "empty_body": [r["empty_body"] for r in rows],
         "teaser_or_empty": [r["teaser"] or r["empty_body"] for r in rows],
         "blank_profile_field": [r["blank_profile"] for r in rows],
         "lives_in_not_given": [r["lives_in_not_given"] for r in rows],
         "agreed_only": [r["agreed"] for r in rows],
         "no_exact_title_in_resume": [not r["title_in_resume"] for r in rows],
         "exact_title_in_resume": [r["title_in_resume"] for r in rows]}
    for k in sorted({r["kind"] for r in rows}):
        s[f"kind:{k}"] = [r["kind"] == k for r in rows]
    if any("employer_in_fresh_test1" in r for r in rows):
        s["employer_new_vs_fresh_test1"] = [not r.get("employer_in_fresh_test1", True) for r in rows]
        s["employer_shared_with_fresh_test1"] = [bool(r.get("employer_in_fresh_test1")) for r in rows]
    return s


def score_fresh(rows: list[dict], models: dict[str, dict[str, str]], chosen: str,
                draws: int = DRAWS, seed: int = SEED) -> dict:
    if chosen not in models:
        raise SystemExit(f"--chosen {chosen}: no score file for it")
    P = systems(rows, models)
    me = f"{chosen}+H'"
    diffs = [(me, k) for k in P if k != me]
    sl = fresh_slices(rows)
    res = {name: score_slice(rows, P, m, diffs, draws, seed) for name, m in sl.items()}
    hp = res["eligibility_pass"]["paired"][f"{me} - rules_skills_free"]
    lo = hp["macro_f1_diff_low_exact"]
    return {"set": "fresh_test", "chosen": me, "draws": draws, "seed": seed,
            "pass": {"rule": "macro-F1 of the chosen system minus skills-free rules on the eligibility-pass slice, "
                             "95% two-way bootstrap lower bound above 0",
                     "system": me, "against": "rules_skills_free", "slice": "eligibility_pass",
                     "macro_f1_diff": hp["macro_f1_diff"], "lower_bound_exact": lo, "accuracy_diff (reported, not for pass)": hp["accuracy_diff"],
                     "passed": bool(lo > 0)},
            "headline_all_rows": res["all"]["paired"][f"{me} - rules_skills_free"],
            "kind_vs_pool_field": {k: dict(Counter(r["pool_field"] for r in rows if r["kind"] == k))
                                   for k in sorted({r["kind"] for r in rows})},
            "title_in_resume_by_archetype": {a: sum(r["title_in_resume"] for r in rows if r["archetype"] == a)
                                             for a in sorted({r["archetype"] for r in rows})},
            "slices": res}


FRESH2_AGAINST = ("rules_full", "base+H'")


def score_fresh2(rows: list[dict], models: dict[str, dict[str, str]], chosen: str,
                 draws: int = DRAWS, seed: int = SEED) -> dict:
    """Amendment 2's pass bar: on the eligibility-pass slice the chosen system beats BOTH rules_full and base+H' on
    macro-F1, each paired difference's two-way bootstrap 95% lower bound above 0."""
    missing = [t for t in (chosen, *FRESH2_REQUIRED) if t not in models]
    if missing:
        raise SystemExit(f"fresh test 2 needs score files for {missing}")
    P = systems(rows, models)
    me = f"{chosen}+H'"
    diffs = [(me, k) for k in P if k != me]
    sl = fresh_slices(rows)
    res = {name: score_slice(rows, P, m, diffs, draws, seed) for name, m in sl.items()}
    ep = res["eligibility_pass"]
    bars = {}
    for b in FRESH2_AGAINST:
        d = ep["paired"][f"{me} - {b}"]
        bars[b] = {"macro_f1_diff": d["macro_f1_diff"], "lower_bound_exact": d["macro_f1_diff_low_exact"],
                   "accuracy_diff (reported, not for pass)": d["accuracy_diff"],
                   "beaten": bool(d["macro_f1_diff_low_exact"] > 0)}
    return {"set": "fresh_test2", "chosen": me, "draws": draws, "seed": seed,
            "eligibility_vs_sealed_design": design_flag_summary(rows),
            "pass": {"rule": "on the eligibility-pass slice, macro-F1 of the chosen system minus each of rules_full and "
                             "base+H', each with a 95% two-way bootstrap lower bound above 0 (amendment 2)",
                     "system": me, "slice": "eligibility_pass", "against": bars,
                     "passed": all(v["beaten"] for v in bars.values())},
            "reported_against": {k: {s_: res[s_]["paired"][f"{me} - {k}"]["macro_f1_diff"]
                                     for s_ in ("all", "eligibility_pass")}
                                 for k in ("h+H'", "d3+H'", "rules_skills_free", *FRESH2_AGAINST) if k != me},
            "kind_vs_pool_field": {k: dict(Counter(r["pool_field"] for r in rows if r["kind"] == k))
                                   for k in sorted({r["kind"] for r in rows})},
            "slices": res}


def fresh3_slices(rows: list[dict]) -> dict[str, list[bool]]:
    """Every slice amendment 3 reports: fresh_slices (all, the recomputed eligibility pass, teaser, kind, ...), the
    sealed-design eligibility pass, posting in cycle 1's dev_fit or not (design.posting_in_dev_fit, and _all), pool
    field and posting length."""
    s = fresh_slices(rows)
    s["eligibility_pass_design"] = [r["eligibility_pass_design"] for r in rows]
    s["posting_in_dev_fit"] = [bool(r.get("posting_in_dev_fit")) for r in rows]
    s["posting_not_in_dev_fit"] = [not r.get("posting_in_dev_fit") for r in rows]
    s["posting_in_dev_fit_all"] = [bool(r.get("posting_in_dev_fit_all")) for r in rows]
    s["posting_not_in_dev_fit_all"] = [not r.get("posting_in_dev_fit_all") for r in rows]
    for k in sorted({r["pool_field"] for r in rows}):
        s[f"field:{k}"] = [r["pool_field"] == k for r in rows]
    for k in sorted({r["posting_length"] for r in rows if r.get("posting_length") is not None}):
        s[f"length:{k}"] = [r.get("posting_length") == k for r in rows]
    return s


FRESH3_HEAD_SLICES = ("all", "eligibility_pass", "eligibility_pass_design")


def score_fresh3(rows: list[dict], models: dict[str, dict[str, str]], chosen: str,
                 draws: int = DRAWS, seed: int = SEED) -> dict:
    """Amendment 3 (amendment 2's bar on another set): on fresh test 3's eligibility-pass slice, recomputed with the
    rules in effect, the chosen system beats BOTH rules_full and base+H' on macro-F1, each paired difference's two-way
    bootstrap 95% lower bound (unrounded) above 0. The sealed-design slice gets the same systems and differences with
    its own draws (DRAWS, SEED), reported, not for pass; so do all rows and every other slice."""
    missing = [t for t in dict.fromkeys((chosen, *FRESH3_REQUIRED)) if t not in models]
    if missing:
        raise SystemExit(f"fresh test 3 needs score files for {missing}")
    P = systems(rows, models)
    me = f"{chosen}+H'"
    diffs = [(me, k) for k in P if k != me]
    sl = fresh3_slices(rows)
    res = {name: score_slice(rows, P, m, diffs, draws, seed) for name, m in sl.items()}
    ep = res["eligibility_pass"]
    if ep is None:
        raise SystemExit("fresh test 3: the recomputed eligibility-pass slice is empty")
    bars = {}
    for b in FRESH2_AGAINST:
        d = ep["paired"][f"{me} - {b}"]
        bars[b] = {"macro_f1_diff": d["macro_f1_diff"], "lower_bound_exact": d["macro_f1_diff_low_exact"],
                   "accuracy_diff (reported, not for pass)": d["accuracy_diff"],
                   "beaten": bool(d["macro_f1_diff_low_exact"] > 0)}
    sens = {}
    if res["eligibility_pass_design"] is not None:
        for b in FRESH2_AGAINST:
            d = res["eligibility_pass_design"]["paired"][f"{me} - {b}"]
            sens[b] = {"macro_f1_diff": d["macro_f1_diff"], "lower_bound_exact": d["macro_f1_diff_low_exact"],
                       "would_be_beaten": bool(d["macro_f1_diff_low_exact"] > 0)}
    return {"set": "fresh_test3", "amendment": AMENDMENT3_NAME, "chosen": me, "draws": draws, "seed": seed,
            "eligibility_vs_sealed_design": design_flag_summary3(rows),
            "pass": {"rule": "on the eligibility-pass slice recomputed with the rules in effect, macro-F1 of the chosen "
                             "system minus each of rules_full and base+H', each with a 95% two-way bootstrap lower bound "
                             "above 0 (amendment 2's bar, read on fresh test 3 by amendment 3)",
                     "system": me, "slice": "eligibility_pass", "against": bars,
                     "passed": all(v["beaten"] for v in bars.values())},
            "sealed_design_slice (sensitivity, not for pass)": {"slice": "eligibility_pass_design", "against": sens},
            "reported_against": {k: {s_: (res[s_]["paired"][f"{me} - {k}"]["macro_f1_diff"] if res[s_] else None)
                                     for s_ in FRESH3_HEAD_SLICES}
                                 for k in ("h+H'", "d3+H'", "rules_skills_free", *FRESH2_AGAINST) if k != me},
            "kind_vs_pool_field": {k: dict(Counter(r["pool_field"] for r in rows if r["kind"] == k))
                                   for k in sorted({r["kind"] for r in rows})},
            "slices": res}


def select_fresh1(rows: list[dict], models: dict[str, dict[str, str]], draws: int = DRAWS, seed: int = SEED) -> dict:
    """Amendment 2 selection on all fresh test 1 rows: the candidate (h, h2*) with the highest TAG+H' macro-F1; when
    the best is within TIE_MARGIN of the incumbent h, h stays."""
    cands = sorted(t for t in models if CANDIDATE_RX.match(t))
    if INCUMBENT not in cands:
        raise SystemExit(f"selection needs the incumbent {INCUMBENT}'s scores")
    P = {k: v for k, v in systems(rows, models).items() if k in {f"{t}+H'" for t in cands}}
    res = score_slice(rows, P, [True] * len(rows), [(f"{t}+H'", f"{INCUMBENT}+H'") for t in cands if t != INCUMBENT],
                      draws, seed)
    f1 = {t: res["systems"][f"{t}+H'"]["macro_f1"][0] for t in cands}
    best = max(cands, key=lambda t: (f1[t], t == INCUMBENT))
    if best != INCUMBENT and round(f1[best] - f1[INCUMBENT], 6) <= TIE_MARGIN:
        pick_, why = INCUMBENT, f"{best} is within {TIE_MARGIN} macro-F1 of {INCUMBENT}: the incumbent stays"
    elif best == INCUMBENT:
        pick_, why = INCUMBENT, "the incumbent has the highest macro-F1"
    else:
        pick_, why = best, f"higher macro-F1 than {INCUMBENT} by more than the tie margin {TIE_MARGIN}"
    return {"set": "fresh_test (selection set, amendment 2)", "amendment": "eval/protocol_amendment_2.json",
            "candidates": cands, "ignored_tags": sorted(set(models) - set(cands)), "draws": draws, "seed": seed,
            "selection": {"macro_f1": f1, "pick": pick_, "why": why, "tie_margin": TIE_MARGIN, "incumbent": INCUMBENT},
            "pick": pick_, "eligibility_vs_sealed_design": design_flag_summary(rows), "all_rows": res}


def score_dev(rows: list[dict], models: dict[str, dict[str, str]], draws: int = DRAWS, seed: int = SEED) -> dict:
    P = systems(rows, models)
    cands = [t for t in SELECTION_TAGS if t in models]
    diffs = [(f"{a}+H'", f"{b}+H'") for a in cands for b in cands if a < b]
    diffs += [(f"{t}+H'", "rules_skills_free") for t in models]
    sl = {"all": [True] * len(rows), "eligibility_pass": [r["eligibility_pass"] for r in rows]}
    for s in sorted({r["source"] for r in rows}):
        sl[f"source:{s}"] = [r["source"] == s for r in rows]
    res = {name: score_slice(rows, P, m, diffs, draws, seed) for name, m in sl.items()}
    f1 = {t: res["all"]["systems"][f"{t}+H'"]["macro_f1"][0] for t in cands}
    pick_ = None
    if len(cands) == 2:
        a, b = cands
        if round(abs(f1[a] - f1[b]), 6) <= TIE_MARGIN:
            pick_, why = TIE_PICK, f"within {TIE_MARGIN} on macro-F1: the incumbent {TIE_PICK} stays"
        else:
            pick_ = max(cands, key=f1.get)
            why = "higher macro-F1 by more than the tie margin"
    elif cands:
        pick_, why = cands[0], "the only candidate scored"
    else:
        why = "no candidate scored"
    return {"set": "dev_fit", "draws": draws, "seed": seed,
            "selection": {"macro_f1": f1, "pick": pick_, "why": why, "tie_margin": TIE_MARGIN},
            "eligibility_pass_rows": {"amendment definition (no HP_CATS Skip, no salary cap)": sum(r["eligibility_pass"] for r in rows),
                                      "dev_fit stored flag (no HP_CATS Skip)": sum(r["stored_eligibility_pass"] for r in rows)},
            "note": "dev_fit gold is rules.label on the rows' own cards, so rules_full scores 1.0 here by construction",
            "slices": res}


# ---------------------------------------------------------------- output


def fmt(x: list[float]) -> str:
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"


def markdown(res: dict) -> str:
    if res.get("set") == "fresh_test4":
        return markdown4(res)
    L = [f"# {res['set']}: systems and baselines", "",
         f"Two-way bootstrap (postings, then candidates within them), {res['draws']} draws, seed {res['seed']}; "
         "95% percentile intervals.", ""]
    if "pass" in res and isinstance(res["pass"]["against"], dict):
        p = res["pass"]
        L += [f"## Pass rule (amendment {3 if res.get('set') == 'fresh_test3' else 2})", "",
              f"**{'PASS' if p['passed'] else 'NOT PASSED'}**: {p['rule']}", ""]
        L += [f"- {p['system']} minus {b}: {fmt(v['macro_f1_diff'])} -> {'beaten' if v['beaten'] else 'not beaten'} "
              f"(accuracy diff, reported only: {fmt(v['accuracy_diff (reported, not for pass)'])})"
              for b, v in p["against"].items()]
        L.append("")
        sens = res.get("sealed_design_slice (sensitivity, not for pass)")
        if sens is not None:
            L += ["Sealed-design eligibility-pass slice (sensitivity, not for pass):", ""]
            L += [f"- {p['system']} minus {b}: {fmt(v['macro_f1_diff'])} -> "
                  f"{'would be beaten' if v['would_be_beaten'] else 'would not be beaten'}" for b, v in sens["against"].items()]
            L.append("")
        ev = res.get("eligibility_vs_sealed_design")
        if ev is not None and res.get("set") == "fresh_test3":
            L += [f"Eligibility pass recomputed: {ev['recomputed_pass_rows']} rows; sealed design: "
                  f"{ev['sealed_design_pass_rows']} rows; moved out {ev['rows_moved_out']} ({ev['moved_out_by']}), "
                  f"moved in {ev['rows_moved_in']}.", ""]
    elif "pass" in res:
        p = res["pass"]
        L += ["## Pass rule", "", f"{p['system']} minus {p['against']}, macro-F1, {p['slice']}: "
              f"{fmt(p['macro_f1_diff'])} -> **{'PASS' if p['passed'] else 'NOT PASSED'}** "
              f"(accuracy diff, reported only: {fmt(p['accuracy_diff (reported, not for pass)'])})", ""]
    if "selection" in res:
        s = res["selection"]
        L += ["## Selection", "", f"macro-F1 on all rows: {s['macro_f1']}; pick **{s['pick']}** ({s['why']}).", ""]
    heads = FRESH3_HEAD_SLICES if res.get("set") == "fresh_test3" else ("all", "eligibility_pass")
    for name in heads if "slices" in res else ():
        sl = res["slices"][name]
        if sl is None:
            L += [f"## {name} (n=0)", ""]
            continue
        L += [f"## {name} (n={sl['n']}, postings={sl['postings']})", "", "| system | accuracy | macro-F1 |", "|---|---|---|"]
        L += [f"| {k} | {fmt(v['accuracy'])} | {fmt(v['macro_f1'])} |" for k, v in sl["systems"].items()]
        if sl.get("labeler_ceiling"):
            c = sl["labeler_ceiling"]
            L += [f"| labeler ceiling (A vs B) | {c['raw_agreement']:.3f} | kappa {c['kappa']} |"]
        L.append("")
    me = res.get("chosen")
    if me and res.get("set") in ("fresh_test2", "fresh_test3"):
        L += ["## Slices (reported, not for pass)", "",
              f"| slice | n | {me} macro-F1 | rules_full macro-F1 | base+H' macro-F1 | {me} - rules_full | "
              f"{me} - base+H' |", "|---|---|---|---|---|---|---|"]
        for name, sl in res["slices"].items():
            if sl is None:
                L.append(f"| {name} | 0 | | | | | |")
                continue
            S, D, bh = sl["systems"], sl["paired"], "base+H'"
            L.append(f"| {name} | {sl['n']} | {fmt(S[me]['macro_f1'])} | {fmt(S['rules_full']['macro_f1'])} | "
                     f"{fmt(S[bh]['macro_f1'])} | {fmt(D[me + ' - rules_full']['macro_f1_diff'])} | "
                     f"{fmt(D[me + ' - ' + bh]['macro_f1_diff'])} |")
        L.append("")
    elif me:
        L += ["## Slices (reported, not for pass)", "",
              f"| slice | n | {me} macro-F1 | rules_skills_free macro-F1 | diff |", "|---|---|---|---|---|"]
        for name, sl in res["slices"].items():
            if sl is None:
                L.append(f"| {name} | 0 | | | |")
                continue
            d = sl["paired"].get(f"{me} - rules_skills_free", {}).get("macro_f1_diff", [0.0, 0.0, 0.0])
            L.append(f"| {name} | {sl['n']} | {fmt(sl['systems'][me]['macro_f1'])} | "
                     f"{fmt(sl['systems']['rules_skills_free']['macro_f1'])} | {fmt(d)} |")
        L.append("")
    return "\n".join(L)


def check_hashes(names: list[str]) -> None:
    """The files a run reads must be the ones eval/fit_eval_hashes.json pinned."""
    pinned = json.loads(HASHES.read_text())["sha256"]
    for n in names:
        if sha256(ROOT / n) != pinned[n]:
            raise SystemExit(f"{n} is not the file eval/fit_eval_hashes.json pinned")


def parse_scores(items: list[str]) -> dict[str, Path]:
    out = {}
    for it in items:
        tag, path = it.split("=", 1)
        out[tag] = Path(path)
    return out


def write_once(path: Path, res: dict) -> None:
    """Write a result json (and its .md beside it); refuse when either already exists: a set is scored once."""
    md = path.with_suffix(".md")
    for f in (path, md):
        if f.exists():
            raise SystemExit(f"{f} exists: refusing to overwrite a result (each set is scored once)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")
    md.write_text(markdown(res))
    print(markdown(res))


def write(out: Path, stem: str, res: dict) -> None:
    write_once(out / f"{stem}.json", res)


def committed(path: Path) -> bool:
    """The file is tracked by git and identical to HEAD (nothing staged or unstaged)."""
    import subprocess
    path = Path(path).resolve()
    top = subprocess.run(["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if top.returncode:
        return False
    rel = str(path.relative_to(Path(top.stdout.strip()).resolve()))
    tracked = subprocess.run(["git", "-C", top.stdout.strip(), "ls-files", "--error-unmatch", rel], capture_output=True)
    if tracked.returncode:
        return False
    clean = subprocess.run(["git", "-C", top.stdout.strip(), "diff", "--quiet", "HEAD", "--", rel], capture_output=True)
    return clean.returncode == 0


def read_pick(selection: Path, need_amendment: str | None = None) -> str:
    """The committed pick: a selection json that git tracks unchanged from HEAD, with a 'pick' key."""
    selection = Path(selection)
    if not selection.exists():
        raise SystemExit(f"{selection}: no such selection file")
    if not committed(selection):
        raise SystemExit(f"{selection} is not committed (untracked or changed since HEAD): commit the pick first")
    d = json.loads(selection.read_text())
    pick_ = d.get("pick") or (d.get("selection") or {}).get("pick")
    if not pick_:
        raise SystemExit(f"{selection} holds no pick")
    if need_amendment and d.get("amendment") != need_amendment:
        raise SystemExit(f"{selection} is not a selection made under {need_amendment}")
    return pick_


def cmd_dev(a: argparse.Namespace) -> None:
    check_hashes(["data/built/fit/dev_fit.jsonl", "eval/protocol_amendment_1.json"])
    rows = load_dev()
    models = {t: read_scores(p, [r["row_id"] for r in rows]) for t, p in parse_scores(a.score).items()}
    write(Path(a.out), "dev_fit", score_dev(rows, models))


def cmd_fresh(a: argparse.Namespace) -> None:
    check_hashes(["data/built/fresh_test/rows.jsonl", "data/built/fresh_test/cards.jsonl",
                  "data/built/fresh_test/gold.jsonl", "eval/protocol_amendment_1.json"])
    chosen = read_pick(Path(a.selection))
    rows = load_fresh()
    models = {t: read_scores(p, [r["row_id"] for r in rows]) for t, p in parse_scores(a.score).items()}
    write(Path(a.out), "fresh_test", score_fresh(rows, models, chosen))


def cmd_select(a: argparse.Namespace) -> None:
    check_hashes(["data/built/fresh_test/rows.jsonl", "data/built/fresh_test/cards.jsonl",
                  "data/built/fresh_test/gold.jsonl"])
    if not committed(AMENDMENT2):
        raise SystemExit("eval/protocol_amendment_2.json is not committed unchanged")
    rows = load_fresh()
    paths = parse_scores(a.score)
    models = {t: read_scores(p, [r["row_id"] for r in rows]) for t, p in paths.items()}
    res = select_fresh1(rows, models)
    res["score_files"] = {t: {"path": str(p.resolve().relative_to(ROOT)), "sha256": sha256(p)} for t, p in paths.items()}
    res["archives"] = {t: {"file": Path(p).name, "sha256": sha256(Path(p))} for t, p in parse_scores(a.archive or []).items()}
    if a.note:
        res["notes"] = a.note
    write_once(Path(a.out), res)


def check_fresh2() -> None:
    """Fresh test 2's rows and cards must be the sealed ones, its gold the one its gold manifest pins, and amendment 2
    committed unchanged."""
    man = json.loads(FRESH2_MANIFEST.read_text())["files"]
    for n in ("data/built/fresh_test2/rows.jsonl", "data/built/fresh_test2/cards.jsonl"):
        if sha256(ROOT / n) != man[n]:
            raise SystemExit(f"{n} is not the file eval/fresh_test2_manifest.json sealed")
    if not FRESH2_GOLD_MANIFEST.exists():
        raise SystemExit("eval/fresh_test2_gold_manifest.json is missing: fresh test 2 has no sealed gold yet")
    if sha256(FRESH2 / "gold.jsonl") != json.loads(FRESH2_GOLD_MANIFEST.read_text())["gold_sha256"]:
        raise SystemExit("data/built/fresh_test2/gold.jsonl is not the gold eval/fresh_test2_gold_manifest.json pins")
    for f in (AMENDMENT2, FRESH2_MANIFEST, FRESH2_GOLD_MANIFEST):
        if not committed(f):
            raise SystemExit(f"{f.relative_to(ROOT)} is not committed unchanged")


def cmd_fresh2(a: argparse.Namespace) -> None:
    if FRESH2_RESULT.exists() or FRESH2_RESULT.with_suffix(".md").exists():
        raise SystemExit(f"{FRESH2_RESULT.relative_to(ROOT)} exists: fresh test 2 is scored once")
    chosen = read_pick(Path(a.selection), need_amendment="eval/protocol_amendment_2.json")
    check_fresh2()
    rows = load_fresh(FRESH2)
    models = {t: read_scores(p, [r["row_id"] for r in rows]) for t, p in parse_scores(a.score).items()}
    write_once(FRESH2_RESULT, score_fresh2(rows, models, chosen))


def selection_amendments(d: dict) -> list[str]:
    a = d.get("amendments") or []
    return ([d["amendment"]] if d.get("amendment") else []) + (list(a) if isinstance(a, list) else [a])


def read_pick3(selection: Path) -> str:
    """The committed pick for fresh test 3: read_pick's checks, and the selection names amendment 3 (as its
    'amendment' or among its 'amendments')."""
    pick_ = read_pick(selection)
    if AMENDMENT3_NAME not in selection_amendments(json.loads(Path(selection).read_text())):
        raise SystemExit(f"{selection} does not name {AMENDMENT3_NAME}: a fresh test 3 selection must")
    return pick_


def check_fresh3() -> None:
    """Fresh test 3's files are the sealed ones before any row is read: every hash in its manifest's files, in
    eval/fresh_test3_data_hashes.json and the gold its gold manifest pins; amendment 3's own pins (the manifest and
    the protocol files it amends); the gold manifest's row count; amendments 2 and 3, the manifest, the gold
    manifest and the data-hash file committed unchanged."""
    for f in (FRESH3_MANIFEST, FRESH3_DATA_HASHES, FRESH3_GOLD_MANIFEST, AMENDMENT3):
        if not f.exists():
            raise SystemExit(f"{f.relative_to(ROOT)} is missing")
    for f in (AMENDMENT2, AMENDMENT3, FRESH3_MANIFEST, FRESH3_GOLD_MANIFEST, FRESH3_DATA_HASHES):
        if not committed(f):
            raise SystemExit(f"{f.relative_to(ROOT)} is not committed unchanged")
    am = json.loads(AMENDMENT3.read_text())
    if sha256(FRESH3_MANIFEST) != am["fresh_test_3"]["manifest_sha256"]:
        raise SystemExit("eval/fresh_test3_manifest.json is not the manifest eval/protocol_amendment_3.json pins")
    for n, h in am["amends"]["files"].items():
        if sha256(ROOT / n) != h:
            raise SystemExit(f"{n} is not the file eval/protocol_amendment_3.json amends")
    man = json.loads(FRESH3_MANIFEST.read_text())
    for src, pins in (("eval/fresh_test3_manifest.json", man["files"]),
                      ("eval/fresh_test3_data_hashes.json", json.loads(FRESH3_DATA_HASHES.read_text())["sha256"])):
        if not pins:
            raise SystemExit(f"{src} pins no file")
        for n, h in pins.items():
            if not (ROOT / n).exists():
                raise SystemExit(f"{n} is missing ({src} pins it)")
            if sha256(ROOT / n) != h:
                raise SystemExit(f"{n} is not the file {src} pinned")
    gm = json.loads(FRESH3_GOLD_MANIFEST.read_text())
    if sha256(FRESH3 / "gold.jsonl") != gm["gold_sha256"]:
        raise SystemExit("data/built/fresh_test3/gold.jsonl is not the gold eval/fresh_test3_gold_manifest.json pins")
    if gm.get("n_gold") != man["n_rows"] or gm.get("missing_rows"):
        raise SystemExit(f"fresh test 3 gold holds {gm.get('n_gold')} rows (missing {gm.get('missing_rows')}), "
                         f"the manifest {man['n_rows']}")


def cmd_fresh3(a: argparse.Namespace) -> None:
    if FRESH3_RESULT.exists() or FRESH3_RESULT.with_suffix(".md").exists():
        raise SystemExit(f"{FRESH3_RESULT.relative_to(ROOT)} exists: fresh test 3 is scored once")
    sel = Path(a.selection)
    chosen = read_pick3(sel)
    check_fresh3()
    rows = load_fresh(FRESH3, strict_design=False)
    if len(rows) != json.loads(FRESH3_MANIFEST.read_text())["n_rows"]:
        raise SystemExit("fresh test 3 rows.jsonl does not hold the manifest's n_rows")
    paths = parse_scores(a.score)
    models = {t: read_scores(p, [r["row_id"] for r in rows]) for t, p in paths.items()}
    res = score_fresh3(rows, models, chosen)
    def rel(f: Path) -> str:
        f, top = Path(f).resolve(), ROOT.resolve()
        return str(f.relative_to(top) if f.is_relative_to(top) else f)

    res["selection_file"] = {"path": rel(sel), "sha256": sha256(sel), "pick": chosen}
    res["score_files"] = {t: {"path": rel(p), "sha256": sha256(Path(p))} for t, p in paths.items()}
    write_once(FRESH3_RESULT, res)


# ---------------------------------------------------------------- fresh test 4 (amendments 4, 4b, 4c, 5, 5b), once


FRESH4 = ROOT / "data/built/fresh_test4"
FRESH4_MANIFEST = ROOT / "eval/fresh_test4_manifest.json"
FRESH4_DATA_HASHES = ROOT / "eval/fresh_test4_data_hashes.json"
FRESH4_GOLD_MANIFEST = ROOT / "eval/fresh_test4_gold_manifest.json"
FRESH4_INPUTS = ROOT / "eval/fresh_test4_eval_inputs.json"
FRESH4_RESULT = ROOT / "eval/fresh_test4_result.json"
FRESH4_ROWS_OUT = ROOT / "data/models/c3_eval/test4/test4_rows.jsonl"
SELECTION_C3 = ROOT / "eval/selection_cycle3.json"
FRESH4_AMENDMENTS = tuple(ROOT / f"eval/protocol_amendment_{k}.json" for k in ("2", "4", "4b", "4c", "5", "5b"))
FRESH4_NEEDS = ("h", "base")  # beside the pick's model: h + H' (the recorded system) and the untrained base
FRESH4_HEAD_SLICES = ("all", "eligibility_pass", "agreed_only", "agreed_eligibility_pass")
FRESH4_SCORE_HEADER = {"framing": "chat", "dtype": "bfloat16", "dev": "fresh_test4.jsonl"}
NO_CARD_VERDICT = {"rule": None, "hard_cats": [], "salary_soft": False, "reasons": ["no extracted card"]}
NO_CARD_RULES = "Explore"


def load_fresh4(d: Path | None = None) -> list[dict]:
    """Fresh test 4's rows (amendment 5: the slice recomputed with the rules in effect, the sealed design flag beside
    it, as fresh test 3). Rows whose posting has no extracted card (eval/fresh_test4_eval_inputs.json
    no_card_rows_policy, fixed before any inference): no rule verdict (H' and H'' override nothing), rules_full and
    rules_skills_free Explore, outside the eligibility-pass slice."""
    d = Path(d or FRESH4)
    rows = jsonl(d / "rows.jsonl")
    posts = {p["key"]: p for p in jsonl(d / "postings.jsonl")}
    cards = {r["id"]: r["card"] for r in jsonl(d / "cards.jsonl")}
    gold = {g["row_id"]: g for g in jsonl(d / "gold.jsonl")}
    if set(gold) != {r["row_id"] for r in rows} or len(gold) != len(rows):
        raise SystemExit("gold.jsonl and rows.jsonl hold different rows")
    out = []
    for r in rows:
        p, g, dsg = posts[r["posting_id"]], gold[r["row_id"]], r["design"]
        has_card = r["posting_id"] in cards
        if has_card != bool(dsg["has_card"]):
            raise SystemExit(f"{r['row_id']}: design.has_card {dsg['has_card']} but cards.jsonl says {has_card}")
        if has_card:
            card = mark_thin(RequirementCard.from_dict(cards[r["posting_id"]]), p.get("text") or "")
            x = base_row(r["row_id"], r["posting_id"], g["gold"], CandidateCard.from_dict(r["profile"]), card,
                         r["title"], r["resume_text"])
        else:
            if dsg["eligibility_pass"] is not None:
                raise SystemExit(f"{r['row_id']}: no card but a sealed eligibility flag")
            x = {"row_id": r["row_id"], "posting_id": r["posting_id"], "gold": g["gold"], "verdict": dict(NO_CARD_VERDICT),
                 "eligibility_pass": False, "rules_full": NO_CARD_RULES, "rules_skills_free": NO_CARD_RULES,
                 "title_in_resume": title_in_resume(r["title"], r["resume_text"])}
        x |= {"has_card": has_card, "eligibility_pass_design": bool(dsg["eligibility_pass"]),
              "label_A": g["label_A"], "label_B": g["label_B"], "agreed": g["label_A"] == g["label_B"],
              "teaser": bool(dsg["teaser"]), "empty_body": bool(dsg["empty_body"]),
              "blank_profile": blank_profile(r["profile"]), "lives_in_not_given": not r["profile"].get("current_location"),
              "kind": kind_of(p["title"], p.get("text") or ""), "pool_field": r["field"], "archetype": dsg["archetype"],
              "posting_length": dsg["posting_length"], "employer": r["employer"],
              "employer_in_fresh_test1_3": bool(dsg["employer_in_fresh_test1_3"]),
              "extension_4b": dsg.get("extension") == "4b"}
        out.append(x)
    return out


def fresh4_slices(rows: list[dict]) -> dict[str, list[bool]]:
    """fresh_slices (all, eligibility_pass, agreed_only, teaser, kind, ...) and test 4's own: rows with a card, the
    agreement subset of the eligibility-pass slice, the sealed design slice, the new-employer slice (employer in no
    fresh test 1-3), amendment 4b's extension and the first 940 rows, pool field and posting length."""
    s = fresh_slices(rows)
    s["all_with_card"] = [r["has_card"] for r in rows]
    s["agreed_eligibility_pass"] = [r["agreed"] and r["eligibility_pass"] for r in rows]
    s["eligibility_pass_design"] = [r["eligibility_pass_design"] for r in rows]
    s["employer_new_vs_fresh_tests_1_3"] = [not r["employer_in_fresh_test1_3"] for r in rows]
    s["employer_shared_with_fresh_tests_1_3"] = [r["employer_in_fresh_test1_3"] for r in rows]
    s["extension_4b"] = [r["extension_4b"] for r in rows]
    s["first_940"] = [not r["extension_4b"] for r in rows]
    for k in sorted({r["pool_field"] for r in rows}):
        s[f"field:{k}"] = [r["pool_field"] == k for r in rows]
    for k in sorted({r["posting_length"] for r in rows}):
        s[f"length:{k}"] = [r["posting_length"] == k for r in rows]
    return s


def fresh4_labels(rows: list[dict], scores: dict[str, np.ndarray], spec: dict) -> tuple[dict[str, list[str]], dict]:
    """The four systems amendment 5 scores on test 4, as label lists in rows order, and each decider model's s:
    the pick (its model, H'' variant, readout and committed cut-offs), h + H' (argmax, H' = HP_CATS + salary cap),
    rules_full, and base with the pick's H'' variant and readout (its own committed cut-offs for a decider).
    scores: tag -> rows x (Keep, Explore, Skip) float32 first-token scores. Nothing is fitted here."""
    import fit_select_cycle3 as C3
    pk, bs = spec["pick"], spec["base"]
    cats = C3.VARIANTS[pk["variant"]]
    out: dict[str, list[str]] = {}
    svals = {}

    def readout(tag: str, cut: dict) -> list[str]:
        S = scores[tag]
        if pk["readout"] == "decider":
            s = C3.decider_score(S)
            svals[tag] = s
            lab = C3.decider_idx(s, cut["t_lo"], cut["t_hi"])
        else:
            lab = C3.argmax_idx(S)
        return [LABELS[int(i)] for i in C3.apply_h_rows(lab, rows, cats)]

    out[spec["pick_system"]] = readout(pk["model"], pk)
    out["h+H'"] = [hybrid(LABELS[pick(list(x))], r["verdict"], HP_CATS, True) for x, r in zip(scores["h"], rows)]
    out["rules_full"] = [r["rules_full"] for r in rows]
    out[spec["base_system"]] = readout("base", bs)
    return out, svals


def keep_line(rows: list[dict], preds: list[str], mask: list[bool]) -> dict:
    """Keep recall with the gold Keep count beside it (amendment 4c) and Keep precision, exact counts."""
    g = [r["gold"] for r, m in zip(rows, mask) if m]
    p = [x for x, m in zip(preds, mask) if m]
    gk = sum(x == "Keep" for x in g)
    pk = sum(x == "Keep" for x in p)
    tp = sum(a == b == "Keep" for a, b in zip(g, p))
    return {"gold_keeps": gk, "found": tp, "recall": round(tp / gk, 4) if gk else None,
            "predicted_keeps": pk, "precision": round(tp / pk, 4) if pk else None,
            "recall_step": round(1 / gk, 4) if gk else None}


def headline4(rows: list[dict], P: dict[str, list[str]], sl: dict[str, list[bool]], res: dict) -> dict:
    out = {}
    for name in FRESH4_HEAD_SLICES:
        r = res[name]
        out[name] = {"n": r["n"], "postings": r["postings"], "gold": r["gold"], "systems": {}}
        for k, m in r["systems"].items():
            out[name]["systems"][k] = {
                "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                "recall": {w: m["per_label"][w]["recall"] for w in LABELS},
                "keep": keep_line(rows, P[k], sl[name]), "unsafe_keep": m["unsafe_keep"]}
    return out


def bar(res_slice: dict, me: str, against: list[str]) -> dict:
    out = {}
    for b in against:
        d = res_slice["paired"][f"{me} - {b}"]
        out[b] = {"macro_f1_diff": d["macro_f1_diff"], "lower_bound_exact": d["macro_f1_diff_low_exact"],
                  "accuracy_diff (reported, not for pass)": d["accuracy_diff"],
                  "beaten": bool(d["macro_f1_diff_low_exact"] > 0)}
    return out


def score_fresh4(rows: list[dict], scores: dict[str, np.ndarray], spec: dict,
                 draws: int = DRAWS, seed: int = SEED) -> tuple[dict, dict[str, list[str]], dict]:
    """Amendment 5 on fresh test 4 under amendment 2's bar: on the eligibility-pass slice the pick beats BOTH
    rules_full and base (the pick's H'' variant and readout) on macro-F1, each paired difference (same rows, same
    draws) with a 95% two-way bootstrap lower bound (unrounded) above 0. Every slice gets its own draws (DRAWS, SEED),
    shared by every system on it. The same differences on the agreement subset (rows where labelers A and B agree),
    all rows and every other slice are reported, not for pass."""
    missing = [t for t in dict.fromkeys((spec["pick"]["model"], *FRESH4_NEEDS)) if t not in scores]
    if missing:
        raise SystemExit(f"fresh test 4 needs score files for {missing}")
    for t, S in scores.items():
        if S.shape != (len(rows), 3):
            raise SystemExit(f"{t}: scores of shape {S.shape}, rows {len(rows)}")
    P, svals = fresh4_labels(rows, scores, spec)
    me, base = spec["pick_system"], spec["base_system"]
    against = ["rules_full", base]
    diffs = [(me, k) for k in P if k != me]
    sl = fresh4_slices(rows)
    res = {name: score_slice(rows, P, m, diffs, draws, seed) for name, m in sl.items()}
    ep = res["eligibility_pass"]
    if ep is None:
        raise SystemExit("fresh test 4: the eligibility-pass slice is empty")
    bars = bar(ep, me, against)
    agree = {name: bar(res[name], me, against) if res[name] else None
             for name in ("agreed_eligibility_pass", "agreed_only")}
    new = res["employer_new_vs_fresh_tests_1_3"]
    new_rows = [r for r, m in zip(rows, sl["employer_new_vs_fresh_tests_1_3"]) if m]
    return ({"set": "fresh_test4", "amendments": [str(f.relative_to(ROOT)) for f in FRESH4_AMENDMENTS],
             "chosen": me, "systems": list(P), "base_comparator": base, "draws": draws, "seed": seed,
             "spec": spec,
             "pass": {"rule": "on the eligibility-pass slice, macro-F1 of the pick minus each of rules_full and base "
                              "(the pick's H'' variant and readout), each with a 95% two-way bootstrap lower bound "
                              "(2.5th percentile, unrounded) above 0 (amendment 2's bar, amendment 5 on fresh test 4)",
                      "system": me, "slice": "eligibility_pass", "n": ep["n"], "postings": ep["postings"],
                      "gold": ep["gold"], "against": bars, "passed": all(v["beaten"] for v in bars.values())},
             "sealed_design_slice (sensitivity, not for pass)": {
                 "slice": "eligibility_pass_design",
                 "what": "the sealed design flag (build_fresh_test.verdict on the card without rules.mark_thin; the "
                         "scored slice applies mark_thin on the posting body first, as every earlier test did)",
                 "n": res["eligibility_pass_design"]["n"] if res["eligibility_pass_design"] else 0,
                 "gold": res["eligibility_pass_design"]["gold"] if res["eligibility_pass_design"] else {},
                 "against": {b: {k: v[k] for k in ("macro_f1_diff", "lower_bound_exact")} | {"would_be_beaten": v["beaten"]}
                             for b, v in (bar(res["eligibility_pass_design"], me, against)
                                          if res["eligibility_pass_design"] else {}).items()}},
             "agreement_subset (reported, not for pass)": {
                 "what": "the bar's differences on rows where labelers A and B agree (amendment 5 also_reported)",
                 **{k: {"n": res[k]["n"] if res[k] else 0, "gold": res[k]["gold"] if res[k] else {}, "against": v,
                        "would_pass": bool(v) and all(x["beaten"] for x in v.values())} for k, v in agree.items()}},
             "headline": headline4(rows, P, sl, res),
             "new_employer_slice": {
                 "slice": "employer_new_vs_fresh_tests_1_3", "n": new["n"] if new else 0,
                 "postings": new["postings"] if new else 0, "employers": sorted({r["employer"] for r in new_rows}),
                 "eligibility_pass_rows": sum(r["eligibility_pass"] for r in new_rows),
                 "gold": new["gold"] if new else {},
                 "flag": "too small to carry the bar: reported, not for pass (fresh test 4 is new at the posting "
                         "level, not the employer level; eval/fresh_test4_manifest.json employers_vs_fresh_tests_1_3)",
                 "macro_f1": {k: v["macro_f1"] for k, v in new["systems"].items()} if new else {}},
             "eligibility_vs_sealed_design": design_flag_summary3(rows),
             "no_card_rows": {"rows": [r["row_id"] for r in rows if not r["has_card"]],
                              "gold": dict(Counter(r["gold"] for r in rows if not r["has_card"])),
                              "policy": "eval/fresh_test4_eval_inputs.json no_card_rows_policy: no rule verdict (no "
                                        "H'/H'' override), rules_full Explore, outside the eligibility-pass slice"},
             "kind_vs_pool_field": {k: dict(Counter(r["pool_field"] for r in rows if r["kind"] == k))
                                    for k in sorted({r["kind"] for r in rows})},
             "slices": res}, P, svals)


def read_spec4(selection: Path = SELECTION_C3) -> dict:
    """The committed pick and base comparator: the selection json is tracked unchanged from HEAD, names amendments 5
    and 5b with their sha256 equal to the files now, holds a pick that cleared its floors, and the cut-offs the pick
    and base comparator carry equal the fitted ones it records; nothing is typed in."""
    import fit_select_cycle3 as C3
    selection = Path(selection)
    if not selection.exists():
        raise SystemExit(f"{selection}: no such selection file")
    if not committed(selection):
        raise SystemExit(f"{selection} is not committed (untracked or changed since HEAD): commit the pick first")
    d = json.loads(selection.read_text())
    names = selection_amendments(d)
    for a in ("eval/protocol_amendment_5.json", "eval/protocol_amendment_5b.json"):
        if a not in names:
            raise SystemExit(f"{selection} does not name {a}")
        if sha256(ROOT / a) != (d.get("amendments_sha256") or {}).get(a):
            raise SystemExit(f"{a} is not the file {selection.name} was selected under")
    pk = d.get("pick")
    if not isinstance(pk, dict) or not pk.get("system"):
        raise SystemExit(f"{selection} holds no cycle-3 pick")
    if pk.get("no_system_cleared_floors"):
        raise SystemExit(f"{selection}: no system cleared the floors; amendment 5b's fallback is not scored here")
    m, v, r = C3.parse(pk["system"])
    if (m, v, r) != (pk["model"], pk["eligibility_layer"], pk["readout"]) or v not in C3.VARIANTS:
        raise SystemExit(f"{selection}: the pick's fields disagree with its name {pk['system']}")
    if d["definitions"][v] != sorted(C3.VARIANTS[v]) + ["salary cap"]:
        raise SystemExit(f"{selection}: {v} is not the variant fit_select_cycle3 defines")
    bc = d["base_comparator_for_test_4"]
    if bc["system"] != C3.system_name("base", v, r):
        raise SystemExit(f"{selection}: the base comparator {bc['system']} is not base with the pick's {v} and {r}")
    spec = {"pick_system": pk["system"], "base_system": bc["system"],
            "pick": {"model": m, "variant": v, "readout": r}, "base": {},
            "selection": {"path": str(selection.resolve().relative_to(ROOT.resolve())), "sha256": sha256(selection)}}
    if r == "decider":
        fit_p, fit_b = d["decider_cut_offs"][f"{m}+{v}"], d["base_decider_cut_offs"][f"base+{v}"]
        for who, cut, fit in (("pick", pk["cut_offs"], fit_p), ("base", bc["cut_offs"], fit_b)):
            if cut["t_lo"] is None or (cut["t_lo"], cut["t_hi"]) != (fit["t_lo"], fit["t_hi"]):
                raise SystemExit(f"{selection}: the {who}'s cut-offs are not the fitted ones it records")
            spec[who] |= {"t_lo": float(cut["t_lo"]), "t_hi": float(cut["t_hi"]), "floors_met": cut["floors_met"]}
    return spec


def check_fresh4() -> dict:
    """Fresh test 4's files are the sealed ones before any row is read: every hash in its manifest's files and in
    eval/fresh_test4_data_hashes.json (and the manifest's own sha256 there), the gold its gold manifest pins, 1,139 gold
    rows none missing; amendments 2, 4, 4b, 4c, 5 and 5b, the three manifests and the prompt pins committed unchanged;
    the pins amendments 5 (2, 4), 5b (5) and 4c (4, 4b, the gold manifest) carry equal the files now."""
    for f in (FRESH4_MANIFEST, FRESH4_DATA_HASHES, FRESH4_GOLD_MANIFEST, FRESH4_INPUTS, *FRESH4_AMENDMENTS):
        if not f.exists():
            raise SystemExit(f"{f.relative_to(ROOT)} is missing")
        if not committed(f):
            raise SystemExit(f"{f.relative_to(ROOT)} is not committed unchanged")
    for a in FRESH4_AMENDMENTS:
        am = json.loads(a.read_text())
        if a.name in ("protocol_amendment_4b.json", "protocol_amendment_4.json"):
            continue  # 4b pins the pre-extension manifests that 4c and the data-hash file supersede; 4 pins none
        for n, h in (am.get("amends") or {}).get("files", {}).items():
            if sha256(ROOT / n) != h:
                raise SystemExit(f"{n} is not the file {a.relative_to(ROOT)} amends")
    dh = json.loads(FRESH4_DATA_HASHES.read_text())
    if sha256(FRESH4_MANIFEST) != dh["manifest_sha256"]:
        raise SystemExit("eval/fresh_test4_manifest.json is not the manifest eval/fresh_test4_data_hashes.json records")
    man = json.loads(FRESH4_MANIFEST.read_text())
    for src, pins in (("eval/fresh_test4_manifest.json", man["files"]), ("eval/fresh_test4_data_hashes.json", dh["sha256"])):
        if not pins:
            raise SystemExit(f"{src} pins no file")
        for n, h in pins.items():
            if not (ROOT / n).exists():
                raise SystemExit(f"{n} is missing ({src} pins it)")
            if sha256(ROOT / n) != h:
                raise SystemExit(f"{n} is not the file {src} pinned")
    gm = json.loads(FRESH4_GOLD_MANIFEST.read_text())
    if sha256(FRESH4 / "gold.jsonl") != gm["gold_sha256"]:
        raise SystemExit("data/built/fresh_test4/gold.jsonl is not the gold eval/fresh_test4_gold_manifest.json pins")
    for n, h in gm.get("input_sha256", {}).items():
        if sha256(ROOT / n) != h:
            raise SystemExit(f"{n} is not the file eval/fresh_test4_gold_manifest.json pinned")
    if gm.get("n_gold") != man["n_rows"] or gm.get("missing_rows"):
        raise SystemExit(f"fresh test 4 gold holds {gm.get('n_gold')} rows (missing {gm.get('missing_rows')}), "
                         f"the manifest {man['n_rows']}")
    inp = json.loads(FRESH4_INPUTS.read_text())
    prompts = {**inp["B"], **inp["h"], **inp["base"]}
    for n, h in prompts.items():
        if sha256(ROOT / n) != h:
            raise SystemExit(f"{n} is not the prompt file eval/fresh_test4_eval_inputs.json pins")
    return {"n_rows": man["n_rows"], "gold_sha256": gm["gold_sha256"], "prompt_files": prompts}


def read_raw4(path: Path, ids: list[str], tag: str) -> np.ndarray:
    """rows x 3 float scores in `ids` order after read_scores' checks (every row once, finite, a stored pred equal to
    the label its scores read), from a colab_eval run on test 4's prompt file in bf16 at batch 1 with chat framing."""
    d = json.loads(Path(path).read_text())
    for k, v in FRESH4_SCORE_HEADER.items():
        if d.get(k) != v:
            raise SystemExit(f"{path}: {k} is {d.get(k)!r}, not {v!r}")
    if (d.get("timing") or {}).get("batch") != 1:
        raise SystemExit(f"{path}: batch {(d.get('timing') or {}).get('batch')}, not 1")
    want_model = "base_Qwen--Qwen3.5-4B" if tag == "base" else f"merged_{tag}"
    if d.get("model") != want_model:
        raise SystemExit(f"{path}: model {d.get('model')!r}, not {want_model!r}")
    read_scores(path, ids)
    by = {r.get("pair_id", r.get("row_id")): [float(x) for x in r["scores"]] for r in d["rows"]}
    return np.array([by[i] for i in ids], dtype=np.float64)


def write_rows4(path: Path, rows: list[dict], scores: dict[str, np.ndarray], P: dict[str, list[str]], svals: dict) -> None:
    """One line per row for an independent recount: gold, labelers, verdict, the slice flags, every model's raw
    scores and decider s, and every system's label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ("posting_id", "gold", "label_A", "label_B", "agreed", "eligibility_pass", "eligibility_pass_design",
            "has_card", "employer", "employer_in_fresh_test1_3", "extension_4b", "teaser", "empty_body", "kind",
            "pool_field", "posting_length", "rules_full")
    with path.open("w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"row_id": r["row_id"], **{k: r[k] for k in keys},
                                "verdict": {k: r["verdict"][k] for k in ("rule", "hard_cats", "salary_soft")},
                                "scores": {t: [float(x) for x in S[i]] for t, S in scores.items()},
                                "s": {t: float(s[i]) for t, s in svals.items()},
                                "systems": {k: v[i] for k, v in P.items()}}) + "\n")


def cmd_fresh4(a: argparse.Namespace) -> None:
    for f in (FRESH4_RESULT, FRESH4_RESULT.with_suffix(".md"), FRESH4_ROWS_OUT):
        if f.exists():
            raise SystemExit(f"{f.relative_to(ROOT)} exists: fresh test 4 is scored once")
    spec = read_spec4(Path(a.selection))
    if not FRESH4_GOLD_MANIFEST.exists() or not committed(FRESH4_GOLD_MANIFEST):
        raise SystemExit("eval/fresh_test4_gold_manifest.json is not committed: fresh test 4 has no sealed gold")
    pins = check_fresh4()
    rows = load_fresh4()
    if len(rows) != pins["n_rows"]:
        raise SystemExit("fresh test 4 rows.jsonl does not hold the manifest's n_rows")
    paths = parse_scores(a.score)
    want = {spec["pick"]["model"], *FRESH4_NEEDS}
    if set(paths) != want:
        raise SystemExit(f"fresh test 4 takes score files for exactly {sorted(want)}, got {sorted(paths)}")
    ids = [r["row_id"] for r in rows]
    scores = {t: read_raw4(p, ids, t) for t, p in paths.items()}
    res, P, svals = score_fresh4(rows, scores, spec)

    def rel(f: Path) -> str:
        f, top = Path(f).resolve(), ROOT.resolve()
        return str(f.relative_to(top) if f.is_relative_to(top) else f)

    write_rows4(FRESH4_ROWS_OUT, rows, scores, P, svals)
    res["selection_file"] = spec["selection"]
    res["score_files"] = {t: {"path": rel(p), "sha256": sha256(Path(p))} for t, p in paths.items()}
    res["prompt_files"] = pins["prompt_files"]
    res["gold_sha256"] = pins["gold_sha256"]
    res["per_row_file"] = {"path": rel(FRESH4_ROWS_OUT), "sha256": sha256(FRESH4_ROWS_OUT)}
    if a.session:
        res["colab_sessions"] = a.session
    write_once(FRESH4_RESULT, res)


def _kr(k: dict) -> str:
    return f"{k['recall']:.3f} ({k['found']}/{k['gold_keeps']})" if k["recall"] is not None else f"- (0/{k['gold_keeps']})"


def markdown4(res: dict) -> str:
    p = res["pass"]
    L = ["# fresh_test4: the cycle-3 pick, scored once", "",
         f"Pick **{res['chosen']}** (cut-offs t_lo {res['spec']['pick'].get('t_lo')}, t_hi "
         f"{res['spec']['pick'].get('t_hi')}); base comparator {res['base_comparator']} (t_lo "
         f"{res['spec']['base'].get('t_lo')}, t_hi {res['spec']['base'].get('t_hi')}).",
         f"Two-way bootstrap (postings, then rows within them), {res['draws']} draws, seed {res['seed']}, one draw set "
         "per slice; 95% percentile intervals. Keep recall is shown with found / gold Keeps (amendment 4c).", "",
         "## Pass bar (amendment 2, on the eligibility-pass slice)", "",
         f"**{'PASS' if p['passed'] else 'NOT PASSED'}** (n={p['n']}, postings={p['postings']}, gold {p['gold']})", ""]
    L += [f"- {p['system']} minus {b}: macro-F1 {fmt(v['macro_f1_diff'])}, lower bound {v['lower_bound_exact']:.6f} -> "
          f"{'beaten' if v['beaten'] else 'not beaten'} (accuracy diff, reported only: "
          f"{fmt(v['accuracy_diff (reported, not for pass)'])})" for b, v in p["against"].items()]
    L.append("")
    ag = res["agreement_subset (reported, not for pass)"]
    L += ["## Agreement subset (labelers A and B agree; reported, not for pass)", ""]
    for k in ("agreed_eligibility_pass", "agreed_only"):
        x = ag[k]
        if not x["against"]:
            L.append(f"- {k}: empty")
            continue
        L += [f"- {k} (n={x['n']}, gold {x['gold']}): {p['system']} minus {b}: {fmt(v['macro_f1_diff'])}, lower bound "
              f"{v['lower_bound_exact']:.6f} -> {'beaten' if v['beaten'] else 'not beaten'}" for b, v in x["against"].items()]
    L.append("")
    for name, h in res["headline"].items():
        L += [f"## {name} (n={h['n']}, postings={h['postings']}, gold {h['gold']})", "",
              "| system | macro-F1 | accuracy | Keep recall (found/gold) | Explore recall | Skip recall | Keep precision | unsafe Keeps |",
              "|---|---|---|---|---|---|---|---|"]
        for k, v in h["systems"].items():
            kp = v["keep"]["precision"]
            L.append(f"| {k} | {fmt(v['macro_f1'])} | {fmt(v['accuracy'])} | {_kr(v['keep'])} | "
                     f"{v['recall']['Explore']} | {v['recall']['Skip']} | "
                     f"{kp if kp is not None else '-'} ({v['keep']['found']}/{v['keep']['predicted_keeps']}) | "
                     f"{v['unsafe_keep']} |")
        c = res["slices"][name].get("labeler_ceiling")
        if c:
            L += [f"| labeler ceiling (A vs B) | | {c['raw_agreement']:.3f} (kappa {c['kappa']}) | | | | | |"]
        L.append("")
    ne = res["new_employer_slice"]
    L += ["## New-employer slice", "",
          f"{ne['slice']}: n={ne['n']}, postings={ne['postings']}, employers {ne['employers']}, eligibility-pass rows "
          f"{ne['eligibility_pass_rows']}, gold {ne['gold']}. {ne['flag']}.", ""]
    L += [f"- {k}: macro-F1 {fmt(v)}" for k, v in ne["macro_f1"].items()]
    L.append("")
    sd = res["sealed_design_slice (sensitivity, not for pass)"]
    L += [f"Sealed-design eligibility-pass slice (sensitivity, not for pass; n={sd['n']}, gold {sd['gold']}):", ""]
    L += [f"- {p['system']} minus {b}: {fmt(v['macro_f1_diff'])}, lower bound {v['lower_bound_exact']:.6f} -> "
          f"{'would be beaten' if v['would_be_beaten'] else 'would not be beaten'}" for b, v in sd["against"].items()]
    L.append("")
    ev = res["eligibility_vs_sealed_design"]
    nc = res["no_card_rows"]
    L += [f"Eligibility pass recomputed: {ev['recomputed_pass_rows']} rows; sealed design: {ev['sealed_design_pass_rows']}; "
          f"moved out {ev['rows_moved_out']} ({ev['moved_out_by']}), moved in {ev['rows_moved_in']}.",
          f"Rows with no extracted card: {len(nc['rows'])} (gold {nc['gold']}); {nc['policy']}.", ""]
    me, base, hh = res["chosen"], res["base_comparator"], "h+H'"
    L += ["## Slices (reported, not for pass)", "",
          f"| slice | n | gold Keeps | {me} | rules_full | {base} | h+H' | pick - rules_full | pick - base |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, sl in res["slices"].items():
        if sl is None:
            L.append(f"| {name} | 0 | | | | | | | |")
            continue
        S, D = sl["systems"], sl["paired"]
        L.append(f"| {name} | {sl['n']} | {sl['gold'].get('Keep', 0)} | {fmt(S[me]['macro_f1'])} | "
                 f"{fmt(S['rules_full']['macro_f1'])} | {fmt(S[base]['macro_f1'])} | {fmt(S[hh]['macro_f1'])} | "
                 f"{fmt(D[me + ' - rules_full']['macro_f1_diff'])} | {fmt(D[me + ' - ' + base]['macro_f1_diff'])} |")
    L.append("")
    if res.get("colab_sessions"):
        L += [f"Colab sessions: {', '.join(res['colab_sessions'])}.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dev")
    d.add_argument("--score", action="append", required=True, help="TAG=RESULT.json")
    d.add_argument("--out", default=str(OUT))
    d.set_defaults(fn=cmd_dev)
    f = sub.add_parser("fresh")
    f.add_argument("--score", action="append", required=True, help="TAG=RESULT.json")
    f.add_argument("--selection", required=True, help="the committed selection json whose 'pick' is scored")
    f.add_argument("--out", default=str(OUT))
    f.set_defaults(fn=cmd_fresh)
    s = sub.add_parser("select")
    s.add_argument("--score", action="append", required=True, help="TAG=RESULT.json on fresh test 1 (h and h2*)")
    s.add_argument("--out", required=True, help="the selection json to write (refused if it exists)")
    s.add_argument("--archive", action="append", help="TAG=ARCHIVE.tar.gz: record the adapter archive's sha256")
    s.add_argument("--note", action="append", help="a note recorded in the selection json")
    s.set_defaults(fn=cmd_select)
    f2 = sub.add_parser("fresh2")
    f2.add_argument("--score", action="append", required=True, help="TAG=RESULT.json on fresh test 2")
    f2.add_argument("--selection", required=True, help="the committed amendment-2 selection json")
    f2.set_defaults(fn=cmd_fresh2)
    f3 = sub.add_parser("fresh3")
    f3.add_argument("--score", action="append", required=True, help="TAG=RESULT.json on fresh test 3")
    f3.add_argument("--selection", required=True, help="the committed selection json naming the pick and amendment 3")
    f3.set_defaults(fn=cmd_fresh3)
    f4 = sub.add_parser("fresh4")
    f4.add_argument("--score", action="append", required=True, help="TAG=RESULT.json on fresh test 4 (the pick's model, h, base)")
    f4.add_argument("--selection", default=str(SELECTION_C3), help="the committed cycle-3 selection json")
    f4.add_argument("--session", action="append", help="a Colab session the score files came from (recorded)")
    f4.set_defaults(fn=cmd_fresh4)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
