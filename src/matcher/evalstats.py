"""Metrics for the Keep / Explore / Skip matcher, and the one label read every scorer shares. numpy only:
the label read (LabelHead) calls methods of a model it is given and never imports torch.

Used by scripts/local_eval.py (model scoring), the demo scorers and scripts/build_real_test.py and baselines.py.
Scores are three floats per row, in LABELS order. Intervals resample postings (base_rq_id).
"""
from __future__ import annotations

import json
import math

import numpy as np

LABELS = ("Keep", "Explore", "Skip")
WORD = {"For you": "Keep", "Explore": "Explore", "Excluded": "Skip"}
TIE_ORDER = (2, 1, 0)  # Skip, Explore, Keep: the safer label wins an exact tie
N_BINS = 15

# Work-authorization reasons that make a row Excluded on their own (flag names from rules.py).
HARD_AUTH_FLAGS = ("work_auth:citizens_only", "work_auth:needs_sponsorship_no_sponsor", "work_auth:no_opt")
HARD_AUTH_TEXT = ("limited to US citizens or permanent residents", "will not sponsor visas and you need sponsorship",
                  "does not accept OPT or STEM OPT")

# Easy Keep kinds, each decided by profile fields alone against the posting header: the three the real
# test adds to reach its For you share (a relocation city, a wanted-industry list, a work-mode preference)
# and the training generator's own relocation, same-metro and harmless-preference Keeps (variants.py).
# Every headline is also given without them.
ADDED_KEEP_KINDS = ("relocate", "pref_mode_match", "industry_list", "location_willing", "location_anywhere",
                    "location_same_metro", "pref_remote_match", "pref_industry_avoided_other", "pref_role_avoided_other")
# Kinds whose label needs the resume read (skills; the degree field is in the resume and, since the profile
# gained its education line, in the profile too); the rest are decided from the profile and the posting header.
RESUME_KINDS = ("full_match", "must_skill", "must_skill_alternative", "must_skill_synonym", "nice_skill",
                "nice_skill_one_missing", "soft_skill_missing", "degree_field", "degree_field_adjacent",
                "degree_field_related")

# ---------------------------------------------------------------- the label read (model side)
# local_eval.py, the demo scorers (demo_backend_torch.py, Modal) and Modal's CPU reference all read a label
# through these, so the demo's label is the eval's label. torch is never imported here: the functions only
# call methods of the model and tensors they are given, so this module still loads with numpy alone.

def label_token_ids(tok) -> list[int]:
    """Token ids of Keep, Explore, Skip (LABELS order); each label must be one token."""
    ids = {w: tok.encode(w, add_special_tokens=False) for w in LABELS}
    assert all(len(v) == 1 for v in ids.values()), ids
    return [ids[w][0] for w in LABELS]


class LabelHead:
    """The three label scores as the eval reads them: the final hidden state (the output head's input, after
    the last norm) cast to float32, times the head's rows for the three label tokens, also in float32 (the
    embedding rows when the head is tied, as in Qwen3.5). In a bfloat16 model the logits themselves are
    bfloat16 (steps of 0.125 at magnitude 16-32), so two labels can tie there; these float32 scores almost
    never do. A forward hook on the head keeps the last input it saw."""

    def __init__(self, model, tok) -> None:
        head = model.get_output_embeddings()
        text_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        # the label scores are hidden @ head rows: a head bias or a final logit softcap would make them differ
        # from the model's own logits, so a model with either is refused
        assert getattr(head, "bias", None) is None, "output head has a bias: the float32 label scores would skip it"
        assert not getattr(text_cfg, "final_logit_softcapping", None), "final_logit_softcapping is set"
        self.tied = bool(getattr(model.config, "tie_word_embeddings", False)
                         or getattr(text_cfg, "tie_word_embeddings", False))
        self.ids = label_token_ids(tok)
        self.rows = head.weight[self.ids].detach().float()  # (3, hidden)
        self.last = None
        head.register_forward_hook(self._keep)

    def _keep(self, module, inp, out) -> None:
        self.last = inp[0].detach()

    def scores(self):
        """(B, 3) float32 label scores at the last position of the head's last call."""
        return self.last[:, -1].float() @ self.rows.T


def pick(scores) -> int:
    """Index of the largest score; an exact tie goes to Skip, then Explore, then Keep."""
    best = max(scores)
    for i in TIE_ORDER:
        if scores[i] == best:
            return i
    raise ValueError("empty scores")


# ---------------------------------------------------------------- pure metrics (numpy only)

def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll(scores: np.ndarray, gold: np.ndarray, temperature: float) -> float:
    p = softmax(scores, temperature)
    return float(-np.log(np.clip(p[np.arange(len(gold)), gold], 1e-12, 1.0)).mean())


def fit_temperature(scores: np.ndarray, gold: np.ndarray) -> float:
    """The temperature that minimizes the NLL of softmax(scores / T) (golden section on log T)."""
    lo, hi = math.log(0.05), math.log(50.0)
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = nll(scores, gold, math.exp(a)), nll(scores, gold, math.exp(b))
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = nll(scores, gold, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = nll(scores, gold, math.exp(b))
    return float(math.exp((lo + hi) / 2))


def top(probs: np.ndarray) -> np.ndarray:
    """The predicted label of each row, ties broken as pick() breaks them (Skip, Explore, Keep)."""
    return np.array([pick(p) for p in probs], dtype=int)


def ece(probs: np.ndarray, gold: np.ndarray, bins: int = N_BINS) -> float:
    """Expected calibration error of the top-label confidence, `bins` equal-width bins."""
    conf, pred = probs.max(axis=1), top(probs)
    hit = (pred == gold).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[0]) & (conf <= edges[1])
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def reliability(probs: np.ndarray, gold: np.ndarray, bins: int = N_BINS) -> list[dict]:
    conf, pred = probs.max(axis=1), top(probs)
    hit = (pred == gold).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[0]) & (conf <= edges[1])
        if m.any():
            out.append({"lo": round(edges[i], 4), "hi": round(edges[i + 1], 4), "n": int(m.sum()),
                        "confidence": round(float(conf[m].mean()), 4), "accuracy": round(float(hit[m].mean()), 4)})
    return out


def brier(probs: np.ndarray, gold: np.ndarray) -> float:
    """Multi-class Brier score: the squared error summed over the three labels, averaged over rows.
    Range 0 (perfect) to 2, not 0 to 1; quote it with that range."""
    onehot = np.eye(probs.shape[1])[gold]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def confusion(gold: np.ndarray, pred: np.ndarray) -> np.ndarray:
    m = np.zeros((3, 3), dtype=int)
    np.add.at(m, (gold, pred), 1)
    return m


def prf(conf: np.ndarray) -> dict[str, dict]:
    out = {}
    for i, w in enumerate(LABELS):
        tp, fp, fn = conf[i, i], conf[:, i].sum() - conf[i, i], conf[i, :].sum() - conf[i, i]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[w] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                  "f1": round(float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0, 4), "support": int(conf[i, :].sum())}
    return out


def macro_f1_of(gold, pred) -> float:
    """Macro-F1 over Keep / Explore / Skip; gold and pred are label words or indices (same type)."""
    gold, pred = np.asarray(gold), np.asarray(pred)
    f1s = []
    for w in (LABELS if gold.dtype.kind in "US" else range(3)):
        tp = int(((gold == w) & (pred == w)).sum())
        fp = int(((gold != w) & (pred == w)).sum())
        fn = int(((gold == w) & (pred != w)).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / 3


def cluster_ci(groups, fn, draws: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """(point, low, high): fn(index array) on all rows, and a 95% interval from resampling the
    groups (postings) with replacement. fn must return a float (nan is skipped)."""
    groups = np.asarray(groups)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    point = float(fn(np.arange(len(groups))))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        pick_g = rng.integers(0, len(uniq), len(uniq))
        v = fn(np.concatenate([members[g] for g in pick_g]))
        if not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cluster_bootstrap(gold, pred, groups, metric, draws: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    gold, pred = np.asarray(gold), np.asarray(pred)
    return cluster_ci(groups, lambda idx: metric(gold[idx], pred[idx]), draws, seed)


def base_of(row: dict) -> str:
    return row.get("base_rq_id") or str(row.get("rq_id", row["pair_id"])).split("~")[0]


def is_hard_auth(row: dict) -> bool:
    """Gold Skip with a work-authorization hard rule among its reasons."""
    if row["gold"] != "Skip":
        return False
    flags = row.get("flags")
    if flags is not None:
        return any(f in HARD_AUTH_FLAGS for f in flags)
    return any(t in row.get("completion", "") for t in HARD_AUTH_TEXT)


def ci(t: tuple[float, float, float]) -> dict:
    return {"value": round(t[0], 4), "ci95": [round(t[1], 4), round(t[2], 4)]}


def subset_f1(rows: list[dict], draws: int = 1000) -> dict | None:
    """Accuracy, macro-F1 (posting-clustered CI) and Keep recall of rows carrying gold and pred."""
    if not rows:
        return None
    gold = [r["gold"] for r in rows]
    pred = [r["pred"] for r in rows]
    groups = [base_of(r) for r in rows]
    keep = [p for g, p in zip(gold, pred) if g == "Keep"]
    return {"n": len(rows), "postings": len(set(groups)),
            "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(rows), 4),
            "macro_f1": ci(cluster_bootstrap(gold, pred, groups, macro_f1_of, draws)),
            "keep_recall": round(sum(p == "Keep" for p in keep) / len(keep), 4) if keep else None}


def side_headlines(rows: list[dict], draws: int = 1000) -> dict:
    """The headline again without the added Keep kinds, and on the resume-dependent kinds alone.
    Rows need gold, pred and variant_kind; rows without variant_kind are not in either subset."""
    if not any(r.get("variant_kind") for r in rows):
        return {}
    return {"without_added_keep_kinds": subset_f1([r for r in rows if r.get("variant_kind")
                                                   and r["variant_kind"] not in ADDED_KEEP_KINDS], draws),
            "resume_kinds": subset_f1([r for r in rows if r.get("variant_kind") in RESUME_KINDS], draws)}


def is_decidable(row: dict) -> bool:
    """False when the prompt does not show what the label rests on (assemble's decidable, carried as
    visible by the evaluator)."""
    return bool(row.get("visible", row.get("decidable", True)))


def analyze(rows: list[dict], temperature: float | None = None, temperature_source: str | None = None,
            draws: int = 1000, include_undecidable: bool = False) -> dict:
    """All metrics from rows carrying gold, scores (three float32 label scores) and optionally
    pred_old, flags, variant_kind, visible.

    Rows marked not decidable (visible / decidable False) are left out of every number, as
    build_real_test.py score and baselines.py leave them out; `include_undecidable` keeps them, and
    the result always reports how many were kept and dropped and the all-rows accuracy and macro-F1."""
    all_rows = rows
    dropped = [r for r in rows if not is_decidable(r)]
    if not include_undecidable:
        rows = [r for r in rows if is_decidable(r)]
    if not rows:
        raise ValueError(f"no decidable rows to score ({len(all_rows)} rows, all not decidable)")
    gold = np.array([LABELS.index(r["gold"]) for r in rows])
    scores = np.array([r["scores"] for r in rows], dtype=np.float64)
    pred = np.array([pick(r["scores"]) for r in rows])
    groups = [base_of(r) for r in rows]
    if temperature is None:
        temperature = fit_temperature(scores, gold)
        temperature_source = "fitted on these rows (optimistic)"
    probs = softmax(scores, temperature)
    raw = softmax(scores, 1.0)
    conf = confusion(gold, pred)

    def acc(idx):
        return float((gold[idx] == pred[idx]).mean())

    def f1(idx):
        return macro_f1_of(gold[idx], pred[idx])

    hard = np.array([is_hard_auth(r) for r in rows])

    def viol(idx):
        h = hard[idx]
        return float((pred[idx][h] == 0).mean()) if h.any() else float("nan")

    ties = int(sum(sorted(r["scores"])[-1] == sorted(r["scores"])[-2] for r in rows))
    res: dict = {
        "n": len(rows), "postings": len(set(groups)),
        "rows": {"in_file": len(all_rows), "scored": len(rows), "not_decidable": len(dropped),
                 "not_decidable_left_out": not include_undecidable and bool(dropped)}, "temperature": round(temperature, 4),
        "temperature_source": temperature_source,
        "accuracy": ci(cluster_ci(groups, acc, draws)),
        "macro_f1": ci(cluster_ci(groups, f1, draws)),
        "ece15": ci(cluster_ci(groups, lambda i: ece(probs[i], gold[i]), draws)),
        "ece15_uncalibrated": round(ece(raw, gold), 4),
        "brier": ci(cluster_ci(groups, lambda i: brier(probs[i], gold[i]), draws)),
        "brier_uncalibrated": round(brier(raw, gold), 4),
        "nll": round(nll(scores, gold, temperature), 4), "nll_uncalibrated": round(nll(scores, gold, 1.0), 4),
        "per_class": prf(conf),
        "confusion": {LABELS[g]: {LABELS[p]: int(conf[g, p]) for p in range(3)} for g in range(3)},
        "pred_counts": {w: int((pred == i).sum()) for i, w in enumerate(LABELS)},
        "exact_ties": ties,
        "hard_filter": {"gold_skip_on_work_auth_hard_rule": int(hard.sum()),
                        "predicted_keep": int((pred[hard] == 0).sum()) if hard.any() else 0,
                        "violation_rate": ci(cluster_ci(groups, viol, draws)) if hard.any() else None},
        "unsafe_keep_rate": round(float(((gold != 0) & (pred == 0)).sum() / max((gold != 0).sum(), 1)), 4),
        "reliability": reliability(probs, gold),
    }
    if all("pred_old" in r for r in rows):
        old = np.array([LABELS.index(r["pred_old"]) for r in rows])
        res["agreement_with_old_argmax"] = round(float((old == pred).mean()), 4)
        res["accuracy_old_argmax"] = round(float((old == gold).mean()), 4)
    if dropped:
        g_all = [r["gold"] for r in all_rows]
        p_all = [LABELS[pick(r["scores"])] for r in all_rows]
        res["all_rows"] = {"n": len(all_rows), "accuracy": round(sum(a == b for a, b in zip(g_all, p_all)) / len(all_rows), 4),
                           "macro_f1": round(macro_f1_of(g_all, p_all), 4)}
    side = side_headlines([r | {"pred": LABELS[pred[i]]} for i, r in enumerate(rows)], draws)
    if side:
        res["side_headlines"] = side
    kinds = [r.get("variant_kind") for r in rows]
    if any(kinds):
        res["per_kind_accuracy"] = {k: {"n": int(sum(x == k for x in kinds)),
                                        "accuracy": round(float(np.mean([gold[i] == pred[i] for i, x in enumerate(kinds) if x == k])), 4)}
                                    for k in sorted({k for k in kinds if k})}
    return res


def report(res: dict) -> None:
    def line(name, d):
        print(f"{name:26s} {d['value']:.3f}  95% CI [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]")

    rs = res.get("rows", {})
    left = ("left out" if rs.get("not_decidable_left_out") else "included") if rs.get("not_decidable") else ""
    print(f"{res['n']} rows from {res['postings']} postings (intervals resample postings); "
          f"{rs.get('not_decidable', 0)} not decidable{' (' + left + ')' if left else ''} of {rs.get('in_file', res['n'])} "
          f"in the file; temperature {res['temperature']} ({res['temperature_source']})")
    line("accuracy", res["accuracy"])
    line("macro-F1", res["macro_f1"])
    for name, d in (res.get("side_headlines") or {}).items():
        if d:
            kr = f"; Keep recall {d['keep_recall']:.3f}" if d["keep_recall"] is not None else ""
            line(f"macro-F1, {name.replace('_', ' ')}", d["macro_f1"])
            print(f"{'':26s} accuracy {d['accuracy']:.3f} on {d['n']} rows{kr}")
    if "all_rows" in res:
        print(f"{'all rows (undecidable in)':26s} accuracy {res['all_rows']['accuracy']:.3f}, macro-F1 "
              f"{res['all_rows']['macro_f1']:.3f} on {res['all_rows']['n']} rows")
    line("ECE (15 bins, calibrated)", res["ece15"])
    print(f"{'ECE uncalibrated':26s} {res['ece15_uncalibrated']:.3f}")
    line("Brier (0-2, calibrated)", res["brier"])
    print(f"{'Brier uncalibrated':26s} {res['brier_uncalibrated']:.3f}; NLL {res['nll']} (uncalibrated {res['nll_uncalibrated']})")
    h = res["hard_filter"]
    if h["violation_rate"]:
        line(f"hard-filter violations", h["violation_rate"])
    print(f"  {h['predicted_keep']} of {h['gold_skip_on_work_auth_hard_rule']} gold-Skip work-auth rows predicted Keep; "
          f"unsafe Keep rate (gold not Keep, predicted Keep) {res['unsafe_keep_rate']:.3f}; exact ties {res['exact_ties']}")
    if "agreement_with_old_argmax" in res:
        print(f"agreement with the old argmax scoring {res['agreement_with_old_argmax']:.1%} "
              f"(old accuracy {res['accuracy_old_argmax']:.3f})")
    print("per label (precision / recall / F1 / support):")
    for w, d in res["per_class"].items():
        print(f"  {w:8s} {d['precision']:.3f} {d['recall']:.3f} {d['f1']:.3f} {d['support']}")
    print("confusion (rows gold, columns predicted):")
    print("         " + " ".join(f"{w:>8}" for w in LABELS))
    for g in LABELS:
        print(f"{g:>8} " + " ".join(f"{res['confusion'][g][p]:>8}" for p in LABELS))
    for k, d in res.get("per_kind_accuracy", {}).items():
        print(f"  kind {k:28s} n={d['n']:4d} accuracy {d['accuracy']:.3f}")
