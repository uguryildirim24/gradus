"""Shortcut/leakage checks: cheap baselines and a swap-invariance harness.

If a bag-of-words model on only the resume or only the posting scores well,
the labels leak through surface text. The full model should do better than
either half, and a shuffled-posting control should fall toward chance.
"""
from __future__ import annotations

import random
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

RESUME, POSTING, LABEL, ID = "resume", "posting", "label", "id"
PROFILE = "profile"  # optional: the candidate profile block, its own text column when present


def load_table(path: str | Path) -> pd.DataFrame:
    """Rows with columns resume, posting, label (and optional id)."""
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in (".jsonl", ".json"):
        df = pd.read_json(p, lines=p.suffix == ".jsonl")
    else:
        df = pd.read_csv(p)
    missing = {RESUME, POSTING, LABEL} - set(df.columns)
    if missing:
        raise ValueError(f"{p}: missing columns {sorted(missing)}")
    if ID not in df.columns:
        df[ID] = np.arange(len(df)).astype(str)
    df[ID] = df[ID].astype(str)
    return df


def load_splits(path: str | Path) -> dict[str, str]:
    """File with columns id, split -> {id: split}."""
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in (".jsonl", ".json"):
        df = pd.read_json(p, lines=p.suffix == ".jsonl")
    else:
        df = pd.read_csv(p)
    return dict(zip(df[ID].astype(str), df["split"].astype(str)))


def split_rows(df, splits, train="train", dev="dev"):
    s = df[ID].map(splits)
    return df[s == train].reset_index(drop=True), df[s == dev].reset_index(drop=True)


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _words(t: str) -> int:
    return len(t.split())


def length_features(df) -> np.ndarray:
    r = df[RESUME].map(_words).to_numpy(float)
    p = df[POSTING].map(_words).to_numpy(float)
    return np.column_stack([np.log1p(r), np.log1p(p), np.log1p(r) - np.log1p(p)])


def _fit_predict_text(train, dev, cols, dev_posting=None):
    mats_tr, mats_dev = [], []
    for c in cols:
        v = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True, max_features=50000)
        try:
            xt = v.fit_transform(train[c])
        except ValueError:  # empty vocabulary (tiny fixtures)
            v = TfidfVectorizer(sublinear_tf=True)
            xt = v.fit_transform(train[c])
        src = dev_posting if (c == POSTING and dev_posting is not None) else dev[c]
        mats_tr.append(xt)
        mats_dev.append(v.transform(src))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(hstack(mats_tr).tocsr(), train[LABEL])
    return clf.predict(hstack(mats_dev).tocsr())


def text_columns(df) -> list[str]:
    return [RESUME, *([PROFILE] if PROFILE in df.columns else []), POSTING]


def full_name(cols) -> str:
    return "BoW: " + " + ".join(cols)


def shuffle_postings(df, seed: int) -> list[str]:
    """Re-pair postings randomly within the frame, keeping the multiset."""
    rng = random.Random(seed)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    return df[POSTING].iloc[idx].tolist()


def run_baselines(train, dev, seed: int = 0, shuffle_repeats: int = 5) -> list[dict]:
    y = dev[LABEL]
    rows = []

    def add(name, f1, note=""):
        rows.append({"baseline": name, "macro_f1": f1, "note": note})

    maj = train[LABEL].value_counts().idxmax()
    add("majority class", macro_f1(y, [maj] * len(dev)), f"always '{maj}'")

    def length_model(cols):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(length_features(train)[:, cols], train[LABEL])
        return macro_f1(y, clf.predict(length_features(dev)[:, cols]))

    add("length: resume only", length_model([0]))
    add("length: posting only", length_model([1]))
    add("length: resume + posting + ratio", length_model([0, 1, 2]))
    cols = text_columns(train)
    for c in cols:
        add(f"BoW: {c} only", macro_f1(y, _fit_predict_text(train, dev, [c])))
    if PROFILE in cols:
        add("BoW: resume + profile", macro_f1(y, _fit_predict_text(train, dev, [RESUME, PROFILE])),
            "the candidate side alone")
    add(full_name(cols), macro_f1(y, _fit_predict_text(train, dev, cols)))

    scores = []
    for k in range(shuffle_repeats):
        sp = shuffle_postings(dev, seed + k)
        scores.append(macro_f1(y, _fit_predict_text(train, dev, cols, dev_posting=sp)))
    add("shuffled-posting control (full BoW)", float(np.mean(scores)),
        f"mean of {shuffle_repeats} shuffles of dev postings; should fall toward chance")
    return rows


def chance_f1(labels) -> float:
    k = len(set(labels))
    return 1.0 / k if k else 0.0


def report_markdown(rows, n_train, n_dev, dev_labels) -> str:
    counts = pd.Series(list(dev_labels)).value_counts().to_dict()
    out = [
        "# Leakage report", "",
        f"Train rows: {n_train}. Dev rows: {n_dev}. Dev label counts: {counts}.",
        f"Uniform-guess macro-F1 is about {chance_f1(dev_labels):.2f}.", "",
        "| Baseline | Dev macro-F1 | Note |", "|---|---|---|",
    ]
    out += [f"| {r['baseline']} | {r['macro_f1']:.3f} | {r['note']} |" for r in rows]
    by = {r["baseline"]: r["macro_f1"] for r in rows}
    flags = []
    for k in ("BoW: resume only", "BoW: profile only", "BoW: posting only", "length: resume + posting + ratio"):
        if k in by and by[k] > by["majority class"] + 0.15:
            flags.append(f"- {k} reaches {by[k]:.2f}, well above majority: surface text carries the label.")
    full = next((v for k, v in by.items() if k.startswith("BoW: resume + ") and k.endswith("posting")), None)
    shuf = by.get("shuffled-posting control (full BoW)")
    if full is not None and shuf is not None and shuf > full - 0.05 and full > by["majority class"] + 0.05:
        flags.append("- Shuffled-posting control did not collapse: the model ignores the pairing.")
    out += ["", "## Flags", ""] + (flags or ["- None."])
    return "\n".join(out) + "\n"


# --- swap invariance --------------------------------------------------------

# Fixed lists. Each entry is (original marker, replacement). Matching is
# case-sensitive on whole phrases; every occurrence in the resume is replaced.
COUNTRY_SWAPS = [
    ("Turkey", "Canada"), ("Türkiye", "Canada"), ("Turkish", "Canadian"),
    ("India", "Germany"), ("Indian", "German"),
    ("United States", "Brazil"), ("USA", "Brazil"), ("American", "Brazilian"),
    ("China", "Kenya"), ("Chinese", "Kenyan"),
    ("Nigeria", "Japan"), ("Nigerian", "Japanese"),
]
SCHOOL_SWAPS = [
    ("Istanbul", "Toronto"), ("Ankara", "Munich"), ("Izmir", "Montreal"),
    ("Boston", "Sao Paulo"), ("Newton, MA", "Nairobi"), ("Mumbai", "Berlin"),
    ("Beijing", "Lagos"), ("Lagos", "Osaka"),
]
LANGUAGE_SWAPS = [
    ("Turkish", "French"), ("Hindi", "Spanish"), ("Mandarin", "Portuguese"),
    ("Yoruba", "Swahili"), ("Arabic", "Dutch"),
]
SWAP_KINDS = {"country": COUNTRY_SWAPS, "school": SCHOOL_SWAPS, "language": LANGUAGE_SWAPS}


def apply_swaps(text: str, swaps) -> str:
    """Single-pass replacement so a replacement is never re-swapped."""
    pairs = {a: b for a, b in swaps}
    pat = re.compile("|".join(re.escape(a) for a in sorted(pairs, key=len, reverse=True)))
    return pat.sub(lambda m: pairs[m.group(0)], text)


def swap_invariance(model: Callable[[str, str], str], rows, kinds=None) -> dict:
    """Flip rate per swap kind: share of rows where the swap changed the model's bucket.

    Only rows whose resume actually contains a marker of that kind are counted.
    """
    kinds = kinds or list(SWAP_KINDS)
    res = {}
    for kind in kinds:
        swaps = SWAP_KINDS[kind]
        n = flips = 0
        for resume, posting in rows:
            swapped = apply_swaps(resume, swaps)
            if swapped == resume:
                continue
            n += 1
            flips += model(resume, posting) != model(swapped, posting)
        res[kind] = {"affected": n, "flips": flips, "flip_rate": flips / n if n else None}
    return res


def swap_markdown(res) -> str:
    out = ["## Swap invariance", "", "| Swap | Rows with marker | Flips | Flip rate |", "|---|---|---|---|"]
    for k, v in res.items():
        fr = "n/a" if v["flip_rate"] is None else f"{v['flip_rate']:.3f}"
        out.append(f"| {k} | {v['affected']} | {v['flips']} | {fr} |")
    return "\n".join(out) + "\n"
