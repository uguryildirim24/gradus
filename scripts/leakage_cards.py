"""Leakage baselines on the pair manifest, before any text is rendered (0 credits).

Each row gets three proxy texts built from its cards: the resume facts (the fields the
resume is rendered from), the candidate profile (render_profile, exactly what the prompt
shows) and the posting facts (the posting's card as the renderer sees it). A TF-IDF +
logistic-regression model is fit on train rows from ONE of these texts and scored on
dev + test rows. If one side alone predicts the bucket well, the label leaks through it.

Gate: for each single side, accuracy stays within --margin (0.10) of the majority-class
rate, and balanced macro-F1 within --margin of uniform guessing (1/3). Reports the
all-sides fit too. The macro-F1 column uses matcher.leakage.macro_f1 (read-only); the fits
mirror its TF-IDF + logistic-regression baselines on a third input column.

A linear model on words misses interactions ("remote only" and "internship wanted" together),
so a boosted-tree probe (HistGradientBoosting) also reads the structured candidate fields
alone (degree, graduation year, years, work authorization, every profile preference, skill
and license counts) and the structured posting fields alone, under the same gate. The
builder steers picks on those same candidate fields (fingerprints.py), so a third probe adds
what it does not steer on: a bag of the 200 commonest skills and licenses, the home metro
and the degree field. On the v3 integrate manifest the candidate probes sit at +0.03 to +0.08
accuracy and +0.04 to +0.09 macro-F1 over chance across build seeds 0-3, under the 0.10 gate.
Before the builder steered on salary floor, industries wanted, remote preference and work
authorization together (and on relocation x enrollment x degree, industries x years), the
candidate-field probe read +0.102 macro-F1 on seed 0 and the edge probe +0.107 on seed 0 and +0.103
on seed 3 once CPT students stopped avoiding "intern"; every seed 0-3 must pass, not one picked.

A fourth row, the edge probe, reads every candidate field one-hot (each skill, license,
relocation metro, wanted or avoided industry, avoided role and job type with at least
ALL_MIN_HITS train rows, plus the salary floor, years, graduation year and the
years-vs-graduation gap) with a deeper fit. It is the strongest candidate-only probe found so
far. It was over the margin before the builder stopped steering on fit features and pulled
citizens' bucket mix level across posting stances (+0.11 at d9d423a); it is now +0.04 to +0.08
accuracy and +0.05 to +0.08 macro-F1 across seeds 0-3, so it is gated with the others. The
signal is spread over many preference interactions (salary floor, number of industries
wanted, enrollment, work authorization, years and roles avoided each carry a little). The
200-column skill bag makes the third probe weaker than the first, not stronger.

uv run python scripts/leakage_cards.py [--pairs data/built/pairs] [--out data/built/pairs/leakage_cards.md]
(about 35 s; OMP_NUM_THREADS=4 halves it on a laptop, where the boosted trees fight over cores)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import hstack  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.feature_extraction import DictVectorizer  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

from matcher import leakage as L  # noqa: E402
from matcher.degrees import canon_field  # noqa: E402
from matcher.metros import metro_of  # noqa: E402
from matcher.profile import render_profile  # noqa: E402
from matcher.rules import is_enrolled  # noqa: E402
from matcher.schema import (  # noqa: E402
    CURRENT_YEAR, DEGREE_LEVELS, EMPLOYMENT_TYPES, REMOTE_PREFS, SENIORITIES, SPONSORSHIP, WORK_AUTH, WORK_MODES,
    CandidateCard,
    RequirementCard, skill_canon,
)

RESUME_FIELDS = ("field", "degree_level", "degree_field", "graduation_year", "years", "skills", "licenses",
                 "current_location", "education_country", "language")
POSTING_DROP = ("source_posting_id", "onet_soc", "related_ok")  # related_ok: a card-side flag, as in render_cards


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in open(p) if x.strip()] if p.exists() else []


def flat(d: dict) -> str:
    """'key: value' lines; lists joined, so every value word is a token."""
    out = []
    for k, v in d.items():
        if isinstance(v, (list, tuple)):
            v = "; ".join(" or ".join(x) if isinstance(x, (list, tuple)) else str(x) for x in v)
        out.append(f"{k}: {v}")
    return "\n".join(out)


def n_preferences(c: CandidateCard) -> int:
    """How many profile preferences are set (salary floor, job types, industries, roles, remote, relocation)."""
    return sum([c.salary_floor is not None, bool(c.employment_types_wanted), bool(c.industries_wanted),
                bool(c.industries_avoided), bool(c.roles_avoided), c.remote_preference != "any",
                bool(c.willing_locations)])


def candidate_features(c: CandidateCard) -> dict[str, float]:
    """Structured candidate-only fields, as numbers, for the boosted probe."""
    return {
        "salary_floor": c.salary_floor if c.salary_floor is not None else -1,
        "n_types": len(c.employment_types_wanted),
        **{f"wants_{t}": t in c.employment_types_wanted for t in EMPLOYMENT_TYPES},
        "n_industries_wanted": len(c.industries_wanted), "n_industries_avoided": len(c.industries_avoided),
        "n_roles_avoided": len(c.roles_avoided), "remote": REMOTE_PREFS.index(c.remote_preference),
        "n_willing": len(c.willing_locations), "anywhere": any("anywhere" in x for x in c.willing_locations),
        "work_auth": WORK_AUTH.index(c.work_auth), "degree_level": DEGREE_LEVELS.index(c.degree_level),
        "enrolled": is_enrolled(c), "graduation_year": c.graduation_year or -1, "years": c.years,
        "n_skills": len(c.skills), "n_licenses": len(c.licenses), "n_preferences": n_preferences(c),
    }


def all_candidate_fields(c: CandidateCard) -> dict[str, float]:
    """Every candidate field the builder sets, one-hot where it is a list, for the edge probe."""
    gy = c.graduation_year
    d: dict[str, float] = {
        "degree_level=" + c.degree_level: 1,
        "degree_field=" + (canon_field(c.degree_field) if c.degree_field else "-"): 1,
        "work_auth=" + c.work_auth: 1, "remote=" + c.remote_preference: 1,
        "home=" + (metro_of(c.current_location) or "other"): 1, "enrolled": float(is_enrolled(c)),
        "graduation_year": gy if gy is not None else -1, "years": c.years,
        "gap": (CURRENT_YEAR - gy - c.years) if gy is not None and gy <= CURRENT_YEAR else -99,
        "salary_floor": c.salary_floor if c.salary_floor is not None else -1,
        "job_types=" + ("/".join(sorted(c.employment_types_wanted)) or "any"): 1,
        "n_skills": len(c.skills), "n_willing": len(c.willing_locations), "n_preferences": n_preferences(c),
        "n_industries_wanted": len(c.industries_wanted), "n_industries_avoided": len(c.industries_avoided),
        "n_roles_avoided": len(c.roles_avoided),
    }
    for prefix, items in (("skill", (skill_canon(x) for x in c.skills)), ("license", c.licenses),
                          ("willing", (metro_of(x) or x for x in c.willing_locations)),
                          ("wants_type", c.employment_types_wanted), ("industry_wanted", c.industries_wanted),
                          ("industry_avoided", c.industries_avoided), ("role_avoided", c.roles_avoided)):
        for x in items:
            d[f"{prefix}={x}"] = 1
    return d


ALL_MIN_HITS = 20


def all_candidate_matrix(train: pd.DataFrame, test: pd.DataFrame):
    dv = DictVectorizer(sparse=False)
    xtr, xte = dv.fit_transform(train["c_all"]), dv.transform(test["c_all"])
    keep = (xtr != 0).sum(axis=0) >= ALL_MIN_HITS
    return xtr[:, keep], xte[:, keep]


def posting_features(r: RequirementCard) -> dict[str, float]:
    """Structured posting-only fields, as numbers, for the boosted probe."""
    return {
        "employment_type": EMPLOYMENT_TYPES.index(r.employment_type), "work_mode": WORK_MODES.index(r.work_mode),
        "sponsorship": SPONSORSHIP.index(r.sponsorship), "citizens_only": r.citizens_only, "e_verify": r.e_verify,
        "seniority": SENIORITIES.index(r.seniority), "degree_level": DEGREE_LEVELS.index(r.degree_level),
        "enrolled_in": DEGREE_LEVELS.index(r.enrolled_in), "n_degree_fields": len(r.degree_fields),
        "min_years": r.min_years, "n_must": len(r.must_skills), "n_nice": len(r.nice_skills),
        "licensed": bool(r.license), "n_locations": len(r.locations),
        "salary_max": r.salary_max if r.salary_max is not None else -1, "has_industry": bool(r.industry),
    }


def table(pairs: Path, cards: Path) -> pd.DataFrame:
    cd = {r["id"]: r["card"] for p in (cards / "candidate_cards.jsonl", pairs / "variant_candidates.jsonl")
          for r in read_jsonl(p)}
    rq = {r["id"]: r["card"] for p in (cards / "requirement_cards.jsonl", pairs / "variant_requirements.jsonl")
          for r in read_jsonl(p)}
    rows = []
    for r in read_jsonl(pairs / "manifest.jsonl"):
        c, card = cd[r["cd_id"]], rq[r["rq_id"]]
        cand, req = CandidateCard.from_dict(c), RequirementCard.from_dict(card)
        rows.append({
            "id": r["pair_id"], "split": r["split"], "label": r["bucket"],
            "resume": flat({k: c[k] for k in RESUME_FIELDS if k in c}),
            "profile": render_profile(cand),
            "posting": flat({k: v for k, v in req.to_dict().items() if k not in POSTING_DROP}),
            **{f"c.{k}": float(v) for k, v in candidate_features(cand).items()},
            **{f"p.{k}": float(v) for k, v in posting_features(req).items()},
            "skill_keys": tuple(skill_canon(s) for s in (*cand.skills, *cand.licenses)),
            "home": metro_of(cand.current_location) or cand.current_location,
            "degree_field": canon_field(cand.degree_field) if cand.degree_field else "",
            "c_all": all_candidate_fields(cand),
        })
    return pd.DataFrame(rows)


BAG_SIZE = 200


def rich_candidate_matrix(train: pd.DataFrame, test: pd.DataFrame):
    """(X_train, X_test, categorical mask): the candidate fields plus a bag of the BAG_SIZE commonest
    train skills and licenses, the home metro and the degree field (as categories)."""
    vocab = [k for k, _ in Counter(k for keys in train["skill_keys"] for k in keys).most_common(BAG_SIZE)]
    idx = {k: i for i, k in enumerate(vocab)}
    cats = {col: {v: i for i, v in enumerate(sorted(set(train[col])))} for col in ("home", "degree_field")}
    cols = [c for c in train.columns if c.startswith("c.")]

    def mat(d: pd.DataFrame) -> np.ndarray:
        bag = np.zeros((len(d), len(vocab)))
        for i, keys in enumerate(d["skill_keys"]):
            for k in keys:
                if k in idx:
                    bag[i, idx[k]] = 1
        extra = np.array([[cats[col].get(v, len(cats[col])) for col, v in (("home", h), ("degree_field", f))]
                          for h, f in zip(d["home"], d["degree_field"])], dtype=np.float64)
        return np.hstack([d[cols].to_numpy(np.float64), bag, extra])

    return mat(train), mat(test), [False] * (len(cols) + len(vocab)) + [True, True]


def fit_trees(train: pd.DataFrame, test: pd.DataFrame, prefix: str, balanced: bool):
    if prefix == "all":
        xtr, xte = all_candidate_matrix(train, test)
        clf = HistGradientBoostingClassifier(random_state=0, max_iter=300, learning_rate=0.05,
                                             class_weight="balanced" if balanced else None)
        return clf.fit(xtr, train["label"]).predict(xte)
    if prefix == "rich":
        xtr, xte, cat = rich_candidate_matrix(train, test)
    else:
        cols = [c for c in train.columns if c.startswith(prefix)]
        xtr, xte, cat = train[cols].to_numpy(np.float64), test[cols].to_numpy(np.float64), None
    clf = HistGradientBoostingClassifier(random_state=0, class_weight="balanced" if balanced else None,
                                         categorical_features=cat)
    clf.fit(xtr, train["label"])
    return clf.predict(xte)


def fit(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], balanced: bool):
    mats_tr, mats_te = [], []
    for c in cols:
        v = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True, max_features=50000)
        mats_tr.append(v.fit_transform(train[c]))
        mats_te.append(v.transform(test[c]))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced" if balanced else None)
    clf.fit(hstack(mats_tr).tocsr(), train["label"])
    return clf.predict(hstack(mats_te).tocsr())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=ROOT / "data/built/pairs")
    ap.add_argument("--cards", default=ROOT / "data/built/cards")
    ap.add_argument("--out", default=None)
    ap.add_argument("--margin", type=float, default=0.10)
    a = ap.parse_args(argv)
    pairs, cards = Path(a.pairs), Path(a.cards)
    df = table(pairs, cards)
    train = df[df.split == "train"].reset_index(drop=True)
    test = df[df.split.isin(["dev", "test"])].reset_index(drop=True)
    y = test["label"]
    majority = y.value_counts(normalize=True).max()
    lines = [
        "# Leakage baselines on card proxy text",
        "",
        f"Train rows {len(train)}, scored on dev + test rows {len(test)}. Majority-class rate {majority:.3f}; "
        f"uniform-guess macro-F1 0.333. Gate: each single side within {a.margin:.2f} of both.",
        "Texts are built from the cards (resume fields, render_profile, posting card), not rendered prose.",
        "",
        "| Input | Accuracy | Acc - majority | Macro-F1 (balanced) | F1 - 1/3 | Gate |",
        "|---|---|---|---|---|---|",
    ]
    failed = []
    for name, cols in (("profile only", ["profile"]), ("resume only", ["resume"]), ("posting only", ["posting"]),
                       ("resume + profile", ["resume", "profile"]), ("all three", ["resume", "profile", "posting"])):
        acc = accuracy_score(y, fit(train, test, cols, balanced=False))
        f1 = L.macro_f1(y, fit(train, test, cols, balanced=True))
        single = len(cols) == 1
        ok = bool(acc - majority <= a.margin and f1 - 1 / 3 <= a.margin) if single else None
        if ok is False:
            failed.append(name)
        lines.append(f"| {name} | {acc:.3f} | {acc - majority:+.3f} | {f1:.3f} | {f1 - 1 / 3:+.3f} | "
                     f"{'pass' if ok else 'FAIL' if ok is False else '(not gated)'} |")
    for name, prefix in (("candidate fields, boosted trees", "c."),
                         ("candidate fields + skills, home metro, degree field, boosted trees", "rich"),
                         ("posting fields, boosted trees", "p.")):
        acc = accuracy_score(y, fit_trees(train, test, prefix, balanced=False))
        f1 = L.macro_f1(y, fit_trees(train, test, prefix, balanced=True))
        ok = bool(acc - majority <= a.margin and f1 - 1 / 3 <= a.margin)
        if not ok:
            failed.append(name)
        lines.append(f"| {name} | {acc:.3f} | {acc - majority:+.3f} | {f1:.3f} | {f1 - 1 / 3:+.3f} | "
                     f"{'pass' if ok else 'FAIL'} |")
    name = "every candidate field one-hot, boosted trees (edge probe)"
    acc = accuracy_score(y, fit_trees(train, test, "all", balanced=False))
    f1 = L.macro_f1(y, fit_trees(train, test, "all", balanced=True))
    ok = bool(acc - majority <= a.margin and f1 - 1 / 3 <= a.margin)
    if not ok:
        failed.append(name)
    lines.append(f"| {name} | {acc:.3f} | {acc - majority:+.3f} | {f1:.3f} | {f1 - 1 / 3:+.3f} | "
                 f"{'pass' if ok else 'FAIL'} |")
    md = "\n".join(lines) + "\n"
    out = Path(a.out) if a.out else pairs / "leakage_cards.md"
    out.write_text(md)
    print(md)
    if failed:
        print("LEAKAGE GATE FAILED:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
