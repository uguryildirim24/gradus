"""Cycle-2 training sets: careful-reading labels where fit decides, fixed rules where eligibility decides.

  build_fit_train2.py build     data/built/fit/fit2/pool.jsonl (every train_fit.csv row with its kind, label and
                                completion), then `variants`
  build_fit_train2.py variants  debias, then data/built/fit/train_fit2_bal.csv, train_fit2_nat.csv (prompt,
                                completion), data/built/fit/fit2/{rows.jsonl, report.json}
  build_fit_train2.py check     data/built/fit/fit2/checks.json: leakage guessers on each variant, fresh test 1 and 2
                                postings and employers against both variants

Fresh test 1 showed train_fit.csv's labels (rules on cards) disagree with careful reading where fit decides: hedged
and duty phrases read as must-skills, seniority, domain, work mode. This set keeps one definition per row:

  eligibility-decided  rules.label on the row's own card with the fixed rules (work mode a hard Skip, commit c6c4790)
                       is Skip with a hard reason in ELIG_CATS: work authorization, enrollment, degree level or field,
                       employment type, location, license, work mode. The label and completion are the fixed rules'
                       own (format_completion). This is the split H' acts on at inference, so a relabeled row the
                       fixed rules now Skip on work mode takes the rules Skip too (H' overrides the model there), and
                       a fit-dependent row that was never relabeled but the fixed rules now Skip on work mode joins
                       as a rules Skip.
  careful              every other row that labeler B relabeled (data/built/fit/relabel/labels/B, the rubric the
                       fresh-test gold uses) with confidence high or medium: B's label, completion "Label — reason."
                       with B's reason on one line (B writes in the third person, the rules in the second; kept as B
                       wrote it).
  dropped              B's low-confidence rows, and fit-dependent rows B never saw: no training set teaches both the
                       rules' and careful reading's definition of fit. Also dropped: every careful row of a B chunk
                       whose labels are scrambled (SCRAMBLED_CHUNKS, caught by scramble_check, which stops the build
                       on a new one), B Explores whose reason says a required item is not shown (NOT_SHOWN; the rubric
                       says Skip there, B drifts to Explore), and B Keeps or Skips on a thin posting (Explore by the
                       rubric).
  eligibility reasons  an eligibility completion names only its reasons in ELIG_CATS, never a skills or years reason
                       the rules wrote beside them.

Cards are the v2 cards the rows were built on: extractor v3 was not adopted (eval/fix_cycle2.json) and no v3 card
exists for a training posting. The rows and cards come from build_fit_train's row builders joined to train_fit.csv
by prompt, as relabel_prep does (about 7 minutes).

Debias (AFLite on the profile, before either variant is cut): B Skips a set preference the posting meets (an avoided
role or industry, pay under the floor) where the rules said Explore, and the generators drew preferences to be met, so
the profile alone guessed B's label (grouped-CV balanced accuracy about 0.47 unfiltered). Rows a profile-only guesser
finds easiest are dropped, each variant on its own, until it is at most DEBIAS_TO (see debias).

Variants (seeded):
  bal  equal Keep / Explore / Skip, each label at the smallest label's count. Within a label careful rows go first,
       eligibility rows fill the rest.
  nat  every kept row, except that a label over NAT_CAP of the set is cut to NAT_CAP, eligibility rows cut first.
The reasoning_trace column of train_fit.csv is not carried: it is the rules' trace, and train_main's raw upload
never sends it.

Never reads data/built/fresh_test2/ (fresh test 2's postings come from the frozen test pool by the manifest's ids).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

FIT = ROOT / "data/built/fit"
TRAIN_CSV = FIT / "train_fit.csv"
RELABEL = FIT / "relabel"
LABELS_B = RELABEL / "labels/B"
OUT = FIT / "fit2"
BAL_CSV = FIT / "train_fit2_bal.csv"
NAT_CSV = FIT / "train_fit2_nat.csv"
SEED = 20261022
NAT_CAP = 0.60
WORDS = ("Keep", "Explore", "Skip")
ELIG_CATS = frozenset({"work_auth", "enrollment", "degree_level", "degree_field", "employment_type", "location",
                       "license", "work_mode"})
DROP_CONF = {"low"}
# Labeler B chunks whose labels are scrambled: row_id keys right, label, reason and confidence belong to other rows
# (scramble_check). Their B labels are not used; their rows that the fixed rules Skip on eligibility keep the rules
# label, every other row is dropped (kind careful_scrambled).
SCRAMBLED_CHUNKS = frozenset({"chunk_037", "chunk_040", "chunk_049"})
SCRAMBLE_MIN = 0.45   # a chunk passes when at least this share of B's reasons overlap their own row best in the chunk
# A B Explore whose reason says a required item is not shown on the resume: the rubric makes a clearly required item
# the resume does not show a Skip, B often says Explore (fresh test 1: 15 of 33 such B Explores are gold Skip, against
# 16 of 111 other B Explores), so these rows are dropped (kind careful_soft_explore) unless the posting is thin.
NOT_SHOWN = re.compile(
    r"(not shown|not on (the|her|his|their) resume|isn't shown|aren't shown|no evidence|not listed|not evident|"
    r"not mentioned|doesn't show|does not show|not demonstrated|not stated on the resume|resume (shows|lists) no|"
    r"lacks|lacking|missing|without (any )?(evidence|experience)|no (direct |clear |explicit )?(experience|mention)|"
    r"(does not|doesn't|never) (list|name|mention|include)s?|not (named|included)|resume shows only|not on (the )?resume)",
    re.I)
DEBIAS_TO = 0.37      # profile-only grouped-CV balanced accuracy each variant is filtered down to (train_fit.csv: 0.356)
DEBIAS_STEP = 0.02    # share of a label's rows dropped a round
DEBIAS_ROUNDS = 30
EXPLORE_FLOOR = 150   # the scarcest label is never filtered below this (bal shrinks with it: a floor that keeps
                      # profile-guessable Explores in bal holds the profile guesser at 0.47)
FT_MANIFESTS = (ROOT / "eval/fresh_test_manifest.json", ROOT / "eval/fresh_test2_manifest.json")

csv.field_size_limit(10**9)


def jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open() if line.strip()] if p.exists() else []


def write_jsonl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def careful_completion(word: str, reason: str) -> str:
    text = " ".join(reason.split()).rstrip(".").strip()
    if word not in WORDS or not text:
        raise ValueError(f"bad label {word!r} / reason {reason!r}")
    return f"{word} — {text}."


def _toks(s: str) -> set[str]:
    stop = {"the", "and", "for", "with", "that", "this", "from", "are", "but", "not", "has", "have", "candidate",
            "candidate's", "posting", "role", "job", "required", "requires", "require", "does", "also", "only",
            "their", "they", "them", "which", "into", "than", "more", "over", "about", "there", "would", "while",
            "within", "being", "both", "such"}
    return {w for w in re.findall(r"[a-z0-9][a-z0-9+#\-]{2,}", s.lower()) if w not in stop}


def scramble_check() -> dict[str, float]:
    """Each B chunk's share of reasons whose word overlap is highest with their own row among the chunk's rows
    (posting, resume and profile as the labeler saw them). A sound chunk sits at 0.77-0.96; a chunk whose labels
    were written against the wrong rows falls to 0.08-0.27. Stops the build when a chunk under SCRAMBLE_MIN is
    not in SCRAMBLED_CHUNKS, or one in it passes."""
    out = {}
    for f in sorted((RELABEL / "chunks").glob("chunk_*.jsonl")):
        rows = {x["row_id"]: _toks(f"{x['posting']} {x['resume']} {x['profile']}") for x in jsonl(f)}
        labs = jsonl(LABELS_B / f.name)
        if {x["row_id"] for x in labs} != set(rows):
            sys.exit(f"{f.name}: B's row ids differ from the chunk's")
        own = 0
        for x in labs:
            t = _toks(x["reason"])
            ov = {i: len(t & w) / max(1, len(t)) for i, w in rows.items()}
            own += max(ov, key=lambda i: (ov[i], i == x["row_id"])) == x["row_id"]
        out[f.stem] = round(own / len(labs), 3)
    bad = {c for c, v in out.items() if v < SCRAMBLE_MIN}
    if bad != SCRAMBLED_CHUNKS:
        sys.exit(f"scramble check: chunks under {SCRAMBLE_MIN}: {sorted(bad)}; SCRAMBLED_CHUNKS: {sorted(SCRAMBLED_CHUNKS)}")
    return out


def joined_rows() -> list[dict]:
    """train_fit.csv rows in file order with their pool rows (relabel_prep.joined_rows, keeping posting key and
    employer for the checks)."""
    import build_fit_train as bft
    mix, tw, vr = bft.mix3_rows(), bft.twin_rows(), bft.venator_rows()
    pool = [*mix, *[r for r in tw if r["split"] == "train"], *[r for r in vr if r["split"] == "train"]]
    idx: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(pool):
        idx[r["prompt"]].append(i)
    out = []
    for n, t in enumerate(csv.DictReader(TRAIN_CSV.open(newline=""))):
        hits = idx.get(t["prompt"], [])
        if len(hits) != 1:
            sys.exit(f"train_fit.csv row {n}: {len(hits)} pool rows match (want 1)")
        p = pool[hits[0]]
        out.append({"csv_row": n, "prompt": t["prompt"], "completion": t["completion"], "source": p["source"],
                    "sub": p["sub"], "cand": p["cand"], "card": p["card"], "posting_key": p["posting_key"],
                    "employer": p.get("employer", "")})
    return out


def take_preferring(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """n rows: careful rows first, then eligibility rows, each group in seeded random order."""
    careful = sorted((r for r in rows if r["origin"] == "careful"), key=lambda r: r["csv_row"])
    elig = sorted((r for r in rows if r["origin"] != "careful"), key=lambda r: r["csv_row"])
    rng.shuffle(careful)
    rng.shuffle(elig)
    return (careful + elig)[:n]


def mix(rows: list[dict]) -> dict:
    c = Counter(r["label"] for r in rows)
    n = len(rows)
    return {"rows": n, "labels": {w: c[w] for w in WORDS}, "frac": {w: round(c[w] / n, 4) for w in WORDS},
            "by_origin": dict(Counter(r["origin"] for r in rows)),
            "by_source_label": {s: dict(Counter(r["label"] for r in rows if r["source"] == s))
                                for s in ("mix3", "twins", "venator", "teaser")}}


def build(_: argparse.Namespace) -> None:
    import hybrid_rules as hr
    from matcher.rules import format_completion, label
    from relabel_prep import row_id
    scramble = scramble_check()
    index = {x["row_id"]: x for x in jsonl(RELABEL / "index.jsonl")}
    lab = {}
    for f in sorted(LABELS_B.glob("chunk_*.jsonl")):
        for x in jsonl(f):
            if x["row_id"] in lab:
                sys.exit(f"{x['row_id']} labeled twice")
            lab[x["row_id"]] = x
    selected = {k for k, x in index.items() if x["selected"]}
    if set(lab) != selected:
        sys.exit(f"labels cover {len(set(lab) & selected)} of {len(selected)} selected rows, {len(set(lab) - selected)} extra")
    rows = joined_rows()
    out, kinds, changed = [], Counter(), []
    for r in rows:
        rid = row_id(r["prompt"])
        x = index[rid]
        if x["csv_row"] != r["csv_row"]:
            sys.exit(f"{rid}: index csv_row {x['csv_row']} != {r['csv_row']}")
        v = hr.rule_verdict(r["cand"], r["card"])
        elig = v["rule"] == "Skip" and bool(set(v["hard_cats"]) & ELIG_CATS)
        scrambled = x["chunk"] in SCRAMBLED_CHUNKS
        b = None if scrambled else lab.get(rid)
        thin = bool(r["card"].thin_posting) or r["source"] == "teaser"
        if x["eligibility_decided"] and not elig:
            kind = "elig_lost"  # an old eligibility Skip the fixed rules no longer Skip on eligibility: never relabeled
        elif elig:
            kind = "elig" if x["eligibility_decided"] else ("elig_mode_relabeled" if b or scrambled else "elig_mode_new")
        elif scrambled:
            kind = "careful_scrambled"
        elif b is None:
            kind = "fit_unlabeled"
        elif b["confidence"] in DROP_CONF:
            kind = "careful_low"
        elif thin and b["label"] != "Explore":
            kind = "careful_thin_off_rubric"  # a title-and-place posting is Explore by the rubric
        elif b["label"] == "Explore" and not thin and NOT_SHOWN.search(b["reason"]):
            kind = "careful_soft_explore"
        else:
            kind = "careful"
        kinds[(x["source"], kind)] += 1
        rec = {"row_id": rid, "csv_row": r["csv_row"], "source": r["source"], "sub": r["sub"],
               "posting_key": r["posting_key"], "employer": r["employer"], "kind": kind,
               "rules_label_old": x["rules_label"], "rules_completion_old": x["rules_completion"],
               "hard_cats": v["hard_cats"], "thin": bool(r["card"].thin_posting), "chunk": x["chunk"],
               "b_label": b and b["label"], "b_confidence": b and b["confidence"]}
        if kind.startswith("elig") and kind != "elig_lost":
            bucket, reasons, _ = label(r["cand"], r["card"])
            by_check = hr.reasons_by_check(r["cand"], r["card"])
            # only the eligibility reasons: a skills or years reason beside them is the extractor's must-skill text
            reasons = [y for y in reasons if hr.category(by_check.get(y), y) in ELIG_CATS]
            if bucket != "Excluded" or not reasons:
                sys.exit(f"{rid}: eligibility row without an eligibility reason")
            comp = format_completion(bucket, reasons)
            if comp != x["rules_completion"]:
                changed.append({"row_id": rid, "old": x["rules_completion"], "new": comp})
            rec |= {"origin": "eligibility", "label": comp.split()[0], "completion": comp}
        elif kind == "careful":
            rec |= {"origin": "careful", "label": b["label"], "completion": careful_completion(b["label"], b["reason"])}
        else:
            rec |= {"origin": None, "label": None, "completion": None}
        rec["prompt"] = r["prompt"]
        out.append(rec)
    write_jsonl(OUT / "pool.jsonl", out)
    rep = {
        "cards": "v2 cards throughout (extractor v3 not adopted; no v3 card exists for a training posting)",
        "rules": "fixed rules (work mode a hard Skip); eligibility = rules Skip with a hard reason in " + ", ".join(sorted(ELIG_CATS)),
        "train_fit_rows": len(out),
        "scramble_check": {"min": SCRAMBLE_MIN, "dropped_chunks": sorted(SCRAMBLED_CHUNKS), "own_row_best_share": scramble,
                           "rows_in_dropped_chunks": dict(Counter(r["kind"] for r in out if r["chunk"] in SCRAMBLED_CHUNKS))},
        "kinds_by_source": {s: {k: n for (s2, k), n in sorted(kinds.items()) if s2 == s} for s in ("mix3", "twins", "venator", "teaser")},
        "kinds": dict(Counter(r["kind"] for r in out)),
        "eligibility_completions_changed": {"rows": len(changed), "examples": changed[:5]},
        "elig_mode_relabeled_b_labels": dict(Counter(r["b_label"] for r in out if r["kind"] == "elig_mode_relabeled")),
        "elig_mode_relabeled_b_labels_by_source": {s: dict(Counter(r["b_label"] for r in out if r["kind"] == "elig_mode_relabeled" and r["source"] == s))
                                                   for s in ("mix3", "twins", "venator", "teaser")},
        "agreement_old_rules_vs_b": agreement(out),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pool_report.json").write_text(json.dumps(rep, indent=1))
    variants(argparse.Namespace())


def profile_oof(rows: list[dict]) -> tuple[float, list[float]]:
    """Grouped 5-fold (by posting) out-of-fold balanced accuracy of a profile-only TF-IDF + logistic guesser
    (build_fit_train._profile_oof's guesser), and each row's out-of-fold probability of its own label."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from matcher.pairs import prompt_parts
    x = [prompt_parts(r["prompt"])[1] for r in rows]
    y = np.array([r["label"] for r in rows])
    g = [r["posting_key"] for r in rows]
    p_gold = np.zeros(len(rows))
    pred = np.empty(len(rows), dtype=object)
    for tr, te in GroupKFold(n_splits=5).split(x, y, g):
        v = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True, max_features=50000)
        xt = v.fit_transform([x[i] for i in tr])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(xt, y[tr])
        pr = clf.predict_proba(v.transform([x[i] for i in te]))
        cls = list(clf.classes_)
        p_gold[te] = pr[np.arange(len(te)), [cls.index(c) for c in y[te]]]
        pred[te] = np.array(cls)[pr.argmax(1)]
    return float(balanced_accuracy_score(y, pred)), p_gold.tolist()


def make_variants(kept: list[dict], seed: int) -> tuple[list[dict], list[dict], str | None]:
    rng = random.Random(seed)
    by = {w: [r for r in kept if r["label"] == w] for w in WORDS}
    per = min(len(v) for v in by.values())
    bal = [r for w in WORDS for r in take_preferring(by[w], per, rng)]
    nat, capped = list(kept), None
    top = max(WORDS, key=lambda w: len(by[w]))
    if len(by[top]) > NAT_CAP * len(kept):
        n = int(NAT_CAP / (1 - NAT_CAP) * (len(kept) - len(by[top])))
        nat, capped = [r for w in WORDS if w != top for r in by[w]] + take_preferring(by[top], n, rng), top
    rng.shuffle(bal)
    rng.shuffle(nat)
    return bal, nat, capped


def debias(kept: list[dict], which: int) -> tuple[list[dict], dict]:
    """AFLite on the profile, one variant at a time (which: 0 bal, 1 nat). The careful labels Skip a set preference
    the posting meets (an avoided role or industry, pay under the floor) where the rules said Explore, and the
    generators drew preferences to be met, so a profile alone guesses the label ('avoid technology' -> Skip, 'avoid
    none' -> Explore). Each round cuts the variant from the rows not yet dropped and scores it with a grouped 5-fold
    profile-only guesser (balanced accuracy); while that is over DEBIAS_TO it drops, per label, the variant's rows the
    guesser finds easiest (own-label probability over 0.5), DEBIAS_STEP of the label's rows a round, any source and
    origin, and a dropped row's place goes back to the label's other rows. Explore never goes below EXPLORE_FLOOR.
    Small steps and stopping at the first round at or under the target keep the filter from overshooting into an
    inverted shortcut (below chance)."""
    banned: set[str] = set()
    log = []
    for rnd in range(DEBIAS_ROUNDS + 1):
        rows = make_variants([r for r in kept if r["row_id"] not in banned], SEED)[which]
        acc, pg = profile_oof(rows)
        n = Counter(r["label"] for r in rows)
        log.append({"round": rnd, "rows": len(rows), "labels": {w: n[w] for w in WORDS}, "profile_bacc": round(acc, 4)})
        print(f"debias {('bal', 'nat')[which]} round {rnd}: {len(rows)} rows {dict(n)}, profile-only {acc:.4f}")
        if acc <= DEBIAS_TO or rnd == DEBIAS_ROUNDS:
            break
        left = Counter(r["label"] for r in kept if r["row_id"] not in banned)
        for w in WORDS:
            step = max(5, int(n[w] * DEBIAS_STEP))
            if w == "Explore":
                step = min(step, left[w] - EXPLORE_FLOOR)
            easy = sorted((-p, r["row_id"]) for r, p in zip(rows, pg) if r["label"] == w and p > 0.5)
            banned |= {i for _, i in easy[:max(0, step)]}
    dropped = [r for r in kept if r["row_id"] in banned]
    return rows, {
        "target": DEBIAS_TO, "step": DEBIAS_STEP, "explore_floor": EXPLORE_FLOOR, "rounds": log, "dropped": len(dropped),
        "reached_target": log[-1]["profile_bacc"] <= DEBIAS_TO,
        "dropped_by_source_label": dict(sorted(Counter(f"{r['source']}:{r['label']}" for r in dropped).items())),
        "dropped_by_origin_label": dict(sorted(Counter(f"{r['origin']}:{r['label']}" for r in dropped).items()))}


def variants(_: argparse.Namespace) -> None:
    out = jsonl(OUT / "pool.jsonl")
    kept = [r for r in out if r["origin"]]
    before = {"kept_pool": mix(kept)}
    bal, deb_bal = debias(kept, 0)
    nat, deb_nat = debias(kept, 1)
    capped = make_variants(kept, SEED)[2]
    for name, rs in (("bal", bal), ("nat", nat)):
        ids = {r["row_id"] for r in rs}
        for r in out:
            r[f"in_{name}"] = r["row_id"] in ids
    for path, rs in ((BAL_CSV, bal), (NAT_CSV, nat)):
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["prompt", "completion"])
            for r in rs:
                if r["completion"].split()[0] != r["label"]:
                    sys.exit(f"{r['row_id']}: completion {r['completion']!r} vs label {r['label']}")
                w.writerow([r["prompt"], r["completion"]])
    write_jsonl(OUT / "rows.jsonl", [{k: v for k, v in r.items() if k != "prompt"} for r in out])
    rep = json.loads((OUT / "pool_report.json").read_text()) | before | {
        "debias": {"bal": deb_bal, "nat": deb_nat},
        "bal": mix(bal) | {"csv": str(BAL_CSV.relative_to(ROOT))},
        "nat": mix(nat) | {"csv": str(NAT_CSV.relative_to(ROOT)), "cap": NAT_CAP, "capped_label": capped}}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("debias", "bal", "nat")}, indent=1))


def agreement(out: list[dict]) -> dict:
    """train_fit.csv's rules label against B's on every relabeled row, by source: all confidences, and high or
    medium only; with the old -> new confusion."""
    res = {}
    for scope, keep in (("all", lambda r: True), ("high_medium", lambda r: r["b_confidence"] not in DROP_CONF)):
        res[scope] = {}
        for s in ("mix3", "twins", "venator", "teaser", "all"):
            rs = [r for r in out if r["b_label"] and keep(r) and (s == "all" or r["source"] == s)]
            agree = sum(r["b_label"] == r["rules_label_old"] for r in rs)
            res[scope][s] = {"rows": len(rs), "agree": agree, "rate": round(agree / max(1, len(rs)), 4),
                             "old_to_new": dict(sorted(Counter(f"{r['rules_label_old']}->{r['b_label']}" for r in rs).items()))}
    # real mix3 rows (real postings on extracted cards) apart from synthetic ones
    for sub in ("real", "synthetic"):
        rs = [r for r in out if r["b_label"] and r["source"] == "mix3" and r["sub"] == sub]
        agree = sum(r["b_label"] == r["rules_label_old"] for r in rs)
        res["all"][f"mix3_{sub}"] = {"rows": len(rs), "agree": agree, "rate": round(agree / max(1, len(rs)), 4)}
    return res


# ---------------------------------------------------------------- checks


def _company(posting: str) -> str:
    import re
    m = re.search(r"^Company: (.+)$", posting, re.M)
    return m.group(1).strip() if m else ""


def variant_frame(csv_path: Path, meta: dict[str, dict]):
    import pandas as pd
    from matcher.pairs import prompt_parts
    from relabel_prep import row_id
    out = []
    for t in csv.DictReader(csv_path.open(newline="")):
        m = meta[row_id(t["prompt"])]
        resume, profile, posting = prompt_parts(t["prompt"])
        out.append({"id": m["row_id"], "resume": resume, "profile": profile, "posting": posting,
                    "label": t["completion"].split()[0], "posting_key": m["posting_key"], "employer": m["employer"],
                    "origin": m["origin"], "source": m["source"], "thin": m["thin"] or m["source"] == "teaser"})
    return pd.DataFrame(out)


def guessers(df) -> dict:
    """Grouped 5-fold by posting (a posting's rows never on both sides): profile, posting, resume and length
    guessers, scored on all rows and on the careful rows alone; plus matcher.leakage's baselines (full BoW and the
    shuffled-posting control) with fold 0 held out."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from matcher import leakage as L

    def guess(cols, tr, te, length=False):
        if length:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(L.length_features(tr), tr.label)
            return clf.predict(L.length_features(te))
        return L._fit_predict_text(tr, te, cols)

    def sc(y, p):
        return {"accuracy": round(accuracy_score(y, p), 4), "macro_f1": round(f1_score(y, p, average="macro"), 4),
                "balanced_accuracy": round(balanced_accuracy_score(y, p), 4)}

    gkf = GroupKFold(n_splits=5)
    careful = (df.origin == "careful").to_numpy()
    out = {"rows": len(df), "majority_rate": round(df.label.value_counts(normalize=True).max(), 4),
           "careful_rows": int(careful.sum()),
           "careful_majority_rate": round(df.label[careful].value_counts(normalize=True).max(), 4), "grouped_cv": {}}
    for name, cols, length in (("profile only", ["profile"], False), ("posting only", ["posting"], False),
                               ("resume only", ["resume"], False), ("resume + profile + posting", ["resume", "profile", "posting"], False),
                               ("length", [], True)):
        pred = np.empty(len(df), dtype=object)
        for tr_i, te_i in gkf.split(df, df.label, df.posting_key):
            pred[te_i] = guess(cols, df.iloc[tr_i], df.iloc[te_i], length)
        out["grouped_cv"][name] = sc(df.label, pred) | {"careful_rows": sc(df.label[careful], pred[careful])}
    # thin postings say Explore by the rubric, readable from the posting alone: the posting guesser without them
    full = df[~df.thin].reset_index(drop=True)
    pred = np.empty(len(full), dtype=object)
    for tr_i, te_i in gkf.split(full, full.label, full.posting_key):
        pred[te_i] = guess(["posting"], full.iloc[tr_i], full.iloc[te_i])
    out["posting_only_without_thin"] = sc(full.label, pred) | {"rows": len(full)}
    tr_i, te_i = next(gkf.split(df, df.label, df.posting_key))
    out["fold0_baselines_macro_f1"] = {r["baseline"]: round(r["macro_f1"], 4)
                                       for r in L.run_baselines(df.iloc[tr_i].reset_index(drop=True),
                                                                df.iloc[te_i].reset_index(drop=True))}
    return out


def fresh_overlap(frames: dict) -> dict:
    """Fresh test 1 and 2 postings and employers against each variant: posting keys against both manifests'
    posting_ids; company keys (the venator rows' employer field and every prompt's Company line) against both
    manifests' employers and the freeze's test-pool employer names; and each distinct training posting's 6-word
    windows against the fresh-test posting texts (containment >= 0.5 is a near copy). Fresh test 2's texts come from
    the frozen test pool by the manifest's ids, never from data/built/fresh_test2/."""
    from fit_pools import load_pool
    from matcher.nearcopy import company_key, shingles
    man = [json.loads(p.read_text()) for p in FT_MANIFESTS]
    ids = {i for m in man for i in m["posting_ids"]}
    emps = {company_key(e): f"fresh_test{n + 1}" for n, m in enumerate(man) for e in m["employers"]}
    fr = json.loads((ROOT / "eval/fit_freeze.json").read_text())
    test_emps = {company_key(n) for e in fr["employers"]["test"] for n in [e["employer"], *e["names"]]}
    ft1 = {p["key"]: f"{p.get('title', '')}\n{p.get('text', '')}" for p in jsonl(ROOT / "data/built/fresh_test/postings.jsonl")}
    pool = {p["key"]: p for p in load_pool("test")}
    ft2_ids = json.loads(FT_MANIFESTS[1].read_text())["posting_ids"]
    ft_texts = ft1 | {i: f"{pool[i].get('title', '')}\n{pool[i].get('text', '')}" for i in ft2_ids}
    inv: dict[str, set[str]] = defaultdict(set)
    for k, t in ft_texts.items():
        for s in shingles(t):
            inv[s].add(k)
    res = {"fresh_posting_ids": len(ids), "fresh_employers": len(emps), "fresh_texts": len(ft_texts)}
    for name, df in frames.items():
        keys = set(df.posting_key)
        cos = {company_key(x) for x in [*df.employer, *map(_company, df.posting)] if x}
        near = []
        for pk, text in dict(zip(df.posting_key, df.posting)).items():
            sa = shingles(text)
            if len(sa) < 5:
                continue
            hits = Counter(k for s in sa for k in inv.get(s, ()))
            for k, n in hits.most_common(1):
                if n / len(sa) >= 0.5:
                    near.append({"train_posting": pk, "fresh_posting": k, "containment": round(n / len(sa), 3)})
        res[name] = {"posting_keys": len(keys), "posting_keys_in_fresh": sorted(keys & ids),
                     "company_keys": len(cos), "fresh_employers_in_train": sorted(k for k in cos if k in emps),
                     "test_pool_employers_in_train": sorted(cos & test_emps), "near_copies": near,
                     "pass": not (keys & ids) and not any(k in emps for k in cos) and not near}
    return res


def check(_: argparse.Namespace) -> None:
    meta = {r["row_id"]: r for r in jsonl(OUT / "rows.jsonl")}
    frames = {"bal": variant_frame(BAL_CSV, meta), "nat": variant_frame(NAT_CSV, meta)}
    out = {name: guessers(df) for name, df in frames.items()}
    out["fresh_overlap"] = fresh_overlap(frames)
    (OUT / "checks.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(fn=build)
    sub.add_parser("variants", help="debias and variants from fit2/pool.jsonl (build's first stage)").set_defaults(fn=variants)
    sub.add_parser("check").set_defaults(fn=check)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
