"""Real-posting training pairs: build_real_test's rows on the real postings outside the real test.

The real test (build_real_test.py, 86 postings with rows) reads real posting text; training so far never did. This
builds pairs the same way on every other real posting with a v2 extracted card, with the extracted card in place of
the hand-checked one, so its labels carry the extraction's errors where the card is wrong on what a row tests.

  build_real_train.py build [--seed 0] [--dev-share 0.10] [--per-posting 0] [--dry-run]
      Writes data/built/real_train/
        rows.jsonl          one row per (posting, candidate), with its split; labels, no prompt yet
        dropped.jsonl       every row left out, with the reason (build_real_test's drops, plus resume_shared_*)
        excluded.jsonl      every posting kept out as a near copy of a test posting: rule, test id, scores
        borderline.jsonl    kept postings just under a near-copy rule, with the closest test posting
        splits.json         posting id -> train | dev, for every posting with a split
        resume_cards.jsonl  the candidate cards of the kept rows, for render_cards.py resumes
        resume_ids.txt      their ids
        report.json         counts: pool, exclusion, split, buckets and kinds per split, resumes and credits
      --dry-run prints the report and writes nothing. --per-posting K keeps at most K rows a posting, round-robin over
      variant kinds (0, the default, keeps every row). Nothing here calls Adaption: the resume count and the credit
      estimate are computed from the rendered store and the earlier render's state files.
  build_real_train.py assemble
      Joins rendered resume texts (render_cards' context hash) with render_profile and the real posting text through
      build_prompt, as build_real_test.assemble_rows does for the test (the same call, the same posting fit), and
      writes data/built/real_train/train.jsonl and dev.jsonl. A row whose label the rendered resume does not show
      (assemble_rows' decidable False) is left out, as is a row with no rendered resume.

Pool: data/built/cards/extracted_real_cards_v2.jsonl, minus the 87 test cards (data/built/audit/test_cards.jsonl), minus
rows blueprint.training_ok rejects (meta.in_real_test: a test description or a nearcopy.NearIndex hit, same company and
title or 80% of the 6-word windows shared; dup_of and same_description_as: one row per posting text; must_phrases and
must_many: cards whose required groups read as phrases or are too many to trust).

Near copy of a test posting (near_test), on top of that, any one of:
  1. same company (nearcopy.company_key) and a similar title: the title's core words (title_core: no intern, co-op,
     season, year, "university" or other filler, a light stem) at TITLE_SIM Jaccard or more;
  2. most of the text shared: the descriptions' 6-word windows at TEXT_JACCARD Jaccard or more, any company;
  3. one employer under another name: another company string, and CONTAINMENT or more of the shorter description's
     6-word windows in the other (PennState / Pennsylvania State University, CoverMyMeds / McKesson, xAI / SpaceX).
  Within one company the employer's boilerplate (about us, benefits, EEO, the sponsorship line) is shared by every
  role, so containment alone is not used there: HPE's Optical Engineering Intern holds 80% of the HPE ASIC test
  posting's windows and is another job. Rules 1-3 remove 122 pool postings (near 40 test postings); a whole-company exclusion would remove
  331 (report.json, exclusion.whole_company_alternative).

Split: postings are grouped into components (same company_key, or CONTAINMENT of the shorter description shared,
which joins "Nissan" and "Nissan Global"), and a component goes to train or dev whole, so dev shares no employer with
train. The 50 fresh postings with checked cards (data/built/audit/fresh_checked_cards.jsonl) go to dev with their
components: their rows are also labelled with the checked card (report.json, checked_card_agreement), a first measure
of how often the extracted card's label is wrong. That file holds the cards as read, not corrected; `build
--checked hand_truth` measures against score_extraction.hand_truth instead, and `agreement` reruns the measure on the
built rows against any of CHECKED_SOURCES without rebuilding. Other components follow in a fixed hash order until
dev holds about --dev-share of the postings.

Rows: build_real_test.build_rows on each posting (candidates_for, its guards and its posting-side visibility checks,
imported, not copied), then build_real_test.person_problems. A row whose resume context is one a test row also uses is
dropped (resume_shared_with_test: the same resume text in train and test); a dev row whose resume context a train row
uses is dropped too (resume_shared_with_train).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import build_real_test as brt  # noqa: E402
from matcher.blueprint import training_ok  # noqa: E402
from matcher.nearcopy import company_key, norm_text, shingles  # noqa: E402
from matcher.rules import label  # noqa: E402
from matcher.schema import CandidateCard, RequirementCard  # noqa: E402

V2 = ROOT / "data/built/cards/extracted_real_cards_v2.jsonl"
TEST_CARDS = brt.AUDIT
CHECKED = ROOT / "data/built/audit/fresh_checked_cards.jsonl"
FRESH_VERDICTS = ROOT / "data/built/audit/fresh_verdicts.jsonl"
CLEAN_CARDS = ROOT / "data/built/real_train/clean_checked/cards.jsonl"
# Which checked cards checked_cards returns. "raw" is CHECKED as it was read, not corrected (the default, what build
# has always used); "hand_truth" is score_extraction.hand_truth, CHECKED with every "wrong" and "debatable" verdict
# of FRESH_VERDICTS applied (the truth for those 50 postings); "clean" is clean_real_checked's cards.jsonl (the four
# cardcheck batches plus hand_truth for the 28 fresh dev postings, nice list the extractor's: the cards the checked
# real dev labels come from).
CHECKED_SOURCES = ("raw", "hand_truth", "clean")
RAW_POSTINGS = brt.RAW_POSTINGS
OUT = ROOT / "data/built/real_train"
STATE = ROOT / "data/adaption"
SPLITS = ("train", "dev")

TITLE_SIM = 0.5
TEXT_JACCARD = 0.5
CONTAINMENT = 0.5
DEV_SHARE = 0.10
COMMON_DF = 300  # a 6-word window in more postings than this is generic text; it does not propose a pair
MIN_SHARED = 20  # a pair sharing fewer specific windows than this is not checked further
CREDITS_PER_TRY = 0.0105  # fallback: Adaption's estimate for the 10,206-row resume batch, 107 credits
TRIES_PER_CONTEXT = 1.8  # fallback: the real test's render (state files give the measured value)
CAPS = (5, 7, 10)  # per-posting caps the report prices

# ---------------------------------------------------------------- inputs


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_raw(path: Path = RAW_POSTINGS) -> dict[str, dict]:
    """posting id -> the raw ATS posting (title, company, locations, description)."""
    return {"ats_" + p["simplify_id"]: p for p in read_jsonl(path)}


def checked_card_dicts(source: str = "raw", checked: Path = CHECKED, verdicts: Path = FRESH_VERDICTS,
                       clean: Path = CLEAN_CARDS) -> dict[str, dict]:
    """posting id -> checked card dict, from the CHECKED_SOURCES source named."""
    if source == "raw":
        return {r["id"]: r["card"] for r in read_jsonl(checked)}
    if source == "hand_truth":
        import score_extraction as se
        return se.hand_truth(read_jsonl(verdicts), {r["id"]: r["card"] for r in read_jsonl(checked)})
    if source == "clean":
        return {r["id"]: r["card"] for r in read_jsonl(clean)}
    raise ValueError(f"checked card source {source!r}, not one of {CHECKED_SOURCES}")


def checked_cards(source: str = "raw", **paths) -> dict[str, RequirementCard]:
    return {i: RequirementCard.from_dict(c) for i, c in checked_card_dicts(source, **paths).items()}


def test_ids(path: Path = TEST_CARDS) -> list[str]:
    return sorted(r["id"] for r in read_jsonl(path))


def drop_reason(meta: dict | None) -> str | None:
    """The first reason training_ok rejects a v2 row, in a fixed order, or None when it accepts it."""
    if training_ok(meta):
        return None
    m = meta or {}
    for key, why in (("in_real_test", "in_real_test (test description or nearcopy.NearIndex hit)"),
                     ("dup_of", "dup_of (identical posting under another id)"),
                     ("same_description_as", "same_description_as (one description under several titles)"),
                     ("must_phrases", "must_phrases (a required group reads as a phrase)"),
                     ("must_many", "must_many (too many required groups)")):
        if m.get(key):
            return why
    return "not US or not English"


def why_out(i: str, tests: list[str], meta: dict[str, dict | None], excluded: list[dict]) -> str:
    """Why a posting is in no split: no v2 card, a test posting, a training_ok reason, or a near-copy rule."""
    if i in tests:
        return "test posting"
    if i not in meta:
        return "no v2 card"
    hit = next((e for e in excluded if e["id"] == i), None)
    return drop_reason(meta[i]) or (f"near copy: {hit['rule']}" if hit else "in a split")


def load_pool(v2: Path = V2, tests: list[str] | None = None) -> tuple[dict[str, RequirementCard], dict]:
    """(posting id -> extracted card for every v2 row that may train, counts of what was left out)."""
    tests = set(test_ids() if tests is None else tests)
    rows = read_jsonl(v2)
    cards: dict[str, RequirementCard] = {}
    why = Counter()
    for r in rows:
        if r["id"] in tests:
            why["test posting"] += 1
            continue
        reason = drop_reason(r.get("meta"))
        if reason:
            why[reason] += 1
            continue
        cards[r["id"]] = RequirementCard.from_dict(r["card"])
    return cards, {"v2_rows": len(rows), "left_out": dict(why), "pool": len(cards)}


# ---------------------------------------------------------------- near copies of the test

FILLER = frozenset("""intern interns internship internships co op coop summer fall spring winter autumn university
student students program programme new grad graduate early career undergraduate part full time the of and for in to
a an i ii iii iv usa us remote hybrid onsite games season term quarter""".split())


def stem(w: str) -> str:
    """Strip -ing, -ers, -er, -s until none is left ("engineering", "engineers", "engineer" -> "engine")."""
    while True:
        suf = next((s for s in ("ing", "ers", "er", "s") if len(w) > len(s) + 3 and w.endswith(s)), None)
        if suf is None:
            return w
        w = w[: -len(suf)]


def title_core(title: str) -> frozenset[str]:
    """The words of a title that name the job: no intern, co-op, season, year or other filler, lightly stemmed
    ("Software Engineering Intern - Summer 2027" -> {softwar..., engine...})."""
    return frozenset(stem(w) for w in norm_text(title).split() if w not in FILLER and not w.isdigit())


def title_sim(a: str, b: str) -> float:
    """Jaccard of the title cores; two titles with no core word at all ("Summer Intern 2027", "Intern") are alike."""
    x, y = title_core(a), title_core(b)
    if not x and not y:
        return 1.0
    return len(x & y) / len(x | y) if x and y else 0.0


def overlap(a: frozenset[str], b: frozenset[str]) -> tuple[float, float]:
    """(Jaccard, containment: the share of the smaller set found in the other) of two shingle sets."""
    if not a or not b:
        return 0.0, 0.0
    n = len(a & b)
    return n / len(a | b), n / min(len(a), len(b))


RULES = ("same company, similar title", "most of the text shared", "same employer under another name")


class TestIndex:
    """The test postings, ready for `near` and `closest`."""

    def __init__(self, tests: dict[str, dict]):  # id -> raw posting
        self.ids = sorted(tests)
        self.post = tests
        self.co = {i: company_key(tests[i].get("company", "")) for i in self.ids}
        self.sh = {i: shingles(tests[i].get("description", "")) for i in self.ids}

    def scores(self, post: dict) -> list[dict]:
        """One score row per test posting: same company, title similarity, Jaccard, containment."""
        co, sh = company_key(post.get("company", "")), shingles(post.get("description", ""))
        out = []
        for i in self.ids:
            j, c = overlap(sh, self.sh[i])
            same = bool(co) and co == self.co[i]
            ts = title_sim(post.get("title", ""), self.post[i].get("title", "")) if same else 0.0
            out.append({"test_id": i, "same_company": same, "title_sim": round(ts, 3), "jaccard": round(j, 3),
                        "containment": round(c, 3)})
        return out

    @staticmethod
    def rule_of(s: dict) -> str | None:
        if s["same_company"] and s["title_sim"] >= TITLE_SIM:
            return RULES[0]
        if s["jaccard"] >= TEXT_JACCARD:
            return RULES[1]
        if not s["same_company"] and s["containment"] >= CONTAINMENT:
            return RULES[2]
        return None

    def near(self, post: dict) -> dict | None:
        """The strongest near-copy hit {test_id, rule, scores} (rule order, then Jaccard), or None."""
        hits = [(RULES.index(r), -s["jaccard"], s["test_id"], r, s) for s in self.scores(post) if (r := self.rule_of(s))]
        if not hits:
            return None
        *_, rule, s = min(hits)
        return s | {"rule": rule}

    def closest(self, post: dict) -> dict:
        """For the borderline list: the highest Jaccard, the highest cross-company containment and the most similar
        same-company title, each with its test id."""
        sc = self.scores(post)
        j = max(sc, key=lambda s: (s["jaccard"], s["test_id"]))
        c = max((s for s in sc if not s["same_company"]), key=lambda s: (s["containment"], s["test_id"]), default=None)
        t = max((s for s in sc if s["same_company"]), key=lambda s: (s["title_sim"], s["test_id"]), default=None)
        return {"jaccard": j, "cross_company_containment": c, "same_company_title": t}


def exclude_near(pool: list[str], raw: dict[str, dict], tests: list[str]) -> tuple[list[str], list[dict], list[dict], int]:
    """(pool ids kept, excluded rows, borderline rows, pool postings that share a company with a test posting).

    Borderline: kept postings just under a rule: Jaccard 0.4-0.5, cross-company containment 0.3-0.5, or the same
    company with a title similarity of 0.3-0.5."""
    index = TestIndex({i: raw[i] for i in tests if i in raw})
    kept, excluded, border = [], [], []
    test_cos = set(index.co.values()) - {""}
    same_co = 0
    for i in sorted(pool):
        p = raw[i]
        same_co += company_key(p.get("company", "")) in test_cos
        hit = index.near(p)
        if hit:
            excluded.append({"id": i, "company": p.get("company"), "title": p.get("title"),
                             "test_company": raw[hit["test_id"]].get("company"), "test_title": raw[hit["test_id"]].get("title"),
                             **hit})
            continue
        kept.append(i)
        cl = index.closest(p)
        why = []
        if 0.4 <= cl["jaccard"]["jaccard"] < TEXT_JACCARD:
            why.append(("jaccard", cl["jaccard"]))
        if cl["cross_company_containment"] and 0.3 <= cl["cross_company_containment"]["containment"] < CONTAINMENT:
            why.append(("cross_company_containment", cl["cross_company_containment"]))
        if cl["same_company_title"] and 0.3 <= cl["same_company_title"]["title_sim"] < TITLE_SIM:
            why.append(("same_company_title", cl["same_company_title"]))
        for kind, s in why:
            border.append({"id": i, "company": p.get("company"), "title": p.get("title"), "near": kind,
                           "test_company": raw[s["test_id"]].get("company"), "test_title": raw[s["test_id"]].get("title"),
                           **s})
    return kept, excluded, border, same_co


# ---------------------------------------------------------------- split


def components(ids: list[str], raw: dict[str, dict]) -> list[list[str]]:
    """Groups of postings that must share a split: the same company_key, or CONTAINMENT of the shorter description's
    6-word windows shared (one employer under two names). Pairs are proposed by windows in at most COMMON_DF postings
    and checked on the full sets."""
    ids = sorted(ids)
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first: dict[str, str] = {}
    for i in ids:
        k = company_key(raw[i].get("company", ""))
        if k:
            union(i, first.setdefault(k, i))
    sh = {i: shingles(raw[i].get("description", "")) for i in ids}
    inv: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        for s in sh[i]:
            inv[s].append(i)
    shared: Counter = Counter()
    for s, group in inv.items():
        if 2 <= len(group) <= COMMON_DF:
            for x in range(len(group)):
                for y in range(x + 1, len(group)):
                    shared[(group[x], group[y])] += 1
    for (a, b), n in shared.items():
        if n >= MIN_SHARED and find(a) != find(b) and overlap(sh[a], sh[b])[1] >= CONTAINMENT:
            union(a, b)
    groups: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def comp_order(comp: list[str], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{comp[0]}".encode()).hexdigest()


def split_components(comps: list[list[str]], forced_dev: set[str], dev_share: float = DEV_SHARE,
                     seed: int = 0) -> dict[str, str]:
    """posting id -> train | dev. Components holding a forced id go to dev; then components in a fixed hash order join
    dev while it stays at or under dev_share of all postings; every other component is train."""
    total = sum(len(c) for c in comps)
    target = round(dev_share * total)
    out: dict[str, str] = {}
    dev = 0
    rest = []
    for c in comps:
        if forced_dev & set(c):
            out.update({i: "dev" for i in c})
            dev += len(c)
        else:
            rest.append(c)
    for c in sorted(rest, key=lambda c: comp_order(c, seed)):
        if dev + len(c) <= target:
            out.update({i: "dev" for i in c})
            dev += len(c)
        else:
            out.update({i: "train" for i in c})
    return out


# ---------------------------------------------------------------- rows


def build_posting_rows(cards: dict[str, RequirementCard], postings: dict[str, str],
                       seed: int = 0) -> tuple[list[dict], list[dict], dict[str, str]]:
    """(kept, dropped, posting id -> error) from build_real_test.build_rows, one posting at a time, so a card the
    builder cannot take loses its own rows only. A kept row whose candidate could not exist fails the build, as in
    build_real_test.build."""
    kept, dropped, errors = [], [], {}
    for i in sorted(cards):
        try:
            k, d = brt.build_rows({i: cards[i]}, {i: postings[i]}, seed)
        except Exception as e:  # noqa: BLE001 — recorded, the posting has no rows
            errors[i] = f"{type(e).__name__}: {e}"
            continue
        kept += k
        dropped += d
    bad = [(r["pair_id"], p) for r in kept if (p := brt.person_problems(CandidateCard.from_dict(r["candidate"])))]
    if bad:
        raise SystemExit(f"build check failed, nothing written: {len(bad)} kept candidates could not exist, e.g. {bad[:3]}")
    return kept, dropped, errors


def kind_order(kind: str, rq_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{rq_id}:{kind}".encode()).hexdigest()


def cap_rows(rows: list[dict], k: int, seed: int = 0) -> list[dict]:
    """At most k rows a posting, round-robin over its variant kinds (kinds in a fixed hash order per posting, rows of
    a kind by pair id), so no kind is cut first. k <= 0 keeps every row."""
    if k <= 0:
        return rows
    by: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["rq_id"]][r["variant_kind"]].append(r)
    keep: set[str] = set()
    for rq, kinds in by.items():
        queues = [sorted(v, key=lambda r: r["pair_id"]) for _, v in sorted(kinds.items(), key=lambda kv: kind_order(kv[0], rq, seed))]
        picked = 0
        while picked < k and any(queues):
            for q in queues:
                if q and picked < k:
                    keep.add(q.pop(0)["pair_id"])
                    picked += 1
    return [r for r in rows if r["pair_id"] in keep]


def ctx_of(card: dict) -> str:
    from render_cards import ctx_hash, ctx_str
    return ctx_hash(ctx_str("resumes", card))


def drop_shared_resumes(rows: list[dict], test_ctx: set[str]) -> tuple[list[dict], list[dict]]:
    """(kept, dropped): a row whose resume context a test row uses is dropped (the same resume text would sit in
    training and in the test), and a dev row whose resume context a train row uses (the same in train and dev)."""
    for r in rows:
        r.setdefault("ctx", ctx_of(r["candidate"]))
    train_ctx = {r["ctx"] for r in rows if r["split"] == "train" and r["ctx"] not in test_ctx}
    kept, dropped = [], []
    for r in rows:
        if r["ctx"] in test_ctx:
            dropped.append(r | {"drop": "resume_shared_with_test", "detail": "a test row's resume has this context"})
        elif r["split"] == "dev" and r["ctx"] in train_ctx:
            dropped.append(r | {"drop": "resume_shared_with_train", "detail": "a train row's resume has this context"})
        else:
            kept.append(r)
    return kept, dropped


# ---------------------------------------------------------------- resumes and credits


def measured_render_rate(test_cards: list[dict]) -> dict:
    """Tries a resume context took and credits a try cost on the real test's render: tries per context from the kept
    store and the quarantine file (render_cards.tries) over the test's resume contexts that had a try; credits per try
    from Adaption's estimates over the resume batches of 100 rows or more (smaller ones pay the 1-credit floor)."""
    from render_cards import texts_by_ctx, tries
    store, failed = texts_by_ctx("resumes"), tries("resumes")
    ctxs = {ctx_of(r["card"]) for r in test_cards}
    tried = [h for h in ctxs if h in store or failed[h]]
    n_tries = sum(failed[h] + (h in store) for h in tried)
    rows = credits = 0.0
    for p in glob.glob(str(STATE / "render_resumes_*.json")):
        s = json.loads(Path(p).read_text())
        if s.get("rows", 0) >= 100 and s.get("estimate_credits"):
            rows += s["rows"]
            credits += s["estimate_credits"]
    per_try = credits / rows if rows else CREDITS_PER_TRY
    return {"test_contexts_tried": len(tried), "tries": int(n_tries),
            "tries_per_context": round(n_tries / len(tried), 3) if tried else TRIES_PER_CONTEXT,
            "kept_share": round(sum(h in store for h in tried) / len(tried), 3) if tried else None,
            "first_try_pass": round(sum(h in store and not failed[h] for h in tried) / len(tried), 3) if tried else None,
            "credits_per_try": round(per_try, 5), "batch_rows_priced": int(rows), "batch_credits_priced": credits}


def cap_plan(rows: list[dict], k: int, seed: int, rate: dict) -> dict:
    """resume_plan for the rows a per-posting cap keeps, with each split's bucket mix (the cap round-robins over kinds,
    and there are more Skip kinds than Keep kinds)."""
    capped = cap_rows(rows, k, seed)
    mix = {}
    for s in SPLITS:
        c = Counter(r["bucket"] for r in capped if r["split"] == s)
        n = max(sum(c.values()), 1)
        mix[s] = {"bucket_counts": {b: c[b] for b in brt.BUCKETS}, "bucket_share": {b: round(c[b] / n, 3) for b in brt.BUCKETS}}
    return resume_plan(capped, rate) | {"mix": mix}


def resume_plan(rows: list[dict], rate: dict) -> dict:
    """Unique resume contexts of the rows, how many already have a kept text, how many gave up (3 failed tries), how
    many must be rendered, and the credit estimate for those at the measured rate."""
    from render_cards import MAX_REDO, texts_by_ctx, tries
    store, failed = texts_by_ctx("resumes"), tries("resumes")
    ctxs = {r["ctx"] for r in rows}
    have = {h for h in ctxs if h in store}
    gave_up = {h for h in ctxs - have if failed[h] > MAX_REDO}
    todo = ctxs - have - gave_up
    credits = len(todo) * rate["tries_per_context"] * rate["credits_per_try"]
    kept = len(todo) * (rate["kept_share"] or 1.0)
    return {"rows": len(rows), "candidate_cards": len({r["cd_id"] for r in rows}), "unique_contexts": len(ctxs),
            "already_rendered": len(have), "gave_up": len(gave_up), "to_render": len(todo),
            "estimated_tries": round(len(todo) * rate["tries_per_context"]),
            "estimated_credits": round(credits, 1), "expected_kept_resumes": round(kept)}


# ---------------------------------------------------------------- checked cards (dev)


CARD_FIELDS = ("degree_level", "enrolled_in", "min_years", "must_skills", "license", "work_mode", "sponsorship",
               "citizens_only", "employment_type", "locations", "salary_min", "salary_max")


def card_diff(a: RequirementCard, b: RequirementCard) -> list[str]:
    """Label-relevant fields on which two cards differ (degree_fields left out: the cleaned cards add "related field")."""
    return [f for f in CARD_FIELDS if getattr(a, f) != getattr(b, f)]


def checked_agreement(rows: list[dict], checked: dict[str, RequirementCard],
                      extracted: dict[str, RequirementCard] | None = None, postings: dict[str, str] | None = None) -> dict:
    """On the rows of postings with a checked card: how often rules.label on the checked card gives the row's bucket
    (the extracted card's). Per bucket, per variant kind, per posting, and apart for the postings whose two cards
    agree on enrolled_in (the field the disagreement concentrates on).

    With `postings`, also over the rows whose checked label the shown posting states (build_real_test.posting_misses
    on the checked card's reasons, as the test's build checks its own rows): a checked fact the text the model reads
    does not show ("enrolled_in bachelor" on a posting that never names students) is the checker's knowledge, not an
    error a reader of the prompt could see. The raw agreement is then a floor, this one the comparable number."""
    from matcher.profile import render_profile
    sub = [r for r in rows if r["rq_id"] in checked]
    if not sub:
        return {"rows": 0}
    agree, readable = Counter(), Counter()
    per_bucket: dict[str, Counter] = defaultdict(Counter)
    readable_bucket: dict[str, Counter] = defaultdict(Counter)
    per_kind: dict[str, Counter] = defaultdict(Counter)
    per_post: dict[str, Counter] = defaultdict(Counter)
    confusion, readable_confusion = Counter(), Counter()
    shown = {i: brt.shown_posting(postings[i]) for i in {r["rq_id"] for r in sub}} if postings else {}
    for r in sub:
        cand, card = CandidateCard.from_dict(r["candidate"]), checked[r["rq_id"]]
        gold, reasons, _ = label(cand, card)
        ok = gold == r["bucket"]
        agree[ok] += 1
        per_bucket[r["bucket"]][ok] += 1
        per_kind[r["variant_kind"]][ok] += 1
        per_post[r["rq_id"]][ok] += 1
        confusion[f"{r['bucket']} -> {gold}"] += 1
        if shown and not brt.posting_misses(cand, card, reasons, render_profile(cand), shown[r["rq_id"]],
                                            hard=gold == "Excluded"):
            readable[ok] += 1
            readable_bucket[r["bucket"]][ok] += 1
            readable_confusion[f"{r['bucket']} -> {gold}"] += 1
    rate = lambda c: round(c[True] / max(c[True] + c[False], 1), 3)  # noqa: E731
    out = {"rows": len(sub), "postings": len(per_post), "agreement": rate(agree),
           "per_bucket": {b: {"n": sum(c.values()), "agreement": rate(c)} for b, c in sorted(per_bucket.items())},
           "extracted_label_to_checked_label": dict(confusion.most_common()),
           "worst_kinds": {k: {"n": sum(c.values()), "agreement": rate(c)}
                           for k, c in sorted(per_kind.items(), key=lambda kv: rate(kv[1]))[:10]}}
    if shown:
        out["checked_label_readable"] = {
            "rows": sum(readable.values()), "agreement": rate(readable),
            "per_bucket": {b: {"n": sum(c.values()), "agreement": rate(c)} for b, c in sorted(readable_bucket.items())},
            "extracted_label_to_checked_label": dict(readable_confusion.most_common())}
    if extracted:
        diffs = {i: card_diff(extracted[i], checked[i]) for i in per_post if i in extracted}
        same_enr = Counter()
        for i, c in per_post.items():
            if "enrolled_in" not in diffs.get(i, ()):
                same_enr.update(c)
        out["agreement_where_cards_agree_on_enrolled_in"] = {"rows": sum(same_enr.values()), "agreement": rate(same_enr)}
        out["per_posting"] = [{"id": i, "rows": sum(c.values()), "agreement": rate(c), "card_diff": diffs.get(i)}
                              for i, c in sorted(per_post.items(), key=lambda kv: (rate(kv[1]), kv[0]))]
    return out


# ---------------------------------------------------------------- build


def split_report(rows: list[dict], dropped: list[dict], cards: dict[str, RequirementCard]) -> dict:
    rep = brt.report_of(rows, dropped, len(cards), cards)
    keep = ("n_rows", "n_postings_with_rows", "n_postings", "posting_employment_types", "bucket_counts", "bucket_share",
            "per_kind", "dropped", "dropped_by_guard", "unique_resume_cards", "keep_rows", "resume_kind_rows")
    return {k: rep[k] for k in keep}


def build(a: argparse.Namespace) -> dict:
    raw = load_raw()
    postings = {i: brt.posting_text(p) for i, p in raw.items()}
    tests = test_ids()
    pool_cards, pool_counts = load_pool(tests=tests)
    kept_ids, excluded, border, same_co = exclude_near(list(pool_cards), raw, tests)
    by_rule = Counter(e["rule"] for e in excluded)
    print(f"pool: {pool_counts['pool']} of {pool_counts['v2_rows']} v2 cards may train ({pool_counts['left_out']}); "
          f"near copies of a test posting: {len(excluded)} more out {dict(by_rule)}; {len(kept_ids)} postings left")
    checked = checked_cards(a.checked)  # the agreement measure's cards; the forced dev ids are CHECKED's either way
    comps = components(kept_ids, raw)
    forced = set(checked_card_dicts("raw")) & set(kept_ids)
    split = split_components(comps, forced, a.dev_share, a.seed)
    cards = {i: pool_cards[i] for i in kept_ids}
    rows, dropped, errors = build_posting_rows(cards, postings, a.seed)
    for r in rows + dropped:
        r["split"] = split[r["rq_id"]]
    from render_cards import ctx_hash, ctx_str  # noqa: F401  (render_cards needs the adaption package)
    test_rows_cards = read_jsonl(brt.OUT / "resume_cards.jsonl")
    test_ctx = {ctx_of(r["card"]) for r in test_rows_cards}
    rows, shared = drop_shared_resumes(rows, test_ctx)
    dropped += shared
    full = rows
    rows = cap_rows(rows, a.per_posting, a.seed)
    capped = [r | {"drop": "per_posting_cap", "detail": f"over {a.per_posting} rows a posting"} for r in full
              if r["pair_id"] not in {x["pair_id"] for x in rows}]
    dropped += capped
    rate = measured_render_rate(test_rows_cards)
    v2_meta = {r["id"]: r.get("meta") for r in read_jsonl(V2)}
    rep = {
        "seed": a.seed, "per_posting": a.per_posting,
        "pool": pool_counts,
        "exclusion": {
            "rules": {"same company, similar title": f"company_key equal and title_core Jaccard >= {TITLE_SIM}",
                      "most of the text shared": f"description 6-word-window Jaccard >= {TEXT_JACCARD}",
                      "same employer under another name": f"other company_key and >= {CONTAINMENT} of the shorter "
                                                          "description's 6-word windows shared"},
            "excluded": len(excluded), "by_rule": dict(by_rule),
            "test_postings_matched": len({e["test_id"] for e in excluded}),
            "whole_company_alternative": {"pool_postings_sharing_a_company_with_a_test_posting": same_co,
                                          "note": "a whole-company exclusion would remove these (the near-copy "
                                                  "postings among them included)"},
            "borderline_kept": len(border), "borderline_examples": border[:12]},
        "split": {"components": len(comps), "largest_components": sorted((len(c) for c in comps), reverse=True)[:8],
                  "postings": dict(Counter(split.values())),
                  "dev_share": round(sum(v == "dev" for v in split.values()) / max(len(split), 1), 3),
                  "checked_postings_in_dev": len(forced),
                  "checked_postings_not_in_dev": dict(Counter(why_out(i, tests, v2_meta, excluded) for i in set(checked) - forced))},
        "build_errors": errors,
        "rows": {s: split_report([r for r in rows if r["split"] == s], [r for r in dropped if r["split"] == s],
                                 {i: c for i, c in cards.items() if split[i] == s}) for s in SPLITS},
        "dropped_resume_shared": dict(Counter(r["drop"] for r in shared)),
        "dropped_per_posting_cap": len(capped),
        "checked_card_agreement": checked_agreement([r for r in rows if r["split"] == "dev"], checked, cards, postings),
        "render_rate": rate,
        "resumes": {s: resume_plan([r for r in rows if r["split"] == s], rate) for s in SPLITS}
        | {"all": resume_plan(rows, rate)},
        "resumes_at_cap": {str(k): cap_plan(full, k, a.seed, rate) for k in CAPS},
        "credit_note": "the estimate is a floor: tries a context were measured on the real test's candidates, built "
                       "from hand-checked cards; extracted cards carry longer skill lists, and the resume gates fail a "
                       "text that drops or adds a skill, so expect more tries",
        "test_rebuild_note": "build_real_test.build_rows on the 87 test cards now keeps 1,190 rows; the committed "
                             "real_test/rows.jsonl (1,200) predates the onsite-guard commit d1b02b4",
    }
    print_report(rep)
    if a.dry_run:
        print("dry run: nothing written")
        return rep
    OUT.mkdir(parents=True, exist_ok=True)
    brt.write_jsonl(OUT / "rows.jsonl", rows)
    brt.write_jsonl(OUT / "dropped.jsonl", dropped)
    brt.write_jsonl(OUT / "excluded.jsonl", excluded)
    brt.write_jsonl(OUT / "borderline.jsonl", border)
    (OUT / "splits.json").write_text(json.dumps(dict(sorted(split.items())), indent=0))
    resume_cards = {r["cd_id"]: {"id": r["cd_id"], "card": r["candidate"]} for r in rows}
    brt.write_jsonl(OUT / "resume_cards.jsonl", list(resume_cards.values()))
    (OUT / "resume_ids.txt").write_text("\n".join(resume_cards) + "\n")
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}/{{rows,dropped,excluded,borderline,resume_cards}}.jsonl, splits.json, "
          "resume_ids.txt, report.json")
    return rep


def print_report(rep: dict) -> None:
    ex = rep["exclusion"]
    print(f"near copies: {ex['excluded']} postings out {ex['by_rule']} (matching {ex['test_postings_matched']} test "
          f"postings); whole-company exclusion would remove "
          f"{ex['whole_company_alternative']['pool_postings_sharing_a_company_with_a_test_posting']}; "
          f"{ex['borderline_kept']} borderline kept")
    sp = rep["split"]
    print(f"split: {sp['postings']} postings in {sp['components']} components (dev share {sp['dev_share']}, largest "
          f"{sp['largest_components']}); {sp['checked_postings_in_dev']} checked-card postings in dev")
    if rep["build_errors"]:
        print(f"build errors on {len(rep['build_errors'])} postings: {list(rep['build_errors'].items())[:3]}")
    for s in SPLITS:
        r = rep["rows"][s]
        print(f"{s}: {r['n_rows']} rows over {r['n_postings_with_rows']} of {r['n_postings']} postings; buckets "
              f"{r['bucket_counts']} shares {r['bucket_share']}")
    print("dropped (resume shared):", rep["dropped_resume_shared"], "| per-posting cap:", rep["dropped_per_posting_cap"])
    ca = rep["checked_card_agreement"]
    if ca.get("rows"):
        print(f"checked-card agreement on dev: {ca['agreement']} over {ca['rows']} rows of {ca['postings']} postings; "
              f"per bucket {ca['per_bucket']}; where the cards agree on enrolled_in "
              f"{ca.get('agreement_where_cards_agree_on_enrolled_in')}")
        if ca.get("checked_label_readable"):
            cr = ca["checked_label_readable"]
            print(f"  where the checked label is readable from the shown posting: {cr['agreement']} over {cr['rows']} rows; "
                  f"per bucket {cr['per_bucket']}")
    rt = rep["render_rate"]
    print(f"render rate (real test): {rt['tries_per_context']} tries a context, first-try pass {rt['first_try_pass']}, "
          f"kept {rt['kept_share']}; {rt['credits_per_try']} credits a try")
    for s, v in rep["resumes"].items():
        print(f"resumes {s}: {v['unique_contexts']} contexts, {v['already_rendered']} rendered, {v['gave_up']} gave up, "
              f"{v['to_render']} to render, ~{v['estimated_credits']} credits, ~{v['expected_kept_resumes']} kept")
    for k, v in rep["resumes_at_cap"].items():
        print(f"  at {k} rows a posting: {v['rows']} rows, {v['to_render']} to render, ~{v['estimated_credits']} credits; "
              f"shares {({s: m['bucket_share'] for s, m in v['mix'].items()})}")
    print("credits:", rep["credit_note"])


# ---------------------------------------------------------------- assemble


def assemble(a: argparse.Namespace) -> dict[str, list[dict]]:
    from render_cards import texts_by_ctx
    rows = read_jsonl(OUT / "rows.jsonl")
    if not rows:
        sys.exit(f"no rows: run `build` first ({OUT.relative_to(ROOT)}/rows.jsonl)")
    raw = load_raw()
    postings = {i: brt.posting_text(raw[i]) for i in {r["rq_id"] for r in rows}}
    cards = {r["id"]: RequirementCard.from_dict(r["card"]) for r in read_jsonl(V2) if r["id"] in postings}
    store = texts_by_ctx("resumes")
    out = {}
    for s in SPLITS:
        sub = [r for r in rows if r["split"] == s]
        # the card's current context, as build_real_test.assemble looks it up (a context changed since build has no text)
        got, missing, undecidable = brt.assemble_rows(sub, cards, postings, lambda r: store.get(ctx_of(r["candidate"])))
        keep = [r | {"candidate_kind": "real_posting", "split": s} for r in got if r["decidable"]]
        brt.write_jsonl(OUT / f"{s}.jsonl", keep)
        out[s] = keep
        print(f"{s}.jsonl: {len(keep)} rows over {len({r['rq_id'] for r in keep})} postings; {len(missing)} with no "
              f"rendered resume, {len(got) - len(keep)} not decidable from the rendered text {dict(undecidable)}; "
              f"buckets {dict(Counter(r['bucket'] for r in keep))}")
    return out


def agreement(a: argparse.Namespace) -> dict:
    """checked_agreement on the built dev rows (rows.jsonl, as build measures it) for each card source in
    a.checked, without rebuilding anything. For "clean" also apart for its two parts (the 28 fresh postings, the
    cardcheck ones). Writes a.out."""
    rows = [r for r in read_jsonl(OUT / "rows.jsonl") if r["split"] == "dev"]
    if not rows:
        sys.exit(f"no dev rows: run `build` first ({OUT.relative_to(ROOT)}/rows.jsonl)")
    raw = load_raw()
    postings = {i: brt.posting_text(raw[i]) for i in {r["rq_id"] for r in rows}}
    cards = {r["id"]: RequirementCard.from_dict(r["card"]) for r in read_jsonl(V2) if r["id"] in postings}
    fresh = set(checked_card_dicts("raw"))
    out = {}
    for s in a.checked:
        checked = checked_cards(s)
        out[s] = checked_agreement(rows, checked, cards, postings)
        if s == "clean":
            for part, keep in (("clean_fresh_postings", lambda i: i in fresh),
                               ("clean_cardcheck_postings", lambda i: i not in fresh)):
                out[part] = checked_agreement(rows, {i: c for i, c in checked.items() if keep(i)}, cards, postings)
    for k, ca in out.items():
        if ca.get("rows"):
            cr = ca.get("checked_label_readable", {})
            print(f"{k}: {ca['agreement']} over {ca['rows']} rows of {ca['postings']} postings; per bucket "
                  f"{ca['per_bucket']}; readable {cr.get('agreement')} over {cr.get('rows')} rows "
                  f"{cr.get('per_bucket')}")
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--dev-share", type=float, default=DEV_SHARE)
    b.add_argument("--per-posting", type=int, default=0)
    b.add_argument("--dry-run", action="store_true")
    b.add_argument("--checked", choices=CHECKED_SOURCES[:2], default="raw",
                   help="cards for report.json's checked_card_agreement (default raw, as always)")
    sub.add_parser("assemble")
    g = sub.add_parser("agreement")
    g.add_argument("--checked", nargs="+", choices=CHECKED_SOURCES, default=list(CHECKED_SOURCES))
    g.add_argument("--out", default=str(OUT / "clean_checked/checked_agreement.json"))
    a = ap.parse_args()
    {"build": build, "assemble": assemble, "agreement": agreement}[a.cmd](a)


if __name__ == "__main__":
    main()
