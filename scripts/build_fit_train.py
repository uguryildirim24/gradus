"""v4 fit-layer training data and the selection dev set, built through Adaption from the train and dev pools only.

The model's job in v4 is the fit layer: among jobs the eligibility rules let through (src/matcher/hybrid.py H' on
Adaption's extracted card; scripts/hybrid_rules.py on this branch), does the resume meet what the posting asks, read
with hedged wording ("such as", "e.g.", "preferred", "bonus", duties) and saying Explore when a posting says too little.
Nothing here reads the test pool (data/built/fit/pools/test.jsonl) or data/built/fresh_test: load_pool refuses "test".

  build_fit_train.py sample            pick the venator postings: ~1,500 train-pool postings stratified by field and
                                        length (employer cap), the train-pool teasers (screened) and a share of
                                        empty-body postings, and every dev-pool posting -> build/sample.jsonl
  build_fit_train.py extract estimate|run|collect [--max-credits N]
                                        the v2 extractor unchanged (extract_postings.adapt_kwargs, blueprint.parse_card,
                                        posting_meta) on the sample -> build/extract_cards.jsonl; thin_posting is set
                                        from the card and the posting body (rules.mark_thin)
  build_fit_train.py twins             counterfactual twin postings (see TWINS) -> build/twin_cards.jsonl
  build_fit_train.py candidates        candidates for the venator, teaser and dev rows -> build/candidates.jsonl
  build_fit_train.py render postings|resumes estimate|run [--max-credits N]
                                        Adaption adapt renders (twin postings; rich resumes), gated as render_cards.py
                                        gates them, failed texts sent again with their findings (at most 3 tries)
  build_fit_train.py assemble          label, guard, trace and balance -> data/built/fit/train_fit.csv,
                                        dev_fit.jsonl, dpo_pairs.csv, build/report.json
  build_fit_train.py check             leakage guessers, employer and near-copy checks -> build/checks.json

Every Adaption call reads the balance before and after (datasets.invent(estimate=True), free) and logs it in
build/credits.jsonl; `run` estimates first and refuses over --max-credits.

TWINS. A synthetic requirement card with at least 2 must-skill groups, and a must_skill candidate of it whose resume is
already rendered and gated (variant_candidates.jsonl: the full match minus one required skill X, so rules.label says
Skip "missing required skill: X"). Its twin set moves X (one per such candidate on the card), every other group stays
required:
  req   X required, as on the card (the candidate: Skip)
  ex    X as one example of a kind ("experience with a programming language such as Python, Java or C++"): on the card
        one group of alternatives, plus the kind's generic name when schema.SHOWN_BY has one (any programming language
        meets "Programming"). When the candidate holds another skill of that kind it is one of the examples (the
        candidate: Keep); otherwise the examples are skills the candidate lacks (Skip, "needs one of"), so the wording
        alone never decides the label.
  pref  X moved to nice_skills, under preferred or bonus wording (Keep or Explore by the nice-gap rule)
  duty  X off the card, named only in a duty sentence ("you'll build dashboards in X") (Keep)
Every variant is rendered by Adaption from its card with the wording it demands, in one writer, so the four texts share
a style; the gates check that the wording is there (the such-as list in one sentence, a preferred skill only in
preferred wording, a duty skill never stated as required). The same candidates go on every variant, so a label flips
only with the wording; a second profile draw of each candidate (same resume, one preference or job type changed) goes
on two hedged variants, so hedged wording also meets eligibility fails and soft preferences.
Train sets come from the cards mix3 trained on; dev sets from the synthetic cards it never saw (old dev, dev_cross and
synthetic-test cards).

PROFILES. A blank form field must not mark a row as a likely Keep, nor a set one as a likely Explore or Skip: the first
draw of every twin candidate and every venator and teaser row get their blank fields filled to values the posting meets
(neutral_fill; the resume is unchanged, the label is the rule's on the filled card). What a profile guesser still finds
is filtered out: assemble drops the twin, venator and teaser rows a grouped-CV profile-only guesser finds easiest, an
equal number per label, until it scores at most DEBIAS_TO (debias, AFLite-style; mix3 rows are never dropped), and the
dev set keeps, at equal labels, the rows guessers trained on train do not find easy, until none of them beats
DEV_GUESS_MAX (dev_filter; every dev row, with in_dev, is in build/dev_fit_all.jsonl).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from matcher.blueprint import parse_card, phrase_groups, posting_meta, training_ok, MUST_MANY  # noqa: E402
from matcher.rules import mark_thin  # noqa: E402
from matcher.schema import RequirementCard  # noqa: E402

FIT = ROOT / "data/built/fit"
BUILD = FIT / "build"
STATE = ROOT / "data/adaption"
TRAIN_CSV = FIT / "train_fit.csv"
DEV_JSONL = FIT / "dev_fit.jsonl"
DPO_CSV = FIT / "dpo_pairs.csv"
SEED = 20261021

SAMPLE = BUILD / "sample.jsonl"
EXTRACT_CARDS = BUILD / "extract_cards.jsonl"
EXTRACT_RAW = BUILD / "extract_raw.jsonl"
EXTRACT_FAIL = BUILD / "extract_failures.jsonl"
CREDITS_LOG = BUILD / "credits.jsonl"

# sample sizes
TRAIN_REAL = 1500
EMPLOYER_CAP = 25          # train postings per employer (Quest Diagnostics alone has 1,931 US postings)
TEASER_EMPLOYER_CAP = 6
EMPTY_N = 140              # empty-body train postings (title and location only): thin like teasers
EMPTY_EMPLOYER_CAP = 4
LENGTH_SHARE = {"700-1300": 1.0, "1300-3000": 0.6}   # take this share of what a field has in the shorter strata
FIELDS = ("internship", "health_nursing", "lab_bio", "data", "software", "sales_ops", "other")


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []


def write_jsonl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def h(*parts) -> str:
    return hashlib.sha1("\x00".join(str(p) for p in parts).encode()).hexdigest()


def load_pool(name: str) -> list[dict]:
    """A frozen pool, checked against eval/fit_freeze.json. Only train and dev: this package never reads the test."""
    if name not in ("train", "dev"):
        raise SystemExit(f"build_fit_train reads the train and dev pools only, not {name!r}")
    import fit_pools
    return fit_pools.load_pool(name)


# ---------------------------------------------------------------- sample

FOREIGN = re.compile(r"\b(?:India|China|Japan|Tokyo|Shanghai|Beijing|Singapore|Germany|France|Spain|Italy|Ireland|"
                     r"United Kingdom|England|Scotland|Canada|Ontario|Quebec|Toronto|Montreal|Vancouver|Mexico|Brazil|"
                     r"Argentina|Chile|Colombia|Australia|Sydney|Melbourne|Netherlands|NLD|Belgium|Switzerland|Swiss|"
                     r"Austria|Poland|Hungary|Budapest|Czech|Prague|Denmark|Sweden|Norway|Finland|Israel|Korea|Seoul|"
                     r"Taiwan|Philippines|Manila|Taguig|Malaysia|Kuala Lumpur|Thailand|Vietnam|Indonesia|Hyderabad|"
                     r"Bengaluru|Bangalore|Mumbai|Pune|Chennai|Gurgaon|Gurugram|Noida|Hong Kong|Dubai|UAE|Saudi|Egypt|"
                     r"South Africa|Nigeria|Kenya|Portugal|Lisbon|Greece|Turkey|Romania|Bulgaria|Slovakia|Croatia|"
                     r"Serbia|Ukraine|Russia|Puerto Rico|Costa Rica|Panama|Peru|Hennigsdorf|Visp|Basel|Zurich|Geneva|"
                     r"Dublin|Cork|London|Paris|Berlin|Munich|Frankfurt|Madrid|Barcelona|Milan|Rome|Amsterdam|Leiden|"
                     r"Copenhagen|Stockholm|Oslo|Helsinki|Vienna|Warsaw|Krakow)\b|^(?:CH|NL|DE|FR|GB|UK|IE|IN|CN|JP|"
                     r"BE|ES|IT|PL|SG|AU|CA|MX|BR|KR) - ")
JUNK_TITLE = re.compile(r"DO NOT APPLY|TEST REQ|\btest posting\b|\bdo not use\b", re.I)
# a teaser body that is only legal or scam-warning boilerplate says nothing about the job either; it stays a teaser,
# but one copy of such a body is enough
BOILER = re.compile(r"equal opportunity|EEO|scam|disabilit|veteran|accommodation|background check|drug screen", re.I)


def us_like(p: dict) -> bool:
    """Not visibly outside the US by its location line or the start of its text (posting_meta reads venator's
    location strings, 'Boston-MA', '2 Locations', as unknown, so it cannot decide alone)."""
    loc = p.get("locations") or ""
    if FOREIGN.search(loc) or FOREIGN.search(p.get("title", "")):
        return False
    m = posting_meta([loc] if loc else [], p.get("text", "")[:3000])
    return m["country"] != "other" and m["language"] == "en"


def norm_text(s: str) -> str:
    return " ".join(re.sub(r"[\W_]+", " ", s.casefold()).split())


def sample(_: argparse.Namespace) -> None:
    rng = random.Random(SEED)
    train = load_pool("train")
    dev = load_pool("dev")
    out = []
    # 1. real train postings: stratified by field and length, employer cap, one copy of a description
    seen_text: set[str] = set()
    by_emp: Counter = Counter()
    real = [p for p in train if p["chars"] >= 700 and us_like(p) and not JUNK_TITLE.search(p["title"])]
    rng.shuffle(real)
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in real:
        strata[(p["field"], p["length"])].append(p)
    per_field = TRAIN_REAL // len(FIELDS)
    picked: list[dict] = []

    def take(pool: list[dict], n: int) -> list[dict]:
        got = []
        for p in pool:
            if len(got) >= n:
                break
            k = norm_text(p["text"])[:4000]
            if k in seen_text or by_emp[p["emp_key"]] >= EMPLOYER_CAP:
                continue
            seen_text.add(k)
            by_emp[p["emp_key"]] += 1
            got.append(p)
        return got

    short_of = 0
    for f in FIELDS:
        want = per_field
        got = []
        for ln in ("700-1300", "1300-3000"):
            pool = strata.get((f, ln), [])
            got += take(pool, min(int(len(pool) * LENGTH_SHARE[ln]), want - len(got)))
        got += take(strata.get((f, "3000+"), []), want - len(got))
        short_of += want - len(got)
        picked += got
    # a field that ran out (software, internship) leaves its share to the largest fields
    if short_of:
        rest = [p for f in ("other", "lab_bio", "health_nursing", "sales_ops", "data") for p in strata.get((f, "3000+"), [])]
        rng.shuffle(rest)
        picked += take(rest, short_of)
    out += [p | {"split": "train", "kind": "real"} for p in picked]
    # 2. teasers: train-pool postings with a body under 700 characters, screened
    teasers, t_emp, t_text = [], Counter(), set()
    for p in sorted((p for p in train if 0 < p["chars"] < 700), key=lambda p: p["key"]):
        k = norm_text(p["text"])
        if JUNK_TITLE.search(p["title"]) or not us_like(p) or k in t_text or t_emp[p["emp_key"]] >= TEASER_EMPLOYER_CAP:
            continue
        if BOILER.search(p["text"]) and len(k.split()) < 120 and any(norm_text(x["text"])[:200] == k[:200] for x in teasers):
            continue
        t_text.add(k)
        t_emp[p["emp_key"]] += 1
        teasers.append(p)
    out += [p | {"split": "train", "kind": "teaser"} for p in teasers]
    # 3. empty bodies: the title and location are all a reader gets
    empties = [p for p in train if p["chars"] == 0 and us_like(p) and not JUNK_TITLE.search(p["title"])
               and p["locations"] and not re.match(r"^\d+ Locations$", p["locations"])]
    rng.shuffle(empties)
    e_emp, e_title, got = Counter(), set(), []
    for p in empties:
        t = norm_text(p["title"])
        if t in e_title or e_emp[p["emp_key"]] >= EMPTY_EMPLOYER_CAP:
            continue
        e_title.add(t)
        e_emp[p["emp_key"]] += 1
        got.append(p)
        if len(got) >= EMPTY_N:
            break
    out += [p | {"split": "train", "kind": "teaser"} for p in got]
    # 4. dev: every dev-pool posting (dev employers only), as the fresh test will see them
    out += [p | {"split": "dev", "kind": "teaser" if p["chars"] < 700 else "real"} for p in dev]
    write_jsonl(SAMPLE, out)
    c = Counter((r["split"], r["kind"]) for r in out)
    print(f"sample: {len(out)} postings {dict(c)}; train real by field "
          f"{dict(Counter(r['field'] for r in out if r['split'] == 'train' and r['kind'] == 'real'))}, by length "
          f"{dict(Counter(r['length'] for r in out if r['split'] == 'train' and r['kind'] == 'real'))}; "
          f"{len(teasers)} teasers and {len(got)} empty bodies; employers in train real {len(by_emp)}")


# ---------------------------------------------------------------- Adaption plumbing

def client():
    import extract_postings as ep
    return ep.client()


def balance(c) -> float:
    return c.datasets.invent(rows=10, estimate=True, prompt="balance check").available_credits


def log_credits(what: str, before: float, after: float, extra: dict | None = None) -> None:
    CREDITS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with CREDITS_LOG.open("a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "what": what, "before": before, "after": after,
                            "spent": round(before - after, 3)} | (extra or {})) + "\n")


def wait_done(c, did: str, interval: float = 30.0, settle: int = 4):
    """Wait until every row is processed, or the count stops moving for `settle` polls (rows adapt dropped)."""
    last, still = -1, 0
    while True:
        for attempt in range(8):
            try:
                st = c.datasets.get_status(did)
                break
            except Exception as e:  # noqa: BLE001  a dropped connection sleeps, it does not end the wait
                print("status retry", type(e).__name__, e, flush=True)
                time.sleep(min(300, 10 * 2 ** attempt))
        n, tot = (st.progress.processed_rows, st.progress.total_rows) if st.progress else (0, 0)
        print(time.strftime("%H:%M:%S"), st.status, f"{n}/{tot}", flush=True)
        if st.status == "failed":
            return st, n, tot
        if st.status == "succeeded":
            if tot and n >= tot:
                return st, n, tot
            still = still + 1 if n == last else 0
            if still >= settle:
                return st, n, tot
        last = n
        time.sleep(interval)


def download(c, did: str) -> list[dict]:
    res = c.datasets.download(did, file_format="csv")
    data = res.read() if hasattr(res, "read") else res.content
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))


def upload_rows(c, path: Path, header: list[str], rows: list[list[str]], name: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    did = c.datasets.upload_file(str(path), name=name).dataset_id
    for _ in range(120):  # adapt answers 409 until the import has a row_count
        st = c.datasets.get_status(did)
        if st.row_count or st.status == "failed":
            break
        time.sleep(5)
    return did


# Adaption's estimate grows faster than the row count (1,996 extraction rows: 90 credits; the same postings as 300- and
# 400-row datasets: 3 and 5 credits each, about 0.01 a row), so every batch goes up in chunks of this many rows.
CHUNK = 250


def run_chunks(c, tag: str, header: list[str], rows: list[list[str]], kwargs: dict, mode: str, max_credits: float,
               name: str) -> list[dict] | None:
    """rows in CHUNK-row datasets: upload and estimate every chunk, refuse when the estimates add up to more than
    max_credits, launch every chunk, then wait for each and download. Returns all downloaded rows (None for an
    estimate). State files data/adaption/fit_<tag>_<i>.json make a rerun resume."""
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    states = [STATE / f"fit_{tag}_{i}.json" for i in range(len(chunks))]
    total = 0.0
    for i, (sp, ch) in enumerate(zip(states, chunks)):
        st = json.loads(sp.read_text()) if sp.exists() else {}
        if not st.get("dataset_id"):
            st = {"dataset_id": upload_rows(c, BUILD / "batches" / f"{tag}_{i}.csv", header, ch, f"{name}-{i}"),
                  "rows": len(ch)}
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_text(json.dumps(st, indent=1))
        if not st.get("run_id"):
            est = c.datasets.adapt(st["dataset_id"], estimate=True, **kwargs)
            st["estimate"] = est.estimated_credits_consumed
            sp.write_text(json.dumps(st, indent=1))
            total += st["estimate"]
        print(f"chunk {i}: {len(ch)} rows, estimate {st.get('estimate')}, run {st.get('run_id', '-')}")
    print(f"{name}: {len(rows)} rows in {len(chunks)} chunks; estimate for the chunks not yet launched: {total}")
    if mode == "estimate":
        return None
    if total > max_credits:
        sys.exit(f"estimate {total} > --max-credits {max_credits}; not running")
    before = balance(c)
    for sp in states:
        st = json.loads(sp.read_text())
        if not st.get("run_id"):
            r = c.datasets.adapt(st["dataset_id"], estimate=False, **kwargs)
            st.update(run_id=r.run_id, started=time.time())
            sp.write_text(json.dumps(st, indent=1))
    print("launched; balance before", before, flush=True)
    time.sleep(30)
    out = []
    for i, sp in enumerate(states):
        st = json.loads(sp.read_text())
        dl = sp.parent / f"{sp.stem}_download.jsonl"
        if not st.get("downloaded"):
            s2, n, tot = wait_done(c, st["dataset_id"])
            if s2.status != "succeeded":
                print(f"chunk {i} ended with status {s2.status}; its rows stay unrendered")
                continue
            got = download(c, st["dataset_id"])
            write_jsonl(dl, got)
            st.update(downloaded=True, processed=n, total=tot, finished=time.time())
            sp.write_text(json.dumps(st, indent=1))
        out += jsonl(dl)
    after = balance(c)
    log_credits(name, before, after, {"chunks": len(chunks), "rows": len(rows), "estimate": total})
    print(f"{name}: balance {before} -> {after}")
    return out


def run_adapt(c, state_path: Path, csv_path: Path, header: list[str], rows: list[list[str]], kwargs: dict,
              mode: str, max_credits: float, name: str) -> list[dict] | None:
    """Upload once (state file), estimate, refuse over the cap, adapt, wait, download. Returns the downloaded rows
    (None for an estimate). A rerun with the same state file resumes: no second upload or second adapt."""
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if not state.get("dataset_id"):
        state = {"dataset_id": upload_rows(c, csv_path, header, rows, name), "rows": len(rows), "csv": str(csv_path)}
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=1))
        print("UPLOADED", state["dataset_id"], len(rows), "rows")
    did = state["dataset_id"]
    if not state.get("run_id"):
        est = c.datasets.adapt(did, estimate=True, **kwargs)
        print("ESTIMATE", est.estimated_credits_consumed, "credits,", est.estimated_minutes, "min")
        state["estimate"] = est.estimated_credits_consumed
        state_path.write_text(json.dumps(state, indent=1))
        if mode == "estimate":
            return None
        if est.estimated_credits_consumed > max_credits:
            sys.exit(f"estimate {est.estimated_credits_consumed} > --max-credits {max_credits}; not running")
        before = balance(c)
        r = c.datasets.adapt(did, estimate=False, **kwargs)
        state.update(run_id=r.run_id, started=time.time(), balance_before=before)
        state_path.write_text(json.dumps(state, indent=1))
        print("RUN", r.run_id, "balance before", before)
        time.sleep(30)
    elif mode == "estimate":
        print("already launched:", state["run_id"])
        return None
    if not state.get("downloaded"):
        st, n, tot = wait_done(c, did)
        if st.status != "succeeded":
            sys.exit(f"adapt ended with status {st.status}")
        got = download(c, did)
        (state_path.parent / f"{state_path.stem}_download.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in got))
        after = balance(c)
        state.update(downloaded=True, processed=n, total=tot, balance_after=after, finished=time.time())
        state_path.write_text(json.dumps(state, indent=1))
        log_credits(name, state.get("balance_before", after), after, {"dataset_id": did, "rows": state["rows"],
                                                                       "estimate": state.get("estimate")})
    return jsonl(state_path.parent / f"{state_path.stem}_download.jsonl")


# ---------------------------------------------------------------- extraction (the v2 extractor, unchanged)

def as_ats(p: dict) -> dict:
    """A venator posting in the shape extract_postings.posting_text and posting_meta read."""
    return {"title": p["title"], "company": p.get("employer", ""), "locations": [p["locations"]] if p.get("locations") else [],
            "description": p.get("text", "")}


def extract(a: argparse.Namespace) -> None:
    import extract_postings as ep
    posts = {p["key"]: p for p in jsonl(SAMPLE)}
    done = {r["key"] for r in jsonl(EXTRACT_RAW) if r.get("completion")}
    # one row per posting text (adapt drops exact duplicates without a word)
    first: dict[str, str] = {}
    for k in sorted(posts):
        first.setdefault(ep.posting_text(as_ats(posts[k])), k)
    todo = [k for k in sorted(posts) if first[ep.posting_text(as_ats(posts[k]))] == k and k not in done]
    random.Random(SEED).shuffle(todo)  # sorted keys put one source's postings in one chunk; mixed chunks estimate lower
    tag = "extract_" + h(ep.BP_HASH, *todo)[:10]
    print(f"extract: {len(posts)} postings, {len(done)} extracted, {len(todo)} to send (blueprint {ep.BP_HASH})")
    if not todo:
        return parse_extracted()
    c = client()
    rows = [[k, ep.posting_text(as_ats(posts[k]))] for k in todo]
    got = run_chunks(c, tag, ["posting_id", "posting"], rows, ep.adapt_kwargs(), a.mode, a.max_credits, f"fit-{tag}")
    if got is None:
        return
    old = {r["key"]: r for r in jsonl(EXTRACT_RAW)}
    for r in got:
        t = (r.get("enhanced_completion") or "").strip()
        if r.get("posting_id") in posts and t:
            old[r["posting_id"]] = {"key": r["posting_id"], "completion": t}
    write_jsonl(EXTRACT_RAW, old.values())
    print(f"extract: {len(old)} completions on file")
    parse_extracted()


def parse_extracted() -> None:
    """Raw completions -> cards (+ meta), with thin_posting from rules.mark_thin on the posting body. A posting
    with no completion of its own takes the completion of one with the same sent text."""
    import extract_postings as ep
    posts = {p["key"]: p for p in jsonl(SAMPLE)}
    raw = {r["key"]: r["completion"] for r in jsonl(EXTRACT_RAW) if r.get("completion")}
    by_text = {}
    for k in sorted(raw):
        by_text.setdefault(ep.posting_text(as_ats(posts[k])), k)
    cards, fails = [], []
    for k, p in posts.items():
        src = k if k in raw else by_text.get(ep.posting_text(as_ats(p)))
        if src is None:
            continue
        post = as_ats(p)
        body = raw[src].strip()
        if body.startswith("```"):
            body = body.strip("`").removeprefix("json").strip()
        try:
            card = parse_card(k, body, ep.posting_text(post, None), post["locations"])
        except Exception as e:  # noqa: BLE001
            fails.append({"key": k, "error": f"{type(e).__name__}: {e}", "completion": raw[src][:600]})
            continue
        card = mark_thin(card, p.get("text", ""))
        meta = posting_meta(post["locations"], post["description"], card.locations)
        meta["must_phrases"] = phrase_groups([list(g) for g in card.must_skills])
        meta["must_many"] = len(card.must_skills) >= MUST_MANY
        meta["training_ok"] = training_ok(meta)
        if src != k:
            meta["borrowed_from"] = src
        cards.append({"key": k, "split": p["split"], "kind": p["kind"], "card": card.to_dict(), "meta": meta})
    write_jsonl(EXTRACT_CARDS, cards)
    write_jsonl(EXTRACT_FAIL, fails)
    c = Counter((r["split"], r["kind"], r["meta"]["training_ok"], r["card"].get("thin_posting", False)) for r in cards)
    print(f"extract: {len(cards)} cards, {len(fails)} failed to parse; (split, kind, training_ok, thin): {dict(c)}")


# ---------------------------------------------------------------- twins

TWIN_CARDS = BUILD / "twin_cards.jsonl"
TWIN_ROWS = BUILD / "twin_candidates.jsonl"
TWIN_TRAIN_MAX = 10_000    # every eligible train card
TWIN_DEV_SETS = 75
TWIN_KINDS = ("req", "ex", "pref", "duty")
# A skill -> (the kind a posting names it as an example of, the kind's other members, the kind's generic name that
# schema.SHOWN_BY lets any member meet, or None). The examples a twin lists are the moved skill plus two of the others.
_LANGS = ("Python", "Java", "C++", "C#", "JavaScript", "R", "MATLAB", "SAS", "Go", "TypeScript", "C", "Scala")
CATEGORIES = [
    (_LANGS, "a programming language", "Programming"),
    (("SQL Server", "MySQL", "PostgreSQL", "Oracle Database"), "a relational database", "Database software"),
    (("Microsoft Excel", "Microsoft Word", "Microsoft PowerPoint", "Microsoft Outlook", "Microsoft Access"),
     "Microsoft Office applications", "Microsoft Office"),
    (("AWS", "Azure", "Google Cloud"), "a cloud platform", "Cloud platforms"),
    (("PyTorch", "TensorFlow", "Keras", "scikit-learn"), "a machine learning framework", "Machine learning frameworks"),
    (("AutoCAD", "SolidWorks", "Revit", "CATIA"), "CAD software", "CAD"),
    (("Photoshop", "Illustrator", "InDesign", "Premiere Pro", "After Effects"), "Adobe design software",
     "Adobe Creative Cloud"),
    (("Epic", "MEDITECH", "eClinicalWorks"), "an electronic health record system", "Electronic medical record EMR"),
    (("Git", "GitHub", "GitLab", "Bitbucket", "Subversion"), "a version control tool", "Version control"),
    (("Tableau", "Power BI", "Looker", "Qlik"), "a BI and dashboard tool", None),
    (("SPSS", "Stata", "Minitab"), "a statistics package", None),
    (("PCR", "ELISA", "Western blot", "Flow cytometry", "Cell culture"), "laboratory techniques", None),
    (("SAP", "Oracle NetSuite", "Microsoft Dynamics"), "an ERP system", None),
    (("Salesforce", "HubSpot", "Zoho CRM"), "a CRM platform", None),
    (("QuickBooks", "Sage 50 Accounting", "Xero"), "accounting software", None),
    (("Jira", "Asana", "Trello"), "a project tracking tool", None),
    (("ArcGIS", "QGIS"), "GIS software", None),
    (("Biology", "Chemistry", "Physics", "Biochemistry"), "a natural science", None),
    (("Psychology", "Sociology", "Geography", "Anthropology"), "a social science", None),
    (("Microsoft Office", "Google Workspace", "LibreOffice"), "an office suite", None),
    (("Linux", "Windows Server", "macOS"), "an operating system", None),
    (("Statistics", "Linear algebra", "Calculus"), "quantitative coursework", None),
    (("Deep learning", "Natural language processing", "Computer vision"), "a machine learning area", None),
]


def category_of(skill: str):
    from matcher.schema import skill_canon
    k = skill_canon(skill)
    for members, kind, generic in CATEGORIES:
        if k in {skill_canon(m) for m in members}:
            return [m for m in members if skill_canon(m) != k], kind, generic
    return None


def _twin_sources() -> tuple[set[str], set[str]]:
    """(train base cards, held-out base cards): the synthetic cards smoke.csv (mix3's synthetic part) trained on, and
    the ones it never saw (old dev, dev_cross and synthetic-test splits)."""
    base = lambda x: x.split("~")[0]  # noqa: E731
    tr = {base(json.loads(l)["rq_id"]) for l in (ROOT / "data/built/training/train.jsonl").open()}
    held = set()
    for name in ("dev.jsonl", "dev_cross.jsonl"):
        held |= {base(json.loads(l)["rq_id"]) for l in (ROOT / "data/built/training" / name).open()}
    for m in jsonl(ROOT / "data/built/pairs/manifest.jsonl"):
        if m["split"] in ("test", "test_heldout_family", "dev"):
            held.add(base(m["rq_id"]))
    return tr, held - tr


def twin_context(base: dict, display: dict, ex_groups: dict, duty: list[str]) -> dict:
    """What the twin writer sees: render_cards' posting context of the base card (so the four variants share their
    opener, years wording and every other pick), with the display card's skills: example groups written as
    {"kind", "such_as"}, preferred skills, and the duty skills under their own key."""
    import render_cards as rc
    ctx = rc.context_of("postings", base)
    ctx.pop("required_skills", None)
    ctx.pop("preferred_skills", None)
    if display.get("nice_skills"):
        ctx["preferred_skills"] = list(display["nice_skills"])
    if display.get("must_skills"):
        out = []
        for g in display["must_skills"]:
            key = tuple(g)
            if key in ex_groups:
                out.append({"kind": ex_groups[key], "such_as": list(g)})
            else:
                out.append(rc.skill_entries([g])[0])
        ctx["required_skills"] = out
    if duty:
        ctx["duty_skills"] = list(duty)
    return ctx


def twins(_: argparse.Namespace) -> None:
    import render_cards as rc
    from matcher.rules import held_skills, label, required_groups
    from matcher.schema import CandidateCard, group_met, skill_canon
    from matcher.visibility import _names_lacked

    rng = random.Random(SEED + 1)
    reqs, cands = rc.load_cards("postings"), rc.load_cards("resumes")
    ms = {i: c for i, c in cands.items() if "_must_skill_" in i}
    texts = rc.rendered_texts("resumes", ms)
    tr, held = _twin_sources()
    by_rq: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for cid, c in ms.items():
        rq = "rq_" + cid.split("_")[2]
        if rq not in reqs or cid not in texts:
            continue
        card = RequirementCard.from_dict(reqs[rq])
        cand = CandidateCard.from_dict(c)
        groups = required_groups(card)
        if len(groups) < 2 or label(cand, card)[0] != "Excluded":
            continue
        unmet = [g for g in groups if not group_met(g, held_skills(cand))]
        if len(unmet) != 1 or len(unmet[0]) > 2:
            continue
        g = tuple(unmet[0])
        want = [f"missing required skill: {g[0]}"] if len(g) == 1 else [f"needs one of: {', '.join(g)}"]
        if label(cand, card)[1] != want:
            continue  # the candidate misses exactly one group (one skill or a pair of alternatives) and nothing else
        by_rq[rq].append((cid, g))
    rqs = sorted(by_rq)
    rng.shuffle(rqs)
    train_rq = [r for r in rqs if r in tr][:TWIN_TRAIN_MAX]
    dev_rq = [r for r in rqs if r in held][:TWIN_DEV_SETS]
    cards_out, rows_out = [], []
    for split, ids in (("train", train_rq), ("dev", dev_rq)):
        for rq in ids:
            base = reqs[rq]
            members = by_rq[rq]
            if split == "dev":
                members = members[:1]
            groups = [tuple(g) for g in base["must_skills"]]
            moved = list(dict.fromkeys(x for _, x in members))[:max(1, len(required_groups(RequirementCard.from_dict(base))) - 1)]
            members = [(cid, x) for cid, x in members if x in moved]
            flat = [n for g in moved for n in g]
            keep = [g for g in groups if g not in moved]
            variants = {"req": (dict(base), dict(base), {}, [])}
            variants["pref"] = (base | {"must_skills": [list(g) for g in keep], "nice_skills": [*base.get("nice_skills", []), *flat]},) * 2 + ({}, [])
            variants["duty"] = (base | {"must_skills": [list(g) for g in keep]},) * 2 + ({}, flat)
            # ex: each moved group with a kind becomes one example group
            ex_display, ex_label, ex_kinds, ex_any = [], [], {}, False
            for g in groups:
                cats = [category_of(n) for n in g] if g in moved else [None]
                # every alternative of the group must be of one kind ("PCB design or Python" has none)
                cat = cats[0] if all(c and c[1] == cats[0][1] for c in cats) else None
                if cat:
                    others, kind, generic = cat
                    others = [o for o in others if o not in g]
                    holders = [cid for cid, mx in members if mx == g]
                    sib_held = []
                    for cid in holders:
                        have = held_skills(CandidateCard.from_dict(cands[cid]))
                        sib_held += [o for o in others if skill_canon(o) in have]
                    lack = [o for o in others if o not in sib_held and not any(
                        _names_lacked(texts[cid], o, CandidateCard.from_dict(cands[cid])) for cid in holders)]
                    rng.shuffle(lack)
                    pick = ([rng.choice(sib_held)] if sib_held else []) + lack
                    if len(pick) < 2:
                        ex_display.append(list(g))
                        ex_label.append(list(g))
                        continue
                    shown = [*g, *pick[:3 - len(g)]]
                    rng.shuffle(shown)
                    ex_display.append(shown)
                    ex_label.append(shown + ([generic] if generic else []))
                    ex_kinds[tuple(shown)] = kind
                    ex_any = True
                else:
                    ex_display.append(list(g))
                    ex_label.append(list(g))
            if ex_any:
                variants["ex"] = (base | {"must_skills": ex_label}, base | {"must_skills": ex_display}, ex_kinds, [])
            for kind, (lab, disp, exk, duty) in variants.items():
                vid = f"tw_{rq}~{kind}"
                ctx = twin_context(base, disp, dict(exk), duty)
                cards_out.append({"id": vid, "base_rq": rq, "split": split, "variant": kind, "moved": flat,
                                  "label_card": lab, "display_card": disp, "duty_skills": duty,
                                  "ex_groups": [[list(k), v] for k, v in exk.items()], "context": ctx})
            for cid, x in members:
                rows_out.append({"cd_id": cid, "base_rq": rq, "split": split, "lacks": list(x), "draw": 0,
                                 "card": neutral_fill(cands[cid], base, random.Random(f"{SEED}:{cid}:fill")),
                                 "variants": sorted(variants)})
                # more profile draws of the same person (one preference or the job type changed, same resume), each on
                # hedged variants, so hedged wording also meets eligibility fails and soft preferences
                hedged = [k for k in sorted(variants) if k != "req"]
                rng.shuffle(hedged)
                used = set()
                for d, vs in ((1, hedged[:2]), (2, hedged[2:] or hedged[:1])):
                    alt = profile_redraw(cands[cid], base, rng, used)
                    if alt is None:
                        break
                    used.add(alt[1])
                    rows_out.append({"cd_id": cid, "base_rq": rq, "split": split, "lacks": list(x), "draw": d,
                                     "card": alt[0], "redraw": alt[1], "variants": sorted(vs)})
    write_jsonl(TWIN_CARDS, cards_out)
    write_jsonl(TWIN_ROWS, rows_out)
    c = Counter((r["split"], r["variant"]) for r in cards_out)
    print(f"twins: {len(cards_out)} variant cards {dict(c)}; {len(rows_out)} candidate draws "
          f"({Counter(r['split'] for r in rows_out)}); rows {sum(len(r['variants']) for r in rows_out)}")


REDRAWS = ("employment_type", "salary", "remote_only", "role_avoided", "industry_avoided")


def neutral_fill(cand: dict, card: dict, rng: random.Random) -> dict:
    """Form fields set the way people fill them, to values this posting meets (its job type, a salary floor under its
    pay, a work arrangement its mode allows, its industry wanted), each about half the time and only where the form
    left the field blank, so a set field is no sign of an Explore or Skip: the redraws and the failing kinds set the
    same fields to values the posting does not meet. Nothing the resume shows changes (the resume is looked up by
    the unfilled card)."""
    from matcher.schema import EMPLOYMENT_TYPES, CandidateCard
    out = dict(cand)
    fills = {}
    et = card.get("employment_type")
    if et in EMPLOYMENT_TYPES and not cand.get("employment_types_wanted") and rng.random() < 0.5:
        fills["employment_types_wanted"] = [et] if rng.random() < 0.6 else sorted({et, rng.choice(EMPLOYMENT_TYPES)})
    if card.get("salary_max") and cand.get("salary_floor") is None and rng.random() < 0.5:
        fills["salary_floor"] = max(5000, int(card["salary_max"] * rng.uniform(0.6, 0.97)) // 5000 * 5000)
    mode = card.get("work_mode", "onsite")
    if mode != "onsite" and cand.get("remote_preference", "any") == "any" and rng.random() < 0.6:
        fills["remote_preference"] = "remote_only" if mode == "remote" and rng.random() < 0.5 else "hybrid_or_remote"
    if (card.get("industry") and not cand.get("industries_wanted") and not cand.get("industries_avoided")
            and rng.random() < 0.3):
        fills["industries_wanted"] = [card["industry"]]
    for k, v in fills.items():
        try:
            CandidateCard.from_dict(out | {k: v})
        except (ValueError, TypeError):
            continue
        out[k] = v
    return out


def profile_redraw(cand: dict, card: dict, rng: random.Random, used=()) -> tuple[dict, str] | None:
    """The same person with one form field changed (never anything the resume shows: not the city, degree, years or
    skills): a job type the posting is not, a salary floor over its pay, remote only for an onsite or hybrid job, the
    title's role word or the posting's industry avoided."""
    from matcher.rules import industry_ids, role_hits
    from matcher.sample import AVOID_WORDS
    from matcher.schema import CandidateCard
    opts = []
    et = card.get("employment_type", "full_time")
    opts.append(("employment_type", {"employment_types_wanted": [rng.choice([t for t in ("full_time", "part_time",
                                                                              "internship", "contract") if t != et])]}))
    if card.get("salary_max"):
        opts.append(("salary", {"salary_floor": int(card["salary_max"] // 5000 * 5000 + 5000 * rng.randint(2, 6))}))
    if card.get("work_mode", "onsite") != "remote":
        opts.append(("remote_only", {"remote_preference": "remote_only"}))
    words = [w for w in AVOID_WORDS if re.search(rf"\b{w}", card.get("title", ""), re.I)]
    if words:
        opts.append(("role_avoided", {"roles_avoided": [words[0]]}))
    if card.get("industry") and not (set(cand.get("industries_wanted") or ()) and industry_ids(card["industry"])):
        opts.append(("industry_avoided", {"industries_avoided": [card["industry"]], "industries_wanted": []}))
    rng.shuffle(opts)
    for name, edit in opts:
        if name in used:
            continue
        alt = cand | edit
        try:
            CandidateCard.from_dict(alt)
        except ValueError:
            continue
        return alt, name
    return None


# ---------------------------------------------------------------- twin posting render

TWIN_BLUEPRINT_ADD = (
    "\nTwin wording keys (these override the general rules above where they speak):\n"
    "- required_skills entries of the form {\"kind\": K, \"such_as\": [A, B, C]}: one requirement that any K meets. "
    "Write it in one sentence of the requirements part as K followed by 'such as' and the names, each exactly as given "
    "(for example 'Experience with a programming language such as Python, Java or C++'). Never say all of them are "
    "needed and never list them as separate requirements.\n"
    "- duty_skills: tools or subjects the job uses that are not requirements. Name each one only inside a sentence "
    "about the work itself, with an action verb ('you will build weekly dashboards in Tableau', 'you will run assays "
    "on the PCR line'). Never put one in the requirements or preferred part, and never write must, required, "
    "experience with, knowledge of, familiarity, proficiency, ideally, preferred or a plus in a sentence that names it.\n"
    "- preferred_skills: write each one under preferred, bonus or nice-to-have wording only ('Bonus points for ...', "
    "'Preferred: ...', '... is a plus'), never in the requirements part and never as something applicants must have."
)
TWIN_POSTINGS = BUILD / "twin_postings.jsonl"
TWIN_QUARANTINE = BUILD / "twin_postings_quarantine.jsonl"
MAX_TRIES = 3
SUCH_AS = re.compile(r"\b(?:such as|e\.g\.|for example|for instance|including|like)\b", re.I)


def twin_kwargs() -> dict:
    import render_cards as rc
    return dict(
        column_mapping={"context": ["card"], "universal_prompt": rc.POSTING_UNIVERSAL},
        recipe_specification={"recipes": {"prompt_rephrase": False, "deduplication": False, "reasoning_traces": False}},
        brand_controls={"blueprint": rc.POSTING_BLUEPRINT + TWIN_BLUEPRINT_ADD, "length": "detailed",
                        "hallucination_mitigation": False},
        training_type="instruction_dataset",
    )


def twin_wording(tc: dict, text: str) -> list[dict]:
    """The wording gates of one twin text: {"tag", "quote"/"missing", "card_says"} per problem."""
    import render_cards as rc
    from matcher.visibility import has_phrase
    out = []
    sents = rc._sentences_with_heading(text)
    if tc["variant"] == "ex":
        for names, kind in tc["ex_groups"]:
            if not any(SUCH_AS.search(s) and all(has_phrase(s, n) for n in names) for s, _ in sents):
                out.append({"tag": "examples_not_such_as", "missing": f"one sentence with '{kind} such as "
                            f"{', '.join(names[:-1])} or {names[-1]}'",
                            "card_says": f"{kind} such as {', '.join(names[:-1])} or {names[-1]}, in one sentence of "
                                         "the requirements part, each name exactly as given"})
    if tc["variant"] == "pref":
        # a preferred skill: named in preferred wording at least once, and never as a requirement (a duty mention is
        # fine: real postings name preferred tools among the duties too)
        for x in tc["moved"]:
            hits = [(s, head) for s, head in sents if has_phrase(s, x)]
            pref = [s for s, head in hits if is_pref(s, head)]
            req = [s for s, head in hits if not is_pref(s, head) and (rc.REQ_MARK.search(s) or rc.REQ_LABEL.search(s)
                                                                     or rc.REQ_HEADING.search(head))]
            if not pref or req:
                out.append({"tag": "preferred_not_preferred", "quote": (req or [s for s, _ in hits] or [""])[0][:rc.QUOTE_MAX],
                            "named": x, "card_says": f"{x} is preferred only: name it under preferred, bonus or "
                                                     "nice-to-have wording, never as a requirement"})
    if tc["variant"] == "duty":
        req = set(rc._req_or_pref_sentences(text))
        fields = {f.casefold() for f in tc["display_card"].get("degree_fields") or ()}
        for x in tc["duty_skills"]:
            # a duty skill that is also a degree field the card asks for is named in the degree sentence too
            hits = [s for s, _ in sents if has_phrase(s, x) and not (x.casefold() in fields and DEGREE_SENT.search(s))]
            if not hits:
                out.append({"tag": "duty_missing", "missing": x, "card_says": f"duty_skills: name {x} in a sentence "
                            "about the work (an action verb), not as a requirement"})
            bad = [s for s in hits if s in req]
            if bad:
                out.append({"tag": "duty_stated_required", "quote": bad[0][:rc.QUOTE_MAX], "named": x,
                            "card_says": f"{x} is a duty only: name it in a sentence about the work with an action "
                                         "verb, never with must, required, experience with, knowledge of, a plus or "
                                         "preferred, and never in the requirements or preferred part"})
    return out


PREF_EXTRA = re.compile(r"\b(?:beneficial|not (?:a )?(?:strict |hard )?requirement|not required|additional strengths?|"
                        r"welcome|would be great|great to have|excited|valuable|advantage|bonus|a plus|helpful|nice|"
                        r"ideal(?:ly)?|appreciated|desir\w+|prefer\w*|extra credit|stand out|set you apart|valued|favorabl\w*)\b", re.I)
DEGREE_SENT = re.compile(r"\b(?:degree|major|field of study|fields? of study|program|bachelor|master|doctor)", re.I)


def is_pref(sent: str, head: str) -> bool:
    import render_cards as rc
    return bool(rc.PREF_MARK.search(sent) or PREF_EXTRA.search(sent) or rc.PREF_HEADING.search(head)
                or PREF_EXTRA.search(head))


def regate_twins() -> None:
    """Gate the quarantined twin texts again under the current gate code: a text a false positive kept out moves to
    the kept file (no credits). Its try count stays what it was."""
    tcs = {r["id"]: r for r in jsonl(TWIN_CARDS)}
    kept = {r["id"] for r in jsonl(TWIN_POSTINGS)}
    q = jsonl(TWIN_QUARANTINE)
    moved, stay = [], []
    for r in q:
        if r["id"] in kept or r["id"] in {m["id"] for m in moved}:
            stay.append(r)
            continue
        tags = twin_fatal(tcs[r["id"]], r["text"])
        (stay if tags else moved).append(r | {"tags": tags} if tags else r | {"tags": [], "regated": True})
    if moved:
        with TWIN_POSTINGS.open("a") as f:
            for r in moved:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_jsonl(TWIN_QUARANTINE, stay)
    print(f"regate twins: {len(moved)} quarantined texts pass now; {len(stay)} stay "
          f"({dict(Counter(t for r in stay for t in r['tags']).most_common(10))})")


def twin_fatal(tc: dict, text: str) -> list[str]:
    import render_cards as rc
    return list(dict.fromkeys(rc.fatal("postings", tc["display_card"], text) + [f["tag"] for f in twin_wording(tc, text)]))


def twin_retry_context(tc: dict, text: str) -> dict:
    import render_cards as rc
    fs = [f for f in rc.findings("postings", tc["display_card"], text)] + twin_wording(tc, text)
    lines = rc.feedback_lines(fs)
    return tc["context"] | ({rc.FEEDBACK_KEY: lines} if lines else {})


def render_twins(a: argparse.Namespace) -> None:
    import render_cards as rc
    tcs = {r["id"]: r for r in jsonl(TWIN_CARDS)}
    kept = {r["id"] for r in jsonl(TWIN_POSTINGS)}
    quar = jsonl(TWIN_QUARANTINE)
    tries = Counter(r["id"] for r in quar)
    last = {r["id"]: r for r in quar}
    todo = [i for i in tcs if i not in kept and tries[i] < MAX_TRIES]
    gave_up = [i for i in tcs if i not in kept and tries[i] >= MAX_TRIES]
    sent = {}
    for i in todo:
        ctx = twin_retry_context(tcs[i], last[i]["text"]) if i in last else tcs[i]["context"]
        sent[i] = json.dumps(ctx, ensure_ascii=False)
    print(f"twin postings: {len(tcs)} cards, {len(kept)} kept, {len(todo)} to render "
          f"({sum(1 for i in todo if i in last)} retries), {len(gave_up)} out of tries")
    if not todo:
        return
    c = client()
    order = list(todo)
    random.Random(SEED + len(kept)).shuffle(order)
    if a.limit:
        order = order[:a.limit]
    tag = "twinpost_" + h(*(sent[i] for i in order), json.dumps(twin_kwargs(), sort_keys=True))[:10]
    got = run_chunks(c, tag, ["row_id", "card"], [[i, sent[i]] for i in order], twin_kwargs(), a.mode, a.max_credits,
                     f"fit-{tag}")
    if got is None:
        return
    new, bad = [], []
    for r in got:
        i, text = r.get("row_id", ""), (r.get("enhanced_completion") or "").strip()
        if i not in sent or not text or i in kept:
            continue
        tags = twin_fatal(tcs[i], text)
        rec = {"id": i, "text": text, "try": tries[i] + 1, "tags": tags, "feedback": rc.FEEDBACK_KEY in sent[i]}
        (bad if tags else new).append(rec)
        kept |= set() if tags else {i}
    with TWIN_POSTINGS.open("a") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with TWIN_QUARANTINE.open("a") as f:
        for r in bad:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"twin postings: {len(new)} kept, {len(bad)} failed "
          f"({dict(Counter(t for r in bad for t in r['tags']).most_common(12))}); first-try pass "
          f"{sum(1 for r in new if r['try'] == 1)}/{sum(1 for r in new + bad if r['try'] == 1)}")


# ---------------------------------------------------------------- venator candidates

CANDS = BUILD / "candidates.jsonl"
# The kinds a venator row is drawn from, by the label it is built to have, with weights: resume-side kinds (skills,
# years, degree) first, the profile-side ones the eligibility layer also covers at lower weight.
KIND_W = {
    "For you": {"full_match": 2, "must_skill_alternative": 2, "nice_skill_one_missing": 2, "must_skill_synonym": 1,
                "soft_skill_missing": 1, "degree_field_related": 1, "degree_enrolled_higher": 0.5,
                "location_willing": 0.5, "location_same_metro": 0.3, "pref_remote_match": 0.3,
                "work_auth_cpt_internship": 0.5, "work_auth_h1b_sponsors": 0.3, "auth_f1_cpt": 0.3},
    "Explore": {"nice_skill": 2, "years_one_short": 1.5, "overqualified": 1, "degree_field_adjacent": 1,
                "degree_still_enrolled": 1, "salary": 1, "pref_industry_wanted": 0.5, "pref_industry_avoided": 0.5,
                "pref_remote_only": 0.5, "pref_hybrid_or_remote": 0.5, "pref_role_avoided": 0.5,
                "work_auth_f1_opt": 0.5, "work_auth_stem_opt": 0.5, "work_auth_h1b_unstated": 0.5,
                "work_auth_other_visa": 0.3, "auth_f1_opt": 0.3, "auth_needs_h1b_now": 0.3},
    "Excluded": {"must_skill": 2, "years_far_short": 1.5, "degree_level": 1, "degree_field": 1, "license": 1,
                 "location": 0.6, "employment_type": 0.6, "enrollment_graduated": 0.6, "enrollment_level": 0.6,
                 "work_auth_citizens_only": 0.5, "work_auth_needs_h1b": 0.5, "work_auth_no_opt": 0.4},
}
ROWS_PER_POSTING = {"train": 2, "dev": 2}


def posting_text_of(p: dict) -> str:
    """A venator posting as a prompt shows it: textclean.real_posting_text, the format of the real test and demo."""
    from matcher import textclean
    return textclean.real_posting_text({"title": p["title"], "company": p.get("employer", ""),
                                        "locations": [p["locations"]] if p.get("locations") else [],
                                        "description": p.get("text", "")}, key=p["key"])


def label_card_of(cr: dict) -> RequirementCard:
    """The card a venator row is labelled on: the extracted card as it is, except required groups the extractor
    flagged phrase-like (meta.must_phrases: 'Basic business office skills', 'Databases'), which no resume can show
    and the v2 blueprint itself says to leave out as vague; they are dropped, as soft skills are."""
    card = RequirementCard.from_dict(cr["card"])
    drop = {tuple(g) for g in cr["meta"].get("must_phrases") or []}
    if drop:
        card = replace(card, must_skills=tuple(g for g in card.must_skills if tuple(g) not in drop))
    return card


def usable(cr: dict) -> bool:
    """A US posting in English whose card does not list MUST_MANY or more required groups (over-extraction)."""
    m = cr["meta"]
    return bool(m.get("us")) and m.get("language") == "en" and not m.get("must_many")


def candidates(_: argparse.Namespace) -> None:
    import build_real_test as brt
    from matcher.rules import label
    rng = random.Random(SEED + 2)
    posts = {p["key"]: p for p in jsonl(SAMPLE)}
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    out = []
    for r in sorted(jsonl(EXTRACT_CARDS), key=lambda r: r["key"]):
        p = posts[r["key"]]
        card = label_card_of(r)
        if r["kind"] == "teaser" or card.thin_posting:
            continue  # teaser rows borrow a rendered resume at assemble time
        us = usable(r)
        if r["split"] == "train" and not us:
            continue
        text = posting_text_of(p)
        try:
            cs = brt.candidates_for(card, text, SEED)
        except Exception as e:  # noqa: BLE001  one odd card never stops the build
            print("candidates_for failed", r["key"], type(e).__name__, e)
            continue
        by_label: dict[str, list[dict]] = defaultdict(list)
        for c in cs:
            if c["problems"]:
                continue
            b = label(c["candidate"], card)[0]
            if c["expected"] is not None and b != c["expected"]:
                continue
            w = KIND_W.get(b, {}).get(c["kind"])
            if w:
                by_label[b].append(c | {"w": w, "bucket": b})
        n = ROWS_PER_POSTING[r["split"]] if us else 1
        cnt = counts[(r["split"], p["field"])]
        picked = []
        for _ in range(n):
            labs = sorted((b for b in by_label if by_label[b] and b not in {x["bucket"] for x in picked}),
                          key=lambda b: (cnt[b], rng.random()))
            if not labs:
                break
            pool = by_label[labs[0]]
            c = rng.choices(pool, weights=[x["w"] for x in pool])[0]
            picked.append(c)
            cnt[c["bucket"]] += 1
        for c in picked:
            out.append({"key": r["key"], "split": r["split"], "field": p["field"], "kind": c["kind"],
                        "cd_id": c["cd_id"], "built_as": c["bucket"], "candidate": c["candidate"].to_dict()})
    # dev gets one more row on each US posting, drawn with its own generator so the rows above stay as they were
    drng = random.Random(SEED + 22)
    have = defaultdict(set)
    for o in out:
        have[o["key"]].add(o["cd_id"])
    for r in sorted(jsonl(EXTRACT_CARDS), key=lambda r: r["key"]):
        card = label_card_of(r)
        if r["split"] != "dev" or r["kind"] == "teaser" or card.thin_posting or not usable(r):
            continue
        p = posts[r["key"]]
        try:
            cs = brt.candidates_for(card, posting_text_of(p), SEED)
        except Exception:  # noqa: BLE001
            continue
        pool = []
        for c in cs:
            if c["problems"] or c["cd_id"] in have[r["key"]]:
                continue
            b = label(c["candidate"], card)[0]
            if (c["expected"] is None or b == c["expected"]) and KIND_W.get(b, {}).get(c["kind"]):
                pool.append(c | {"bucket": b, "w": KIND_W[b][c["kind"]]})
        if pool:
            c = drng.choices(pool, weights=[x["w"] for x in pool])[0]
            out.append({"key": r["key"], "split": "dev", "field": p["field"], "kind": c["kind"], "cd_id": c["cd_id"],
                        "built_as": c["bucket"], "candidate": c["candidate"].to_dict(), "extra_dev": True})
    write_jsonl(CANDS, out)
    print(f"candidates: {len(out)} rows on {len({r['key'] for r in out})} postings; "
          f"{dict(Counter((r['split'], r['built_as']) for r in out))}; distinct candidates {len({r['cd_id'] for r in out})}")


# ---------------------------------------------------------------- rich resume render

RICH_ADD = (
    "\nA real one-page resume, richer than a skills list:\n"
    "- Projects: when the card has skills, add a Projects section with one to three projects, each a title and one or two "
    "bullets, using only the card's skills (no other tool, language or method).\n"
    "- Coursework: for a student, or a graduate of the last three years, add a 'Relevant coursework' line with three to "
    "five course titles in the degree field; a course title names no skill, tool or subject that is not in the card's "
    "skills or degree field.\n"
    "- Vary the format from resume to resume: reverse-chronological, combination (summary and skills first) or "
    "skills-forward; vary the date style (Jan 2023, 01/2023) and bullet length. Keep it to one page, 300 to 550 words."
)
FIT_RESUMES = BUILD / "resumes.jsonl"
FIT_RESUME_Q = BUILD / "resumes_quarantine.jsonl"


def resume_kwargs() -> dict:
    import render_cards as rc
    return dict(
        column_mapping={"context": ["card"], "universal_prompt": rc.RESUME_UNIVERSAL},
        recipe_specification={"recipes": {"prompt_rephrase": False, "deduplication": False, "reasoning_traces": False}},
        brand_controls={"blueprint": rc.RESUME_BLUEPRINT + RICH_ADD, "length": "detailed", "hallucination_mitigation": False},
        training_type="instruction_dataset",
    )


def resume_jobs() -> dict[str, dict]:
    """ctx hash -> {"card", "lacked"}: one rendered resume per distinct resume context among the venator rows, with
    every skill a posting it is paired with names and the candidate lacks (the resume must not name any)."""
    import render_cards as rc
    from matcher.rules import held_skills
    from matcher.schema import CandidateCard, group_met
    cards = {r["key"]: label_card_of(r) for r in jsonl(EXTRACT_CARDS)}
    jobs: dict[str, dict] = {}
    for r in jsonl(CANDS):
        cd = r["candidate"]
        hsh = rc.ctx_hash(rc.ctx_str("resumes", cd))
        card = cards[r["key"]]
        have = held_skills(CandidateCard.from_dict(cd))
        names = [n for g in card.must_skills for n in g] + list(card.nice_skills)
        lacked = {n for n in names if not group_met((n,), have)}
        j = jobs.setdefault(hsh, {"card": cd, "lacked": set(), "ids": set()})
        j["lacked"] |= lacked
        j["ids"].add(r["cd_id"])
    return jobs


def resume_fatal(card: dict, lacked: set[str], text: str) -> tuple[list[str], list[dict]]:
    """render_cards' resume gates, with one swap: lacked_skill_named reads the whole synthetic training vocabulary
    ('data analysis', 'research', 'customer service': a richer resume's projects name such things on 70% of the pilot
    texts), while a venator resume's labels read only its own posting's skills; names_lacked_skill checks exactly
    those (every required or preferred skill of the paired posting that the candidate lacks)."""
    import render_cards as rc
    from matcher import textclean
    tags = [t for t in rc.fatal("resumes", card, text) if t != "lacked_skill_named"]
    extra = rc.pair_lacked_named(card, textclean.redact(textclean.strip_markdown(text)), lacked)
    fs = []
    if extra:
        tags = tags + ["names_lacked_skill"]
        fs.append({"tag": "names_lacked_skill", "named": ", ".join(extra[:6]),
                   "card_says": f"skills lists only the card's skills; never name {', '.join(extra[:6])} anywhere, not "
                                "even in projects, coursework or bullet points"})
    return tags, fs


def regate_resumes() -> None:
    jobs = resume_jobs()
    kept = {r["ctx"] for r in jsonl(FIT_RESUMES)}
    moved, stay = [], []
    for r in jsonl(FIT_RESUME_Q):
        if r["ctx"] in kept or r["ctx"] not in jobs:
            stay.append(r)
            continue
        tags, _ = resume_fatal(jobs[r["ctx"]]["card"], jobs[r["ctx"]]["lacked"], r["text"])
        if tags:
            stay.append(r | {"tags": tags})
        else:
            moved.append(r | {"tags": [], "regated": True})
            kept.add(r["ctx"])
    with FIT_RESUMES.open("a") as f:
        for r in moved:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_jsonl(FIT_RESUME_Q, stay)
    print(f"regate resumes: {len(moved)} quarantined texts pass now; {len(stay)} stay "
          f"({dict(Counter(t for r in stay for t in r['tags']).most_common(10))})")


def render_resumes(a: argparse.Namespace) -> None:
    import render_cards as rc
    jobs = resume_jobs()
    kept = {r["ctx"] for r in jsonl(FIT_RESUMES)}
    quar = jsonl(FIT_RESUME_Q)
    tries = Counter(r["ctx"] for r in quar)
    last = {r["ctx"]: r for r in quar}
    todo = [k for k in jobs if k not in kept and tries[k] < MAX_TRIES]
    sent = {}
    for k in todo:
        if k in last:
            base = json.loads(rc.retry_context("resumes", jobs[k]["card"], last[k]["text"], last[k]["tags"]))
            _, mine = resume_fatal(jobs[k]["card"], jobs[k]["lacked"], last[k]["text"])
            lines = list(dict.fromkeys([*base.get(rc.FEEDBACK_KEY, []), *rc.feedback_lines(mine)]))
            ctx = rc.context_of("resumes", jobs[k]["card"]) | ({rc.FEEDBACK_KEY: lines} if lines else {})
            sent[k] = json.dumps(ctx, ensure_ascii=False)
        else:
            sent[k] = rc.ctx_str("resumes", jobs[k]["card"])
    print(f"resumes: {len(jobs)} contexts, {len(kept)} kept, {len(todo)} to render ({sum(1 for k in todo if k in last)} "
          f"retries), {sum(1 for k in jobs if k not in kept and tries[k] >= MAX_TRIES)} out of tries")
    if not todo:
        return
    order = list(todo)
    random.Random(SEED + len(kept)).shuffle(order)
    if a.limit:
        order = order[:a.limit]
    c = client()
    tag = "fitres_" + h(*(sent[k] for k in order), json.dumps(resume_kwargs(), sort_keys=True))[:10]
    got = run_chunks(c, tag, ["row_id", "card"], [[k, sent[k]] for k in order], resume_kwargs(), a.mode,
                     a.max_credits, f"fit-{tag}")
    if got is None:
        return
    new, bad = [], []
    for r in got:
        k, text = r.get("row_id", ""), (r.get("enhanced_completion") or "").strip()
        if k not in sent or not text or k in kept:
            continue
        tags, _ = resume_fatal(jobs[k]["card"], jobs[k]["lacked"], text)
        rec = {"ctx": k, "text": text, "try": tries[k] + 1, "tags": tags}
        (bad if tags else new).append(rec)
        if not tags:
            kept.add(k)
    with FIT_RESUMES.open("a") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with FIT_RESUME_Q.open("a") as f:
        for r in bad:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"resumes: {len(new)} kept, {len(bad)} failed ({dict(Counter(t for r in bad for t in r['tags']).most_common(12))}); "
          f"first-try pass {sum(1 for r in new if r['try'] == 1)}/{sum(1 for r in new + bad if r['try'] == 1)}")


# ---------------------------------------------------------------- reasoning traces

CHECK_NAMES = {"work_auth": "work authorization", "employment_type": "job type", "license": "license",
               "enrollment": "enrollment", "degree": "degree", "skills": "required skills", "years": "experience",
               "location": "location", "preferences": "preferences", "thin": "posting detail"}
ELIGIBILITY = ("work_auth", "employment_type", "license", "enrollment", "degree", "location")


def _clip(xs: list[str], n: int) -> str:
    xs = list(dict.fromkeys(xs))
    return ", ".join(xs[:n]) + (f" and {len(xs) - n} more" if len(xs) > n else "")


def reasoning_trace(cand, card, hedged: dict[str, list[str]] | None = None) -> str:
    """Plain-language steps from the cards and the rule, deterministic: each eligibility check that has something to
    check (passes, is a stretch, or fails, with the rule's own reason), the required skills met and missing, the
    skills the posting names that are only examples, preferred or duties (they never block), then the verdict. It runs
    the rule's own checks (rules.CHECKS), so it agrees with label() by construction."""
    from matcher.rules import CHECKS, LABEL_WORDS, _Verdict, label, required_groups
    from matcher.rules import held_skills
    from matcher.schema import group_met
    bucket, reasons, _ = label(cand, card)
    elig = []  # (check name, state, the rule's reason)
    for name, check in CHECKS:
        v = _Verdict()
        check(cand, card, v)
        if name == "skills":
            continue
        if v.hard:
            elig.append((name, "fails", v.hard[0]))
        elif v.soft:
            elig.append((name, "is a stretch", v.soft[0]))
        elif v.good and name in ELIGIBILITY + ("years",):
            elig.append((name, "ok", ""))
    have = held_skills(cand)
    groups = required_groups(card)
    met = [" or ".join(g) if len(g) < 4 else f"one of {g[0]}, {g[1]}, ..." for g in groups if group_met(g, have)]
    miss = [" or ".join(g) for g in groups if not group_met(g, have)]

    def render(level: int) -> str:
        # level 0 says everything; each level up says less: passing checks go, then the stretches' reasons, then the
        # fails' reasons, then the lists shrink. The verdict always keeps the rule's first reason.
        shown = [(n, st, why) for n, st, why in elig if not (level >= 1 and st == "ok")]
        cs = [f"{CHECK_NAMES[n]} {st}" + (f" ({why})" if why and not (level >= 2 and st == "is a stretch")
                                          and not (level >= 3 and st == "fails") else "") for n, st, why in shown]
        k = 2 if level >= 4 else 3
        parts = ["Checks: " + ("; ".join(cs) if cs else "nothing to check beyond skills") + "."]
        if groups and level >= 5:  # counts, and the missing ones by name
            parts.append(f"Required skills met: {len(met)} of {len(groups)}" + (f"; missing: {_clip(miss, 2)}." if miss else "."))
        elif groups:
            parts.append(f"Required skills met: {_clip(met, k + 1) or 'none'}" + (f"; missing: {_clip(miss, k)}." if miss else "."))
        else:
            parts.append("The posting requires no specific skill.")
        soft = []
        for kind, what in (("example", "examples of one ask, any one meets it"), ("preferred", "preferred, never a must"),
                           ("duty", "named as duties, not asked for")):
            names = (hedged or {}).get(kind, [])
            if names and level >= 6 and len(names) > 1:
                soft.append(f"{len(names)} skills are {what}")
            elif names:
                soft.append(f"{_clip(names, k)} are {what}" if len(names) > 1 else f"{names[0]} is {what.replace('examples of', 'an example of')}")
        if soft:
            parts.append("Hedged wording: " + "; ".join(soft) + ".")
        parts.append(f"Verdict: {LABEL_WORDS[bucket]}, {reasons[0]}.")
        return " ".join(" ".join(parts).split())

    for level in range(7):
        out = render(level)
        if trace_tokens(out) <= TRACE_TOKENS:
            break
    return out


TRACE_TOKENS = 90  # words and punctuation marks, as trace_tokens counts them; Qwen3.5's tokenizer gives at most ~1.3x that


def trace_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


# ---------------------------------------------------------------- assemble

TRAIN_TARGET = 11_400       # rows, balanced: 3,800 a label
CHECK_FIELDS = {"work_auth": ["sponsorship", "citizens_only", "e_verify"], "employment_type": ["employment_type"],
                "license": ["license"], "enrollment": ["enrolled_in"], "degree": ["degree_level", "degree_fields"],
                "skills": ["must_skills", "nice_skills"], "years": ["min_years", "seniority"],
                "location": ["locations", "work_mode"], "preferences": ["salary_max", "industry", "title"],
                "thin": ["thin_posting"]}
# fields the extraction audit found reliable (data/built/audit, the v2 re-read): every label field but must_skills
# (about 3 in 4 cards right), nice_skills (never audited), degree_fields and industry
RELIABLE = {"sponsorship", "citizens_only", "e_verify", "employment_type", "license", "enrolled_in", "degree_level",
            "min_years", "seniority", "locations", "work_mode", "salary_max", "title", "thin_posting"}


def label_fields(cand, card, reasons: list[str]) -> list[str]:
    from matcher.rules import reasons_by_check
    by = reasons_by_check(cand, card)
    return sorted({f for r in reasons for f in CHECK_FIELDS.get(by.get(r, ""), [])})


def eligibility_pass(cand, card) -> bool:
    import hybrid_rules as hr
    v = hr.rule_verdict(cand, card)
    return not (v["rule"] == "Skip" and set(v["hard_cats"]) & hr.HP_CATS)


def mix3_rows() -> list[dict]:
    """train_mix3.csv (d3's data) row by row with its cards: the 6,891 synthetic rows by prompt from training/train.jsonl,
    the real rows by completion and profile from mix3/rows.jsonl (cards: clean_checked/cards.jsonl). The 9 Explore
    rows on Ambarella's master's-only posting (ats_fc36587f: 'M.S. degree or M.S. Degree (in progress)'; the card
    said bachelor, so bachelor candidates got Explore where they should get Skip) are left out."""
    import render_cards as rc
    from matcher.profile import render_profile
    from matcher.rules import counted_nice
    from matcher.schema import CandidateCard
    csv.field_size_limit(10**9)
    rows = list(csv.DictReader((ROOT / "data/built/real_train/train_mix3.csv").open(newline="")))
    synth = {json.loads(l)["prompt"]: json.loads(l) for l in (ROOT / "data/built/training/train.jsonl").open()}
    reqs, cands = rc.load_cards("postings"), rc.load_cards("resumes")
    real = [r for r in jsonl(ROOT / "data/built/real_train/mix3/rows.jsonl") if r["in_csv"] in (True, "True")]
    real_cards = {r["id"]: r["card"] for r in jsonl(ROOT / "data/built/real_train/clean_checked/cards.jsonl")}
    out, unmatched, bad, ambiguous = [], 0, 0, 0
    used = set()
    for r in rows:
        s = synth.get(r["prompt"])
        if s:
            cand, card = CandidateCard.from_dict(cands[s["cd_id"]]), RequirementCard.from_dict(reqs[s["rq_id"]])
            out.append({"source": "mix3", "sub": "synthetic", "prompt": r["prompt"], "completion": r["completion"],
                        "cand": cand, "card": card, "posting_key": s["rq_id"].split("~")[0],
                        "hedged": {"preferred": list(counted_nice(card))}})
            continue
        hits = [i for i, x in enumerate(real) if i not in used and x["completion"] == r["completion"]
                and render_profile(CandidateCard.from_dict(_d(x["candidate"]))) in r["prompt"]]
        if len(hits) > 1:  # the same form on two postings: the one whose card title the prompt shows
            hits = [i for i in hits if real_cards[real[i]["rq_id"]]["title"] in r["prompt"]] or hits
        if not hits:
            unmatched += 1
            continue
        # one form and one label on two postings of the same title: either card gives the same label (the trace may
        # name the other posting's skills; counted)
        ambiguous += len({real[i]["rq_id"] for i in hits}) > 1
        x = real[hits[0]]
        used.add(hits[0])
        if x["base_rq_id"] == "ats_fc36587f-acf4-428e-a61f-aab42b07540c" and r["completion"].startswith("Explore"):
            bad += 1
            continue
        cand, card = CandidateCard.from_dict(_d(x["candidate"])), RequirementCard.from_dict(real_cards[x["rq_id"]])
        out.append({"source": "mix3", "sub": "real", "prompt": r["prompt"], "completion": r["completion"],
                    "cand": cand, "card": card, "posting_key": x["base_rq_id"], "hedged": {"preferred": list(counted_nice(card))}})
    if unmatched or bad != 9 or len(out) != len(rows) - 9:
        sys.exit(f"mix3: {unmatched} rows unmatched, {bad} known-bad rows found (want 9), {len(out)} of {len(rows)}")
    print(f"mix3: {len(out)} rows ({len(rows)} less the 9 known-bad Explore rows); {ambiguous} real rows matched one of "
          "two same-title postings")
    return out


def _d(x):
    return x if isinstance(x, dict) else __import__("ast").literal_eval(x)


def twin_rows() -> list[dict]:
    import build_real_test as brt
    import render_cards as rc
    from matcher import textclean
    from matcher.pairs import build_prompt, prompt_parts
    from matcher.profile import render_profile
    from matcher.rules import counted_nice, format_completion, label
    from matcher.schema import CandidateCard
    from matcher.visibility import has_phrase
    tcs = {r["id"]: r for r in jsonl(TWIN_CARDS)}
    texts = {r["id"]: r["text"] for r in jsonl(TWIN_POSTINGS)}
    draws = jsonl(TWIN_ROWS)
    resumes = rc.rendered_texts("resumes", {d["cd_id"]: rc.load_cards("resumes")[d["cd_id"]] for d in draws})
    bank = textclean.training_bank(ROOT)
    out, drop = [], Counter()
    for d in draws:
        for var in d["variants"]:
            vid = f"tw_{d['base_rq']}~{var}"
            if vid not in texts or d["cd_id"] not in resumes:
                drop["no_text"] += 1
                continue
            tc = tcs[vid]
            card = RequirementCard.from_dict(tc["label_card"])
            cand = CandidateCard.from_dict(d["card"])
            posting = textclean.training_posting(tc["display_card"], texts[vid], vid, bank)
            prompt = build_prompt(resumes[d["cd_id"]], render_profile(cand), posting)
            rs, pf, shown = prompt_parts(prompt)
            if rc.posting_missing_facts(tc["display_card"], shown):
                drop["fact_lost_in_fit"] += 1
                continue
            if any(not has_phrase(shown, x) for x in tc["moved"]):
                drop["moved_skill_lost_in_fit"] += 1
                continue
            b, reasons, _ = label(cand, card)
            if brt.person_problems(cand):
                drop["person_problems"] += 1
                continue
            if brt.readable_misses(cand, card, reasons, rs, pf, shown, hard=b == "Excluded"):
                drop["not_decidable"] += 1
                continue
            hedged = {"example": [n for names, _ in tc["ex_groups"] for n in names] if var == "ex" else [],
                      "preferred": list(counted_nice(card)), "duty": list(tc["duty_skills"]) if var == "duty" else []}
            out.append({"source": "twins", "sub": var, "split": d["split"], "prompt": prompt,
                        "completion": format_completion(b, reasons), "cand": cand, "card": card,
                        "posting_key": d["base_rq"], "hedged": hedged, "posting_text": posting,
                        "resume_text": resumes[d["cd_id"]], "draw": d["draw"], "cd_id": d["cd_id"], "variant_id": vid})
    print(f"twins: {len(out)} rows; dropped {dict(drop)}")
    return out


def venator_rows() -> list[dict]:
    """The venator rows and the teaser rows, gated as mix3's real rows were (clean_real_checked.gate: person checks,
    build_real_test.guard for the kinds the reasons name, assemble_real_train's text screens, visibility)."""
    import clean_real_checked as ccr
    import render_cards as rc
    from matcher.rules import counted_nice
    from matcher.pairs import build_prompt
    from matcher.profile import render_profile
    from matcher.schema import CandidateCard
    posts = {p["key"]: p for p in jsonl(SAMPLE)}
    cards = {r["key"]: r for r in jsonl(EXTRACT_CARDS)}
    res = {r["ctx"]: r["text"] for r in jsonl(FIT_RESUMES)}
    out, drop = [], Counter()

    def gate_row(key, cand, src, kind, cd_id, rtext=None):
        p, cr = posts[key], cards[key]
        card = label_card_of(cr)
        rtext = rtext or res.get(rc.ctx_hash(rc.ctx_str("resumes", cand.to_dict())))
        if rtext is None:
            drop["no_resume"] += 1
            return
        posting = posting_text_of(p)
        # the form's blank fields filled to values the posting meets (neutral_fill), when the filled row passes the
        # gates; else the row as drawn
        filled = CandidateCard.from_dict(neutral_fill(cand.to_dict(), card.to_dict(),
                                                      random.Random(f"{SEED}:{key}:{cd_id}:{kind}:fill")))
        for c in dict.fromkeys([filled, cand]):
            prompt = build_prompt(rtext, render_profile(c), posting)
            # the location screen reads the posting's place list: the card's "City, ST" places, since venator's own
            # location strings ("Boston-MA", "2 Locations", "Worcester, MA University Campus") do not parse as places
            g = ccr.gate(c, card, as_ats(p) | {"locations": list(card.locations)}, prompt, p["split"])
            if g["stage"] == "kept" and g.get("decidable", True):
                break
        if g["stage"] != "kept" or not g.get("decidable", True):
            drop[f"{src}:{g['stage'] if g['stage'] != 'kept' else 'undecidable'}"] += 1
            return
        drop["filled"] += c is filled and filled != cand
        cand = c
        out.append({"source": src, "sub": kind, "split": p["split"], "prompt": prompt, "completion": g["completion"],
                    "cand": cand, "card": card, "posting_key": key, "employer": p.get("employer", ""),
                    "field": p["field"], "hedged": {"preferred": list(counted_nice(card))},
                    "posting_text": posting, "resume_text": rtext, "cd_id": cd_id, "meta_us": usable(cr),
                    "phrase_groups_dropped": cr["meta"].get("must_phrases") or []})

    vc = jsonl(CANDS)
    for r in vc:
        gate_row(r["key"], CandidateCard.from_dict(r["candidate"]), "venator", r["kind"], r["cd_id"])
    # teasers: a rendered resume of another posting's candidate (same split, same field when one exists), its form
    # set so the posting's eligibility passes, or, for one row in three, fails on job type or location
    rng = random.Random(SEED + 3)
    donors: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in vc:
        if rc.ctx_hash(rc.ctx_str("resumes", r["candidate"])) in res:
            donors[(r["split"], r["field"])].append(r)
            donors[(r["split"], "*")].append(r)
    for key, cr in sorted(cards.items()):
        if cr["kind"] != "teaser" and not cr["card"].get("thin_posting"):
            continue
        p = posts[key]
        if p["split"] == "train" and not usable(cr):
            continue
        card = label_card_of(cr)
        pool = donors.get((p["split"], p["field"])) or donors[(p["split"], "*")]
        n = 3 if p["split"] == "train" else 2
        for j, dn in enumerate(rng.sample(pool, min(n, len(pool)))):
            c = CandidateCard.from_dict(dn["candidate"])
            places = list(card.locations)
            c = replace(c, willing_locations=tuple(dict.fromkeys([*c.willing_locations, *places]))[:4],
                        employment_types_wanted=(), remote_preference="any", industries_avoided=(), roles_avoided=(),
                        salary_floor=None)
            if j == n - 1 and p["split"] == "train":  # the ineligible one
                if places and card.work_mode != "remote" and rng.random() < 0.5:
                    c = replace(c, willing_locations=())
                else:
                    c = replace(c, employment_types_wanted=(rng.choice([t for t in ("full_time", "part_time", "internship")
                                                                         if t != card.employment_type]),))
            gate_row(key, c, "teaser", "teaser_fail" if j == n - 1 and p["split"] == "train" else "teaser", dn["cd_id"],
                     res[rc.ctx_hash(rc.ctx_str("resumes", dn["candidate"]))])
    print(f"venator and teasers: {len(out)} rows {dict(Counter((r['source'], r['split'], r['completion'].split()[0]) for r in out))}; "
          f"dropped {dict(drop.most_common(20))}")
    return out


def balance_rows(rows: list[dict], target: int, rng: random.Random) -> list[dict]:
    """Equal Keep / Explore / Skip, `target` rows in all. Every twin, venator and teaser row is kept while a label has
    room; mix3 rows fill each label to its share (mix3 is d3's data, so most of it stays)."""
    have = Counter(r["completion"].split()[0] for r in rows)
    per = min(target // 3, *(have[w] for w in ("Keep", "Explore", "Skip")))
    out = []
    for word in ("Keep", "Explore", "Skip"):
        lab = [r for r in rows if r["completion"].startswith(word)]
        new = [r for r in lab if r["source"] != "mix3"]
        old = [r for r in lab if r["source"] == "mix3"]
        rng.shuffle(new)
        rng.shuffle(old)
        take = new[:per]
        out += take + old[:max(0, per - len(take))]
    rng.shuffle(out)
    return out


DEBIAS_TO = 0.37  # profile-only accuracy (grouped 5-fold, balanced) the train set is filtered down to
DEBIAS_ROUNDS = 12


def _profile_oof(rows: list[dict]) -> tuple[float, list[float]]:
    """Grouped 5-fold (by posting) out-of-fold accuracy of a profile-only TF-IDF + logistic guesser, and each row's
    out-of-fold probability of its own label."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    x = [_parts(r["prompt"])[1] for r in rows]
    y = np.array([r["completion"].split()[0] for r in rows])
    g = [r["posting_key"] for r in rows]
    p_gold = np.zeros(len(rows))
    hit = np.zeros(len(rows), dtype=bool)
    for tr, te in GroupKFold(n_splits=5).split(x, y, g):
        v = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True, max_features=50000)
        xt = v.fit_transform([x[i] for i in tr])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(xt, y[tr])
        pr = clf.predict_proba(v.transform([x[i] for i in te]))
        cls = list(clf.classes_)
        p_gold[te] = pr[np.arange(len(te)), [cls.index(c) for c in y[te]]]
        hit[te] = np.array(cls)[pr.argmax(1)] == y[te]
    return float(hit.mean()), p_gold.tolist()


def debias(pool: list[dict], target: int, seed: int) -> tuple[list[dict], dict]:
    """AFLite on the profile: balance, score every row out of fold with a profile-only guesser, and drop the new rows
    (twins, venator, teasers) it finds easiest, an equal number per label, until it does no better than DEBIAS_TO.
    mix3 rows are d3's data and are never dropped; a dropped row's place goes back to the label's other rows."""
    banned: set[int] = set()
    log = []
    for rnd in range(DEBIAS_ROUNDS + 1):
        train = balance_rows([r for r in pool if r["_i"] not in banned], target, random.Random(seed))
        acc, pg = _profile_oof(train)
        log.append({"round": rnd, "rows": len(train), "profile_cv_accuracy": round(acc, 4), "dropped": len(banned)})
        print(f"debias round {rnd}: {len(train)} rows, profile-only cv {acc:.4f}, {len(banned)} dropped")
        if acc <= DEBIAS_TO or rnd == DEBIAS_ROUNDS:
            break
        step = max(30, int(len(train) * 0.02))
        for word in ("Keep", "Explore", "Skip"):
            easy = sorted((-p, r["_i"]) for r, p in zip(train, pg)
                          if r["source"] != "mix3" and r["completion"].startswith(word) and p > 0.5)
            banned |= {i for _, i in easy[:step]}
    dropped = [r for r in pool if r["_i"] in banned]
    return train, {"rounds": log, "dropped_by_source": dict(Counter(r["source"] for r in dropped)),
                   "dropped_by_source_label": dict(Counter(f"{r['source']}:{r['completion'].split()[0]}" for r in dropped))}


DEV_GUESS_MAX = 0.39  # the most a shallow guesser trained on train may score on the dev set
DEV_MIN_PER_LABEL = 170


def dev_filter(train: list[dict], dev: list[dict]) -> tuple[list[dict], dict]:
    """The dev set at equal labels, filtered as train was: profile-only, posting-only and resume-only guessers
    (TF-IDF + logistic, as the checks fit them) are trained on train and score every dev row; the rows they find
    easiest go first, from the labels with rows to spare and then an equal number from each, until none of them
    beats DEV_GUESS_MAX. So a model that reads one part of the prompt and guesses is not what the dev set selects."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    y = np.array([r["completion"].split()[0] for r in dev])
    hits, pg = [], []
    for part in (0, 1, 2):  # resume, profile, posting
        v = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True, max_features=50000)
        xt = v.fit_transform([_parts(r["prompt"])[part] for r in train])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(xt, [r["completion"].split()[0] for r in train])
        pr = clf.predict_proba(v.transform([_parts(r["prompt"])[part] for r in dev]))
        cls = list(clf.classes_)
        hits.append(np.array(cls)[pr.argmax(1)] == y)
        pg.append(pr[np.arange(len(dev)), [cls.index(c) for c in y]])
    hits, pg = np.array(hits), np.array(pg)
    keep = np.ones(len(dev), dtype=bool)
    words = ("Keep", "Explore", "Skip")

    def accs():
        return [round(float(h[keep].mean()), 4) for h in hits]

    def drop_easiest(word, k):
        # easy for a guesser that is over the line: its probability of the gold label, where it guesses the gold label
        over = [g for g, a in enumerate(accs()) if a > DEV_GUESS_MAX] or [0, 1, 2]
        easy = (pg[over] * hits[over]).max(0)
        idx = [i for i in np.argsort(-easy, kind="stable") if keep[i] and y[i] == word][:max(0, k)]
        keep[idx] = False

    start = {"rows": len(dev), "accuracy_resume_profile_posting": accs()}
    n = min(int((y == w).sum()) for w in words)
    for w in words:
        drop_easiest(w, int((y[keep] == w).sum()) - n)
    balanced = accs()
    while max(accs()) > DEV_GUESS_MAX and n > DEV_MIN_PER_LABEL:
        for w in words:
            drop_easiest(w, 2)
        n -= 2
    return [r for r, k in zip(dev, keep) if k], {
        "start": start, "balanced_before_filter": balanced, "rows": int(keep.sum()),
        "accuracy_resume_profile_posting": accs(),
        "kept_by_source": dict(Counter(r["source"] for r, k in zip(dev, keep) if k)),
        "dropped_by_source_label": dict(Counter(f"{r['source']}:{w}" for r, k, w in zip(dev, keep, y) if not k))}


WRONG = {"Keep": ("Explore", "Skip"), "Explore": ("Keep", "Skip"), "Skip": ("Keep", "Explore")}


def wrong_completions(r: dict, rng: random.Random) -> list[str]:
    """Two wrong-label completions with reasons that read plausibly: the mistakes the fit layer is there to stop (a
    named-but-not-required skill called missing, a stretch called fine, a hard fail called a stretch)."""
    from matcher.rules import ALL_SKILLS_MET, LABEL_WORDS, NO_REQUIREMENTS, OTHERWISE_OK, THIN_REASON, label, required_groups
    cand, card = r["cand"], r["card"]
    bucket, reasons, _ = label(cand, card)
    gold = LABEL_WORDS[bucket]
    named = [*r["hedged"].get("example", []), *r["hedged"].get("preferred", []), *r["hedged"].get("duty", [])]
    req = [g[0] for g in required_groups(card)]
    out = []
    for w in WRONG[gold]:
        if w == "Skip":
            if gold == "Explore" and THIN_REASON in reasons:
                why = "the posting names no skill you have"
            elif named:
                why = f"missing required skill: {rng.choice(named)}"
            elif req:
                why = f"missing required skill: {rng.choice(req)}"
            elif reasons and reasons[0] not in (OTHERWISE_OK, NO_REQUIREMENTS, ALL_SKILLS_MET):
                why = reasons[0].replace("you may be ", "you are ")
            else:
                why = "you do not meet the posting's requirements"
            out.append(f"Skip — {why}.")
        elif w == "Keep":
            if THIN_REASON in reasons:
                why = NO_REQUIREMENTS
            elif bucket == "Excluded" and named:
                why = f"{ALL_SKILLS_MET}; {rng.choice(named)} counts as an example"
            else:
                why = ALL_SKILLS_MET if req else "meets the stated requirements"
            out.append(f"Keep — {why}.")
        else:  # Explore
            if bucket == "Excluded":
                why = f"{reasons[0]}, but that may be flexible; {OTHERWISE_OK}"
            elif named:
                why = f"missing nice-to-have skill: {', '.join(named[:2])}; {OTHERWISE_OK}"
            else:
                why = f"the posting does not say much about the role; {OTHERWISE_OK}"
            out.append(f"Explore — {why}.")
    return out


def assemble(_: argparse.Namespace) -> None:
    from matcher.profile import render_profile
    from matcher.rules import LABEL_WORDS, format_completion, label
    rng = random.Random(SEED + 4)
    mix = mix3_rows()
    tw = twin_rows()
    vr = venator_rows()
    pool = [*mix, *[r for r in tw if r["split"] == "train"], *[r for r in vr if r["split"] == "train"]]
    for r in pool:  # every completion is the rule's own on the row's cards
        b, reasons, _ = label(r["cand"], r["card"])
        if r["source"] != "mix3" and format_completion(b, reasons) != r["completion"]:
            sys.exit(f"completion is not the rule's: {r['completion']} vs {format_completion(b, reasons)}")
    for i, r in enumerate(pool):
        r["_i"] = i
    train, deb = debias(pool, TRAIN_TARGET, SEED + 5)
    TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "completion", "reasoning_trace"])
        for r in train:
            r["trace"] = reasoning_trace(r["cand"], r["card"], r["hedged"])
            assert r["trace"].split("Verdict: ", 1)[1].split(",")[0] == r["completion"].split()[0], r["trace"]
            w.writerow([r["prompt"], r["completion"], r["trace"]])
    with DPO_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "chosen", "rejected"])
        n_dpo = 0
        for r in train:
            for bad in wrong_completions(r, rng):
                assert bad.split()[0] != r["completion"].split()[0]
                w.writerow([r["prompt"], r["completion"], bad])
                n_dpo += 1
    # dev: real dev-pool postings with extracted cards, dev teasers, twins of held-out synthetic cards
    dev_all = [r for r in [*vr, *tw] if r["split"] == "dev"]
    dev, dsel = dev_filter(train, dev_all)
    in_dev = {id(r) for r in dev}
    for path, rows in ((BUILD / "dev_fit_all.jsonl", dev_all), (DEV_JSONL, dev)):
        with path.open("w") as f:
            for r in rows:
                b, reasons, _ = label(r["cand"], r["card"])
                f.write(json.dumps({
                    "id": "fd_" + h(r["posting_key"], r.get("variant_id", ""), json.dumps(r["cand"].to_dict(), sort_keys=True))[:12],
                    "source": r["source"], "sub": r["sub"], "posting_key": r["posting_key"], "employer": r.get("employer", ""),
                    "posting": r["posting_text"], "resume": r["resume_text"], "profile": r["cand"].to_dict(),
                    "profile_text": render_profile(r["cand"]), "prompt": r["prompt"], "gold": LABEL_WORDS[b],
                    "gold_completion": r["completion"], "card": r["card"].to_dict(), "eligibility_pass": eligibility_pass(r["cand"], r["card"]),
                    "label_fields": label_fields(r["cand"], r["card"], reasons), "hedged": r["hedged"],
                    "reasoning_trace": reasoning_trace(r["cand"], r["card"], r["hedged"]),
                    "in_dev": id(r) in in_dev}, ensure_ascii=False) + "\n")
    rep = {
        "train_rows": len(train), "label_counts": dict(Counter(r["completion"].split()[0] for r in train)),
        "rows_by_source": dict(Counter(r["source"] for r in train)),
        "rows_by_source_label": {s: dict(Counter(r["completion"].split()[0] for r in train if r["source"] == s))
                                 for s in ("mix3", "twins", "venator", "teaser")},
        "twins_by_variant": dict(Counter(f"{r['sub']}:{r['completion'].split()[0]}" for r in train if r["source"] == "twins")),
        "twin_hedged_keep_explore": sum(1 for r in train if r["source"] == "twins" and r["sub"] != "req"
                                        and not r["completion"].startswith("Skip")),
        "available": {"mix3": len(mix), "twins": sum(r["split"] == "train" for r in tw),
                      "venator": sum(r["split"] == "train" and r["source"] == "venator" for r in vr),
                      "teaser": sum(r["split"] == "train" and r["source"] == "teaser" for r in vr)},
        "venator_label_fields": dict(Counter(f for r in train if r["source"] in ("venator", "teaser")
                                             for f in label_fields(r["cand"], r["card"], label(r["cand"], r["card"])[1]))),
        "venator_rows_on_reliable_fields": sum(1 for r in train if r["source"] in ("venator", "teaser")
                                               and set(label_fields(r["cand"], r["card"], label(r["cand"], r["card"])[1])) <= RELIABLE),
        "dev_rows": len(dev), "dev_filter": dsel, "dev_label_counts": dict(Counter(json.loads(l)["gold"] for l in DEV_JSONL.open())),
        "dev_by_source": dict(Counter(r["source"] for r in dev)),
        "dev_eligibility_pass_frac": round(sum(json.loads(l)["eligibility_pass"] for l in DEV_JSONL.open()) / max(1, len(dev)), 4),
        "dpo_pairs": n_dpo,
        "trace_words_max": max(len(r["trace"].split()) for r in train),
        "debias": deb,
    }
    (BUILD / "report.json").write_text(json.dumps(rep, indent=1))
    # the rows the checks read (no cards: text and label only)
    write_jsonl(BUILD / "leak_rows.jsonl", [
        {"id": f"t{i}", "split": "train", "source": r["source"], "employer": r.get("employer", ""),
         "posting_key": r["posting_key"], **dict(zip(("resume", "profile", "posting"), _parts(r["prompt"]))),
         "label": r["completion"].split()[0]} for i, r in enumerate(train)] + [
        {"id": f"d{i}", "split": "dev", "source": r["source"], "employer": r.get("employer", ""),
         "posting_key": r["posting_key"], **dict(zip(("resume", "profile", "posting"), _parts(r["prompt"]))),
         "label": r["completion"].split()[0]} for i, r in enumerate(dev)])
    print(json.dumps(rep, indent=1))


def _parts(prompt: str) -> tuple[str, str, str]:
    from matcher.pairs import prompt_parts
    return prompt_parts(prompt)


# ---------------------------------------------------------------- checks

def _company(posting: str) -> str:
    m = re.search(r"^Company: (.+)$", posting, re.M)
    return m.group(1).strip() if m else ""


def _near_pairs(a: dict[str, str], b: dict[str, str], near: float = 0.8, rare: int = 30) -> list[tuple[str, str, float]]:
    """(a id, b id, Jaccard) for every pair of texts sharing at least `near` of their 6-word windows; candidates
    come from windows that few texts of b hold."""
    from matcher.nearcopy import shingles
    sb = {k: shingles(v) for k, v in b.items() if v.strip()}
    inv: dict[str, list[str]] = defaultdict(list)
    for k, sh in sb.items():
        for x in sh:
            inv[x].append(k)
    out = []
    for k, v in a.items():
        sa = shingles(v)
        if len(sa) < 5:
            continue
        cand = Counter(j for x in sa if len(inv.get(x, ())) <= rare for j in inv.get(x, ()))
        for j, n in cand.most_common(5):
            jac = len(sa & sb[j]) / len(sa | sb[j])
            if jac >= near:
                out.append((k, j, round(jac, 3)))
    return out


def check(_: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from matcher import leakage as L
    from matcher.nearcopy import company_key
    rows = jsonl(BUILD / "leak_rows.jsonl")
    df = pd.DataFrame(rows)
    train, dev = df[df.split == "train"].reset_index(drop=True), df[df.split == "dev"].reset_index(drop=True)
    # dev at equal labels (the smallest label's count, seeded)
    m = dev.label.value_counts().min()
    devb = pd.concat([g.sample(m, random_state=0) for _, g in dev.groupby("label")]).reset_index(drop=True)
    out: dict = {"train_rows": len(train), "dev_rows": len(dev), "dev_balanced_rows": len(devb)}
    out["train_to_dev_macro_f1"] = {r["baseline"]: round(r["macro_f1"], 4) for r in L.run_baselines(train, devb)}

    def guess(cols, tr, te, length=False):
        if length:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(L.length_features(tr), tr.label)
            return clf.predict(L.length_features(te))
        return L._fit_predict_text(tr, te, cols)

    # within train: grouped 5-fold by posting (a posting's rows never on both sides), every guesser
    cv = {}
    gkf = GroupKFold(n_splits=5)
    for name, cols, length in (("profile only", ["profile"], False), ("posting only", ["posting"], False),
                               ("resume only", ["resume"], False), ("length", [], True)):
        pred = np.empty(len(train), dtype=object)
        for tr_i, te_i in gkf.split(train, train.label, train.posting_key):
            pred[te_i] = guess(cols, train.iloc[tr_i], train.iloc[te_i], length)
        cv[name] = {"accuracy": round(accuracy_score(train.label, pred), 4),
                    "macro_f1": round(f1_score(train.label, pred, average="macro"), 4)}
        # the new rows alone (twins, venator, teasers), as scored inside the same folds
        new = (train.source != "mix3").to_numpy()
        cv[name]["new_rows_accuracy"] = round(accuracy_score(train.label[new], pred[new]), 4)
    out["train_grouped_cv"] = cv
    devp = {}
    for name, cols, length in (("profile only", ["profile"], False), ("posting only", ["posting"], False),
                               ("resume only", ["resume"], False), ("length", [], True)):
        pr = guess(cols, train, devb, length)
        devp[name] = {"accuracy": round(accuracy_score(devb.label, pr), 4),
                      "macro_f1": round(f1_score(devb.label, pr, average="macro"), 4)}
    out["train_to_dev_balanced"] = devp
    # employers: no test- or dev-pool employer in train (company key of every train posting's Company line, and the
    # venator rows' own employer field, against every name of the freeze's test and dev employer clusters)
    fr = json.loads((ROOT / "eval/fit_freeze.json").read_text())
    held = {company_key(n): side for side in ("test", "dev") for e in fr["employers"][side] for n in [e["employer"], *e["names"]]}
    train_cos = {company_key(x) for x in [*train.employer, *map(_company, train.posting)] if x}
    hits = sorted(k for k in train_cos if k in held)
    out["employers"] = {"train_company_keys": len(train_cos), "in_test_or_dev": hits,
                        "dev_venator_employers_in_train": sorted({company_key(x) for x in dev.employer if x} & train_cos)}
    # near copies across train and dev: postings and resumes, 6-word-window Jaccard >= 0.8
    tp = {f"{r.posting_key}": r.posting for r in train.itertuples()}
    dp = {f"{r.posting_key}": r.posting for r in dev.itertuples()}
    tr_res = {f"t{i}": x for i, x in enumerate(dict.fromkeys(train.resume))}
    dv_res = {f"d{i}": x for i, x in enumerate(dict.fromkeys(dev.resume))}
    np_post = _near_pairs(dp, tp)
    np_res = _near_pairs(dv_res, tr_res)
    out["near_copies"] = {"dev_postings": len(dp), "near_train_postings": np_post[:20], "n_posting_pairs": len(np_post),
                          "dev_resumes": len(dv_res), "n_resume_pairs": len(np_res), "resume_pairs": np_res[:10],
                          "same_posting_key": sorted(set(dp) & set(tp))}
    flat = [v["accuracy"] for v in cv.values()] + [v["accuracy"] for v in devp.values()]
    out["max_guesser_accuracy"] = max(flat)
    out["pass"] = {"guessers_at_most_0.40": max(flat) <= 0.40, "no_held_employer": not hits,
                   "no_near_copies": not np_post and not np_res and not out["near_copies"]["same_posting_key"]}
    (BUILD / "checks.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample")
    e = sub.add_parser("extract")
    e.add_argument("mode", choices=("estimate", "run", "parse"))
    e.add_argument("--max-credits", type=float, default=20.0)
    sub.add_parser("twins")
    sub.add_parser("candidates")
    sub.add_parser("regate")
    sub.add_parser("assemble")
    sub.add_parser("check")
    r = sub.add_parser("render")
    r.add_argument("kind", choices=("postings", "resumes"))
    r.add_argument("mode", choices=("estimate", "run"))
    r.add_argument("--max-credits", type=float, default=30.0)
    r.add_argument("--limit", type=int, default=0, help="send only this many of the rows to render (a pilot)")
    a = ap.parse_args()
    if a.cmd == "sample":
        sample(a)
    elif a.cmd == "extract":
        parse_extracted() if a.mode == "parse" else extract(a)
    elif a.cmd == "twins":
        twins(a)
    elif a.cmd == "candidates":
        candidates(a)
    elif a.cmd == "regate":
        regate_twins()
        regate_resumes()
    elif a.cmd == "assemble":
        assemble(a)
    elif a.cmd == "check":
        check(a)
    elif a.cmd == "render":
        render_twins(a) if a.kind == "postings" else render_resumes(a)


if __name__ == "__main__":
    main()
