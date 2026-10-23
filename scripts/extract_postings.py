"""Adaption adapt (universal-prompt mode) turns real postings into RequirementCard JSON (blueprint v2).

Usage: extract_postings.py estimate|run|collect|merge|parse SET [--max-credits N]
  SET is one of
    audit  the 100 postings in data/built/audit/sample_ids.txt (hand-corrected cards exist)
    fresh  50 more, drawn at random (seed 2027) from postings outside the audit
    rest   every other posting (2,505; 2,655 with audit and fresh)
    rest_us  the rest postings that are US and English by posting_meta (2,158): the ones training
           can use (training_ok drops the 347 others anyway). Done: blueprint b2b2ed8e, 2,157 cards.
    all    audit + fresh + rest in one run (merge needs the three above, or this)
  Do not run `rest` or `all`: rest_us covers every posting training can use. What is left (Symbotic e614b55c,
  which adapt dropped, and the 347 non-US or French postings) is not worth credits. todo_ids skips every
  posting with a completion in any set's raw file, so such a run would send only those.
  estimate  upload (once) and print the credit estimate
  run       estimate, launch adapt, wait, then collect
  collect   wait for a launched run, download it, parse and clean every completion
  merge     write data/built/cards/extracted_real_cards_v2.jsonl from the latest collected audit, fresh, rest
  parse     redo the cleaning of a SET from its saved raw completions (no credits)

Cutover: extracted_real_cards_v2.jsonl is the real-card file from now on. The v1 file
(extracted_real_cards.jsonl) stays as it is, because the hand audit was built from it. Each v2 row
carries meta {country, us, mixed, language, must_phrases}; pairs.load_requirements skips rows that
blueprint.training_ok() rejects (not US, French, or a phrase-like required group).

Files are keyed by a hash of the blueprint, so changing the blueprint never reuses an old upload:
  data/built/extract_v2/<SET>_<hash>_postings.csv     what was sent
  data/built/extract_v2/<SET>_<hash>_raw.jsonl        the model's completion per posting
  data/built/extract_v2/<SET>_<hash>_cards.jsonl      cleaned cards: id, source, card, meta
  data/built/extract_v2/<SET>_<hash>_failures.jsonl   completions that did not parse, with the reason
  data/adaption/extract_v2_<SET>_<hash>.json          dataset id, run id, estimate
Postings the pipeline drops come back missing; rerun `run` for the same SET to send only those.

Identical postings: the board lists some postings several times under different ids with the same text
(same title, company, locations and description). Adaption returned only one copy for 58 of the 59 rest_us
ids that came back missing (it keeps some exact-duplicate rows and drops others), and two copies sent
twice came back with different must_skills 11 times in 63 pairs. So:
  - upload sends each posting text once (the first id in sorted order); parse gives every id with that
    text the same completion (meta.borrowed_from names the id whose completion it is);
  - merge compares texts after norm_text (casefold, every run of punctuation or spaces one space) and ignores the
    company string ("KLA" and "KLA Corporation" with one text gave degree_level bachelor and none);
  - merge gives every id in a group of identical postings (title, locations, description) the labels of the
    group's first id (meta.dup_of), and training_ok keeps only that first id, so one text never carries two
    label sets;
  - the same description under another title or location line is not one posting, so each row keeps its own
    card, but merge marks all but one row of the group meta.same_description_as and training_ok leaves those out;
  - the 87 postings of the real-posting test set (data/built/audit/test_cards.jsonl), every row with one
    of their descriptions and every near copy (matcher.nearcopy: same company and title, or 80% of the
    description's 6-word windows shared; meta.near_test names the test posting) get meta.in_real_test, and
    training_ok leaves them out: no test text in training. merge fails if a training row is still a near copy.

Real-card must_skills stay noisy after cleaning: on the 99 hand-checked audit cards about 4 in 10
still carry a required skill the posting does not require. For the pair step, either treat a
missing must skill on a real card (source extracted_real) as soft in the labeler, or keep real
cards out of the first training mix and use them for the demo and the test set only."""
import argparse, csv, hashlib, io, json, os, random, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from matcher.blueprint import BLUEPRINT, MUST_MANY, UNIVERSAL, parse_card, phrase_groups, posting_meta, training_ok  # noqa: E402
from matcher.nearcopy import NearIndex  # noqa: E402

POSTS = ROOT / "data/raw/ats/postings.jsonl"
AUDIT_IDS = ROOT / "data/built/audit/sample_ids.txt"
TEST_CARDS = ROOT / "data/built/audit/test_cards.jsonl"  # the real-posting test set (build_real_test.py)
OUT = ROOT / "data/built/extract_v2"
STATE = ROOT / "data/adaption"
MERGED = ROOT / "data/built/cards/extracted_real_cards_v2.jsonl"
BP_HASH = hashlib.sha1((BLUEPRINT + UNIVERSAL).encode()).hexdigest()[:8]
if "--hash" in sys.argv:  # reparse or merge the files of an earlier blueprint
    i = sys.argv.index("--hash")
    BP_HASH = sys.argv[i + 1]
    del sys.argv[i:i + 2]
FRESH_SEED, FRESH_N = 2027, 50


def load_postings() -> dict[str, dict]:
    out = {}
    for line in POSTS.open():
        d = json.loads(line)
        out["ats_" + d["simplify_id"]] = d
    return out


def posting_text(r: dict, limit: int | None = 12000) -> str:
    """What adapt is sent (the first 12,000 characters). The cleaning code gets the whole posting
    (limit=None), so sponsorship or work-authorization lines past the cut still count."""
    return f"Title: {r['title']}\nCompany: {r['company']}\nLocations: {'; '.join(r.get('locations') or [])}\n\n{r['description'][:limit]}"


def id_sets(posts: dict[str, dict]) -> dict[str, list[str]]:
    audit = [l.split("\t")[0] for l in AUDIT_IDS.read_text().splitlines() if l.strip()]
    outside = sorted(i for i in posts if i not in set(audit))
    fresh = random.Random(FRESH_SEED).sample(outside, FRESH_N)
    rest = [i for i in sorted(posts) if i not in set(audit) and i not in set(fresh)]
    rest_us = [i for i in rest if us_english(posts[i])]
    return {"audit": audit, "fresh": fresh, "rest": rest, "rest_us": rest_us, "all": audit + fresh + rest}


def us_english(r: dict) -> bool:
    """A US posting in English by posting_meta on the ATS locations and the posting text."""
    m = posting_meta(r.get("locations") or [], r["description"])
    return m["us"] and m["language"] == "en"


def paths(name: str) -> dict[str, Path]:
    return {
        "csv": OUT / f"{name}_{BP_HASH}_postings.csv",
        "raw": OUT / f"{name}_{BP_HASH}_raw.jsonl",
        "cards": OUT / f"{name}_{BP_HASH}_cards.jsonl",
        "fail": OUT / f"{name}_{BP_HASH}_failures.jsonl",
        "state": STATE / f"extract_v2_{name}_{BP_HASH}.json",
    }


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []


def client():
    os.environ.setdefault("ADAPTION_API_KEY", Path("~/.config/adaption/api_key").expanduser().read_text().strip())
    from adaption import Adaption
    return Adaption()


def adapt_kwargs() -> dict:
    return dict(
        column_mapping={"context": ["posting"], "universal_prompt": UNIVERSAL},
        recipe_specification={"recipes": {"prompt_rephrase": False, "deduplication": False, "reasoning_traces": False}},
        brand_controls={"blueprint": BLUEPRINT, "length": "concise", "hallucination_mitigation": False},
        training_type="instruction_dataset",
    )


def have_raw(name: str) -> set[str]:
    return {r["posting_id"] for r in read_jsonl(paths(name)["raw"]) if r.get("completion")}


def norm_text(s: str) -> str:
    """Text for comparing two postings: casefolded, every run of non-alphanumeric characters one space, so
    whitespace, punctuation and dash differences do not hide a copy."""
    return " ".join(re.sub(r"[\W_]+", " ", s.casefold()).split())


def desc_key(r: dict) -> str:
    """One description, whatever company string the board gives it ("KLA" and "KLA Corporation")."""
    return norm_text(r["description"])


def twins(posts: dict, ids: list[str], limit: int | None = 12000, norm: bool = False) -> dict[str, str]:
    """id -> the first id (sorted) with the same posting text. limit=12000 is what adapt is sent;
    limit=None compares the whole posting, which the cleaning code reads. norm=True compares title, locations
    and description after norm_text and ignores the company string, so a copy under "KLA Corporation" with
    other spacing is still the same posting."""
    first: dict[str, str] = {}
    out = {}
    for i in sorted(ids):
        r = posts[i]
        key = (norm_text(r["title"]), norm_text("; ".join(r.get("locations") or [])), desc_key(r)) if norm \
            else posting_text(r, limit)
        out[i] = first.setdefault(key, i)
    return out


SETS = ("audit", "fresh", "rest", "rest_us", "all")


def todo_ids(name: str, ids: list[str], posts: dict) -> list[str]:
    """Ids still to send: one per posting text, and none whose text already has a completion in the raw file
    of ANY set (so `run rest` after `run rest_us` sends only the postings rest_us left out, not 2,369 again)."""
    done = set().union(*(have_raw(n) for n in SETS))
    tw = twins(posts, ids)
    done_text = {posting_text(posts[i]) for i in done if i in posts}
    return [i for i in ids if tw[i] == i and i not in done and posting_text(posts[i]) not in done_text]


def same_description(posts: dict, order: list[str], ok: dict[str, bool]) -> dict[str, str]:
    """id -> the id that stands for its group of postings with the same description after norm_text (the title,
    location line or company string may differ: "Hardware Intern" and "Hardware Co-op" with one text, "KLA" and
    "KLA Corporation"). The group's first id that training_ok accepts stands for it, else its first id. One
    description then gives training one label set."""
    groups: dict[str, list[str]] = {}
    for i in sorted(order):
        groups.setdefault(desc_key(posts[i]), []).append(i)
    out = {}
    for ids in groups.values():
        first = next((i for i in ids if ok.get(i)), ids[0])
        out.update({i: first for i in ids})
    return out


def upload(c, name: str, posts: dict, ids: list[str], p: dict) -> str:
    state = json.loads(p["state"].read_text()) if p["state"].exists() else {}
    if state.get("dataset_id") and state.get("sent") == ids:
        return state["dataset_id"]
    p["csv"].parent.mkdir(parents=True, exist_ok=True)
    with p["csv"].open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["posting_id", "posting"])
        for i in ids:
            w.writerow([i, posting_text(posts[i])])
    did = c.datasets.upload_file(str(p["csv"]), name=f"extract-v2-{name}-{BP_HASH}").dataset_id
    state = {"dataset_id": did, "sent": ids, "blueprint": BP_HASH}
    p["state"].write_text(json.dumps(state, indent=1))
    for _ in range(120):  # adapt answers 409 until the import has a row_count
        st = c.datasets.get_status(did)
        if st.row_count or st.status == "failed":
            break
        time.sleep(5)
    return did


def wait_done(c, did: str, settle_polls: int = 4, interval: float = 30.0):
    """Right after upload the dataset already reads "succeeded", so wait for processed_rows to
    reach total_rows, or to stop moving (rows the pipeline dropped)."""
    last, still = -1, 0
    while True:
        st = c.datasets.get_status(did)
        n, tot = (st.progress.processed_rows, st.progress.total_rows) if st.progress else (0, 0)
        print(time.strftime("%H:%M:%S"), st.status, f"{n}/{tot}", flush=True)
        if st.status == "failed":
            return st, n, tot
        if st.status == "succeeded":
            if tot and n >= tot:
                return st, n, tot
            still = still + 1 if n == last else 0
            if still >= settle_polls:
                return st, n, tot
        last = n
        time.sleep(interval)


def collect(c, name: str, posts: dict, p: dict) -> None:
    state = json.loads(p["state"].read_text())
    did = state["dataset_id"]
    st, n, tot = wait_done(c, did)
    print("DONE", st)
    if st.status != "succeeded":
        sys.exit(f"adapt ended with status {st.status}")
    res = c.datasets.download(did, file_format="csv")
    data = res.read() if hasattr(res, "read") else res.content
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))
    sent = set(state["sent"])
    got = {r["posting_id"]: (r.get("enhanced_completion") or "").strip() for r in rows if r.get("posting_id") in sent}
    p["raw"].parent.mkdir(parents=True, exist_ok=True)
    old = {r["posting_id"]: r for r in read_jsonl(p["raw"])}
    for i, t in got.items():
        if t:
            old[i] = {"posting_id": i, "completion": t}
    with p["raw"].open("w") as f:
        for r in old.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    state.update(finished=time.time(), processed=n, total=tot, returned=len(got))
    p["state"].write_text(json.dumps(state, indent=1))
    parse_all(name, posts, p, id_sets(posts)[name])
    print(f"{name}: {len(old)} completions on file for {len(state['sent'])} sent")


def parse_all(name: str, posts: dict, p: dict, ids: list[str] | None = None) -> tuple[int, int]:
    """raw completions -> cleaned cards (+ meta) and a failures file. Rerunnable offline. An id of
    the set with no completion of its own takes the completion of an id with the same sent text."""
    cards, fails = [], []
    raw = {r["posting_id"]: r["completion"] for r in read_jsonl(p["raw"]) if r.get("completion")}
    borrowed: dict[str, str] = {}
    if ids:
        by_text: dict[str, str] = {}
        for i in sorted(raw):
            by_text.setdefault(posting_text(posts[i]), i)
        for i in ids:
            if i not in raw and posting_text(posts[i]) in by_text:
                borrowed[i] = by_text[posting_text(posts[i])]
    order = [*raw, *(i for i in ids or [] if i in borrowed)]
    for i in order:
        text = raw[borrowed.get(i, i)]
        post = posts[i]
        try:
            body = text.strip()
            if body.startswith("```"):
                body = body.strip("`").removeprefix("json").strip()
            card = parse_card(i, body, posting_text(post, None), post.get("locations") or [])
        except Exception as e:  # noqa: BLE001  any bad completion is recorded, not fatal
            fails.append({"posting_id": i, "error": f"{type(e).__name__}: {e}", "completion": text[:600]})
            continue
        meta = posting_meta(post.get("locations") or [], post["description"], card.locations)
        meta["must_phrases"] = phrase_groups([list(g) for g in card.must_skills])
        meta["must_many"] = len(card.must_skills) >= MUST_MANY
        if i in borrowed:
            meta["borrowed_from"] = borrowed[i]
        cards.append({"id": i, "source": "extracted_real", "card": card.to_dict(), "meta": meta})
    with p["cards"].open("w") as f:
        for r in cards:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with p["fail"].open("w") as f:
        for r in fails:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(cards)} cards ({len(borrowed)} from an identical posting's completion), "
          f"{len(fails)} failed to parse")
    return len(cards), len(fails)


def merge(posts: dict) -> None:
    sets = id_sets(posts)
    by: dict[str, dict] = {}
    for name in SETS:
        for r in read_jsonl(paths(name)["cards"]):
            by[r["id"]] = r
    order = [i for i in sets["all"] if i in by]
    # identical postings (title, locations and description equal after norm_text, any company string): every id
    # gets the first id's labels, so one text has one label set; the row keeps its own title
    tw = twins(posts, order, None, norm=True)
    ndup = 0
    for i in order:
        if tw[i] != i:
            ndup += 1
            src = by[tw[i]]
            by[i] = {**src, "id": i, "card": {**src["card"], "source_posting_id": i, "title": by[i]["card"]["title"]},
                     "meta": {**src["meta"], "dup_of": tw[i]}}
    # the real-posting test set stays out of training, and so does every row with a test posting's description
    test = {r["id"] for r in read_jsonl(TEST_CARDS)}
    test_desc = {desc_key(posts[i]) for i in test if i in posts}
    ntest = 0
    for i in order:
        if desc_key(posts[i]) in test_desc:
            ntest += 1
            by[i] = {**by[i], "meta": {**by[i]["meta"], "in_real_test": True}}
    # near copies of a test posting (same company and title, or 80% of the description's 6-word windows): the
    # board lists one job under several ids with another location or pay line (Booz Allen e36f52f0 has 9)
    index = NearIndex({i: posts[i] for i in sorted(test) if i in posts})
    nnear = 0
    for i in order:
        if not by[i]["meta"].get("in_real_test") and (hit := index.near(posts[i])):
            nnear += 1
            by[i] = {**by[i], "meta": {**by[i]["meta"], "in_real_test": True, "near_test": list(hit)}}
    # same company and description under another title or location line: training keeps one row per description
    rep = same_description(posts, order, {i: training_ok(by[i].get("meta")) for i in order})
    nsame = 0
    for i in order:
        if rep[i] != i and not by[i]["meta"].get("dup_of"):
            nsame += 1
            by[i] = {**by[i], "meta": {**by[i]["meta"], "same_description_as": rep[i]}}
    # gate: no row that may train is a near copy of a test posting
    slipped = [i for i in order if training_ok(by[i].get("meta")) and index.near(posts[i])]
    if slipped:
        raise SystemExit(f"{len(slipped)} training rows are near copies of real test postings, e.g. {slipped[:5]}")
    MERGED.parent.mkdir(parents=True, exist_ok=True)
    with MERGED.open("w") as f:
        for i in order:
            f.write(json.dumps(by[i], ensure_ascii=False) + "\n")
    missing = [i for i in sets["all"] if i not in by]
    skipped = sum(not us_english(posts[i]) for i in missing)
    print(f"wrote {len(order)} of {len(sets['all'])} cards to {MERGED}; missing {len(missing)} "
          f"({skipped} of them not US or not English, never sent when rest_us was used)")
    metas = [by[i].get("meta") or {} for i in order]
    print(f"identical posting text: {ndup} rows carry the card of an earlier id (meta.dup_of)")
    print(f"same description (any company string), other title or locations: {nsame} rows left out of training "
          f"(meta.same_description_as)")
    print(f"real-posting test set: {ntest} rows with the description of one of its {len(test)} postings and {nnear} "
          f"near copies (meta.near_test) left out of training (meta.in_real_test)")
    print(f"training_ok {sum(map(training_ok, metas))}; not US {sum(not m.get('us', True) for m in metas)}; "
          f"French {sum(m.get('language') == 'fr' for m in metas)}; phrase-like required group "
          f"{sum(bool(m.get('must_phrases')) for m in metas)}; {MUST_MANY}+ required groups {sum(bool(m.get('must_many')) for m in metas)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("estimate", "run", "collect", "merge", "parse"))
    ap.add_argument("set", nargs="?", default="audit", choices=("audit", "fresh", "rest", "rest_us", "all"))
    ap.add_argument("--max-credits", type=float, default=5.0)
    a = ap.parse_args()
    posts = load_postings()
    if a.mode == "merge":
        return merge(posts)
    sets = id_sets(posts)
    p = paths(a.set)
    if a.mode == "parse":
        parse_all(a.set, posts, p, sets[a.set])
        return
    c = client()
    if a.mode == "collect":
        return collect(c, a.set, posts, p)
    ids = todo_ids(a.set, sets[a.set], posts)
    print(f"blueprint {BP_HASH}; {a.set}: {len(sets[a.set])} postings, {len(ids)} still to extract")
    if not ids:
        return
    did = upload(c, a.set, posts, ids, p)
    est = c.datasets.adapt(did, estimate=True, **adapt_kwargs())
    print("ESTIMATE", est)
    state = json.loads(p["state"].read_text())
    state["estimate_credits"] = est.estimated_credits_consumed
    p["state"].write_text(json.dumps(state, indent=1))
    if a.mode == "estimate":
        return
    if est.estimated_credits_consumed > a.max_credits:
        sys.exit(f"estimate {est.estimated_credits_consumed} > --max-credits {a.max_credits}; not running")
    if "run_id" in state:
        print("a run is already launched for this upload; collecting it")
    else:
        r = c.datasets.adapt(did, estimate=False, **adapt_kwargs())
        print("RUN", r)
        state.update(run_id=r.run_id, started=time.time())
        p["state"].write_text(json.dumps(state, indent=1))
        time.sleep(30)
    collect(c, a.set, posts, p)


if __name__ == "__main__":
    main()
