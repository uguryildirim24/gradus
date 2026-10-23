"""Fix cycle 2: re-extract requirement cards with the v3 draft blueprint, written next to the v2 cards (never over them).

  extract_v3.py estimate|run|parse SET [--blueprint v3|v2] [--max-credits N]

--blueprint v3 (default) sends matcher.blueprint.BLUEPRINT_V3, the fix-cycle-2 draft, which no pipeline sends; v2 sends
the live v2 text again (BLUEPRINT_V2), a same-instruction re-extraction that measures run-to-run noise. Outputs carry
the tag: cards_v3.jsonl / cards_v2rerun.jsonl, raw completions extract_raw_<tag>_<hash>.jsonl.

SET (each read with the posting adapter its v2 cards were made with, so the instruction is the only change):
  ft1   fresh test 1's 150 postings (data/built/fresh_test/postings.jsonl, build_fresh_test.adapter)
        -> data/built/fresh_test/cards_<tag>.jsonl (+ extract_raw_<tag>_<hash>.jsonl, extract_failures_<tag>.jsonl)
  ft2   fresh test 2's 95 postings, read from the frozen test pool by eval/fresh_test2_manifest.json posting_ids, so
        nothing under data/built/fresh_test2/ is opened; only cards_<tag>.jsonl (+ raw, failures) is written there
  fit   the venator and teaser postings train_fit.csv uses (build/leak_rows.jsonl train rows) and every dev-pool posting,
        from build/sample.jsonl (build_fit_train.as_ats, rules.mark_thin on the body as build_fit_train does)
        -> data/built/fit/build/extract_cards_<tag>.jsonl (+ extract_raw_<tag>_<hash>.jsonl, extract_failures_<tag>.jsonl)

Each distinct posting text is sent once, in datasets of at most CHUNK rows (Adaption's estimate grows faster than the
row count). `estimate` uploads and prints the free estimate; `run` refuses when the estimates of the chunks not yet
launched add up to more than --max-credits or would take the cycle past CAP, then launches, waits, downloads and parses.
Credits are read before and after every run (datasets.invent estimate, free) into data/adaption/fixcycle2_credits.json.
State files data/adaption/fixcycle2_<SET>_<hash>_<i>.json make a rerun resume. `parse` redoes the cleaning offline.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import hashlib  # noqa: E402

import extract_postings as ext  # noqa: E402
from matcher.blueprint import BLUEPRINT_V2, BLUEPRINT_V3, MUST_MANY, UNIVERSAL, parse_card, phrase_groups, posting_meta, \
    training_ok  # noqa: E402
from matcher.rules import mark_thin  # noqa: E402

STATE = ROOT / "data/adaption"
CREDITS = STATE / "fixcycle2_credits.json"
CHUNK = 250
SEED = 20261021
CAP = 25.0
FT1 = ROOT / "data/built/fresh_test"
FT2 = ROOT / "data/built/fresh_test2"  # written to only: cards_<tag>.jsonl, extract_raw_<tag>_<hash>.jsonl, failures
FIT_BUILD = ROOT / "data/built/fit/build"
csv.field_size_limit(10**9)


def jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open() if line.strip()] if p.exists() else []


def write_jsonl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- sets


def ft1_posts() -> dict[str, dict]:
    import build_fresh_test as bft
    return {p["key"]: bft.adapter(p) for p in jsonl(FT1 / "postings.jsonl")}


def ft2_posts() -> dict[str, dict]:
    import build_fresh_test as bft
    from fit_pools import load_pool
    ids = json.loads((ROOT / "eval/fresh_test2_manifest.json").read_text())["posting_ids"]
    pool = {p["key"]: p for p in load_pool("test")}
    return {i: bft.adapter(pool[i]) for i in ids}


def fit_rows() -> dict[str, dict]:
    """sample.jsonl rows of the train venator / teaser postings train_fit.csv uses and of every dev-pool posting."""
    used = {r["posting_key"] for r in jsonl(FIT_BUILD / "leak_rows.jsonl")
            if r["source"] in ("venator", "teaser") and r["split"] == "train"}
    dev = {json.loads(line)["key"] for line in (ROOT / "data/built/fit/pools/dev.jsonl").open() if line.strip()}
    return {p["key"]: p for p in jsonl(FIT_BUILD / "sample.jsonl") if p["key"] in used | dev}


def fit_posts() -> dict[str, dict]:
    import build_fit_train as bft
    return {k: bft.as_ats(p) for k, p in fit_rows().items()}


BLUEPRINTS = {"v3": BLUEPRINT_V3, "v2rerun": BLUEPRINT_V2}
TAG = "v3"
BP = BLUEPRINT_V3
BP_HASH = hashlib.sha1((BP + UNIVERSAL).encode()).hexdigest()[:8]
SETS: dict = {}


def use(tag: str) -> None:
    """Point the blueprint, its hash and every output path at one tag (raw completions are kept per hash, so drafts
    never mix)."""
    global TAG, BP, BP_HASH
    TAG, BP = tag, BLUEPRINTS[tag]
    BP_HASH = hashlib.sha1((BP + UNIVERSAL).encode()).hexdigest()[:8]
    SETS.update({
        "ft1": (ft1_posts, FT1 / f"cards_{tag}.jsonl", FT1 / f"extract_raw_{tag}_{BP_HASH}.jsonl",
                FT1 / f"extract_failures_{tag}.jsonl"),
        "ft2": (ft2_posts, FT2 / f"cards_{tag}.jsonl", FT2 / f"extract_raw_{tag}_{BP_HASH}.jsonl",
                FT2 / f"extract_failures_{tag}.jsonl"),
        "fit": (fit_posts, FIT_BUILD / f"extract_cards_{tag}.jsonl", FIT_BUILD / f"extract_raw_{tag}_{BP_HASH}.jsonl",
                FIT_BUILD / f"extract_failures_{tag}.jsonl"),
    })


def adapt_kwargs() -> dict:
    """extract_postings.adapt_kwargs with this tag's blueprint in place of the live one."""
    kw = ext.adapt_kwargs()
    kw["brand_controls"] = {**kw["brand_controls"], "blueprint": BP}
    return kw


use("v3")
CARD_SOURCE = {"ft1": "extracted_fresh_test", "ft2": "extracted_fresh_test2"}


# ---------------------------------------------------------------- Adaption


def credits_log() -> dict:
    return json.loads(CREDITS.read_text()) if CREDITS.exists() else {"runs": []}


def spent() -> float:
    return round(sum(r.get("spent", r["estimate"]) for r in credits_log()["runs"]), 4)


def balance(c) -> float:
    return float(c.datasets.invent(rows=10, estimate=True, prompt="balance check").available_credits)


def upload(c, path: Path, rows: list[list[str]], name: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["posting_id", "posting"])
        w.writerows(rows)
    did = c.datasets.upload_file(str(path), name=name).dataset_id
    for _ in range(120):  # adapt answers 409 until the import has a row count
        st = c.datasets.get_status(did)
        if st.row_count or st.status == "failed":
            break
        time.sleep(5)
    return did


def todo(name: str, posts: dict[str, dict]) -> list[str]:
    """One id per posting text (the first, sorted), none that already has a v3 completion."""
    done = {r["posting_id"] for r in jsonl(SETS[name][2]) if r.get("completion")}
    first: dict[str, str] = {}
    for i in sorted(posts):
        first.setdefault(ext.posting_text(posts[i]), i)
    done_text = {ext.posting_text(posts[i]) for i in done if i in posts}
    out = [i for i in sorted(posts) if first[ext.posting_text(posts[i])] == i and i not in done
           and ext.posting_text(posts[i]) not in done_text]
    random.Random(SEED).shuffle(out)  # sorted keys put one source's postings in one chunk; mixed chunks estimate lower
    return out


def cmd_run(name: str, mode: str, max_credits: float) -> None:
    posts = SETS[name][0]()
    ids = todo(name, posts)
    print(f"{name}: blueprint {TAG} {BP_HASH}; {len(posts)} postings, {len(ids)} texts to send")
    if not ids:
        return parse(name)
    c = ext.client()
    chunks = [ids[i:i + CHUNK] for i in range(0, len(ids), CHUNK)]
    states = [STATE / f"fixcycle2_{name}_{BP_HASH}_{i}.json" for i in range(len(chunks))]
    pending = 0.0
    for i, (sp, ch) in enumerate(zip(states, chunks)):
        st = json.loads(sp.read_text()) if sp.exists() else {}
        if st.get("sent") != ch:
            st = {"sent": ch, "dataset_id": upload(c, FIT_BUILD.parent.parent / "extract_v3" / f"{sp.stem}.csv",
                                                     [[k, ext.posting_text(posts[k])] for k in ch],
                                                     f"fixcycle2-{name}-{BP_HASH}-{i}")}
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_text(json.dumps(st, indent=1))
        if "estimate" not in st:
            st["estimate"] = float(c.datasets.adapt(st["dataset_id"], estimate=True, **adapt_kwargs())
                                   .estimated_credits_consumed)
            sp.write_text(json.dumps(st, indent=1))
        if not st.get("run_id"):
            pending += st["estimate"]
        print(f"  chunk {i}: {len(ch)} rows, estimate {st['estimate']}, run {st.get('run_id', '-')}")
    print(f"{name}: estimate for chunks not yet launched {round(pending, 4)}; cycle spent so far {spent()} (cap {CAP})")
    if mode == "estimate":
        return
    if pending > max_credits or spent() + pending > CAP:
        sys.exit(f"{name}: {pending} over --max-credits {max_credits} or past the {CAP} cap; not running")
    before = balance(c)
    for i, sp in enumerate(states):
        st = json.loads(sp.read_text())
        if not st.get("run_id"):
            r = c.datasets.adapt(st["dataset_id"], estimate=False, **adapt_kwargs())
            assert adapt_kwargs()["brand_controls"]["blueprint"] == BP
            st.update(run_id=r.run_id, started=time.time())
            sp.write_text(json.dumps(st, indent=1))
    time.sleep(20)
    raw = {r["posting_id"]: r for r in jsonl(SETS[name][2])}
    for i, sp in enumerate(states):
        st = json.loads(sp.read_text())
        if not st.get("finished"):
            res, n, tot = ext.wait_done(c, st["dataset_id"], interval=20.0)
            if res.status != "succeeded":
                sys.exit(f"{name} chunk {i}: adapt ended {res.status}")
            got = c.datasets.download(st["dataset_id"], file_format="csv")
            data = got.read() if hasattr(got, "read") else got.content
            (STATE / f"{sp.stem}_download.csv").write_bytes(data)
            st.update(finished=time.time(), processed=n, total=tot)
            sp.write_text(json.dumps(st, indent=1))
        data = (STATE / f"{sp.stem}_download.csv").read_text()
        for r in csv.DictReader(io.StringIO(data, newline="")):
            t = (r.get("enhanced_completion") or "").strip()
            if r.get("posting_id") in posts and t:
                raw[r["posting_id"]] = {"posting_id": r["posting_id"], "blueprint": BP_HASH, "completion": t}
    write_jsonl(SETS[name][2], raw.values())
    after = balance(c)
    log = credits_log()
    log["runs"].append({"set": name, "blueprint": BP_HASH, "chunks": len(chunks), "rows": len(ids),
                        "estimate": round(pending, 4), "balance_before": before, "balance_after": after,
                        "spent": round(before - after, 4), "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    log["total_spent"] = round(sum(r.get("spent", r["estimate"]) for r in log["runs"]), 4)
    CREDITS.write_text(json.dumps(log, indent=1))
    print(f"{name}: balance {before} -> {after}")
    parse(name)


def parse(name: str) -> None:
    """Raw v3 completions -> cards in the shape of that set's v2 card file. A posting with no completion of its own
    takes the completion of one with the same sent text."""
    posts = SETS[name][0]()
    raw = {r["posting_id"]: r["completion"] for r in jsonl(SETS[name][2]) if r.get("completion")}
    by_text: dict[str, str] = {}
    for k in sorted(raw):
        by_text.setdefault(ext.posting_text(posts[k]), k)
    body_of = {k: p.get("text") or "" for k, p in fit_rows().items()} if name == "fit" else {}
    split_kind = {k: (p["split"], p["kind"]) for k, p in fit_rows().items()} if name == "fit" else {}
    cards, fails = [], []
    for k in sorted(posts):
        src = k if k in raw else by_text.get(ext.posting_text(posts[k]))
        if src is None:
            continue
        post = posts[k]
        body = raw[src].strip()
        if body.startswith("```"):
            body = body.strip("`").removeprefix("json").strip()
        try:
            card = parse_card(k, body, ext.posting_text(post, None), post["locations"])
        except Exception as e:  # noqa: BLE001  a bad completion is recorded, not fatal
            fails.append({"posting_id": k, "error": f"{type(e).__name__}: {e}", "completion": raw[src][:600]})
            continue
        meta = posting_meta(post["locations"], post["description"], card.locations)
        if src != k:
            meta["borrowed_from"] = src
        if name == "fit":
            card = mark_thin(card, body_of[k])
            meta["must_phrases"] = phrase_groups([list(g) for g in card.must_skills])
            meta["must_many"] = len(card.must_skills) >= MUST_MANY
            meta["training_ok"] = training_ok(meta)
            split, kind = split_kind[k]
            cards.append({"key": k, "split": split, "kind": kind, "extractor": BP_HASH, "card": card.to_dict(),
                          "meta": meta})
        else:
            cards.append({"id": k, "source": CARD_SOURCE[name], "extractor": BP_HASH, "card": card.to_dict(),
                          "meta": meta})
    write_jsonl(SETS[name][1], cards)
    write_jsonl(SETS[name][3], fails)
    print(f"{name}: {len(cards)} {TAG} cards of {len(posts)} postings, {len(fails)} failed to parse -> {SETS[name][1]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("estimate", "run", "parse"))
    ap.add_argument("set", choices=tuple(SETS))
    ap.add_argument("--blueprint", choices=("v3", "v2"), default="v3")
    ap.add_argument("--max-credits", type=float, default=5.0)
    a = ap.parse_args()
    use("v3" if a.blueprint == "v3" else "v2rerun")
    if a.mode == "parse":
        return parse(a.set)
    cmd_run(a.set, a.mode, a.max_credits)


if __name__ == "__main__":
    main()
