"""The exact prompts each model is scored on, for dev_fit and the sealed fresh test, built before any model runs.

  fit_eval_inputs.py build     data/built/fit/eval_inputs/{h,d3,base}/{dev_fit,fresh_test}.jsonl and
                               data/built/fit/eval_inputs/report.json (format check, parity check, blank-field shares)
  fit_eval_inputs.py build2    data/built/fit/eval_inputs/{h,d3,base}/fresh_test2.jsonl and
                               eval_inputs/report_fresh_test2.json (fresh test 2, amendment 2; refuses to overwrite)
  fit_eval_inputs.py build3    data/built/fit/eval_inputs/{h,d3,base}/fresh_test3.jsonl and
                               eval_inputs/report_fresh_test3.json (fresh test 3, amendment 3; refuses to overwrite;
                               checks fresh test 3's rows, postings and gold against their pins first)
  fit_eval_inputs.py build4    data/built/fit/eval_inputs/B/fresh_test{,2,3}.jsonl and eval_inputs/report_cycle3_B.json
                               (cycle-3 run B's format on fresh tests 1-3, amendments 5 and 5b; refuses to overwrite;
                               checks each test's rows and gold against their pins and B's format against train_B.csv)
  fit_eval_inputs.py build5    data/built/fit/eval_inputs/{h,base,B}/fresh_test4.jsonl, eval_inputs/report_fresh_test4.json
                               and eval/fresh_test4_eval_inputs.json (sealed fresh test 4, amendments 5 and 5b: B in its
                               own format, h and base in h's; refuses to overwrite; checks every test-4 pin first and
                               that each row's texts are what the labelers saw)
  fit_eval_inputs.py parity    the parity check alone (prints it)

Each model gets the prompt format its own training data used (eval/protocol_amendment_1.json, "prompts"):
  h     train_fit.csv (scripts/build_fit_train.py): pairs.build_prompt(resume, profile.render_profile(card), posting),
        a real posting through build_fit_train.posting_text_of (textclean.real_posting_text on the venator pool row:
        title, employer, the pool's location string, the body), a twin posting through textclean.training_posting
  d3    train_mix3.csv (scripts/build_train_mix3.py): the same build_prompt and render_profile; a real posting through
        build_real_test.posting_text (textclean.real_posting_text on an ATS row: the location list, the body); a
        synthetic posting through textclean.training_posting (render_cards.training_postings)
  base  h's format (the untrained Qwen3.5-4B)
The instruction text (pairs.INSTRUCTION), the profile renderer, the prompt frame and the chat framing that scores
them (local_eval.frame: the tokenizer's chat template, enable_thinking False) are one and the same for h and d3; the
report counts how many prompts differ between the h and d3 builds (the only possible source: the location header of a
pool posting, written from the pool's string by h and from that string split on ';' by d3).

The fresh test's blank form fields stay blank (that is part of the test). Training filled them in part
(build_fit_train.neutral_fill; the report gives the shares of each blank in train_fit.csv, train_mix3.csv and the
fresh test, and says whether "Lives in: not given" ever reached a training prompt).

Parity: training prompts are rebuilt from their raw parts with the functions above and must equal rows of
train_fit.csv and train_mix3.csv byte for byte; a sample of 20 from each must match, and the report gives how many of
each source rebuild at all. dev_fit's stored prompt must equal the h build on every row. Any mismatch stops the build.

Output rows: row_id, pair_id (= row_id, the key local_eval/colab_eval read), posting_id (the bootstrap's posting
group), base_rq_id (= posting_id), prompt, gold (Keep / Explore / Skip) and bucket (the gold in local_eval's words).
Loads no model and calls nothing; reads local files only.

Cycle 3 (build4): run A trained on h's format (build_fit_train3.prompt_a is build_prompt), so A is scored on h's
existing fresh test 1-3 files. Run B trained on build_fit_train3.prompt_b: the rubric (system_b) at the top of the
single user turn, then h's layout without INSTRUCTION, with the whole posting. B's rows here are prompt_b on the row's
own resume_text, render_profile(profile) and posting_text (rows.jsonl: the posting as the labelers saw it, which is
what train_B.csv's new rows, built by the same fresh-test pipeline, hold). Before anything is written every kept new
row of train_B.csv is rebuilt this way from data/built/fit/cycle3/rows.jsonl and must equal its CSV row byte for byte.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import build_fit_train as bft  # noqa: E402
from matcher.pairs import build_prompt, prompt_parts  # noqa: E402
from matcher.profile import render_profile  # noqa: E402
from matcher.schema import CandidateCard  # noqa: E402

FIT = ROOT / "data/built/fit"
FRESH = ROOT / "data/built/fresh_test"
FRESH2 = ROOT / "data/built/fresh_test2"
FRESH3 = ROOT / "data/built/fresh_test3"
FRESH3_MANIFEST = ROOT / "eval/fresh_test3_manifest.json"
FRESH3_GOLD_MANIFEST = ROOT / "eval/fresh_test3_gold_manifest.json"
OUT = FIT / "eval_inputs"
DEV_FIT = FIT / "dev_fit.jsonl"
TRAIN_FIT = FIT / "train_fit.csv"
TRAIN_MIX3 = ROOT / "data/built/real_train/train_mix3.csv"
MODELS = ("h", "d3", "base")
SETS = ("dev_fit", "fresh_test")
BUCKET = {"Keep": "For you", "Explore": "Explore", "Skip": "Excluded"}
PARITY_N = 20
PARITY_SEED = 20261021

csv.field_size_limit(10**9)


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- the two formats


def ats_row(p: dict) -> dict:
    """A venator pool posting as the ATS row build_real_test.posting_text reads (mix3's real postings)."""
    return {"title": p["title"], "company": p.get("employer", ""), "simplify_id": p["key"],
            "locations": [x.strip() for x in (p.get("locations") or "").split(";") if x.strip()],
            "description": p.get("text") or ""}


def posting_h(p: dict) -> str:
    """A pool posting as train_fit's venator and teaser rows show one."""
    return bft.posting_text_of(p)


def posting_d3(p: dict) -> str:
    """A pool posting as train_mix3's real rows show one."""
    import build_real_test as brt
    return brt.posting_text(ats_row(p))


POSTING = {"h": posting_h, "d3": posting_d3, "base": posting_h}


def prompt_of(resume: str, profile: dict | CandidateCard, posting: str) -> str:
    c = profile if isinstance(profile, CandidateCard) else CandidateCard.from_dict(profile)
    return build_prompt(resume, render_profile(c), posting)


def twin_posting(posting_key: str, sub: str, cache: dict) -> str:
    """A twin posting from its raw parts (twin card + Adaption's rendered text), as build_fit_train.twin_rows writes it."""
    from matcher import textclean
    if not cache:
        cache["cards"] = {r["id"]: r for r in jsonl(bft.TWIN_CARDS)}
        cache["texts"] = {r["id"]: r["text"] for r in jsonl(bft.TWIN_POSTINGS)}
        cache["bank"] = textclean.training_bank(ROOT)
    vid = f"tw_{posting_key}~{sub}"
    return textclean.training_posting(cache["cards"][vid]["display_card"], cache["texts"][vid], vid, cache["bank"])


# ---------------------------------------------------------------- the eval rows


def dev_fit_rows(model: str) -> list[dict]:
    """dev_fit from its raw parts: a venator or teaser row's dev-pool posting (build/sample.jsonl), a twin's card and
    rendered text; the resume and the profile as dev_fit stores them."""
    pool = {p["key"]: p for p in jsonl(bft.SAMPLE)}
    cache: dict = {}
    out = []
    for r in jsonl(DEV_FIT):
        if r["source"] == "twins":
            posting = twin_posting(r["posting_key"], r["sub"], cache)
        else:
            p = pool[r["posting_key"]]
            assert p["split"] == "dev", r["id"]
            posting = POSTING[model](p)
        out.append(row(r["id"], r["posting_key"], prompt_of(r["resume"], r["profile"], posting), r["gold"]))
    return out


def fresh_rows(model: str, d: Path = FRESH) -> list[dict]:
    posts = {p["key"]: p for p in jsonl(d / "postings.jsonl")}
    gold = {g["row_id"]: g["gold"] for g in jsonl(d / "gold.jsonl")}
    out = []
    for r in jsonl(d / "rows.jsonl"):
        out.append(row(r["row_id"], r["posting_id"],
                       prompt_of(r["resume_text"], r["profile"], POSTING[model](posts[r["posting_id"]])), gold[r["row_id"]]))
    return out


def row(rid: str, posting_id: str, prompt: str, gold: str) -> dict:
    return {"row_id": rid, "pair_id": rid, "posting_id": posting_id, "base_rq_id": posting_id, "prompt": prompt,
            "gold": gold, "bucket": BUCKET[gold]}


BUILDERS = {"dev_fit": dev_fit_rows, "fresh_test": fresh_rows}


# ---------------------------------------------------------------- parity with the training CSVs


def csv_prompts(p: Path) -> list[str]:
    return [r["prompt"] for r in csv.DictReader(p.open(newline=""))]


def rebuild_fit_venator() -> list[str]:
    """train_fit's venator rows from their raw parts: the train-pool posting (build/sample.jsonl), the candidate drawn
    for it (build/candidates.jsonl), its resume (build/resumes.jsonl, by the unfilled candidate's context) and the
    form, filled by neutral_fill with the row's own seed and also as drawn (venator_rows tries filled, then as
    drawn). Every rebuild is returned; the caller counts which are training rows."""
    import render_cards as rc
    posts = {p["key"]: p for p in jsonl(bft.SAMPLE)}
    cards = {r["key"]: r for r in jsonl(bft.EXTRACT_CARDS)}
    res = {r["ctx"]: r["text"] for r in jsonl(bft.FIT_RESUMES)}
    out = []
    for r in jsonl(bft.CANDS):
        if r["split"] != "train" or r["key"] not in cards:
            continue
        rtext = res.get(rc.ctx_hash(rc.ctx_str("resumes", r["candidate"])))
        if rtext is None:
            continue
        card = bft.label_card_of(cards[r["key"]])
        filled = bft.neutral_fill(r["candidate"], card.to_dict(),
                                  random.Random(f"{bft.SEED}:{r['key']}:{r['cd_id']}:{r['kind']}:fill"))
        posting = posting_h(posts[r["key"]])
        out.append([prompt_of(rtext, c, posting) for c in (filled, r["candidate"])])
    return out


def rebuild_mix3_real() -> list[str]:
    """train_mix3's real rows from their raw parts: the ATS posting (build_real_test.RAW_POSTINGS), the drawn form
    (mix3/rows.jsonl), the resume rendered for the row's first form (real_train/rows.jsonl, by context)."""
    import assemble_real_train as art
    import build_real_test as brt
    from build_real_train import ctx_of
    from render_cards import texts_by_ctx
    raw = art.brt_raw()
    store = texts_by_ctx("resumes")
    first = {r["pair_id"]: r["candidate"] for r in jsonl(ROOT / "data/built/real_train/rows.jsonl")}
    out = []
    for r in jsonl(ROOT / "data/built/real_train/mix3/rows.jsonl"):
        if r.get("in_csv") not in (True, "True"):
            continue
        resume = store.get(ctx_of(first[r["old_pair_id"]]))
        if resume is None:
            continue
        out.append([prompt_of(resume, bft._d(r["candidate"]), brt.posting_text(raw[r["rq_id"]]))])
    return out


def mix3_real_count() -> int:
    """train_mix3 rows that are not smoke.csv's synthetic rows (data/built/training/train.jsonl, by prompt)."""
    synth = {json.loads(l)["prompt"] for l in (ROOT / "data/built/training/train.jsonl").open()}
    return sum(p not in synth for p in csv_prompts(TRAIN_MIX3))


def parity() -> dict:
    """Rebuild training prompts from raw parts. Every row of the source (train_fit's venator rows, train_mix3's real
    rows: the real-posting rows, whose format the fresh test gets) must be among the rebuilds, and a seeded sample of
    PARITY_N of them is compared byte for byte (encoded UTF-8) with its CSV row."""
    rng = random.Random(PARITY_SEED)
    rep = {}
    for name, path, rebuild, source_rows in (
            ("train_fit.csv (venator rows)", TRAIN_FIT, rebuild_fit_venator,
             json.loads((bft.BUILD / "report.json").read_text())["rows_by_source"]["venator"]),
            ("train_mix3.csv (real rows)", TRAIN_MIX3, rebuild_mix3_real, mix3_real_count())):
        prompts = csv_prompts(path)
        index = {p: i for i, p in enumerate(prompts)}
        cands = rebuild()
        hits = [(next(p for p in alts if p in index)) for alts in cands if any(p in index for p in alts)]
        hit_rows = sorted({index[p] for p in hits})
        sample = rng.sample(hit_rows, min(PARITY_N, len(hit_rows)))
        # byte for byte: the rebuilt text against the CSV row read back as bytes
        by_row = {index[p]: p for p in hits}
        exact = sum(by_row[i].encode() == prompts[i].encode() for i in sample)
        rep[name] = {"csv_rows": len(prompts), "rebuilt_from_raw_parts": len(cands),
                     "rebuilt_found_in_csv": len(hit_rows), "source_rows_in_csv": source_rows,
                     "sample": len(sample), "sample_byte_equal": exact, "sample_rows": sorted(sample)}
        if len(sample) < PARITY_N or exact != len(sample) or len(hit_rows) != source_rows:
            raise SystemExit(f"parity failed for {name}: {rep[name]}")
    # train_fit's mix3 rows are train_mix3's rows, byte for byte (the d3 side of "same functions")
    fit, mix = csv_prompts(TRAIN_FIT), set(csv_prompts(TRAIN_MIX3))
    rep["train_fit rows that are train_mix3 rows"] = sum(p in mix for p in fit)
    # dev_fit's stored prompt is the h build of every row
    dev = jsonl(DEV_FIT)
    built = dev_fit_rows("h")
    same = sum(a["prompt"] == b["prompt"] for a, b in zip(dev, built))
    rep["dev_fit stored prompt == h build"] = f"{same}/{len(dev)}"
    if same != len(dev):
        raise SystemExit(f"dev_fit: {len(dev) - same} rows differ from the h build")
    return rep


# ---------------------------------------------------------------- what the report says


PROFILE_BLANKS = {"Minimum salary: not set": "salary_floor not set", "Job types wanted: any": "employment_types_wanted any",
                  "Work arrangement: onsite, hybrid or remote": "remote_preference any",
                  "Industries wanted: any": "industries_wanted any", "Lives in: not given": "current_location not given"}


def blank_shares(prompts: list[str]) -> dict:
    n = len(prompts)
    prof = [prompt_parts(p)[1] for p in prompts]
    return {v: {"rows": sum(k in x for x in prof), "share": round(sum(k in x for x in prof) / n, 4)}
            for k, v in PROFILE_BLANKS.items()} | {"rows": n}


def resume_shape(prompts: list[str]) -> dict:
    """Second line of the redacted resume: a 'City, ST' line (as every training resume's contact block leaves it)."""
    city = re.compile(r"^[A-Z][A-Za-z .'-]+, [A-Z]{2}$")
    sec = [prompt_parts(p)[0].split("\n")[1] if "\n" in prompt_parts(p)[0] else "" for p in prompts]
    return {"rows": len(prompts), "second_line_city": sum(bool(city.match(s.strip())) for s in sec)}


def posting_header_diffs(d: Path = FRESH) -> dict:
    """The fresh test's posting as the labelers saw it (rows.jsonl posting_text: extract_postings.posting_text) against
    the h build, both cut by build_prompt the way the model sees them."""
    posts = {p["key"]: p for p in jsonl(d / "postings.jsonl")}
    shown = {r["posting_id"]: r["posting_text"] for r in jsonl(d / "rows.jsonl")}
    diff, cut = [], 0
    for k, p in posts.items():
        a = prompt_parts(build_prompt("", "", posting_h(p)))[2]
        b = prompt_parts(build_prompt("", "", shown[k]))[2]
        if a != b:
            al, bl = a.split("\n"), b.split("\n")
            diff.append({"posting_id": k, "lines": [(x, y) for x, y in zip(bl, al) if x != y][:3]})
        from matcher.textclean import strip_markdown
        cut += len(strip_markdown(posting_h(p)).strip()) > 6000
    return {"postings": len(posts), "differ_after_build_prompt": len(diff), "examples": diff,
            "postings_cut_by_build_prompt_budget": cut}


def build(_: argparse.Namespace) -> None:
    par = parity()
    built = {m: {s: BUILDERS[s](m) for s in SETS} for m in MODELS}
    for m in MODELS:
        for s in SETS:
            ids = [r["row_id"] for r in built[m][s]]
            assert len(ids) == len(set(ids)), (m, s)
            d = OUT / m
            d.mkdir(parents=True, exist_ok=True)
            with (d / f"{s}.jsonl").open("w") as f:
                for r in built[m][s]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fmt = {s: {"rows": len(built["h"][s]),
               "h_vs_d3_prompts_differ": sum(a["prompt"] != b["prompt"] for a, b in zip(built["h"][s], built["d3"][s])),
               "h_vs_base_prompts_differ": sum(a["prompt"] != b["prompt"] for a, b in zip(built["h"][s], built["base"][s]))}
           for s in SETS}
    fit, mix = csv_prompts(TRAIN_FIT), csv_prompts(TRAIN_MIX3)
    fresh = [r["prompt"] for r in built["h"]["fresh_test"]]
    rep = {
        "what": "fit_eval_inputs.py build: prompts for h, d3 and base on dev_fit and the fresh test",
        "parity": par,
        "format": fmt,
        "instruction_same": len({p.split("\nRESUME:\n", 1)[0] for p in fit + mix + fresh}) == 1,
        "blank_profile_fields": {"train_fit.csv": blank_shares(fit), "train_mix3.csv": blank_shares(mix),
                                 "fresh_test": blank_shares(fresh),
                                 "dev_fit": blank_shares([r["prompt"] for r in built["h"]["dev_fit"]])},
        "resume_second_line": {"train_fit.csv": resume_shape(fit), "train_mix3.csv": resume_shape(mix),
                               "fresh_test": resume_shape(fresh)},
        "fresh_posting_vs_labeler_text": posting_header_diffs(),
        "files": {f"{m}/{s}.jsonl": sha256(OUT / m / f"{s}.jsonl") for m in MODELS for s in SETS},
    }
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def h2_prompts_are_train_fit_rows() -> dict:
    """This cycle's runs (h2n: train_fit2_nat.csv, h2b: train_fit2_bal.csv) train on prompts taken unchanged from
    train_fit.csv (build_fit_train2 changes completions only), so they are scored on h's build."""
    fit = set(csv_prompts(TRAIN_FIT))
    out = {}
    for name in ("train_fit2_nat.csv", "train_fit2_bal.csv"):
        p = FIT / name
        if p.exists():
            pr = csv_prompts(p)
            out[name] = {"rows": len(pr), "prompts_that_are_train_fit_rows": sum(x in fit for x in pr)}
            if out[name]["prompts_that_are_train_fit_rows"] != len(pr):
                raise SystemExit(f"{name}: not every prompt is a train_fit.csv row; h2 runs need their own build")
    return out


def build2(_: argparse.Namespace) -> None:
    """Fresh test 2's prompts (eval/protocol_amendment_2.json), built from its sealed rows before any model output on
    it: data/built/fit/eval_inputs/{h,d3,base}/fresh_test2.jsonl and eval_inputs/report_fresh_test2.json. Refuses to
    overwrite a file that exists. The h2 runs take h's build (h2_prompts_are_train_fit_rows)."""
    targets = [OUT / m / "fresh_test2.jsonl" for m in MODELS] + [OUT / "report_fresh_test2.json"]
    if any(t.exists() for t in targets):
        raise SystemExit(f"{[str(t.relative_to(ROOT)) for t in targets if t.exists()]} exist: fresh test 2's prompts are built once")
    par = parity()
    h2 = h2_prompts_are_train_fit_rows()
    built = {m: fresh_rows(m, FRESH2) for m in MODELS}
    for m in MODELS:
        ids = [r["row_id"] for r in built[m]]
        assert len(ids) == len(set(ids)), m
        (OUT / m).mkdir(parents=True, exist_ok=True)
        with (OUT / m / "fresh_test2.jsonl").open("w") as f:
            for r in built[m]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fit, mix = csv_prompts(TRAIN_FIT), csv_prompts(TRAIN_MIX3)
    fresh = [r["prompt"] for r in built["h"]]
    rep = {
        "what": "fit_eval_inputs.py build2: prompts for h (and this cycle's h2 runs), d3 and base on fresh test 2",
        "parity": par,
        "h2_runs_use_h_build": h2,
        "format": {"rows": len(fresh),
                   "h_vs_d3_prompts_differ": sum(a["prompt"] != b["prompt"] for a, b in zip(built["h"], built["d3"])),
                   "h_vs_base_prompts_differ": sum(a["prompt"] != b["prompt"] for a, b in zip(built["h"], built["base"]))},
        "instruction_same": len({p.split("\nRESUME:\n", 1)[0] for p in fit + mix + fresh}) == 1,
        "blank_profile_fields": {"fresh_test2": blank_shares(fresh)},
        "resume_second_line": {"fresh_test2": resume_shape(fresh)},
        "fresh_posting_vs_labeler_text": posting_header_diffs(FRESH2),
        "files": {f"{m}/fresh_test2.jsonl": sha256(OUT / m / "fresh_test2.jsonl") for m in MODELS},
    }
    (OUT / "report_fresh_test2.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k != "parity"}, indent=1, ensure_ascii=False))


def check_fresh3_inputs(d: Path = FRESH3) -> dict:
    """The files build3 reads (rows, postings, gold) are the sealed ones: the manifest's files and the gold manifest's
    gold_sha256. Cards are not read here (amendment 3: only the eligibility recompute at scoring reads them)."""
    files = json.loads(FRESH3_MANIFEST.read_text())["files"]
    got = {}
    for name in ("rows.jsonl", "postings.jsonl"):
        key = f"data/built/fresh_test3/{name}"
        got[key] = sha256(d / name)
        if got[key] != files[key]:
            raise SystemExit(f"{key} is not the file eval/fresh_test3_manifest.json sealed")
    got["data/built/fresh_test3/gold.jsonl"] = sha256(d / "gold.jsonl")
    if got["data/built/fresh_test3/gold.jsonl"] != json.loads(FRESH3_GOLD_MANIFEST.read_text())["gold_sha256"]:
        raise SystemExit("data/built/fresh_test3/gold.jsonl is not the gold eval/fresh_test3_gold_manifest.json pins")
    return got


def build3(_: argparse.Namespace) -> None:
    """Fresh test 3's prompts (eval/protocol_amendment_3.json), built from its sealed rows before any model output on
    it, with the same builders as fresh tests 1 and 2 (h's format for h, the h2 runs and base; d3's for d3):
    data/built/fit/eval_inputs/{h,d3,base}/fresh_test3.jsonl and eval_inputs/report_fresh_test3.json. Refuses to
    overwrite a file that exists; touches no fresh test 1 or 2 file."""
    targets = [OUT / m / "fresh_test3.jsonl" for m in MODELS] + [OUT / "report_fresh_test3.json"]
    if any(t.exists() for t in targets):
        raise SystemExit(f"{[str(t.relative_to(ROOT)) for t in targets if t.exists()]} exist: fresh test 3's prompts are built once")
    pins = check_fresh3_inputs(FRESH3)
    par = parity()
    h2 = h2_prompts_are_train_fit_rows()
    built = {m: fresh_rows(m, FRESH3) for m in MODELS}
    for m in MODELS:
        ids = [r["row_id"] for r in built[m]]
        assert len(ids) == len(set(ids)), m
        (OUT / m).mkdir(parents=True, exist_ok=True)
        with (OUT / m / "fresh_test3.jsonl").open("w") as f:
            for r in built[m]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fit, mix = csv_prompts(TRAIN_FIT), csv_prompts(TRAIN_MIX3)
    fresh = [r["prompt"] for r in built["h"]]
    rep = {
        "what": "fit_eval_inputs.py build3: prompts for h (and h-format runs), d3 and base on fresh test 3",
        "inputs_sha256": pins,
        "parity": par,
        "h2_runs_use_h_build": h2,
        "format": {"rows": len(fresh),
                   "h_vs_d3_prompts_differ": sum(a["prompt"] != b["prompt"] for a, b in zip(built["h"], built["d3"])),
                   "h_vs_base_prompts_differ": sum(a["prompt"] != b["prompt"] for a, b in zip(built["h"], built["base"]))},
        "instruction_same": len({p.split("\nRESUME:\n", 1)[0] for p in fit + mix + fresh}) == 1,
        "blank_profile_fields": {"fresh_test3": blank_shares(fresh)},
        "resume_second_line": {"fresh_test3": resume_shape(fresh)},
        "fresh_posting_vs_labeler_text": posting_header_diffs(FRESH3),
        "files": {f"{m}/fresh_test3.jsonl": sha256(OUT / m / "fresh_test3.jsonl") for m in MODELS},
    }
    (OUT / "report_fresh_test3.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k != "parity"}, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------- cycle 3: run B's format on fresh tests 1-3


C3 = FIT / "cycle3"
B_DIR = OUT / "B"
B_REPORT = OUT / "report_cycle3_B.json"
B_SETS = {"fresh_test": FRESH, "fresh_test2": FRESH2, "fresh_test3": FRESH3}
C3_PINS = ROOT / "eval/cycle3_eval_inputs.json"
H_PINS = {"fresh_test": (ROOT / "eval/fit_eval_hashes.json", "sha256", "data/built/fit/eval_inputs/h/fresh_test.jsonl"),
          "fresh_test2": (OUT / "report_fresh_test2.json", "files", "h/fresh_test2.jsonl"),
          "fresh_test3": (OUT / "report_fresh_test3.json", "files", "h/fresh_test3.jsonl")}


def prompt_b_row(r: dict, system: str) -> str:
    """Run B's prompt for a row built by build_fresh_test's pipeline (a fresh test's or cycle 3's rows.jsonl)."""
    import build_fit_train3 as b3
    return b3.prompt_b(r["resume_text"], render_profile(CandidateCard.from_dict(r["profile"])), r["posting_text"], system)


def fresh_rows_b(d: Path, system: str) -> list[dict]:
    gold = {g["row_id"]: g["gold"] for g in jsonl(d / "gold.jsonl")}
    return [row(r["row_id"], r["posting_id"], prompt_b_row(r, system), gold[r["row_id"]]) for r in jsonl(d / "rows.jsonl")]


def b_parity(system: str, c3: Path = C3) -> dict:
    """Every kept new row of train_B.csv (train3_rows.jsonl: row_id -> csv_row) rebuilt by prompt_b_row from
    cycle3/rows.jsonl equals its CSV row byte for byte (encoded UTF-8); the rendered profile equals the row's own
    profile_text. Any mismatch stops the build."""
    side = {x["row_id"]: x for x in jsonl(c3 / "train3_rows.jsonl") if x.get("kept")}
    csv_b = csv_prompts(c3 / "train_B.csv")
    n = same = prof = 0
    bad = []
    for r in jsonl(c3 / "rows.jsonl"):
        x = side.get(r["row_id"])
        if x is None or x["source"] != "new":
            continue
        n += 1
        prof += render_profile(CandidateCard.from_dict(r["profile"])) == r["profile_text"]
        if prompt_b_row(r, system).encode() == csv_b[x["csv_row"]].encode():
            same += 1
        elif len(bad) < 3:
            bad.append(r["row_id"])
    want = sum(x["source"] == "new" for x in side.values())
    rep = {"train_B.csv rows": len(csv_b), "kept new rows": want, "rebuilt": n, "byte_equal": same,
           "profile_text == render_profile": prof, "first_mismatches": bad}
    if not n or n != want or same != n or prof != n:
        raise SystemExit(f"B parity failed: {rep}")
    return rep


def check_b_inputs() -> dict:
    """Each test's rows and gold are the pinned ones (fresh test 1: eval/fit_eval_hashes.json; 2: its manifest and
    gold manifest; 3: its manifest and gold manifest), checked by fit_eval_score's own checks."""
    import fit_eval_score as F
    F.check_hashes(["data/built/fresh_test/rows.jsonl", "data/built/fresh_test/gold.jsonl"])
    F.check_fresh2()
    check_fresh3_inputs(FRESH3)
    return {f"data/built/{d.name}/{n}": sha256(d / n) for d in B_SETS.values() for n in ("rows.jsonl", "gold.jsonl")}


def b_vs_posting_h(d: Path, system: str) -> dict:
    """How many B prompts would change if the posting came from h's posting builder (posting_h on postings.jsonl)
    instead of the row's posting_text (reported; the difference is the location header of a few pool postings)."""
    import build_fit_train3 as b3
    posts = {p["key"]: p for p in jsonl(d / "postings.jsonl")}
    rows = jsonl(d / "rows.jsonl")
    diff = [r["row_id"] for r in rows if prompt_b_row(r, system) != b3.prompt_b(
        r["resume_text"], render_profile(CandidateCard.from_dict(r["profile"])), posting_h(posts[r["posting_id"]]), system)]
    return {"rows": len(rows), "prompts_that_differ": len(diff), "row_ids": diff}


def build4(_: argparse.Namespace) -> None:
    """Run B's prompts on fresh tests 1-3 (eval/protocol_amendment_5.json runs.B, 5b): B/fresh_test{,2,3}.jsonl and
    report_cycle3_B.json. Refuses to overwrite; touches no h, d3 or base file and no fresh test 4 file."""
    import build_fit_train3 as b3
    targets = [B_DIR / f"{s}.jsonl" for s in B_SETS] + [B_REPORT, C3_PINS]
    if any(t.exists() for t in targets):
        raise SystemExit(f"{[str(t.relative_to(ROOT)) for t in targets if t.exists()]} exist: B's prompts are built once")
    pins = check_b_inputs()
    system = b3.system_b()
    par = b_parity(system)
    built = {s: fresh_rows_b(d, system) for s, d in B_SETS.items()}
    for s, rows in built.items():
        ids = [r["row_id"] for r in rows]
        assert len(ids) == len(set(ids)), s
        if not all(r["prompt"].startswith(system + "\n\nRESUME:\n") and r["prompt"].endswith("\n\nAnswer:") for r in rows):
            raise SystemExit(f"{s}: a B prompt is not rubric + RESUME: ... Answer:")
    B_DIR.mkdir(parents=True, exist_ok=True)
    for s, rows in built.items():
        with (B_DIR / f"{s}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lens = {s: sorted(len(r["prompt"]) for r in rows) for s, rows in built.items()}
    rep = {
        "what": "fit_eval_inputs.py build4: cycle-3 run B's prompts on fresh tests 1-3 (practice sets); run A is "
                "scored on h's files (eval_inputs/h/fresh_test{,2,3}.jsonl, report*.json pins)",
        "format": "build_fit_train3.prompt_b(resume_text, render_profile(profile), posting_text, system_b()): the "
                  "rubric, a blank line, then h's layout without INSTRUCTION (RESUME:/CANDIDATE PROFILE:/JOB POSTING:/"
                  "Answer:) with the whole posting; one user turn, no system message (score with the chat framing)",
        "system_b_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "inputs_sha256": pins,
        "train_B_parity": par,
        "posting_text_vs_posting_h": {s: b_vs_posting_h(d, system) for s, d in B_SETS.items()},
        "rows": {s: len(rows) for s, rows in built.items()},
        "prompt_chars": {s: {"min": v[0], "median": v[len(v) // 2], "max": v[-1]} for s, v in lens.items()},
        "files": {f"B/{s}.jsonl": sha256(B_DIR / f"{s}.jsonl") for s in B_SETS},
    }
    B_REPORT.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    write_c3_pins(rep)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def write_c3_pins(rep: dict) -> dict:
    """eval/cycle3_eval_inputs.json: the prompt files each cycle-3 run is scored on, by sha256 (A on h's files, as
    their own reports pin them and as they are now; B on build4's), committed before any A or B output."""
    a = {}
    for s, (f, key, name) in H_PINS.items():
        pinned, now = json.loads(f.read_text())[key][name], sha256(OUT / "h" / f"{s}.jsonl")
        if pinned != now:
            raise SystemExit(f"eval_inputs/h/{s}.jsonl is not the file {f.name} pins")
        a[f"data/built/fit/eval_inputs/h/{s}.jsonl"] = now
    pins = {"what": "the prompt files cycle-3 runs A and B are scored on for fresh tests 1-3 (scripts/fit_eval_inputs.py "
                    "build4); A takes h's format, B the rubric-at-top format of train_B.csv",
            "A": a, "B": {f"data/built/fit/eval_inputs/{k}": v for k, v in rep["files"].items()},
            "B_format": rep["format"], "system_b_sha256": rep["system_b_sha256"],
            "train_B_parity": rep["train_B_parity"], "inputs_sha256": rep["inputs_sha256"],
            "posting_text_vs_posting_h": {k: {"rows": v["rows"], "prompts_that_differ": v["prompts_that_differ"]}
                                          for k, v in rep["posting_text_vs_posting_h"].items()},
            "report": str(B_REPORT.relative_to(ROOT)), "report_sha256": sha256(B_REPORT)}
    C3_PINS.write_text(json.dumps(pins, indent=1, ensure_ascii=False) + "\n")
    return pins


# ---------------------------------------------------------------- fresh test 4 (sealed, scored once)


FRESH4 = ROOT / "data/built/fresh_test4"
FRESH4_MANIFEST = ROOT / "eval/fresh_test4_manifest.json"
FRESH4_DATA_HASHES = ROOT / "eval/fresh_test4_data_hashes.json"
FRESH4_GOLD_MANIFEST = ROOT / "eval/fresh_test4_gold_manifest.json"
FRESH4_PINS = ROOT / "eval/fresh_test4_eval_inputs.json"
FRESH4_REPORT = OUT / "report_fresh_test4.json"
FRESH4_MODELS = ("h", "base", "B")  # h's format for h and base (posting fitted to 6,000 characters), B's for run B


def check_fresh4_inputs(d: Path = FRESH4) -> dict:
    """Fresh test 4's files are the sealed ones before a prompt is built: every hash in its manifest's files and in
    eval/fresh_test4_data_hashes.json, the gold its gold manifest pins, and those three files committed unchanged."""
    import fit_eval_score as F
    for f in (FRESH4_MANIFEST, FRESH4_DATA_HASHES, FRESH4_GOLD_MANIFEST):
        if not F.committed(f):
            raise SystemExit(f"{f.relative_to(ROOT)} is not committed unchanged")
    pins = {**json.loads(FRESH4_MANIFEST.read_text())["files"], **json.loads(FRESH4_DATA_HASHES.read_text())["sha256"]}
    for n, h in pins.items():
        if sha256(ROOT / n) != h:
            raise SystemExit(f"{n} is not the file fresh test 4's manifests pin")
    if sha256(d / "gold.jsonl") != json.loads(FRESH4_GOLD_MANIFEST.read_text())["gold_sha256"]:
        raise SystemExit("data/built/fresh_test4/gold.jsonl is not the gold eval/fresh_test4_gold_manifest.json pins")
    return {f"data/built/fresh_test4/{n}": sha256(d / n) for n in ("rows.jsonl", "postings.jsonl", "gold.jsonl")}


def labeler_view_check(d: Path = FRESH4) -> dict:
    """Every row's posting_text, resume_text and profile_text equal its blind label batch's posting, resume and profile
    (what the labelers saw), and render_profile(profile) equals profile_text (what B's prompt renders)."""
    rows = {r["row_id"]: r for r in jsonl(d / "rows.jsonl")}
    seen: dict[str, dict] = {}
    for f in sorted((d / "label_batches").glob("batch_*.jsonl")):
        for b in jsonl(f):
            seen.setdefault(b["row_id"], b)
    out = {"rows": len(rows), "rows_in_label_batches": len(set(seen) & set(rows)), "posting_equal": 0, "resume_equal": 0,
           "profile_equal": 0, "render_profile_equal": 0}
    for rid, r in rows.items():
        b = seen.get(rid, {})
        out["posting_equal"] += b.get("posting") == r["posting_text"]
        out["resume_equal"] += b.get("resume") == r["resume_text"]
        out["profile_equal"] += b.get("profile") == r["profile_text"]
        out["render_profile_equal"] += render_profile(CandidateCard.from_dict(r["profile"])) == r["profile_text"]
    if any(v != len(rows) for k, v in out.items() if k != "rows"):
        raise SystemExit(f"fresh test 4 rows differ from what the labelers saw: {out}")
    return out


def build5(_: argparse.Namespace) -> None:
    """Fresh test 4's prompts (eval/protocol_amendment_5.json fresh_test_4, 5b), built from its sealed rows before any
    model output on it: h/ and base/fresh_test4.jsonl in h's format (fresh_rows: build_prompt, posting from
    posting_h fitted to 6,000 characters), B/fresh_test4.jsonl in run B's format (fresh_rows_b: the rubric at the top
    of the user turn, the posting exactly as the labelers saw it), report_fresh_test4.json and eval/fresh_test4_eval_inputs.json
    (the sha256 of each, committed before any inference). Refuses to overwrite; touches no other test's file."""
    import build_fit_train3 as b3
    targets = [OUT / m / "fresh_test4.jsonl" for m in FRESH4_MODELS] + [FRESH4_REPORT, FRESH4_PINS]
    if any(t.exists() for t in targets):
        raise SystemExit(f"{[str(t.relative_to(ROOT)) for t in targets if t.exists()]} exist: fresh test 4's prompts are built once")
    inputs = check_fresh4_inputs()
    view = labeler_view_check()
    system = b3.system_b()
    par_b = b_parity(system)
    par_h = parity()
    built = {"h": fresh_rows("h", FRESH4), "base": fresh_rows("base", FRESH4), "B": fresh_rows_b(FRESH4, system)}
    for m, rows in built.items():
        ids = [r["row_id"] for r in rows]
        if len(ids) != len(set(ids)) or len(ids) != json.loads((ROOT / "eval/fresh_test4_manifest.json").read_text())["n_rows"]:
            raise SystemExit(f"{m}: {len(ids)} rows, {len(set(ids))} distinct; the manifest says otherwise")
    if any(a["prompt"] != b["prompt"] for a, b in zip(built["h"], built["base"])):
        raise SystemExit("h and base prompts differ: base takes h's format")
    if not all(r["prompt"].startswith(system + "\n\nRESUME:\n") and r["prompt"].endswith("\n\nAnswer:") for r in built["B"]):
        raise SystemExit("a B prompt is not rubric + RESUME: ... Answer:")
    from matcher.pairs import INSTRUCTION
    if not all(r["prompt"].startswith(INSTRUCTION + "\nRESUME:\n") for r in built["h"]):
        raise SystemExit("an h prompt does not open with INSTRUCTION")
    for m, rows in built.items():
        (OUT / m).mkdir(parents=True, exist_ok=True)
        with (OUT / m / "fresh_test4.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lens = {m: sorted(len(r["prompt"]) for r in rows) for m, rows in built.items()}
    h_post = [p.split("\nJOB POSTING:\n", 1)[1] for p in (r["prompt"] for r in built["h"])]
    files = {f"data/built/fit/eval_inputs/{m}/fresh_test4.jsonl": sha256(OUT / m / "fresh_test4.jsonl") for m in FRESH4_MODELS}
    rep = {
        "what": "fit_eval_inputs.py build5: prompts for B (run B's format), h and base (h's format) on sealed fresh test 4",
        "inputs_sha256": inputs,
        "labeler_view": view,
        "train_B_parity": par_b,
        "h_parity": par_h,
        "rows": {m: len(rows) for m, rows in built.items()},
        "prompt_chars": {m: {"min": v[0], "median": v[len(v) // 2], "max": v[-1]} for m, v in lens.items()},
        "h_posting_section_chars_max": max(len(x) for x in h_post),
        "B_posting_text_vs_posting_h": b_vs_posting_h(FRESH4, system),
        "blank_profile_fields": {"fresh_test4": blank_shares([r["prompt"] for r in built["h"]])},
        "files": files,
    }
    FRESH4_REPORT.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    pins = {"what": "the prompt files fresh test 4 is scored on (scripts/fit_eval_inputs.py build5), pinned before any "
                    "inference on it: run B on B/fresh_test4.jsonl (rubric at the top of the user turn, the posting as "
                    "the labelers saw it); h and the untrained base on h's format (posting fitted to 6,000 characters)",
            "B": {k: v for k, v in files.items() if "/B/" in k},
            "h": {k: v for k, v in files.items() if "/h/" in k},
            "base": {k: v for k, v in files.items() if "/base/" in k},
            "B_format": "build_fit_train3.prompt_b(resume_text, render_profile(profile), posting_text, system_b())",
            "h_format": "pairs.build_prompt(resume_text, render_profile(profile), build_fit_train.posting_text_of(posting))",
            "system_b_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "inputs_sha256": inputs, "labeler_view": view, "train_B_parity": par_b,
            "report": str(FRESH4_REPORT.relative_to(ROOT)), "report_sha256": sha256(FRESH4_REPORT)}
    FRESH4_PINS.write_text(json.dumps(pins, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("h_parity",)}, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------- fresh test 5 and candidate C (amendments 6, 6b)


FRESH5 = ROOT / "data/built/fresh_test5"
FRESH5_MANIFEST = ROOT / "eval/fresh_test5_manifest.json"
FRESH5_DATA_HASHES = ROOT / "eval/fresh_test5_data_hashes.json"
FRESH5_GOLD_MANIFEST = ROOT / "eval/fresh_test5_gold_manifest.json"
FRESH5_PINS = ROOT / "eval/fresh_test5_eval_inputs.json"
FRESH5_REPORT = OUT / "report_fresh_test5.json"
C_DIR = OUT / "C"
C_SETS = {"fresh_test": FRESH, "fresh_test2": FRESH2, "fresh_test3": FRESH3, "fresh_test4": FRESH4,
          "fresh_test5": FRESH5}
CLAR = "- A required item the resume names but doesn't show doing is not met."
SYSTEM_C_SHA = "eb304433e80f6a71d43d5404cec86131f5094a32487472f4494ae8361c22cc9f"


def check_fresh5_inputs(d: Path = FRESH5) -> dict:
    """Fresh test 5's files are the sealed ones: every hash in its manifest's files and in its data hashes, the gold
    its gold manifest pins, and those three files committed unchanged."""
    import fit_eval_score as F
    for f in (FRESH5_MANIFEST, FRESH5_DATA_HASHES, FRESH5_GOLD_MANIFEST):
        if not F.committed(f):
            raise SystemExit(f"{f.relative_to(ROOT)} is not committed unchanged")
    pins = {**json.loads(FRESH5_MANIFEST.read_text())["files"], **json.loads(FRESH5_DATA_HASHES.read_text())["sha256"]}
    for n, h in pins.items():
        if sha256(ROOT / n) != h:
            raise SystemExit(f"{n} is not the file fresh test 5's manifests pin")
    if sha256(d / "gold.jsonl") != json.loads(FRESH5_GOLD_MANIFEST.read_text())["gold_sha256"]:
        raise SystemExit("data/built/fresh_test5/gold.jsonl is not the gold eval/fresh_test5_gold_manifest.json pins")
    return {f"data/built/fresh_test5/{n}": sha256(d / n) for n in ("rows.jsonl", "postings.jsonl", "gold.jsonl")}


def build6(_: argparse.Namespace) -> None:
    """Prompts for fresh test 5 and for candidate C (eval/protocol_amendment_6b.json eval_inputs), before any model
    output on them: B/fresh_test5.jsonl in run B's format (B and B-colab), C/fresh_test{,2,3,4,5}.jsonl in C's format
    (B's prompt with C's system: B's rubric + the clarification line; every C prompt must equal C's system + the B
    prompt of the same row after B's system), report_fresh_test5.json and eval/fresh_test5_eval_inputs.json pinning
    every prompt file B, B-colab and C are scored on for tests 1-5. Refuses to overwrite."""
    import build_fit_train3 as b3
    targets = [B_DIR / "fresh_test5.jsonl", *(C_DIR / f"{s}.jsonl" for s in C_SETS), FRESH5_REPORT, FRESH5_PINS]
    if any(t.exists() for t in targets):
        raise SystemExit(f"{[str(t.relative_to(ROOT)) for t in targets if t.exists()]} exist: built once")
    inputs5 = check_fresh5_inputs()
    inputs4 = check_fresh4_inputs()
    inputs123 = check_b_inputs()
    view5 = labeler_view_check(FRESH5)
    sb = b3.system_b()
    sc = sb + "\n" + CLAR
    if hashlib.sha256(sc.encode()).hexdigest() != SYSTEM_C_SHA:
        raise SystemExit("C's system differs from amendment 6b's")
    par_b = b_parity(sb)
    b5 = fresh_rows_b(FRESH5, sb)
    if len(b5) != json.loads(FRESH5_MANIFEST.read_text())["n_rows"] or len({r["row_id"] for r in b5}) != len(b5):
        raise SystemExit("fresh test 5: row count differs from the manifest")
    if not all(r["prompt"].startswith(sb + "\n\nRESUME:\n") and r["prompt"].endswith("\n\nAnswer:") for r in b5):
        raise SystemExit("a B prompt is not rubric + RESUME: ... Answer:")
    # B's existing files for tests 1-4 must be the pinned ones and must equal a fresh build
    b_old = {}
    pinned = {**json.loads(C3_PINS.read_text())["B"], **json.loads(FRESH4_PINS.read_text())["B"]}
    for s in ("fresh_test", "fresh_test2", "fresh_test3", "fresh_test4"):
        p = B_DIR / f"{s}.jsonl"
        if sha256(p) != pinned[f"data/built/fit/eval_inputs/B/{s}.jsonl"]:
            raise SystemExit(f"{p.relative_to(ROOT)} is not the pinned B file")
        b_old[s] = jsonl(p)
        if [r["prompt"] for r in b_old[s]] != [r["prompt"] for r in fresh_rows_b(C_SETS[s], sb)]:
            raise SystemExit(f"{s}: B's pinned prompts differ from a rebuild")
    b_all = {**b_old, "fresh_test5": b5}
    c_all = {s: fresh_rows_b(d, sc) for s, d in C_SETS.items()}
    for s in C_SETS:
        for rb, rc in zip(b_all[s], c_all[s], strict=True):
            if rb["row_id"] != rc["row_id"] or rc["prompt"] != sc + rb["prompt"][len(sb):] or \
                    {k: v for k, v in rb.items() if k != "prompt"} != {k: v for k, v in rc.items() if k != "prompt"}:
                raise SystemExit(f"{s} {rb['row_id']}: C's prompt is not C's system + B's prompt body")
    with (B_DIR / "fresh_test5.jsonl").open("w") as f:
        for r in b5:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    C_DIR.mkdir(parents=True, exist_ok=True)
    for s, rows in c_all.items():
        with (C_DIR / f"{s}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    b_files = {f"data/built/fit/eval_inputs/B/{s}.jsonl": sha256(B_DIR / f"{s}.jsonl") for s in C_SETS}
    c_files = {f"data/built/fit/eval_inputs/C/{s}.jsonl": sha256(C_DIR / f"{s}.jsonl") for s in C_SETS}
    lens = sorted(len(r["prompt"]) for r in b5)
    rep = {
        "what": "fit_eval_inputs.py build6: run B's format on fresh test 5 (B and B-colab) and C's format on fresh "
                "tests 1-5 (amendments 6 and 6b)",
        "inputs_sha256": {**inputs123, **inputs4, **inputs5},
        "labeler_view_fresh_test5": view5,
        "train_B_parity": par_b,
        "c_vs_b": "every C prompt equals C's system + the same row's B prompt after B's system; every other field equal",
        "rows": {s: len(v) for s, v in c_all.items()},
        "B_fresh_test5_prompt_chars": {"min": lens[0], "median": lens[len(lens) // 2], "max": lens[-1]},
        "B_posting_text_vs_posting_h_fresh_test5": b_vs_posting_h(FRESH5, sb),
        "files": {**b_files, **c_files},
    }
    FRESH5_REPORT.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    pins = {"what": "the prompt files B, B-colab and C are scored on for fresh tests 1-5 (scripts/fit_eval_inputs.py "
                    "build6; amendment 6b eval_inputs), pinned before any inference of B-colab or C, and of B on test 5",
            "B": b_files, "Bcolab": b_files, "C": c_files,
            "B_format": "build_fit_train3.prompt_b(resume_text, render_profile(profile), posting_text, system_b())",
            "C_format": "the same with C's system = system_b() + '\\n' + the clarification line",
            "system_b_sha256": hashlib.sha256(sb.encode()).hexdigest(), "system_c_sha256": SYSTEM_C_SHA,
            "inputs_sha256": rep["inputs_sha256"], "train_B_parity": par_b,
            "report": str(FRESH5_REPORT.relative_to(ROOT)), "report_sha256": sha256(FRESH5_REPORT)}
    FRESH5_PINS.write_text(json.dumps(pins, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "inputs_sha256"}, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build6").set_defaults(fn=build6)
    sub.add_parser("build").set_defaults(fn=build)
    sub.add_parser("build2").set_defaults(fn=build2)
    sub.add_parser("build3").set_defaults(fn=build3)
    sub.add_parser("build4").set_defaults(fn=build4)
    sub.add_parser("build5").set_defaults(fn=build5)
    sub.add_parser("parity").set_defaults(fn=lambda a: print(json.dumps(parity(), indent=1)))
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
