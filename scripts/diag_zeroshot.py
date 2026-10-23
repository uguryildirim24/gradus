"""Cycle-3 diagnostic (eval/protocol_amendment_4.json): untrained Qwen3.5-4B and 9B, zero-shot with the labelers'
rubric and the full texts the labelers saw, on fresh tests 1-3 (practice sets since amendment 4).

  diag_zeroshot.py build
      one prompt per row of fresh tests 1, 2 and 3: a system message holding the labelers' rubric plus the answer
      format, a user message with the posting, profile and resume exactly as data/built/fresh_test*/label_batches/
      showed them (whole postings, no 6,000-character cut), Qwen's chat template with enable_thinking=False, rendered
      to text (local_eval --framing raw reads it as is). Checks that the label batches equal rows.jsonl, that the two
      models' tokenizers and templates are the same, and that Keep, Explore and Skip each begin with one token right
      after the rendered prompt (prompt + label tokenizes as prompt tokens + that one token, as local_eval's label
      read assumes). Tokenizer only, no model.
      -> data/models/diag_cycle3/zeroshot/prompts/zs_fresh_test{,2,3}.jsonl and prompts_manifest.json (sha256s,
         token-length stats, the check results)
  diag_zeroshot.py colab SESSION --base-model HF_ID --out DIR [--max-min 80] [--gen 8]
      colab_eval.py's own driver (watchdog-guarded L4 session, bf16, batch 1, one model per session) on the three
      prompt files, with every local_eval score call given --framing raw
  diag_zeroshot.py score
      fit_eval_score.py's own functions (load_fresh with the rules in effect, read_scores, systems, score_slice and
      its two-way bootstrap, DRAWS draws from SEED): zs4b and zs9b, raw and + H', per test and pooled, with paired
      differences against h + H', rules_full and the stored base run (our training instruction, postings cut to
      6,000 characters) -> eval/diag_cycle3_zeroshot.json. Never calls a write-once command or path.
  diag_zeroshot.py think-build / think-colab / think-score
      the optional 9B-with-thinking run on a stratified 150-row sample of fresh test 3 (vLLM on the L4), answer
      parsed after </think>. Not completed on 2026-10-21: vLLM 0.32.0 installed and loaded the architecture, but the
      session hit its 40-minute cap before the generations finished (eval/diag_cycle3_zeroshot.json, colab_sessions).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

OUT = ROOT / "data/models/diag_cycle3/zeroshot"
PROMPTS = OUT / "prompts"
TESTS = ("fresh_test", "fresh_test2", "fresh_test3")
MODELS = {"4b": "Qwen/Qwen3.5-4B", "9b": "Qwen/Qwen3.5-9B"}
SUMMARY = ROOT / "eval/diag_cycle3_zeroshot.json"
STORED = ROOT / "data/models/v4_eval"

RUBRIC = """You are labeling job-match rows. Each row has a job posting, a candidate's resume, and the candidate's profile (what they filled in on a form). Decide what a careful career advisor would tell this candidate about applying. Today is October 2026.

- Skip: something the posting states as a hard requirement rules them out. That covers work authorization (US citizens or permanent residents only; no visa sponsorship when the candidate will need it; no OPT or CPT when they are on F-1), student status (a posting for current students when the candidate has graduated, or one for graduates when they are still a student; compare the graduation year with today), a required degree level, or a required degree field they lack unless the posting accepts "related" or "equivalent", a required license they don't hold, an employment type, location or work mode their profile rules out, stated pay below their salary floor, or a skill or amount of experience the posting clearly calls required (must have, required, minimum qualifications) that the resume doesn't show.
- Keep: nothing rules them out and they meet the posting's stated requirements. Skills given as examples ("such as", "e.g.", "or similar", "one of", "experience with a language like") are met by any one of them or a close equivalent. Missing preferred, bonus or nice-to-have items do not block Keep unless most of them are missing. Duties describe the work; they are not requirements.
- Explore: worth a look but not a clean fit. Nothing rules them out, but they are about a year short on required experience, they miss most of the preferred items, a requirement is genuinely ambiguous, or the posting says too little to judge (a short teaser with no real requirements)."""
FORMAT = "Answer with one word first, Keep, Explore or Skip, then ' — ' and a one-line reason."
SYSTEM = RUBRIC + "\n\n" + FORMAT
LABELS = ("Keep", "Explore", "Skip")


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def user_message(b: dict) -> str:
    return f"Job posting:\n{b['posting']}\n\nCandidate profile:\n{b['profile']}\n\nResume:\n{b['resume']}"


def messages(b: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(b)}]


def label_batches(test: str) -> dict[str, dict]:
    out = {}
    for f in sorted((ROOT / "data/built" / test / "label_batches").glob("*.jsonl")):
        for b in jsonl(f):
            if b["row_id"] in out:
                raise SystemExit(f"{test}: {b['row_id']} in two label batches")
            out[b["row_id"]] = b
    return out


def tokenizers():
    from transformers import AutoTokenizer
    toks = {k: AutoTokenizer.from_pretrained(m) for k, m in MODELS.items()}
    return toks


def stats(xs: list[int]) -> dict:
    xs = sorted(xs)
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]  # noqa: E731
    return {"n": len(xs), "mean": round(statistics.mean(xs)), "median": q(0.5), "p90": q(0.9), "p99": q(0.99),
            "max": xs[-1], "min": xs[0]}


def cmd_build(a: argparse.Namespace) -> None:
    toks = tokenizers()
    tok = toks["4b"]
    PROMPTS.mkdir(parents=True, exist_ok=True)
    label_ids = {w: tok.encode(w, add_special_tokens=False) for w in LABELS}
    if not all(len(v) == 1 for v in label_ids.values()):
        raise SystemExit(f"a label is not one token on its own: {label_ids}")
    man = {"what": "zero-shot rubric prompts, cycle-3 diagnostic (eval/protocol_amendment_4.json)",
           "models": MODELS, "system_message": SYSTEM, "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
           "user_message": "Job posting:\\n{posting}\\n\\nCandidate profile:\\n{profile}\\n\\nResume:\\n{resume}, the "
                           "three fields exactly as data/built/<test>/label_batches/*.jsonl showed them to the labelers",
           "template": "tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, "
                       "enable_thinking=False); scored with local_eval --framing raw",
           "label_token_ids": {w: v[0] for w, v in label_ids.items()}, "files": {}, "tests": {}}
    all_lens: list[int] = []
    for test in TESTS:
        rows = jsonl(ROOT / "data/built" / test / "rows.jsonl")
        lb = label_batches(test)
        ei = {r["row_id"]: r for r in jsonl(ROOT / "data/built/fit/eval_inputs/base" / f"{test}.jsonl")}
        if set(lb) != {r["row_id"] for r in rows} or set(ei) != set(lb):
            raise SystemExit(f"{test}: label batches, rows.jsonl and eval_inputs hold different rows")
        diffs = Counter()
        for r in rows:
            b = lb[r["row_id"]]
            diffs["posting"] += b["posting"] != r["posting_text"]
            diffs["profile"] += b["profile"] != r["profile_text"]
            diffs["resume"] += b["resume"] != r["resume_text"]
        if any(diffs.values()):
            raise SystemExit(f"{test}: label batches differ from rows.jsonl {dict(diffs)}")
        out, lens, posting_chars, boundary_bad = [], [], [], Counter()
        for r in rows:
            b = lb[r["row_id"]]
            msgs = messages(b)
            texts = {k: t.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
                     for k, t in toks.items()}
            if texts["4b"] != texts["9b"]:
                raise SystemExit(f"{test} {r['row_id']}: the 4B and 9B templates render differently")
            text = texts["4b"]
            if not text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"):
                raise SystemExit(f"{test} {r['row_id']}: unexpected generation prompt {text[-60:]!r}")
            ids = tok.encode(text, add_special_tokens=False)
            for w in LABELS:
                for tail in (w, w + " — "):
                    full = tok.encode(text + tail, add_special_tokens=False)
                    if full[:len(ids)] != ids or full[len(ids)] != label_ids[w][0]:
                        boundary_bad[w] += 1
            e = ei[r["row_id"]]
            out.append({"row_id": r["row_id"], "pair_id": e["pair_id"], "posting_id": e["posting_id"],
                        "base_rq_id": e["base_rq_id"], "gold": e["gold"], "bucket": e["bucket"], "prompt": text,
                        "prompt_tokens": len(ids)})
            lens.append(len(ids))
            posting_chars.append(len(b["posting"]))
        if any(boundary_bad.values()):
            raise SystemExit(f"{test}: a label does not start with its single token after the prompt {dict(boundary_bad)}")
        p = PROMPTS / f"zs_{test}.jsonl"
        p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
        man["files"][str(p.relative_to(ROOT))] = sha256(p)
        man["tests"][test] = {"rows": len(out), "prompt_tokens": stats(lens), "posting_chars": stats(posting_chars),
                              "postings_over_6000_chars": sum(c > 6000 for c in posting_chars),
                              "label_batches_equal_rows_jsonl": True,
                              "label_first_token_check": {w: f"{len(out)}/{len(out)} rows: prompt + '{w}' and prompt + "
                                                             f"'{w} — ' tokenize as the prompt's tokens + id "
                                                             f"{label_ids[w][0]}" for w in LABELS}}
        all_lens += lens
        print(test, man["tests"][test]["prompt_tokens"], flush=True)
    man["pooled_prompt_tokens"] = stats(all_lens)
    man["tokenizers_same"] = True
    (OUT / "prompts_manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in man.items() if k in ("files", "pooled_prompt_tokens")}, indent=1))


def raw_framing(script: str) -> str:
    n = script.count("python local_eval.py score ")
    if not n:
        raise SystemExit("no local_eval score call in the jobs script")
    return script.replace("python local_eval.py score ", "python local_eval.py score --framing raw ")


def cmd_colab(a: argparse.Namespace) -> None:
    import colab_eval as ce
    orig = ce.jobs_script
    ce.jobs_script = lambda *x, **k: raw_framing(orig(*x, **k))
    argv = ["colab_eval.py", a.session, "--base-model", a.base_model, "--base", "--out", a.out, "--dtype", "bfloat16",
            "--batch", "1", "--max-min", str(a.max_min), "--gen", str(a.gen), "--max-tokens", "48"]
    for t in TESTS:
        argv += ["--dev", str(PROMPTS / f"zs_{t}.jsonl")]
    sys.argv = argv
    ce.main()


# ---------------------------------------------------------------- scoring (fit_eval_score's own functions)


def stored_score(test: str, tag: str) -> Path:
    return STORED / test / f"{tag}_run" / f"{tag}_{test}.json"


def zs_score(test: str, k: str) -> Path:
    return OUT / k / f"base_zs_{test}.json"


def compact(sl: dict | None, systems_: list[str], diffs: list[str]) -> dict | None:
    if sl is None:
        return None
    out = {"n": sl["n"], "postings": sl["postings"], "gold": sl["gold"], "systems": {}, "paired": {}}
    for s in systems_:
        m = sl["systems"][s]
        out["systems"][s] = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                             "keep_recall": m["per_label"]["Keep"]["recall"], "unsafe_keep": m["unsafe_keep"],
                             "pred": {w: m["per_label"][w]["pred"] for w in LABELS}}
    for d in diffs:
        p = sl["paired"][d]
        out["paired"][d] = {"macro_f1_diff": p["macro_f1_diff"], "accuracy_diff": p["accuracy_diff"],
                            "rows_fixed": p["rows_fixed"], "rows_broken": p["rows_broken"]}
    return out


def cmd_score(a: argparse.Namespace) -> None:
    import fit_eval_score as fe
    have = [k for k in MODELS if all(zs_score(t, k).exists() for t in TESTS)]
    if not have:
        raise SystemExit("no zero-shot score files yet")
    per_test, pooled_rows, pooled_models = {}, [], {}
    files = {}
    for test in TESTS:
        rows = fe.load_fresh(ROOT / "data/built" / test, strict_design=False)
        ids = [r["row_id"] for r in rows]
        paths = {"h": stored_score(test, "h"), "base": stored_score(test, "base")}
        paths |= {f"zs{k}": zs_score(test, k) for k in have}
        models = {t: fe.read_scores(p, ids) for t, p in paths.items()}
        files[test] = {t: {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for t, p in paths.items()}
        per_test[test] = (rows, models)
        pooled_rows += rows
        for t, lab in models.items():
            pooled_models.setdefault(t, {}).update(lab)
    show = [f"zs{k}{s}" for k in have for s in (" raw", "+H'")] + ["h+H'", "h raw", "base raw", "base+H'", "rules_full"]
    diffs = []
    for k in have:
        diffs += [f"zs{k} raw - h+H'", f"zs{k}+H' - h+H'", f"zs{k} raw - rules_full", f"zs{k}+H' - rules_full",
                  f"zs{k} raw - base raw", f"zs{k}+H' - base+H'"]
    if len(have) == 2:
        diffs += ["zs9b raw - zs4b raw", "zs9b+H' - zs4b+H'"]
    pairs = [tuple(d.split(" - ")) for d in diffs]
    res = {}
    for name, (rows, models) in [*per_test.items(), ("pooled", (pooled_rows, pooled_models))]:
        P = fe.systems(rows, models)
        res[name] = {}
        for sl, mask in (("all", [True] * len(rows)), ("eligibility_pass", [r["eligibility_pass"] for r in rows])):
            res[name][sl] = compact(fe.score_slice(rows, P, mask, pairs, fe.DRAWS, fe.SEED), show, diffs)
    pm = json.loads((OUT / "prompts_manifest.json").read_text())
    top1 = {}
    for k in have:
        c = Counter()
        for t in TESTS:
            for r in json.loads(zs_score(t, k).read_text())["rows"]:
                c[r["top1"] if r["top1"] in LABELS else "other"] += 1
        top1[f"zs{k}"] = dict(c)
    summary = {
        "what": "cycle-3 diagnostic (eval/protocol_amendment_4.json): untrained Qwen3.5-4B / 9B zero-shot with the "
                "labelers' rubric as the system message and the whole posting, profile and resume as the labelers saw "
                "them, on fresh tests 1-3 (practice sets); no write-once result was read for scoring or rewritten",
        "caveat": "against the stored base run this changes two things at once: the prompt (labelers' rubric in a system "
                  "message vs our training instruction) and the posting length (whole postings vs cut to 6,000 "
                  "characters), so the paired difference against base cannot say which of the two helped",
        "scoring": "fit_eval_score.load_fresh(strict_design=False: eligibility recomputed with the rules in effect), "
                   "read_scores (label re-read from the float32 first-token scores, ties Skip > Explore > Keep), "
                   "systems (TAG raw, TAG+H'), score_slice with the two-way bootstrap (DRAWS %d, SEED %d); pooled = the "
                   "three tests' rows together, postings as the bootstrap clusters" % (fe.DRAWS, fe.SEED),
        "metric_format": "[point, 2.5th percentile, 97.5th percentile]; unsafe_keep = gold Explore or Skip predicted "
                         "Keep (count)",
        "models": {f"zs{k}": MODELS[k] for k in have},
        "prompts": {"manifest": str((OUT / "prompts_manifest.json").relative_to(ROOT)),
                    "manifest_sha256": sha256(OUT / "prompts_manifest.json"), "files": pm["files"],
                    "system_sha256": pm["system_sha256"],
                    "prompt_tokens": {t: pm["tests"][t]["prompt_tokens"] for t in TESTS} | {
                        "pooled": pm["pooled_prompt_tokens"]}},
        "top1_token_is_a_label": top1,
        "score_files": files,
        "results": res,
    }
    if a.extra:
        summary |= json.loads(Path(a.extra).read_text())
    SUMMARY.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print_table(res, show, diffs)


def fmt(x: list[float]) -> str:
    return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"


def print_table(res: dict, show: list[str], diffs: list[str]) -> None:
    for name, by in res.items():
        for sl in ("all", "eligibility_pass"):
            d = by[sl]
            print(f"\n== {name} / {sl}: n {d['n']}, postings {d['postings']}, gold {d['gold']}")
            for s in show:
                m = d["systems"][s]
                print(f"  {s:12s} F1 {fmt(m['macro_f1'])}  acc {fmt(m['accuracy'])}  KeepR {m['keep_recall']}  "
                      f"unsafeK {m['unsafe_keep']}  pred {m['pred']}")
            for k in diffs:
                p = d["paired"][k]
                print(f"  {k:26s} dF1 {fmt(p['macro_f1_diff'])}  dacc {fmt(p['accuracy_diff'])}  "
                      f"+{p['rows_fixed']}/-{p['rows_broken']}")


# ---------------------------------------------------------------- optional: 9B with thinking, 150 rows of test 3

THINK = OUT / "9b_think"
THINK_QUOTA = {"Keep": 50, "Explore": 50}  # capped at what the test holds; Skip fills the rest to 150
THINK_N = 150
THINK_SEED = 20261021
THINK_VM = r'''
import json, sys, time
from vllm import LLM, SamplingParams


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    kw = dict(model=sys.argv[2], dtype="bfloat16", max_model_len=int(sys.argv[4]), gpu_memory_utilization=0.92,
              enable_prefix_caching=True, max_num_seqs=64, limit_mm_per_prompt={"image": 0, "video": 0})
    llm = None
    for extra in ({}, {"enforce_eager": True, "max_num_seqs": 16}, {"enforce_eager": True, "max_num_seqs": 16,
                                                                   "limit_mm_per_prompt": None}):
        try:
            llm = LLM(**{k: v for k, v in (kw | extra).items() if v is not None})
            print("LLM up with", extra, flush=True)
            break
        except Exception as e:
            print("LLM failed with", extra, repr(e)[:400], flush=True)
    if llm is None:
        sys.exit("vllm could not load the model")
    sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=int(sys.argv[5]), seed=0)
    t = time.time()
    outs = llm.generate([r["prompt"] for r in rows], sp)
    with open(sys.argv[3], "w") as f:
        for r, o in zip(rows, outs):
            c = o.outputs[0]
            f.write(json.dumps({"row_id": r["row_id"], "gold": r["gold"], "text": c.text, "tokens": len(c.token_ids),
                                "finish": c.finish_reason}) + "\n")
    print("generated", len(rows), "in", round(time.time() - t), "s", flush=True)


if __name__ == "__main__":  # vLLM spawns its engine process, which re-imports this file
    main()
'''


def cmd_think_build(a: argparse.Namespace) -> None:
    """A stratified sample of fresh test 3 by gold label (seeded): up to 50 Keeps and 50 Explores (test 3 holds 20
    Keeps, so all of them), Skips for the rest of the 150, rendered with enable_thinking=True. Not proportional, so the
    sample's numbers are read against zs9b raw on the same 150 rows, not against the full test."""
    import random
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS["9b"])
    rows = jsonl(ROOT / "data/built/fresh_test3/rows.jsonl")
    gold = {g["row_id"]: g["gold"] for g in jsonl(ROOT / "data/built/fresh_test3/gold.jsonl")}
    lb = label_batches("fresh_test3")
    rng = random.Random(THINK_SEED)
    by = {}
    for r in rows:
        by.setdefault(gold[r["row_id"]], []).append(r["row_id"])
    quota = {w: min(len(by[w]), n) for w, n in THINK_QUOTA.items()}
    quota["Skip"] = THINK_N - sum(quota.values())
    pick = sorted(x for w, v in by.items() for x in rng.sample(sorted(v), quota[w]))
    THINK.mkdir(parents=True, exist_ok=True)
    out = []
    for rid in pick:
        text = tok.apply_chat_template(messages(lb[rid]), add_generation_prompt=True, tokenize=False, enable_thinking=True)
        out.append({"row_id": rid, "gold": gold[rid], "prompt": text, "prompt_tokens": len(tok.encode(text))})
    p = THINK / "think_sample.jsonl"
    p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    (THINK / "think_sample_manifest.json").write_text(json.dumps(
        {"rows": len(out), "quota": quota, "seed": THINK_SEED, "sha256": sha256(p),
         "template": "enable_thinking=True, same system and user messages as the zero-shot prompts",
         "prompt_tokens": stats([x["prompt_tokens"] for x in out])}, indent=1) + "\n")
    print(quota, stats([x["prompt_tokens"] for x in out]))


def cmd_think_colab(a: argparse.Namespace) -> None:
    import subprocess
    import time
    import colab_eval as ce
    s = a.session
    started = time.time()
    print(ce.run([str(ce.GUARD), s, "L4", str(a.max_min), "20"], 1000), flush=True)
    try:
        ce.exec_code(s, f"import os\nos.makedirs({ce.M!r}, exist_ok=True)")
        setup = (f"#!/bin/bash\ncd {ce.M}\npip install -q vllm 2>&1 | tail -3\n"
                 "python -c 'import vllm; print(\"vllm\", vllm.__version__)' || exit 1\n"
                 f"hf download {MODELS['9b']} --local-dir base9b > /dev/null || exit 1\nnvidia-smi --query-gpu=name --format=csv\n")
        ce.start_detached(s, "setup", setup, "SETUP_DONE", "SETUP_FAIL")
        vm = THINK / "think_vm.py"
        vm.write_text(THINK_VM)
        for f in (THINK / "think_sample.jsonl", vm):
            ce.upload(s, f, f"{ce.M}/{f.name}")
        ce.wait_for(s, "SETUP_DONE", "SETUP_FAIL", 1200, "vllm install and 9B download")
        ce.sh(s, f"tail -n 4 {ce.M}/setup.log")
        print(f"setup took {time.time() - started:.0f} s", flush=True)
        job = (f"#!/bin/bash\ncd {ce.M}\npython think_vm.py think_sample.jsonl base9b think_out.jsonl {a.max_len} "
               f"{a.max_new}\n")
        ce.start_detached(s, "think", job, "THINK_DONE", "THINK_FAIL")
        while True:
            time.sleep(30)
            ls = ce.run([ce.COLAB, "ls", "-s", s, ce.M], 120, 3)
            if "THINK_DONE" in ls or "THINK_FAIL" in ls:
                break
        ce.sh(s, f"tail -n 15 {ce.M}/think.log")
        if "THINK_DONE" in ls:
            ce.run([ce.COLAB, "download", "-s", s, f"{ce.M}/think_out.jsonl", str(THINK / "think_out.jsonl")], 300, 3)
            print("downloaded", THINK / "think_out.jsonl", flush=True)
    finally:
        for f in ("setup.log", "think.log"):
            subprocess.run(["timeout", "120", ce.COLAB, "download", "-s", s, f"{ce.M}/{f}", str(THINK / f)],
                           capture_output=True)
        print(ce.run([ce.COLAB, "stop", "-s", s], 180), flush=True)


def parse_think(text: str) -> str | None:
    """The label after the thinking block: the first Keep / Explore / Skip word after </think>."""
    if "</think>" not in text:
        return None
    m = re.search(r"\b(Keep|Explore|Skip)\b", text.split("</think>", 1)[1])
    return m.group(1) if m else None


def cmd_think_score(a: argparse.Namespace) -> None:
    """The thinking 9B on its 150 rows beside zs9b raw (first-token, no thinking) on the same rows: macro-F1,
    accuracy, Keep recall, unsafe Keep, the paired difference (two-way bootstrap), and the rows with no </think> or
    no label after it (counted, scored as Skip, the tie-break's safest label)."""
    import numpy as np
    import fit_eval_score as fe
    outs = {r["row_id"]: r for r in jsonl(THINK / "think_out.jsonl")}
    rows = {r["row_id"]: r for r in fe.load_fresh(ROOT / "data/built/fresh_test3", strict_design=False)}
    ids = [r["row_id"] for r in jsonl(THINK / "think_sample.jsonl")]
    if set(outs) != set(ids):
        raise SystemExit("think_out.jsonl does not hold the sample's rows")
    zs = fe.read_scores(zs_score("fresh_test3", "9b"), list(rows))
    parsed = {i: parse_think(outs[i]["text"]) for i in ids}
    fails = Counter("no </think>" if "</think>" not in outs[i]["text"] else "no label" for i in ids if parsed[i] is None)
    think = [parsed[i] or "Skip" for i in ids]
    sub = [rows[i] for i in ids]
    g = fe.as_idx([r["gold"] for r in sub])
    dr = fe.two_way_draws([r["posting_id"] for r in sub], fe.DRAWS, fe.SEED)
    P = {"zs9b_think raw": fe.as_idx(think), "zs9b raw": fe.as_idx([zs[i] for i in ids])}
    from hybrid_rules import HP_CATS, hybrid
    P["zs9b_think+H'"] = fe.as_idx([hybrid(m, r["verdict"], HP_CATS, True) for m, r in zip(think, sub)])
    P["zs9b+H'"] = fe.as_idx([hybrid(zs[i], rows[i]["verdict"], HP_CATS, True) for i in ids])
    res = {"rows": len(ids), "gold": dict(Counter(r["gold"] for r in sub)), "parse_failures": dict(fails),
           "think_tokens": stats([outs[i]["tokens"] for i in ids]),
           "finish": dict(Counter(outs[i]["finish"] for i in ids)), "systems": {}, "paired": {}}
    for k, p in P.items():
        m = fe.metrics(g, p, dr)
        res["systems"][k] = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                             "keep_recall": m["per_label"]["Keep"]["recall"], "unsafe_keep": m["unsafe_keep"],
                             "pred": {w: m["per_label"][w]["pred"] for w in LABELS}}
    for a_, b_ in (("zs9b_think raw", "zs9b raw"), ("zs9b_think+H'", "zs9b+H'")):
        d = fe.paired(g, P[a_], P[b_], dr)
        res["paired"][f"{a_} - {b_}"] = {k: d[k] for k in ("macro_f1_diff", "accuracy_diff", "rows_fixed", "rows_broken")}
    man = json.loads((THINK / "think_sample_manifest.json").read_text())
    out = {"thinking_9b_sample": {"what": "Qwen3.5-9B with enable_thinking=True on a stratified sample of fresh test 3 "
                                          "(vLLM on a Colab L4, bf16), the label parsed as the first Keep / Explore / "
                                          "Skip after </think>",
                                  "sampling": "temperature 0.6, top_p 0.95, top_k 20, seed 0 (Qwen's thinking-mode "
                                              "settings; sampled, not greedy), up to %d new tokens" % a.max_new,
                                  "sample": man, "sample_file_sha256": sha256(THINK / "think_sample.jsonl"),
                                  "output_file": str((THINK / "think_out.jsonl").relative_to(ROOT)),
                                  "output_sha256": sha256(THINK / "think_out.jsonl"), **res}}
    (THINK / "think_result.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["thinking_9b_sample"], indent=1)[:4000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    c = sub.add_parser("colab")
    c.add_argument("session")
    c.add_argument("--base-model", required=True, choices=list(MODELS.values()))
    c.add_argument("--out", required=True)
    c.add_argument("--max-min", type=int, default=80)
    c.add_argument("--gen", type=int, default=8)
    c.set_defaults(fn=cmd_colab)
    s = sub.add_parser("score")
    s.add_argument("--extra", help="a json whose keys are added to the summary (the thinking run, sessions used)")
    s.set_defaults(fn=cmd_score)
    sub.add_parser("think-build").set_defaults(fn=cmd_think_build)
    tc = sub.add_parser("think-colab")
    tc.add_argument("session")
    tc.add_argument("--max-min", type=int, default=50)
    tc.add_argument("--max-len", type=int, default=8192)
    tc.add_argument("--max-new", type=int, default=4096)
    tc.set_defaults(fn=cmd_think_colab)
    ts = sub.add_parser("think-score")
    ts.add_argument("--max-new", type=int, default=4096)
    ts.set_defaults(fn=cmd_think_score)
    a = ap.parse_args()
    a.fn(a)
