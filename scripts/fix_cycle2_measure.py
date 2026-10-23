"""Fix cycle 2 on fresh test 1 (the selection set): what the work-mode rule and the v3 extractor change, against gold.

  fix_cycle2_measure.py [--out eval/fix_cycle2.json]

Reads data/built/fresh_test (rows, postings, the v2 cards.jsonl, the re-extracted card sets in ALTS, gold) and the stored
per-row model
labels in data/models/v4_eval/fresh_test/{h,d3,base}.jsonl. Loads no model, calls nothing, never reads fresh test 2.

"Old rules" are the rules before fix cycle 2, emulated exactly: the label with remote_preference "any" (the work-mode
check was the only one that read it), then Explore when the profile rules the mode out and the label was not Excluded.
Every card gets rules.mark_thin on the posting body, as fit_eval_score does.

Systems, as fit_eval_score defines them: rules_full, rules_skills_free, TAG+H' (hybrid_rules.hybrid with HP_CATS and the
salary cap). Point values only (no bootstrap): this is diagnosis on the selection set, not a scored result.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from hybrid_rules import HP_CATS, category, hybrid  # noqa: E402
from matcher.evalstats import WORD, macro_f1_of  # noqa: E402
from matcher.rules import label, mark_thin, mode_ruled_out, reasons_by_check, required_groups  # noqa: E402
from matcher.schema import CandidateCard, RequirementCard  # noqa: E402

FT = ROOT / "data/built/fresh_test"
PRED = ROOT / "data/models/v4_eval/fresh_test"
MODELS = ("h", "d3", "base")
SALARY_PREFIX = "pay tops out at"


def jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open() if line.strip()]


def verdict(c: CandidateCard, r: RequirementCard, old: bool = False) -> dict:
    """hybrid_rules.rule_verdict, under the new rules or the old ones (see the module doc)."""
    cc = replace(c, remote_preference="any") if old else c
    b, reasons, _ = label(cc, r)
    if old and b != "Excluded" and mode_ruled_out(c, r):
        b = "Explore"
    by = reasons_by_check(cc, r)
    cats = sorted({category(by.get(x), x) for x in reasons}) if b == "Excluded" else []
    sal = b == "Explore" and any(x.startswith(SALARY_PREFIX) for x in reasons)
    return {"rule": WORD[b], "hard_cats": cats, "salary_soft": sal}


def skills_free(r: RequirementCard) -> RequirementCard:
    return replace(r, must_skills=(), nice_skills=())


def systems(rows: list[dict], cards: dict[str, RequirementCard], old: bool) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"rules_full": [], "rules_skills_free": []}
    for m in MODELS:
        out[f"{m}+H'"] = []
    for r in rows:
        c, card = r["cand"], cards[r["posting_id"]]
        v = verdict(c, card, old)
        out["rules_full"].append(v["rule"])
        out["rules_skills_free"].append(verdict(c, skills_free(card), old)["rule"])
        for m in MODELS:
            out[f"{m}+H'"].append(hybrid(r[m], v, HP_CATS, True))
    return out


def metrics(gold: list[str], pred: list[str]) -> dict:
    keep = [p for g, p in zip(gold, pred) if g == "Keep"]
    return {"n": len(gold), "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(gold), 4),
            "macro_f1": round(macro_f1_of(gold, pred), 4),
            "keep_recall": round(sum(p == "Keep" for p in keep) / len(keep), 4) if keep else None}


def table(rows: list[dict], cols: dict[str, list[str]], mask: list[bool]) -> dict:
    gold = [r["gold"] for r, m in zip(rows, mask) if m]
    return {k: metrics(gold, [p for p, m in zip(v, mask) if m]) for k, v in cols.items()}


def agree(rows: list[dict], a: list[str], b: list[str], mask: list[bool]) -> dict:
    """On the masked rows: n, gold label mix, and how many a and b each get right."""
    idx = [i for i, m in enumerate(mask) if m]
    return {"rows": len(idx), "postings": len({rows[i]["posting_id"] for i in idx}),
            "gold": dict(Counter(rows[i]["gold"] for i in idx)),
            "before_agrees": sum(a[i] == rows[i]["gold"] for i in idx),
            "after_agrees": sum(b[i] == rows[i]["gold"] for i in idx)}


# The re-extractions compared against the v2 cards (scripts/extract_v3.py): a same-instruction rerun (noise floor) and the
# two v3 drafts, by file, with the blueprint hash each was made with.
ALTS = {"v2_rerun": ("cards_v2rerun.jsonl", "b2b2ed8e", "the v2 instruction sent again: run-to-run noise"),
        "v3_draft_a": ("cards_v3_0e7b4209.jsonl", "0e7b4209",
                       "first draft: experience areas excluded from must_skills too (commit 142dc8e)"),
        "v3_draft_b": ("cards_v3.jsonl", "d46a0614",
                       "second draft (matcher.blueprint.BLUEPRINT_V3): required experience areas kept as must_skills")}


def load() -> tuple[list[dict], dict, dict[str, dict], dict[str, list[str]]]:
    """rows, the v2 cards, each ALTS card set (a posting with no card there keeps its v2 card) and the postings that
    had none, by ALTS name."""
    posts = {p["key"]: p for p in jsonl(FT / "postings.jsonl")}
    body = {k: p.get("text") or "" for k, p in posts.items()}

    def cards(name: str) -> dict:
        return {r["id"]: mark_thin(RequirementCard.from_dict(r["card"]), body[r["id"]]) for r in jsonl(FT / name)}
    v2 = cards("cards.jsonl")
    alts, missing = {}, {}
    for k, (f, _, _) in ALTS.items():
        got = cards(f)
        missing[k] = sorted(set(v2) - set(got))
        alts[k] = {i: got.get(i, c) for i, c in v2.items()}
    gold = {g["row_id"]: g["gold"] for g in jsonl(FT / "gold.jsonl")}
    pred = {m: {x["row_id"]: x["pred"] for x in jsonl(PRED / f"{m}.jsonl")} for m in MODELS}
    rows = []
    for r in jsonl(FT / "rows.jsonl"):
        rows.append({"row_id": r["row_id"], "posting_id": r["posting_id"], "gold": gold[r["row_id"]],
                     "cand": CandidateCard.from_dict(r["profile"]), "design_pass": bool(r["design"]["eligibility_pass"]),
                     **{m: pred[m][r["row_id"]] for m in MODELS}})
    return rows, v2, alts, missing


def must_counts(v2: dict, v3: dict) -> dict:
    """Required groups the rules count (rules.required_groups) and nice-to-haves, v2 against another card set, on the
    postings both have; and how many cards changed each eligibility or fit field."""
    n2 = {k: len(required_groups(c)) for k, c in v2.items()}
    n3 = {k: len(required_groups(c)) for k, c in v3.items()}
    ks = sorted(set(n2) & set(n3))
    return {"postings": len(ks),
            "groups_v2": sum(n2[k] for k in ks), "groups_other": sum(n3[k] for k in ks),
            "postings_fewer": sum(n3[k] < n2[k] for k in ks), "postings_more": sum(n3[k] > n2[k] for k in ks),
            "postings_same_count": sum(n3[k] == n2[k] for k in ks),
            "postings_identical_groups": sum(set(v2[k].must_skills) == set(v3[k].must_skills) for k in ks),
            "postings_with_none_v2": sum(n2[k] == 0 for k in ks), "postings_with_none_other": sum(n3[k] == 0 for k in ks),
            "postings_8_plus_v2": sum(n2[k] >= 8 for k in ks), "postings_8_plus_other": sum(n3[k] >= 8 for k in ks),
            "nice_v2": sum(len(v2[k].nice_skills) for k in ks), "nice_other": sum(len(v3[k].nice_skills) for k in ks),
            "other_fields_changed": dict(Counter(f for k in ks for f in ("work_mode", "degree_level", "degree_fields",
                                                                            "min_years", "license", "employment_type",
                                                                            "enrolled_in", "citizens_only", "sponsorship",
                                                                            "locations", "salary_max", "thin_posting")
                                                 if getattr(v2[k], f) != getattr(v3[k], f)))}


def canary() -> dict:
    """Stored synthetic pair labels (data/built/pairs/manifest.jsonl) the work-mode rule moves to Excluded."""
    man = ROOT / "data/built/pairs/manifest.jsonl"
    if not man.exists():
        return {"available": False}
    import render_cards as rc
    reqs, cands = rc.load_cards("postings"), rc.load_cards("resumes")
    n, moved, by = 0, 0, Counter()
    for line in man.open():
        m = json.loads(line)
        if m["rq_id"] not in reqs or m["cd_id"] not in cands:
            continue
        n += 1
        c, r = CandidateCard.from_dict(cands[m["cd_id"]]), RequirementCard.from_dict(reqs[m["rq_id"]])
        b = label(c, r)[0]
        if b != m["bucket"]:
            moved += 1
            by[(m["bucket"], b)] += 1
    return {"pairs": n, "bucket_changed": moved, "by": {f"{a} -> {b}": k for (a, b), k in by.items()}}


def compare(rows: list[dict], v2: dict, other: dict, base: dict[str, list[str]], design: list[bool]) -> dict:
    """What another card set changes against the v2 cards, new rules on both sides (only the card differs)."""
    sy = systems(rows, other, False)
    must_only = {k: replace(v2[k], must_skills=other[k].must_skills, nice_skills=other[k].nice_skills) for k in v2}
    sm = systems(rows, must_only, False)
    flip = [o != n for o, n in zip(base["rules_full"], sy["rules_full"])]
    flip_m = [o != n for o, n in zip(base["rules_full"], sm["rules_full"])]
    return {
        "must_skills": must_counts(v2, other),
        "rules_full_rows_flipped": agree(rows, base["rules_full"], sy["rules_full"], flip),
        "rules_full_rows_flipped_eligibility_pass_design": agree(rows, base["rules_full"], sy["rules_full"],
                                                                 [f and d for f, d in zip(flip, design)]),
        "rules_full_rows_flipped_by_must_and_nice_alone": agree(rows, base["rules_full"], sm["rules_full"], flip_m),
        "flips": dict(Counter(f"{o}->{n}" for o, n, f in zip(base["rules_full"], sy["rules_full"], flip) if f)),
        "h_prime_rows_flipped": {f"{m}+H'": agree(rows, base[f"{m}+H'"], sy[f"{m}+H'"],
                                                  [o != n for o, n in zip(base[f"{m}+H'"], sy[f"{m}+H'"])])
                                 for m in MODELS},
    }, sy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "eval/fix_cycle2.json"))
    a = ap.parse_args()
    rows, v2, alts, missing = load()
    allr = [True] * len(rows)
    design = [r["design_pass"] for r in rows]
    old_v2, new_v2 = systems(rows, v2, True), systems(rows, v2, False)

    ruled = [mode_ruled_out(r["cand"], v2[r["posting_id"]]) for r in rows]
    thin = [x and v2[r["posting_id"]].thin_posting for x, r in zip(ruled, rows)]
    flip_wm = [o != n for o, n in zip(old_v2["rules_full"], new_v2["rules_full"])]
    work_mode = {
        "what": "a work mode the profile rules out (remote only vs hybrid/onsite, hybrid or remote only vs onsite) is a "
                "hard rules Skip (rules._check_location), soft only on a thin posting; H' Skips it (category work_mode)",
        "broke": "rules._check_location wrote it as a soft reason, so the rules said Explore and H' (hard Skips only) "
                 "never fired; the card (work_mode), the profile (remote_preference) and H' were otherwise fine. The "
                 "card cannot tell a stated onsite from an unstated one (onsite is the default), which is why thin "
                 "postings keep it soft.",
        "ruled_out_rows_v2_cards": {"rows": sum(ruled), "thin": sum(thin), "gold": dict(Counter(r["gold"] for r, x in zip(rows, ruled) if x))},
        "agreement_on_ruled_out_rows": {k: agree(rows, old_v2[k], new_v2[k], ruled)
                                        for k in ("rules_full", *(f"{m}+H'" for m in MODELS))},
        "rules_full_on_rows_it_flips": agree(rows, old_v2["rules_full"], new_v2["rules_full"], flip_wm),
        "h_prime_on_rows_it_flips": {f"{m}+H'": agree(rows, old_v2[f"{m}+H'"], new_v2[f"{m}+H'"],
                                                      [o != n for o, n in zip(old_v2[f"{m}+H'"], new_v2[f"{m}+H'"])])
                                     for m in MODELS},
        "synthetic_pairs_moved": canary(),
    }
    # eligibility pass recomputed (amendment definition) vs the sealed design flag
    def elig(cards: dict, old: bool) -> list[bool]:
        out = []
        for r in rows:
            v = verdict(r["cand"], cards[r["posting_id"]], old)
            out.append(not (v["rule"] == "Skip" and set(v["hard_cats"]) & HP_CATS) and not v["salary_soft"])
        return out
    e_old, e_new2 = elig(v2, True), elig(v2, False)
    eligibility = {
        "sealed_design_pass_rows": sum(design),
        "old_rules_v2_cards_reproduce_design_flag": sum(a == b for a, b in zip(e_old, design)),
        "new_rules_v2_cards_pass_rows": sum(e_new2),
        "new_rules_v2_cards_rows_differing_from_design_flag": sum(a != b for a, b in zip(e_new2, design)),
        "new_rules_by_card_set": {k: {"pass_rows": sum(e), "rows_differing_from_design_flag":
                                      sum(x != y for x, y in zip(e, design))}
                                  for k, e in ((k, elig(c, False)) for k, c in alts.items())},
        "note": "fit_eval_score.load_fresh stops when the recomputed pass differs from the sealed design flag, so "
                "fresh/select/fresh2 refuse to run under the new rules until that check and the slice definition are "
                "decided (keep the sealed design flag as the slice, or recompute it as amendment 2 words it).",
    }
    extractor = {"what": "each card set against the v2 cards (data/built/fresh_test/cards.jsonl) on fresh test 1, new "
                         "rules on both sides; v2_rerun is the noise floor any instruction change has to clear",
                 "decision": "v3 not adopted: neither draft makes the full rules agree with gold more often on the rows "
                             "it flips than the v2 cards do, so no fresh test 2 or fit cards were re-extracted"}
    tables = {"v4_old_rules_v2_cards": old_v2, "new_rules_v2_cards": new_v2}
    for k, (f, hsh, what) in ALTS.items():
        res, sy = compare(rows, v2, alts[k], new_v2, design)
        extractor[k] = {"file": f"data/built/fresh_test/{f}", "blueprint": hsh, "what": what,
                        "postings_without_a_card_kept_v2": missing[k], **res}
        tables[f"new_rules_{k}_cards"] = sy
    tables = {k: {"all_rows": table(rows, c, allr), "eligibility_pass_sealed_design_flag": table(rows, c, design)}
              for k, c in tables.items()}
    log = ROOT / "data/adaption/fixcycle2_credits.json"
    credits = json.loads(log.read_text()) if log.exists() else {"runs": []}
    out = {"cycle": "fix cycle 2 (amendment 2: fresh test 1 is the selection set; fresh test 2 untouched)",
           "set": "fresh test 1, 592 rows, 150 postings, sealed gold (data/built/fresh_test/gold.jsonl)",
           "code": {"rules": "src/matcher/rules.py _check_location, mode_ruled_out, WORK_MODE_PREFIX",
                    "h_prime": "scripts/hybrid_rules.py category() work_mode, HP_CATS",
                    "extractor_draft": "src/matcher/blueprint.py BLUEPRINT_V3 (live BLUEPRINT stays v2)",
                    "re_extraction": "scripts/extract_v3.py", "this_file": "scripts/fix_cycle2_measure.py"},
           "work_mode": work_mode, "eligibility_pass": eligibility, "extractor_v3": extractor,
           "systems_point_values": tables,
           "credits": {"cap": 25, "spent_by_balance": round(sum(r["spent"] for r in credits["runs"]), 4),
                       "runs": [{k: r[k] for k in ("set", "blueprint", "rows", "estimate", "balance_before",
                                                   "balance_after", "spent")} for r in credits["runs"]]}}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"work_mode": work_mode["agreement_on_ruled_out_rows"],
                      **{k: {x: extractor[k][x] for x in ("must_skills", "rules_full_rows_flipped",
                                                          "rules_full_rows_flipped_by_must_and_nice_alone")}
                         for k in ALTS}}, indent=1))
    for k, t in tables.items():
        print(k)
        for part, cols in t.items():
            print(" ", part, {s: (m["accuracy"], m["macro_f1"], m["keep_recall"]) for s, m in cols.items()})


if __name__ == "__main__":
    main()
