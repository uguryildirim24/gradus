"""Build data/built/pairs/{manifest.jsonl,splits.json,...} from the card files, then check it.

Checks (the build fails when one trips):
- every row reproduces under label() + format_completion (a regression check: rows come from
  label(), so this catches a stale manifest or a reason formatter change, not a wrong rule; the
  independent check is test_rules' construction-versus-label comparison);
- no candidate has a work authorization that contradicts their education;
- no candidate holds a license their finished degree could not lead to;
- every candidate's years of work fit their graduation year (variants.years_fit);
- no candidate wants internships, or is on CPT, and avoids "intern" titles (variants.prefs_fit), none
  lists a skill twice in any wording (variants.unique_skills), and none wants and avoids one
  industry (variants.industry_clash, by rules.industry_ids);
- no surface feature value (fingerprints.py) puts more than 80% of its rows in one bucket;
- For you and Explore each take at least 25% of their rows from repaired real candidates;
- STEM OPT and needs-H-1B candidates each have For you rows against sponsoring postings;
- a status the rule checks the same on every posting stance (citizen, CPT, other visa) keeps its
  bucket mix on each stance: no (status, stance) cell's Keep, Explore or Skip rate departs from
  the status's own by more than STANCE_MAX_GAP (a one-vs-rest check the 80% gate cannot make);
- no real posting card in the pairs is a near copy of a real test posting (matcher.nearcopy);
- every determinant keeps its rows: the share of Skip (or Explore) rows that one check decides
  stays above COVERAGE_FLOORS, so steering cannot starve one silently (it once left 45
  years-short rows of 10,500).
Writes data/built/pairs/fingerprints.md with the full table and the coverage numbers.
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from matcher.nearcopy import load_index
from matcher.fingerprints import FIT_FEATURES, MAX_SHARE, MIN_ROWS, STANCE_DECIDES, stance_cell, tally_rows, \
    worst
from matcher.pairs import build_rows, load_candidate_socs, load_candidates, load_requirements, write_outputs
from matcher.rules import format_completion, label, license_problems, status_problems
from matcher.schema import CandidateCard, RequirementCard
from matcher.variants import industry_clash, prefs_fit, unique_skills, years_fit

ROOT = Path(__file__).resolve().parent.parent
REPAIRED_MIN = 0.25
STANCE_MAX_GAP = 0.15
# (bucket, reason pattern, least share of that bucket's rows with such a reason). About half of
# what the v3 build gives, so a seed change does not trip them; years short is the 3% asked for.
COVERAGE_FLOORS = {
    "years short": ("Excluded", r"^needs \d+\+ years of experience", 0.03),
    "one year short": ("Explore", r"year short of the", 0.015),
    "overqualified": ("Explore", r"^you may be overqualified", 0.015),
    "license": ("Excluded", r"^requires the ", 0.008),
    "enrollment": ("Excluded", r"^for students currently pursuing", 0.03),
    "degree level": ("Excluded", r"^needs (a|an) .*(or higher|doctorate);", 0.03),
    "degree field": ("Excluded", r"^degree field should be", 0.03),
    "citizens only": ("Excluded", r"^limited to US citizens", 0.025),
    "no sponsorship": ("Excluded", r"will not sponsor visas", 0.025),
    "no OPT": ("Excluded", r"does not accept OPT", 0.005),
}


def check(res: dict, reqs, cands):
    """(report lines, failures, fingerprint tally)."""
    rows = res["rows"]
    cd = dict(cands) | {v["id"]: CandidateCard.from_dict(v["card"]) for v in res["variant_candidates"]}
    rq = {r.rq_id: r.card for r in reqs} | {v["id"]: RequirementCard.from_dict(v["card"]) for v in res["variant_requirements"]}
    out, fail = [], []
    n = len(rows)
    tot = Counter(r["bucket"] for r in rows)
    out.append(f"rows {n}: " + ", ".join(f"{b} {tot[b]} ({tot[b] / n:.0%})" for b in ("For you", "Explore", "Excluded")))
    for b in tot:
        if not 0.28 <= tot[b] / n <= 0.42:
            fail.append(f"bucket mix: {b} is {tot[b] / n:.0%}")
    # (c) every row reproduces
    bad = [r["pair_id"] for r in rows
           if (lambda b, reasons, flags: (b, reasons, flags, format_completion(b, reasons)) !=
               (r["bucket"], r["reasons"], r["flags"], r["completion"]))(*label(cd[r["cd_id"]], rq[r["rq_id"]]))]
    out.append(f"rows that do not reproduce under label() + format_completion (regression check): {len(bad)}")
    if bad:
        fail.append(f"{len(bad)} rows do not reproduce, e.g. {bad[:3]}")
    probs = [r["cd_id"] for r in rows if status_problems(cd[r["cd_id"]])]
    out.append(f"candidates with a contradictory work authorization: {len(set(probs))}")
    if probs:
        fail.append(f"{len(probs)} rows with contradictory work authorization")
    lic = [r["cd_id"] for r in rows if license_problems(cd[r["cd_id"]])]
    out.append(f"candidates holding a license their degree cannot lead to: {len(set(lic))}")
    if lic:
        fail.append(f"{len(lic)} rows with a license that does not fit the degree, e.g. {lic[:3]}")
    yrs = [r["cd_id"] for r in rows if not years_fit(cd[r["cd_id"]])]
    out.append(f"candidates whose years of work do not fit their graduation year: {len(set(yrs))}")
    if yrs:
        fail.append(f"{len(yrs)} rows whose years do not fit the graduation year, e.g. {yrs[:3]}")
    clash = [r["cd_id"] for r in rows if not prefs_fit(cd[r["cd_id"]])]
    out.append(f"candidates who want internships or are on CPT and avoid 'intern' titles: {len(set(clash))}")
    if clash:
        fail.append(f"{len(clash)} rows whose candidate wants internships or is on CPT and avoids 'intern', e.g. {clash[:3]}")
    twice = [r["cd_id"] for r in rows if unique_skills(cd[r["cd_id"]].skills) != cd[r["cd_id"]].skills]
    out.append(f"candidates listing a skill twice: {len(set(twice))}")
    if twice:
        fail.append(f"{len(twice)} rows whose candidate lists a skill twice, e.g. {twice[:3]}")
    clash_ind = [r["cd_id"] for r in rows if industry_clash(cd[r["cd_id"]])]
    out.append(f"candidates who want and avoid one industry: {len(set(clash_ind))}")
    if clash_ind:
        fail.append(f"{len(clash_ind)} rows whose candidate wants and avoids one industry, e.g. {clash_ind[:3]}")
    out.append(f"builder stats: {res.get('stats', {})}")
    # (d) repaired share
    for b in ("For you", "Explore"):
        k = Counter(r["candidate_kind"] for r in rows if r["bucket"] == b)
        share = k["repaired"] / max(sum(k.values()), 1)
        out.append(f"{b} rows by candidate kind: {dict(k)}; repaired share {share:.0%}")
        if share < REPAIRED_MIN:
            fail.append(f"{b}: repaired share {share:.0%} < {REPAIRED_MIN:.0%}")
    # (e) visa coverage
    for st in ("stem_opt", "needs_h1b_now", "f1_opt", "f1_cpt", "other_visa"):
        c = Counter(r["bucket"] for r in rows if cd[r["cd_id"]].work_auth == st)
        sp = sum(1 for r in rows if cd[r["cd_id"]].work_auth == st and r["bucket"] == "For you"
                 and rq[r["rq_id"]].sponsorship == "sponsors")
        out.append(f"work_auth {st}: {dict(c)}; For you against a sponsoring posting: {sp}")
        if st in ("stem_opt", "needs_h1b_now") and sp == 0:
            fail.append(f"{st} has no For you row against a sponsoring posting")
    # stance-blind statuses keep their bucket mix on every stance
    cells: dict[str, Counter] = {}
    groups: dict[str, Counter] = {}
    for r in rows:
        cell = stance_cell(cd[r["cd_id"]], rq[r["rq_id"]])
        if cell == STANCE_DECIDES:
            continue
        cells.setdefault(cell, Counter())[r["bucket"]] += 1
        groups.setdefault(cell.split(" / ")[0], Counter())[r["bucket"]] += 1
    for cell, c in sorted(cells.items()):
        g, n = groups[cell.split(" / ")[0]], sum(c.values())
        gap = max(abs(c[b] / n - g[b] / sum(g.values())) for b in ("For you", "Explore", "Excluded"))
        out.append(f"stance cell {cell}: n={n}, " + ", ".join(f"{b} {c[b] / n:.0%}" for b in ("For you", "Explore",
                                                                                             "Excluded"))
                   + f"; largest gap from the status's mix {gap:.0%}")
        if n >= MIN_ROWS and gap > STANCE_MAX_GAP:
            fail.append(f"stance cell {cell}: a bucket rate is {gap:.0%} off the status's own (n={n})")
    # determinant coverage
    for name, (b, pat, floor) in COVERAGE_FLOORS.items():
        rx = re.compile(pat)
        k = sum(1 for r in rows if r["bucket"] == b and any(rx.search(x) for x in r["reasons"]))
        share = k / max(tot[b], 1)
        out.append(f"{name}: {k} {b} rows ({share:.1%}; floor {floor:.1%})")
        if share < floor:
            fail.append(f"coverage: {name} is {share:.1%} of {b} rows, under {floor:.1%}")
    # no real posting that trains is a near copy of a real test posting (matcher.nearcopy; merge marks them
    # in_real_test and load_requirements skips them, so this fails only if a card file bypasses that)
    real = [r for r in reqs if r.source == "extracted_real"]
    if real:
        index, posts = load_index(ROOT)
        near = [r.rq_id for r in real if r.rq_id in posts and index.near(posts[r.rq_id])]
        out.append(f"real postings in the pairs: {len(real)}; near copies of a real test posting: {len(near)}")
        if near:
            fail.append(f"{len(near)} real postings are near copies of real test postings, e.g. {near[:3]}")
    else:
        out.append("real postings in the pairs: 0, so none is a near copy of a real test posting")
    # render bill: unique resume and posting contexts behind the rows
    # the contexts render_cards sends (one render each): the bill of the full render, at about 0.01 credit a context
    sys.path.insert(0, str(ROOT / "scripts"))
    from render_cards import ctx_hash, ctx_str

    resumes = {ctx_hash(ctx_str("resumes", cd[r["cd_id"]].to_dict())) for r in rows}
    postings = {ctx_hash(ctx_str("postings", rq[r["rq_id"]].to_dict())) for r in rows}
    out.append(f"unique resume contexts {len(resumes)}, unique posting contexts {len(postings)} as render_cards sends "
               f"them (edited postings: {len(res['variant_requirements'])})")
    # (a) fingerprints
    t = tally_rows(rows, cd, rq)
    bad_fp = worst(t)
    out.append(f"fingerprint values over {MAX_SHARE:.0%} in one bucket (n >= {MIN_ROWS}): {len(bad_fp)}")
    for x in bad_fp:
        fail.append(f"fingerprint {x['feature']}={x['value']}: {x['top_share']:.0%} {x['top_bucket']} (n={x['n']})")
    return out, fail, t


def fingerprint_md(t, lines: list[str]) -> str:
    md = ["# Pair-build checks", "", "Written by `scripts/build_pairs.py`.", ""] + [f"- {x}" for x in lines]
    md += ["", f"## Fingerprints (values with n >= {MIN_ROWS}; top bucket share must stay <= {MAX_SHARE:.0%})", "",
           f"Fit features ({', '.join(sorted(FIT_FEATURES))}) compare the two sides the way the rule does; "
           "they are reported, not steered or gated.", "",
           "| feature | value | n | top bucket | share | For you | Explore | Excluded |", "|---|---|---|---|---|---|---|---|"]
    for x in sorted(t.table(), key=lambda x: (x["feature"], -x["n"])):
        c = x["counts"]
        name = x["feature"] + (" (fit, not gated)" if x["feature"] in FIT_FEATURES else "")
        md.append(f"| {name} | {x['value']} | {x['n']} | {x['top_bucket']} | {x['top_share']:.0%} | "
                  f"{c.get('For you', 0)} | {c.get('Explore', 0)} | {c.get('Excluded', 0)} |")
    return "\n".join(md) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requirements", default=ROOT / "data/built/cards/requirement_cards.jsonl")
    ap.add_argument("--candidates", default=ROOT / "data/built/cards/candidate_cards.jsonl")
    ap.add_argument("--out", default=ROOT / "data/built/pairs")
    ap.add_argument("--per-card", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    reqs = load_requirements(Path(a.requirements))
    cands = load_candidates(Path(a.candidates))
    res = build_rows(reqs, cands, per_card=a.per_card, seed=a.seed, cand_socs=load_candidate_socs(Path(a.candidates)))
    s = write_outputs(res, Path(a.out))
    print("held-out families:", s["heldout_families"])
    for split, c in s["counts"].items():
        print(split, c)
    lines, fail, t = check(res, reqs, cands)
    md = fingerprint_md(t, lines)
    (Path(a.out) / "fingerprints.md").write_text(md)
    print("\n".join(lines))
    print("\nTop fingerprint shares:")
    for x in sorted(t.table(), key=lambda x: -x["top_share"])[:15]:
        print(f"  {x['feature']}={x['value']}: {x['top_share']:.0%} {x['top_bucket']} (n={x['n']})")
    if fail:
        print("\nBUILD CHECKS FAILED:\n  " + "\n  ".join(fail))
        sys.exit(1)
    print("\nall build checks passed")


if __name__ == "__main__":
    main()
