"""Build requirement and candidate cards from O*NET.

Writes data/built/cards/{requirement_cards,candidate_cards}.jsonl and README.md.
Usage: uv run python scripts/build_cards.py [--requirements 1500] [--candidates 1200]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matcher.onet import load_occupations  # noqa: E402
from matcher.sample import sample_candidate_cards, sample_requirement_cards  # noqa: E402

OUT = ROOT / "data" / "built" / "cards"


def pct(counter: Counter, total: int, top: int | None = None) -> str:
    rows = counter.most_common(top)
    return "\n".join(f"| {k if k != '' else '(none)'} | {v} | {100 * v / total:.1f}% |" for k, v in rows)


def table(title: str, counter: Counter, total: int, top: int | None = None) -> str:
    return f"### {title}\n\n| value | n | share |\n|---|---|---|\n{pct(counter, total, top)}\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requirements", type=int, default=1500)
    ap.add_argument("--candidates", type=int, default=1200)
    ap.add_argument("--req-seed", type=int, default=20261017)
    ap.add_argument("--cand-seed", type=int, default=20261018)
    args = ap.parse_args()

    occs = load_occupations()
    reqs = sample_requirement_cards(occs, args.requirements, args.req_seed)
    cands = sample_candidate_cards(occs, args.candidates, args.cand_seed)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "requirement_cards.jsonl").open("w") as fh:
        for rid, card in reqs:
            fh.write(json.dumps({"id": rid, "source": "onet", "card": card.to_dict()}, ensure_ascii=False) + "\n")
    with (OUT / "candidate_cards.jsonl").open("w") as fh:
        for cid, card, soc in cands:
            fh.write(json.dumps({"id": cid, "target_soc": soc, "card": card.to_dict()}, ensure_ascii=False) + "\n")

    rc = [c for _, c in reqs]
    cc = [c for _, c, _ in cands]
    nr, nc = len(rc), len(cc)
    parts = [
        "# Built cards\n",
        "Written by `scripts/build_cards.py` (seeded, deterministic). Occupation facts come from O*NET 31.0 "
        "(CC BY 4.0, USDOL/ETA; modified). Location, work mode, sponsorship, salary, employment type and seniority are injected by the sampler; "
        "skill names are shown the way postings write them (onet.display_name), and computing and engineering postings "
        "draw on O*NET Hot Technologies (Python, SQL, C++ ...) plus field concepts. A student slice of internships "
        "for current students is added (sample.STUDENT_SHARE). "
        "pay is a job-zone x field heuristic, not O*NET wage data. Candidates have no names.\n",
        "`must_skills` is a list of groups; a group is met by any one of its names.\n",
        "Files: `requirement_cards.jsonl` (`id` rq_*, `source`, `card`), `candidate_cards.jsonl` (`id` cd_*, `target_soc`, `card`). "
        "Rendered posting and resume text is joined later by `id`.\n",
        f"## Requirement cards ({nr})\n",
        table("Field", Counter(c.field for c in rc), nr),
        table("Employment type", Counter(c.employment_type for c in rc), nr),
        table("Seniority", Counter(c.seniority for c in rc), nr),
        table("Degree level", Counter(c.degree_level for c in rc), nr),
        table("Enrolled in (for current students at this level or higher)", Counter(c.enrolled_in for c in rc), nr),
        f"Must-skill groups: {sum(len(c.must_skills) for c in rc)}, of which any-of (2+ alternatives): "
        f"{sum(len(g) > 1 for c in rc for g in c.must_skills)}.\n",
        table("Sponsorship stance", Counter(c.sponsorship for c in rc), nr),
        table("Work mode", Counter(c.work_mode for c in rc), nr),
        f"Student-facing coverage: internships {sum(c.employment_type == 'internship' for c in rc) / nr:.1%}, "
        f"enrolled_in set {sum(c.enrolled_in != 'none' for c in rc) / nr:.1%}, "
        f"citizens-only {sum(c.citizens_only for c in rc) / nr:.1%}, "
        f"0-1 must-skill groups {sum(len(c.must_skills) <= 1 for c in rc) / nr:.1%}.\n",
        f"Citizens-only: {sum(c.citizens_only for c in rc)}; E-Verify: {sum(c.e_verify for c in rc)}; "
        f"with license: {sum(bool(c.license) for c in rc)}; with salary: {sum(c.salary_min is not None for c in rc)}; "
        f"distinct occupations: {len({c.onet_soc for c in rc})}.\n",
        table("Licenses", Counter(c.license for c in rc if c.license), nr),
        f"## Candidate cards ({nc})\n",
        table("Field", Counter(c.field for c in cc), nc),
        table("Work authorization", Counter(c.work_auth for c in cc), nc),
        table("Degree level", Counter(c.degree_level for c in cc), nc),
        table("Education country (top 20)", Counter(c.education_country for c in cc), nc, 20),
        f"Distinct education countries: {len({c.education_country for c in cc})}; languages: {len({c.language for c in cc})}.\n",
        table("Language (top 12)", Counter(c.language for c in cc), nc, 12),
        table("Years of experience", Counter(min(c.years, 15) // 3 * 3 for c in cc), nc),
        table("Remote preference", Counter(c.remote_preference for c in cc), nc),
        table("Employment types wanted", Counter("/".join(c.employment_types_wanted) or "any" for c in cc), nc),
        f"Salary floor set: {sum(c.salary_floor is not None for c in cc)}; holds a license: {sum(bool(c.licenses) for c in cc)}.\n",
    ]
    (OUT / "README.md").write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
