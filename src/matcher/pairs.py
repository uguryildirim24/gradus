"""Training-pair manifest: (requirement card, candidate) rows labeled by rules.label.

Text is joined in later by id: resume text by cd_id, posting text by rq_id.
Variant candidates (one-edit copies) are new cards with ids cd_v_*; when a
variant also edits the posting card (work-auth cases) it gets an rq id
"<rq_id>~<kind>" and its card is written to variant_requirements.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rules import format_completion, label
from .schema import BUCKETS, CandidateCard, RequirementCard
from .textclean import fit_posting, redact, strip_markdown
from .splits import CAND_RATIOS, HELDOUT, RATIOS, assign, family_of, pick_heldout_families
from .variants import make_variants

TARGET_MIX = {"Excluded": 0.35, "Explore": 0.35, "For you": 0.30}
INSTRUCTION = (
    "You match a candidate to a job posting. Read the resume, the candidate profile (education, years of "
    "work experience, work authorization and job preferences from the candidate's form) and the posting, "
    "then answer with one label, "
    "Keep, Explore or Skip, followed by a short reason.\n"
    "Keep = meets every requirement and every preference in the profile (soft skills such as communication or "
    "teamwork do not decide the label). "
    "Explore = meets the requirements, but something is a stretch or does not fit the profile. "
    "Skip = misses a must-have: degree, experience, license, required skill, location, job type "
    "or work authorization.\n"
)


def build_prompt(resume_text: str, profile_text: str, posting_text: str, max_posting_tokens: int = 1500) -> str:
    """Instruction + resume + candidate profile + posting.

    The resume is redacted (contact block becomes "Candidate") and both texts lose markdown. The posting
    is fitted to a token budget (~4 chars/token) by textclean.fit_posting, which drops boilerplate first
    and keeps the header, the work-authorization sentences, then pay / years / degree / enrollment /
    work-mode lines, then the other requirement sentences. The 1500 here is the old ~4 chars/token count
    (6,000 characters, about 1,180 Qwen tokens). On the 1,322 of the 2,655 real postings that have a
    work-authorization sentence, every such sentence reaches the fitted text whole, as written in the posting
    (textclean.survival checks it with its own splitter and detector, not fit_posting's; before the splitter
    stopped splitting at 'U.S.' and 'Inc.', 18 postings lost or mangled one). A plain cut at 4,800 characters
    loses one in 36% of those postings. Against
    the v2 extraction cards of the 726 over-budget real postings, every pay and years figure survives, 99.9% of
    work-mode sentences and 99.2% of the must-skill groups the posting names (7 postings lose one; the losses
    are duty-line mentions in long multi-role postings); on the 87 real test cards every must-skill group
    survives (`python -m matcher.textclean REPORT`)."""
    resume = redact(strip_markdown(resume_text))
    posting = fit_posting(strip_markdown(posting_text), max_posting_tokens * 4)
    return (f"{INSTRUCTION}\nRESUME:\n{resume}\n\nCANDIDATE PROFILE:\n{profile_text.strip()}"
            f"\n\nJOB POSTING:\n{posting}\n\nAnswer:")


def prompt_parts(prompt: str) -> tuple[str, str, str]:
    """(resume, profile, posting) exactly as build_prompt wrote them into `prompt`: the resume redacted and
    without markdown, the posting fitted to the budget. Visibility checks read these, not the raw texts."""
    rest = prompt.split("\nRESUME:\n", 1)[1]
    resume, rest = rest.split("\n\nCANDIDATE PROFILE:\n", 1)
    profile, rest = rest.split("\n\nJOB POSTING:\n", 1)
    return resume, profile, rest.rsplit("\n\nAnswer:", 1)[0]


@dataclass(frozen=True)
class Req:
    rq_id: str
    card: RequirementCard
    source: str
    family: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _card_of(row: dict[str, Any]) -> dict[str, Any]:
    inner = row.get("card")
    return inner if isinstance(inner, dict) else row


def load_requirements(path: Path) -> list[Req]:
    """Rows of a card file. Real-card rows whose meta fails blueprint.training_ok (not US, French,
    phrase-like required group) are skipped."""
    from .blueprint import training_ok

    out = []
    for row in read_jsonl(path):
        if not training_ok(row.get("meta")):
            continue
        card = RequirementCard.from_dict(_card_of(row))
        rid = row.get("id") or row.get("rq_id")
        source = row.get("source") or ("onet" if card.onet_soc else "extracted_real")
        out.append(Req(rid, card, source, family_of(card.field, card.onet_soc)))
    return out


def load_candidates(path: Path) -> list[tuple[str, CandidateCard]]:
    out = []
    for row in read_jsonl(path):
        out.append((row.get("id") or row.get("cd_id"), CandidateCard.from_dict(_card_of(row))))
    return out


def load_candidate_socs(path: Path) -> dict[str, str]:
    """cd_id -> target_soc for candidate files that carry one."""
    return {(row.get("id") or row.get("cd_id")): row["target_soc"] for row in read_jsonl(path) if row.get("target_soc")}


# ---------------------------------------------------------------- pair building
# Kept below the prompt code on purpose: the prompt and the pair builder change separately.

from .fingerprints import STANCE_DECIDES, Tally, features, stance_cell  # noqa: E402
from .rules import license_problems, status_problems  # noqa: E402
from .variants import repair  # noqa: E402

# Wanted share of each work authorization among rows: international students are the users
# the pitch is about, so they get more rows than their share of sampled candidates.
AUTH_SHARE = {"citizen_or_pr": 0.6, "f1_cpt": 0.1, "f1_opt": 0.08, "stem_opt": 0.06, "needs_h1b_now": 0.1,
              "other_visa": 0.06}
RESUME_REUSE_BONUS = 0.5  # prefer a resume already picked for this posting (same text, other profile)
# Wanted share of repaired real candidates among each bucket's rows (gate: at least 25%).
REPAIRED_SHARE = {"For you": 0.42, "Explore": 0.42, "Excluded": 0.3}
N_REPAIRED = 3  # real candidates repaired toward each posting
N_REAL = 22  # real candidates labelled as they are, per posting
MAX_EDITED = 2  # rows per posting whose posting card is a work-authorization copy ("rq~sponsors")
# How hard picks steer against fingerprints, against the pulls toward repaired candidates and the
# work-authorization mix. At 1.0 a boosted-tree probe on candidate fields alone read the bucket at
# +0.11 over the majority rate; at 2.0 it stays under the 0.10 leakage margin on three build seeds.
FINGERPRINT_WEIGHT = 2.0
NOISE = 0.4  # random tie-breaking between near-equal picks
# Pull toward AUTH_SHARE within each bucket, scaled by the share of the target still missing: rare
# statuses (other visa, STEM OPT) keep their rows against the fingerprint steering.
AUTH_WEIGHT = 2.0
# Pull toward TARGET_MIX within each (stance-blind status, posting stance) cell: the AUTH pull
# otherwise hands a sponsoring posting's For you slots to visa holders, leaving citizens there
# 9% Keep against 36% elsewhere, which a model can read as "citizen + sponsors = not Keep".
STANCE_WEIGHT = 2.0


# The candidate fields a rendered resume is written from (scripts/render_cards.py RESUME_FIELDS):
# candidates equal on these share one resume text.
RESUME_FIELDS = ("field", "degree_level", "degree_field", "graduation_year", "years", "skills", "licenses",
                 "current_location", "education_country", "language")


def resume_key(c: CandidateCard) -> str:
    d = c.to_dict()
    return hashlib.sha1(json.dumps([d[k] for k in RESUME_FIELDS], ensure_ascii=False).encode()).hexdigest()[:16]


def _variant_id(prefix: str, rq_id: str, kind: str, payload: dict) -> str:
    h = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]
    return f"{prefix}{rq_id.split('_', 1)[-1]}_{kind}_{h}"


def _pair_id(rq_id: str, cd_id: str) -> str:
    return "pr_" + hashlib.sha1(f"{rq_id}|{cd_id}".encode()).hexdigest()[:12]


def _edit_name(base: RequirementCard, card: RequirementCard) -> str:
    """Name a work-authorization copy of a posting by what changed, so copies that make the
    same change share one id (and one rendered posting)."""
    if card.citizens_only and not base.citizens_only:
        return "citizens"
    return {"sponsors": "sponsors", "will_not_sponsor": "no_sponsor", "will_not_sponsor_or_opt": "no_opt",
            "unstated": "unstated"}[card.sponsorship]


def _quotas(rng: random.Random, n: int, counts: Counter, total: int) -> Counter:
    """Draw n bucket slots, steering toward the global target mix."""
    q: Counter = Counter()
    for _ in range(n):
        done = total + sum(q.values())
        w = {}
        for b in BUCKETS:
            share = (counts[b] + q[b]) / done if done else 0
            w[b] = max(TARGET_MIX[b] - share + 0.05, 0.01)
        q[rng.choices(list(w), weights=list(w.values()))[0]] += 1
    return q


def assign_soc_splits(reqs: list[Req], held_fams: list[str], seed: int = 0) -> dict[str, str]:
    """Occupation (SOC, or the card id when it has none) -> split. Every posting of one
    occupation lands in one split, so dev and test occupations are never seen in train."""
    keys = sorted({r.card.onet_soc or r.rq_id for r in reqs if r.family not in held_fams})
    out = assign(keys, RATIOS, seed)
    for r in reqs:
        if r.family in held_fams:
            out[r.card.onet_soc or r.rq_id] = HELDOUT
    return out


def _cand_split(cid: str, soc: str | None, soc_split: dict[str, str], held_fams: list[str],
                fallback: dict[str, str]) -> str:
    if soc:
        if family_of("", soc) in held_fams:
            return HELDOUT
        if soc in soc_split:
            return soc_split[soc]
    return fallback[cid]


def _field_pools(reqs: list[Req]) -> dict[str, tuple[str, ...]]:
    """Skills other postings in a field use: what copies list beyond the card's own."""
    pools: dict[str, list[str]] = {}
    for r in reqs:
        lst = pools.setdefault(r.card.field, [])
        lst += [g[0] for g in r.card.must_skills] + list(r.card.nice_skills)
    return {f: tuple(sorted(set(v))) for f, v in pools.items()}


def build_rows(
    reqs: list[Req],
    cands: list[tuple[str, CandidateCard]],
    per_card: int = 7,
    seed: int = 0,
    n_heldout_families: int = 2,
    cand_socs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Pairs per posting from three sources: real candidates as they are, real candidates
    repaired toward the posting (plus their one-edit breaks), and copies built from the
    posting (plus theirs). Picks steer toward TARGET_MIX and against surface fingerprints
    (fingerprints.py); every row's label is label() of its cards."""
    cand_socs = cand_socs or {}
    fam_counts = Counter(r.family for r in reqs)
    held_fams = pick_heldout_families(dict(fam_counts), n_heldout_families, seed)
    soc_split = assign_soc_splits(reqs, held_fams, seed)
    rq_split = {r.rq_id: soc_split[r.card.onet_soc or r.rq_id] for r in reqs}
    fallback = assign([c for c, _ in cands], CAND_RATIOS, seed)
    cd_split = {cid: _cand_split(cid, cand_socs.get(cid), soc_split, held_fams, fallback) for cid, _ in cands}
    pools: dict[str, list[tuple[str, CandidateCard]]] = {s: [] for s in CAND_RATIOS}
    for cid, card in cands:
        pools[cd_split[cid]].append((cid, card))
    field_pools = _field_pools(reqs)

    rows: list[dict[str, Any]] = []
    var_cands: dict[str, dict] = {}
    var_reqs: dict[str, dict] = {}
    counts: Counter = Counter()
    kind_counts: Counter = Counter()
    auth_counts: Counter = Counter()
    stance_counts: Counter = Counter()  # (stance cell, bucket) and stance cell
    tally = Tally()
    stats: Counter = Counter()
    for req in sorted(reqs, key=lambda r: r.rq_id):
        split = rq_split[req.rq_id]
        rng = random.Random(f"{seed}:{req.rq_id}")
        pool = pools[split]
        q = _quotas(rng, per_card, counts, len(rows))
        opts: list[dict] = []
        seen_opts: set[tuple[str, CandidateCard]] = set()

        def option(kind: str, ckind: str, cid: str, cand: CandidateCard, card: RequirementCard,
                   src: str | None = None, expected: str | None = None) -> None:
            b, reasons, flags = label(cand, card)
            if expected is not None and b != expected:
                stats["expected_mismatch"] += 1
                return
            if status_problems(cand):
                stats["status_problem"] += 1
                return
            if license_problems(cand):
                stats["license_problem"] += 1
                return
            edited = card != req.card
            rq_id = f"{req.rq_id}~{_edit_name(req.card, card)}" if edited else req.rq_id
            key = (rq_id, cand)
            if key in seen_opts:
                stats["duplicate_option"] += 1
                return  # two edits that made the same pair (CPT on a posting that already refuses OPT)
            seen_opts.add(key)
            opts.append(dict(kind=kind, ckind=ckind, cid=cid, cand=cand, card=card, rq_id=rq_id, bucket=b,
                             reasons=reasons, flags=flags, src=src, edited=edited, resume=resume_key(cand),
                             feats=features(cand, card, ckind, edited)))

        same = [x for x in pool if x[1].field == req.card.field]
        other = [x for x in pool if x[1].field != req.card.field]
        for cid, cand in rng.sample(same, min(N_REAL // 2, len(same))) + rng.sample(other, min(N_REAL // 2, len(other))):
            option("real", "real", cid, cand, req.card)
        major = (req.card.onet_soc or "")[:2]
        near = sorted(same, key=lambda x: (cand_socs.get(x[0], "")[:2] != major, rng.random()))
        for n, (cid, cand) in enumerate(near[: N_REPAIRED * 2]):
            if sum(o["ckind"] == "repaired" and o["kind"] in ("full_match", "repaired_soft") for o in opts) >= N_REPAIRED:
                break
            fixed = repair(cand, req.card, random.Random(f"{seed}:{req.rq_id}:{cid}"))
            if fixed is None:
                stats["repair_failed"] += 1
                continue
            base, expected, note = fixed
            if expected != "For you":
                rid = _variant_id("cd_v_", req.rq_id, f"r{n}_soft", base.to_dict())
                option("repaired_soft", "repaired", rid, base, req.card, cid, expected)
                continue
            for v in make_variants(req.card, seed, base=base, base_kind="repaired", extra_pool=field_pools.get(req.card.field)):
                rid = _variant_id("cd_v_", req.rq_id, f"r{n}_{v.kind}", v.candidate.to_dict())
                option(v.kind, "repaired", rid, v.candidate, v.card, cid, v.expected)
        for v in make_variants(req.card, seed, extra_pool=field_pools.get(req.card.field)):
            rid = _variant_id("cd_v_", req.rq_id, v.kind, v.candidate.to_dict())
            option(v.kind, "copy", rid, v.candidate, v.card, None, v.expected)

        chosen: list[dict] = []
        used_resumes: set[str] = set()
        used_ids: set[str] = set()
        used_kinds: set[tuple[str, str]] = set()
        n_edited = 0

        def score(o: dict, b: str) -> float:
            s = FINGERPRINT_WEIGHT * tally.share_after(o["feats"], b)
            want = REPAIRED_SHARE[b]
            have = kind_counts[(b, "repaired")] / max(counts[b], 1)
            if o["ckind"] == "repaired" and have < want:
                s -= 2.0 * (want - have) * 4
            a = o["cand"].work_auth
            # each bucket keeps the overall work-authorization mix, so a status alone tells no bucket
            have_a = auth_counts[(b, a)] / max(counts[b], 1)
            if have_a < AUTH_SHARE[a]:  # pull by the share of the target still missing
                s -= AUTH_WEIGHT * (AUTH_SHARE[a] - have_a) / AUTH_SHARE[a]
            cell = o["feats"]["work_auth_x_stance"]
            # other visas are never Keep; posting copies already cost a render, so they get no pull
            if cell != STANCE_DECIDES and o["cand"].work_auth != "other_visa" and not o["edited"]:
                have_c = stance_counts[(cell, b)] / max(stance_counts[cell], 1)
                if have_c < TARGET_MIX[b]:
                    s -= STANCE_WEIGHT * (TARGET_MIX[b] - have_c) / TARGET_MIX[b]
            if o["resume"] in used_resumes:
                s -= RESUME_REUSE_BONUS
            if o["edited"]:  # a posting copy costs one more render; the citizens-only copy is worth it
                s += 0.1 if o["rq_id"].endswith("~citizens") else 0.6
            return s + rng.random() * NOISE

        for b in sorted(BUCKETS, key=lambda x: -q[x]):
            for _ in range(q[b]):
                cands_b = [o for o in opts if o["bucket"] == b and o["cid"] not in used_ids
                           and (o["kind"], o["ckind"]) not in used_kinds
                           and not (o["edited"] and n_edited >= MAX_EDITED)]
                if not cands_b:
                    stats[f"short_{b}"] += 1
                    continue
                o = min(cands_b, key=lambda x: score(x, b))
                chosen.append(o)
                used_ids.add(o["cid"])
                used_resumes.add(o["resume"])
                auth_counts[o["cand"].work_auth] += 1
                auth_counts[(b, o["cand"].work_auth)] += 1
                stance_counts[o["feats"]["work_auth_x_stance"]] += 1
                stance_counts[(o["feats"]["work_auth_x_stance"], b)] += 1
                used_kinds.add((o["kind"], o["ckind"]))
                n_edited += o["edited"]
                tally.add(o["feats"], b)
                counts[b] += 1
                kind_counts[(b, o["ckind"])] += 1
        for o in chosen:
            if o["ckind"] != "real":
                var_cands[o["cid"]] = {"id": o["cid"], "base": o["ckind"], "from": o["src"], "kind": o["kind"],
                                       "card": o["cand"].to_dict()}
            if o["edited"]:
                var_reqs[o["rq_id"]] = {"id": o["rq_id"], "base_rq_id": req.rq_id, "card": o["card"].to_dict()}
            rows.append({
                "pair_id": _pair_id(o["rq_id"], o["cid"]),
                "rq_id": o["rq_id"],
                "cd_id": o["cid"],
                "base_rq_id": req.rq_id,
                "bucket": o["bucket"],
                "reasons": o["reasons"],
                "flags": o["flags"],
                "variant_kind": o["kind"] if o["ckind"] != "real" else None,
                "candidate_kind": o["ckind"],
                "repaired_from": o["src"],
                "source": req.source,
                "family": req.family,
                "split": split,
                "completion": format_completion(o["bucket"], o["reasons"]),
            })
    return {
        "rows": rows,
        "variant_candidates": list(var_cands.values()),
        "variant_requirements": list(var_reqs.values()),
        "heldout_families": held_fams,
        "rq_split": rq_split,
        "cd_split": cd_split,
        "soc_split": soc_split,
        "stats": dict(stats),
        "tally": tally,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, Counter] = {}
    for r in rows:
        out.setdefault(r["split"], Counter())[r["bucket"]] += 1
    return {s: dict(c) | {"total": sum(c.values())} for s, c in sorted(out.items())}


def write_outputs(result: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    def dump(name: str, items: list[dict]) -> None:
        with open(out_dir / name, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    dump("manifest.jsonl", result["rows"])
    dump("variant_candidates.jsonl", result["variant_candidates"])
    dump("variant_requirements.jsonl", result["variant_requirements"])
    by_split: dict[str, dict[str, list[str]]] = {}
    for r in result["rows"]:
        d = by_split.setdefault(r["split"], {"pair_ids": []})
        d["pair_ids"].append(r["pair_id"])
    summary = {
        "heldout_families": result["heldout_families"],
        "counts": summarize(result["rows"]),
        "bucket_totals": dict(Counter(r["bucket"] for r in result["rows"])),
        "requirement_split": result["rq_split"],
        "candidate_split": result["cd_split"],
        "occupation_split": result.get("soc_split", {}),
        "pair_ids": {s: d["pair_ids"] for s, d in by_split.items()},
    }
    (out_dir / "splits.json").write_text(json.dumps(summary, indent=1))
    return summary
