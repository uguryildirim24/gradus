"""Clean real-posting rows from the hand-checked cards: relabel, break the profile shortcut, train_mix2.csv and a
checked real dev set.

The real_train pairs (build_real_train.py) carry labels from rules.label on Adaption's extracted card, and their
candidates are built to meet that card. This script keeps only the rows of postings whose card was checked by hand
and rebuilds what those rows can teach:

  1. relabel   every rendered row (data/built/real_train/{train,dev}.jsonl, before the screen) on a checked posting:
               rules.label on (candidate card, checked card), the rendered resume kept as is. Then the build's own
               gates, in order: person_problems; build_real_test.guard for every kind the new reasons name (a Skip
               that is now a Keep never went through the guard as a Keep); the text screens of
               assemble_real_train for non-Skip rows; and visibility (build_real_test.readable_misses on the full
               prompt, plus resume_month_ok). Train drops a row any gate rejects. Dev drops posting-side misses, as
               the test build does, and keeps a resume-side miss with decidable False, as test.jsonl does.
  2. shuffle   the profile fields no resume shows (render_cards.RESUME_FIELDS: willing_locations,
               remote_preference, employment_types_wanted, salary_floor, industries_wanted, industries_avoided,
               roles_avoided) are drawn again for every row from fixed seeded distributions that do not look at the
               row's variant kind or label (draw_hidden). work_auth and current_location stay: the resume shows the
               education a status implies and the city. A drawn preference whose reason the shown text does not
               let a reader check (an industry the posting never names, pay it never states) is set back to its
               default, as the build only sets a preference when the posting shows its fact. Then relabel and the
               same gates.
  3. assemble  train_mix2.csv: smoke.csv (6,891 rows, md5 checked) plus the cleaned real train rows, downsampled to
               the smallest label (seeded), one seeded shuffle, prompt and completion only. checked_real_dev.jsonl:
               the cleaned real dev rows in test.jsonl's keys. checked_real_dev_builder.jsonl: the same dev rows
               after step 1 only (the profile as the builder made it, as on the real test), for reference.
  4. report    the ask gap of the cleaned Keep rows against the synthetic Keeps (select_cardcheck.ask_gaps, the
               diagnosis's measure), the leakage and duplicate checks (assemble_real_train.checks; a leak stops the
               build), employer overlap, and the shortcut guessers (leakage.run_baselines) before and after step 2.

Checked cards (load_checked):
  - data/built/real_train/cardcheck/checked_{1..4}.jsonl: 162 postings (120 train, 42 dev). A posting a checker
    flagged "drop" is left out (a Canadian posting, a program for high-school graduates).
  - the 28 real dev postings of data/built/audit/fresh_checked_cards.jsonl. That file holds the cards as they were
    read, not corrected: the truth is score_extraction.hand_truth, the card with every "wrong" and "debatable"
    verdict of fresh_verdicts.jsonl applied (the project's own definition for those postings).
  nice_skills (--nice): "extracted" (default) keeps the extracted card's nice list, moved only where a checker moved
  an item between must and nice. The test cards' nice lists are the extractor's (the audit never judged that
  field), and the candidates were built to the extracted list. "checked" takes the checkers' nice lists as written.

  clean_real_checked.py build [--seed 0] [--nice extracted|checked] [--draws 1] [--no-checks]
      Writes data/built/real_train/clean_checked/{cards.jsonl, rows.jsonl, report.json},
      data/built/real_train/train_mix2.csv, checked_real_dev.jsonl and checked_real_dev_builder.jsonl.
      --draws K draws K profiles per row (K > 1 multiplies rows; the default 1 is what is written up).
      Deterministic, local files only; loads no model and calls nothing.

  clean_real_checked.py dev-v2 [--dry-run]
      checked_real_dev_v2_builder.jsonl (test-like, the primary real-posting selection set) and
      checked_real_dev_v2.jsonl (profile-redrawn): the two v1 dev files without every posting that has a
      restriction the card cannot hold (RESTRICTIONS: one school, class standing, a graduation window, a degree
      level the rules do not encode, must be local; a degree ceiling or experience cap only when a non-Skip row's
      candidate is past it). Each of the 64 v1 dev postings was read once, with the checkers' flags; a posting
      in neither RESTRICTIONS nor READ_CLEAR stops it. Rows are copied byte for byte. Also writes
      clean_checked/dev_v2/{restrictions.jsonl, prompt_identical_to_v1.json, report.json}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import assemble_real_train as art  # noqa: E402
import build_real_test as brt  # noqa: E402
from matcher.metros import ANYWHERE_US, places_match  # noqa: E402
from matcher.pairs import build_prompt, prompt_parts  # noqa: E402
from matcher.profile import render_profile  # noqa: E402
from matcher.rules import format_completion, industry_ids, is_enrolled, label, location_ok, \
    reasons_by_check  # noqa: E402
from matcher.sample import INDUSTRIES, INDUSTRY_W, METRO_NAMES, ROLE_W, TYPE_W, sample_willing, wanted_role_words, \
    weighted_sample  # noqa: E402
from matcher.schema import DEGREE_LEVELS, EMPLOYMENT_TYPES, CandidateCard, RequirementCard, skill_canon  # noqa: E402
from matcher.variants import _ind_own  # noqa: E402

RT = ROOT / "data/built/real_train"
CARDCHECK = RT / "cardcheck"
OUT = RT / "clean_checked"
V2 = ROOT / "data/built/cards/extracted_real_cards_v2.jsonl"
FRESH_CARDS = ROOT / "data/built/audit/fresh_checked_cards.jsonl"
FRESH_VERDICTS = ROOT / "data/built/audit/fresh_verdicts.jsonl"
SPLITS_FILE = RT / "splits.json"
CSV_OUT = RT / "train_mix2.csv"
DEV_OUT = RT / "checked_real_dev.jsonl"
DEV_BUILDER_OUT = RT / "checked_real_dev_builder.jsonl"
SPLITS = ("train", "dev")
SKIP = "Excluded"
WORD = {"For you": "Keep", "Explore": "Explore", "Excluded": "Skip"}
TEST_KEYS = art.TEST_KEYS

# The profile fields no rendered resume shows (render_cards.RESUME_FIELDS writes the resume from field, degree,
# graduation year, years, skills, licenses, current_location, education_country and language).
HIDDEN = ("willing_locations", "remote_preference", "employment_types_wanted", "salary_floor", "industries_wanted",
          "industries_avoided", "roles_avoided")
DEFAULT = {"willing_locations": (), "remote_preference": "any", "employment_types_wanted": (), "salary_floor": None,
           "industries_wanted": (), "industries_avoided": (), "roles_avoided": ()}

# draw_hidden's rates. A preference is mostly one the posting meets, so a set value says little about the label:
# the builder set a field only in the variant whose label it decides.
P_TYPES = 0.45  # a job-type list instead of "any"
P_TYPES_MISS = 0.2  # of those, a list without the posting's type
P_REMOTE = {"onsite": 0.10, "hybrid": 0.30, "remote": 0.40}  # a hybrid-or-remote or remote-only preference
P_FLOOR = 0.4
P_FLOOR_OVER = 0.25  # of the floors on a posting that states pay, one above its top
P_WANT = 0.4
P_WANT_MISS = 0.2  # of the wanted lists, one without the posting's industry
P_AVOID = 0.18
P_AVOID_HIT = 0.2  # of the avoided industries, the posting's own
P_ROLE = 0.08
P_MOVE_THERE = 0.5  # a candidate who lives away from the posting lists one of its places


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def _h(*parts) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()


# ---------------------------------------------------------------- checked cards


def _flat(groups) -> set[str]:
    return {skill_canon(x) for g in groups for x in g}


def nice_extracted(checked: dict, extracted: dict) -> list[str]:
    """The extracted card's nice list, with the checker's moves between must and nice: an extracted nice item the
    checked card requires leaves it, and an extracted must item the checker moved to nice joins it."""
    must = _flat(checked["must_skills"])
    out = [s for s in extracted["nice_skills"] if skill_canon(s) not in must]
    have = {skill_canon(s) for s in out}
    was_must = _flat(extracted["must_skills"])
    for s in checked["nice_skills"]:
        k = skill_canon(s)
        if k in was_must and k not in must and k not in have:
            out.append(s)
            have.add(k)
    return out


def drop_flag(flags) -> str | None:
    for f in flags or ():
        if isinstance(f, dict) and f.get("field") == "drop":
            return f.get("note") or f.get("quote") or "drop"
    return None


def load_checked(nice: str = "extracted", cardcheck: Path = CARDCHECK, v2: Path = V2, fresh_cards: Path = FRESH_CARDS,
                 fresh_verdicts: Path = FRESH_VERDICTS, splits: dict[str, str] | None = None) -> tuple[dict, dict]:
    """(posting id -> {card, split, source}, report). Cards are RequirementCards; `nice` picks the nice list."""
    import score_extraction as se
    ext = {r["id"]: r["card"] for r in read_jsonl(v2)}
    splits = json.loads(SPLITS_FILE.read_text()) if splits is None else splits
    out, dropped, rep = {}, {}, Counter()
    for p in sorted(cardcheck.glob("checked_*.jsonl")):
        for r in read_jsonl(p):
            i = r["posting_id"]
            if (why := drop_flag(r.get("flags"))):
                dropped[i] = why
                continue
            c = dict(r["card"])
            if nice == "extracted":
                c["nice_skills"] = nice_extracted(c, ext[i])
            out[i] = {"card": RequirementCard.from_dict(c), "split": r["split"], "source": p.name}
            rep[f"{p.name}:{r['split']}"] += 1
    fresh = {r["id"]: r["card"] for r in read_jsonl(fresh_cards)}
    truth = se.hand_truth(read_jsonl(fresh_verdicts), fresh) if fresh else {}
    for i, c in sorted(truth.items()):
        if splits.get(i) != "dev" or i in out:
            continue
        c = dict(c)
        # nice_skills was never judged: the extracted card's list the candidates were built to, with the moves
        c["nice_skills"] = nice_extracted({**c, "nice_skills": c["nice_skills"]}, ext[i])
        out[i] = {"card": RequirementCard.from_dict(c), "split": "dev", "source": "fresh_verdicts"}
        rep["fresh_verdicts:dev"] += 1
    wrong_split = sorted(i for i, v in out.items() if splits.get(i) != v["split"])
    if wrong_split:
        raise SystemExit(f"checked postings whose split differs from splits.json: {wrong_split[:5]}")
    return out, {"by_source": dict(rep), "dropped": dropped, "nice": nice, "postings": len(out)}


# ---------------------------------------------------------------- gates


def effective_kinds(c: CandidateCard, card: RequirementCard, bucket: str, reasons: list[str]) -> set[str]:
    """The variant kinds whose guards fit this label: the kind a row would have been built as, read off its reasons
    (a full match that now misses a checked must skill is a must_skill row)."""
    owner = reasons_by_check(c, card)
    kinds = set()
    for r in reasons:
        check = owner.get(r)
        if check == "employment_type" and bucket == SKIP and "you want" in r:
            kinds.add("employment_type")
        elif check == "skills" and (r.startswith("missing required skill") or r.startswith("needs one of")):
            kinds.add("must_skill")
        elif check == "skills" and r.startswith("missing nice-to-have"):
            kinds.add("nice_skill")
        elif check == "location":
            if r.startswith("based in "):
                kinds.add("location")
            elif r.startswith("location works") and not location_ok(replace(c, willing_locations=()), card):
                kinds.add("relocate")
            elif r.endswith("matches your preference"):
                kinds.add("pref_mode_match")  # its guard: the posting also names a mode the preference rejects
        elif check == "enrollment" and r.startswith("for students"):
            kinds.add("enrollment_graduated" if "you have graduated" in r else "enrollment_level")
        elif check == "degree":
            if r.startswith("degree field should be"):
                kinds.add("degree_field_adjacent" if r.endswith("is close") else "degree_field")
            elif r.startswith("needs a finished"):
                kinds.add("degree_still_enrolled")
            elif r.startswith("needs "):
                kinds.add("degree_level")
        elif check == "years" and "overqualified" in r:
            kinds.add("overqualified")
        elif check == "preferences" and "industry you said you avoid" in r:
            kinds.add("pref_industry_avoided")
    return kinds


def guard_all(kinds: set[str], c: CandidateCard, card: RequirementCard, bucket: str, shown: str) -> str | None:
    """build_real_test.guard for each kind (and once with a kind of its own, for the kind-free checks)."""
    for k in sorted(kinds | {"full_match"}):
        if (why := brt.guard(k, c, card, bucket, shown)):
            return f"{k}: {why}"
    return None


def screen_all(kinds: set[str], c: CandidateCard, raw: dict, resume: str, completion: str) -> tuple[str, str] | None:
    """assemble_real_train's screens on a non-Skip row, with the kinds its reasons name."""
    text = raw.get("description", "")
    checks = (
        ("location", lambda: art.screen_location(c, raw)),
        ("student_only", lambda: art.screen_student_only(c, text)),
        ("class_standing", lambda: art.screen_class_standing(c, text)),
        ("grad_window", lambda: art.screen_grad_window(c, text)),
        ("prior_internship", lambda: art.screen_prior_internship(resume, text)),
        ("school_only", lambda: art.screen_school_only(resume, text)),
        ("permanent_auth", lambda: art.screen_permanent_auth(c, text)),
        ("no_sponsor", lambda: art.screen_no_sponsor(c, text)),
        ("citizen_only", lambda: art.screen_citizen_only(c, text)),
        ("industry_conflict", lambda: art.screen_industry(
            c, raw, "pref_industry_avoided" if "pref_industry_avoided" in kinds else "")),
        ("soft_nice_skill", lambda: art.screen_soft_nice("nice_skill" if "nice_skill" in kinds else "", completion)),
        ("broad_degree_field", lambda: art.screen_broad_degree(
            c, "degree_field_adjacent" if "degree_field_adjacent" in kinds else "", text)),
    )
    for name, f in checks:
        if (why := f()):
            return name, why
    return None


def visibility(c: CandidateCard, card: RequirementCard, bucket: str, reasons: list[str], prompt: str):
    """(posting-side misses, every miss) for the full prompt, as build_rows and assemble_rows check them."""
    resume, profile, shown = prompt_parts(prompt)
    hard = bucket == SKIP
    post = brt.posting_misses(c, card, reasons, profile, shown, hard=hard)
    full = brt.readable_misses(c, card, reasons, resume, profile, shown, hard=hard)
    if bucket != SKIP and (why := brt.resume_month_ok(resume, shown, c.graduation_year)):
        full.append(("enrollment", why))
    return post, full


def gate(c: CandidateCard, card: RequirementCard, raw: dict, prompt: str, split: str) -> dict:
    """Label and run the gates. Returns the label fields plus stage: kept, or the gate that dropped the row."""
    bucket, reasons, flags = label(c, card)
    out = {"bucket": bucket, "reasons": reasons, "flags": flags, "completion": format_completion(bucket, reasons),
           "decidable": True, "missing": []}
    if (pp := brt.person_problems(c)):
        return out | {"stage": "person_problems", "detail": pp}
    resume, profile, shown = prompt_parts(prompt)
    kinds = effective_kinds(c, card, bucket, reasons)
    if (why := guard_all(kinds, c, card, bucket, shown)):
        return out | {"stage": "guard", "detail": why}
    if bucket != SKIP and (hit := screen_all(kinds, c, raw, resume, out["completion"])):
        return out | {"stage": f"screen:{hit[0]}", "detail": hit[1]}
    post, full = visibility(c, card, bucket, reasons, prompt)
    if post:
        return out | {"stage": "posting_does_not_state", "detail": [list(m) for m in post][:4]}
    if full:
        if split == "train":
            return out | {"stage": "not_decidable", "detail": [list(m) for m in full][:4]}
        out |= {"decidable": False, "missing": [list(m) for m in full]}
    return out | {"stage": "kept", "detail": ""}


# ---------------------------------------------------------------- the hidden profile


def posting_metro(loc: str) -> str:
    """The metro name a candidate would list for a posting place (a METRO_NAMES entry it matches), else the place."""
    return next((m for m in METRO_NAMES if places_match(m, loc)), loc)


def _sorted_types(types) -> tuple[str, ...]:
    return tuple(t for t in EMPLOYMENT_TYPES if t in types)


def draw_hidden(c: CandidateCard, card: RequirementCard, rng: random.Random, shown: str = "",
                facts: dict[str, bool] | None = None) -> CandidateCard:
    """The candidate with every HIDDEN field drawn again. The draws look at the posting (its type, mode, pay,
    industry, places and text) but never at the row's kind or label. Lists are written in a fixed order, so the
    posting's own item is never first by construction.

    `facts` (shown_facts) says which preference facts the shown posting states; a preference whose fact it does not
    state stays at its default, for every row of the posting (the build sets a preference only when the posting
    shows the fact it is compared with). A wanted job type the posting names is never the one that fails it
    (build_real_test.guard drops that row)."""
    facts = facts if facts is not None else {}
    t = card.employment_type
    types: tuple[str, ...] = ()
    if rng.random() < P_TYPES and facts.get("employment_types_wanted", True):
        others = {x: TYPE_W[x] for x in EMPLOYMENT_TYPES if x != t}
        unnamed = {x: w for x, w in others.items() if not re.search(brt.TYPE_PHRASE[x], shown, re.I)}
        if rng.random() < P_TYPES_MISS and unnamed:
            types = weighted_sample(rng, unnamed, rng.choice((1, 1, 2)))
        else:
            types = (t, *weighted_sample(rng, others, 1)) if rng.random() < 0.4 else (t,)
    types = _sorted_types(types)

    remote = "any"
    if rng.random() < P_REMOTE.get(card.work_mode, 0.1) and facts.get("remote_preference", True):
        remote = rng.choice(("hybrid_or_remote", "remote_only"))

    floor = None
    if rng.random() < P_FLOOR and facts.get("salary_floor", True):
        if card.salary_max:
            over = rng.random() < P_FLOOR_OVER
            mult = rng.uniform(1.05, 1.5) if over else rng.uniform(0.5, 0.95)
            floor = max(5000, int(card.salary_max * mult / 5000) * 5000)
            if over and floor <= card.salary_max:
                floor = (card.salary_max // 5000 + 1) * 5000
        else:
            floor = int(rng.uniform(35000, 110000) / 5000) * 5000

    own = _ind_own(card) if card.industry else ""
    own_ids = industry_ids(card.industry) if card.industry else frozenset()
    other = {i: w for i, w in INDUSTRY_W.items() if not (industry_ids(i) & own_ids)}
    wanted: tuple[str, ...] = ()
    if rng.random() < P_WANT and facts.get("industries_wanted", True):
        n = rng.choice((1, 2, 2, 3))
        if not own_ids or rng.random() < P_WANT_MISS:
            wanted = weighted_sample(rng, other if own_ids else INDUSTRY_W, n)
        else:
            wanted = (own, *weighted_sample(rng, other, n - 1))
        wanted = tuple(sorted(dict.fromkeys(wanted)))
    wanted_ids = {k for x in wanted for k in industry_ids(x)}
    avoided: tuple[str, ...] = ()
    if rng.random() < P_AVOID and facts.get("industries_avoided", True):
        if own_ids and rng.random() < P_AVOID_HIT and not (own_ids & wanted_ids):
            avoided = (own,)
        else:
            pool = {i: w for i, w in other.items() if not (industry_ids(i) & wanted_ids) and i in INDUSTRIES}
            avoided = weighted_sample(rng, pool, 1)

    roles = weighted_sample(rng, ROLE_W, 1, wanted_role_words(types, c.work_auth)) if rng.random() < P_ROLE else ()

    away = card.work_mode != "remote" and bool(card.locations) and \
        not location_ok(replace(c, willing_locations=()), card)
    there = [posting_metro(x) for x in card.locations]
    if away and rng.random() < P_MOVE_THERE:
        willing = [w for w in sample_willing(rng, c.current_location) if w != ANYWHERE_US][:2]
        willing.insert(rng.randrange(len(willing) + 1), rng.choice(there))
        willing = list(dict.fromkeys(willing))
    else:
        near = (lambda m: any(places_match(m, x) for x in card.locations)) if away else (lambda m: False)
        willing = list(sample_willing(rng, c.current_location, avoid=near))
        if away and willing == [ANYWHERE_US]:
            willing = []  # "anywhere" would meet the posting: that is the branch above
    return replace(c, willing_locations=tuple(willing), remote_preference=remote, employment_types_wanted=types,
                   salary_floor=floor, industries_wanted=wanted, industries_avoided=avoided, roles_avoided=roles)


PROBES = ("employment_types_wanted", "remote_preference", "salary_floor", "industries_wanted")


def shown_facts(c: CandidateCard, card: RequirementCard, resume: str, posting: str) -> dict[str, bool]:
    """Which preference facts a reader of this prompt can check: for each probe (the posting's own job type wanted,
    a hybrid-or-remote preference, a floor at half the posting's top pay, the posting's own industry wanted), is
    the reason it gives readable (build_real_test.readable_misses), whatever the row's label? A probe that gives no reason (no pay stated,
    no industry the rule knows) is not shown. industries_avoided follows industries_wanted (the same fact)."""
    base = replace(c, **DEFAULT)
    probe = {"employment_types_wanted": (card.employment_type,), "remote_preference": "hybrid_or_remote",
             "salary_floor": max(5000, (card.salary_max or 0) // 2) if card.salary_max else None,
             "industries_wanted": (_ind_own(card),) if card.industry else ()}
    out = {}
    for f, v in probe.items():
        if v in (None, ()):
            out[f] = False
            continue
        p = replace(base, **{f: v})
        # every reason the checks write, not only the label's: a Skip for another reason hides the good ones, and
        # the fact must not depend on the row's label
        owner = reasons_by_check(p, card)
        mine = [r for r, chk in owner.items() if f in hidden_fields_of(r, chk)]
        prompt = build_prompt(resume, render_profile(p), posting)
        r_text, profile, shown = prompt_parts(prompt)
        out[f] = bool(mine) and not brt.readable_misses(p, card, mine, r_text, profile, shown, hard=False)
    out["industries_avoided"] = out["industries_wanted"]
    return out


def hidden_fields_of(reason: str, check: str) -> tuple[str, ...]:
    """The HIDDEN fields a reason rests on (none for a reason the resume or the posting alone decides)."""
    if check == "employment_type":
        return ("employment_types_wanted",)
    if check == "preferences":
        if "pay" in reason or "floor" in reason:
            return ("salary_floor",)
        if "you said you avoid" in reason and reason.startswith("title includes"):
            return ("roles_avoided",)
        if "avoid" in reason:
            return ("industries_avoided",)
        return ("industries_wanted",)
    if check == "location" and ("preference" in reason or "remote only" in reason or "prefer hybrid" in reason):
        return ("remote_preference",)
    return ()


def settle_hidden(c: CandidateCard, card: RequirementCard, resume: str, posting: str) -> tuple[CandidateCard, list[str]]:
    """(the candidate, fields set back): a drawn preference whose reason the shown text does not let a reader check
    goes back to its default, until the row reads or no hidden field is left to blame."""
    reset: list[str] = []
    for _ in range(len(HIDDEN)):
        bucket, reasons, _ = label(c, card)
        prompt = build_prompt(resume, render_profile(c), posting)
        r_text, profile, shown = prompt_parts(prompt)
        if not brt.readable_misses(c, card, reasons, r_text, profile, shown, hard=bucket == SKIP):
            break
        owner = reasons_by_check(c, card)
        blame = []
        for r in reasons:
            fields = [f for f in hidden_fields_of(r, owner.get(r, "")) if getattr(c, f) != DEFAULT[f]]
            if fields and brt.readable_misses(c, card, [r], r_text, profile, shown, hard=False):
                blame += fields
        if not blame:
            break
        c = replace(c, **{f: DEFAULT[f] for f in dict.fromkeys(blame)})
        reset += blame
    return c, reset


# ---------------------------------------------------------------- the pipeline


def load_inputs():
    from build_real_train import ctx_of
    from render_cards import texts_by_ctx
    rows = {s: read_jsonl(RT / f"{s}.jsonl") for s in SPLITS}
    cands = {r["pair_id"]: r["candidate"] for r in read_jsonl(RT / "rows.jsonl")}
    raw = art.brt_raw()
    store = texts_by_ctx("resumes")
    return rows, cands, raw, store, ctx_of


def relabel_rows(rows: dict[str, list[dict]], cands: dict[str, dict], cards: dict, raw: dict, resume_of,
                 seed: int = 0, draws: int = 1) -> tuple[list[dict], list[dict]]:
    """(step 1 records, step 2 records): one record per input row (step 1) and per draw (step 2), kept or not."""
    step1, step2 = [], []
    taken: set[str] = set()  # new pair ids: two rows of one posting can draw into one candidate
    for s in SPLITS:
        for r in rows[s]:
            i = r["rq_id"]
            if i not in cards:
                continue
            card = cards[i]["card"]
            c0 = CandidateCard.from_dict(cands[r["pair_id"]])
            posting = brt.posting_text(raw[i])
            resume = resume_of(c0)
            prompt0 = build_prompt(resume, render_profile(c0), posting)
            base = {"old_pair_id": r["pair_id"], "rq_id": i, "base_rq_id": r["base_rq_id"], "split": s,
                    "variant_kind": r["variant_kind"], "old_bucket": r["bucket"], "old_completion": r["completion"]}
            g = gate(c0, card, raw[i], prompt0, s)
            step1.append(base | {"pair_id": r["pair_id"], "cd_id": r["cd_id"], "candidate": c0.to_dict(),
                                 "prompt": prompt0, "step1_bucket": g["bucket"]} | g)
            facts = shown_facts(c0, card, resume, posting)
            shown = prompt_parts(prompt0)[2]
            for k in range(draws):
                rng = random.Random(f"{seed}:{r['pair_id']}:profile:{k}")
                c = draw_hidden(c0, card, rng, shown, facts)
                c, reset = settle_hidden(c, card, resume, posting)
                cd = brt.cd_id_of(c)
                pid = brt.pair_id_of(i, cd)
                prompt = build_prompt(resume, render_profile(c), posting)
                g2 = gate(c, card, raw[i], prompt, s)
                if pid in taken:
                    g2 = g2 | {"stage": "duplicate", "detail": "another row drew the same candidate"}
                taken.add(pid)
                step2.append(base | {"pair_id": pid, "cd_id": cd, "draw": k, "reset": reset,
                                     "candidate": c.to_dict(), "prompt": prompt, "step1_bucket": g["bucket"]} | g2)
    return step1, step2


def balance(rows: list[dict], seed: int = 0) -> list[dict]:
    """Every label cut to the smallest label's count, by a seeded hash order spread over postings: rows are taken
    round-robin over postings (postings and rows in hash order), so a cut label keeps as many postings as it can."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["bucket"]].append(r)
    if not by:
        return []
    n = min(len(v) for v in by.values())
    keep = []
    for b, rs in sorted(by.items()):
        posts: dict[str, list[dict]] = defaultdict(list)
        for r in sorted(rs, key=lambda r: _h(seed, "row", r["pair_id"])):
            posts[r["rq_id"]].append(r)
        queues = [posts[p] for p in sorted(posts, key=lambda p: _h(seed, "post", b, p))]
        got = []
        while len(got) < n:
            for q in queues:
                if q and len(got) < n:
                    got.append(q.pop(0))
        keep += got
    ids = {r["pair_id"] for r in keep}
    return [r for r in rows if r["pair_id"] in ids]


def confusion(recs: list[dict], a: str, b: str) -> dict:
    out: dict[str, dict] = {}
    for s in SPLITS:
        c = Counter(f"{WORD[r[a]]} -> {WORD[r[b]]}" for r in recs if r["split"] == s)
        out[s] = dict(sorted(c.items()))
    return out


def gate_counts(recs: list[dict]) -> dict:
    out = {}
    for s in SPLITS:
        sub = [r for r in recs if r["split"] == s]
        kept = [r for r in sub if r["stage"] == "kept"]
        out[s] = {"rows": len(sub), "kept": len(kept), "postings_kept": len({r["rq_id"] for r in kept}),
                  "dropped_by_stage": dict(Counter(r["stage"].split(":")[0] if r["stage"].startswith("screen") else
                                                   r["stage"] for r in sub if r["stage"] != "kept").most_common()),
                  "dropped_by_screen": dict(Counter(r["stage"] for r in sub if r["stage"].startswith("screen")).most_common()),
                  "guard_reasons": dict(Counter(r["detail"].split(": ", 1)[1][:90] for r in sub
                                                if r["stage"] == "guard").most_common(8)),
                  "kept_labels": dict(Counter(WORD[r["bucket"]] for r in kept)),
                  "kept_decidable_labels": dict(Counter(WORD[r["bucket"]] for r in kept if r["decidable"]))}
    return out


def field_by_label(recs: list[dict]) -> dict:
    """For each HIDDEN field: the share of rows with a non-default value, per label (kept rows). Before the shuffle
    a set field sits almost only in one label; after it, in all three."""
    out = {}
    for f in HIDDEN:
        row = {}
        for b in ("For you", "Explore", "Excluded"):
            sub = [r for r in recs if r["bucket"] == b]
            row[WORD[b]] = round(sum(_set(r["candidate"][f], f) for r in sub) / max(len(sub), 1), 3)
        out[f] = row
    return out


def _set(v, f) -> bool:
    d = DEFAULT[f]
    return (tuple(v) if isinstance(v, list) else v) != d


# ---------------------------------------------------------------- guessers


# render_profile's lines for the HIDDEN fields; the other lines (education, years, work authorization, city) say
# what the resume says, or the status its education implies
HIDDEN_LINES = ("Willing to relocate to:", "Work arrangement:", "Job types wanted:", "Minimum salary:",
                "Industries wanted:", "Industries to avoid:", "Roles to avoid:")
PROFILE_HIDDEN, PROFILE_OTHER = "profile_hidden", "profile_other"


def profile_lines(profile: str, hidden: bool) -> str:
    return "\n".join(x for x in profile.split("\n") if x.startswith(HIDDEN_LINES) == hidden)


def frame(rows: list[dict]):
    import pandas as pd
    from matcher import leakage as L
    parts = [prompt_parts(r["prompt"]) for r in rows]
    df = pd.DataFrame({L.ID: [r["pair_id"] for r in rows], L.LABEL: [r["bucket"] for r in rows],
                       L.RESUME: [p[0] for p in parts], L.PROFILE: [p[1] for p in parts],
                       L.POSTING: [p[2] for p in parts]})
    return df


def profile_ablation(train: list[dict], dev: list[dict]) -> dict:
    """Profile-only guessers on the HIDDEN lines alone and on the other lines alone, train -> dev."""
    from matcher import leakage as L
    tr, dv = frame(train), frame(dev)
    out = {}
    for name, hidden in (("BoW: profile hidden-field lines only", True), ("BoW: profile other lines only", False)):
        a, b = tr.copy(), dv.copy()
        a[L.PROFILE] = a[L.PROFILE].map(lambda t: profile_lines(t, hidden))
        b[L.PROFILE] = b[L.PROFILE].map(lambda t: profile_lines(t, hidden))
        out[name] = round(L.macro_f1(b[L.LABEL], L._fit_predict_text(a, b, [L.PROFILE])), 3)
    return out


def _short(rows: list[dict]) -> dict:
    keep = ("majority class", "BoW: resume only", "BoW: profile only", "BoW: posting only", "BoW: resume + profile",
            "BoW: resume + profile + posting", "shuffled-posting control (full BoW)")
    return {r["baseline"]: round(r["macro_f1"], 3) for r in rows if r["baseline"] in keep}


def grouped_cv(rows: list[dict], raw: dict, folds: int = 5, seed: int = 0) -> dict:
    """leakage.run_baselines in `folds` folds over every cleaned row (train and dev together), each employer in one
    fold, macro-F1 pooled over the folds' predictions."""
    from matcher import leakage as L
    from matcher.nearcopy import company_key
    from sklearn.metrics import f1_score
    import numpy as np
    co = {r["rq_id"]: company_key(raw[r["rq_id"]].get("company", "")) or r["rq_id"] for r in rows}
    order = sorted(set(co.values()), key=lambda k: _h(seed, "fold", k))
    fold = {k: j % folds for j, k in enumerate(order)}
    df = frame(rows)
    df["fold"] = [fold[co[r["rq_id"]]] for r in rows]
    df[PROFILE_HIDDEN] = df[L.PROFILE].map(lambda t: profile_lines(t, True))
    df[PROFILE_OTHER] = df[L.PROFILE].map(lambda t: profile_lines(t, False))
    preds: dict[str, list] = defaultdict(lambda: [None] * len(df))
    for k in range(folds):
        tr, dv = df[df.fold != k].reset_index(drop=True), df[df.fold == k].reset_index(drop=True)
        idx = np.where(df.fold == k)[0]
        for name, cols in (("BoW: profile only", [L.PROFILE]), ("BoW: profile hidden-field lines only", [PROFILE_HIDDEN]),
                           ("BoW: profile other lines only", [PROFILE_OTHER]), ("BoW: resume only", [L.RESUME]),
                           ("BoW: posting only", [L.POSTING]), ("BoW: resume + profile + posting",
                                                                [L.RESUME, L.PROFILE, L.POSTING])):
            p = L._fit_predict_text(tr, dv, cols)
            for j, v in zip(idx, p):
                preds[name][j] = v
    y = df[L.LABEL].tolist()
    return {name: round(float(f1_score(y, p, average="macro", zero_division=0)), 3) for name, p in preds.items()}


def old_shortcut_transfer(dev_sets: dict[str, list[dict]]) -> dict:
    """The profile-only guesser trained on the old real train (screened_train.jsonl, 4,350 rows, the builder's
    profiles), scored on each dev set: it reads the builder's signatures, so it falls when they are gone."""
    from matcher import leakage as L
    old = read_jsonl(RT / "screened_train.jsonl")
    if not old:
        return {}
    tr = frame(old)
    out = {}
    for name, rows in dev_sets.items():
        dv = frame(rows)
        out[name] = round(L.macro_f1(dv[L.LABEL], L._fit_predict_text(tr, dv, [L.PROFILE])), 3)
    return out


# ---------------------------------------------------------------- ask gap


def gap_stats(gaps: list[int]) -> dict:
    if not gaps:
        return {"rows": 0}
    g = sorted(gaps)
    return {"rows": len(g), "over_4": round(sum(x > 4 for x in g) / len(g), 3),
            "over_7": round(sum(x > 7 for x in g) / len(g), 3), "median": g[len(g) // 2],
            "mean": round(sum(g) / len(g), 2)}


def ask_gap_report(real: dict[str, list[dict]], synth: list[dict]) -> dict:
    import select_cardcheck as sc
    rows = {k: [r for r in v if r["bucket"] == "For you"] for k, v in real.items()}
    syn = [{"pair_id": f"synth_{j}", "prompt": r["prompt"]} for j, r in enumerate(synth)
           if r["completion"].startswith("Keep")]
    allrows = [r for v in rows.values() for r in v] + syn
    gaps = sc.ask_gaps(allrows)
    out = {k: gap_stats([gaps[r["pair_id"]] for r in v]) for k, v in rows.items()}
    out["synthetic_keep"] = gap_stats([gaps[r["pair_id"]] for r in syn])
    return out


# ---------------------------------------------------------------- build


def employer_overlap(train: list[dict], dev: list[dict], raw: dict) -> dict:
    from matcher.nearcopy import company_key
    test = read_jsonl(art.TEST)
    co = lambda ids: {company_key(raw[i].get("company", "")) for i in ids} - {""}  # noqa: E731
    tr, dv = co({r["rq_id"] for r in train}), co({r["rq_id"] for r in dev})
    te = co({r["rq_id"] for r in test})
    return {"train_dev": sorted(tr & dv), "train_test": sorted(tr & te), "dev_test": sorted(dv & te)}


def lineage(rows: list[dict]) -> list[dict]:
    """The rows as assemble_real_train.checks reads them: its resume-context lookup goes by the old pair id (the
    resume fields, and so the context, are the old candidate's)."""
    return [r | {"pair_id": r["old_pair_id"]} for r in rows]


def build(a: argparse.Namespace) -> dict:
    splits = json.loads(SPLITS_FILE.read_text())
    cards, card_rep = load_checked(a.nice, splits=splits)
    rows, cands, raw, store, ctx_of = load_inputs()
    resume_of = lambda c: store.get(ctx_of(c.to_dict()))  # noqa: E731
    step1, step2 = relabel_rows(rows, cands, cards, raw, resume_of, a.seed, a.draws)
    other = None
    if a.nice == "extracted":  # the alternative, counted only: the checkers' nice lists as written
        alt, _ = load_checked("checked", splits=splits)
        other = Counter()
        for r in step1:
            b, _, _ = label(CandidateCard.from_dict(r["candidate"]), alt[r["rq_id"]]["card"])
            other[f"{r['split']}: {WORD[r['old_bucket']]} -> {WORD[b]}"] += 1

    kept1 = [r for r in step1 if r["stage"] == "kept"]
    kept2 = [r for r in step2 if r["stage"] == "kept"]
    train2 = [r for r in kept2 if r["split"] == "train"]
    dev2 = [r for r in kept2 if r["split"] == "dev"]
    dev1 = [r for r in kept1 if r["split"] == "dev"]
    train1 = [r for r in kept1 if r["split"] == "train"]
    real = balance(train2, a.seed)

    synth = art.read_synth()
    bad = art.frame_problems(real + dev2 + dev1, synth)
    if bad:
        raise SystemExit(f"{len(bad)} rows out of the synthetic frame, e.g. {bad[:3]}; nothing written")
    csv_rows = art.mix(synth, real, a.seed)

    rep = {"seed": a.seed, "nice": a.nice, "draws": a.draws, "cards": card_rep,
           "step1_relabel": {"label_change": confusion(step1, "old_bucket", "bucket"),
                             "label_change_nice_checked_alternative": dict(sorted(other.items())) if other else None,
                             "gates": gate_counts(step1)},
           "step2_shuffle": {"label_change_from_step1": confusion(step2, "step1_bucket", "bucket"),
                             "resets": dict(Counter(f for r in step2 for f in r["reset"])),
                             "gates": gate_counts(step2),
                             "set_share_by_label_before": field_by_label(kept1),
                             "set_share_by_label_after": field_by_label(kept2),
                             "rates": {k: v for k, v in globals().items() if k.startswith("P_")}},
           "mix": {"synthetic_rows": len(synth), "synthetic_md5": art.SYNTH_MD5,
                   "real_train_cleaned": dict(Counter(WORD[r["bucket"]] for r in train2)),
                   "real_train_in_csv": dict(Counter(WORD[r["bucket"]] for r in real)),
                   "real_train_postings_in_csv": len({r["rq_id"] for r in real}),
                   "csv_rows": len(csv_rows), "csv_labels": dict(Counter(c.split()[0] for _, c in csv_rows))},
           "checked_real_dev": {"rows": len(dev2), "postings": len({r["rq_id"] for r in dev2}),
                                "labels": dict(Counter(WORD[r["bucket"]] for r in dev2)),
                                "decidable": sum(r["decidable"] for r in dev2),
                                "decidable_labels": dict(Counter(WORD[r["bucket"]] for r in dev2 if r["decidable"]))},
           "checked_real_dev_builder": {"rows": len(dev1), "labels": dict(Counter(WORD[r["bucket"]] for r in dev1)),
                                        "decidable": sum(r["decidable"] for r in dev1)}}
    if a.draws > 1:
        rep["note"] = f"{a.draws} profile draws per row"

    # guessers: the same rows before and after the shuffle, train -> dev; all cleaned rows in employer folds; and
    # the old builder-trained profile guesser on the new rows
    rep["guessers"] = {
        "before_shuffle_train_to_dev": _short(art.baselines(train1, dev1)) | profile_ablation(train1, dev1),
        "after_shuffle_train_to_dev": _short(art.baselines(train2, dev2)) | profile_ablation(train2, dev2),
        "before_shuffle_employer_folds": grouped_cv(train1 + dev1, raw, seed=a.seed),
        "after_shuffle_employer_folds": grouped_cv(train2 + dev2, raw, seed=a.seed),
        "old_profile_guesser_on": old_shortcut_transfer({"checked_real_dev_builder": dev1, "checked_real_dev": dev2,
                                                         "cleaned_real_train": train2}),
        "random_guess": {"checked_real_dev": art.random_f1([r["bucket"] for r in dev2]),
                         "employer_folds_after": art.random_f1([r["bucket"] for r in train2 + dev2])}}
    rep["ask_gap_keep"] = ask_gap_report({"real_train_cleaned": train2, "real_train_in_csv": real,
                                          "checked_real_dev": dev2}, synth)
    rep["employer_overlap"] = employer_overlap(train2, dev2, raw)
    if not a.no_checks:
        ch = art.checks(lineage(real), lineage(dev2), synth, csv_rows)
        ch.pop("shortcuts", None)  # reported above, on these rows
        stop = art.problems_of(ch)
        test_prompts = {r["prompt"] for r in read_jsonl(art.TEST)}
        csv_prompts = {p for p, _ in csv_rows}
        b_prompts = Counter(r["prompt"] for r in dev1)
        ch["builder_dev"] = {"prompts_seen_twice": sum(v > 1 for v in b_prompts.values()),
                             "prompts_in_test": len(set(b_prompts) & test_prompts),
                             "prompts_in_csv": len(set(b_prompts) & csv_prompts)}
        stop += [f"builder_dev.{k}: {v}" for k, v in ch["builder_dev"].items() if v]
        if rep["employer_overlap"]["train_dev"]:
            stop.append(f"employer in train and dev: {rep['employer_overlap']['train_dev'][:5]}")
        rep["leakage"] = ch
        rep["leakage_problems"] = stop
        if stop:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "report.json").write_text(json.dumps(rep, indent=1))
            raise SystemExit(f"checks failed, nothing else written ({OUT / 'report.json'}): {stop}")

    OUT.mkdir(parents=True, exist_ok=True)
    brt.write_jsonl(OUT / "cards.jsonl", [{"id": i, "split": v["split"], "source": v["source"], "card": v["card"].to_dict()}
                                          for i, v in sorted(cards.items())])
    drop_prompt = lambda r: {k: v for k, v in r.items() if k != "prompt"}  # noqa: E731
    brt.write_jsonl(OUT / "rows_step1.jsonl", [drop_prompt(r) for r in step1])
    brt.write_jsonl(OUT / "rows.jsonl", [drop_prompt(r) | {"in_csv": r["pair_id"] in {x["pair_id"] for x in real}}
                                         for r in step2])
    rep["csv_md5"] = art.write_csv(CSV_OUT, csv_rows)
    brt.write_jsonl(DEV_OUT, [{k: r[k] for k in TEST_KEYS} for r in dev2])
    brt.write_jsonl(DEV_BUILDER_OUT, [{k: r[k] for k in TEST_KEYS} for r in dev1])
    rep["checked_real_dev"]["md5"] = art.md5(DEV_OUT)
    rep["checked_real_dev_builder"]["md5"] = art.md5(DEV_BUILDER_OUT)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("step1_relabel", "mix", "checked_real_dev", "guessers", "ask_gap_keep")},
                     indent=1, default=str))
    print(f"wrote {CSV_OUT.relative_to(ROOT)} ({rep['csv_md5']}), {DEV_OUT.relative_to(ROOT)}, "
          f"{DEV_BUILDER_OUT.relative_to(ROOT)}, {OUT.relative_to(ROOT)}/{{cards,rows,rows_step1}}.jsonl, report.json")
    return rep


# ---------------------------------------------------------------- checked real dev v2

# Every posting in the v1 dev files, read once (2026-10-20) against the restrictions a RequirementCard cannot hold.
# A posting with one of REMOVE_KINDS loses all its rows, Skips included, so the cut does not look at labels. A
# degree ceiling ("pursuing a bachelor's", "undergrad or master's") or an experience cap ("0-2 years") is exact on
# the candidate card, so it removes the posting only when a non-Skip row's candidate is past it (CHECK_KINDS). The
# source says who named it: "checker" (a flag in cardcheck/checked_*.jsonl), "read" (this read only) or "both".
REMOVE_KINDS = ("school_only", "class_standing", "grad_window", "degree_level", "must_be_local")
CHECK_KINDS = ("degree_ceiling", "exp_cap")
RESTRICTIONS: dict[str, tuple[str, str, str, object]] = {  # posting id: (kind, source, posting quote, limit)
    "ats_0caf3dfe-2e90-4b65-8cdb-cb79345145bc": (
        "school_only", "both", "ONLY CURRENTLY ENROLLED MIT STUDENTS WILL BE ELIGIBLE", None),
    "ats_72d70be2-7543-43d5-a8cd-9b80189884ad": (
        "school_only", "read", "ONLY CURRENTLY ENROLLED MIT STUDENTS WILL BE ELIGIBLE", None),
    "ats_69e6a885-ba62-4040-becf-35815be46d4e": (
        "school_only", "both", "Currently enrolled as a 2nd or 3rd year student at a leading Grande École", None),
    "ats_12a95e7d-c62e-4b8c-b8ff-491b854e7a20": ("class_standing", "both", "Senior or Grad student", None),
    "ats_d25191cb-2a6a-41b6-aebd-15d609a98d7b": (
        "class_standing", "both", "Junior, Senior or Graduate student pursuing a degree in engineering", None),
    "ats_318f1739-5d28-4330-afd4-1427be4f950e": (
        "class_standing", "both", "class standing of a sophomore or above ... must not have a graduation date prior to "
                                  "the start of the co-op term", None),
    "ats_46b35ec8-88e8-4800-ad1a-d17665e52fa1": (
        "class_standing", "read", "Must have a minimum of 30 hours of credit by the start of the internship", None),
    "ats_674b6824-6b78-406d-80dc-1f26ba47bbd1": (
        "class_standing", "read", "completion of at least two years toward a degree", None),
    "ats_9a42eb0e-d390-47e7-85fa-9d1494c41240": (
        "class_standing", "read", "Completion of Undergraduate Freshman year at an accredited University is required",
        None),
    "ats_a094bbba-b159-429f-905a-e52b9cadc157": (
        "class_standing", "read", "(graduate or upper-level undergraduate) student worker; Ability to commute on-site "
                                  "to Foster City", None),
    "ats_0503901e-622f-4d1b-b7f9-29d44e59eaab": (
        "grad_window", "read", "Must be planning on graduating in 2028. This should be your final internship", None),
    "ats_5370153e-8623-4e1d-b4f6-986a7ab410a1": (
        "grad_window", "both", "Must be able to begin full-time employment ... between January 2028 – August 2028",
        None),
    "ats_d7d2d9a2-c865-47e5-9029-23b28fb08f05": (
        "grad_window", "read", "Must be able to begin full time employment ... between January 2028 – August 2028",
        None),
    "ats_69f5edde-2904-4610-a7eb-8132ce17ead9": (
        "grad_window", "both", "Scheduled to obtain a Bachelor's degree ... by Summer 2028", None),
    "ats_e500e206-2c16-4d2e-ae52-93231ed58452": (
        "grad_window", "both", "Scheduled to obtain a Bachelor's degree ... by Summer 2028", None),
    "ats_c0f5bb29-73b3-4d6d-ae4a-8630ce8569e6": (
        "grad_window", "read", "Scheduled to obtain a Bachelor's degree by Winter 2028", None),
    "ats_bcda5eeb-196c-4f8a-a46a-519f22b3a082": (
        "grad_window", "read", "anticipated graduation date between December 2027 and June 2028", None),
    "ats_7e5feb76-9fea-420d-a0ea-32c19e256975": (
        "grad_window", "read", "eligible for full-time roles starting in 2028 or 2029", None),
    "ats_c4745709-5fa9-4f56-b6f8-9e2f7d7a0cb3": (
        "grad_window", "both", "not graduate before Fall 2027", None),
    "ats_f7007672-60be-4e79-8cc0-b6eee4d0c1f5": (
        "grad_window", "both", "the expected graduation date must not be before September of the same year", None),
    "ats_9158832e-544f-42af-af55-12573eabe341": (
        "grad_window", "read", "Currently pursuing a Bachelor's degree ... and returning to school following the "
                               "internship", None),
    "ats_e83d5205-2c76-4f34-adbc-45a543b7b26b": (
        "grad_window", "read", "Currently pursuing a Bachelor's degree ... and returning to school following the "
                               "internship", None),
    "ats_22d00642-f6f4-4b29-8f19-16686852eeb5": (
        "degree_level", "read", "Currently pursuing or recently completed a Master's, or Ph.D.", None),
    "ats_63b2b431-dd8a-4ed5-8d8d-ae9abcc4878a": (
        "must_be_local", "both", "part-time (approximately 20 hours per week ...) ... an excellent opportunity for "
                                 "local students", None),
    "ats_832d71bf-22d8-4dec-a04b-a08291150417": (
        "must_be_local", "read", "Ability to work part time (15-20 hours) during the school year", None),
    "ats_1e9bfb18-fe9d-4d19-82ab-a87461ef3ff7": (
        "must_be_local", "read", "Availability to work approximately 20 hours per week during the school year", None),
    "ats_c57ba1d3-0591-408a-a691-b05626be84f0": (
        "must_be_local", "read", "meaningful, ongoing work alongside your studies; Able to work on-site at one of our "
                                 "office locations (Foster City)", None),
    "ats_4ba241d0-99b3-4fa5-a8cc-38e1bc93ce59": (
        "degree_ceiling", "both", "Currently pursuing a Bachelor's degree", "bachelor"),
    "ats_b854fca7-7546-478f-937b-a76b8dbbddae": (
        "degree_ceiling", "read", "Currently pursuing a Bachelor's degree", "bachelor"),
    "ats_ca96c441-d218-4a8a-b6f6-a035b101c51b": (
        "degree_ceiling", "read", "Currently pursuing a Bachelor's degree", "bachelor"),
    "ats_4c33b0d7-ab72-41f6-a429-46ce9f78dc99": (
        "degree_ceiling", "read", "Currently pursuing an undergraduate or master's degree", "master"),
    "ats_b8c23e53-79db-45ab-a105-78a3d344b881": (
        "degree_ceiling", "read", "Current CS or ML undergrad or Master's student", "master"),
    "ats_11e12aaa-6e95-477a-a2ef-b0995c36edee": (
        "degree_ceiling", "read", "Currently pursuing a Bachelor's or Master's degree", "master"),
    "ats_19e66723-f0ca-497d-881a-e6d544f3ab31": (
        "degree_ceiling", "read", "Actively pursuing a Bachelor's or Master's degree", "master"),
    "ats_1b670e5e-9edc-4255-93f8-5fdac2fb70ac": (
        "degree_ceiling", "read", "Currently enrolled in a B.S. or M.S. program", "master"),
    "ats_366e5594-f4cf-486f-b2e7-0c33b41e8b18": (
        "degree_ceiling", "read", "Currently pursuing a Bachelor's or Master's degree", "master"),
    "ats_8ba55980-bb00-4f5d-8d2a-4b40243aa2ac": (
        "degree_ceiling", "read", "Currently pursuing a Bachelor's or Master's degree", "master"),
    "ats_46c77ff4-d626-47a9-bdb5-d46938c07a50": (
        "degree_ceiling", "read", "Currently enrolled in a Bachelor's or Master's degree", "master"),
    "ats_47fb79bc-9f49-407b-bb0c-52352eba40be": (
        "degree_ceiling", "read", "Currently pursuing a Bachelor's or Master's degree", "master"),
    "ats_caf607e7-c4db-41eb-b84a-606fa07f4ae6": (
        "degree_ceiling", "read", "pursuing a Bachelor's, Master's, or MBA", "master"),
    "ats_eed65ade-e9b6-4281-bfea-94e030327a6d": (
        "degree_ceiling", "read", "Enrolled in university for bachelor or masters program", "master"),
    "ats_02bb784d-be4f-49f0-a8b5-b63b8705968d": ("exp_cap", "read", "0-2 years of relevant experience", 2),
    "ats_f79a534a-9fa4-4f5a-970a-603c7c67ce1e": ("exp_cap", "read", "0-2 years of relevant experience", 2),
    "ats_6e326004-e660-4a06-a3f7-9378a0812a4e": (
        "exp_cap", "read", "Pursuing a PhD degree ... with 0-1 years of hands-on experience", 1),
}
# Read and nothing found the card cannot hold (a GPA, program dates, a weekly schedule or a "3+ months" floor are
# not among the kinds above: no profile shows a GPA, and the test is built the same way).
READ_CLEAR = frozenset({
    "ats_02d9e13e-0a78-4ac3-ac88-bcc860a02ee5", "ats_14baa8ae-f151-483f-b5e0-77ca5086d04e",
    "ats_1736ee29-1f1e-4421-8616-9c616abc3f8e", "ats_25634961-9424-41ec-9553-925a6e400699",
    "ats_2d77d594-bd99-46dd-bb12-5e46c01f41e8", "ats_40fdd912-45b2-44ba-8a04-89566decd578",
    "ats_4770a4e3-28be-40d7-b5a2-700c0307960d", "ats_501aa792-9adc-45ac-b213-6e5af6627213",
    "ats_5725405b-6f22-485b-a418-ad3c6148afb4", "ats_6cc400a9-db6f-4550-9989-36f65cbe70b1",
    "ats_96ffbc65-720f-4b45-965f-6ddc2b4a4207", "ats_9c3c759c-4a1c-4b6f-8f1c-51e5c4fdb90d",
    "ats_a13245a8-c3d3-4790-9835-386cfb36742e", "ats_a4415818-9dfc-4624-8235-e94c953af5e4",
    "ats_ac297577-f4f9-4b73-ab38-00c094342140", "ats_ac51c04c-ef1b-4705-8b2c-8001da135c28",
    "ats_c3f5e2ad-b12d-4ea0-bcce-3fc9a30ae812", "ats_d7612fa7-5009-43fc-a9d3-8f16c3478eb2",
    "ats_f455fb7a-6e82-4af2-b378-c5deb178c84d", "ats_fbbaa89d-fca9-49aa-9d4a-14d71883ea4a",
})
DEV_V2_OUT = RT / "checked_real_dev_v2.jsonl"
DEV_V2_BUILDER_OUT = RT / "checked_real_dev_v2_builder.jsonl"
DEV_V2_DIR = OUT / "dev_v2"


def past_limit(kind: str, limit, c: CandidateCard) -> bool:
    """For a CHECK_KINDS restriction: is this candidate past it (studying above the ceiling, more years than the
    cap)? A graduate on a "currently pursuing" posting already fails the card's enrollment check."""
    if kind == "degree_ceiling":
        return is_enrolled(c) and DEGREE_LEVELS.index(c.degree_level) > DEGREE_LEVELS.index(limit)
    if kind == "exp_cap":
        return c.years > limit
    raise ValueError(kind)


def dev_v2_decisions(files: dict[str, list[dict]], cands: dict[str, dict[str, dict]],
                     restrictions: dict = RESTRICTIONS, clear: frozenset = READ_CLEAR) -> dict[str, dict]:
    """posting id -> {kind, source, quote, limit, remove, past_limit_rows} for every posting in `files`. A posting
    neither restricted nor read-clear stops the build: it was never read."""
    ids = sorted({r["rq_id"] for rows in files.values() for r in rows})
    unread = [i for i in ids if i not in restrictions and i not in clear]
    if unread:
        raise SystemExit(f"{len(unread)} dev postings were never read for restrictions: {unread[:5]}")
    out = {}
    for i in ids:
        if i in clear:
            out[i] = {"kind": None, "source": "read", "quote": "", "limit": None, "remove": False,
                      "past_limit_rows": 0}
            continue
        kind, source, quote, limit = restrictions[i]
        past = 0
        if kind in CHECK_KINDS:
            past = sum(past_limit(kind, limit, CandidateCard.from_dict(cands[f][r["pair_id"]]))
                       for f, rows in files.items() for r in rows if r["rq_id"] == i and r["bucket"] != SKIP)
        elif kind not in REMOVE_KINDS:
            raise ValueError(f"{i}: unknown restriction kind {kind}")
        out[i] = {"kind": kind, "source": source, "quote": quote, "limit": limit,
                  "remove": kind in REMOVE_KINDS or past > 0, "past_limit_rows": past}
    return out


def _counts(rows: list[dict]) -> dict:
    return {"rows": len(rows), "postings": len({r["rq_id"] for r in rows}),
            "labels": {w: sum(WORD[r["bucket"]] == w for r in rows) for w in ("Keep", "Explore", "Skip")},
            "decidable": sum(r["decidable"] for r in rows),
            "decidable_labels": {w: sum(WORD[r["bucket"]] == w and r["decidable"] for r in rows)
                                 for w in ("Keep", "Explore", "Skip")}}


def dev_v2(a: argparse.Namespace) -> dict:
    """checked_real_dev_v2{_builder}.jsonl: the v1 files without the postings dev_v2_decisions removes. Rows are
    copied byte for byte (no relabel: the cards are the ones v1 was built on), so every prompt is one in v1."""
    v1 = {"builder": DEV_BUILDER_OUT, "redrawn": DEV_OUT}
    v2 = {"builder": DEV_V2_BUILDER_OUT, "redrawn": DEV_V2_OUT}
    lines = {k: [ln for ln in p.read_text().splitlines() if ln.strip()] for k, p in v1.items()}
    files = {k: [json.loads(ln) for ln in v] for k, v in lines.items()}
    step = {"builder": OUT / "rows_step1.jsonl", "redrawn": OUT / "rows.jsonl"}
    cands = {k: {r["pair_id"]: r["candidate"] for r in read_jsonl(p) if r["split"] == "dev" and r["stage"] == "kept"}
             for k, p in step.items()}
    for k, rows in files.items():
        miss = [r["pair_id"] for r in rows if r["pair_id"] not in cands[k]]
        if miss:
            raise SystemExit(f"{k}: {len(miss)} v1 rows have no candidate in {step[k].name}: {miss[:3]}")
    dec = dev_v2_decisions(files, cands)
    gone = {i for i, d in dec.items() if d["remove"]}
    kept = {k: [(ln, r) for ln, r in zip(lines[k], files[k]) if r["rq_id"] not in gone] for k in files}
    v1_prompts = {r["prompt"] for rows in files.values() for r in rows}
    same = {k: [r["pair_id"] for _, r in v if r["prompt"] in v1_prompts] for k, v in kept.items()}
    if any(len(same[k]) != len(kept[k]) for k in kept):
        raise SystemExit("a v2 row's prompt is not in v1")

    raw = art.brt_raw()
    removed = []
    for i in sorted(dec, key=lambda i: (not dec[i]["remove"], dec[i]["kind"] or "~", i)):
        d = dec[i]
        if d["kind"] is None:
            continue
        removed.append({"posting_id": i, "company": raw[i].get("company", ""), "title": raw[i].get("title", ""),
                        **{k: d[k] for k in ("kind", "source", "quote", "limit", "remove", "past_limit_rows")},
                        "rows": {k: dict(Counter(WORD[r["bucket"]] for r in rows if r["rq_id"] == i))
                                 for k, rows in files.items()}})
    rep = {"v1": {k: _counts(v) | {"md5": art.md5(v1[k])} for k, v in files.items()},
           "postings_read": len(dec), "postings_removed": len(gone),
           "removed_by_kind": dict(Counter(dec[i]["kind"] for i in gone)),
           "checked_not_removed": {k: sorted(i for i, d in dec.items() if d["kind"] == k and not d["remove"])
                                   for k in CHECK_KINDS},
           "rows_removed": {k: _counts([r for r in rows if r["rq_id"] in gone]) for k, rows in files.items()},
           "prompt_identical_to_v1": {k: f"{len(same[k])} of {len(kept[k])}" for k in kept}}
    if a.dry_run:
        print(json.dumps(rep, indent=1, default=str))
        return rep
    DEV_V2_DIR.mkdir(parents=True, exist_ok=True)
    for k, v in kept.items():
        v2[k].write_text("".join(ln + "\n" for ln, _ in v))
        rep[f"v2_{k}"] = _counts([r for _, r in v]) | {"file": str(v2[k].relative_to(ROOT)), "md5": art.md5(v2[k])}
    brt.write_jsonl(DEV_V2_DIR / "restrictions.jsonl", removed)
    (DEV_V2_DIR / "prompt_identical_to_v1.json").write_text(json.dumps(
        {"note": "pair ids of v2 rows whose prompt is byte-identical to a v1 row's (v1: checked_real_dev.jsonl or "
                 "checked_real_dev_builder.jsonl); a stored v1 prediction for that pair id can be reused",
         **same}, indent=0))
    (DEV_V2_DIR / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    print(f"wrote {DEV_V2_BUILDER_OUT.relative_to(ROOT)}, {DEV_V2_OUT.relative_to(ROOT)}, "
          f"{DEV_V2_DIR.relative_to(ROOT)}/{{restrictions.jsonl, prompt_identical_to_v1.json, report.json}}")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--nice", choices=("extracted", "checked"), default="extracted")
    b.add_argument("--draws", type=int, default=1)
    b.add_argument("--no-checks", action="store_true")
    d = sub.add_parser("dev-v2")
    d.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    {"build": build, "dev-v2": dev_v2}[a.cmd](a)


if __name__ == "__main__":
    main()
