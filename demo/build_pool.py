"""Build demo/local/pool.jsonl: ~150 real US internship postings joined with their extracted cards.

    uv run python demo/build_pool.py [--n 150] [--seed 0]

Reads data/built/cards/extracted_real_cards_v2.jsonl (the v2 extraction, `extract_postings.py merge`) and
data/raw/ats/postings.jsonl. Keeps rows whose meta says US and English, one row per posting text (no
meta.dup_of or meta.same_description_as copy), and of those the US-located internships whose term is
Spring/Summer/Fall 2027 or N/A (no past seasons) and not reserved for a conference's attendees, at most 2
per company, with about one in ten citizens-only so the work-authorization switch has something to move.
The posting text is textclean.real_posting_text (the Title/Company/Locations header and markdown-free body
the real test and training carry). It is employer copyright: demo/local/ is git-ignored, so the pool is
rebuilt locally and never published.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from matcher import textclean  # noqa: E402

STATES = ("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK "
          "OR PA RI SC SD TN TX UT VT VA WA WV WI WY").split()
US_PLACE = re.compile(r",\s*(?:" + "|".join(STATES) + r")\b(?!,\s*[A-Za-z]{3,})")


# A posting that turns away F-1 students or CPT, or that asks for permanent work authorization, ITAR /
# export-control "U.S. person" status or a clearance, is a citizens-only posting under the label rules
# (rules.py: citizens_only covers "permanent" work authorization, ITAR and clearance, and "a posting that
# does refuse CPT is in practice ... citizens_only"). When the extracted card missed that (citizens_only
# false), the H'' rules let it through and the featured F-1 CPT persona can get Keep on a posting
# that plainly says no, so the pool leaves it out. Dropping a posting by mistake only costs a pool slot.
# The whole text is searched (run B reads the whole posting, so the model can see a refusal anywhere in it).
_NEG = r"(?:\bnot\b|\bno\b|n't\b|\bunable\b|\bcannot\b|\bineligible\b|\bonly\b)"
_F1 = r"(?:\bF-?1\b|\bCPT\b|student\s+visas?|temporary\s+visas?)"
F1_REFUSAL = re.compile(rf"{_F1}[^\n]{{0,200}}?{_NEG}|{_NEG}[^\n]{{0,400}}?{_F1}", re.I)  # F-1 / CPT near a negation
CITIZENS_ONLY_TEXT = re.compile(
    r"\bpermanent(?:ly)?\s+(?:U\.?S\.?\s+)?(?:work|employment)\s+authori|\bpermanently\s+authori[sz]ed"
    r"|\bwork\s+permanently\b|\bunlimited\s+(?:amount|period)\s+of\s+time\b|\bindefinite(?:ly|\s+period)\b"
    r"|\b(?:unlimited|unrestricted)\s+(?:work|employment|authori)|\bwithout\s+restrictions?\b"  # permanent authorization
    r"|\bfor\s+anyone\s+other\s+than\b|\bon\s+a\s+permanent\s+basis\b"
    r"|\bcitizen(?:ship)?\s*(?:,|or|/)\s*(?:a\s+)?(?:lawful\s+)?(?:U\.?S\.?\s+)?permanent\s+residen"
    r"|(?:citizen|national)[^\n]{0,80}?permanent\s+resident[^\n]{0,250}?(?:refugee|asyl)"   # the ITAR "U.S. person" list
    r"|\bITAR\b|International\s+Traffic\s+in\s+Arms|\bU\.\s?S\.?\s+persons?\b"         # ITAR / U.S. person
    r"|export[- ]control\w*[^\n]{0,600}?(?:citizen|permanent\s+resident|green\s+card)"  # export control, citizens named
    r"|(?:security|secret|public\s+trust)\s+clearance|\bclearance\s+required|\bpublic\s+trust\b|\bTS/SCI\b"
    r"|\btop\s+secret\b", re.I)


# "F-1 students are eligible if they are using CPT" (DoorDash) or "CPT & Pre-OPT candidates invited to
# apply" (Compeer) welcomes CPT: that overrides a nearby F-1 negation, never ITAR, clearance or permanence.
WELCOMES = re.compile(r"\b(?:F-?1|CPT)\b[^.\n]{0,60}?\b(?:(?:are|is)\s+eligible|invited\s+to\s+apply|welcome)", re.I)


# A posting only a conference's attendees may apply through ("reserved for registered 2026 AfroTech attendees",
# "must be attending ... the Society of Women Engineers Conference") is closed to the persona whatever the
# rules say, so a Keep on it would be a visible demo error. Plain "reserved for" is too broad (HPE's
# accommodation form, award blurbs), so the attendee wording is required. The cut comes after the shuffle, so
# the rest of the pool keeps its draw.
EVENT_ONLY = re.compile(r"\breserved\s+for\s+[^.\n]{0,60}?\b(?:attendees|participants)\b|\bmust\s+be\s+attending\b", re.I)


def event_only(post: dict) -> bool:
    return bool(EVENT_ONLY.search(post.get("description", "")))


def refuses_f1(card: dict, post: dict) -> bool:
    if card.get("citizens_only"):
        return False
    text = post.get("description", "")
    return bool(CITIZENS_ONLY_TEXT.search(text)) or bool(F1_REFUSAL.search(text)) and not WELCOMES.search(text)


def eligible(card: dict, post: dict) -> bool:
    terms = post.get("terms") or ["N/A"]
    ok_term = any(t == "N/A" or (re.search(r"2027", t) and not t.startswith("Winter")) for t in terms)
    locs = card.get("locations") or []
    return (card["employment_type"] == "internship" and ok_term and len(post.get("description", "")) > 800
            and (card["work_mode"] == "remote" or (locs and all(US_PLACE.search(x) for x in locs))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "demo" / "local" / "pool.jsonl"))
    a = ap.parse_args()
    need = [ROOT / "data/raw/ats/postings.jsonl", ROOT / "data/built/cards/extracted_real_cards_v2.jsonl"]
    missing = [str(p.relative_to(ROOT)) for p in need if not p.exists()]
    if missing:
        sys.exit("The demo pool is built from real postings and their extracted cards, which are not in the public repo\n"
                 "(employer text): missing " + ", ".join(missing) + ".\n"
                 "The live demo linked at the top of README.md runs the same page on this pool.")
    posts = {}
    for line in open(need[0]):
        p = json.loads(line)
        posts[p["simplify_id"]] = p
    rows, refused = [], []
    for line in open(need[1]):
        r = json.loads(line)
        meta = r.get("meta") or {}
        if not meta.get("us") or meta.get("language") != "en" or meta.get("dup_of") or meta.get("same_description_as"):
            continue
        post = posts.get(r["id"].removeprefix("ats_"))
        if post and eligible(r["card"], post):
            if refuses_f1(r["card"], post):
                refused.append(post["simplify_id"])
            else:
                rows.append((r, post))
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    events = [post["simplify_id"] for _, post in rows if event_only(post)]
    rows = [x for x in rows if not event_only(x[1])]
    chosen, per_company = [], defaultdict(int)
    cit = [x for x in rows if x[0]["card"]["citizens_only"]]
    rest = [x for x in rows if not x[0]["card"]["citizens_only"]]
    for group, want in ((cit, a.n // 10), (rest, a.n)):
        for r, post in group:
            if len(chosen) >= a.n or sum(1 for c in chosen if c[2] is group) >= want:
                break
            co = post["company"].strip().lower()
            if per_company[co] >= 2:
                continue
            per_company[co] += 1
            chosen.append((r, post, group))
    rng.shuffle(chosen)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r, post, _ in chosen:
            f.write(json.dumps({"id": post["simplify_id"], "company": post["company"].strip(), "title": post["title"],
                                "locations": post["locations"], "url": post["url"], "terms": post.get("terms"),
                                "text": textclean.real_posting_text(post), "card": r["card"]}, ensure_ascii=False) + "\n")
    print(f"left out {len(refused)} eligible postings whose text is citizens-only in effect (F-1/CPT refused, "
          "permanent authorization, ITAR, clearance) but whose card is not citizens_only")
    print(f"left out {len(events)} postings reserved for a conference's attendees")
    print(f"{len(chosen)} postings ({sum(1 for c in chosen if c[0]['card']['citizens_only'])} citizens-only) -> {out}")


if __name__ == "__main__":
    main()
