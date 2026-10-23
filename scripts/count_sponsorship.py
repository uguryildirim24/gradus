"""Count how many real US internships refuse sponsorship or are citizens-only (the README's problem numbers).

Usage: count_sponsorship.py [CARDS.jsonl]   (default data/built/cards/extracted_real_cards_v2.jsonl)

Steps, each printed: every card in the file; US and English postings not mixed with other countries; minus
repeat postings (meta.dup_of); internships among those; then the work-authorization split.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "data/built/cards/extracted_real_cards_v2.jsonl"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else CARDS
    rows = [json.loads(l) for l in path.open() if l.strip()]
    us = [r for r in rows if r["meta"].get("us") and r["meta"].get("language") == "en" and not r["meta"].get("mixed")]
    uniq = [r for r in us if "dup_of" not in r["meta"]]
    intern = [r["card"] for r in uniq if r["card"]["employment_type"] == "internship"]
    no_sponsor = [c for c in intern if c["sponsorship"] in ("will_not_sponsor", "will_not_sponsor_or_opt")]
    no_opt = [c for c in intern if c["sponsorship"] == "will_not_sponsor_or_opt"]
    citizens = [c for c in intern if c["citizens_only"]]
    either = [c for c in intern if c["citizens_only"] or c["sponsorship"] in ("will_not_sponsor", "will_not_sponsor_or_opt")]
    sponsors = [c for c in intern if c["sponsorship"] == "sponsors"]
    n = len(intern)
    print(f"cards in file:                      {len(rows):,}")
    print(f"US, English, not mixed:             {len(us):,}")
    print(f"minus repeat postings (dup_of):     {len(uniq):,}")
    print(f"internships:                        {n:,}")
    print(f"will not sponsor:                   {len(no_sponsor):,} (of which also no OPT: {len(no_opt):,})")
    print(f"citizens or permanent residents:    {len(citizens):,}")
    print(f"either (refuse or citizens-only):   {len(either):,} = {100 * len(either) / n:.1f}%")
    print(f"say they sponsor:                   {len(sponsors):,} = {100 * len(sponsors) / n:.1f}%")


if __name__ == "__main__":
    main()
