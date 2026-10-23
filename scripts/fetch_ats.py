"""Usage: uv run python scripts/fetch_ats.py [--limit N] [--report]"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from matcher.ats.fetch import report, run

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int)
ap.add_argument("--report", action="store_true")
a = ap.parse_args()
out = ROOT / "data/raw/ats"
if not a.report:
    asyncio.run(run(ROOT / "data/raw/simplify/listings.json", out, a.limit))
report(out)
