"""Check an invented candidate sample against CandidateCard."""
import csv, json, sys, re, collections
sys.path.insert(0, "src")
from matcher.schema import CandidateCard
csv.field_size_limit(10**8)
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
print("rows", len(rows), "columns", list(rows[0].keys()))
pcol = next(k for k in rows[0] if "prompt" in k.lower())
ccol = next(k for k in rows[0] if "completion" in k.lower())
ok, bad, cards = 0, collections.Counter(), []
for r in rows:
    t = r[pcol].strip()
    m = re.search(r"\{.*\}", t, re.S)
    try:
        d = json.loads(m.group(0) if m else t)
        d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
        cards.append(CandidateCard(**d)); ok += 1
    except Exception as e:
        bad[type(e).__name__ + ": " + str(e)[:80]] += 1
print("valid cards", ok, "invalid", sum(bad.values())); print(bad.most_common(8))
C = collections.Counter
print("work_auth", C(c.work_auth for c in cards).most_common())
print("degree", C(c.degree_level for c in cards).most_common())
print("field top", C(c.field for c in cards).most_common(25))
print("lang", C(c.language for c in cards).most_common(12))
print("edu country", C(c.education_country for c in cards).most_common(12))
print("years", sorted(C(c.years for c in cards).items()))
lens = sorted(len(r[ccol]) for r in rows); print("resume chars median", lens[len(lens)//2], "min", lens[0], "max", lens[-1])
names = sum(bool(re.search(r"@|\(\d{3}\)|\d{3}-\d{3}-\d{4}", r[ccol])) for r in rows); print("rows with email/phone", names)
print("\n--- SAMPLE PROMPT ---\n", rows[0][pcol][:1200], "\n--- SAMPLE COMPLETION ---\n", rows[0][ccol][:1500])
