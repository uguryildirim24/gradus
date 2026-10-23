"""Resume-writing test: upload 100 candidate cards, have Adaption adapt write a resume for each (universal-prompt mode)."""
import csv, json, os, re, sys
from pathlib import Path
sys.path.insert(0, "src")
from adaption import Adaption
from matcher.schema import CandidateCard
os.environ.setdefault("ADAPTION_API_KEY", Path("~/.config/adaption/api_key").expanduser().read_text().strip())
c = Adaption()
csv.field_size_limit(10**8)
src = "data/adaption/d20a7c9a-900e-4441-8cf0-62a575d7f713/download.bin"
cards = []
for r in csv.DictReader(open(src, newline="")):
    m = re.search(r"\{.*\}", r["enhanced_prompt"], re.S)
    try:
        d = json.loads(m.group(0) if m else r["enhanced_prompt"])
        card = CandidateCard(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})
    except Exception:
        continue
    cards.append(card.to_dict())
cards = cards[:100]
out = Path("data/built/render_test_cards.csv"); out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["row_id", "card"])
    for i, d in enumerate(cards): w.writerow([f"c{i:04d}", json.dumps(d, ensure_ascii=False)])
print("cards", len(cards))

UNIVERSAL = ("Write a realistic one-page resume as plain text for the job seeker described by the JSON card in the context. "
             "Use only the facts in the card: field, degree, graduation year, years of experience, skills, licenses, current location. "
             "Invent plausible but generic employer names and bullet points consistent with those facts.")
BLUEPRINT = ("Header is the single word 'Candidate'. Never include a name, email, phone number, street address, URL, photo, age, gender, "
             "nationality, or religion. Do not mention work authorization, visa status, salary expectations, preferred locations, remote "
             "preference, industries wanted or avoided, or roles avoided: those are private preferences, not resume content. "
             "Mention the country of education only as the school's location. Write in English. Vary layout and tone across resumes.")
mode = sys.argv[1] if len(sys.argv) > 1 else "estimate"
state = Path("data/adaption/render_test.json")
if state.exists():
    did = json.loads(state.read_text())["dataset_id"]
else:
    up = c.datasets.upload_file(str(out), name="render-test-cards")
    print("UPLOAD", up)
    did = getattr(up, "dataset_id", None) or getattr(up, "id")
    state.write_text(json.dumps({"dataset_id": did}))
    c.datasets.wait_for_completion(did) if False else None
kw = dict(column_mapping={"context": ["card"], "universal_prompt": UNIVERSAL},
          recipe_specification={"recipes": {"prompt_rephrase": False, "deduplication": False, "reasoning_traces": False}},
          brand_controls={"blueprint": BLUEPRINT, "length": "detailed", "hallucination_mitigation": False},
          training_type="instruction_dataset")
r = c.datasets.adapt(did, estimate=(mode == "estimate"), **kw)
print(mode.upper(), r)
