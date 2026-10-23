"""Spend ~20 trial credits on a 200-row invented candidate sample to learn Adaption's output shape."""
import json, os, sys
from pathlib import Path
from adaption import Adaption

os.environ.setdefault("ADAPTION_API_KEY", Path("~/.config/adaption/api_key").expanduser().read_text().strip())
c = Adaption()
PROMPT = """Synthetic job-seeker profiles for a job-matching model. Cover every kind of job seeker, not just tech:
nursing, pharmacy, biology labs, accounting, marketing, teaching, social work, design, law enforcement and forensics,
skilled trades, hospitality, data science, software, finance, communications. Vary seniority from current students
to 15-year veterans, career changers and people returning after a break.

Each PROMPT is one JSON object with exactly these keys and allowed values:
field (string), degree_level ("none"|"associate"|"bachelor"|"master"|"doctoral"), degree_field (string),
graduation_year (int or null), years (int), skills (list of 4-10 strings), licenses (list, often empty),
current_location ("City, ST" or "City, Country"), willing_locations (list), remote_preference ("any"|"hybrid_or_remote"|"remote_only"),
work_auth ("citizen_or_pr"|"f1_cpt"|"f1_opt"|"stem_opt"|"needs_h1b_now"|"other_visa"),
salary_floor (int USD per year or null), employment_types_wanted (subset of "internship","full_time","part_time","contract"),
industries_wanted (list), industries_avoided (list), roles_avoided (list of words), education_country (string), language (ISO 639-1 code).
About a third of profiles are international students or graduates on F-1 CPT, F-1 OPT or STEM OPT. No names, emails, phone numbers or addresses.

Each COMPLETION is that same person's one-page resume as plain text, using only the facts in the JSON (no invented employers' secrets,
no name: use "Candidate" as the header). Keep it realistic and varied in style."""
mode = sys.argv[1] if len(sys.argv) > 1 else "estimate"
r = c.datasets.invent(name="candidate-sample-200", rows=200, dataset_prompt=PROMPT,
                      training_type="instruction_dataset", estimate=(mode == "estimate"))
print(r)
if mode == "run":
    Path("data/adaption/sample_invent.json").write_text(json.dumps({"id": r.id, "name": r.name}))
