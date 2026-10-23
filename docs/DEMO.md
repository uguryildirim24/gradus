# Demo

## For judges

Live demo: https://rolfyildirim--gradus.modal.run

1. Open the link. The first request after a quiet spell wakes the GPU (about 110 s); the page says so while it
   waits.
2. Pick the saved persona "Alex" (synthetic: a computer science junior on F-1 CPT in Boston, will move anywhere in
   the US), or paste or upload your own resume (.pdf, .docx, .txt) and fill in the profile.
3. Press "Score postings". The pool is about 150 real US tech internship postings. Each lands in For you (Keep),
   Explore or Excluded (Skip) with P(Keep) and a one-line reason.
4. Look for the marker "decided by the rule on Adaption's extracted card": those postings were set to Skip by the
   H'' rules before the model ran (work authorization, enrollment, degree level and field, location). Every other
   card is the model's call.
5. Switch work authorization to "US citizen or permanent resident" and sort again: the citizens-only postings lose
   the rule marker and the model sorts them on fit.

The page runs the shipped system: run B (Qwen3.5-4B LoRA trained with Adaption AutoScientist, run
`8dc3d64a-cd0f-4807-a6e9-4eadb542ed3d`) merged into the public weights, in bfloat16 on one Modal L4, read with the
decider, behind H'' without work mode. "Remove name, email, phone and links" is on by default; resumes and profiles
are never written to disk or logs. A monthly spend cap protects the GPU budget: once it is used, the page says the
model is paused and links the video and the repo.

The evidence that the page gives the evaluated labels is `tests/test_demo_b_fresh.py`. It runs every row of fresh tests
1-4 through the page's scoring path with B's stored float32 scores in place of the model (and a label word the page
must ignore). The page's final label equals the committed B+H''_without_work_mode|decider label on every row, its s
equals the stored s exactly, and the prompt it sends (or would send, for a row H'' decides before the model) equals
B's eval prompt byte for byte. The posting text is taken from B's prompt, so this checks rubric, resume, profile and
split, not the live pool's own posting text.

| set | rows through the page | agree | decided before the model | prompts equal B's |
|---|---|---|---|---|
| fresh test 1 | 592 | 592 | 145 | 592/592 |
| fresh test 2 | 371 | 371 | 86 | 371/371 |
| fresh test 3 | 437 | 437 | 99 | 437/437 |
| fresh test 4 | 1,135 | 1,135 | 315 | 1,135/1,135 |

Fresh test 4's other 4 rows have no extracted card; the demo's pool never holds such a posting
(`demo/build_pool.py` takes postings with a card), and for those rows the committed label is the decider's alone, as
the pinned no-card policy says. The closest any row's s comes to a cut-off is 0.0002 (fresh test 3). The test needs
the private `data/` and skips without it.

The page's labels stand on sealed fresh test 4, scored once on 1,139 rows (`eval/fresh_test4_result.md`): macro-F1
0.679 [0.599, 0.747], 28 of 43 gold Keeps found, Keep precision 0.549 (28/51), 23 wrong Keeps. On the 513-row
eligibility-pass slice it passes amendment 2's bar against both the full rules (+0.192 macro-F1, lower bound 0.079)
and the untrained base behind the same H'' and decider (+0.217, lower bound 0.124).

## How the page works

```
browser ──> demo/server.py, or its Modal wrapper demo/modal_web.py   (standard library only, no model code)
              │  POST /score  (JSON, shared prompt prefix + one suffix per posting)
              ▼
            scorer, same JSON API either way:
              the Modal app's GPU scorer (demo/modal_app.py, one L4, bfloat16)
              scripts/demo_backend_torch.py   (torch on CPU, float32; minutes per request)
```

The server never imports a model library, directly or through another module (a test checks the import graph). The
prompt is run B's (`prompt_b` in `src/matcher/prompt_b.py`: the labelers' rubric on top, then `build_prompt`'s
layout from `src/matcher/pairs.py` without h's instruction, with the whole posting), and the profile text is
`render_profile` from `src/matcher/profile.py`, so neither can drift from training.

- The prompt is B's. `matcher.prompt_b` puts the rubric the labelers applied (`SYSTEM_B`, the same text
  `scripts/build_fit_train3.py` reads from `eval/protocol_amendment_5.json`, sha256 `1d12b0d0...cb2e0d` as
  `eval/cycle3_eval_inputs.json` pins it) at the top of the single user turn, then resume, profile and the whole
  posting, no cut. `demo_lib.split_prompt` caches rubric, resume and profile as the shared prefix.
- The readout is the decider. `matcher.decider` (standard library, so the server and the scorers import it):
  s = P(Keep) + 0.5 P(Explore), softmax at temperature 1 over the three float32 first-token label scores; s >=
  0.5890379172624224 Keep, s < 0.2731045004103801 Skip, else Explore (the exact cut-offs of
  `eval/selection_cycle3.json`). The page reads the label from the scores, live or precomputed; the scorers use
  the same decider to pick the word they force as the reason's first token.
- H'' runs on the page. `Demo.prepare` applies `matcher.hybrid`'s `rule_verdict` and `hybrid` (the functions
  `scripts/hybrid_rules.py` and `scripts/fit_select_cycle3.py` score with) on each pool row's Adaption-extracted
  card, marked thin or not from the card and the posting body first (`rules.mark_thin`, as the eval does), with
  `HPP_CATS`: degree field, degree level, enrollment, location, work authorization, plus the salary cap. Employment
  type and work mode are left to the model. A posting H'' sets to Skip whatever the model says is decided there and
  never sent to the GPU; the salary cap runs after the decider.

### What the page does

- Resume: paste, or upload .pdf / .docx / .txt (any file name, non-ASCII letters included; a corrupt
  or encrypted file gives a plain error, and a Word text box is read once). "Remove name, email, phone
  and links" is on by default: a simple local function in `demo/demo_lib.py` (Unicode names, name
  particles such as "de la", the name anywhere in the first three non-heading lines, also when it shares
  the line with contact details or the city: "Name | email | phone | City", "Name — City",
  "Name, City, ST", "Surname, Given", "NAME (he/him)"; 10-digit phones, extensions). It runs before
  `textclean.redact`, which every prompt goes through (the contact block becomes "Candidate") but which alone misses
  a name that shares a line with contact details. "What the model sees as your resume" shows the exact text after
  redaction.
- Profile: the fields of `render_profile`, plus highest degree, field, graduation year and years of
  experience. `render_profile` writes those four into the prompt, so they reach the model
  ("Highest degree: Bachelor's degree in Computer Science, expected 2028", "Years of work experience: 1").
  "Willing to relocate" has an explicit choice: anywhere in the US, these places, nowhere
  else. An empty list would render "nowhere else" and silently exclude every other city.
  "What the model sees as your profile" shows the exact text.
- H'' rules (the shipped eligibility layer, `eval/selection_cycle3.json`): before the model, the rules on each
  posting's Adaption-extracted card set it to Skip when H'' would whatever the model says: work authorization
  (citizens-only, no sponsorship for someone who needs it now, no OPT), enrollment, degree level, degree field and
  location. Never job type, work mode, skills or years: those stay the model's call. Those postings show in
  Excluded with the rule's reason and the marker "decided by the rule on Adaption's extracted card", and never
  reach the model. After the decider, the salary cap turns a Keep into Explore when the pay tops out under the
  candidate's floor; that card carries the marker too and says what the model said. An empty form field never
  fires an override (see "What the form cannot say").
- Scoring: *live* postings are scored now (20 by default; the live link allows up to 25 a request), and a
  *precomputed* results file, if its prompt prefix matches this exact resume and profile, fills in the rest.
  Citizens-only postings that survive the rules (the candidate is a citizen or permanent resident) are scored on top
  of the live limit, so flipping F-1 to citizen sends the 15 citizens-only postings from the H'' rules to the model
  and keeps the same others. The model decides their bucket: those that fit move to For you or Explore, the rest
  (several are EE, RF or marine roles) stay in Excluded, now with a model score. Anything else is listed under "Not
  scored this time", so no posting is dropped silently.
- Pool: `build_pool.py` leaves out postings whose text is citizens-only in effect (F-1 or CPT
  refused, permanent or unrestricted work authorization, ITAR / U.S. person, clearance) while the
  extracted card says `citizens_only: false`; the H'' rules would let those through and the F-1 CPT
  persona could get Keep on a posting that plainly says no. It also leaves out postings reserved for a
  conference's attendees ("reserved for registered 2026 AfroTech attendees", "must be attending"). On the
  2026-10-19 build (v2 extraction cards) that left out 101 eligible postings as citizens-only in effect and 1
  as event-only; the 15 citizens-only pool cards each have an explicit line (ITAR, clearance, citizenship).
  Rebuild and reread the work-authorization sentences whenever the extraction cards change.

### What the form cannot say

H'' reads the candidate through the profile form, which has no skills or licenses (the model reads those in the
resume). Neither is an H'' category, so the page leaves the card's skill and license asks out of the rule verdict;
left in, nearly every verdict would be a skills Skip, which would switch off the salary cap (it needs the rules'
verdict to be Explore). An empty form field is read as unknown and never fires an override, the way a reason
outside the H'' categories never does in `hybrid_rules`: an empty graduation year drops enrollment (unless the
degree level is "none"), an empty field of study drops degree field, an empty "Lives in" drops location (unless
relocation is anywhere in the US), and a missing degree level (the page always sends one) drops all three degree
categories. Work authorization and salary floor need nothing: their empty values never fire. No fresh-test
candidate needed any of this: on all four fresh tests the page, with skills and licenses left out, gives the
committed label on every row.

### Scorer API

```
GET  /health
POST /score {"prefix": str, "items": [{"id", "suffix"}], "reasons": bool, "max_reason_tokens": 40}
  -> {"backend", "model", "results": [{"id", "scores", "p", "label", "reason", "prompt_tokens",
       "cached_tokens", "seconds"}], "stats": {...}}
```

`prefix + suffix` is the exact `prompt_b` text. The scores are read the way `local_eval.py` reads them, with
the same code (`evalstats.LabelHead`): the final hidden state in float32 times the float32 head rows of Keep,
Explore and Skip. A bfloat16 model's own logits move in steps of 0.125 near 16-32, so reading them would be
coarser. `p` = softmax over the three scores at temperature 1; `label` = the decider on the unrounded scores
(`demo_common.decider_label`); the reason is generated with that label forced as the first token, so label and
reason agree. The page reads the label from `scores` itself and applies H'' after it. Framing is the Qwen3.5 chat
template with thinking off, the same as `local_eval.py`.

## Run the page locally

Needs the private `data/` (the real postings and their extracted cards) and a scorer serving run B merged into
Qwen3.5-4B. `scripts/demo_backend_torch.py --model MERGED_DIR` serves it with torch on CPU: Qwen3.5's
linear-attention layers use slow reference kernels there, so a request takes minutes. That suits a precompute and is too slow
for a live take.

```sh
uv run python demo/build_pool.py                                             # ~150 real US internships, never committed
uv run --group demo python demo/server.py --scorer http://127.0.0.1:18765     # http://127.0.0.1:8077
```

`--group demo` adds `pypdf` (pure Python) for PDF upload; DOCX is read from the zip's XML, TXT/MD as is.

To precompute a persona (local only: the Modal app does not ship `demo/results/`, so the live link always scores
live):

```sh
uv run python demo/server.py precompute --persona demo/personas/alex_cs_f1.json --out demo/results/alex_cs_f1.json
uv run python demo/server.py precompute --persona demo/personas/alex_cs_f1.json --work-auth citizen_or_pr --out demo/results/alex_cs_citizen.json
```

It scores every posting the H'' rules let through (scores only, in chunks of 20; the file holds the model's scores
and decider labels, and the page reads the label from the scores again and applies the salary cap), then generates
reasons for the 8 most confident postings of each label (`--reasons-per-label`, so about 24 reason generations). The
file records the prompt-prefix hash, so it applies only to that resume and profile, plus a hash of each posting's
text and the model name: a rebuilt pool drops stale scores, and the page shows which model a file came from and
warns when it differs from the live scorer. Results hold ids, labels, probabilities and reasons, and no posting text.

## Deploy on Modal

`demo/modal_app.py` serves the page and its API from a small CPU container and the model from one L4. Settings live
in `demo/modal_settings.json` (model, dtype, monthly caps, links). Every container writes its uptime cost to a spend
ledger, and the app refuses a request once the month's total plus that request's worst case would pass the month's
cap. Rates are `demo/spend.py`'s: the L4 container (2 cores, 16 GiB) $1.02/h, the 4-core 32 GiB build container
$0.45/h, the web container $0.03/h.

```sh
PY=$(uv tool dir)/modal/bin/python   # a Python with the modal package (uv tool install modal)
D=data                               # the private data/ folder
```

Deploy run B with one command. It unpacks B's archive (zstd despite the name) into a temporary folder, puts the
adapter on the app's Volume, runs `prepare` (downloads Qwen/Qwen3.5-4B after checking its config against the
adapter's, merges on the build container), writes the settings, deploys and stops older containers. `caps` replaces
the whole table, and any month missing from it gets `cap_default` ($15); a cap of 0 deploys with the GPU off.

```sh
$PY scripts/modal_demo_deploy.py --adapter $D/models/main_4b_c3/B_8dc3d64a-cd0f-4807-a6e9-4eadb542ed3d.tar.gz \
  --set model=main_4b_B --set dtype=bfloat16 --set gpu_mem_gib=16 \
  --set 'caps={"2026-10": CAP, "2026-11": CAP}' \
  --set video_url=VIDEO_URL --set repo_url=REPO_URL
```

The merge takes about 5-15 minutes on the build container, roughly $0.04-0.11 (estimate). Archive sha256
`e0895cbe...2700eb35`, as `eval/selection_cycle3.json` pins it (`sha256sum` the file first).

Check the deployed model against B's eval (bfloat16, batch 1, as the eval ran), on fresh test 1, a selection set
(fresh test 4 stays scored once):

```sh
$PY scripts/modal_demo_check.py https://rolfyildirim--gradus.modal.run \
  --eval $D/models/c3_eval/B/B_fresh_test.json --rows $D/built/fit/eval_inputs/B/fresh_test.jsonl --n 40
```

The eval's label is the decider on its stored scores. It passes when at least 98% of labels agree once near-cuts
(the eval's s within 0.02 of a cut-off) are set aside. Measured on the first deploy (2026-10-22): 40/40 labels
agree, 0 flips; the 40 rows took 30 s and $0.009 on a warm GPU. A cold start adds about 100 s (107 s the first
time) and the idle tail about 3 minutes.

To look at H'' on the live page (one persona, through the public URL; about $0.03 warm or $0.13 from cold):

```sh
$PY scripts/modal_demo_check.py https://rolfyildirim--gradus.modal.run --pool demo/local/pool.jsonl
```

After a pool rebuild, run `modal volume put -f matcher-demo demo/local/pool.jsonl pool.jsonl` (the Volume kept the
project's working name). The page reads each pool row's card, so H'' uses whatever cards that file carries.

A visitor costs about $0.015 per warm request and $0.10 per cold visit (estimates); the cap stops all GPU use once
the month's total would pass it.

## Known limits

- "Anywhere in the US" is the demo's relocation default. The training pairs write it too (`metros.ANYWHERE_US`,
  `sample.py`, `variants.py`), so a model trained on them sees that profile line in distribution.
- The pool is real US tech internships. Gradus scores any candidate against any posting, but a non-tech pool would
  need its own build (`--pool` takes any jsonl in the same shape).
- Uploaded real resumes are out of the training distribution (Adaption-written resumes).
