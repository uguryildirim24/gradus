# Gradus

Live demo: https://rolfyildirim--gradus.modal.run (the first request after a quiet spell wakes the GPU, about
110 s)

Video (2:35): https://youtu.be/h3V7mkQGKT8

Model (run B's LoRA adapter): https://huggingface.co/Rolfier/gradus-qwen3.5-4b-lora

Give it a candidate's resume, a short profile (work authorization, degree, where they live and will move, job type,
salary floor) and a job posting. It answers Keep, Explore or Skip, with one line of reasons. The demo page
sorts a pool of about 150 real US tech internship postings into For you, Explore and Excluded for the
resume and profile you give it.

Built for the Best Use of Adaption Challenge (2026). Adaption does three jobs here: it extracts the requirements
from every real posting, writes the synthetic resumes the model is trained and tested on, and trains the model with
AutoScientist. The shipped system is run B + H'' without work mode, read with the decider, and it passed its sealed
test (fresh test 4, scored once).

The real postings come from venator, a store of public job postings collected from employers' applicant-tracking
boards; "venator" in file names and records means that store.

## What it does

A posting can rule a candidate out for reasons that have nothing to do with skill: US citizens only, no visa
sponsorship, current students only, a degree level or field, a place the candidate can't move to. And among the
postings that let a candidate through, the question is fit: does the resume show what the posting requires, read the
way a careful advisor would ("such as" lists, preferred items, duties that are not requirements).

Gradus answers both, for any candidate:

- Keep: nothing rules them out and they meet the posting's stated requirements.
- Explore: worth a look but not a clean fit (about a year short, most preferred items missing, an ambiguous
  requirement, or a posting that says too little to judge).
- Skip: something the posting states as a hard requirement rules them out.

## Two layers

```
 posting text ──> [Adaption] requirement card ──> rules on card + profile ──> Skip?  (eligibility, H'')
                                                                  │ no
 rubric + resume + profile + whole posting ──> run B (Qwen3.5-4B, trained on Adaption) ──> decider ──> Keep / Explore / Skip
```

1. Eligibility comes from Adaption-extracted requirement cards and rules. Adaption turns each posting into a requirement
   card (degree level and fields, enrollment, sponsorship stance, citizens-only, locations, salary, job type, work
   mode, skills). `src/matcher/rules.py` compares the card with the profile. Where the rules say Skip for work
   authorization, enrollment, degree level, degree field or location, the answer is Skip whatever the model says;
   when the posting's pay tops out under the candidate's salary floor, a Keep becomes Explore
   (`src/matcher/hybrid.py`, the H'' categories). Skills, years, job type and work mode are left to the model: a
   card can't hold how a posting hedges its skills, the job type override did more harm than good in the cycle-3
   ablation (`eval/diag_cycle3_cpu.json`, as `eval/protocol_amendment_5.json` records), and the variant without
   work mode won the cycle-3 pick.
2. Fit comes from the Adaption-trained 4B. Run B, a Qwen3.5-4B LoRA trained with Adaption AutoScientist, reads the
   labelers' rubric, the resume, the profile and the whole posting (`src/matcher/prompt_b.py`) and gives three
   first-token label scores. The decider (`src/matcher/decider.py`) turns them into a label: s = P(Keep) + 0.5
   P(Explore) at temperature 1; Keep when s >= 0.589, Skip when s < 0.273, else Explore. The cut-offs were fitted on
   fresh tests 1-3 before test 4 was scored (`eval/selection_cycle3.json`). With the label forced as its first token,
   the model writes the one-line reason.

## What Adaption does

[`docs/ADAPTION_FEATURES.md`](docs/ADAPTION_FEATURES.md) lists every Adaption run for test 4 and for run B's new training rows, with its ids and credits.

- Adapt extracts the requirement cards. In universal-prompt mode with a blueprint that fixes the card schema
  (`scripts/extract_postings.py`), it reads every real posting: 289 of test 4's 290 postings got a card (one extraction
  failed), and the 520 postings sampled for run B's new training rows went through the same extractor. At
  inference, the eligibility layer reads those cards.
- Adapt writes the resumes. Candidates are sampled in code as cards; Adapt writes each resume from its card, code
  gates check the text against the card, and a failed text goes back with the failure quoted (up to three tries).
  Test 4's 1,139 resumes: 910 passed on the first try, 216 on the second, 13 on the third
  (`eval/fresh_test4_manifest.json`). Run B's training set also holds synthetic twin postings Adapt wrote from cards.
  For run B, Adaption built 2,553 new rows (cards and resumes) for 40 credits; 1,501 of them are in B's 2,686-row
  training set (`eval/cycle3_train_build.json`, `eval/cycle3_train_csvs.json`).
- AutoScientist trains the model. The labeled rows go up raw, so the labels reach training exactly as written. Run B
  (`8dc3d64a-cd0f-4807-a6e9-4eadb542ed3d`) was one AutoScientist iteration of LoRA on Qwen3.5-4B: 3 epochs, lr 2e-4,
  r16, 33 minutes (`eval/selection_cycle3.json`, `archives.B`). Training charged no credits we can attribute to it:
  runs A and B shared one window in which the balance went from 8 to 5 while other work was also spending (the
  `credits_spent: 3.0` on both runs in that file logs the shared window and can't be split between the runs). The
  adapter downloads as a PEFT archive and is merged into the public Qwen3.5-4B weights for scoring and for the demo.

The labels themselves come from Claude labelers applying one written rubric: two of them, blind to each other, for
the test gold, with disagreements adjudicated; one of them for B's training rows.

## Result: sealed fresh test 4

1,139 rows over 290 real postings, scored once after the system was picked
([`eval/fresh_test4_result.md`](eval/fresh_test4_result.md)). No test-4 posting is in any training file or in the
sets that picked the system (`eval/fresh_test4_manifest.json`, `freshness`). The test was extended by 199 rows
before anything was scored on it, when its first 940 rows held too few gold Keeps (amendments 4b and 4c). 43 rows are gold Keep. A wrong Keep is a predicted Keep
whose gold is Explore or Skip.

| system | macro-F1 [95%] | Keeps found (of 43) | wrong Keeps |
|---|---|---|---|
| B + H'', decider (shipped) | 0.679 [0.599, 0.747] | 28 | 23 |
| h + H' (the previous pick) | 0.582 [0.505, 0.653] | 15 | 21 |
| rules on the extracted cards | 0.456 [0.390, 0.526] | 3 | 7 |
| untrained Qwen3.5-4B + H'', decider | 0.490 [0.435, 0.549] | 24 | 84 |

The pass bar, fixed before the test was built, is read on the 513 rows the hard filters let through (36 gold Keeps):
B must beat both the full rules and the untrained model behind the same eligibility layer, each paired macro-F1
difference with its 95% bootstrap lower bound above 0. B clears both: +0.192 [0.079, 0.286] over the rules and +0.217 [0.124,
0.300] over the untrained model. A second Claude reviewer (Sonnet) recomputed all 160 headline values and the
bootstrap bounds from the raw score files; all matched ([`eval/fresh_test4_review.md`](eval/fresh_test4_review.md)).
Without the rule layer, B alone scores 0.670 on all rows; H'' adds +0.009 [-0.010, 0.026], which is not
significant (`eval/test4_rule_layer.json`). Tests 1-3, the cycles from h to B and the protocol: [`docs/RESULTS.md`](docs/RESULTS.md).

Fresh test 5, a second sealed test with resumes written by Claude Sonnet instead of Adaption (1,187 rows on 300 real
postings), scored B after it shipped: on its eligibility-pass slice B finds 139 of 188 Keeps (Keep-F1 0.678), and
macro-F1 drops to 0.621 from test 4's 0.676, with the loss on Skip and Explore
([`docs/RESULTS.md`](docs/RESULTS.md#test-5-a-second-sealed-test-with-a-different-resume-writer)).

## Limits

- The labels are Claude labelers applying a written rubric, not recruiters. On test 4 the two labelers agree on
  92% of rows (kappa 0.73). B learned from one of those labelers, so it imitates the rubric as they read it.
- The resumes are synthetic, written by Adaption from sampled cards (test 5's by Claude Sonnet). The postings are real.
- The lead over the old model is not significant on the rows where the resume decides. On the 513-row
  eligibility-pass slice, B minus h + H' is +0.070 macro-F1 [-0.016, 0.152]. On all rows it is +0.097 [0.019, 0.180].
- About half the Keeps are wrong: Keep precision is about 55% (28 of 51), and B finds 28 of the 43 Keeps.
- The counts are small: 36 gold Keeps on the slice the bar reads. Test 4's postings are new, but almost all its
  employers also appear in tests 1-3, which chose the system and fitted the cut-offs.
- A requirement the extraction misses can't fire, so the eligibility layer is only as good as the extracted card.
- The demo pool is US tech internships. Gradus scores any candidate against any posting, but the page only holds
  this pool.
- Not legal advice on work authorization. Read the posting before you apply.

## Run it

The live demo is the quickest way to try Gradus. Running the page locally needs the private `data/` folder (the
real postings and their extracted cards stay out of the repo because the posting text is the employers'), and a
scorer serving run B. Without `data/`, `demo/build_pool.py` stops and says what is missing.

Tests run without `data/`; the ones that need it skip:

```sh
uv run --group demo pytest -q
```

The offline parity check (no model; needs `data/`) is the evidence that the demo gives the evaluated labels:
every row of fresh tests 1-4 through the page's scoring path with B's stored scores gives the committed label
(592/592, 371/371, 437/437, 1,135/1,135; test 4's 4 card-less rows are never in the demo pool), and every prompt
equals B's eval prompt byte for byte. The per-set table is in [`docs/DEMO.md`](docs/DEMO.md).

```sh
MATCHER_DATA_DIR=/path/to/data uv run --group demo pytest -q -s tests/test_demo_b_fresh.py
```

To run the page with `data/` (scorer details and the Modal deploy: [`docs/DEMO.md`](docs/DEMO.md)):

```sh
uv run python demo/build_pool.py                                            # real postings, kept local
uv run --group demo python demo/server.py --scorer http://127.0.0.1:18765    # http://127.0.0.1:8077
```

The scorer is the Modal app in `demo/modal_app.py` (one L4) or `scripts/demo_backend_torch.py` (torch on CPU,
minutes per request), both serving run B merged into Qwen3.5-4B. The page sends B's prompt, reads the label from the
three scores with the decider, and applies H'' before and after the model. No model runs on a laptop; every
evaluation ran on a Colab L4.

## Where things are

| path | what |
|---|---|
| `src/matcher/rules.py`, `hybrid.py` | the label rules on two cards; the eligibility layer (H, H', H'') |
| `src/matcher/prompt_b.py`, `decider.py` | run B's prompt; the decider readout and its cut-offs |
| `scripts/extract_postings.py`, `build_fresh_test.py` | Adaption extraction; the fresh-test and cycle-3 row pipeline (cards, Adaption-written resumes, gates) |
| `scripts/build_fit_train3.py`, `train_main.py` | cycle-3 training sets; AutoScientist runs |
| `scripts/fit_select_cycle3.py`, `fit_eval_score.py` | the pick on tests 1-3; write-once scoring of the sealed tests |
| `eval/` | the protocol, its amendments, selections and every committed result ([`eval/README.md`](eval/README.md)) |
| `demo/` | the page, its server, the Modal app |
| `docs/` | [`RESULTS.md`](docs/RESULTS.md) (tests 1-5, the cycles and run C), [`ADAPTION_FEATURES.md`](docs/ADAPTION_FEATURES.md), [`DEMO.md`](docs/DEMO.md) |

The Python package is still called `matcher`, the project's working name.

## Data and licences

Code: MIT ([`LICENSE`](LICENSE)). O*NET 31.0 (CC BY 4.0, USDOL/ETA; modified). Postings from public listings and ATS
boards; employer text stays out of the repo (the demo pool is rebuilt locally). Synthetic cards and resumes are ours.
