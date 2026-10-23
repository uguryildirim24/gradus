# Results: fresh tests 1-5, from run h to run B

Every number here is copied from a committed file under `eval/`; the link next to each table says which. The
per-row files those results were computed from live in `data/` (gitignored); the committed files pin their sha256.

## The first four sealed tests

Each fresh test is a new set of real postings (none in any training file), with synthetic candidates whose resumes
Adaption wrote from sampled candidate cards, and gold Keep / Explore / Skip labels from two blind labelers (Claude
Opus and Claude Sonnet, both applying the same written rubric) with every disagreement adjudicated. Each test was
built and sealed (a data-hash manifest committed) before any system was scored on it, and its recorded result below
was scored once. Tests 1-3 were later reused as selection and practice sets (see the cycles below), and test 3's
postings come from the dev pool: 63 of its 112 had rows in the dev set that picked h over d3 (selection exposure,
not training; [`protocol_amendment_3.json`](../eval/protocol_amendment_3.json), `disclosures.dev_fit`).

| test | rows / postings | gold Keeps | system scored | macro-F1, all rows [95%] | Keeps found | pass bar | verdict |
|---|---|---|---|---|---|---|---|
| 1 | 592 / 150 | 57 | h + H' | 0.539 [0.463, 0.614] | 16 of 57 | beat skills-free rules | passed |
| 2 | 371 / 95 | 20 | h + H' | 0.637 [0.523, 0.723] | 10 of 20 | beat full rules and base + H' | not passed |
| 3 | 437 / 112 | 20 | h + H' | 0.492 [0.407, 0.583] | 3 of 20 | beat full rules and base + H' | not passed |
| 4 | 1,139 / 290 | 43 | B + H'' (decider) | 0.679 [0.599, 0.747] | 28 of 43 | beat full rules and base + H'' | passed |

Sources: [`fresh_test_result_v4.json`](../eval/fresh_test_result_v4.json),
[`fresh_test2_result.md`](../eval/fresh_test2_result.md), [`fresh_test3_result.md`](../eval/fresh_test3_result.md),
[`fresh_test4_result.md`](../eval/fresh_test4_result.md). The bar is always read on the eligibility-pass slice (the
rows the hard filters let through, which are the rows where the resume decides), as a paired macro-F1 difference
whose 95% two-way bootstrap lower bound (postings, then rows within them) must be above 0:

| test | slice rows (gold Keeps) | difference against | result [95%] | beaten |
|---|---|---|---|---|
| 1 | 344 (52) | skills-free rules | +0.127 [0.024, 0.223] | yes |
| 2 | 164 (19) | full rules | +0.152 [-0.002, 0.273] | no |
| 2 | 164 (19) | base + H' | +0.345 [0.208, 0.480] | yes |
| 3 | 213 (18) | full rules | +0.056 [-0.046, 0.171] | no |
| 3 | 213 (18) | base + H' | +0.055 [-0.077, 0.195] | no |
| 4 | 513 (36) | full rules | +0.192 [0.079, 0.286] | yes |
| 4 | 513 (36) | base + H'' (decider, its own cut-offs) | +0.217 [0.124, 0.300] | yes |

Test 1 passed the original protocol's bar, which only asked to beat the rules with the skill checks taken out. On
the same slice the full rules scored 0.505 macro-F1 and h + H' 0.529 (`fresh_test_result_v4.json`), so
amendment 2 raised the bar for every later test: beat the full rules and the untrained model behind the same
eligibility layer. h did not clear that bar on test 2 or test 3. B cleared it on test 4.

"Full rules" is `src/matcher/rules.py` on the Adaption-extracted card, skills included. "Base" is the untrained
Qwen3.5-4B behind the same eligibility layer; on test 4 it is read with the decider too, with cut-offs fitted the
same way on tests 1-3 (amendment 5b), so both models share the readout and the bar measures the training.

Labeler agreement, the ceiling for any system (raw agreement, Cohen's kappa): test 1 0.892 (0.775), test 2 0.898
(0.804), test 3 0.892 (0.757), test 4 0.923 (0.730).

## Test 4 in full

From [`fresh_test4_result.md`](../eval/fresh_test4_result.md), all 1,139 rows (968 gold Skip, 128 Explore, 43 Keep).
A wrong Keep is a predicted Keep whose gold is Explore or Skip (the result file calls it an unsafe Keep).

| system | macro-F1 [95%] | accuracy | Keeps found (of 43) | Keep precision | wrong Keeps |
|---|---|---|---|---|---|
| B + H'' without work mode, decider (shipped) | 0.679 [0.599, 0.747] | 0.877 | 28 | 0.549 (28 of 51) | 23 |
| h + H' (the previous pick) | 0.582 [0.505, 0.653] | 0.856 | 15 | 0.417 (15 of 36) | 21 |
| full rules on the extracted card | 0.456 [0.390, 0.526] | 0.860 | 3 | 0.300 (3 of 10) | 7 |
| untrained Qwen3.5-4B + H'', decider | 0.490 [0.435, 0.549] | 0.734 | 24 | 0.222 (24 of 108) | 84 |

B against h + H', paired macro-F1 difference (reported, not part of the bar; `fresh_test4_result.json`,
`slices.*.paired`):

| slice | rows | B minus h + H' [95%] |
|---|---|---|
| all rows | 1,139 | +0.097 [0.019, 0.180] |
| eligibility pass | 513 | +0.070 [-0.016, 0.152] (interval includes 0) |
| labelers agree | 1,051 | +0.112 [0.022, 0.200] |
| labelers agree, eligibility pass | 455 | +0.108 [0.025, 0.199] |

On the slice the bar reads, B's lead over the old model is not significant. On the point estimates B finds more
Keeps (28 against 15) at a similar number of wrong Keeps (23 against 21); no interval was computed for those counts,
and the two systems also differ in their eligibility layer (H'' against H').

What the eligibility layer adds on test 4 (reported, not part of the bar; recomputed on CPU from B's stored scores
by `scripts/test4_rule_layer.py`, [`test4_rule_layer.json`](../eval/test4_rule_layer.json)):

| system | macro-F1 [95%] | Keeps found (of 43) | wrong Keeps |
|---|---|---|---|
| B alone, decider | 0.670 [0.591, 0.735] | 29 | 27 |
| B + H'' (shipped) | 0.679 [0.599, 0.747] | 28 | 23 |

H'' changes 17 of B's 1,139 labels (12 Explore to Skip, 4 Keep to Skip, 1 Keep to Explore): 11 become right, 2
become wrong. The paired difference is +0.009 [-0.010, 0.026], so the gain is small and not significant. On the
513-row eligibility-pass slice the two are identical (0.676) by construction, so the pass bar measures the model.

A second Claude reviewer (Sonnet) recomputed the 160 headline values and the bootstrap bounds from the raw score files
and found them all equal: [`fresh_test4_review.md`](../eval/fresh_test4_review.md). Four rows (one posting) have no extracted card;
the policy pinned before inference ([`fresh_test4_eval_inputs.json`](../eval/fresh_test4_eval_inputs.json)) gives
them no rule verdict and keeps them out of the slice. All four are gold Skip.

## Test 5: a second sealed test with a different resume writer

Test 5 was built after B shipped, to check B on new rows and to judge one more candidate (run C, next section). It
holds 1,187 rows on 300 real postings from 222 employers, and none of its postings are in tests 1-4
([`fresh_test5_manifest.json`](../eval/fresh_test5_manifest.json)). The postings come from an older pool, and 21 of
them share text with a training file: their own older rows in the files that trained the retired runs h and d3,
and for two (Moderna, Regeneron) employer boilerplate in B's training set, where the overlap rule found no near copy
(`freshness.training_text`). Two things differ from test 4. The Adaption
credits were spent, so Claude Sonnet wrote the resumes, with test 4's writer instruction, gates and retries (921
passed on the first try, 253 on the second, 13 on the third); the postings' requirement cards are Adaption
extractions made earlier. And the candidate mix leans toward Keep by design, as amendment 6 declared before the
build ([`protocol_amendment_6.json`](../eval/protocol_amendment_6.json)). Gold Keeps are 17.8% of rows against test
4's 3.8%: 211 of 1,187, and 188 on the 676-row eligibility-pass slice, far above the 60-70 amendment 6 expected
for this design. The gold manifest traces the larger part of the rise to the candidates, not the writer. Within
every candidate type, test 5's skills lists hold about three times as many of the card's required skills as test
4's, since these tech postings name concrete tools the sampler finds; the resume adds about 1 point over its skills
list on test 5 and 2-3 points on test 4 ([`fresh_test5_gold_manifest.json`](../eval/fresh_test5_gold_manifest.json)).
Gold comes from the same setup as test 4: Opus and Sonnet label blind, and Opus adjudicates the 179 disagreements.
The two labelers agree on 84.9% of rows (kappa 0.719).

Before anything was scored, amendment 6 made test 4 a dev set. The comparison model for the bar, refit-B, is B with
its decider cut-offs refit on tests 1-4 pooled ([`cutoffs_B_t1to4.json`](../eval/cutoffs_B_t1to4.json)); the shipped
B keeps the cut-offs it shipped with. B as shipped on the eligibility-pass slice of each test
([`fresh_test5_result.md`](../eval/fresh_test5_result.md), [`fresh_test4_result.md`](../eval/fresh_test4_result.md)):

| B + H'', decider (shipped) | test 4: 513 rows, 36 gold Keeps | test 5: 676 rows, 188 gold Keeps |
|---|---|---|
| macro-F1 [95%] | 0.676 [0.592, 0.745] | 0.621 [0.571, 0.667] |
| Keep-F1 [95%] | not in the test-4 result | 0.678 [0.603, 0.739] |
| Keep recall | 0.667 (24 of 36) | 0.739 (139 of 188) |
| Keep precision | 0.545 (24 of 44) | 0.626 (139 of 222) |
| wrong Keeps | 20 | 83 |
| Explore recall | 0.528 | 0.448 |
| Skip recall | 0.892 | 0.701 |

Macro-F1 fell from 0.68 to 0.62 on this slice, and from 0.679 to 0.650 on all rows. The loss is on Skip and
Explore. B finds a larger share of the Keeps than on test 4, and more of its Keeps are right, but it lets through
more rows the gold calls Skip or Explore. The strong candidates show it most: on their 302 pass-slice rows (115 gold
Keep, 62 Explore, 125 Skip) B's Explore recall is 0.323 and its Skip recall 0.640, and 44 of its 83 wrong Keeps are
there (`slices.archetype:strong_eligibility_pass` in [`fresh_test5_result.json`](../eval/fresh_test5_result.json)).
On the 540 pass-slice rows where both labelers agree, B scores 0.652 macro-F1 and 0.734 Keep-F1. Test 5 changed the
writer and the candidate mix together, so it can't say how much of the drop each one causes.

Refit-B has the same Keep-F1 and 0.616 macro-F1. The full rules on the cards score 0.358 macro-F1 and 0.160 Keep-F1
on the same rows. A Sonnet reviewer recounted the headline numbers and the bootstrap bounds with its own code and
found nothing to fix ([`fresh_test5_review.md`](../eval/fresh_test5_review.md)).

## What we tried after: run C

Run C was an attempt to cut B's wrong Keeps, judged on test 5. Amendment 6b fixed the plan before any C label or
test-5 score existed ([`protocol_amendment_6b.json`](../eval/protocol_amendment_6b.json)):

- Opus relabeled the 1,292 Keep and Explore rows of B's training set with the rubric plus one line: "A required
  item the resume names but doesn't show doing is not met." 288 labels changed, 262 of them to a stricter label
  ([`cycleC_train_csv.json`](../eval/cycleC_train_csv.json)). An outside Sonnet reader graded 50 relabeled rows and
  found 0 clear errors ([`cycleC_gates.json`](../eval/cycleC_gates.json)). After the label-mix band, C's training
  set has 2,281 rows.
- The Adaption credits were spent, so C was trained on Colab (an A100) with B's recipe. A control, B-colab, trained
  B's own training set the same way: C against B-colab measures the relabel, and B-colab against refit-B measures
  the trainer ([`colab_train_C.json`](../eval/colab_train_C.json),
  [`colab_train_Bcolab.json`](../eval/colab_train_Bcolab.json)). A Colab-trained model can't ship as the contest
  model, so a pass would only have shown the relabel works.
- The bar, on test 5's eligibility-pass slice: C minus refit-B needs a Keep-F1 lower bound above 0 and a macro-F1
  lower bound above -0.03.

On that slice ([`fresh_test5_result.md`](../eval/fresh_test5_result.md)):

| system | Keep-F1 [95%] | macro-F1 [95%] | Keeps found (of 188) | wrong Keeps |
|---|---|---|---|---|
| B as shipped | 0.678 [0.603, 0.739] | 0.621 [0.571, 0.667] | 139 | 83 |
| refit-B | 0.678 [0.603, 0.739] | 0.616 [0.565, 0.659] | 139 | 83 |
| B-colab | 0.674 [0.598, 0.740] | 0.613 [0.565, 0.656] | 128 | 64 |
| C | 0.651 [0.563, 0.720] | 0.603 [0.552, 0.650] | 111 | 42 |

C cut the wrong Keeps from 83 to 42, and it also found fewer real ones: 111 of 188 against 139. C minus refit-B is
-0.027 Keep-F1 [-0.091, 0.028] and -0.013 macro-F1 [-0.063, 0.034]. Both lower bounds miss the bar, so C failed.
B-colab minus refit-B is -0.004 Keep-F1 [-0.052, 0.037] and -0.003 macro-F1, so the trainer makes no detectable
difference; C minus B-colab, which isolates the relabel, is -0.023 Keep-F1 [-0.079, 0.029]. B stays the shipped model.

## Three cycles: h, h kept, then B

Cycle 1 (run h) was a Qwen3.5-4B LoRA trained with Adaption AutoScientist on `train_fit.csv`: labels written by the
rules on each pair's cards, h's prompt (an instruction, the resume, the profile, the posting cut to 6,000
characters), argmax label. It was picked over the earlier d3 on a dev set only
([`selection_v4.json`](../eval/selection_v4.json): 0.813 against 0.771 macro-F1 behind H'), and scored once on test 1.

In cycle 2, run h2n, trained on careful-reading labels where fit decides, came within the 0.01 tie margin
of h on test 1, now a selection set, so the incumbent h stayed
([`selection_cycle2.md`](../eval/selection_cycle2.md)). Test 2 and then test 3 were scored once with h + H': neither
passed.

In cycle 3 (run B), the diagnosis after test 3 found that the rule labels h was trained on agree with careful
reading on only 53% of fit rows, and that runs h, d3 and h2n all plateau there; the test labelers agree with each
other about 89% of the time on tests 1-3 ([`protocol_amendment_5.json`](../eval/protocol_amendment_5.json),
`teacher.why`). So cycle 3 changed the labels:

| | run h | run B |
|---|---|---|
| training labels | the rules on the cards | a Claude labeler (Opus) applying the rubric the test labelers use, one label per row |
| training rows | `train_fit.csv` | 2,686 rows: 1,501 new rows built with the fresh-test pipeline, plus earlier training rows (real postings, short teasers, synthetic twin postings) relabeled; Skip downsampled into the label-mix band (Keep 26%, Explore 22%, Skip 52%) |
| prompt | h's instruction, resume, profile, posting cut to 6,000 characters | the rubric on top, then resume, profile and the whole posting |
| hyperparameters | 3 epochs, lr 2e-4, LoRA r16 alpha 32, one AutoScientist iteration | the same |
| eligibility layer | H': work authorization, enrollment, degree level and field, employment type, location, work mode, salary cap | H'' without work mode: H' minus employment type and work mode |
| readout | argmax of the three label scores | decider: s = P(Keep) + 0.5 P(Explore), Keep at s >= 0.589, Skip below 0.273 |

Sources: [`cycle3_train_csvs.json`](../eval/cycle3_train_csvs.json) (rows, mix, prompts),
[`protocol_amendment_5.json`](../eval/protocol_amendment_5.json) (teacher, runs, H''),
[`protocol_amendment_5b.json`](../eval/protocol_amendment_5b.json) (decider, a 0.60 Keep-recall floor on the pick),
[`selection_cycle3.json`](../eval/selection_cycle3.json) (the pick, the cut-offs, B's archive).

Run A got the same new labels in h's prompt format, so the pick on tests 1-3 shows where the gain came from. Pooled
macro-F1 on the 1,400 rows of tests 1-3 (behind H'' without work mode; [`selection_cycle3.json`](../eval/selection_cycle3.json)):

| | argmax | decider |
|---|---|---|
| h | 0.582 | 0.525 |
| A (new labels, h's prompt) | 0.727 | 0.742 |
| B (new labels, rubric and whole posting) | 0.758 | 0.766 |

On these point estimates (no paired test), most of the gain came with the new labels (h to A) and B's prompt added a
little more. These are practice-set numbers:
tests 1-3 had already been scored and read, they chose the pick, and the decider's cut-offs were fitted on these same
rows, so the decider column is in-sample. Test 4 is the only clean number for B.

## The protocol

- Sealed tests: each fresh test was built from postings no earlier test used and (test 5's 21 older-pool
  postings aside, above) no training file used, labeled by
  two blind labelers, and sealed with a data-hash manifest ([`fresh_test4_manifest.json`](../eval/fresh_test4_manifest.json),
  [`fresh_test4_gold_manifest.json`](../eval/fresh_test4_gold_manifest.json)) before any system ran on it.
- Choices pinned before results: the original protocol ([`protocol.json`](../eval/protocol.json)) was frozen
  before any fit data was built, and each amendment carries a `written_before` field: amendment 1 pinned every scoring choice before any model output existed;
  2 raised the bar and made test 1 a selection set; 3 and 3b sealed and then scored test 3; 4 made tests 1-3
  practice sets and asked for a new sealed test 4; 4b extended test 4 when its first 940 rows held only 34
  eligibility-pass gold Keeps; 4c accepted 36 (under the 40 asked for); 5 and 5b fixed the teacher, the runs, H'',
  the decider and how the pick is chosen, before any cycle-3 label or score existed; 6 and 6b made test 4 a dev
  set, sealed test 5 and fixed run C's relabel, its Colab training and its bar before any test-5 score existed.
- Selection never touches the sealed test: h was picked on a dev set; cycle 2 picked on test 1; cycle 3 picked
  on tests 1-3 pooled ([`selection_cycle3.json`](../eval/selection_cycle3.json)), before test 4's prompts were pinned
  ([`fresh_test4_eval_inputs.json`](../eval/fresh_test4_eval_inputs.json)).
- Write-once scoring: each `fresh_test*_result` file is written once by `scripts/fit_eval_score.py`, which refuses
  to overwrite it; amendment 4 says the recorded results of tests 1-3 stand as written and are never rescored.
- Same reading everywhere: every model was scored on Colab L4 in bfloat16, batch 1, from the float32 first-token
  label scores; macro-F1 and the two-way bootstrap (2,000 draws, seed 20261021; 20261022 on test 5) are defined once and reused. The
  eligibility-pass slice is recomputed with the rules in effect when a test is scored: test 1's recorded 344 rows
  predate the work-mode rule (271 rows under the later rules, [`selection_cycle2.json`](../eval/selection_cycle2.json)),
  and tests 3 and 4 report the sealed-design slice beside the recomputed one (205 against 213, 511 against 513).

## Limits

- The gold is Claude labelers applying a written rubric, not recruiters or hiring outcomes. On test 4 the two
  labelers agree on 92.3% of rows (kappa 0.73). B's training labels come from one of those labeler setups, so B
  learns to imitate the labeler.
- The resumes are synthetic (written by Adaption from sampled candidate cards, and by Claude Sonnet on test 5);
  the postings are real. On test 5, Sonnet wrote the resumes that one of the two labelers, also Sonnet, labels;
  amendment 6 records that coupling and does not correct it.
- Test 4 is new at the posting level; almost all of its employers also appear in tests 1-3,
  which chose the pick. Its 20 rows from a new employer are all gold Skip
  ([`fresh_test4_result.md`](../eval/fresh_test4_result.md), new-employer slice).
- Small counts: 43 gold Keeps on test 4, 36 on the slice the bar reads, so Keep recall moves in steps of about 3
  points.
- Keep precision is about 55%: almost half of B's Keeps are wrong by the gold. It finds 28 of 43 Keeps.
- B's lead over h + H' on the eligibility-pass slice is not significant (+0.070 [-0.016, 0.152]).
