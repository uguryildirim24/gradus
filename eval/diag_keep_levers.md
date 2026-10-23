# Keep levers: more real Keeps, fewer wrong Keeps

Written 2026-10-21. CPU read of stored files only: no model run, no training, no Adaption credit spent. Numbers come from `scripts/diag_keep_levers.py` → `eval/diag_keep_levers.json` (commits b2f7d1c, 0fdbf50), plus two row reads (all 40 missed Keeps, all 66 wrong Keeps on tests 1-4) and a training/cost read. A Sonnet reviewer recomputed the numbers from the raw score files and spot-read 16 rows (PASS with notes), and a second Sonnet pass checked this file against the JSON and the raw files; its corrections are folded in. Aggregates only: no row ids, employers or row text here.

Terms. True Keep: the pick (B + H'' without work_mode, decider) says Keep and gold is Keep. Wrong Keep: the pick says Keep and gold is Explore or Skip. "Pass slice" is the eligibility-pass slice the bar is read on. Test 4 is used up, so every fitted threshold below is fitted on tests 1-3 (or leave-one-test-out) and only read on test 4.

## Short answer

- Nothing in the data raises true Keeps by more than a handful. Most misses sit where the teacher itself was unsure, and a 4B student can't be surer than its teacher.
- One lever cuts wrong Keeps by about three quarters at a small cost in true Keeps: a second-pass rubric re-read of B's Keep calls only. Pair it with a lower Keep cut-off fitted on tests 1-3 and true Keeps come back to about the pick's level, with wrong Keeps still down by 11 to 15 of 17-21 on test 4.
- Cut-off moves, the P(Keep) readout, averaging runs and eligibility-layer tweaks don't move both numbers. Cut-offs trade one for the other; the rest are within noise or flip sign between tests.
- No lever is proven. Test 4 is used, and a fresh test 5 needs Adaption credits that aren't there (balance about 1).

## 1. Where the missed Keeps go

| | gold Keeps | found | missed | Explore band | low-score Skip | H'' Skip | salary cap |
|---|---|---|---|---|---|---|---|
| tests 1-3 | 97 | 72 | 25 | 17 | 5 | 3 | 0 |
| test 4 | 43 | 28 | 15 | 9 | 4 | 2 | 0 |

- **Explore band (26 of 40).** B's top class is Explore, not "just under the cut-off". On test 4 the band misses have s 0.39-0.56 (median 0.48) and P(Keep) under 0.2 on 6 of 9. Only two misses in all four tests sit within 0.01 of t_hi.
- **Low-score Skips (9).** Confident Skips (s under 0.27, most under 0.21). No cut-off reaches them. On test 4 all 4 are rows both labelers called Keep.
- **H'' Skips (5).** 3 of the 5 cards are wrong (a dropped or invented degree level, a "one of" list read as a required degree); one more card misses an "equivalent experience" clause; one is the exact-city rule failing a short commute. Only one of the five would have been a Keep without H''.
- **Labeler disagreement.** Every one of the 40 missed Keeps has labeler A (Opus, the teacher's setup) = Keep, 38 of them at A's "medium" confidence. B finds 46 of 48 gold Keeps A called with high confidence, and 53 of 91 it called with medium confidence. Split gold Keeps (A Keep, B Explore, adjudicated Keep) are missed 10 of 17 times; agreed ones 30 of 122. On test 4, 5 of the 9 band misses are split rows.

What the 40 misses are, by reading them:

| bucket | tests 1-3 | test 4 |
|---|---|---|
| genuinely borderline (one stated line not shown, years right at the minimum, management shown only by a title, labelers split) | 14 | 8 |
| B misread one line's status (a leveled posting's higher-level minimums, "a minimum of N years ... preferred" read as required, a boilerplate classification line, a "such as" list it didn't credit, a current student read as graduated) | 5 | 4 |
| no-requirements posting read as "too little to judge" | 2 | 0 |
| H'' Skip | 3 | 2 |
| gold looks wrong | 1 | 1 |

So about 22 of 40 misses are judgment calls where B's Explore is fair. About 11 are fixable reading errors, and 5 of those are confident Skips. On test 4, 5 of the 15 misses are fixable in principle (4 misreads, 1 card error).

## 2. What the wrong Keeps have in common

| | tests 1-3 | test 4 |
|---|---|---|
| wrong Keeps | 43 (31 gold Explore, 12 gold Skip) | 23 (15 Explore, 8 Skip) |
| both labelers non-Keep | 30 | 18 |
| B's top class is Keep | 34 | 18 |
| s ≥ 0.75 | 21 | 11 |

They aren't near-threshold hedges: B is confident. On the test-4 pass slice, 16 of the 20 wrong Keeps have no labeler voting Keep, so they are real errors, not ambiguity.

By reading them:

| bucket | tests 1-3 | test 4 |
|---|---|---|
| a required skill, license or kind of experience the resume names but doesn't show doing | 17 | 7 |
| profile mismatch the eligibility layer doesn't cover or gets wrong (work mode, job type, state residency, territory) | 9 | 5 |
| teaser, or too little information | 8 | 3 |
| an eligibility clause the card missed (enrollment, finished degree, F-1 CPT, local work right) | 4 | 3 |
| about a year or more short on required experience | 3 | 4 |
| most preferred items missing | 2 | 1 |

The main wrong Keep: the posting has a hard "required" line (a tool, a kind of work, a language, a state license), the resume carries the keyword or a nearby one, and it doesn't show doing it. Both labelers see the gap (mostly Explore, sometimes Skip); B calls Keep. The rules' skills check fires on most of these rows but can't be the catch: it also fires on 43 of 72 true Keeps on tests 1-3.

"A year short / most preferred missing" accounts for only 10 of 66.

## 3. Why test 4's Keep rate is lower

Pass-slice Keep rate: 17.7%, 11.6%, 8.5%, 7.0% across tests 1-4. A shift-share split puts almost all of the fall on the labels, not the build mix: within the same design cell (archetype × target_pass) the Keep rate fell (−0.060 of the −0.061 change). Strong designs fell from 0.258 on test 1 to 0.070 on test 4. Both labelers fell together (A 0.181 → 0.078, B 0.199 → 0.070). The resume-title copy that the generator fix removed doesn't explain it: on tests 1-3 pass-slice strong designs, rows whose resume holds the posting's title have a Keep rate no higher than rows without (0.15-0.19 against 0.20-0.22, depending on how the title match is defined). Harder postings within a cell can't be told apart from stricter labels here. For sizing test 5, use test 4's rate.

## 4. The levers

Change against the pick, test-4 pass slice (pick: 24 true / 20 wrong Keeps, 36 gold Keeps), then tests 1-3 (leave-one-test-out where a lever is fitted). 95% two-way bootstrap intervals in brackets.

| lever | test 4: Δ true | Δ wrong | tests 1-3: Δ true | Δ wrong | verdict |
|---|---|---|---|---|---|
| t_hi 0.50 | +1 [0, 4] | +8 [2, 17] | +8 [2, 16] | +54 [35, 75] | pure trade |
| t_hi 0.70 | −5 [−11, 0] | −9 [−17, −3] | −13 [−24, −5] | −16 [−27, −7] | pure trade |
| Keep when P(Keep) ≥ τ (τ 0.365, count-matched) | −1 [−4, 0] | −1 [−4, 0] | −2 [−10, 5] | −1 [−11, 8] | nothing |
| base-rate correction (π = test 3's 8.45%) | −3 [−8, 0] | −6 [−12, −1] | – | – | a t_hi raise |
| demote Keep on years or license flags | 0 to −1 | −1 to −3 | 0 to −2 | −2 to −3 | small, mostly clean |
| demote Keep on skills flag | −13 [−21, −5] | −11 [−20, −4] | −43 | −21 | harmful |
| average B with h | −2 [−6, 2] | −4 [−12, 4] | −10 [−21, −1] | +5 [−11, 20] | flips sign |
| average B with base | −3 [−9, 3] | −10 [−19, −2] | −14 [−27, −2] | +20 [0, 40] | flips sign |
| average A with B | – (A not scored on test 4) | – | −3 [−10, 4] | +1 [−9, 12] | nothing |
| veto: Keep only if base also clears a bar | −3 [−7, 0] | −6 [−13, −1] | −12 [−22, −4] | −9 [−18, −2] | a t_hi raise |
| drop H'' entirely (all rows) | +1 | +4 | – | – | H'' barely matters |
| second-pass check of B's Keeps (cross-labeler) | −1 to −5 | −16 | see below | see below | best single lever |

Notes:
- **Cut-offs** move along one curve. Keep AUC outside H'' is 0.951 (s) and 0.960 (P(Keep)) on test 4; at the pick's 23 wrong Keeps (all rows) every single-model readout reaches 28-29 true Keeps.
- **Averaging** with base looked best on test 4 and costs true Keeps and adds wrong ones on tests 1-3. On tests 1-3, A+B moves Keep AUC 0.935 → 0.941 on s and 0.951 → 0.952 on P(Keep) (most of the gap is the readout), and changes nothing at the operating point.
- **Eligibility layer.** Each H'' category kills at most 1 true Keep and stops at most 2 wrong Keeps per test. Card fixes would reach about 1 missed Keep and about 9 wrong Keeps across all four tests (per diem, "APAC remote", territory, finished-degree and student clauses), mostly outside the pass slice.
- **Cheap flags.** Teaser, years and license flags catch 13 wrong Keeps for 4 true on tests 1-3 and 4 for 1 on test 4. Two teaser-like wrong Keeps on test 4 carry no teaser flag; a "no requirements section" check would catch them. The fitted flag combination picked a different set in every leave-one-test-out fold, so treat only the single flags as real.

### The second-pass Keep check

A rubric reader re-reads only the rows B calls Keep and demotes them to Explore when it says not-Keep. Simulated with the labelers. Scoring a labeler against gold is circular (gold is built from those labels), so the honest estimate is cross-labeler: one labeler as the reader, the other as the reference.

Test-4 pass slice, true / wrong Keeps:

| reader → reference | pick | pick + check | check + t_hi 0.41 |
|---|---|---|---|
| labeler B → labeler A | 27 / 17 | 22 / 1 | 24 / 2 |
| labeler A → labeler B | 23 / 21 | 22 / 5 | 24 / 10 |
| rows read, of 513 | – | 44 | 102 |

Tests 1-3 pass slice, same layout: B → A 66 / 31, 61 / 9, 72 / 11; A → B 70 / 27, 61 / 5, 72 / 8.

- t_hi 0.41 is fitted on tests 1-3 with the check in place: the grid value (0.30-0.59, step 0.01, plus the pick's t_hi) with the highest mean true-minus-wrong Keeps over both directions, ties to the higher t_hi. 0.33-0.41 tie on a plateau, so there's no sharp optimum. The tests 1-3 cells at 0.41 are in-sample; test 4 is read afterwards.
- The pick + check and check + 0.41 cells come from the author's recompute, not the JSON; the reviewer reproduced every cell. The JSON's `4f_second_pass.both` is a different variant (a band re-read) and gives 24/2 and 25/10.
- With the check alone, the change on test 4 is −1 to −5 true Keeps and −16 wrong Keeps; the wrong-Keep interval is [−27, −8]. The true-Keep interval touches zero on one bootstrap and not on another, so the honest wording is "costs 1-5 true Keeps" on test 4. On tests 1-3 the check alone costs more: −5 and −9 on the pass slice; on all rows −6 [−13, −1] and −11 [−20, −4].
- With the lowered cut-off, true Keeps are −3 to +1 against the pick and wrong Keeps −15 to −11. That is the only combination that holds true Keeps and cuts wrong ones.
- The reader reads about 86 rows per 1,000 pass-slice rows at the pick's cut-off and about 200 at t_hi 0.41 on test 4 (about 150 and 410 on tests 1-3, which have more Keeps). That's the cost driver.
- Both labelers are LLMs reading the same rubric. A bias they share goes unmeasured, and the cross-labeler numbers are an estimate for a third reader, not a measurement of one.
- A reader in the teacher's setup (Opus, rubric) is the obvious production choice. It isn't an Adaption model, so whether it belongs in the entry is a product call.

## 5. Training data, model size, teacher fixes

- **Teacher's unsure zone.** In the cycle-3 labels without venator-source rows, Keep is 416 rows (48% high confidence), Explore 491 (2% high), Skip 2,465 (75% high). B's misses track A's medium-confidence Keeps (above). More data of the same kind won't make B surer than the teacher. A sharper Keep line in the rubric, or training labels kept only where two labelers agree, is the data change that could raise true Keeps; there's no evidence yet that it would.
- **Free hard negatives.** The 1,416 rows dropped by the band downsample are already labeled. 524 of them are new rows on pass designs, Skips for years (187), skills (103) and degree (75): the same kind as the main wrong Keep. Adding them back breaks amendment 5's Skip band, so it needs an amendment. Expected effect: fewer wrong Keeps; the effect on true Keeps is unknown and could be negative.
- **Graduation timing.** About 7 train_B rows turn on it. No labeler or adjudicator reason on any of the 2,539 test rows mentions it. A teacher fix there moves tests 1-4 by about zero. It's a fix for real F-1 students, not a metric lever.
- **Bigger model.** The only 9B run (f, an earlier cycle with rules gold) was level with the 4B on the old real test (macro-F1 +0.011 [−0.029, +0.055]). No cycle-3 evidence. Its training cost isn't recorded (0-4 credits).
- **Retrain cost.** Runs A and B ran together on one balance, 8 → 5 credits, about 33 minutes each. The 4b extension's extraction took one of those credits, so a 4B run costs about 1-1.5 credits.

## 6. Costs and how to prove it

| lever | Adaption credits | Colab L4 | other |
|---|---|---|---|
| cut-off, readout, flags, card fixes | 0 | 0 | code |
| second-pass check | 0 | 0 | one reader call per Keep candidate (9-20% of pass rows on test 4, 15-41% on tests 1-3) |
| average A with B | 0 | about 18 min per test to score A | – |
| retrain with hard negatives or relabeled rows | about 1-1.5 | about 18 min per test | an amendment |
| 9B | unknown, likely more | longer sessions | – |
| any proof: fresh test 5 | see below | about 15-20 min per system | gold labeling |

Test builds so far cost 15, 9, 10, 18 and 5 credits (test 4 was 4 for extraction, 14 for rendering). About 0.015-0.02 a posting for extraction, 0.012-0.015 a row for rendering. None of the unused reserved postings has an extracted card yet.

What a test 5 can and can't prove:

- **A Keep-enriched test 5** (Keep-leaning candidate designs, declared enriched before the build): about 600 rows on about 150 postings for 50 pass-slice Keeps, about 10-12 credits. Wrong Keeps must be reported per design and reweighted to test 4's mix.
- **It can prove the wrong-Keep cut.** A drop of about 16 of 20 is far outside the noise at 50 Keeps.
- **It can't prove a true-Keep gain.** For 80% power on +4 true Keeps per 43, the conservative setting (each net gain made of 6 gained and 2 lost Keeps) needs about 170 gold Keeps on a strong-design-only mix (about 1,400 rows; about 2,000 at the strong + near mix above) and about 260 Keeps (6,900 rows) at test 4's mix; power at 43 gold Keeps is 0.06. With no churn it needs 86 and 129 Keeps. Either way it is well past 50.
- **Gold source.** If the lever is an Opus or Sonnet rubric reader, test 5's gold can't be the same Opus + Sonnet rubric setup. Options: a labeler from another model family plus adjudication, or a person labeling blind only the rows any system calls Keep or near-Keep (about 100-150).
- **Resumes.** Claude-written resumes would bring the build to about 2-3 credits, but every training and test row so far is Adaption-rendered, so the style shift would be mixed into whatever is tested.
- **Money.** The balance is about 1 credit, at about $1 a credit. The enriched test 5 needs about $10-12 of credits; a retrain plus that test, about $12-14.

## 7. Ranking

By how much each lever helps both numbers at once, with the evidence behind it:

1. **Second-pass Keep check plus a lower cut-off fitted on tests 1-3.** True Keeps held (−3 to +1), wrong Keeps −11 to −15 of 17-21 on test 4. 0 credits to build; one reader call per Keep candidate. Cross-labeler estimate only.
2. **Second-pass Keep check alone.** Wrong Keeps −16, true Keeps −1 to −5. Half the reads of (1).
3. **Cheap flags (teaser or no-requirements section, years, license).** About −4 wrong / −1 true on test 4, −13 / −4 on tests 1-3. Free, deterministic.
4. **Card fixes in the eligibility layer.** About 1 missed Keep and 9 wrong Keeps across four tests, mostly off the pass slice. Free.
5. **Retrain with the 524 free hard negatives.** Targets the main wrong Keep; about 1-1.5 credits plus an amendment; effect on true Keeps unknown.
6. **Teacher fix at the Keep line (two-labeler training labels).** The only data lever aimed at true Keeps; agent time plus a retrain; no evidence yet.
7. **Averaging runs, P(Keep) readout, dropping H''.** Within noise or sign-flipping.
8. **Cut-off moves alone.** A straight trade.
9. **9B.** No cycle-3 evidence.
10. **Graduation-timing teacher fix.** Zero on tests 1-4; a product fix for F-1 students.

Recommendation: build lever 1 with lever 3's flags, fix the cut-off on tests 1-3 before any test-5 row exists, and prove it on a Keep-enriched test 5 whose gold doesn't come from the reader's own setup. That test can show the wrong-Keep cut. It can't show a true-Keep gain, and nothing here promises one.
