# Fresh test 4: independent review of the one scoring

Reviewer: Sonnet (Agent tool, effort high), 2026-10-21, after `fit_eval_score.py fresh4` wrote eval/fresh_test4_result.json (c922be2). Report only: the reviewer changed nothing in the result and did not rerun fresh4. Its scratch scripts sit outside the project.

## Recomputed from the raw score files

The reviewer wrote its own code for argmax, softmax decider, H''/H' and macro-F1, starting from the three raw score files (data/models/c3_eval/test4/{B,h,base}/*_fresh_test4.json) and the cut-offs in eval/selection_cycle3.json.

- It recomputed rule verdicts for all 1,135 carded rows with rule_verdict on mark_thin(card, posting text). Rule, hard_cats, salary_soft, rules_full and eligibility_pass match on every row (eligibility-pass slice 513 rows).
- It compared 160 headline values (4 systems x 4 slices: all, eligibility_pass, agreed_only, agreed_eligibility_pass): macro-F1, accuracy, three recalls, Keep found/gold/predicted/precision, unsafe Keeps. All 160 match.
- It recomputed the bootstrap (2,000 draws, default_rng(20261021), postings then rows). The unrounded lower bounds match the result on every slice:

| slice | pick minus | lower bound |
|---|---|---|
| eligibility_pass (513, 36 gold Keeps) | rules_full | 0.07868158579058981 |
| eligibility_pass | base+H''_without_work_mode\|decider | 0.12360132947699122 |
| agreed_eligibility_pass (455, 29 Keeps) | rules_full | 0.12448706187340573 |
| agreed_eligibility_pass | base | 0.1542117125326819 |
| agreed_only (1,051, 34 Keeps) | rules_full | 0.13185933023925472 |
| agreed_only | base | 0.13001823351303626 |
| eligibility_pass_design (511, sensitivity) | rules_full | 0.08394643523940049 |
| eligibility_pass_design | base | 0.1213676859307353 |

Reviewer's verdict: PASS, the same as the result.

## Protocol checks (all pass)

- Commit order: gold manifest 039378a, then amendment 4c and selection_cycle3.json 0d127b8, then prompt pins 9f7ddff, scoring code 2490e58, result c922be2. Every score file's mtime falls after the code commit. fit_select_cycle3.py never reads test 4.
- The prompt files' sha256 equal eval/fresh_test4_eval_inputs.json. The score files' sha256 equal the result's score_files. Score headers: merged_B, merged_h, base_Qwen--Qwen3.5-4B; fresh_test4.jsonl; chat; bfloat16; batch 1; 1,139 unique rows.
- gold.jsonl sha256 equals the gold manifest's gold_sha256.
- The pick and its cut-offs equal selection_cycle3.json exactly. Base's comparator is base + H'' without work_mode on the decider, with its own committed cut-offs. Nothing was refit.
- Amendment 4c: the gold Keep count sits beside every Keep recall. Amendment 5: the agreement subset and the labeler ceiling are reported (0.9227 / kappa 0.7304 overall; 0.8869 / 0.7369 on the eligibility-pass slice). Amendment 5b: unsafe Keeps are reported. The new-employer slice (20 rows, one employer, all gold Skip) is flagged and kept out of the bar.

## The two choices the scorer made

- (a) The scored slice is the eligibility pass recomputed with mark_thin: 513 rows, against 511 under the sealed design flag. The two rows that move in are empty-body teasers, gold Explore. This follows amendments 2 and 5 ("recomputed from profile and card") and what tests 1-3 did. It can't change the verdict, because the sealed 511-row slice also clears both bars.
- (b) Four no-card rows (one QIAGEN posting) get no rule verdict and rules_full Explore, and sit outside the slice. The policy was pinned before inference and applied the same way to every system. All four are gold Skip, so the bar never reads them. On the 1,135 carded rows the pick's lead over rules_full is 0.219 [0.116, 0.312].

## Concerns (none can change the verdict)

- The tightest margin is pick minus rules_full, lower bound 0.079, on 36 gold Keeps (Keep recall moves in steps of 0.028).
- On the eligibility-pass slice the pick's accuracy lead over rules_full is not significant (0.019 [-0.032, 0.070]). The pick also has 20 unsafe Keeps among 44 predicted Keeps (precision 0.55), where rules_full has 7 among 10. The bar is macro-F1 only.
- data/ is gitignored. The integrity of the score files, gold and prompts rests on the sha256 values in committed files. All of them hold.
- The h run took a second Colab session. The first timed out during setup and produced no output (recorded in colab_sessions).
