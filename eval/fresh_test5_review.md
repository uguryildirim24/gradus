# Fresh test 5: independent recount (step 7 of the run-C brief)

Reader: Claude Sonnet, effort high, own code (no import of scripts/fresh_test5_score.py), no model loaded.
Inputs checked: test5_rows.jsonl sha256 = result per_row_file; the three raw score files match their recorded sha256s
and the per-row scores row for row (0 mismatches, B / B-colab / C); every system label re-derived from scores, cut-offs
and the H'' override (HP_CATS minus employment_type and work_mode; salary-soft Keep -> Explore) matches on all 1,187
rows; pass-slice and agreed-pass-slice Keep-F1, macro-F1, Keep found/gold, precision and wrong Keeps match to 1e-12;
the two-way bootstrap (2,000 draws, seed 20261022) reproduces C - refit-B Keep-F1 -0.0270 [-0.0913, 0.0282] and
macro-F1 -0.0134 [-0.0625, 0.0341], and B-colab - refit-B -0.0044 [-0.0520, 0.0371] / -0.0029 [-0.0427, 0.0353].
Pass call: NO PASS is right (both lower bounds fail). Nothing found to fix.
Not re-derived: the eligibility-pass flags themselves (read from the row file; the result records 676 recomputed = 676
design, 0 differences).
