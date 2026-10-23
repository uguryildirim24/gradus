# eval/: the records, and what the commit ids in them point to

This folder holds the protocol, its amendments, the selections and every committed result. The public repo is a
fresh-history export: one commit made from the final tree of the private working repo. The records were written
during the work, so the commit ids inside them (`parent_commit`, `computed_at_commit`, `commit_of_builder`, a commit
named in a note) refer to that private history and do not resolve here.

You can verify the sha256 chain here. A manifest pins its data files; each amendment pins the amendment
and manifests it builds on; selections and results pin the inputs they read. `scripts/fit_eval_score.py` checks
those pins before it scores. The data files live in `data/` (gitignored), so with `data/` you can check
every data hash too. [`REDACTIONS.md`](REDACTIONS.md) lists the files edited before release and their old and new
hashes.

The private commits the records cite, in order (UTC):

| commit | when | cited in | what it was |
|---|---|---|---|
| `c6c814b` | 2026-10-21 00:01 | `protocol_amendment_1.json`, `fit_eval_hashes.json`, `fresh_test3_manifest.json` | froze the fresh-test, dev and train pools and the original protocol |
| `857e829` | 2026-10-21 00:43 | `protocol_amendment_1.json` | rules: thin postings are Explore |
| `9ec9082` | 2026-10-21 06:18 | `protocol_amendment_2.json` | fresh test 1 scored once |
| `c6c4790` | 2026-10-21 07:24 | `protocol_amendment_3.json`, `fresh_test3_manifest.json` | rules: a ruled-out work mode is a hard Skip |
| `142dc8e` | 2026-10-21 07:25 | `fix_cycle2.json` | extractor v3 |
| `1c2825f` | 2026-10-21 12:17 | `protocol_amendment_3.json` | fresh test 2 sensitivity checks |
| `73990e9` | 2026-10-21 13:09 | `fresh_test3_data_hashes.json` | fresh test 3 gold assembled |
| `02f4851` | 2026-10-21 13:26 | `protocol_amendment_3b.json` | fresh test 3 scoring package, nothing scored |
| `406135c` | 2026-10-21 18:13 | `diag_cycle3_rowread.md` | amendment 4: tests 1-3 become practice sets |
| `01d1b7f` | 2026-10-21 18:26 | `protocol_amendment_5.json` | cycle-3 diagnostic on tests 2 and 3 |
| `8c6bac5` | 2026-10-21 18:50 | `protocol_amendment_5b.json`, `fresh_test4_manifest.json`, `cycle3_train_build.json` | candidate generator fix used by test 4 and the cycle-3 rows |
| `7883b96` | 2026-10-21 18:55 | `cycle3_train_build.json` | the builder for test 4 and the cycle-3 training rows |
| `2b13479` | 2026-10-21 19:53 | `fresh_test4_data_hashes.json` | zero-shot diagnostic on tests 1-3 |
| `0aa5173` | 2026-10-21 20:01 | `fresh_test4_data_hashes.json` | fresh test 4 sealed (first 940 rows) |
| `322517d` | 2026-10-21 20:18 | `protocol_amendment_4b.json` | test 4 gold for the first 940 rows |
| `4ca7027` | 2026-10-21 20:57 | `fresh_test4_data_hashes.json` | the cycle-3 selection script on tests 1-3 |
| `039378a` | 2026-10-21 21:12 | `protocol_amendment_4c.json`, `fresh_test4_review.md` | test 4 gold over all 1,139 rows |
| `0d127b8` | 2026-10-21 21:50 | `fresh_test4_review.md` | `selection_cycle3.json`: the cycle-3 pick |
| `9f7ddff` | 2026-10-21 21:56 | `fresh_test4_review.md` | test 4's pinned prompts (`fresh_test4_eval_inputs.json`) |
| `2490e58` | 2026-10-21 22:08 | `fresh_test4_review.md` | the write-once scoring code for test 4 |
| `c922be2` | 2026-10-21 23:19 | `fresh_test4_review.md` | fresh test 4 scored once |
| `b2f7d1c`, `0fdbf50` | 2026-10-22 01:49, 01:57 | `diag_keep_levers.md` | the Keep-lever diagnostic |

`tests/test_label_read_1762a19.py` is named after private commit `1762a19` (2026-10-20 08:00), the merge change whose
label read it tests.

The records rely on this order. Amendments 5 and 5b were committed at 18:42 and 18:54 UTC, before test 4 was sealed
(`0aa5173`, 20:01). The pick on tests 1-3 (`selection_cycle3.json`, committed 21:50) and test 4's pinned prompts
(`fresh_test4_eval_inputs.json`, 21:56) came before test 4 was scored once (`c922be2`, 23:19). The `written_before`
field in each amendment says the same in words.
