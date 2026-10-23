# Which Adaption features Gradus uses

The shipped system is run B (cycle 3) behind H'' without work mode, read with the decider. This page lists each
Adaption feature in that system with the run or dataset ids and credits the `eval/` files record. Credits marked
"estimate" are the launch estimates the build scripts recorded per run; the cycle-3 build total is also the
measured change in the account balance.

## Used by the shipped system

| Adaption feature | Job in Gradus | Ids and counts | Credits |
|---|---|---|---|
| Adapt as extractor (universal prompt, blueprint `b2b2ed8e` fixes the card schema; `scripts/extract_postings.py`) | Reads each real posting into a requirement card: degree level and fields, enrollment, sponsorship, citizens-only, locations, salary, job type, work mode, skills. At inference the H'' rules read these cards | Test 4: `dataset-731a90af-53e5-46b6-bd2a-fd267ab35727-1792611525563` (240 postings) and the 4b extension's `dataset-60ad2625-c9dc-41c0-9d8d-4efe1f5254ca-1792614196153` (50); 289 of the 290 got a card. Cycle-3 training rows: `dataset-54d10abb-d5dc-4012-a92d-1bb94d16c0a0-1792608947525` (520 postings) | 4 + 1 (estimate); 7 (estimate) |
| Adapt as writer (universal prompt + blueprint) | Writes each synthetic resume from a candidate card sampled in code; code gates check the text against the card and send failures back with the failure quoted, up to three tries | Test 4: `dataset-cd07a1ff-55d1-45c8-92ae-ae735108d00b-1792611799677`, `dataset-5312280c-3917-4d26-9b0e-28f5205dd76e-1792612174052`, `dataset-33050660-3ecf-4571-a2ef-e57e2c35f4d1-1792612478890`, and for the extension `dataset-a5d1b674-a9db-4a5e-904e-96f2d40f6e66-1792616070286`, `dataset-eaf38430-7d6e-4838-b8cd-f73f5a63dd7f-1792616317180`, `dataset-e916d331-593a-419f-bc06-e9b9b3f973c4-1792616511037`. 1,139 resumes: 910 passed on the first try, 216 on the second, 13 on the third. Cycle 3: `dataset-61216c09-1068-40f0-ab3f-8d9afa984127-1792609270184`, `dataset-b734b664-5bf4-48c7-93ab-1808e7ee5c10-1792610630483`, `dataset-7761f7c7-25b8-43af-9f85-9a83c111479b-1792611260788`; 2,553 rows over 520 postings (2,117 first try, 397 second, 39 third) | Test 4: 11 + 2 + 1, extension 2 + 1 + 1 (estimate). Cycle 3: 27 + 5 + 1 (estimate) |
| Raw dataset upload (`processing_mode="raw"`, `scripts/train_main.py`) | Sends B's labeled rows up unchanged, so the labels reach training exactly as written | `train_B.csv`, 2,686 rows (1,501 of them new cycle-3 rows), md5 `6fb0f861e38924aac848a43c009d229e` | none recorded |
| AutoScientist SFT, one iteration with our hyperparameters | Trains run B: LoRA on Qwen3.5-4B, 3 epochs, lr 2e-4, r16 alpha 32, batch "max" | Run `8dc3d64a-cd0f-4807-a6e9-4eadb542ed3d`, succeeded, 33.0 minutes, AutoScientist's own win rate 0.8128 | Not separable: see below |
| AutoScientist download | The trained adapter as a PEFT archive (zstd despite the `.tar.gz` name), merged into the public Qwen3.5-4B weights by `scripts/merge_lora.py` for the eval and the demo | Archive sha256 `e0895cbe622cad1efce65807dbd6d3004b6bb5419a99cc6b77d7a9fd2700eb35` | none |

Sources: extraction and writer runs for test 4 in [`fresh_test4_manifest.json`](../eval/fresh_test4_manifest.json)
(`credits`, `extension_4b.credits`, `writer.tries`); for the cycle-3 rows in
[`cycle3_train_build.json`](../eval/cycle3_train_build.json) (`credits`, `render_tries`); the upload in
[`cycle3_train_csvs.json`](../eval/cycle3_train_csvs.json); run B in
[`selection_cycle3.json`](../eval/selection_cycle3.json) (`archives.B`).

Test 4 cost an estimated 18 credits, plus 5 for the 4b extension. The cycle-3 training rows cost
40: the balance went from 66 to 26 (`cycle3_train_build.json`). Training charged nothing we can attribute to run B.
Runs A and B trained in the same window, and over that window the shared balance went from 8 to 5 while the 4b
extension's extraction was also spending from it, so the 3 credits can't be split between them
([`diag_keep_levers.md`](../eval/diag_keep_levers.md)). `selection_cycle3.json` records `credits_spent: 3.0` on both
A and B because it logs the shared window.

Adaption supplies neither the labels nor the candidate cards. The Keep / Explore / Skip labels come from Claude labelers applying one written rubric: two of
them, blind to each other, for the test gold, with disagreements adjudicated; one of them (Opus) for B's training
rows. Candidate cards are sampled in code, so each resume's facts are known exactly.

## Not used by the shipped system

- Invent: tried early only. Cards are sampled in code instead, so the facts behind each label are exact, and
  Adapt writes the text.
- `recommend_hyperparams`: gave the first recipe (run a, which learned nothing); B uses the hyperparameters of
  the earlier runs h and d3 instead.
- AutoScientist's own hyperparameter search (`max_iterations` above 1) and a 9B model: tried in earlier cycles
  (runs e and f), level with the 4B recipe; B is one iteration on the 4B.
- Translate, language expansion, DPO or preference pairs, augmentation rows, reasoning traces, quality
  evaluation: not used. Gradus is US and English, and B is plain SFT on the answer.
- Hosting: AutoScientist offers downloads, so the live demo runs the merged model on Modal
  ([`DEMO.md`](DEMO.md)).

## History: runs a to f and d3

Before cycle 3, runs a to f and d3 trained on labels written by the rules on each pair's cards and were scored on
an older real-posting test. Run d3 was picked then, and run h replaced it on a dev set; cycle 3 replaced the rule
labels with the rubric labels above. B's 1,185 older training rows (558 from real
venator postings, 188 short teasers, 439 synthetic twin postings) were built with Adaption in those earlier cycles and
relabeled for B ([`cycle3_train_csvs.json`](../eval/cycle3_train_csvs.json), `by_source_kept`); the table above lists
only the cycle-3 builds. The earlier runs are not part of the shipped system; their selection records are
[`selection_v4.json`](../eval/selection_v4.json) and [`selection_cycle2.md`](../eval/selection_cycle2.md), and
[`RESULTS.md`](RESULTS.md) covers the path from h to B.
