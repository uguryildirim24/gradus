"""The main AutoScientist runs on the assembled training set: Qwen3.5-4B (a, b, c one iteration each; e AutoScientist's
loop), Qwen3.5-9B (f: run c's recipe on the bigger model) and Qwen3.5-4B on train_mix3.csv (d: run c's recipe on the
new data; d3: the same at 3 epochs), Qwen3.5-4B on the v4 fit set (h: d3's recipe on train_fit.csv), on the
cycle-2 sets (h2b: h's recipe on train_fit2_bal.csv; h2n: h's recipe on train_fit2_nat.csv) and on the cycle-3 rubric
teacher's labels (A: h's recipe on cycle3/train_A.csv; B: h's recipe on cycle3/train_B.csv).

  train_main.py upload                       raw upload of the CSV (prompt/completion), wait until it is processed,
                                             refuse when Adaption's row count differs from the CSV's
  train_main.py f upload                     the same for run f, into f's own state file (a second dataset id);
                                             d upload and d3 upload the same for d and d3 (train_mix3.csv by default),
                                             h upload for h (train_fit.csv by default), h2b upload and h2n
                                             upload for h2b and h2n (train_fit2_bal.csv, train_fit2_nat.csv),
                                             A upload and B upload for A and B (cycle3/train_A.csv, train_B.csv)
  train_main.py hparams / f hparams          recommend_hyperparams for the model on that dataset (free)
  train_main.py config [RUN]                 print what create would send (no network; RUN defaults to a)
  train_main.py RUN seed                     (c, e) start the run's own state file from main_4b.json's dataset and
                                             recommendation (reads main_4b.json, never writes it; refuses if it exists);
                                             (f, d, d3) start the run's file from run c's sent recipe in
                                             main_4b_c.json (read only); h from run d3's in main_4b_d3.json;
                                             h2b, h2n, A and B from run h's in main_4b_h.json
  train_main.py RUN create --max-credits N   read credits, launch the run, read credits again
  train_main.py RUN wait [--max-credits N]   poll until terminal; cancel once the run has spent more than N credits
  train_main.py RUN download                 save the best checkpoint archive to data/models/main_4b/
  train_main.py status [RUN] / RUN status     free: dataset rows, each run's status, iterations, best win rate,
                                             download_available, spend against its cap, and the balance
RUN is a, b, c, e, f, d, d3, h, h2b, h2n, A or B (A and B are capital letters: not runs a and b). Run a took the recommendation as it was and learned almost nothing (21 steps at lr 1e-5); runs b
and c are the same dataset with RUN_OVERRIDES["b"] / ["c"] over the recommendation. c runs beside b, longer and
hotter; the two are chosen between on dev and dev_cross, not on the real test.

Run e (SEARCH) is AutoScientist's own search: a, b and c were each one iteration of a recipe set up front. Adaption's
docs say a supplied hyperparam replaces the derived value for that field ("any field supplied here replaces the
derived value"; "Unset hyperparameter keys remain under platform control"; "An override can prevent AutoScientist from
exploring the configuration it would otherwise select"), so a supplied field is pinned, not a seed for the search. e
therefore sends no recipe: only PINS (training_type lora, train_on_inputs False), augmentation 0, max_iterations 5 and
target_win_rate 0.95, so the loop starts from AutoScientist's recommendation (the recipe that left run a at 0.5533)
and revises learning rate, epochs, warmup, scheduler and the LoRA shape between iterations. The two pins only remove
lora-vs-full (colab_eval and merge_lora merge an adapter) and the prompt masking choice (~5,800-character prompts
against a ~20-word answer). --epochs is refused for e: it would pin n_epochs. The loop stops early once an iteration
reaches target_win_rate; c's single iteration scored 0.8579, so 0.95 keeps it from stopping after a c-like pass.
e's state file is data/adaption/main_4b_e.json.

State lives in its own file, data/adaption/main_4b.json (never smoke_run_v2.json): the dataset, its CSV's md5 and row
count, the recommendation, and one record per run. Run c has a file of its own, data/adaption/main_4b_c.json, picked
by the run letter (STATES), and so has run e (main_4b_e.json): b's wait loaded main_4b.json once and rewrites all of it
on every poll, so a run c record in that file would be erased within two minutes. a, b, upload and hparams keep
main_4b.json; c's and e's commands never open it, except `RUN seed`, which only reads it. upload, hparams and create refuse when the CSV on disk is not the
one the stored dataset was uploaded from (md5), so a rebuilt set can never train under an old dataset id, and `upload`
refuses to replace a stored dataset (move the state file away to start over). wait, download and status only warn: a
run already going must stay under its cap.

What create sends: the recommendation's hyperparams as they are (hparams must have run), with training_type lora
(colab_eval/merge_lora merge a LoRA adapter only) and train_on_inputs False pinned over it (prompts are ~5,800
characters against a ~20-word answer; only the answer is trained), the run's loop settings (LOOPS: max_iterations 1
and target_win_rate 0.8 for a, b and c), and both augmentation row counts 0. A run's RUN_OVERRIDES go over the recommendation before the pins, so no override can
undo a pin; --epochs overrides n_epochs after both. Settings outside Adaption's documented limits are refused.

The credit cap is ours, not AutoScientist's: autoscientist.create takes no budget or credit limit. `create` refuses
when the balance is under the cap; `wait` reads the balance (datasets.invent(estimate=True), free) on each poll, every
30 s growing to 120 s, and cancels the run once it has spent more than the cap. Every network call in the wait loop
(run get, credits read, cancel) retries on any exception with backoff (10 s doubling to 300 s, without end), so a
dropped connection sleeps instead of ending the cap while the run keeps spending. Each poll and each retry is logged
with a UTC timestamp. The cap only holds while `wait` runs: run it on oci-pi in tmux, not on a laptop that sleeps. A
poll interval can overshoot by what the run spends in two minutes. A run cancelled mid-iteration leaves nothing to
download, so the cap is set high enough for the one iteration to finish.

Several iterations (e): cancel is whole-run; Adaption has no per-iteration cancel, budget or checkpoint call. The run
object reports only iterations_completed, best_win_rate and best_hyperparams, never an iteration's own score, so when
iterations_completed goes up the wait logs the best at that moment and whether it changed (unchanged: that iteration
scored at or under the old best, by how much is not reported) and keeps the list in the run's "iterations". Each time
download_available is true while the best is new (a best win rate and hyperparams not yet saved), the wait saves that
best archive, so a cap cancel mid-loop still leaves the best finished iteration on disk, if Adaption allows a download
before the run ends (its docs only promise one after `succeeded`). At the end (succeeded, or cancelled with
download_available) it downloads again when the final best differs from the last one saved: the archive Adaption
serves is the best iteration's, not the last one's. A cancel mid-iteration loses the iteration in progress.

Run f (Qwen/Qwen3.5-9B, listed by GET /autoscientist/models on 2026-10-20: lora only) is run c with only the model
changed. It sends RECIPES["f"], the 15 hyperparams c's create sent as recorded in main_4b_c.json, as they are: no
recommendation goes under them, so no 9B-specific default (min_lr_ratio, max_grad_norm, weight_decay ...) can slip in.
`f seed` starts main_9b_f.json from c's file (read only) and refuses unless c's recorded hyperparams equal RECIPES["f"].
The docs' limits for the 9B (/autoscientist/supported-models, read 2026-10-20) take c's recipe unchanged: SFT batch
"max" (8, as for the 4B), gradient accumulation 1, lora_r up to 64, lora_alpha 1x or 2x lora_r; the general ranges
(epochs 1-5, lr 1e-6 to 5e-4) are the same for every model. What does differ: the 9B's SFT training context is 65,536
tokens, the 4B's 131,072 (rows are at most ~9,000 characters, so nothing is truncated), so if Adaption packs rows to the
context an epoch takes up to twice c's 21 steps at half the tokens a step (84 to 168 steps for 4 epochs).
f trains on a dataset of its own: Adaption allows one training run per dataset at a time (a second create on
66d2ad3d while e runs returned 409 TRAINING_LAUNCH_IN_FLIGHT, "A training run is already in progress for this
dataset"), so `f upload` uploads the same CSV again under its own name (main-train-<md5[:8]>-9b) into main_9b_f.json,
with the same md5 and row-count checks, and reads the balance before and after. f's commands never open main_4b.json,
main_4b_c.json (except `f seed`, which only reads it) or main_4b_e.json; f's archive goes to data/models/main_9b/ and its
create line to data/adaption/main_9b_run_f.txt.

Runs d and d3 are run c on new data: train_mix3.csv (scripts/build_train_mix3.py; 7,362 rows, md5 8393d474..., the
6,891 smoke.csv rows unchanged plus 471 real-posting rows). d sends RECIPES["d"], c's 15 sent hyperparams as they are;
d3 sends the same with n_epochs 3 (run e's search landed on 3 epochs at lr 2e-4, and run b's fewer passes kept more
high-ask-gap Keeps). As for f, no recommendation goes under them: on a new dataset Adaption could recommend other
values for the fields c never set (min_lr_ratio, scheduler_num_cycles, max_grad_norm, weight_decay), so the recipe is
copied whole instead of laid over a recommendation as RUN_OVERRIDES are. `d seed` and `d3 seed` start
main_4b_d.json and main_4b_d3.json from c's file (read only) and refuse unless c's sent hyperparams, with the run's
SEED_DIFF (d3: n_epochs 3; d: nothing), equal its RECIPES entry. Each uploads train_mix3.csv as a dataset of its own
(main-train-8393d474-d and -d3; one training run per dataset at a time, so the two can train side by side as e and f
did), and `upload` refuses a CSV whose md5 is not the reviewed one (CSV_MD5). Their create lines go to
main_4b_run_d.txt / main_4b_run_d3.txt and their archives to data/models/main_4b/ as d_<run>.tar.gz / d3_<run>.tar.gz.
train_mix3.csv holds about 8% more characters than smoke.csv, so an epoch may take up to 23 steps against run a's 21
(d: 84 to 91 steps, d3: 63 to 68). Both waits read the same account balance, so each cap also sees the other run's
spend.

Run h is d3's recipe on the v4 fit set: data/built/fit/train_fit.csv (scripts/build_fit_train.py; 10,563 rows, md5
b1fdfa85..., 3,521 each of Keep, Explore and Skip), the only change the data. It sends RECIPES["h"], d3's 15 sent
hyperparams as they are (RECIPES["d3"]); `h seed` starts main_4b_h.json from d3's file (read only) and refuses unless
d3's sent hyperparams equal it. `h upload` sends train_fit.csv raw as dataset main-train-b1fdfa85-h with the
prompt/completion mapping: the raw upload keeps only original_prompt and original_completion, so the CSV's
reasoning_trace column never reaches Adaption. h is the no-trace run. Its with-trace twin (g) cannot be built from a
raw upload: probes on 2026-10-20 found the trace column dropped under {prompt, completion} and under a
reasoning_trace key in the upload mapping, and folded into original_prompt after "Answer:" under `context` (the
verdict leaked into the prompt); the only documented trace path is adapt with recipes.reasoning_traces, which writes
Adaption's traces, not ours. train_fit.csv holds 63,867,052 prompt and completion characters against smoke.csv's
40,576,158 (1.574x), so an epoch of packed rows may take up to 57% more steps than run a's (63 to 99 steps).

Runs h2b and h2n are run h's recipe on the cycle-2 sets (scripts/build_fit_train2.py, reviewed after commit
1ef158a): careful-reading labels where fit decides, the fixed rules' labels where eligibility decides. h2b trains on
train_fit2_bal.csv (564 rows, md5 dc5ea96f..., 188 each of Keep, Explore and Skip, every row careful), h2n on
train_fit2_nat.csv (4,829 rows, md5 0aeb3812..., Keep 1,674, Explore 258, Skip 2,897; 2,794 careful and 2,035
eligibility rows). Both send RECIPES["h"] as it is; `h2b seed` and `h2n seed` start main_4b_h2b.json and
main_4b_h2n.json from h's file (read only) and refuse unless h's sent hyperparams equal it. Each uploads raw as a
dataset of its own (main-train-dc5ea96f-h2b, main-train-0aeb3812-h2n), so the two can train side by side; both CSVs
hold only prompt and completion. Steps: every run so far, a to h, took 21 steps an epoch (trainer_state.json
max_steps 21, 42, 63 or 84 for 1 to 4 epochs) on 6,891 to 10,563 rows, h's 1.574x characters included, so 63 steps is
the expected count for both. No run has trained on fewer than 6,891 rows; if an epoch did scale with the data,
h2b's 3,682,551 characters (0.091x smoke.csv's) would take as few as 6 steps and h2n's 30,141,166 (0.743x) 47.

Runs A and B (eval/protocol_amendment_5.json runs) are run h's recipe on the cycle-3 sets of scripts/build_fit_train3.py:
the rubric teacher's labels (labeler A's setup, data/built/fit/cycle3/labels/A) on the same rows, after the label-mix
band check. A's prompt is h's (matcher.pairs.build_prompt, the posting fitted to 6,000 characters); B's opens with the
rubric (labeler_prompt_test3's opening and rubric_bullets, verbatim) and carries the whole posting. Both completions
are "Label — reason." with the teacher's reason. Both send RECIPES["h"] as it is; `A seed` and `B seed` start
main_4b_A.json and main_4b_B.json from h's file (read only) and refuse unless h's sent hyperparams equal it. Each
uploads raw as a dataset of its own (main-train-<md5[:8]>-A / -B), so the two can train side by side. Their create
records and archives go under the tag main_4b_c3 (data/adaption/main_4b_c3_run_A.txt, data/models/main_4b_c3/
A_<run>.tar.gz), apart from run a's and b's names on a case-insensitive disk.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
from adaption import Adaption  # noqa: E402
from adaption.types.autoscientist_create_params import Hyperparams as CreateHyperparams  # noqa: E402

STATE = ROOT / "data/adaption/main_4b.json"
# a run listed here keeps its state in its own file; every other run, upload and hparams use STATE
STATES = {"c": ROOT / "data/adaption/main_4b_c.json", "e": ROOT / "data/adaption/main_4b_e.json",
          "f": ROOT / "data/adaption/main_9b_f.json", "d": ROOT / "data/adaption/main_4b_d.json",
          "d3": ROOT / "data/adaption/main_4b_d3.json", "h": ROOT / "data/adaption/main_4b_h.json",
          "h2b": ROOT / "data/adaption/main_4b_h2b.json", "h2n": ROOT / "data/adaption/main_4b_h2n.json",
          "A": ROOT / "data/adaption/main_4b_A.json", "B": ROOT / "data/adaption/main_4b_B.json"}
CSV = ROOT / "data/built/training/smoke.csv"  # assemble_training.py build writes the training set here
MIX3 = ROOT / "data/built/real_train/train_mix3.csv"  # build_train_mix3.py build writes it here
FIT = ROOT / "data/built/fit/train_fit.csv"  # build_fit_train.py writes the v4 fit set here
FIT2_BAL = ROOT / "data/built/fit/train_fit2_bal.csv"  # build_fit_train2.py variants writes the cycle-2 sets here
FIT2_NAT = ROOT / "data/built/fit/train_fit2_nat.csv"
C3_A = ROOT / "data/built/fit/cycle3/train_A.csv"  # build_fit_train3.py writes the cycle-3 sets here
C3_B = ROOT / "data/built/fit/cycle3/train_B.csv"
# a run listed here trains on another CSV by default (--csv still overrides); every other run uses CSV
RUN_CSVS = {"d": MIX3, "d3": MIX3, "h": FIT, "h2b": FIT2_BAL, "h2n": FIT2_NAT, "A": C3_A, "B": C3_B}
# the md5 a run's upload must find: the train_mix3.csv that passed review (7,362 rows); h: the reviewed train_fit.csv
# (10,563 rows); h2b and h2n: the reviewed cycle-2 sets (564 and 4,829 rows)
CSV_MD5 = {"d": "8393d4746f887332b20254ef7135db29", "d3": "8393d4746f887332b20254ef7135db29",
           "h": "b1fdfa85f3cb4b406f692e48b07f29fb", "h2b": "dc5ea96f8eab21a567637f3ec2b941c2",
           "h2n": "0aeb3812df83ce5db175258ad1ce4532"}
# A and B: the reviewed cycle-3 sets (eval/cycle3_train_csvs.json, 2,686 rows each; upload refuses None)
CSV_MD5 |= {"A": "45baa1edd61628a0bb33eba8372a0231", "B": "6fb0f861e38924aac848a43c009d229e"}
MODELS = ROOT / "data/models/main_4b"
MODEL = "Qwen/Qwen3.5-4B"  # the id GET /autoscientist/models lists (lora and full, 262k context)
# a run listed here trains another base model; every other run trains MODEL
RUN_MODELS = {"f": "Qwen/Qwen3.5-9B"}  # GET /autoscientist/models: 9B, lora only, 262k serving context
# the stem of a run's create record (<tag>_run_<run>.txt) and its archive directory (data/models/<tag>)
TAGS = {"f": "main_9b", "A": "main_4b_c3", "B": "main_4b_c3"}
# runs that upload the CSV as a dataset of their own (in their own state file) instead of training on main_4b.json's
OWN_DATASET = frozenset({"f", "d", "d3", "h", "h2b", "h2n", "A", "B"})
# f's dataset name: main-train-<md5[:8]>-9b, apart from main_4b.json's main-train-<md5[:8]>; d's and d3's -d and -d3
UPLOAD_SUFFIX = {"f": "-9b", "d": "-d", "d3": "-d3", "h": "-h", "h2b": "-h2b", "h2n": "-h2n", "A": "-A", "B": "-B"}
MAPPING = {"prompt": "prompt", "completion": "completion"}
# a raw upload canonicalizes the mapped columns to original_prompt / original_completion
TRAIN_MAPPING = {"prompt": "original_prompt", "completion": "original_completion"}
RUNS = ("a", "b", "c", "e", "f", "d", "d3", "h", "h2b", "h2n", "A", "B")
MAX_ITERATIONS = 1
TARGET_WIN_RATE = 0.8  # the platform default for larger models; pinned so the stopping bar is known before launch
# Runs whose recipe is left to AutoScientist's loop: create sends PINS as the only hyperparams (a supplied field is
# pinned for the run, per Adaption's docs), no recommendation and no overrides.
SEARCH = frozenset({"e"})
# the loop settings each run sends: one iteration for a, b and c; e gets the documented maximum of 5 iterations
# (a and c measured 32.9 and 33.5 min a run) and a target above c's 0.8579 so a c-like pass does not stop the loop
LOOPS = {n: {"max_iterations": MAX_ITERATIONS, "target_win_rate": TARGET_WIN_RATE}
         for n in ("a", "b", "c", "f", "d", "d3", "h", "h2b", "h2n", "A", "B")}
LOOPS["e"] = {"max_iterations": 5, "target_win_rate": 0.95}
PINS = {"training_type": "lora", "train_on_inputs": False}
# Run b: run a's 21 steps at lr 1e-5 left the model at its base. batch_size stays "max": it is the only SFT batch
# Adaption takes (8 for the 4B; the decided 16 is not allowed). Packed as in run a, "max" ran 21 steps an epoch, so
# 2 epochs is about 42 steps, warmup 0.03 about 1 step.
RUN_OVERRIDES = {
    "a": {},
    "b": {"learning_rate": 1e-4, "n_epochs": 2, "warmup_ratio": 0.03, "lr_scheduler_type": "cosine",
          "lora_r": 16, "lora_alpha": 32, "lora_trainable_modules": "all-linear", "lora_dropout": 0.0,
          "batch_size": "max"},
}
# Run c, beside b: everything as b but 4 epochs at lr 2e-4 and warmup 0.05, about 84 steps (warmup about 4).
RUN_OVERRIDES["c"] = RUN_OVERRIDES["b"] | {"learning_rate": 2e-4, "n_epochs": 4, "warmup_ratio": 0.05}
# Runs sent a whole recipe as it is, with no recommendation under it (the pins still go over it). f is run c's create
# exactly: the hyperparams main_4b_c.json records as sent on 2026-10-20 (run 1bbd8313, 0.8579 win rate, 33.5 min).
RECIPES = {"f": {"training_type": "lora", "n_epochs": 4, "learning_rate": 2e-4, "batch_size": "max", "lora_r": 16,
                 "lora_alpha": 32, "lora_dropout": 0.0, "lora_trainable_modules": "all-linear",
                 "lr_scheduler_type": "cosine", "min_lr_ratio": 0.1, "scheduler_num_cycles": 0.5, "warmup_ratio": 0.05,
                 "max_grad_norm": 2.0, "weight_decay": 0.0, "train_on_inputs": False}}
# d: c's recipe on train_mix3.csv, the only change the data; d3: the same at 3 epochs
RECIPES["d"] = dict(RECIPES["f"])
RECIPES["d3"] = RECIPES["f"] | {"n_epochs": 3}
# h: d3's recipe on train_fit.csv, the only change the data
RECIPES["h"] = dict(RECIPES["d3"])
# h2b, h2n: h's recipe on the cycle-2 sets, the only change the data
RECIPES["h2b"] = dict(RECIPES["h"])
RECIPES["h2n"] = dict(RECIPES["h"])
# A, B: h's recipe on the cycle-3 sets, the only change the data (amendment 5: "run h's hyperparameters unchanged")
RECIPES["A"] = dict(RECIPES["h"])
RECIPES["B"] = dict(RECIPES["h"])
# the seed reads this run's state file for the recipe it copies, not main_4b.json
SEED_FROM = {"f": "c", "d": "c", "d3": "c", "h": "d3", "h2b": "h", "h2n": "h", "A": "h", "B": "h"}
# the only fields a seeded recipe may differ in from what SEED_FROM's run sent
SEED_DIFF = {"d3": {"n_epochs": 3}}
RUN_A_STEPS_PER_EPOCH = 21  # run a's trainer_state.json: max_steps 21, num_train_epochs 1, train_batch_size 1 (packed)
# steps an epoch as a multiple of run a's, where the training context differs: the 9B trains at 65,536 tokens against
# the 4B's 131,072, so packed rows fill up to twice the batches (1x if Adaption does not pack to the context)
STEP_SPAN = {"f": (1, 2)}
# train_mix3.csv: 43,778,708 prompt and completion characters against smoke.csv's 40,576,158 (1.079x), so an epoch of
# packed rows may take up to 8% more steps than run a's
STEP_SPAN |= {"d": (1, 1.079), "d3": (1, 1.079)}
# train_fit.csv: 63,867,052 characters against smoke.csv's 40,576,158 (1.574x)
STEP_SPAN["h"] = (1, 1.574)
# h2b, h2n: 21 steps an epoch measured on every run a to h (6,891 to 10,563 rows); if an epoch scaled with the data
# below that, as few as their character share of smoke.csv's: 3,682,551 (0.091x) and 30,141,166 (0.743x)
STEP_SPAN |= {"h2b": (0.091, 1), "h2n": (0.743, 1)}
# A, B: 21 steps an epoch on every run so far (h2n's 4,829 rows included); fewer only if an epoch scales with the set:
# train_A.csv 19,253,789 prompt and completion characters (0.475x smoke.csv's), train_B.csv 24,091,860 (0.594x)
STEP_SPAN |= {"A": (0.475, 1), "B": (0.594, 1)}
STEP_NOTE = {"f": "up to twice that at the 9B's 65,536-token training context",
             "d": "up to 8% more for train_mix3.csv's extra characters",
             "d3": "up to 8% more for train_mix3.csv's extra characters",
             "h": "up to 57% more for train_fit.csv's extra characters",
             "h2b": "fewer only if an epoch scales with train_fit2_bal.csv's 0.091x characters; runs a to h all took 21",
             "h2n": "fewer only if an epoch scales with train_fit2_nat.csv's 0.743x characters; runs a to h all took 21",
             "A": "fewer only if an epoch scales with train_A.csv's 0.475x characters; every run so far took 21",
             "B": "fewer only if an epoch scales with train_B.csv's 0.594x characters; every run so far took 21"}
AUGMENT = {"augmentation_domain_rows": 0, "augmentation_general_rows": 0}
CREATE_HP_KEYS = frozenset(CreateHyperparams.__annotations__)  # the recommendation also lists dpo_* fields create lacks
# Adaption's limits for an SFT LoRA run of Qwen/Qwen3.5-4B and of Qwen/Qwen3.5-9B (the same for both), read 2026-10-20
# from docs.adaptionlabs.ai /autoscientist/supported-models (max SFT batch 8, gradient accumulation 1, lora_r up to 64;
# the 9B's SFT context is 65,536 tokens, the 4B's 131,072) and
# /autoscientist/recommended-hyperparameters ("In SFT runs, it must be 'max'"; any other integer fails after the job is
# accepted). The SDK docstring ("'max' or an integer of 1 or more") is looser than the docs; the docs win.
# max_iterations 1 to 5 (/autoscientist/running-autoscientist); target_win_rate "greater than 0.5 and at most 1" (same
# page; the SDK docstring says 0.5 to 1 inclusive, the stricter reading is kept)
LOOP_LIMITS = {"max_iterations": (1, 5), "target_win_rate": (0.5, 1.0)}
LIMITS = {"batch_size": "max", "n_epochs": (1, 5), "learning_rate": (1e-6, 5e-4), "warmup_ratio": (0.0, 1.0),
          "lora_r": (1, 64), "lora_dropout": (0.0, 1.0)}
POLL_START, POLL_MAX = 30.0, 120.0
RETRY_START, RETRY_MAX = 10.0, 300.0
TERMINAL = ("succeeded", "failed", "cancelled")


def client() -> Adaption:
    os.environ.setdefault("ADAPTION_API_KEY", Path("~/.config/adaption/api_key").expanduser().read_text().strip())
    return Adaption()


def credits(c: Adaption) -> float:
    return c.datasets.invent(rows=10, estimate=True, prompt="balance check").available_credits


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(*parts) -> None:
    print(ts(), *parts, flush=True)


def retry(fn: Callable, what: str, sleep: Callable[[float], None] = time.sleep,
          start: float = RETRY_START, cap: float = RETRY_MAX):
    """fn() until it returns; on any exception log it, sleep, and back off (start doubling to cap), without end."""
    delay, attempt = start, 0
    while True:
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - a network error of any kind must not end the wait loop
            attempt += 1
            log(f"retry {what}: attempt {attempt} failed ({type(e).__name__}: {str(e)[:200]}); sleeping {delay:g}s")
            sleep(delay)
            delay = min(delay * 2, cap)


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def csv_rows(p: Path) -> int:
    with open(p, newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def state_path(name: str | None) -> Path:
    """The state file a run reads and writes (None: upload and hparams)."""
    return STATES.get(name, STATE)


def csv_of(name: str | None) -> Path:
    """The CSV a run uploads and checks against by default: smoke.csv, or the run's RUN_CSVS entry."""
    return RUN_CSVS.get(name, CSV)


def model_of(name: str | None) -> str:
    """The base model a run trains (None: upload and hparams, the 4B)."""
    return RUN_MODELS.get(name, MODEL)


def tag_of(name: str | None) -> str:
    """The stem of a run's create record and archive directory: main_4b, or the run's TAGS entry."""
    return TAGS.get(name, "main_4b")


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save(s: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s, indent=1, default=str))


def dump(obj) -> dict:
    return json.loads(obj.model_dump_json()) if hasattr(obj, "model_dump_json") else obj


def dataset_problem(s: dict, csv_path: Path, model: str = MODEL) -> str | None:
    """Why the stored dataset may not be used with the CSV on disk for this model (None: it may)."""
    d = s.get("dataset")
    if not d:
        return None
    if d.get("model") != model:
        return f"the state file's dataset was set up for {d.get('model')}, not {model}"
    if not csv_path.exists():
        return f"no CSV at {csv_path}"
    now = md5_of(csv_path)
    if d.get("csv_md5") != now:
        return (f"the stored dataset {d.get('dataset_id')} was uploaded from a CSV with md5 {d.get('csv_md5')} "
                f"({d.get('csv_rows')} rows); {csv_path.name} on disk has md5 {now} ({csv_rows(csv_path)} rows). Refusing: "
                "move the state file away to upload the new set as a new dataset")
    return None


def row_count_problem(s: dict) -> str | None:
    """Why Adaption's processed dataset does not hold every CSV row (None: it does)."""
    d = s.get("dataset") or {}
    st = d.get("status") or {}
    got = st.get("row_count")
    if st.get("status") != "succeeded":
        return f"dataset {d.get('dataset_id')} is {st.get('status')}, not succeeded"
    if got != d.get("csv_rows"):
        return f"Adaption holds {got} rows for dataset {d.get('dataset_id')}; the CSV has {d.get('csv_rows')}"
    return None


def run_settings(s: dict, epochs: float | None = None, name: str = "a") -> dict:
    """create() keyword arguments for a run (no dataset id): the recommendation, the run's overrides, the pins, its
    loop settings. A SEARCH run sends the pins alone, so every other field stays under AutoScientist's control."""
    model = model_of(name)
    if name in SEARCH:
        if epochs is not None:
            raise SystemExit(f"run {name} leaves n_epochs to AutoScientist's search; --epochs would pin it")
        hp = dict(PINS)
    elif name in RECIPES:  # the whole recipe as it is: no recommendation under it
        hp = dict(RECIPES[name])
        hp.update(PINS)
        if epochs is not None:
            hp["n_epochs"] = epochs
    else:
        rec = s.get("recommended") or {}
        if rec.get("model") != MODEL or not rec.get("hyperparams"):
            raise SystemExit(f"run hparams first (c: `c seed`): no recommendation for {MODEL} in the state file")
        hp = {k: v for k, v in rec["hyperparams"].items() if v is not None and k in CREATE_HP_KEYS}
        hp.update(RUN_OVERRIDES[name])
        hp.update(PINS)
        if epochs is not None:
            hp["n_epochs"] = epochs
    loop = LOOPS[name]
    if (bad := limit_problems(hp, model) + loop_problems(loop)):
        raise SystemExit("refusing, outside Adaption's limits: " + "; ".join(bad))
    return {**loop, "hyperparams": hp, **AUGMENT}


def loop_problems(loop: dict) -> list[str]:
    """Where max_iterations / target_win_rate break Adaption's documented limits (empty: they keep to them)."""
    out = []
    lo, hi = LOOP_LIMITS["max_iterations"]
    m = loop["max_iterations"]
    if not (lo <= m <= hi and float(m) == int(m)):
        out.append(f"max_iterations {m}: a whole number from {lo} to {hi}")
    lo, hi = LOOP_LIMITS["target_win_rate"]
    if not lo < loop["target_win_rate"] <= hi:
        out.append(f"target_win_rate {loop['target_win_rate']}: above {lo:g} and at most {hi:g}")
    return out


def create_kwargs(s: dict, name: str, epochs: float | None = None) -> dict:
    """Everything autoscientist.create gets for a run, its own idempotency key included."""
    did = s["dataset"]["dataset_id"]
    return {"dataset_id": did, "model": model_of(name), "training_method": "instruction", "data_format": "chat",
            "column_mapping": TRAIN_MAPPING, "idempotency_key": f"main-{name}-{did}", **run_settings(s, epochs, name)}


def step_estimate(hp: dict) -> int | None:
    """Optimizer steps a run will take at batch 'max', from run a's measured steps per epoch (None: epochs left to
    AutoScientist)."""
    return round(RUN_A_STEPS_PER_EPOCH * float(hp["n_epochs"])) if "n_epochs" in hp else None


def step_range(hp: dict, name: str) -> tuple[int, int] | None:
    """The fewest and most optimizer steps a run may take: run a's measured steps an epoch, times the run's STEP_SPAN
    where its training context differs from the 4B's (None: epochs left to AutoScientist)."""
    n = step_estimate(hp)
    if n is None:
        return None
    lo, hi = STEP_SPAN.get(name, (1, 1))
    return round(n * lo), round(n * hi)


def create_run(c, s: dict, name: str, epochs: float | None = None):
    """Send create once and return (run, the kwargs sent)."""
    kw = create_kwargs(s, name, epochs)
    return c.autoscientist.create(**kw), kw


def limit_problems(hp: dict, model: str = MODEL) -> list[str]:
    """Where the hyperparams break Adaption's documented limits (empty: they keep to them)."""
    out = []
    if hp.get("batch_size", "max") != LIMITS["batch_size"]:
        out.append(f"batch_size {hp['batch_size']!r}: an SFT run takes only 'max' (8 for {model})")
    for k in ("n_epochs", "learning_rate", "warmup_ratio", "lora_dropout"):
        lo, hi = LIMITS[k]
        if k in hp and not lo <= hp[k] <= hi:
            out.append(f"{k} {hp[k]} outside [{lo:g}, {hi:g}]")
    if "n_epochs" in hp and float(hp["n_epochs"]) != int(hp["n_epochs"]):
        out.append(f"n_epochs {hp['n_epochs']} is not a whole number")
    r = hp.get("lora_r")
    if r is not None:
        lo, hi = LIMITS["lora_r"]
        if not (lo <= r <= hi and float(r) == int(r) and int(r) & (int(r) - 1) == 0):
            out.append(f"lora_r {r}: a power of 2 from {lo} to {hi}")
        elif hp.get("lora_alpha") is not None and hp["lora_alpha"] not in (r, 2 * r):
            out.append(f"lora_alpha {hp['lora_alpha']}: 1x or 2x lora_r {r}")
    return out


def over_cap(before: float, now: float, cap: float) -> bool:
    return before - now > cap


def archive_dir(name: str) -> Path:
    """Where a run's archives go: data/models/main_4b, or data/models/<tag> for a run with a TAGS entry."""
    return MODELS if name not in TAGS else MODELS.parent / TAGS[name]


def save_archive(c: Adaption, s: dict, name: str, stem: str, save_fn: Callable) -> None:
    out = archive_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    r = s["runs"][name]
    res = c.autoscientist.download(r["run_id"], timeout=3600)
    path = out / f"{name}_{stem}.tar.gz"
    res.write_to_file(path)
    r["archive"] = str(path.relative_to(ROOT))
    r["archive_bytes"] = path.stat().st_size
    r["archive_best"] = best_key(r.get("last") or {})
    r.setdefault("archives", []).append({"at": ts(), "path": r["archive"], "bytes": r["archive_bytes"],
                                         "best": r["archive_best"]})
    save_fn(s)
    log("saved", path, path.stat().st_size, "bytes")


def try_archive(c, s: dict, name: str, stem: str, save_fn: Callable) -> None:
    """A download failure is logged, not raised: it must not end the wait loop (`RUN download` fetches it later)."""
    try:
        save_archive(c, s, name, stem, save_fn)
    except Exception as e:  # noqa: BLE001
        log(f"download failed ({type(e).__name__}: {str(e)[:200]}); fetch it later with: train_main.py {name} download")


def best_key(run: dict) -> dict:
    """What identifies the best iteration so far: its win rate and hyperparameters."""
    return {"best_win_rate": run.get("best_win_rate"), "best_hyperparams": run.get("best_hyperparams")}


HP_SHOWN = ("learning_rate", "n_epochs", "warmup_ratio", "lr_scheduler_type", "lora_r", "lora_alpha", "lora_dropout",
            "min_lr_ratio", "weight_decay", "training_type", "train_on_inputs")


def hp_line(hp: dict | None) -> str:
    return " ".join(f"{k}={hp[k]}" for k in HP_SHOWN if hp.get(k) is not None) if hp else "none"


def note_iterations(r: dict, last: dict) -> None:
    """Log and record each newly completed iteration with the best at that moment (the run reports no per-iteration
    score: an unchanged best means the iteration scored at or under it)."""
    its = r.setdefault("iterations", [])
    done = int(last.get("iterations_completed") or 0)
    while len(its) < done:
        prev = its[-1]["best"] if its else None
        cur = best_key(last)
        new = cur != prev
        its.append({"iteration": len(its) + 1, "at": ts(), "best": cur, "new_best": new,
                    "reported_after": done})
        log(f"iteration {len(its)} done" + (f" (seen with {done} done)" if done > len(its) else "") + ":",
            f"best win {cur['best_win_rate']}", "NEW BEST" if new else "(best unchanged: this iteration scored at or "
            "under it)", "| best hyperparams:", hp_line(cur["best_hyperparams"]))


def wait_loop(c, s: dict, name: str, cap: float, *, save_fn: Callable, sleep: Callable[[float], None] = time.sleep,
              archive: Callable = try_archive):
    """Poll the run until it is terminal, cancelling it once it has spent more than cap. Returns the last run."""
    r = s["runs"][name]
    rid = r["run_id"]
    delay = POLL_START
    while True:
        run = retry(lambda: c.autoscientist.get(rid), "run get", sleep)
        now = retry(lambda: credits(c), "credits read", sleep)
        log(run.status, "iter", f"{run.iterations_completed:g}/{run.max_iterations:g}", "win", run.best_win_rate,
            "dl", run.download_available, "credits", now, f"spent {r['credits_before'] - now:g} of {cap:g}")
        r["last"] = dump(run)
        r.setdefault("credit_trace", []).append([ts(), now, run.status])
        note_iterations(r, r["last"])
        save_fn(s)
        if run.download_available and run.status == "running" and r.get("archive_best") != best_key(r["last"]):
            # keep each new best checkpoint as it appears, in case the cap cancels a later iteration
            archive(c, s, name, f"{rid}_iter{int(run.iterations_completed)}", save_fn)
        if run.status in TERMINAL:
            break
        if over_cap(r["credits_before"], now, cap) and not r.get("cancelled_for_budget"):
            log(f"spent {r['credits_before'] - now:g} > {cap:g}; cancelling")
            log("CANCEL", retry(lambda: c.autoscientist.cancel(rid), "cancel", sleep))
            r["cancelled_for_budget"] = True
            save_fn(s)
        sleep(delay)
        delay = min(delay * 1.5, POLL_MAX)
    r["credits_after"] = retry(lambda: credits(c), "credits read", sleep)
    start = datetime.fromisoformat(str(run.created_at))
    end = datetime.fromisoformat(str(run.completed_at)) if run.completed_at else datetime.now(timezone.utc)
    r["run_minutes"] = round((end - start).total_seconds() / 60, 1)
    r["credits_spent"] = round(r["credits_before"] - r["credits_after"], 2)
    save_fn(s)
    if (run.status in ("succeeded", "cancelled") and run.download_available
            and r.get("archive_best") != best_key(r["last"])):
        # the archive is the best iteration's; fetch it unless that best is already on disk
        archive(c, s, name, rid, save_fn)
    return run


def seed(name: str) -> Path:
    """Start a run's own state file from STATE's dataset and recommendation. STATE is only read. A run in SEED_FROM
    (f) instead starts from that run's recorded create: its hyperparams must equal the run's RECIPES entry; no dataset
    is copied (f uploads its own)."""
    dest = state_path(name)
    if dest == STATE:
        raise SystemExit(f"run {name} keeps its state in {STATE.name}; nothing to seed")
    if name in SEED_FROM:
        return seed_recipe(name, dest)
    src = json.loads(STATE.read_text())
    if not src.get("dataset") or not src.get("recommended"):
        raise SystemExit(f"{STATE.name} has no dataset and recommendation to seed from")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "x") as f:  # never over an existing file: it may hold a created run
        f.write(json.dumps({"dataset": src["dataset"], "recommended": src["recommended"],
                            "seeded_from": STATE.name, "seeded_at": ts()}, indent=1, default=str))
    return dest


def seed_recipe(name: str, dest: Path) -> Path:
    """Start a run's state file from run SEED_FROM[run]'s recorded create (read only), refusing unless the hyperparams
    it sent, with the run's SEED_DIFF over them, are RECIPES[run]."""
    frm = SEED_FROM[name]
    src_path = state_path(frm)
    r = json.loads(src_path.read_text()).get("runs", {}).get(frm, {})
    sent = (r.get("settings") or {}).get("hyperparams")
    if not r.get("run_id") or sent is None:
        raise SystemExit(f"no created run {frm} with sent settings in {src_path.name}")
    want = sent | SEED_DIFF.get(name, {})
    if want != RECIPES[name]:
        raise SystemExit(f"run {frm}'s sent hyperparams (with {SEED_DIFF.get(name, {})}) differ from RECIPES[{name!r}]: "
                         f"{ {k: (want.get(k), RECIPES[name].get(k)) for k in set(want) | set(RECIPES[name]) if want.get(k) != RECIPES[name].get(k)} }")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "x") as f:  # never over an existing file: it may hold a dataset or a created run
        f.write(json.dumps({"recipe_from": {"run": frm, "run_id": r["run_id"], "hyperparams": sent,
                                            "model": (r.get("last") or {}).get("model"),
                                            "best_win_rate": (r.get("last") or {}).get("best_win_rate")},
                            "seed_diff": SEED_DIFF.get(name, {}), "recipe": RECIPES[name],
                            "model": model_of(name), "seeded_from": src_path.name, "seeded_at": ts()},
                           indent=1, default=str))
    return dest


def status_lines(s: dict, runs: dict, c) -> list[str]:
    """s gives the dataset; runs maps each run name to its record, read from that run's own state file."""
    d = s.get("dataset")
    out = []
    if d:
        st = retry(lambda: c.datasets.get_status(d["dataset_id"]), "dataset status")
        out.append(f"dataset {d['dataset_id']} {st.status}: Adaption rows {st.row_count}, CSV rows {d.get('csv_rows')}, "
                   f"md5 {d.get('csv_md5')}")
    else:
        out.append("no dataset uploaded")
    for n, r in runs.items():
        if not r.get("run_id"):
            out.append(f"run {n}: not created")
            continue
        run = retry(lambda: c.autoscientist.get(r["run_id"]), "run get")
        out.append(f"run {n} {r['run_id']}: {run.status}, iterations {run.iterations_completed:g}/{run.max_iterations:g}, "
                   f"best win rate {run.best_win_rate}, download_available {run.download_available}, "
                   f"cap {r.get('max_credits')}, credits before {r.get('credits_before')}"
                   + (f", error {run.error}" if run.error else ""))
        if run.max_iterations > 1:
            out.append(f"  best hyperparams: {hp_line(dump(run.best_hyperparams) if run.best_hyperparams else None)}")
    out.append(f"credits now {retry(lambda: credits(c), 'credits read')}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="+", help="upload | hparams | config [RUN] | status [RUN] | RUN seed | "
                    "RUN create|wait|download | f upload | f hparams")
    ap.add_argument("--max-credits", type=float, help="the run's credit cap (create needs it; wait uses the stored one)")
    ap.add_argument("--epochs", type=float, help="n_epochs over the recommendation and the run's overrides")
    ap.add_argument("--csv", type=Path, help="the training CSV (default: smoke.csv; train_mix3.csv for d and d3; "
                    "train_fit.csv for h; train_fit2_bal.csv for h2b; train_fit2_nat.csv for h2n)")
    a = ap.parse_args()
    w = a.words
    if "status" in w:  # status, status a, a status
        given = [x for x in w if x in RUNS]
        recs = {n: load(state_path(n)).get("runs", {}).get(n, {}) for n in given or RUNS}
        runs = {n: r for n, r in recs.items() if given or r}
        s = load(state_path(given[0]) if given else STATE)
        if (why := dataset_problem(s, a.csv or csv_of(given[0] if given else None),
                                   model_of(given[0] if given else None))):
            print("warning:", why)
        print("\n".join(status_lines(s, runs, client())))
        return
    if len(w) == 2 and w[1] == "seed" and w[0] in STATES:
        dest = seed(w[0])
        print(f"seeded {dest} from {state_path(SEED_FROM[w[0]]) if w[0] in SEED_FROM else STATE} (read only)")
        return
    if len(w) == 2 and w[0] in RUNS and w[1] in ("upload", "hparams"):  # f upload, f hparams: its own dataset
        if w[0] not in OWN_DATASET:
            sys.exit(f"run {w[0]} trains on {STATE.name}'s dataset: use `train_main.py {w[1]}`")
        name, op = w
    else:
        op = w[0]
        name = (w[1] if len(w) > 1 else "a") if op == "config" else op if op in RUNS else None
    model = model_of(name)
    path = state_path(name)
    a.csv = a.csv or csv_of(name)
    if path != STATE and not path.exists():
        sys.exit(f"no {path.name} yet: run `train_main.py {name} seed` first")
    s = load(path)
    if op == "config":
        if name not in RUNS:
            sys.exit(__doc__)
        kw = create_kwargs(s, name, a.epochs) if s.get("dataset") else run_settings(s, a.epochs, name)
        print(json.dumps(kw, indent=1))
        steps = step_range(kw["hyperparams"], name)
        print(f"model: {kw.get('model', model)}")
        print((f"steps: about {steps[0]}" if steps[0] == steps[1] else f"steps: {steps[0]} to {steps[1]}") +
              f" ({RUN_A_STEPS_PER_EPOCH} an epoch at batch 'max', measured on run a" +
              (f", {STEP_NOTE[name]})" if steps[0] != steps[1] else ")")
              if steps else
              f"steps: set by AutoScientist each iteration; up to {kw['max_iterations']:g} iterations, stopping early at "
              f"win rate {kw['target_win_rate']:g}")
        rec = (s.get("recommended") or {}).get("hyperparams") or {}
        print(f"csv: {a.csv}" + (f" (upload needs md5 {CSV_MD5[name]})" if name in CSV_MD5 else ""))
        if name in RECIPES and rec:  # informational: a RECIPES run sends its recipe, never the recommendation
            diff = {k: [rec.get(k), v] for k, v in kw["hyperparams"].items() if rec.get(k) is not None and rec.get(k) != v}
            print(f"{model} recommendation (not sent), where it differs from what {name} sends [recommended, sent]:",
                  json.dumps(diff))
        return
    if op == "upload":
        if s.get("dataset"):
            if (why := dataset_problem(s, a.csv, model)):
                sys.exit(why)
            print("dataset already uploaded from this CSV:", s["dataset"]["dataset_id"])
        c = client()
        if not s.get("dataset"):
            md5 = md5_of(a.csv)
            if name in CSV_MD5 and CSV_MD5[name] is None:
                sys.exit(f"refusing: run {name} has no reviewed CSV md5 in CSV_MD5 yet")
            if name in CSV_MD5 and md5 != CSV_MD5[name]:
                sys.exit(f"refusing: {a.csv} has md5 {md5}; run {name} trains on the reviewed CSV, md5 {CSV_MD5[name]}")
            before = credits(c)
            log("credits before upload", before)
            up = c.datasets.upload_file(str(a.csv), name=f"main-train-{md5[:8]}{UPLOAD_SUFFIX.get(name, '')}",
                                        processing_mode="raw", column_mapping=MAPPING)
            print("UPLOAD", up)
            s["dataset"] = {"dataset_id": up.dataset_id, "csv": str(a.csv), "csv_md5": md5,
                            "csv_rows": csv_rows(a.csv), "model": model, "uploaded_at": ts(),
                            "credits_before_upload": before}
            save(s, path)
        st = c.datasets.wait_for_completion(s["dataset"]["dataset_id"], timeout=3600)
        print("STATUS", st)
        s["dataset"]["status"] = dump(st)
        if "credits_before_upload" in s["dataset"] and "credits_after_upload" not in s["dataset"]:
            s["dataset"]["credits_after_upload"] = credits(c)
            log("credits after upload", s["dataset"]["credits_after_upload"], "(the balance is the account's: other "
                "jobs' spend shows here too)")
        save(s, path)
        if (why := row_count_problem(s)):
            sys.exit(why)
        print("rows match:", s["dataset"]["csv_rows"])
        return
    if not s.get("dataset"):
        sys.exit("upload first")
    if (why := dataset_problem(s, a.csv, model)):
        if op == "hparams" or w[-1] == "create":
            sys.exit(why)
        print("warning:", why, flush=True)
    did = s["dataset"]["dataset_id"]
    if op == "hparams":
        if (why := row_count_problem(s)):
            sys.exit(why)
        r = client().autoscientist.recommend_hyperparams(dataset_id=did, model=model, **AUGMENT)
        print("RECOMMEND", r)
        s["recommended"] = dump(r)
        save(s, path)
        return
    if len(w) != 2 or w[0] not in RUNS or w[1] not in ("create", "wait", "download"):
        sys.exit(__doc__)
    name, cmd = w
    runs = s.setdefault("runs", {})
    r = runs.setdefault(name, {})
    c = client()
    if cmd == "create":
        if r.get("run_id"):
            sys.exit(f"run {name} already created: {r['run_id']}")
        if a.max_credits is None or a.max_credits <= 0:
            sys.exit("create needs --max-credits (the run is cancelled once it spends more)")
        if (why := row_count_problem(s)):
            sys.exit(why)
        create_kwargs(s, name, a.epochs)  # refuse bad settings before the balance read
        before = credits(c)
        log("credits before", before)
        if before < a.max_credits:
            sys.exit(f"refusing: {before} credits left, under the {a.max_credits} cap")
        run, kw = create_run(c, s, name, a.epochs)
        settings = {k: v for k, v in kw.items() if k not in ("dataset_id", "model", "training_method", "data_format",
                                                             "column_mapping", "idempotency_key")}
        r.update(run_id=run.id, credits_before=before, max_credits=a.max_credits, settings=settings,
                 idempotency_key=kw["idempotency_key"], created=dump(run), created_at=ts())
        save(s, path)
        with open(path.with_name(f"{tag_of(name)}_run_{name}.txt"), "a") as f:
            f.write(f"{ts()} run {name} {run.id} {kw['model']} created; credits before {before}, cap {a.max_credits}\n")
        log("CREATE", run)
        time.sleep(20)
        r["credits_after_create"] = retry(lambda: credits(c), "credits read")
        log("credits after create", r["credits_after_create"], "| now run:", f"train_main.py {name} wait")
        save(s, path)
    elif cmd == "wait":
        if not r.get("run_id"):
            sys.exit(f"no run {name} yet")
        cap = a.max_credits if a.max_credits is not None else r["max_credits"]
        run = wait_loop(c, s, name, cap, save_fn=lambda s: save(s, path))
        log("FINAL", run)
        log(f"run {r['run_minutes']} min, credits {r['credits_before']} -> {r['credits_after']} ({r['credits_spent']})")
    elif cmd == "download":
        save_archive(c, s, name, r["run_id"], lambda s: save(s, path))


if __name__ == "__main__":
    main()
