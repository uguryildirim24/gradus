#!/usr/bin/env bash
# launch run h once: run d3's recipe on the v4 fit set (data/built/fit/train_fit.csv, 10,563 rows, md5 b1fdfa85...);
# besides the data nothing changes. Qwen/Qwen3.5-4B, SFT (training_method instruction, data_format chat), lora r16
# alpha 32 dropout 0 all-linear, lr 2e-4, 3 epochs, warmup 0.05, cosine (min_lr_ratio 0.1, 0.5 cycles), max_grad_norm
# 2.0, weight_decay 0, batch "max", train_on_inputs false, augmentation 0, 1 iteration, target 0.8. Column mapping
# prompt=original_prompt, completion=original_completion: no reasoning trace (the raw upload drops that column).
# 63 to 99 steps (21 an epoch on run a, up to 57% more for train_fit.csv's extra characters). Lives at ~/matcher-train/
# on oci-pi.
# h trains on a dataset of its own (one training run per dataset at a time). Once, before this:
#   train_main.py h seed     (h's state file from d3's sent recipe, main_4b_d3.json read only)
#   train_main.py h upload   (train_fit.csv, md5 b1fdfa85..., as dataset main-train-b1fdfa85-h, row count checked)
# run h keeps its state in data/adaption/main_4b_h.json, never main_4b.json or another run's file.
# the cap: CAP credits (default 80). create refuses when the balance is under the cap, and the balance was 91 on
# 2026-10-20, so a 100 cap cannot launch; 80 keeps the guard. Training has charged 0 credits on every run so far.
# the cap reads the whole account balance, so any other job's spend counts against it.
# the key comes from ~/.config, never argv. train_main refuses if run h already has a run id, and create sends
# idempotency key main-h-<h's dataset>, so a second launch cannot create a second run.
# after create: start the wait at once (run_wait_h.sh in tmux) and append the printed main_4b_run_h.txt line to the
# session's scratchpad main_4b_run.txt.
set -euo pipefail
cd ~/matcher-train
CAP="${CAP:-80}"
S=data/adaption/main_4b_h.json
test -f $S || { echo "no $S: run train_main.py h seed, then h upload, first" >&2; exit 1; }
export ADAPTION_API_KEY="$(cat ~/.config/adaption/api_key)"
L=data/adaption/main_4b_h_create.log
PYTHONPATH=src .venv/bin/python -u scripts/train_main.py config h 2>&1 | tee -a $L
PYTHONPATH=src .venv/bin/python -u scripts/train_main.py status h 2>&1 | tee -a $L
PYTHONPATH=src .venv/bin/python -u scripts/train_main.py h create --max-credits "$CAP" 2>&1 | tee -a $L
cat data/adaption/main_4b_run_h.txt
