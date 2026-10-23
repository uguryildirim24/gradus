#!/usr/bin/env bash
# Fresh test 3 inference on Colab L4 (eval/protocol_amendment_3.json, 3b): one model per session, sequential,
# bf16 batch 1, under the watchdog (colab_eval.py starts it through colab_guard.sh). Run under `caffeinate -is`.
#   test3_colab.sh [TAG ...]   score each of h, d3, base (default all three, pick first) on fresh test 3, only once
#                              eval/selection_test3.json is committed unchanged; a model with a result file is skipped,
#                              never rescored. A session that drops before its job ran leaves no result: run again.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
IN=data/built/fit/eval_inputs
O=data/models/v4_eval/fresh_test3
S=eval/selection_test3.json
declare -A ARCH=(
  [h]=data/models/main_4b/h_63659c74-57fd-4353-8a03-07bc11dc0fb6.tar.gz
  [d3]=data/models/main_4b/d3_5be8d991-d02e-4293-bded-3d1374c55e7f.tar.gz
)
declare -A FMT=([h]=h [d3]=d3 [base]=base)
git ls-files --error-unmatch $S >/dev/null 2>&1 && git diff --quiet HEAD -- $S \
  || { echo "$S is not committed unchanged"; exit 1; }
pick=$($PY -c "import json;print(json.load(open('$S'))['pick'])")
tags=${*:-$pick h d3 base}
mkdir -p $O
for t in $(printf '%s\n' $tags | awk '!seen[$0]++'); do
  test -f $O/${t}_run/${t}_fresh_test3.json && { echo "$t already scored on fresh_test3"; continue; }
  if [ "$t" = base ]; then m=(--base); else m=(--model "$t=${ARCH[$t]}"); fi
  if [ -e $O/eval_${t}.log ]; then n=1; while [ -e $O/eval_${t}_try${n}.log ]; do n=$((n+1)); done
    mv $O/eval_${t}.log $O/eval_${t}_try${n}.log; fi
  $PY scripts/colab_eval.py matcher-$t-fresh-test3 --base-model Qwen/Qwen3.5-4B "${m[@]}" \
    --dev $IN/${FMT[$t]}/fresh_test3.jsonl --out $O/${t}_run --dtype bfloat16 --batch 1 --max-min 70 \
    > $O/eval_${t}.log 2>&1
  echo "$t exit $?"
done
echo ALL_SESSIONS_DONE
