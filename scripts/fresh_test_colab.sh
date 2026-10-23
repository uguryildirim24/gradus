#!/bin/bash
# One adapter per Colab L4 session, sequential, under the watchdog (colab_eval.py starts it through colab_guard.sh).
cd "$(dirname "$0")/.."
O=data/models/v4_eval/fresh_test
mkdir -p $O
PY=.venv/bin/python
run1() { # tag session extra...
  tag=$1; s=$2; shift 2
  test -f $O/${tag}_run/${tag}_fresh_test.json && { echo "$tag already scored"; return 0; }
  $PY scripts/colab_eval.py $s --base-model Qwen/Qwen3.5-4B "$@" \
    --dev data/built/fit/eval_inputs/$tag/fresh_test.jsonl --out $O/${tag}_run --dtype bfloat16 --batch 1 --max-min 70 \
    > $O/eval_${tag}.log 2>&1
  echo "$tag exit $?"
}
run1 h matcher-h-fresh --model h=data/models/main_4b/h_63659c74-57fd-4353-8a03-07bc11dc0fb6.tar.gz
run1 d3 matcher-d3-fresh --model d3=data/models/main_4b/d3_5be8d991-d02e-4293-bded-3d1374c55e7f.tar.gz
run1 base matcher-base-fresh --base
echo ALL_SESSIONS_DONE
