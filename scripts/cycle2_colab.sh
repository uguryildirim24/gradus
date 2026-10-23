#!/usr/bin/env bash
# Cycle 2 (eval/protocol_amendment_2.json) inference on Colab L4: one model per session, sequential, bf16 batch 1,
# under the watchdog (colab_eval.py starts it through colab_guard.sh). Run under `caffeinate -is`.
#   cycle2_colab.sh select          h2n on fresh test 1 (h, d3, base are scored already in v4_eval/fresh_test)
#   cycle2_colab.sh fresh2 PICK     PICK, h, d3 and base on fresh test 2, only once eval/selection_cycle2.json is
#                                   committed and names PICK
# h2n trains on train_fit.csv prompts unchanged (fit_eval_inputs.h2_prompts_are_train_fit_rows), so it reads h's build.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
IN=data/built/fit/eval_inputs
declare -A ARCH=(
  [h]=data/models/main_4b/h_63659c74-57fd-4353-8a03-07bc11dc0fb6.tar.gz
  [d3]=data/models/main_4b/d3_5be8d991-d02e-4293-bded-3d1374c55e7f.tar.gz
  [h2n]=data/models/main_4b/h2n_0a978d55-5a40-4de4-86c9-e2558c5a3607.tar.gz
)
declare -A FMT=([h]=h [h2n]=h [d3]=d3 [base]=base)
run1() { # out_dir set tag
  O=$1; set_=$2; tag=$3
  mkdir -p $O
  test -f $O/${tag}_run/${tag}_${set_}.json && { echo "$tag already scored on $set_"; return 0; }
  if [ "$tag" = base ]; then m=(--base); else m=(--model "$tag=${ARCH[$tag]}"); fi
  $PY scripts/colab_eval.py matcher-$tag-${set_//_/-} --base-model Qwen/Qwen3.5-4B "${m[@]}" \
    --dev $IN/${FMT[$tag]}/$set_.jsonl --out $O/${tag}_run --dtype bfloat16 --batch 1 --max-min 70 \
    > $O/eval_${tag}.log 2>&1
  echo "$tag exit $?"
}
case "${1:-}" in
  select)
    run1 data/models/v4_eval/fresh_test fresh_test h2n ;;
  fresh2)
    pick=${2:?pick}
    S=eval/selection_cycle2.json
    git ls-files --error-unmatch $S >/dev/null 2>&1 && git diff --quiet HEAD -- $S \
      || { echo "$S is not committed unchanged"; exit 1; }
    [ "$($PY -c "import json;print(json.load(open('$S'))['pick'])")" = "$pick" ] || { echo "$S does not pick $pick"; exit 1; }
    for t in $(printf '%s\n' $pick h d3 base | awk '!seen[$0]++'); do
      run1 data/models/v4_eval/fresh_test2 fresh_test2 $t
    done ;;
  *) echo "usage: $0 select | fresh2 PICK"; exit 2 ;;
esac
echo ALL_SESSIONS_DONE
