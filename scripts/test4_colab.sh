#!/usr/bin/env bash
# Fresh test 4 inference on Colab L4 (eval/protocol_amendment_5.json fresh_test_4, 5b): one model per session,
# sequential, bf16 batch 1, under the watchdog (colab_eval.py starts it through colab_guard.sh). Run under `caffeinate -is`.
#   test4_colab.sh [TAG ...]   score each of B (the pick's model), h, base (default all three, in that order) on fresh
#                              test 4, only once eval/selection_cycle3.json and eval/fresh_test4_eval_inputs.json are
#                              committed unchanged and the prompt file and adapter archive are the pinned ones; a model
#                              with a result file is skipped, never rescored. B on B's prompts (rubric at the top), h and
#                              base on h's. A session that drops before its job ran leaves no result: run again.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
IN=data/built/fit/eval_inputs
O=data/models/c3_eval/test4
for f in eval/selection_cycle3.json eval/fresh_test4_eval_inputs.json eval/fit_eval_hashes.json; do
  git ls-files --error-unmatch $f >/dev/null 2>&1 && git diff --quiet HEAD -- $f \
    || { echo "$f is not committed unchanged"; exit 1; }
done
pick=$($PY -c "import json;print(json.load(open('eval/selection_cycle3.json'))['pick']['model'])")
[ "$pick" = B ] || { echo "the committed pick's model is $pick, this script scores B"; exit 1; }
declare -A ARCH=(
  [B]=data/models/main_4b_c3/B_8dc3d64a-cd0f-4807-a6e9-4eadb542ed3d.tar.gz
  [h]=data/models/main_4b/h_63659c74-57fd-4353-8a03-07bc11dc0fb6.tar.gz
)
declare -A FMT=([B]=B [h]=h [base]=base)
declare -A SESSION=([B]=matcher-c3b-test4 [h]=matcher-h-test4 [base]=matcher-base-test4)
tags=${*:-B h base}
mkdir -p $O
for t in $(printf '%s\n' $tags | awk '!seen[$0]++'); do
  case $t in B|h|base) ;; *) echo "unknown tag $t"; exit 1;; esac
  test -f $O/$t/${t}_fresh_test4.json && { echo "$t already scored on fresh_test4"; continue; }
  dev=$IN/${FMT[$t]}/fresh_test4.jsonl
  $PY - "$t" "$dev" "${ARCH[$t]:-}" <<'EOF' || exit 1
import hashlib, json, sys
t, dev, arch = sys.argv[1:4]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
pins = json.load(open("eval/fresh_test4_eval_inputs.json"))[t]
if pins.get(dev) != sha(dev):
    sys.exit(f"{dev} is not the prompt file eval/fresh_test4_eval_inputs.json pins for {t}")
if t == "B":
    rec = json.load(open("data/models/c3_eval/B/archive.json"))
    want = rec["sha256"] if rec["local"] == arch else None
elif t == "h":
    rec = json.load(open("eval/fit_eval_hashes.json"))["model_archives"]["h"]
    want = rec["sha256"] if arch.endswith(rec["file"]) else None
else:
    want = None
if arch and sha(arch) != want:
    sys.exit(f"{arch} is not {t}'s pinned archive")
print(f"{t}: prompt file and archive match their pins")
EOF
  if [ "$t" = base ]; then m=(--base); else m=(--model "$t=${ARCH[$t]}"); fi
  mkdir -p $O/$t
  if [ -e $O/eval_${t}.log ]; then n=1; while [ -e $O/eval_${t}_try${n}.log ]; do n=$((n+1)); done
    mv $O/eval_${t}.log $O/eval_${t}_try${n}.log; fi
  echo "$t session ${SESSION[$t]} start $(date -u +%FT%TZ)"
  $PY scripts/colab_eval.py ${SESSION[$t]} --base-model Qwen/Qwen3.5-4B "${m[@]}" \
    --dev $dev --out $O/$t --dtype bfloat16 --batch 1 --max-min 70 \
    > $O/eval_${t}.log 2>&1
  echo "$t exit $? end $(date -u +%FT%TZ)"
done
echo ALL_SESSIONS_DONE
