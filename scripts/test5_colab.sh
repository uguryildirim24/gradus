#!/usr/bin/env bash
# Amendment 6b inference on Colab L4: one model per session, bf16 batch 1, under the watchdog (colab_eval.py starts
# it through colab_guard.sh). Run under `caffeinate -is`.
#   test5_colab.sh TAG [ARCHIVE]   B: run B (its pinned archive) on fresh test 5 (B's format)
#                                  Bcolab / C: the Colab-trained adapter ARCHIVE (data/models/c3_colab/TAG/archive.json
#                                  must name it with its sha256) on fresh tests 1-5, Bcolab on B's prompt files, C on C's
# Every prompt file is checked against eval/fresh_test5_eval_inputs.json (committed unchanged) first; a test whose
# score file exists is skipped, never rescored.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
IN=data/built/fit/eval_inputs
O=data/models/c3_eval/test5
t=${1:?TAG}
for f in eval/fresh_test5_eval_inputs.json eval/protocol_amendment_6b.json; do
  git ls-files --error-unmatch $f >/dev/null 2>&1 && git diff --quiet HEAD -- $f \
    || { echo "$f is not committed unchanged"; exit 1; }
done
case $t in
  B) arch=data/models/main_4b_c3/B_8dc3d64a-cd0f-4807-a6e9-4eadb542ed3d.tar.gz; fmt=B; sets="fresh_test5"; mm=70;;
  Bcolab) arch=${2:?ARCHIVE}; fmt=B; sets="fresh_test fresh_test2 fresh_test3 fresh_test4 fresh_test5"; mm=110;;
  C) arch=${2:?ARCHIVE}; fmt=C; sets="fresh_test fresh_test2 fresh_test3 fresh_test4 fresh_test5"; mm=110;;
  *) echo "unknown tag $t"; exit 1;;
esac
devs=()
for s in $sets; do
  test -f $O/$t/${t}_${s}.json && { echo "$t already scored on $s"; continue; }
  devs+=(--dev $IN/$fmt/$s.jsonl)
done
[ ${#devs[@]} -gt 0 ] || { echo "nothing to score"; exit 0; }
$PY - "$t" "$arch" "${devs[@]}" <<'PYEOF' || exit 1
import hashlib, json, sys
t, arch, *devs = sys.argv[1:]
devs = devs[1::2]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
pins = json.load(open("eval/fresh_test5_eval_inputs.json"))[t]
for d in devs:
    if pins.get(d) != sha(d):
        sys.exit(f"{d} is not the prompt file eval/fresh_test5_eval_inputs.json pins for {t}")
if t == "B":
    rec = json.load(open("data/models/c3_eval/B/archive.json"))
    ok = rec["local"] == arch and rec["sha256"] == sha(arch)
else:
    rec = json.load(open(f"data/models/c3_colab/{t}/archive.json"))
    ok = rec["archive"] == arch and rec["archive_sha256"] == sha(arch)
if not ok:
    sys.exit(f"{arch} is not {t}'s recorded archive")
print(f"{t}: {len(devs)} prompt files and the archive match their pins")
PYEOF
mkdir -p $O/$t
if [ -e $O/eval_${t}.log ]; then n=1; while [ -e $O/eval_${t}_try${n}.log ]; do n=$((n+1)); done
  mv $O/eval_${t}.log $O/eval_${t}_try${n}.log; fi
echo "$t start $(date -u +%FT%TZ)"
$PY scripts/colab_eval.py matcher-$(echo $t | tr A-Z a-z)-t5 --base-model Qwen/Qwen3.5-4B --model "$t=$arch" \
  "${devs[@]}" --out $O/$t --dtype bfloat16 --batch 1 --max-min $mm > $O/eval_${t}.log 2>&1
echo "$t exit $? end $(date -u +%FT%TZ)"
