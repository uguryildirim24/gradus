#!/bin/bash
# Start a Colab A100/H100 session for training under an external watchdog (amendment 6b: the author asked for the
# fastest GPU for the C / B-colab training runs). Same as the external colab_guard.sh that colab_eval.py uses, except
# the GPU it allows; the watchdog is an external colab_watchdog.sh (COLAB_WATCHDOG), unchanged (stop on the hard
# deadline, a stale /content/heartbeat, or three failed checks). Log: $HOME/.colab-guard/<session-name>.log
#
# usage: colab_guard_fast.sh <session-name> <gpu: H100|A100> <max-minutes> <idle-minutes>
set -u
NAME="$1"; GPU="$2"; MAX_MIN="$3"; IDLE_MIN="$4"
WATCHDOG="${COLAB_WATCHDOG:-colab_watchdog.sh}"
COLAB="$HOME/.local/bin/colab"
LOGDIR="$HOME/.colab-guard"; mkdir -p "$LOGDIR"; LOG="$LOGDIR/$NAME.log"

case "$GPU" in H100|A100) ;; *) echo "this guard is for the training runs only: H100 or A100"; exit 2;; esac
command -v "$WATCHDOG" >/dev/null || { echo "no watchdog at $WATCHDOG"; exit 2; }

START=$(date +%s)
echo "$(date -u +%FT%TZ) new $NAME gpu=$GPU max=${MAX_MIN}m idle=${IDLE_MIN}m" >>"$LOG"
if ! timeout 900 "$COLAB" new -s "$NAME" --gpu "$GPU" >>"$LOG" 2>&1; then
  echo "colab new failed on $GPU; see $LOG"; timeout 120 "$COLAB" stop -s "$NAME" >>"$LOG" 2>&1; exit 1
fi
echo 'import pathlib; pathlib.Path("/content/heartbeat").touch()' | timeout 300 "$COLAB" exec -s "$NAME" >>"$LOG" 2>&1

nohup "$WATCHDOG" "$NAME" "$MAX_MIN" "$IDLE_MIN" "$START" >/dev/null 2>&1 &
echo "session $NAME up on $GPU; watchdog pid $! (log $LOG)"
