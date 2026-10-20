#!/usr/bin/env bash
# Train a set of agent variants one after another and let them compete in the arena.
# Usage: scripts/run_experiments.sh [iterations] [python]
set -euo pipefail
cd "$(dirname "$0")/.."
ITERS="${1:-150}"
PY="${2:-.venv/bin/python}"
COMMON=(--games 2048 --rounds 32 --iters "$ITERS" --eval-every 10)

run() { local name="$1"; shift; echo "=== $name ==="; "$PY" -m nochmal.train "${COMMON[@]}" --out "checkpoints/$name" "$@"; }

[ -f checkpoints/duo/best.pt ]        || run duo        --players 2
[ -f checkpoints/duo-blind/best.pt ]  || run duo-blind  --players 2 --opponent-view none
[ -f checkpoints/a2c/best.pt ]        || run a2c        --players 2 --algo a2c
[ -f checkpoints/vpg/best.pt ]        || run vpg        --players 2 --algo vpg
[ -f checkpoints/ppo-small/best.pt ]  || run ppo-small  --players 2 --channels 48 --blocks 3
[ -f checkpoints/trio/best.pt ]       || run trio       --players 3
[ -f checkpoints/quad/best.pt ]       || run quad       --players 4 --games 1536

mkdir -p checkpoints/arena
"$PY" -m nochmal.arena --players 1 2 3 4 --games 240 \
  --agents heuristic random duo/best.pt duo-blind/best.pt a2c/best.pt vpg/best.pt ppo-small/best.pt trio/best.pt quad/best.pt \
  --out "checkpoints/arena/$(date +%Y%m%d-%H%M%S).json"
