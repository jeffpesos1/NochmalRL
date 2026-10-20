#!/usr/bin/env bash
# Train inside the container.  Usage: scripts/train.sh [run-name] [extra args for nochmal.train]
set -euo pipefail
RUN="${1:-duo}"; shift || true
docker compose run --rm train --board classic --players 2 --games 2048 --iters 2000 --out "/app/checkpoints/${RUN}" "$@"
