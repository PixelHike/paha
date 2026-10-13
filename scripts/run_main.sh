#!/usr/bin/env bash
# Single run: every setting is taken from the arguments, nothing is defaulted.
#
#   bash scripts/run_main.sh -- --dataset <name> --num-tasks <T> --clients <N> \
#       --alpha <a> --participation <p> --rounds <R> --local-epochs <I> \
#       --seeds <s,...> --device <dev> --data-root <dir> --out-dir <dir>
set -euo pipefail

if [ "$#" -eq 0 ]; then
    sed -n '2,8p' "$0" >&2
    exit 2
fi

exec python -m paha_fc.cli "$@"
