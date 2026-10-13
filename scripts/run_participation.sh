#!/usr/bin/env bash
# Swept runs through scripts/run_matrix.py; all settings come from the arguments.
#
#   bash scripts/run_participation.sh --grid participation=<p1,p2,...> -- \
#       --dataset <name> --num-tasks <T> --clients <N> --alpha <a,...> \
#       --rounds <R> --local-epochs <I> --seeds <s,...> \
#       --device <dev> --data-root <dir> --out-dir <dir>
set -euo pipefail

if [ "$#" -eq 0 ]; then
    sed -n '2,8p' "$0" >&2
    exit 2
fi

exec python scripts/run_matrix.py "$@"
