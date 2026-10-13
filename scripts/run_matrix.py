#!/usr/bin/env python
"""Grid driver over the command-line interface.

Nothing is defaulted here: every setting comes from the arguments you pass. The
base CLI arguments are forwarded verbatim, and each ``--grid`` adds one swept
setting:

    python scripts/run_matrix.py \
        --grid lr=<lr1,lr2> \
        --grid batch_size=<b1,b2> \
        -- --dataset <dataset> --num-tasks <T> --clients <N> --alpha <alpha> \
           --participation <p> --rounds <R> --local-epochs <I> \
           --seeds <s,...> --device <device> --data-root <dir> --out-dir <dir>

Sweeps are expanded as a cross product, one CLI call per combination. Note that
``--num-tasks``, ``--alpha``, ``--participation`` and ``--seeds`` already accept
comma-separated lists inside the CLI itself, so they normally do not need
``--grid``.

Every swept key must be a field of ``paha_fc.Config``; underscores are mapped to
the dashed option names automatically.
"""

from __future__ import annotations

import itertools
import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paha_fc.cli import main as cli_main  # noqa: E402
from paha_fc.config import Config  # noqa: E402

_CONFIG_KEYS = set(Config.__dataclass_fields__)


class GridError(ValueError):
    """Raised for a malformed or unknown ``--grid`` specification."""


def parse_grid(spec: str) -> Tuple[str, List[str]]:
    """Split ``key=v1,v2`` into a CLI option name and its values."""
    if "=" not in spec:
        raise GridError(f"--grid expects key=value[,value...], got {spec!r}")
    key, _, raw = spec.partition("=")
    key = key.strip().replace("-", "_")
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not key or not values:
        raise GridError(f"--grid needs a non-empty key and at least one value: {spec!r}")
    if key not in _CONFIG_KEYS:
        raise GridError(f"unknown configuration key {key!r} in --grid {spec!r}")
    return key.replace("_", "-"), values


def build_argv(base: List[str], grids: List[str]) -> List[List[str]]:
    """One argument list per grid point, each starting from the base arguments.

    Unless the caller pinned ``--tag`` already, each point is tagged with its swept
    values so that the runs land in separate result folders.
    """
    parsed = dict(parse_grid(spec) for spec in grids)
    if len(parsed) != len(grids):
        raise GridError("duplicate keys in --grid specifications")
    tagged = "--tag" not in base
    argvs = []
    for values in itertools.product(*parsed.values()):
        label = "-".join(f"{key}{value}" for key, value in zip(parsed, values))
        extra = sum(([f"--{key}", value] for key, value in zip(parsed, values)), [])
        if tagged:
            extra += ["--tag", label]
        argvs.append(base + extra)
    return argvs


def main(argv: List[str] | None = None) -> int:
    known, base = _parser().parse_known_args(argv)
    base = [token for token in base if token != "--"]      # optional separator
    if not base:
        _parser().error("no CLI arguments given: pass the base settings after --grid, "
                        "for example -- --dataset <dataset> --rounds <R> ...")
    if not known.grid:
        _parser().error("no --grid given: nothing to sweep; call paha_fc.cli directly "
                        "for a single configuration")
    try:
        argvs = build_argv(base, known.grid)
    except GridError as exc:
        _parser().error(str(exc))
        return 2

    print(f"# {len(argvs)} configuration(s)", flush=True)
    for index, args in enumerate(argvs, start=1):
        print(f"--- [{index}/{len(argvs)}] {' '.join(args)}", flush=True)
        status = cli_main(args)
        if status != 0:
            return status
    return 0


def _parser():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Unrecognised arguments are forwarded to paha_fc.cli unchanged.")
    parser.add_argument("--grid", action="append", default=[], metavar="KEY=V1,V2",
                        help="swept configuration key and values; repeatable")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
