"""Command-line driver.

Every run-defining setting is taken from the command line or from a JSON file
passed with ``--config``; nothing is invented here. Examples::

    python -m paha_fc.cli --dataset <name> --data-root <dir> --out-dir <dir> \
        --device <device> --num-tasks <T> --clients <N> --alpha <alpha> \
        --participation <p> --rounds <R> --local-epochs <I> --seeds <s,...> \
        <objective, memory, solver and alignment settings>

    python -m paha_fc.cli --config run.json --seeds <s,...>
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from typing import Any, Dict, List, Optional

from .config import (ABLATIONS, DATASETS, Config, ablation_preset, legacy_overrides)
from .engine import ResultStore, aggregate_runs, run
from .logging_utils import set_verbosity


def _numbers(spec: str, cast):
    return [cast(v) for v in str(spec).split(",") if v.strip()]


def _dimension(spec: Optional[str], key: str, merged: Dict[str, Any], cast) -> List[Any]:
    """Values for a swept dimension: a comma list, one supplied value, or unset."""
    if spec:
        return _numbers(spec, cast)
    value = merged.get(key)
    return [cast(value)] if value is not None else [None]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="paha", description="Participation-aware historical "
                                "aggregation for federated class-incremental learning")

    data = p.add_argument_group("data and federation")
    data.add_argument("--dataset", choices=DATASETS, default=None)
    data.add_argument("--data-root", default=None)
    data.add_argument("--num-tasks", default=None, help="tasks per run; comma list sweeps")
    data.add_argument("--clients", dest="n_clients", type=int, default=None)
    data.add_argument("--alpha", default=None, help="Dirichlet concentration; 0 = IID. "
                                                    "Comma list sweeps.")
    data.add_argument("--resize", type=int, default=None)
    data.add_argument("--task-order-seed", type=int, default=None)

    fed = p.add_argument_group("optimisation")
    fed.add_argument("--rounds", type=int, default=None)
    fed.add_argument("--participation", default=None, help="p in (0,1]; comma list sweeps")
    fed.add_argument("--batch-size", type=int, default=None)
    fed.add_argument("--local-epochs", type=int, default=None)
    fed.add_argument("--lr", type=float, default=None)
    fed.add_argument("--momentum", type=float, default=None)
    fed.add_argument("--weight-decay", type=float, default=None)
    fed.add_argument("--max-grad-norm", type=float, default=None,
                     help="clip the local update by global norm; 0 leaves it as is")
    fed.add_argument("--server-lr", type=float, default=None)
    fed.add_argument("--optimizer", choices=("sgd", "adam"), default=None)

    obj = p.add_argument_group("objective and memory")
    obj.add_argument("--latent-dim", type=int, default=None)
    obj.add_argument("--lam-rec", type=float, default=None)
    obj.add_argument("--lam-cluster", type=float, default=None)
    obj.add_argument("--lam-enc", type=float, default=None)
    obj.add_argument("--lam-dec", type=float, default=None)
    obj.add_argument("--kl-weight", type=float, default=None)
    obj.add_argument("--sigma-syn", type=float, default=None)
    obj.add_argument("--local-replay-per-class", type=int, default=None)
    obj.add_argument("--global-replay-per-class", type=int, default=None)
    obj.add_argument("--memory-sample-per-class", type=int, default=None)

    hist = p.add_argument_group("historical aggregation")
    hist.add_argument("--rho-base", type=float, default=None)
    hist.add_argument("--gamma", type=float, default=None)
    hist.add_argument("--no-projection", action="store_true",
                      help="aggregate with plain FedAvg instead")
    hist.add_argument("--qp-solver", choices=("pgd", "active_set"), default=None)
    hist.add_argument("--qp-tol", type=float, default=None)
    hist.add_argument("--qp-max-iters", type=int, default=None)

    align = p.add_argument_group("centroid alignment")
    align.add_argument("--lj-eps", type=float, default=None)
    align.add_argument("--lj-sigma", type=float, default=None)
    align.add_argument("--lj-delta", type=float, default=None)
    align.add_argument("--lj-lr", type=float, default=None)
    align.add_argument("--lj-iters", type=int, default=None)
    align.add_argument("--lj-trust-region", type=float, default=None)

    run_g = p.add_argument_group("run control")
    run_g.add_argument("--config", help="JSON file with Config overrides")
    run_g.add_argument("--ablation", choices=ABLATIONS, default="full")
    run_g.add_argument("--seeds", default=None, help="comma list of seeds, e.g. 1,2,3")
    run_g.add_argument("--seed", type=int, default=None)
    run_g.add_argument("--device", default=None)
    run_g.add_argument("--out-dir", default=None)
    run_g.add_argument("--tag", default="")
    run_g.add_argument("--log-every", type=int, default=None)
    run_g.add_argument("--legacy-behaviour", action="store_true",
                       help="use the older non-default settings "
                            "(see config.legacy_overrides)")
    run_g.add_argument("--dry-run", action="store_true", help="print the resolved configs")
    run_g.add_argument("--quiet", action="store_true")
    run_g.add_argument("--verbose", action="store_true")
    return p


_DIRECT_FIELDS = (
    "dataset", "data_root", "num_tasks", "n_clients", "alpha", "resize", "task_order_seed",
    "rounds", "participation", "batch_size", "local_epochs", "lr", "momentum",
    "weight_decay", "max_grad_norm", "server_lr", "optimizer", "latent_dim", "lam_rec",
    "lam_cluster", "lam_enc", "lam_dec", "kl_weight", "sigma_syn",
    "local_replay_per_class", "global_replay_per_class", "memory_sample_per_class",
    "rho_base", "gamma", "lj_eps", "lj_sigma", "lj_delta", "lj_lr", "lj_iters",
    "lj_trust_region", "qp_solver", "qp_tol", "qp_max_iters", "device", "out_dir",
    "log_every",
)


def resolve_configs(args: argparse.Namespace) -> List[Config]:
    """JSON overrides, then CLI overrides, expanded over the swept dimensions."""
    merged: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as fh:
            merged.update(json.load(fh))
    for name in _DIRECT_FIELDS:
        value = getattr(args, name, None)
        if value is not None:
            merged[name] = value
    if args.no_projection:
        merged["projection"] = False
    if args.ablation != "full":
        merged.update(ablation_preset(args.ablation))
    if args.legacy_behaviour:
        merged.update(legacy_overrides())
    merged["tag"] = args.tag

    # Swept dimensions keep their comma lists; anything unset stays unset.
    tasks = _dimension(args.num_tasks, "num_tasks", merged, int)
    alphas = _dimension(args.alpha, "alpha", merged, float)
    parts = _dimension(args.participation, "participation", merged, float)
    if args.seeds:
        seeds = _numbers(args.seeds, int)
    elif args.seed is not None:
        seeds = [int(args.seed)]
    else:
        seeds = _dimension(None, "seed", merged, int)

    configs = []
    for num_tasks, alpha, participation, seed in itertools.product(tasks, alphas, parts, seeds):
        payload = dict(merged)
        payload.update(num_tasks=num_tasks, alpha=alpha, participation=participation, seed=seed)
        configs.append(Config.from_dict(payload))
    return configs


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    set_verbosity(verbose=args.verbose, quiet=args.quiet)

    try:
        configs = resolve_configs(args)
    except Exception as exc:                      # incomplete or inconsistent settings
        print(f"paha: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        for cfg in configs:
            print(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
        print(f"# {len(configs)} configuration(s) resolved", file=sys.stderr)
        return 0

    all_runs: List[Dict] = []
    for index, cfg in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {cfg.dataset} T={cfg.num_tasks} "
              f"N={cfg.n_clients} p={float(cfg.participation):g} alpha={cfg.alpha} "
              f"seed={cfg.seed}", flush=True)
        store = ResultStore(cfg, ablation=args.ablation)
        result = run(cfg, verbose=not args.quiet, result_store=store)
        entry = store.record_run(result, seed=cfg.seed, extra={"ablation": args.ablation})
        store.summarize()
        all_runs.append(entry)
        print(f"  final_acc={result.final_acc:.2f} forgetting={result.forgetting:.2f} "
              f"-> {store.dir}", flush=True)

    summary = aggregate_runs(all_runs)
    if summary:
        print(json.dumps(summary, indent=2))
    if len(configs) > 1:
        ResultStore(configs[0], ablation=args.ablation,
                    subdir=f"sweep_{args.ablation}").write_matrix(all_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
