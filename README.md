# PAHA — Participation-Aware Historical Aggregation for FCIL

Implementation of participation-aware historical aggregation (PAHA) for federated
class-incremental learning: a latent-variable autoencoder trained by replay-
supported local updates and aggregated on the server under **client update memory
(CUM)** and **layerwise historical projection (LHP)**, with task-boundary centroid
exchange and latent-memory refresh.

## Layout

```
paha/
├── paha_fc/
│   ├── config.py            Config dataclass, validation, setting bundles
│   ├── cli.py               command-line driver (`python -m paha_fc.cli`)
│   ├── data/                loaders, task splits, IID/Dirichlet client partitions
│   ├── models/              probabilistic encoder, decoder, VAE helpers, W packing
│   ├── memory/              latent exemplar bank with per-class capacity
│   ├── replay/              local + global replay, mixed set, balanced batches
│   ├── centroids/           centroid exchange, coverage check, LJ alignment, matrix
│   ├── aggregation/         FedAvg, CUM, dual QP solvers, LHP, server round step
│   ├── training/            composite loss, local update, task-boundary refresh
│   ├── evaluation/          nearest-centroid read-out, accuracy and forgetting
│   ├── engine/              participation streams, task loop, run records
│   ├── utils.py              seeding, device resolution, numeric helpers
│   └── logging_utils.py
├── scripts/                 run drivers (run_matrix.py, run_*.sh)
├── pyproject.toml
└── requirements.txt
```

## Install

```bash
pip install -r requirements.txt        # or: pip install -e .
```

Datasets are read from `--data-root`:

* **CIFAR-100** — downloaded by torchvision on first use (a preprocessed
  `<root>/cifar100_npz/` folder is used when present).
* **Tiny-ImageNet** — put `tiny-imagenet-200.zip` in the root, or pre-extract
  `tiny-imagenet-200/`.
* **Stanford Dogs** — the mirror layout `<root>/Images/{train,test}/<breed>/*.jpg`;
  images are decoded at a fixed resolution, as the decoder needs a fixed input size.
* **synthetic** — small deterministic colour blobs, for quick pipeline checks only.

## Usage

Pass the settings you want; the drivers do not supply any of them.

```bash
# single run
python -m paha_fc.cli --dataset <dataset> --data-root <dir> --out-dir <dir> \
    --num-tasks <T> --clients <N> --alpha <alpha> --participation <p> \
    --rounds <R> --local-epochs <I> --seeds <s,...> --device <device> \
    --lr <lr> --batch-size <B> --latent-dim <m> <remaining settings>

# sweep participation or task counts (comma lists are expanded by the CLI)
python -m paha_fc.cli --dataset <dataset> --data-root <dir> --out-dir <dir> \
    --num-tasks <T,...> --clients <N> --alpha <a,...> --participation <p1,p2,...> \
    --rounds <R> --local-epochs <I> --seeds <s,...> --device <device>

# component ablation
python -m paha_fc.cli --ablation <full|no_history|no_local|no_global|no_replay> ...

# cross product over any configuration key
bash scripts/run_participation.sh --grid <key>=<v1,v2,...> -- <cli arguments>

# print the resolved configuration without touching the data
python -m paha_fc.cli --dataset <dataset> --dry-run
```

`bash scripts/run_main.sh -- <cli arguments>` forwards a single run unchanged.
`python -m paha_fc.cli --help` lists every option; `paha_fc.config.Config` holds
the full field set, including the optimiser, loss weights, memory budgets,
centroid-alignment and dual-solver controls.

Every run writes `config.json`, `runs.jsonl` (one line per seed) and `summary.csv`
across seeds into `--out-dir/<cell>`; `tasks.jsonl` keeps per-task diagnostics
(centroid support, alignment shift, repository occupancy, per-task accuracies).

## Behaviour switches

* `anchors_from_task1` — whether the distillation and clustering anchors use the
  task-start model from the first task onward.
* `eval_with_aligned_centroids` — whether the classifier reads out the centroid set
  produced at the task boundary, which then stays fixed for the task.
* `refresh_old_codes` — whether stored codes are re-encoded through the frozen
  anchor decoder before the new-class codes are enqueued.
* `refresh_participants_only` — whether only clients that took part in a task's
  training rounds refresh their latent bank.
* `gamma_strict` — whether the history decay must stay inside its supported range.
* `projection` / `qp_solver` — historical projection on or off, and which dual
  solver solves the per-layer problem.
* `max_grad_norm` — clips the local update by global norm; `0` leaves it as is.

`--legacy-behaviour` turns on an older bundle of these switches in one flag; it is
kept for side-by-side comparison.

Settings that define a run are taken from the arguments or from `--config`; the
per-run `config.json` is a plain dump of the resolved configuration and can be fed
straight back through `--config`.

## Implementation notes

* **Dual solver.** The projection solves one quadratic program per layer, with as
  many variables as there are retained client directions. The default accelerated
  projected-gradient solver is batched over layers and stops on a relative KKT
  residual; the achieved residual is recorded per round as `lhp_kkt`.
* **Centroid alignment.** The Lennard-Jones potential is evaluated in float64,
  because coincident centroids overflow single precision. A trust-region cap limits
  how far one centroid may move per iteration while keeping the descent direction
  equal to the negative gradient.
* **Local optimiser.** Momentum SGD without Nesterov acceleration, with the momentum
  buffer reset at the start of every round.
