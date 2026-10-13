"""Central run configuration for PAHA.

Field groups:

``rho_base`` / ``gamma``   client-update-memory refresh coefficients
``lam_cluster``            clustering weight of the composite objective
``lam_enc`` / ``lam_dec``  weights of the two distillation terms
``sigma_syn``              perturbation std of the global replay channel
``server_lr``              server step size applied to the projected aggregate
``lj_*``                   Lennard-Jones centroid-alignment parameters

The settings that define a run carry no values of their own: they are populated
from the caller, and :meth:`Config.validate` reports every one that is still
missing in a single message. Options that only select a code path (booleans and
named modes) are listed with their switch semantics below.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional

DATASETS = ("cifar100", "tinyimagenet", "stanforddogs", "synthetic")
ABLATIONS = ("full", "no_history", "no_local", "no_global", "no_replay")
QP_SOLVERS = ("pgd", "active_set")

#: Fields a run cannot be described without; ``None`` means "not supplied".
REQUIRED_FIELDS = (
    "dataset", "data_root", "out_dir", "device", "seed",
    "num_tasks", "n_clients", "alpha", "participation", "task_order_seed",
    "rounds", "local_epochs", "batch_size",
    "lr", "momentum", "weight_decay", "server_lr", "max_grad_norm",
    "latent_dim",
    "lam_rec", "lam_cluster", "lam_enc", "lam_dec", "kl_weight", "sigma_syn",
    "local_replay_per_class", "global_replay_per_class", "memory_sample_per_class",
    "rho_base", "gamma",
    "lj_eps", "lj_sigma", "lj_delta", "lj_lr", "lj_iters", "lj_trust_region",
    "qp_tol", "qp_max_iters",
)


class ConfigError(ValueError):
    """Raised when a configuration is incomplete or internally inconsistent."""


def option_name(field_name: str) -> str:
    """CLI spelling of a configuration field."""
    if field_name == "n_clients":
        return "--clients"
    return "--" + field_name.replace("_", "-")


@dataclass
class Config:
    # ------------------------------------------------------------------ data
    dataset: Optional[str] = None      # one of DATASETS
    data_root: Optional[str] = None
    num_tasks: Optional[int] = None
    n_clients: Optional[int] = None
    alpha: Optional[float] = None      # 0 selects the IID partition, else Dirichlet(alpha)
    resize: Optional[int] = None       # resolution override; default comes from the data
    seed: Optional[int] = None
    task_order_seed: Optional[int] = None

    # ------------------------------------------------------------ federation
    rounds: Optional[int] = None       # communication rounds per task
    participation: Optional[float] = None   # p in (0, 1]; active clients = ceil(p * N)
    batch_size: Optional[int] = None
    local_epochs: Optional[int] = None

    # ------------------------------------------------------------- optimizer
    optimizer: str = "sgd"             # "sgd" (momentum) or "adam"
    lr: Optional[float] = None         # local step size
    momentum: Optional[float] = None
    weight_decay: Optional[float] = None
    reset_momentum: bool = True        # zero the momentum buffer at the start of a round
    max_grad_norm: Optional[float] = None   # clip by global norm; 0 leaves the update as is
    server_lr: Optional[float] = None  # step applied to the projected aggregate

    # ----------------------------------------------------------------- model
    latent_dim: Optional[int] = None
    input_size: Optional[int] = None   # fallback resolution when the data gives none

    # ---------------------------------------------------------------- losses
    lam_rec: Optional[float] = None        # weight of the reconstruction term
    lam_cluster: Optional[float] = None    # weight of the centroid-clustering term
    lam_enc: Optional[float] = None        # weight of the encoder-drift term
    lam_dec: Optional[float] = None        # weight of the decoder-drift term
    kl_weight: Optional[float] = None      # scale of the analytic KL term
    sigma_syn: Optional[float] = None      # global-replay perturbation std
    mse_reduction: str = "sum"         # "sum" = Gaussian likelihood, "mean"

    # -------------------------------------------------- memory / replay size
    local_replay_per_class: Optional[int] = None    # per-class capacity of the bank
    global_replay_per_class: Optional[int] = None   # synthetic samples per class
    memory_sample_per_class: Optional[int] = None   # codes enqueued per new class
    enqueue_sample: str = "random"     # "random" or "head"
    use_local_replay: bool = True      # False -> No-Local
    use_global_replay: bool = True     # False -> No-Global

    # ---------------------------------------------- participation-aware CUM
    rho_base: Optional[float] = None   # base of the participation-dependent coefficient
    gamma: Optional[float] = None      # history decay applied to inactive clients
    gamma_strict: bool = True          # require 0.9 < gamma < 1.0 on validation

    # -------------------------------------------------- centroid alignment
    lj_eps: Optional[float] = None         # potential depth epsilon_LJ
    lj_sigma: Optional[float] = None       # equilibrium spacing, in latent-space units
    lj_delta: Optional[float] = None       # offset keeping the potential finite
    lj_lr: Optional[float] = None          # centroid descent step size
    lj_iters: Optional[int] = None
    lj_trust_region: Optional[float] = None    # per-iteration move cap, in units of sigma
    lj_sigma_mode: str = "fixed"       # "fixed" = sigma as given; "mean_pairwise" = rescale
    lj_step_normalize: bool = False    # step along the raw gradient by default

    # ---------------------------------------------------- LHP dual solver
    projection: bool = True            # False -> plain FedAvg aggregation
    qp_solver: str = "pgd"             # "pgd" (accelerated) or "active_set" (exact)
    qp_tol: Optional[float] = None     # relative KKT tolerance of the dual solve
    qp_max_iters: Optional[int] = None

    # ---------------------------------------------------- task-level state
    anchors_from_task1: bool = True        # anchor terms active on the first task too
    eval_with_aligned_centroids: bool = True   # read out the aligned centroid set
    refresh_old_codes: bool = True         # re-encode stored codes each boundary
    refresh_participants_only: bool = True  # refresh only clients that trained this task
    verify_centroid_coverage: bool = True  # require a centroid for every new class

    # ----------------------------------------------------------------- misc
    log_every: int = 0                 # round print interval (0 -> rounds // 5)
    out_dir: Optional[str] = None
    tag: str = ""
    device: Optional[str] = None

    # ----------------------------------------------------------------- I/O
    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------ validation
    def missing_fields(self) -> List[str]:
        """Run-defining fields that were never populated."""
        return [name for name in REQUIRED_FIELDS if getattr(self, name) is None]

    def validate(self) -> None:
        """Reject incomplete or inconsistent configurations."""
        missing = self.missing_fields()
        if missing:
            raise ConfigError("missing settings: " + ", ".join(option_name(n) for n in missing))

        if self.dataset not in DATASETS:
            raise ConfigError(f"dataset must be one of {DATASETS}, got {self.dataset!r}")
        for name in ("num_tasks", "n_clients", "rounds", "local_epochs", "batch_size",
                     "latent_dim", "lj_iters", "qp_max_iters"):
            if int(getattr(self, name)) <= 0:
                raise ConfigError(f"{name} must be positive")
        if not 0.0 < float(self.participation) <= 1.0:
            raise ConfigError(f"participation must lie in (0, 1], got {self.participation}")
        if self.alpha < 0:
            raise ConfigError("alpha must be >= 0 (0 selects the IID partition)")
        if self.lr <= 0 or self.server_lr <= 0:
            raise ConfigError("local and server learning rates must be positive")
        if self.max_grad_norm < 0:
            raise ConfigError("max_grad_norm must be >= 0 (0 disables clipping)")
        if self.momentum < 0:
            raise ConfigError("momentum must be >= 0")
        if self.weight_decay < 0:
            raise ConfigError("weight_decay must be >= 0")
        if min(self.lam_rec, self.lam_cluster, self.lam_enc, self.lam_dec,
               self.kl_weight, self.sigma_syn) < 0:
            raise ConfigError("loss weights and sigma_syn must be >= 0")
        if self.optimizer not in ("sgd", "adam"):
            raise ConfigError("optimizer must be 'sgd' or 'adam'")
        if self.mse_reduction not in ("sum", "mean"):
            raise ConfigError("mse_reduction must be 'sum' or 'mean'")
        if self.enqueue_sample not in ("random", "head"):
            raise ConfigError("enqueue_sample must be 'random' or 'head'")
        for name in ("local_replay_per_class", "global_replay_per_class",
                     "memory_sample_per_class"):
            if int(getattr(self, name)) < 0:
                raise ConfigError(f"{name} must be >= 0")
        if not 0.0 < self.rho_base <= 1.0:
            raise ConfigError("rho_base must lie in (0, 1]")
        if self.gamma_strict and not 0.9 < self.gamma < 1.0:
            raise ConfigError(
                "gamma must lie in the supported range (0.9, 1.0); set gamma_strict="
                "False to run a decay outside that range"
            )
        if not 0.0 < self.gamma <= 1.0:
            raise ConfigError("gamma must lie in (0, 1] to keep the historical half-spaces")
        if self.lj_eps <= 0 or self.lj_sigma <= 0 or self.lj_delta <= 0 or self.lj_lr <= 0:
            raise ConfigError("lj_eps, lj_sigma, lj_delta and lj_lr must be positive")
        if self.lj_trust_region < 0:
            raise ConfigError("lj_trust_region must be >= 0 (0 removes the cap)")
        if self.lj_sigma_mode not in ("fixed", "mean_pairwise"):
            raise ConfigError("lj_sigma_mode must be 'fixed' or 'mean_pairwise'")
        if self.qp_solver not in QP_SOLVERS:
            raise ConfigError(f"qp_solver must be one of {QP_SOLVERS}")
        if self.qp_tol <= 0 or self.qp_max_iters <= 0:
            raise ConfigError("qp_tol and qp_max_iters must be positive")
        if self.resize is not None and self.resize <= 0:
            raise ConfigError("resize must be None or a positive integer")

    # --------------------------------------------------------------- helpers
    def resolve_input_size(self) -> int:
        """Resolution the decoder is built for: ``resize`` wins over ``input_size``."""
        size = self.resize if self.resize is not None else self.input_size
        if size is None:
            raise ConfigError("no image resolution: supply resize or input_size")
        base = 4
        scaled = base
        while scaled < size:
            scaled *= 2
        if scaled != size:
            raise ConfigError(f"decoder requires a resolution of 4*2^k, got {size}")
        return int(size)

    def n_active(self) -> int:
        """Number of clients sampled per training round, ``ceil(p * N)``."""
        return max(1, int(math.ceil(float(self.participation) * int(self.n_clients))))

    def step_log(self) -> int:
        return self.log_every if self.log_every > 0 else max(1, int(self.rounds) // 5)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], strict: bool = True) -> "Config":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown and strict:
            raise ConfigError(f"unknown configuration keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, path: str, **overrides: Any) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        payload.update(overrides)
        return cls.from_dict(payload)

    def copy(self, **overrides: Any) -> "Config":
        merged = self.to_dict()
        merged.update(overrides)
        return Config.from_dict(merged)


# ------------------------------------------------------------- setting bundles ---
# Component-removal bundles: each one only turns a mechanism off.
_ABLATION_PRESETS = {
    "full": {},
    "no_history": dict(projection=False),
    "no_local": dict(use_local_replay=False),
    "no_global": dict(use_global_replay=False),
    "no_replay": dict(use_local_replay=False, use_global_replay=False),
}

# Non-default behaviour retained behind a single flag: first-task anchoring
# switched off, classifier read-out re-estimated from the refreshed codes, no
# stored-code adaptation, a decay outside the supported range, a rescaled LJ
# spacing with a normalised step, the leading samples of a class instead of a
# random draw, and a fixed-budget dual solve.
_LEGACY_OVERRIDES = dict(
    anchors_from_task1=False,
    eval_with_aligned_centroids=False,
    refresh_old_codes=False,
    refresh_participants_only=False,
    gamma=1.0,
    gamma_strict=False,
    lj_sigma_mode="mean_pairwise",
    lj_step_normalize=True,
    enqueue_sample="head",
    qp_max_iters=500,
    qp_tol=1e-6,
)


def ablation_preset(name: str) -> Dict[str, Any]:
    if name not in _ABLATION_PRESETS:
        raise ConfigError(f"ablation must be one of {ABLATIONS}, got {name!r}")
    return dict(_ABLATION_PRESETS[name])


def legacy_overrides() -> Dict[str, Any]:
    return dict(_LEGACY_OVERRIDES)
