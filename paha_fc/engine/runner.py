"""End-to-end PAHA run over a task sequence.

Each task executes the same sequence of steps:

1. freeze the task anchor ``W_{t-1}`` and broadcast it to every client;
2. collect new-class centroids from all ``N`` clients, aggregate, verify
   coverage and align them against the fixed historical centroids;
3. run ``R`` participation-limited rounds of client training and server
   aggregation;
4. let the participating clients refresh their latent memories;
5. evaluate class-incrementally over all classes seen so far.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import torch

from ..aggregation import PAHAServer
from ..centroids import CentroidMatrix
from ..config import Config
from ..data import build_client_data, load_dataset, make_task_splits
from ..data.loaders import labels_to_tensor, to_device_tensor
from ..evaluation import class_il_per_task_accuracy, compute_final_metrics
from ..logging_utils import get_logger
from ..memory import LatentBank
from ..models import anchor_forward, build_autoencoder
from ..training import ClientWorker, local_update, refresh_client_memories
from ..training.task_boundary import collect_and_align_centroids
from ..utils import enable_tf32, resolve_device, set_seed
from .participation import partition_seed, sample_active_clients, task_round_seed

LOG = get_logger("engine")


@dataclass
class RunResult:
    final_acc: float
    forgetting: float
    acc_history: List[List[float]]
    task_splits: List[List[int]]
    elapsed_s: float
    config: Dict = field(default_factory=dict)
    per_task: List[Dict[str, float]] = field(default_factory=list)
    rounds: List[Dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {"final_acc": self.final_acc, "forgetting": self.forgetting,
                "acc_history": self.acc_history, "task_splits": self.task_splits,
                "elapsed_s": self.elapsed_s, "config": self.config,
                "per_task": self.per_task, "rounds": self.rounds}


def _task_config(cfg: Config, task: int) -> Config:
    """Task-start anchoring needs a task-start model; ``anchors_from_task1=False``
    switches the distillation and clustering terms off for the first task only."""
    if cfg.anchors_from_task1 or task == 0:
        return cfg
    return cfg.copy(lam_enc=0.0, lam_dec=0.0, lam_cluster=0.0)


def _readout_matrix(cfg: Config, centroids: CentroidMatrix, banks: Sequence[LatentBank],
                    device: torch.device) -> CentroidMatrix:
    """Centroid set the classifier reads out.

    The default read-out is the aligned centroid set itself. The alternative
    re-estimates each class mean from the refreshed codes of the clients holding
    that class, and is reachable only through ``eval_with_aligned_centroids``.
    """
    if cfg.eval_with_aligned_centroids:
        return centroids
    alt = CentroidMatrix(centroids.num_classes, centroids.latent_dim, device)
    alt.update(centroids.as_dict())
    grouped: Dict[int, List[torch.Tensor]] = {}
    for bank in banks:
        for code, label in bank.items():
            grouped.setdefault(label, []).append(code)
    alt.update({c: torch.stack(v).mean(dim=0) for c, v in grouped.items()})
    return alt


def run(cfg: Config, verbose: bool = True,
        result_store=None) -> RunResult:
    set_seed(cfg.seed)
    enable_tf32(True)
    device = resolve_device(cfg.device)
    cfg = cfg.copy(device=str(device))

    bundle = load_dataset(cfg.dataset, cfg.data_root, cfg.resize)
    cfg = cfg.copy(input_size=int(bundle.input_size), resize=int(bundle.input_size))
    input_size = cfg.resolve_input_size()
    LOG.info("dataset %s | %s", cfg.dataset, bundle.summary())

    train_x = to_device_tensor(bundle.train_x, device)
    train_y = labels_to_tensor(bundle.train_y, device)
    test_x = to_device_tensor(bundle.test_x, device)
    test_y = labels_to_tensor(bundle.test_y, device)

    task_splits = make_task_splits(bundle.num_classes, cfg.num_tasks, cfg.task_order_seed)
    autoencoder = build_autoencoder(cfg.latent_dim, input_size, device)
    anchor = build_autoencoder(cfg.latent_dim, input_size, device).freeze()
    server = PAHAServer(cfg, autoencoder=autoencoder)
    centroids = CentroidMatrix(bundle.num_classes, cfg.latent_dim, device)
    banks = [LatentBank(cfg.local_replay_per_class) for _ in range(cfg.n_clients)]
    workers = [ClientWorker(cfg, j, autoencoder) for j in range(cfg.n_clients)]

    acc_history: List[List[float]] = []
    seen_splits: List[List[int]] = []
    per_task_logs: List[Dict[str, float]] = []
    round_logs: List[Dict[str, float]] = []
    started = time.time()

    for task in range(cfg.num_tasks):
        task_cfg = _task_config(cfg, task)
        task_classes = [int(c) for c in task_splits[task]]
        seen_splits.append(task_classes)
        LOG.info("[task %d/%d] %d new classes, e.g. %s", task + 1, cfg.num_tasks,
                 len(task_classes), task_classes[:6])

        # --- task anchor: broadcast W_{t-1} ------------------------------------
        anchor_state = server.snapshot()
        anchor.set_state(anchor_state)

        cxs, cys = build_client_data(train_x, train_y, task_classes, cfg.n_clients,
                                     cfg.alpha, partition_seed(cfg, task))

        # --- centroid exchange: all N clients, then alignment ------------------
        boundary = collect_and_align_centroids(task_cfg, anchor, cxs, cys, task_classes,
                                               centroids)

        # --- anchor outputs of the real data, constant during the task ---------
        real_anchor = {}
        with torch.no_grad():
            for j in range(cfg.n_clients):
                if cxs[j].numel():
                    real_anchor[j] = anchor_forward(anchor, cxs[j])
                else:
                    real_anchor[j] = (torch.empty(0, cfg.latent_dim, device=device),
                                      torch.empty(0, 3, input_size, input_size, device=device))

        participants = set()
        for round_index in range(task_cfg.rounds):
            active = sample_active_clients(task_cfg, task, round_index)
            participants.update(active)
            deltas, round_terms = {}, []
            for j in active:
                delta, terms = local_update(
                    task_cfg, server.global_state, anchor, autoencoder, workers[j],
                    cxs[j], cys[j], banks[j], centroids, task_classes,
                    task_round_seed(task_cfg, task, round_index), real_anchor=real_anchor[j])
                if delta is None:
                    deltas[j] = server.zero_delta()      # client contributed no batch
                else:
                    deltas[j] = delta
                    round_terms.append(terms)
            sizes = {int(j): int(cys[j].numel()) for j in active}
            step = server.round_step(active, deltas, sizes)
            record = {"task": task + 1, "round": round_index + 1, "n_active": len(active)}
            record.update(step.info)
            if round_terms:
                record.update({k: float(np.mean([getattr(t, k) for t in round_terms]))
                               for k in ("total", "rec", "kl", "cluster", "enc", "dec")})
            round_logs.append(record)
            if verbose and (round_index + 1) % task_cfg.step_log() == 0:
                LOG.info("   task %d round %d/%d  |d_agg|=%.3f |d_proj|=%.3f kkt=%.2e  "
                         "rec=%.3f kl=%.3f clu=%.3f enc=%.3f dec=%.3f  (%.0fs)",
                         task + 1, round_index + 1, task_cfg.rounds, step.agg_norm,
                         step.proj_norm, record.get("lhp_kkt", 0.0), record.get("rec", 0.0),
                         record.get("kl", 0.0), record.get("cluster", 0.0),
                         record.get("enc", 0.0), record.get("dec", 0.0), time.time() - started)

        # --- task-boundary memory refresh of the participating clients ---------
        server.load_into(autoencoder)
        refresh = refresh_client_memories(task_cfg, autoencoder, anchor, banks, cxs, cys,
                                          task_classes, sorted(participants),
                                          seed=cfg.seed + 1000 * task)

        # --- class-incremental evaluation over all seen classes ----------------
        server.load_into(autoencoder)
        autoencoder.eval()
        readout = _readout_matrix(cfg, centroids, banks, device)
        accs = class_il_per_task_accuracy(autoencoder.enc, test_x, test_y, seen_splits,
                                          readout)
        acc_history.append(accs)

        task_info = {"task": task + 1, "classes": task_classes,
                     "participants": sorted(participants), "acc_after_task": accs}
        task_info.update(boundary)
        task_info.update(refresh)
        task_info.update(server.repository_summary())
        per_task_logs.append(task_info)
        if verbose:
            LOG.info("   -> after task %d acc: %s", task + 1,
                     " ".join("%.2f" % a for a in accs))

        if result_store is not None:
            result_store.append_task(task_info)

    metrics = compute_final_metrics(acc_history)
    elapsed = time.time() - started
    LOG.info("final acc %.2f | forgetting %.2f | %.0fs", metrics.final_acc,
             metrics.forgetting, elapsed)
    return RunResult(final_acc=metrics.final_acc, forgetting=metrics.forgetting,
                            acc_history=metrics.curve,
                            task_splits=[[int(c) for c in s] for s in task_splits],
                            elapsed_s=elapsed, config=cfg.to_dict(),
                            per_task=per_task_logs, rounds=round_logs)
