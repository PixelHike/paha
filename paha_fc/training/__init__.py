"""Local training: composite objective, one client round, task-boundary state."""

from .losses import CompositeLoss, LossTerms, composite_loss
from .local_trainer import ClientWorker, local_update
from .task_boundary import collect_and_align_centroids, refresh_client_memories

__all__ = [
    "CompositeLoss",
    "LossTerms",
    "composite_loss",
    "ClientWorker",
    "local_update",
    "collect_and_align_centroids",
    "refresh_client_memories",
]
