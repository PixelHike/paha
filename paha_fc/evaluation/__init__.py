"""Class-incremental evaluation and the reported metrics."""

from .classifier import class_il_per_task_accuracy, nearest_centroid_predict
from .metrics import RunMetrics, compute_final_metrics, final_accuracy, forgetting

__all__ = [
    "nearest_centroid_predict",
    "class_il_per_task_accuracy",
    "compute_final_metrics",
    "final_accuracy",
    "forgetting",
    "RunMetrics",
]
