"""Final accuracy and average forgetting.

``acc_history[t][k]`` is the class-incremental accuracy of task ``k`` recorded
after task ``t`` was trained, with tasks numbered from 0 and ``k <= t``.

* final accuracy -- mean over tasks of the row recorded after the last task;
* average forgetting -- mean over the earlier tasks of (best recorded accuracy of
  that task) minus (its accuracy after the last task).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

import numpy as np

AccHistory = List[List[float]]


def final_accuracy(acc_history: Sequence[Sequence[float]]) -> float:
    if not acc_history:
        return 0.0
    return float(np.mean(acc_history[-1]))


def forgetting(acc_history: Sequence[Sequence[float]]) -> float:
    """Mean drop from each earlier task's peak accuracy to its final accuracy."""
    n_tasks = len(acc_history)
    if n_tasks < 2:
        return 0.0
    final_row = acc_history[-1]
    drops = []
    for task in range(n_tasks - 1):
        recorded = [row[task] for row in acc_history if len(row) > task]
        drops.append(max(recorded) - final_row[task])
    return float(np.mean(drops))


def per_task_forgetting(acc_history: Sequence[Sequence[float]]) -> List[float]:
    n_tasks = len(acc_history)
    if n_tasks < 2:
        return []
    final_row = acc_history[-1]
    return [max(row[task] for row in acc_history if len(row) > task) - final_row[task]
            for task in range(n_tasks - 1)]


@dataclass
class RunMetrics:
    final_acc: float
    forgetting: float
    per_task_forgetting: List[float]
    curve: List[List[float]]

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def compute_final_metrics(acc_history: AccHistory) -> RunMetrics:
    return RunMetrics(final_acc=final_accuracy(acc_history),
                             forgetting=forgetting(acc_history),
                             per_task_forgetting=per_task_forgetting(acc_history),
                             curve=[list(row) for row in acc_history])
