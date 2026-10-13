"""Dataset loading, task splitting and client partitioning."""

from .loaders import DatasetBundle, load_dataset, to_device_tensor
from .partitioning import build_client_data, dirichlet_assign, iid_assign
from .task_splits import make_task_splits

__all__ = [
    "DatasetBundle",
    "load_dataset",
    "to_device_tensor",
    "make_task_splits",
    "build_client_data",
    "iid_assign",
    "dirichlet_assign",
]
