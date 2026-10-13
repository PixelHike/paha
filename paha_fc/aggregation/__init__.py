"""Server-side aggregation: FedAvg, CUM memory and layerwise projection."""

from .cum import ClientUpdateMemory
from .dual_qp import kkt_residual, solve_dual, solve_dual_active_set, solve_dual_pgd
from .fedavg import fedavg, zero_delta
from .lhp import LayerwiseHistoricalProjection
from .server import PAHAServer, ServerStep

__all__ = [
    "ClientUpdateMemory",
    "LayerwiseHistoricalProjection",
    "PAHAServer",
    "ServerStep",
    "fedavg",
    "zero_delta",
    "solve_dual",
    "solve_dual_pgd",
    "solve_dual_active_set",
    "kkt_residual",
]
