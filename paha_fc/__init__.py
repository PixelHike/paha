"""PAHA: participation-aware historical aggregation for FCIL.

Package layout
--------------
``config``          :class:`Config`, presets and validation
``data``            dataset loaders, task splits, client partitioning
``models``          probabilistic encoder, decoder, latent-variable helpers
``memory``          client-side latent exemplar bank
``replay``          local and global replay channels, mixed set, balanced batches
``centroids``       task-boundary centroid exchange and LJ alignment
``aggregation``     FedAvg, client update memory (CUM), layerwise projection (LHP)
``training``        composite objective, local update, task-boundary refresh
``evaluation``      nearest-centroid read-out and reported metrics
``engine``          participation sampling, task loop, run records
``cli``             command-line driver
"""

from .config import ABLATIONS, DATASETS, Config, ConfigError
from .engine import RunResult, run

__all__ = ["Config", "ConfigError", "DATASETS", "ABLATIONS", "run",
           "RunResult"]

__version__ = "1.0.0"
