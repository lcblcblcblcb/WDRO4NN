"""
dnn_dro  –  Torch-centric helpers for our DRO experiments.

Top-level re-exports
--------------------
>>> from dnn_dro import gaussian, relu, softmax, cross_entropy
>>> from dnn_dro import global_L, local_L          # ← Lipschitz stubs
"""

from importlib.metadata import version as _pkg_version

# ------------------------------------------------------------------
# Version string (0+dev when running from source tree)
# ------------------------------------------------------------------
try:
    __version__ = _pkg_version(__name__)
except Exception:        # pragma: no cover
    __version__ = "0+dev"

# ------------------------------------------------------------------
# Low-level utilities
# ------------------------------------------------------------------
from .utils import (
    make_rng,
    manual_seed_all,
    gaussian,
    spec_norm,
    batched_spec_norm
)

from .activations import (
    relu,
    softmax,
    cross_entropy,      # alias to torch.nn.functional.cross_entropy
)

# ------------------------------------------------------------------
# Lipschitz placeholders (raise NotImplementedError for now)
# ------------------------------------------------------------------
from .lipschitz import (
    global_L,          # TODO: implement
    local_L,           # TODO: implement
)

# ------------------------------------------------------------------
# Expose `models` namespace even if still empty
# ------------------------------------------------------------------
from . import models

__all__ = [
    # utils
    "make_rng", "manual_seed_all", "gaussian", "spec_norm", "batched_spec_norm",
    # activations
    "relu", "softmax", "cross_entropy",
    # Lipschitz (stubs for now)
    "global_L", "local_L",
    # sub-packages
    "models",
    # meta
    "__version__",
]