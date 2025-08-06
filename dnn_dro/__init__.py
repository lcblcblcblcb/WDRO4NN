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
    batched_spec_norm,
    relu,
    softmax,
    cross_entropy,
)

# ------------------------------------------------------------------
# Expose `models` namespace even if still empty
# ------------------------------------------------------------------
from . import models

__all__ = [
    # utils
    "make_rng", "manual_seed_all", 
    "gaussian", "spec_norm", "batched_spec_norm",
    "relu", "softmax", "cross_entropy",    
    # sub-packages
    "models", "experiments",
    # meta
    "__version__",
]