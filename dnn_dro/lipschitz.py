from __future__ import annotations
import torch
from torch import Tensor

# ---------------------------------------------------------------------
# Global bound – to be implemented
# ---------------------------------------------------------------------
def global_L(model, **kwargs) -> Tensor:   # noqa: N802
    """
    Placeholder for

        L_global = max_{D} √2 · ‖W₂ D W₁‖₂

    or the analogue you’ll derive for deeper nets.

    Parameters
    ----------
    model : nn.Module
        The network whose Lipschitz constant you want.
    **kwargs
        Optional future arguments (device, max_masks, rng, …).

    Returns
    -------
    0-D tensor
        Raises `NotImplementedError` for now.
    """
    raise NotImplementedError(
        "global_L is not yet implemented – decide on the formula first."
    )


# ---------------------------------------------------------------------
# Per-sample (local) bound – to be implemented
# ---------------------------------------------------------------------
def local_L(model, x: Tensor, **kwargs) -> Tensor:   # noqa: N802
    """
    Placeholder for a *sample-dependent* Lipschitz constant, e.g.

        L_i = √2 · max_{D ∈ 𝒟(x)} ‖W₂ D W₁‖₂
              where 𝒟(x) are masks reachable from x.

    Parameters
    ----------
    model : nn.Module
    x     : Tensor
        One input sample.
    **kwargs
        Extra knobs you may add later (include_anchor, device, …).

    Returns
    -------
    0-D tensor
        Raises `NotImplementedError` for now.
    """
    raise NotImplementedError(
        "local_L is not yet implemented – decide on the formula first."
    )