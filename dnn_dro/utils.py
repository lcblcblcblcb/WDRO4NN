import torch
from torch import Tensor
from typing import Tuple

# ──────────────────────────────────────────────────────────────────────
# RNG plumbing
# ──────────────────────────────────────────────────────────────────────
def make_rng(seed: int | None = None,
             *,
             device: str | torch.device = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return g

def manual_seed_all(seed: int) -> None:
    """
    Convenience: makes CPU, current CUDA device, and Python's
    built-in `random` deterministic in one call.
    """
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ──────────────────────────────────────────────────────────────────────
# Linear-algebra
# ──────────────────────────────────────────────────────────────────────
def spec_norm(mat: Tensor) -> Tensor:
    return torch.norm(mat, p=2)  # spectral norm

def batched_spec_norm(mats: Tensor,
                      *,
                      n_steps: int | None = None) -> Tensor:
    """
    Shape (B, m, n) → (B,) of spectral norms using the same back-end
    (`svdvals` or power-iter) as `spec_norm`.
    """
    if n_steps is None:
        return torch.linalg.svdvals(mats).amax(-1)
    out = []
    for M in mats:                       # loop is fine – B is small
        out.append(spec_norm(M, n_steps=n_steps))
    return torch.stack(out)

# ──────────────────────────────────────────────────────────────────────
# Gaussian sampling
# ──────────────────────────────────────────────────────────────────────
def gaussian(shape: Tuple[int, ...],
             *,
             mean: float = 0.0,
             std: float = 1.0,
             rng: torch.Generator | None = None,
             device: str | torch.device = "cpu",
             dtype: torch.dtype = torch.float32) -> Tensor:
    g = torch.randn(shape, generator=rng, device=device, dtype=dtype)
    return g.mul_(std).add_(mean)