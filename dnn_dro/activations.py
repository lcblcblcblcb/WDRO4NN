import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Any

# --------------------------------------------------------------------
# Element-wise ReLU
# --------------------------------------------------------------------
def relu(z: Tensor) -> Tensor:
    return F.relu(z)

# --------------------------------------------------------------------
# Softmax  (numerically stable)
# --------------------------------------------------------------------
def softmax(logits: Tensor, dim: int = -1) -> Tensor:
    return F.softmax(logits, dim=dim)

# --------------------------------------------------------------------
# Cross-entropy
# --------------------------------------------------------------------
cross_entropy = F.cross_entropy