"""Trunk + VA head (PROJECT_SPEC.md Section 3.1/3.2).

The only trained parameters in the whole pipeline. Both modules are
applied per-window with weights shared across time (no cross-window
mixing) -- windows are independent samples, not a sequence the model
attends over. That keeps Step 1 training a plain per-window regression
problem; "within-clip dynamic tracking" (Step 2) is checked at
evaluation time by grouping a clip's windows back together, not by
anything architectural in the trunk itself.
"""

from __future__ import annotations

import torch
from torch import nn


class _ResidualBlock(nn.Module):
    """linear -> norm -> GELU -> linear, residual add (spec Section 3.1)."""

    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class FmriTrunk(nn.Module):
    """Per-window squeeze (P -> 512, shared weights) + 2 residual blocks.

    MindEye1's squeeze-plus-residual-blocks shape, applied independently
    to each window rather than once on a single pooled input (spec
    Section 3.1) -- deliberately *not* per-timestep distinct weights
    (that's Dynadiff's contribution, cited as inspiration but not what
    this spec calls for: "one linear layer P -> 512, applied
    independently to each window with shared weights").

    Input: (..., P). Output: (..., 512) -- one fingerprint per window,
    never pooled over the window/time dimension.
    """

    def __init__(self, n_parcels: int = 400, width: int = 512, n_blocks: int = 2):
        super().__init__()
        self.squeeze = nn.Linear(n_parcels, width)
        self.squeeze_dropout = nn.Dropout(0.5)
        self.blocks = nn.ModuleList([_ResidualBlock(width, dropout=0.15) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.squeeze(x)
        h = self.squeeze_dropout(h)
        for block in self.blocks:
            h = block(h)
        return h


class VAHead(nn.Module):
    """512 -> 128 -> 2, one hidden layer, GELU, dropout 0.1 (spec Section 3.2).

    Output columns are (valence, arousal). Dominance is dropped -- none
    of the approved datasets label it.
    """

    def __init__(self, width: int = 512, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class FmriTrunkVA(nn.Module):
    """Trunk + VA head, trained jointly with a single MSE/Huber VA loss."""

    def __init__(self, n_parcels: int = 400, width: int = 512, n_blocks: int = 2):
        super().__init__()
        self.trunk = FmriTrunk(n_parcels=n_parcels, width=width, n_blocks=n_blocks)
        self.va_head = VAHead(width=width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.va_head(self.trunk(x))
