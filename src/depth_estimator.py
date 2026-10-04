"""Monocular depth estimation (Phase 3 — not yet implemented).

Planned backend: Intel MiDaS via torch.hub, or the newer Depth Anything V2.
Interface is designed so callers never see the backend.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np


log = logging.getLogger("stereoscopy.depth")


class DepthEstimator:
    """Produces normalized depth maps (float32, range [0, 1]; near = 1, far = 0)."""

    SUPPORTED_MODELS = ("DPT_Large", "DPT_Hybrid", "MiDaS_small")

    def __init__(self, model_type: str = "MiDaS_small", device: str = "auto", batch_size: int = 4):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model {model_type!r}; choose from {self.SUPPORTED_MODELS}")
        self.model_type = model_type
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._transform = None

    def load(self) -> "DepthEstimator":
        raise NotImplementedError("Phase 3: load MiDaS via torch.hub")

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """BGR frame (H, W, 3) → depth map (H, W) float32 in [0, 1]."""
        raise NotImplementedError("Phase 3: single-frame inference")

    def batch_estimate(self, frames: Iterable[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError("Phase 3: batched inference")

    def smooth_temporal(self, depth_maps: list[np.ndarray], alpha: float = 0.5) -> list[np.ndarray]:
        """Exponential moving average across time to reduce flicker."""
        raise NotImplementedError("Phase 3: temporal EMA")
