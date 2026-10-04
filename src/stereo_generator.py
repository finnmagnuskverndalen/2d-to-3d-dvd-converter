"""Stereo view synthesis from a frame + depth map (Phase 4 — not yet implemented)."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np


log = logging.getLogger("stereoscopy.stereo")

OutputFormat = Literal["side-by-side", "top-bottom", "anaglyph", "frame-sequential"]


class StereoGenerator:
    def __init__(
        self,
        baseline_px: int = 20,
        max_disparity: int = 50,
        convergence: float = 0.5,
        inpainting_method: str = "telea",
    ):
        self.baseline_px = baseline_px
        self.max_disparity = max_disparity
        self.convergence = convergence
        self.inpainting_method = inpainting_method

    def generate_stereo_pair(self, frame: np.ndarray, depth_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (left_eye, right_eye) BGR frames of the same shape as `frame`."""
        raise NotImplementedError("Phase 4: depth-based disparity warping")

    def apply_disparity_map(self, frame: np.ndarray, disparity: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Phase 4: horizontal pixel shift per column")

    def blend_occlusion_areas(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Phase 4: Telea / Navier-Stokes inpainting")

    def pack(self, left: np.ndarray, right: np.ndarray, output_format: OutputFormat) -> np.ndarray:
        """Combine the stereo pair into a single frame in the requested layout."""
        raise NotImplementedError("Phase 4: SBS / TB / anaglyph packing")

    @staticmethod
    def generate_anaglyph(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Red/cyan anaglyph for quick preview without 3D glasses hardware."""
        raise NotImplementedError("Phase 4: channel multiplex")
