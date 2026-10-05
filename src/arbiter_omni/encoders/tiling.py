"""
Dynamic High-Resolution Image Tiling (LLaVA-NeXT / S2-Wrapper Style).
Decomposes high-resolution photos and science diagrams into a multi-scale grid:
1 global downsampled overview tile + 4 localized high-resolution quadrant sub-crops,
producing fine-grained spatial patch tokens (up to 980 patches) without exceeding 1 GB VRAM.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn


def load_as_pil_image(image_input: Any) -> Optional[Image.Image]:
    """Converts a file path, numpy array, torch tensor, or PIL Image into a PIL Image."""
    if image_input is None:
        return None
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, str):
        if os.path.exists(image_input):
            try:
                return Image.open(image_input).convert("RGB")
            except Exception:
                return None
        return None
    if isinstance(image_input, np.ndarray):
        if image_input.dtype != np.uint8:
            image_input = (image_input * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(image_input).convert("RGB")
    if isinstance(image_input, torch.Tensor):
        arr = image_input.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # [C, H, W]
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    return None


class DynamicImageTiler:
    """
    High-Resolution Dynamic Multi-Scale Image Tiler.
    
    Splits high-resolution inputs (diagrams, photos, inspection images) into:
    - 1 Global Overview Tile (224x224): captures global visual semantics and scene composition.
    - 4 Quadrant Sub-Crops (224x224 each): captures fine-grained text, tiny symbols, and local textures.
    """

    def __init__(
        self,
        min_dim_for_tiling: int = 336,
        target_tile_size: int = 224,
    ):
        self.min_dim_for_tiling = min_dim_for_tiling
        self.target_tile_size = target_tile_size

    def tile_image(
        self,
        image_input: Any,
        force_tiling: bool = False,
    ) -> List[Image.Image]:
        """
        Decomposes an image into multi-scale tiles.
        
        Args:
            image_input: PIL Image, filepath, or array.
            force_tiling: If True, produces 5 tiles even if resolution is small.
            
        Returns:
            tiles: List of PIL Images [1 tile] if small, or [5 tiles] (1 global + 4 quadrants).
        """
        img = load_as_pil_image(image_input)
        if img is None:
            # Fallback 1x1 black tile
            return [Image.new("RGB", (self.target_tile_size, self.target_tile_size), color=(0, 0, 0))]

        w, h = img.size
        # Always generate global overview tile
        global_tile = img.resize(
            (self.target_tile_size, self.target_tile_size),
            resample=Image.Resampling.BICUBIC,
        )

        should_tile = force_tiling or max(w, h) >= self.min_dim_for_tiling
        if not should_tile:
            return [global_tile]

        # 4 Quadrant Sub-Crops
        x_mid = w // 2
        y_mid = h // 2

        # Bounding boxes: (left, upper, right, lower)
        boxes = [
            (0, 0, x_mid, y_mid),          # Top-Left
            (x_mid, 0, w, y_mid),          # Top-Right
            (0, y_mid, x_mid, h),          # Bottom-Left
            (x_mid, y_mid, w, h),          # Bottom-Right
        ]

        quadrants = []
        for box in boxes:
            crop = img.crop(box)
            crop_resized = crop.resize(
                (self.target_tile_size, self.target_tile_size),
                resample=Image.Resampling.BICUBIC,
            )
            quadrants.append(crop_resized)

        # Total 5 tiles: [Global, TL, TR, BL, BR]
        return [global_tile] + quadrants
