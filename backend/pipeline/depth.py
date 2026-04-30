"""
Depth estimation using Depth Anything V2 (Small) from HuggingFace.

Convention throughout this project:
  depth value 0.0 = nearest (foreground)
  depth value 1.0 = farthest (background)
"""

import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Module-level singleton so the model is loaded once per process lifetime
_pipe = None

# Maximum input side before resizing for VRAM safety (pixels)
_MAX_SIDE = 1024


def _get_pipe():
    """Lazy-load the depth-estimation pipeline onto the best available device."""
    global _pipe
    if _pipe is not None:
        return _pipe

    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    device_label = (
        f"CUDA ({torch.cuda.get_device_name(0)})" if device == 0 else "CPU"
    )
    logger.info("Loading Depth Anything V2 Large on %s …", device_label)

    _pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device=device,
    )
    logger.info("Depth model ready.")
    return _pipe


def estimate_depth(image: Image.Image, max_side: int = _MAX_SIDE) -> np.ndarray:
    """
    Estimate a per-pixel relative depth map from an RGB PIL image.

    The model is run at *max_side* resolution for VRAM efficiency and the
    resulting depth map is bilinearly upscaled back to the original image size.

    Args:
        image:    RGB PIL Image (any size).
        max_side: Longest side used for depth inference.  Lower = faster / less VRAM.

    Returns:
        Float32 numpy array shaped (H, W) with values in [0.0, 1.0].
        0.0 = closest to camera (foreground), 1.0 = farthest (background).
    """
    orig_w, orig_h = image.size

    # Downscale for inference if needed
    scale = min(max_side / max(orig_w, orig_h), 1.0)
    if scale < 1.0:
        proc_w = max(int(orig_w * scale), 1)
        proc_h = max(int(orig_h * scale), 1)
        depth_input = image.resize((proc_w, proc_h), Image.LANCZOS)
        logger.debug("Resized input to %dx%d for depth inference.", proc_w, proc_h)
    else:
        depth_input = image

    pipe = _get_pipe()
    result = pipe(depth_input)

    # `predicted_depth` is a torch.Tensor; shape is [H, W] or [1, H, W]
    depth_tensor = result["predicted_depth"]
    depth_np = depth_tensor.squeeze().cpu().numpy().astype(np.float32)

    # Upscale to original image dimensions (H × W)
    if scale < 1.0:
        depth_pil = Image.fromarray(depth_np)
        depth_pil = depth_pil.resize((orig_w, orig_h), Image.BILINEAR)
        depth_np = np.array(depth_pil, dtype=np.float32)

    # Normalize to [0, 1].
    # Depth Anything V2 outputs metric-style depth: higher value = farther away.
    # That matches our convention directly (no inversion needed).
    # If the parallax effect appears inverted on your footage, set INVERT_DEPTH=1
    # in the environment and the line below will flip the map.
    import os
    if os.getenv("INVERT_DEPTH", "0") == "1":
        depth_np = depth_np.max() - depth_np  # flip near/far

    d_min, d_max = float(depth_np.min()), float(depth_np.max())
    if d_max - d_min > 1e-6:
        depth_normalized = (depth_np - d_min) / (d_max - d_min)
    else:
        depth_normalized = np.zeros_like(depth_np)

    logger.debug(
        "Depth map: shape=%s  min=%.3f  max=%.3f",
        depth_normalized.shape, depth_normalized.min(), depth_normalized.max(),
    )
    return depth_normalized
