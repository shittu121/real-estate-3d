"""
3D parallax warp engine — single-pass backward remap.

Camera motion formula (per mode):
  displacement = BASE + PARALLAX * proximity

  BASE     — the part every pixel shares regardless of depth
             (global camera translation/zoom)
  PARALLAX — the extra amount foreground moves over background
             (depth-differential, small)

This separation is critical for dolly_in:
  Old (broken): scale = 1 + zoom * t * prox
    → background barely zooms, foreground zooms a lot → scene distorts
  New (correct): scale = (1 + base_zoom * t) + parallax_zoom * t * prox
    → everything zooms the same base amount, foreground just slightly more
    → looks like a real camera pushing forward

Depth convention: 0.0 = nearest (foreground), 1.0 = farthest (background).
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CAMERA_MODES: Tuple[str, ...] = (
    "dolly_in",
    "pan_right",
    "pan_left",
    "orbit_right",
    "crane_up",
)

# Each mode splits its motion into a BASE component (global, all depths move)
# and a PARALLAX component (foreground moves this much MORE than background).
# Keep parallax values small — revealing hidden space looks bad on a 2D photo.
_PARAMS = {
    "dolly_in": {
        "base_zoom":      0.30,   # strong forward push — clearly visible zoom
        "parallax_zoom":  0.14,   # foreground pushes extra — real depth cue
    },
    "pan_right": {
        "base_shift":     0.10,   # wide lateral sweep
        "parallax_shift": 0.10,   # foreground leads the pan
        "base_zoom":      0.12,
        "parallax_zoom":  0.06,
    },
    "pan_left": {
        "base_shift":     0.10,
        "parallax_shift": 0.10,
        "base_zoom":      0.12,
        "parallax_zoom":  0.06,
    },
    "orbit_right": {
        "base_shift":     0.08,
        "parallax_shift": 0.09,
        "base_zoom":      0.16,
        "parallax_zoom":  0.08,
    },
    "crane_up": {
        "base_shift":     0.09,   # noticeable vertical lift
        "parallax_shift": 0.09,
        "base_zoom":      0.12,
        "parallax_zoom":  0.06,
    },
}


def _ease(t: float) -> float:
    """Cubic ease-in-out: slow start, smooth peak, slow end."""
    return t * t * (3.0 - 2.0 * t)


# Secondary camera sway — uniform drift layered on top of the primary move.
# Gives the "operator on a slow dolly track" feeling.
_SWAY_AMP_X  = 0.022   # ±2.2 % of width  — visible left/right camera float
_SWAY_AMP_Y  = 0.009   # ±0.9 % of height — subtle breathing
_SWAY_CYCLES = 0.75    # ~¾ of a left→right→left cycle over the clip


def _sway(t_raw: float, w: int, h: int) -> Tuple[float, float]:
    """Return (dx, dy) pixel offset for the camera sway at raw frame progress."""
    angle = 2.0 * np.pi * _SWAY_CYCLES * t_raw
    dx = _SWAY_AMP_X * w * np.sin(angle)
    dy = _SWAY_AMP_Y * h * np.sin(angle * 0.55)   # slightly different freq
    return dx, dy


def generate_frames(
    image: np.ndarray,
    depth_map: np.ndarray,
    camera_mode: str,
    num_frames: int,
    output_resolution: Tuple[int, int],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if camera_mode not in CAMERA_MODES:
        raise ValueError(
            f"Unknown camera_mode '{camera_mode}'. Valid: {CAMERA_MODES}"
        )

    h, w = image.shape[:2]
    out_w, out_h = output_resolution

    depth = cv2.resize(
        depth_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
    )

    # Proximity map — light blur only to avoid single-pixel seams.
    # The bilateral filter in depth.py already sharpened object boundaries,
    # so we don't need a heavy blur here.
    prox = cv2.GaussianBlur((1.0 - depth).astype(np.float32), (0, 0), 4.0)
    prox = np.clip(prox, 0.0, 1.0)

    # Coordinate grids (computed once, reused every frame)
    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.5, h * 0.5

    # Over-crop buffer: resize to 8 % larger then center-crop so reflected
    # border pixels are always outside the visible area.
    _BUF = 1.08
    buf_w = int(out_w * _BUF)
    buf_h = int(out_h * _BUF)
    x0 = (buf_w - out_w) // 2
    y0 = (buf_h - out_h) // 2

    frames: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    empty_mask = np.zeros((out_h, out_w), dtype=np.uint8)

    p = _PARAMS[camera_mode]

    for i in range(num_frames):
        t_raw = i / max(num_frames - 1, 1)
        t = _ease(t_raw)
        sway_x, sway_y = _sway(t_raw, w, h)

        # ── Build displacement maps ────────────────────────────────────────────
        if camera_mode == "dolly_in":
            base_scale = 1.0 + p["base_zoom"] * t
            scale = base_scale + p["parallax_zoom"] * t * prox
            map_x = cx + (x_grid - cx) / scale + sway_x
            map_y = cy + (y_grid - cy) / scale + sway_y

        elif camera_mode == "pan_right":
            base_shift   = w * p["base_shift"]   * t
            para_shift   = w * p["parallax_shift"] * t * prox
            total_shift  = base_shift + para_shift
            base_scale   = 1.0 + p["base_zoom"] * t
            scale        = base_scale + p["parallax_zoom"] * t * prox
            map_x = cx + (x_grid - cx) / scale + total_shift + sway_x
            map_y = cy + (y_grid - cy) / scale + sway_y

        elif camera_mode == "pan_left":
            base_shift   = w * p["base_shift"]   * t
            para_shift   = w * p["parallax_shift"] * t * prox
            total_shift  = base_shift + para_shift
            base_scale   = 1.0 + p["base_zoom"] * t
            scale        = base_scale + p["parallax_zoom"] * t * prox
            map_x = cx + (x_grid - cx) / scale - total_shift + sway_x
            map_y = cy + (y_grid - cy) / scale + sway_y

        elif camera_mode == "orbit_right":
            base_shift   = w * p["base_shift"]   * t
            para_shift   = w * p["parallax_shift"] * t * prox
            total_shift  = base_shift + para_shift
            base_scale   = 1.0 + p["base_zoom"] * t
            scale        = base_scale + p["parallax_zoom"] * t * prox
            map_x = cx + (x_grid - cx) / scale + total_shift + sway_x
            map_y = cy + (y_grid - cy) / scale + sway_y

        elif camera_mode == "crane_up":
            base_shift   = h * p["base_shift"]   * t
            para_shift   = h * p["parallax_shift"] * t * prox
            total_shift  = base_shift + para_shift
            base_scale   = 1.0 + p["base_zoom"] * t
            scale        = base_scale + p["parallax_zoom"] * t * prox
            map_x = cx + (x_grid - cx) / scale + sway_x
            map_y = cy + (y_grid - cy) / scale + total_shift + sway_y

        # ── Remap ─────────────────────────────────────────────────────────────
        # cv2.remap requires float32 maps exactly — cast in case any scalar
        # arithmetic (sway offsets etc.) promoted the arrays to float64.
        frame = cv2.remap(
            image,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        frame_buf = cv2.resize(frame, (buf_w, buf_h), interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame_buf[y0:y0 + out_h, x0:x0 + out_w])
        masks.append(empty_mask)

    logger.info(
        "Warp: %d frames — mode=%s  out=%dx%d",
        num_frames, camera_mode, out_w, out_h,
    )
    return frames, masks
