"""
Video assembly: pipe raw BGR frames into FFmpeg and write an H.264 MP4.

Using FFmpeg directly (via subprocess) gives reliable H.264 output with
browser-compatible yuv420p pixel format and faststart moov atom for
streaming/preview in the frontend <video> element.
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)


def write_video(
    frames: List[np.ndarray],
    output_path: Union[str, Path],
    fps: int = 30,
    crf: int = 18,
) -> Path:
    """
    Write a list of BGR uint8 frames to an H.264 MP4 file.

    Args:
        frames:      Non-empty list of BGR uint8 numpy arrays (all same size).
        output_path: Destination .mp4 path.
        fps:         Frames per second.
        crf:         H.264 quality (0 = lossless, 51 = worst; 18 ≈ visually lossless).

    Returns:
        Path to the written file.

    Raises:
        RuntimeError: if the frame list is empty or FFmpeg exits non-zero.
    """
    if not frames:
        raise RuntimeError("write_video: frame list is empty.")

    output_path = Path(output_path)
    h, w = frames[0].shape[:2]

    cmd = [
        "ffmpeg", "-y",
        # ── Input: raw BGR frames piped to stdin ──────────────────────────
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        # ── Output: H.264 MP4 ─────────────────────────────────────────────
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",     # required for browser / QuickTime compat
        "-crf", str(crf),
        "-preset", "fast",
        "-movflags", "+faststart",  # put moov atom at front for streaming
        str(output_path),
    ]

    logger.info(
        "FFmpeg encode: %d frames  %dx%d @ %dfps  crf=%d → %s",
        len(frames), w, h, fps, crf, output_path,
    )

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if frame.shape[:2] != (h, w):
                import cv2
                frame = cv2.resize(frame, (w, h))
            proc.stdin.write(frame.tobytes())

        proc.stdin.close()
        _, stderr = proc.communicate(timeout=600)

        if proc.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed (exit {proc.returncode}):\n"
                + stderr.decode(errors="replace")[-2000:]
            )

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise RuntimeError("FFmpeg timed out (>10 min).  Try a shorter duration or lower resolution.")

    size_kb = output_path.stat().st_size / 1024
    logger.info("Video ready: %s  (%.0f KB)", output_path, size_kb)
    return output_path


def concat_videos(
    clip_paths: List[Path],
    output_path: Union[str, Path],
    fps: int = 30,
    crossfade_secs: float = 0.5,
    clip_duration_secs: float = 5.0,
) -> Path:
    """
    Concatenate multiple MP4 clips into one video with crossfade transitions.

    Uses FFmpeg's xfade filter.  All clips must share the same resolution and fps.

    Args:
        clip_paths:          Ordered list of input .mp4 paths.
        output_path:         Destination .mp4 path.
        fps:                 Frame rate (for logging).
        crossfade_secs:      Duration of the crossfade between each pair of clips.
        clip_duration_secs:  Duration of each individual clip in seconds.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    n = len(clip_paths)

    if n == 0:
        raise RuntimeError("concat_videos: clip list is empty.")

    # Build FFmpeg inputs
    cmd = ["ffmpeg", "-y"]
    for p in clip_paths:
        cmd += ["-i", str(p)]

    # Build xfade filter chain.
    # offset[i] = i * (clip_duration - crossfade_duration)
    # so each subsequent crossfade starts exactly at the end of the previous combined segment.
    step = clip_duration_secs - crossfade_secs
    parts = []
    prev = "[0:v]"
    for i in range(1, n):
        tag = "[vout]" if i == n - 1 else f"[v{i}]"
        offset = i * step
        parts.append(
            f"{prev}[{i}:v]xfade=transition=fade"
            f":duration={crossfade_secs:.3f}"
            f":offset={offset:.3f}{tag}"
        )
        prev = tag

    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", "[vout]",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.info(
        "Concatenating %d clips  crossfade=%.1fs → %s", n, crossfade_secs, output_path
    )

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _, stderr = proc.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise RuntimeError("FFmpeg concat timed out (>10 min).")

    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg concat failed (exit {proc.returncode}):\n"
            + stderr.decode(errors="replace")[-2000:]
        )

    size_kb = output_path.stat().st_size / 1024
    logger.info("Joined video ready: %s  (%.0f KB)", output_path, size_kb)
    return output_path
