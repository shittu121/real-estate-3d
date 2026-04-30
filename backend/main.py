"""
Real Estate Video Generator — FastAPI backend.

Job lifecycle
─────────────
  POST /upload  → creates job directory, enqueues job, returns {job_id}
  GET /status/{job_id} → reads jobs/{job_id}/status.json
  GET /download/{job_id} → streams jobs/{job_id}/output.mp4
  GET /health   → liveness probe

Supports bulk uploads: up to MAX_IMAGES photos processed in sequence and
joined into one video with crossfade transitions.

Jobs run one-at-a-time in a thread-pool executor (single GPU worker) so the
asyncio event loop is never blocked by CPU/GPU-intensive work.
"""

import asyncio
import json
import logging
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from backend.pipeline import depth as depth_mod
from backend.pipeline import dof as dof_mod
from backend.pipeline import grade as grade_mod
from backend.pipeline import inpaint as inpaint_mod
from backend.pipeline import kenburns as kenburns_mod
from backend.pipeline import video as video_mod
from backend.pipeline import warp as warp_mod

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

JOBS_DIR    = Path(os.getenv("JOBS_DIR", "/jobs"))
FPS         = 30
MAX_IMAGES  = 20       # maximum photos per job
CROSSFADE_S = 0.5      # crossfade duration between clips (seconds)

RESOLUTIONS: Dict[str, tuple] = {
    "720p":  (1280, 720),
    "1080p": (1920, 1080),
}

VALID_CAMERA_MODES = warp_mod.CAMERA_MODES
VALID_PAN_DIRS     = kenburns_mod.PAN_DIRECTIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ─────────────────────────────────────────────────────────────────────────────
# Job queue + worker
# ─────────────────────────────────────────────────────────────────────────────

_job_queue: asyncio.Queue = asyncio.Queue()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-worker")


def _write_status(
    job_dir: Path,
    status: str,
    progress: int = 0,
    message: str = "",
) -> None:
    payload = {"status": status, "progress": progress, "message": message}
    tmp = job_dir / "status.tmp"
    tmp.write_text(json.dumps(payload))
    tmp.replace(job_dir / "status.json")


def _load_image(path: Path) -> tuple:
    pil = Image.open(path).convert("RGB")
    max_pixels = 24_000_000
    w, h = pil.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        logger.info("Input image downscaled to %dx%d (was %dx%d)", *pil.size, w, h)
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return pil, bgr


def _process_one(
    pil_image,
    image_bgr,
    mode: str,
    camera_mode: str,
    pan_direction: str,
    num_frames: int,
    out_res: tuple,
    progress_fn,
) -> List[np.ndarray]:
    """Run the full animation pipeline for a single image."""
    if mode == "kenburns":
        progress_fn(0.20, "Generating Ken Burns animation…")
        frames = kenburns_mod.generate_frames(
            image=image_bgr,
            num_frames=num_frames,
            pan_direction=pan_direction,
            output_resolution=out_res,
        )
    else:
        # ── 3D Parallax pipeline ──────────────────────────────────────────────
        progress_fn(0.10, "Estimating scene depth…")
        depth_map = depth_mod.estimate_depth(pil_image)

        progress_fn(0.40, "Computing camera movement…")
        frames_raw, masks = warp_mod.generate_frames(
            image=image_bgr,
            depth_map=depth_map,
            camera_mode=camera_mode,
            num_frames=num_frames,
            output_resolution=out_res,
        )

        progress_fn(0.70, "Filling parallax holes…")
        frames = [inpaint_mod.inpaint_frame(f, m) for f, m in zip(frames_raw, masks)]

        progress_fn(0.82, "Applying depth of field…")
        frames = [dof_mod.apply_dof(f, depth_map) for f in frames]

    progress_fn(0.90, "Colour grading…")
    frames = [grade_mod.apply_grade(f) for f in frames]

    return frames


def _run_job(job_id: str, job_dir: Path, params: Dict[str, Any]) -> None:
    """
    Blocking job runner — executed inside the thread-pool executor.
    Processes every uploaded image, writes a per-image clip, then joins them.
    """
    try:
        mode          = params.get("mode", "parallax")
        duration      = max(1, min(int(params.get("duration", 5)), 30))
        resolution    = params.get("resolution", "1080p")
        camera_mode   = params.get("camera_mode", "dolly_in")
        pan_direction = params.get("pan_direction", "right")
        image_count   = int(params.get("image_count", 1))

        out_res     = RESOLUTIONS.get(resolution, RESOLUTIONS["1080p"])
        num_frames  = FPS * duration
        # Extra frames added to every clip except the last so the xfade
        # crossfade consumes the padding rather than eating into the user's
        # requested duration.  Total video = image_count × duration exactly.
        crossfade_pad = int(FPS * CROSSFADE_S)
        output_path = job_dir / "output.mp4"
        clip_paths: List[Path] = []

        for idx in range(image_count):
            # Each image owns a proportional slice of 0–85 % of total progress
            prog_lo   = int(idx * 85 / image_count)
            prog_span = max(int(85 / image_count), 1)
            label     = f"Photo {idx + 1}/{image_count}"

            def prog(frac, msg, _lo=prog_lo, _span=prog_span, _lbl=label):
                _write_status(job_dir, "processing",
                              _lo + int(frac * _span), f"{_lbl} — {msg}")

            prog(0.02, "Loading image…")
            pil_image, image_bgr = _load_image(job_dir / f"input_{idx}.jpg")

            is_last = (idx == image_count - 1)
            nf = num_frames if (image_count == 1 or is_last) else num_frames + crossfade_pad

            frames = _process_one(
                pil_image, image_bgr, mode, camera_mode, pan_direction,
                nf, out_res, prog,
            )

            clip_path = job_dir / f"clip_{idx}.mp4"
            prog(0.93, "Encoding clip…")
            video_mod.write_video(frames, clip_path, fps=FPS)
            clip_paths.append(clip_path)

        if image_count == 1:
            _write_status(job_dir, "processing", 90, "Finalising video…")
            shutil.move(str(clip_paths[0]), str(output_path))
        else:
            _write_status(job_dir, "processing", 87,
                          f"Joining {image_count} clips with transitions…")
            video_mod.concat_videos(
                clip_paths, output_path,
                fps=FPS,
                crossfade_secs=CROSSFADE_S,
                clip_duration_secs=float(duration),
            )
            for p in clip_paths:
                p.unlink(missing_ok=True)

        _write_status(job_dir, "done", 100, "Your video is ready!")
        logger.info("Job %s completed successfully.", job_id)

    except Exception as exc:
        logger.exception("Job %s failed.", job_id)
        _write_status(job_dir, "error", 0, _friendly_error(exc))


def _friendly_error(exc: Exception) -> str:
    msg = str(exc)
    if "CUDA out of memory" in msg:
        return "GPU ran out of memory. Try a shorter duration or 720p resolution."
    if "FFmpeg" in msg or "FileNotFoundError" in msg:
        return "Video encoding failed. Make sure FFmpeg is installed on the server."
    if "No such file" in msg:
        return "Upload file was not found on disk. Please try uploading again."
    return f"Processing failed: {msg[:300]}"


async def _worker() -> None:
    loop = asyncio.get_event_loop()
    logger.info("GPU worker started — waiting for jobs.")
    while True:
        job_id, job_dir, params = await _job_queue.get()
        logger.info("Dequeued job %s", job_id)
        try:
            await loop.run_in_executor(_executor, _run_job, job_id, job_dir, params)
        except Exception:
            logger.exception("Unexpected dispatch error for job %s", job_id)
        finally:
            _job_queue.task_done()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    worker_task = asyncio.create_task(_worker())
    logger.info("Server ready.  Jobs dir: %s", JOBS_DIR)
    yield
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Real Estate Video Generator",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload(
    files: List[UploadFile] = File(...),
    mode: str = Form("parallax"),
    camera_mode: str = Form("dolly_in"),
    pan_direction: str = Form("right"),
    duration: int = Form(5),
    resolution: str = Form("1080p"),
):
    """
    Accept one or more image uploads and enqueue a generation job.
    Returns {"job_id": "<uuid>"}.
    """
    if not files:
        raise HTTPException(status_code=422, detail="At least one image is required.")
    if len(files) > MAX_IMAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Maximum {MAX_IMAGES} photos per job. You sent {len(files)}.",
        )
    for i, f in enumerate(files):
        if not f.content_type or not f.content_type.startswith("image/"):
            raise HTTPException(
                status_code=422,
                detail=f"File {i + 1} is not an image. Only JPEG, PNG, WEBP, etc. are supported.",
            )
    if mode not in ("parallax", "kenburns"):
        raise HTTPException(status_code=422, detail=f"Unknown mode: {mode}")
    if mode == "parallax" and camera_mode not in VALID_CAMERA_MODES:
        raise HTTPException(status_code=422, detail=f"Unknown camera_mode: {camera_mode}")
    if mode == "kenburns" and pan_direction not in VALID_PAN_DIRS:
        raise HTTPException(status_code=422, detail=f"Unknown pan_direction: {pan_direction}")
    if resolution not in RESOLUTIONS:
        raise HTTPException(status_code=422, detail=f"Unknown resolution: {resolution}")
    if not (1 <= duration <= 30):
        raise HTTPException(status_code=422, detail="Duration must be 1–30 seconds.")

    job_id  = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    for idx, f in enumerate(files):
        image_bytes = await f.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=422, detail=f"Photo {idx + 1} is empty.")
        (job_dir / f"input_{idx}.jpg").write_bytes(image_bytes)

    params = {
        "mode": mode,
        "camera_mode": camera_mode,
        "pan_direction": pan_direction,
        "duration": duration,
        "resolution": resolution,
        "image_count": len(files),
    }
    (job_dir / "params.json").write_text(json.dumps(params, indent=2))
    _write_status(job_dir, "queued", 0, "Job queued — waiting for GPU…")

    await _job_queue.put((job_id, job_dir, params))
    logger.info(
        "Job %s queued  images=%d  mode=%s  camera=%s  dur=%ds  res=%s",
        job_id, len(files), mode, camera_mode, duration, resolution,
    )
    return JSONResponse({"job_id": job_id})


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    status_file = JOBS_DIR / job_id / "status.json"
    if not status_file.exists():
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(json.loads(status_file.read_text()))


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    job_dir  = JOBS_DIR / job_id
    out_path = job_dir / "output.mp4"

    if not out_path.exists():
        status_file = job_dir / "status.json"
        if status_file.exists():
            st = json.loads(status_file.read_text())
            if st["status"] != "done":
                raise HTTPException(
                    status_code=409,
                    detail=f"Job is not finished yet (status: {st['status']}).",
                )
        raise HTTPException(status_code=404, detail="Output video not found.")

    return FileResponse(
        path=str(out_path),
        media_type="video/mp4",
        filename="real-estate-video.mp4",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "queue_depth": _job_queue.qsize()}
