# Real Estate Video Generator

Turn a single property photo into a smooth, cinematic video with 3D depth-parallax
camera movement or classic Ken Burns pan + zoom — everything runs on your own GPU, no subscriptions.

---

## How it works

| Step | What happens |
|------|-------------|
| Upload | Client uploads a JPEG/PNG room photo |
| Depth | Depth Anything V2 (Small) estimates a per-pixel depth map |
| Warp  | OpenCV remaps pixels frame-by-frame using depth-based displacement |
| Inpaint | TELEA fast-marching fills edge holes created by the camera movement |
| Encode | FFmpeg writes an H.264 MP4 at 30 fps |

---

## Provisioning a vast.ai GPU instance

1. Go to [vast.ai](https://vast.ai) and create an account.
2. Click **Create** → **Instance** and search for a template:
   - **Recommended GPU:** RTX 3090 (24 GB VRAM) or A5000 (24 GB VRAM)
   - **Minimum:** RTX 3080 (10 GB VRAM) — works for 720p, slower for 1080p
   - **Template:** search for `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
     (or use "PyTorch" pre-built templates and adjust the Dockerfile base)
3. Set **disk space** to at least **30 GB** (model cache + job storage).
4. Under **Port Forwarding**, expose port `8000`.
5. Click **Rent** and wait for the instance to start.
6. Click the **SSH** button to connect, or use the provided SSH command.

---

## Setup and first run

uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info


```bash
# 1. Clone the repository onto the instance
git clone https://github.com/your-username/real-estate-3d.git
cd real-estate-3d

# 2. (Docker — recommended on vast.ai)
docker build -t re-video .
docker run --gpus all -p 8000:8000 -v /jobs:/jobs re-video

# ── OR ──────────────────────────────────────────────────────────────────────

# 2. (Bare metal / direct Python)
pip install -r backend/requirements.txt
# Install GPU-enabled PyTorch (match your CUDA version):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Start the server:
bash start.sh
```

The first run downloads the Depth Anything V2 Small model (~100 MB) from
HuggingFace and caches it under `~/.cache/huggingface/`. All subsequent
starts are instant.

---

## Accessing the web portal

### Option A — Open the HTML file directly (simplest)

1. Download `frontend/index.html` to your local machine.
2. Open it in any browser.
3. At the top of the `<script>` block, change `API_BASE`:

```js
// Before (localhost):
const API_BASE = ...

// After (vast.ai remote):
const API_BASE = "http://YOUR_INSTANCE_IP:8000";
```

4. Drag in a photo and click **Generate Video**.

### Option B — Serve via nginx/caddy on the instance

```bash
# Serve the frontend on port 80 alongside the API on 8000
docker run -d -p 80:80 -v $(pwd)/frontend:/usr/share/nginx/html:ro nginx:alpine
```

Then open `http://YOUR_INSTANCE_IP` in a browser.  The `API_BASE` auto-detection
in `index.html` will point correctly to port 8000 on the same host.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JOBS_DIR` | `/jobs` | Directory where job files are stored |
| `INVERT_DEPTH` | `0` | Set to `1` if parallax direction looks backwards |

---

## Camera modes (3D Parallax)

| Mode | Effect |
|------|--------|
| `dolly_in` | Camera pushes forward; near objects zoom in more than far |
| `pan_right` | Camera slides right; foreground shifts left more than background |
| `pan_left` | Mirror of pan_right |
| `orbit_right` | Rightward arc with depth-varying zoom for perspective feel |
| `crane_up` | Camera rises; foreground rises faster than background |

---

## Swapping in LaMa inpainting (optional upgrade)

The inpainter in `backend/pipeline/inpaint.py` is intentionally thin.
To plug in LaMa for higher-quality hole filling:

1. Install LaMa inference (`pip install lama-cleaner` or clone the repo).
2. Replace the body of `inpaint_frame()` with a call to your LaMa wrapper.
3. No other files need changing.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUDA out of memory` | Use 720p resolution or reduce duration to ≤ 5 s |
| Parallax moves backwards | Set `INVERT_DEPTH=1` in your environment |
| FFmpeg not found | `apt-get install ffmpeg` on Ubuntu |
| HuggingFace download fails | `export HF_HUB_OFFLINE=0` and check internet access |
| Browser can't reach API | Check port 8000 is open in vast.ai port settings |

---

## Project layout

```
real-estate-3d/
├── backend/
│   ├── main.py              FastAPI server, job queue, REST endpoints
│   ├── pipeline/
│   │   ├── depth.py         Depth Anything V2 inference
│   │   ├── warp.py          OpenCV depth-based mesh warp
│   │   ├── inpaint.py       TELEA hole-filling (swappable)
│   │   ├── kenburns.py      Classic zoom + pan
│   │   └── video.py         FFmpeg H.264 encoding
│   └── requirements.txt
├── frontend/
│   └── index.html           Single-file web portal
├── Dockerfile
├── start.sh
└── README.md
```
