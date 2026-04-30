#!/usr/bin/env bash
# start.sh — One-command startup for the Real Estate Video Generator.
# Run from the project root:  bash start.sh
set -euo pipefail

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║      Real Estate Video Generator  v1.0           ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── 1. Create required directories ───────────────────────────────────────────
mkdir -p /jobs
echo "✓  Jobs directory: /jobs"

# ── 2. Report GPU status ──────────────────────────────────────────────────────
python3 - <<'PYEOF'
import sys
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓  GPU detected: {name}  ({vram:.1f} GB VRAM)")
    else:
        print("⚠  No CUDA GPU detected — running on CPU (significantly slower).")
except ImportError:
    print("✗  PyTorch not installed. Run:  pip install -r backend/requirements.txt")
    sys.exit(1)
PYEOF

# ── 3. Pre-load and warm-up the depth model ───────────────────────────────────
echo ""
echo "Loading Depth Anything V2 Small …"
echo "(First run downloads ~100 MB from HuggingFace — subsequent runs are instant)"
echo ""

python3 - <<'PYEOF'
import sys, os
try:
    import torch
    from transformers import pipeline
    from PIL import Image

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )
    # Warm-up forward pass so the first real job starts instantly
    dummy = Image.new("RGB", (224, 224), color=(100, 120, 140))
    pipe(dummy)
    print("✓  Depth model loaded and warmed up.")

except Exception as e:
    print(f"⚠  Model pre-load failed: {e}")
    print("   The model will be loaded on the first request instead.")
PYEOF

# ── 4. Start the FastAPI server ───────────────────────────────────────────────
echo ""
echo "Starting server …"
echo ""
echo "  Local access:   http://localhost:8000"
echo "  Remote access:  http://<your-instance-ip>:8000"
echo "  Health check:   http://localhost:8000/health"
echo ""
echo "Open  frontend/index.html  in your browser to use the web portal."
echo "On vast.ai, serve the HTML from any static host or open it locally"
echo "and point it at your instance IP (edit API_BASE in index.html)."
echo ""
echo "Press Ctrl+C to stop."
echo ""

exec uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info
