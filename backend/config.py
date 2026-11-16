"""
config.py
---------
Centralized configuration for the Crowd Counting backend.
Keeping these values in one place makes thresholds, paths, and limits
easy to tune without hunting through the codebase.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "models", "csrnet.pth")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DENSITY_MAP_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "density_maps")
PROCESSED_VIDEO_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "processed_videos")

for folder in (UPLOAD_FOLDER, DENSITY_MAP_FOLDER, PROCESSED_VIDEO_FOLDER):
    os.makedirs(folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Upload validation / security
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
MAX_CONTENT_LENGTH_MB = 200
MAX_CONTENT_LENGTH = MAX_CONTENT_LENGTH_MB * 1024 * 1024  # bytes

# ---------------------------------------------------------------------------
# Video / frame sampling
# ---------------------------------------------------------------------------
# Process 1 out of every N frames. Raising this speeds up processing on
# CPU-only machines at the cost of temporal resolution.
FRAME_SAMPLE_INTERVAL = int(os.environ.get("FRAME_SAMPLE_INTERVAL", 60))

# Hard cap on number of frames analyzed per video (safety limit so a very
# long video can't hang the server for an unbounded amount of time).
MAX_FRAMES_TO_PROCESS = int(os.environ.get("MAX_FRAMES_TO_PROCESS", 150))

# Resize frames to this size before feeding to CSRNet. CSRNet is fully
# convolutional so it can technically accept any size, but capping the
# size keeps inference time and memory usage predictable, especially on
# CPU-only machines.
FRAME_RESIZE_MAX_DIM = 640

# ---------------------------------------------------------------------------
# Crowd density thresholds (people count -> density level)
# These are configurable and should be tuned to the camera's field of view
# and typical monitored area for a real deployment.
# ---------------------------------------------------------------------------
DENSITY_THRESHOLDS = {
    "LOW": 20,       # count <= 20  -> LOW
    "MEDIUM": 50,     # 20 < count <= 50 -> MEDIUM
    "HIGH": 100,      # 50 < count <= 100 -> HIGH
    # count > 100 -> CRITICAL
}

# Mapping from density level to a human-readable safety status
SAFETY_STATUS_MAP = {
    "LOW": "SAFE",
    "MEDIUM": "MODERATE",
    "HIGH": "WARNING",
    "CRITICAL": "EMERGENCY",
}

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
