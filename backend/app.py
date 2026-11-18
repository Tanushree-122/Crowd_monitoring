"""
Flask REST API for the Crowd Counting Surveillance System.

Endpoints:
    POST /api/upload   - upload a video file
    POST /api/analyze  - run CSRNet analysis
    GET  /api/status   - get analysis status
    POST /api/reset   - reset analysis state
    POST /api/stop    - request cancellation between frames
    GET  /api/health  - health check
"""

import os
import logging
import traceback
import threading
import uuid

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import config
from inference import process_video, load_model, ModelLoadError, AnalysisCancelledError
from crowd_analyzer import analyze_counts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
@app.route("/")
def home():
    return "Visora Crowd Counting API is running!"
# Allow the frontend (served from a different port, e.g. via `python -m
# http.server` or Live Server) to call this API.
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER

# ---------------------------------------------------------------------------
# In-memory analysis state.
# For an academic/local project a simple in-process dict is sufficient;
# a production system would use a proper job queue (Celery/RQ) + database.
# ---------------------------------------------------------------------------
_STATE_LOCK = threading.Lock()
_CANCEL_EVENT = threading.Event()
_STATE = {
    "status": "IDLE",           # IDLE | PROCESSING | DONE | ERROR | CANCELLED
    "progress": 0,               # frames processed so far
    "video_name": None,
    "error": None,
    "result": None,              # last full analysis result dict
}


def _reset_state():
    with _STATE_LOCK:
        _STATE.update({
            "status": "IDLE",
            "progress": 0,
            "video_name": None,
            "error": None,
            "result": None,
        })


def _allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


@app.route("/api/health", methods=["GET"])
def health():
    """Basic health check: confirms server is up and reports device/model status."""
    model_status = "not_loaded"
    device = "unknown"
    try:
        model = load_model()
        model_status = "loaded"
        device = str(next(model.parameters()).device)
    except ModelLoadError as exc:
        model_status = f"missing_weights: {exc}"
    except Exception as exc:  # pragma: no cover - defensive
        model_status = f"error: {exc}"

    return jsonify({
        "server": "ok",
        "model_status": model_status,
        "device": device,
    })


@app.route("/api/upload", methods=["POST"])
def upload():
    """
    Accepts a video file via multipart/form-data (field name: "video"),
    validates it, saves it securely, and returns a video_id + filename
    that can later be passed to /api/analyze.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file part in the request."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No video selected."}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "error": (
                f"Unsupported file type. Allowed extensions: "
                f"{', '.join(sorted(config.ALLOWED_EXTENSIONS))}"
            )
        }), 400

    safe_name = secure_filename(file.filename)
    video_id = uuid.uuid4().hex[:12]
    stored_name = f"{video_id}_{safe_name}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)

    try:
        file.save(save_path)
    except Exception as exc:
        logger.exception("Failed to save uploaded video")
        return jsonify({"error": f"Failed to save uploaded video: {exc}"}), 500

    if os.path.getsize(save_path) == 0:
        os.remove(save_path)
        return jsonify({"error": "Uploaded video file is empty."}), 400

    return jsonify({
        "message": "Video uploaded successfully.",
        "video_id": video_id,
        "stored_filename": stored_name,
        "original_filename": safe_name,
    })


def _run_analysis(video_path: str, video_name: str):
    """Runs CSRNet analysis on a background thread and updates _STATE."""
    def progress_cb(processed, total, latest):
        with _STATE_LOCK:
            _STATE["progress"] = processed

    try:
        with _STATE_LOCK:
            _STATE["status"] = "PROCESSING"
            _STATE["video_name"] = video_name
            _STATE["error"] = None

        video_result = process_video(
            video_path,
            progress_callback=progress_cb,
            cancel_event=_CANCEL_EVENT,
        )
        analytics = analyze_counts(video_result["frame_counts"])

        full_result = {
            **analytics,
            "frames_processed": video_result["frames_processed"],
            "processing_fps": video_result["processing_fps"],
            "video_fps": video_result["video_fps"],
            "video_duration_sec": video_result["video_duration_sec"],
            "total_frames_in_video": video_result["total_frames_in_video"],
            "density_map_path": video_result["density_map_path"],
            "video_name": video_name,
            "model_used": "CSRNet",
            "frame_sample_interval": config.FRAME_SAMPLE_INTERVAL,
        }

        with _STATE_LOCK:
            if _CANCEL_EVENT.is_set():
                _STATE["status"] = "CANCELLED"
                _STATE["error"] = "Analysis was stopped by the user."
                return
            _STATE["status"] = "DONE"
            _STATE["result"] = full_result

    except AnalysisCancelledError as exc:
        logger.info("Analysis cancelled: %s", exc)
        with _STATE_LOCK:
            _STATE["status"] = "CANCELLED"
            _STATE["error"] = str(exc)
    except (ModelLoadError, FileNotFoundError, IOError, ValueError) as exc:
        logger.error("Analysis failed: %s", exc)
        with _STATE_LOCK:
            _STATE["status"] = "ERROR"
            _STATE["error"] = str(exc)
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Unexpected error during analysis")
        with _STATE_LOCK:
            _STATE["status"] = "ERROR"
            _STATE["error"] = f"Unexpected error: {exc}"


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Runs CSRNet crowd analysis on an uploaded video.

    Accepts EITHER:
      - multipart/form-data with a fresh "video" file field (uploads +
        analyzes in one call), OR
      - JSON/form field "stored_filename" referring to a file already
        saved via /api/upload.

    This call runs synchronously (blocking) and returns the full result
    as JSON, matching the required response shape. For large videos this
    may take a while on CPU; frame sampling (config.FRAME_SAMPLE_INTERVAL)
    and MAX_FRAMES_TO_PROCESS keep this bounded.
    """
    video_path = None
    video_name = None

    if "video" in request.files and request.files["video"].filename != "":
        file = request.files["video"]
        if not _allowed_file(file.filename):
            return jsonify({
                "error": (
                    f"Unsupported file type. Allowed extensions: "
                    f"{', '.join(sorted(config.ALLOWED_EXTENSIONS))}"
                )
            }), 400
        safe_name = secure_filename(file.filename)
        stored_name = f"{uuid.uuid4().hex[:12]}_{safe_name}"
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
        file.save(video_path)
        video_name = safe_name
    else:
        stored_filename = request.form.get("stored_filename") or (
            request.json.get("stored_filename") if request.is_json else None
        )
        if not stored_filename:
            return jsonify({
                "error": "No video uploaded and no stored_filename provided."
            }), 400
        video_path = os.path.join(
            app.config["UPLOAD_FOLDER"], secure_filename(stored_filename)
        )
        video_name = stored_filename

    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found on server."}), 404

    if os.path.getsize(video_path) == 0:
        return jsonify({"error": "Video file is empty."}), 400

    with _STATE_LOCK:
        if _STATE["status"] == "PROCESSING":
            return jsonify({
                "error": "An analysis is already in progress. "
                         "Call /api/reset or wait for it to finish."
            }), 409
        _CANCEL_EVENT.clear()

    # Run synchronously so the HTTP response directly contains the real
    # CSRNet-derived results (simplest, most robust approach for a local
    # academic project; see README for notes on making this async).
    _run_analysis(video_path, video_name)

    with _STATE_LOCK:
        if _STATE["status"] == "CANCELLED":
            return jsonify({"error": _STATE["error"], "status": "CANCELLED"}), 409
        if _STATE["status"] == "ERROR":
            return jsonify({"error": _STATE["error"]}), 500
        result = _STATE["result"]

    return jsonify(result)


@app.route("/api/status", methods=["GET"])
def status():
    """Returns the current analysis status/progress for polling-based UIs."""
    with _STATE_LOCK:
        return jsonify({
            "status": _STATE["status"],
            "progress": _STATE["progress"],
            "video_name": _STATE["video_name"],
            "error": _STATE["error"],
        })


@app.route("/api/reset", methods=["POST"])
def reset():
    """Clears server-side analysis state so the dashboard can start fresh."""
    _reset_state()
    return jsonify({"message": "State reset successfully."})


@app.route("/api/stop", methods=["POST"])
def stop():
    """Request cancellation between frames in the active CSRNet analysis."""
    with _STATE_LOCK:
        if _STATE["status"] != "PROCESSING":
            return jsonify({"error": "No analysis is currently in progress."}), 409
        _CANCEL_EVENT.set()
    return jsonify({"message": "Analysis stop requested."}), 202


@app.route("/outputs/density_maps/<path:filename>")
def serve_density_map(filename):
    """Serves generated density map images so the frontend can <img> them."""
    return send_from_directory(config.DENSITY_MAP_FOLDER, filename)


@app.errorhandler(413)
def too_large(_exc):
    return jsonify({
        "error": f"File too large. Maximum allowed size is "
                 f"{config.MAX_CONTENT_LENGTH_MB} MB."
    }), 413


@app.errorhandler(500)
def internal_error(exc):
    logger.error("Internal server error: %s\n%s", exc, traceback.format_exc())
    return jsonify({"error": "Internal server error.", "details": str(exc)}), 500


if __name__ == "__main__":
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
