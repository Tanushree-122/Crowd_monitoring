"""
inference.py
------------
Core inference pipeline that ties together CSRNet, OpenCV video reading,
and density-map post-processing.

Pipeline for a single frame:
    Frame (BGR, HxWx3)
        -> preprocess_frame()      [preprocessing.py]
        -> CSRNet forward pass     [csrnet.py]
        -> density map (1xHxWx... squeezed to HxW numpy array)
        -> sum(density map)        -> estimated head count for that frame

Pipeline for a video:
    process_video() opens the video with OpenCV, samples frames at a
    configurable interval (config.FRAME_SAMPLE_INTERVAL), runs
    predict_frame() on each sampled frame, and yields progress/results
    frame by frame via a callback so the Flask layer can report live
    progress without blocking indefinitely.
"""

import os
import time
import logging

import cv2
import numpy as np
import torch

from csrnet import CSRNet
from preprocessing import preprocess_frame
from utils.density_utils import save_density_map
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device selection: use CUDA if available, otherwise fall back to CPU.
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelLoadError(Exception):
    """Raised when the CSRNet checkpoint is missing or incompatible."""
    pass


class AnalysisCancelledError(Exception):
    """Raised when the caller requests cancellation between video frames."""
    pass


_MODEL_CACHE = {"model": None}


def load_model(model_path: str = config.MODEL_PATH) -> CSRNet:
    """
    Load CSRNet and its pretrained weights from `model_path`.

    The model is cached in-process after the first successful load so
    repeated requests don't reload weights from disk every time.

    Raises:
        ModelLoadError: if the checkpoint file does not exist, or if the
            checkpoint's state_dict is not compatible with the CSRNet
            architecture defined in csrnet.py.
    """
    if _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"]

    if not os.path.exists(model_path):
        raise ModelLoadError(
            f"CSRNet weight file not found at '{model_path}'. "
            "This project does NOT ship with pretrained weights. "
            "Please download compatible CSRNet weights (trained on "
            "ShanghaiTech Part A/B or UCF-QNRF) and place the .pth file "
            "at this exact path. See README.md, section 'CSRNet Model "
            "Weight Setup', for download links and instructions."
        )

    model = CSRNet(load_weights=True)  # skip ImageNet init, we load full weights next

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)

        # Some public CSRNet checkpoints store the raw state_dict directly;
        # others wrap it in a dict with a 'state_dict' key (common when the
        # checkpoint also stores optimizer state / epoch / etc).
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Strip a leading "module." prefix if the checkpoint was saved
        # from a model wrapped in nn.DataParallel during training.
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[7:] if k.startswith("module.") else k
            cleaned_state_dict[new_key] = v

        model.load_state_dict(cleaned_state_dict)
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(
            "Failed to load CSRNet weights. The checkpoint at "
            f"'{model_path}' is not compatible with the CSRNet "
            f"architecture defined in csrnet.py. Original error: {exc}"
        ) from exc

    model.to(DEVICE)
    model.eval()  # inference mode: disables dropout/batchnorm training behavior

    _MODEL_CACHE["model"] = model
    logger.info("CSRNet model loaded successfully on device: %s", DEVICE)
    return model


def predict_frame(model: CSRNet, frame: np.ndarray):
    """
    Run CSRNet inference on a single BGR frame.

    Returns:
        (count: float, density_map: np.ndarray[H, W])
    """
    tensor = preprocess_frame(frame).to(DEVICE)

    with torch.no_grad():  # disable gradient tracking for inference speed/memory
        output = model(tensor)

    density_map = output.squeeze().detach().cpu().numpy()
    if density_map.ndim == 0:
        # Degenerate case safeguard (shouldn't normally happen)
        density_map = np.array([[density_map]])

    count = float(np.sum(density_map))
    count = max(count, 0.0)  # density maps should be non-negative; guard against noise
    return count, density_map


def process_video(video_path: str, progress_callback=None, cancel_event=None):
    """
    Process an entire video: extract frames at config.FRAME_SAMPLE_INTERVAL,
    run CSRNet on each sampled frame, and collect per-frame counts.

    Args:
        video_path: path to the video file on disk.
        progress_callback: optional callable(processed, total_sampled, latest_result_dict)
            invoked after each sampled frame is processed, so a Flask route
            (or a background job) can report live progress.
        cancel_event: optional threading.Event checked between video frames.

    Returns:
        dict with keys:
            frame_counts: list[float]      - crowd count per sampled frame
            density_map_path: str | None   - path to a saved sample density map image
            frames_processed: int
            processing_fps: float
            video_fps: float
            video_duration_sec: float
            total_frames_in_video: int
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(
            f"OpenCV could not open video file: {video_path}. "
            "The file may be corrupted or in an unsupported codec/container."
        )

    model = load_model()

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_duration_sec = (total_frames_in_video / video_fps) if video_fps > 0 else 0.0

    frame_counts = []
    saved_density_map_path = None
    frame_index = 0
    processed_count = 0

    start_time = time.time()

    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise AnalysisCancelledError("Analysis was stopped by the user.")

            ret, frame = cap.read()
            if not ret:
                break  # end of video

            # Only run CSRNet on every Nth frame to keep processing time
            # bounded, especially important on CPU-only machines.
            if frame_index % config.FRAME_SAMPLE_INTERVAL == 0:
                count, density_map = predict_frame(model, frame)
                frame_counts.append(count)
                processed_count += 1

                # Save one representative density map (the first sampled
                # frame, then periodically overwrite with the latest) so
                # the frontend always has something recent to display.
                if processed_count == 1 or processed_count % 5 == 0:
                    saved_density_map_path = save_density_map(
                        density_map, config.DENSITY_MAP_FOLDER, frame=frame
                    )

                if progress_callback is not None:
                    progress_callback(
                        processed_count,
                        None,  # total sampled frame count is unknown until video ends
                        {
                            "frame_index": frame_index,
                            "count": round(count, 2),
                            "density_map_path": saved_density_map_path,
                        },
                    )

                if processed_count >= config.MAX_FRAMES_TO_PROCESS:
                    logger.info(
                        "Reached MAX_FRAMES_TO_PROCESS (%d); stopping early.",
                        config.MAX_FRAMES_TO_PROCESS,
                    )
                    break

            frame_index += 1
    finally:
        cap.release()

    elapsed = max(time.time() - start_time, 1e-6)
    processing_fps = processed_count / elapsed

    if processed_count == 0:
        raise ValueError(
            "No frames could be extracted/processed from this video. "
            "It may be empty, corrupted, or too short for the configured "
            "FRAME_SAMPLE_INTERVAL."
        )

    return {
        "frame_counts": frame_counts,
        "density_map_path": saved_density_map_path,
        "frames_processed": processed_count,
        "processing_fps": round(processing_fps, 2),
        "video_fps": round(video_fps, 2),
        "video_duration_sec": round(video_duration_sec, 2),
        "total_frames_in_video": total_frames_in_video,
    }
