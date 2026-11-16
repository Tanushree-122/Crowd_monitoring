"""
density_utils.py
-----------------
Utilities for turning a raw CSRNet density map (a single-channel array of
non-negative floats, where the sum over the map equals the estimated head
count) into a human-viewable colorized image, optionally overlaid on the
original frame.
"""

import os
import time
import cv2
import numpy as np


def normalize_density_map(density_map: np.ndarray) -> np.ndarray:
    """
    Normalize a raw density map to the 0-255 uint8 range for visualization.
    Uses min-max scaling; if the map is completely flat (e.g. an empty
    scene with ~0 density everywhere) it returns an all-zero image rather
    than dividing by zero.
    """
    dmap = density_map.astype(np.float32)
    dmin, dmax = float(dmap.min()), float(dmap.max())
    if dmax - dmin < 1e-8:
        return np.zeros_like(dmap, dtype=np.uint8)
    normalized = (dmap - dmin) / (dmax - dmin)
    return (normalized * 255).astype(np.uint8)


def colorize_density_map(density_map: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Convert a raw density map into a BGR heatmap image using an OpenCV
    colormap (JET by default: blue = low density, red = high density).
    """
    normalized = normalize_density_map(density_map)
    heatmap = cv2.applyColorMap(normalized, colormap)
    return heatmap


def overlay_density_on_frame(frame: np.ndarray, density_map: np.ndarray,
                              alpha: float = 0.5) -> np.ndarray:
    """
    Resize the density map heatmap to match the original frame's
    dimensions and alpha-blend it on top of the frame, so the density
    hotspots are visible in the context of the actual scene.
    """
    heatmap = colorize_density_map(density_map)
    heatmap_resized = cv2.resize(
        heatmap, (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_resized, alpha, 0)
    return overlay


def save_density_map(density_map: np.ndarray, output_dir: str,
                      frame: np.ndarray = None, prefix: str = "density") -> str:
    """
    Save a colorized (and optionally frame-overlaid) density map image to
    `output_dir`. Returns the path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time() * 1000)
    filename = f"{prefix}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    if frame is not None:
        image_to_save = overlay_density_on_frame(frame, density_map)
    else:
        image_to_save = colorize_density_map(density_map)

    cv2.imwrite(filepath, image_to_save)
    return filepath
