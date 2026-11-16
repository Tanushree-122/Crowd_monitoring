"""
preprocessing.py
-----------------
Frame preprocessing utilities that convert a raw OpenCV (BGR, HxWx3, uint8)
frame into a normalized PyTorch tensor suitable for CSRNet inference.

CSRNet was trained on images normalized with ImageNet mean/std statistics
(since its frontend is initialized from ImageNet-pretrained VGG16), so we
apply the same normalization here at inference time for consistency with
the training distribution.
"""

import cv2
import numpy as np
import torch

from config import FRAME_RESIZE_MAX_DIM

# ImageNet normalization statistics (RGB order)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_frame_if_needed(frame: np.ndarray, max_dim: int = FRAME_RESIZE_MAX_DIM) -> np.ndarray:
    """
    Downscale a frame so its longer side does not exceed `max_dim`.
    This keeps inference time/memory bounded on CPU-only machines while
    preserving aspect ratio. CSRNet's backend is fully convolutional
    (no fixed-size FC layers) so variable input sizes are supported, but
    to keep density-map-to-pixel-count math simple, dimensions are rounded
    to a multiple of 8 (matching CSRNet's downsampling factor from the
    3 max-pool layers in the frontend, 2^3 = 8).
    """
    h, w = frame.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))

    new_h = int(round((h * scale) / 8) * 8)
    new_w = int(round((w * scale) / 8) * 8)
    new_h = max(new_h, 8)
    new_w = max(new_w, 8)

    if (new_h, new_w) != (h, w):
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """
    Convert a raw BGR OpenCV frame into a normalized CHW float tensor
    with a batch dimension, ready to be passed into CSRNet.

    Steps:
      1. Resize to a bounded size (multiple of 8 on each dimension).
      2. Convert BGR -> RGB (OpenCV loads as BGR, models expect RGB).
      3. Scale pixel values to [0, 1].
      4. Normalize using ImageNet mean/std.
      5. Transpose HWC -> CHW.
      6. Add batch dimension -> 1xCxHxW.

    Returns:
        torch.Tensor of shape (1, 3, H, W), dtype float32.
    """
    frame = resize_frame_if_needed(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(chw).unsqueeze(0).float()  # add batch dim
    return tensor
