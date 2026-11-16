"""
crowd_analyzer.py
------------------
Takes the raw per-frame crowd counts produced by CSRNet inference
(inference.process_video) and derives the aggregate statistics and
safety classification that the dashboard displays:

    - current_count   (most recent sampled frame's count)
    - peak_count      (max across all sampled frames)
    - average_count   (mean across all sampled frames)
    - density_level   (LOW / MEDIUM / HIGH / CRITICAL, threshold-based)
    - safety_status   (SAFE / MODERATE / WARNING / EMERGENCY)
    - alerts          (human-readable alert messages)

All numbers here are derived strictly from the `frame_counts` list that
inference.py produced from real CSRNet density-map sums -- nothing here
is randomly generated.
"""

from typing import List, Dict, Any

import config


def classify_density(count: float) -> str:
    """
    Map an estimated crowd count to a density level using the
    configurable thresholds in config.DENSITY_THRESHOLDS.
    """
    thresholds = config.DENSITY_THRESHOLDS
    if count <= thresholds["LOW"]:
        return "LOW"
    elif count <= thresholds["MEDIUM"]:
        return "MEDIUM"
    elif count <= thresholds["HIGH"]:
        return "HIGH"
    else:
        return "CRITICAL"


def classify_safety_status(density_level: str) -> str:
    """Map a density level to a safety status label."""
    return config.SAFETY_STATUS_MAP.get(density_level, "UNKNOWN")


def generate_alerts(frame_counts: List[float]) -> List[Dict[str, Any]]:
    """
    Generate a chronological list of alert events: whenever the density
    level for a sampled frame changes relative to the previous frame,
    or whenever a HIGH/CRITICAL level is reached, an alert entry is
    recorded. This gives the frontend a real, data-driven alert feed
    instead of a static/fake one.
    """
    alerts = []
    previous_level = None

    for idx, count in enumerate(frame_counts):
        level = classify_density(count)
        status = classify_safety_status(level)

        if level != previous_level:
            severity = "info"
            if level in ("HIGH", "CRITICAL"):
                severity = "critical" if level == "CRITICAL" else "warning"

            alerts.append({
                "frame_index": idx,
                "count": round(count, 1),
                "density_level": level,
                "status": status,
                "severity": severity,
                "message": (
                    f"Crowd density changed to {level} "
                    f"(~{round(count)} people) — status: {status}"
                ),
            })
        previous_level = level

    return alerts


def analyze_counts(frame_counts: List[float]) -> Dict[str, Any]:
    """
    Compute the full set of aggregate analytics from a list of per-frame
    crowd counts produced by real CSRNet inference.

    Returns a dict matching the shape the frontend/API expects:
        current_count, peak_count, average_count, density, status,
        alerts, frame_counts (rounded for display), min_count
    """
    if not frame_counts:
        raise ValueError("frame_counts is empty; cannot compute analytics.")

    current_count = frame_counts[-1]
    peak_count = max(frame_counts)
    min_count = min(frame_counts)
    average_count = sum(frame_counts) / len(frame_counts)

    # Density/status classification is based on the CURRENT (most recent)
    # count, since that reflects the live/latest state of the scene.
    density_level = classify_density(current_count)
    safety_status = classify_safety_status(density_level)

    alerts = generate_alerts(frame_counts)

    return {
        "current_count": round(current_count),
        "peak_count": round(peak_count),
        "average_count": round(average_count),
        "min_count": round(min_count),
        "density": density_level,
        "status": safety_status,
        "alerts": alerts,
        "frame_counts": [round(c, 1) for c in frame_counts],
    }


def evaluate_against_ground_truth(predicted_counts: List[float],
                                   ground_truth_counts: List[float]) -> Dict[str, float]:
    """
    Compute MAE (Mean Absolute Error) and MSE (Mean Squared Error) between
    predicted CSRNet counts and known ground-truth counts.

    IMPORTANT: This is ONLY meaningful when `ground_truth_counts` come from
    an annotated dataset (e.g. ShanghaiTech, UCF-QNRF) where the true head
    count per image/frame is known. Ordinary surveillance videos uploaded
    by a user do NOT have ground-truth annotations, so running this
    function against arbitrary demo videos would NOT produce a valid
    accuracy metric -- such videos should be used for demonstration and
    qualitative inference only, never presented as an accuracy evaluation.
    """
    if len(predicted_counts) != len(ground_truth_counts):
        raise ValueError(
            "predicted_counts and ground_truth_counts must be the same "
            "length to compute MAE/MSE."
        )
    n = len(predicted_counts)
    if n == 0:
        raise ValueError("Cannot evaluate an empty set of counts.")

    abs_errors = [abs(p - g) for p, g in zip(predicted_counts, ground_truth_counts)]
    sq_errors = [(p - g) ** 2 for p, g in zip(predicted_counts, ground_truth_counts)]

    mae = sum(abs_errors) / n
    mse = sum(sq_errors) / n

    return {"mae": round(mae, 3), "mse": round(mse, 3), "n_samples": n}
