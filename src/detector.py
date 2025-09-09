from typing import List, Tuple

import cv2
import numpy as np

from .schemas import DetectionResult, Slot


def _mask_from_polygon(image_shape: Tuple[int, int], polygon: List[List[int]]) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _compute_features(gray: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
    # Edge density
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    slot_edges = cv2.bitwise_and(edges, edges, mask=mask)
    edge_density = float(np.count_nonzero(slot_edges)) / float(np.count_nonzero(mask) + 1e-6)

    # Texture via Laplacian variance
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_masked = cv2.bitwise_and(cv2.convertScaleAbs(lap), cv2.convertScaleAbs(lap), mask=mask)
    texture_var = float(np.var(lap_masked[np.where(mask > 0)])) if np.count_nonzero(mask) else 0.0

    # Brightness (mean)
    mean_brightness = float(np.mean(gray[np.where(mask > 0)])) if np.count_nonzero(mask) else 0.0

    return edge_density, texture_var, mean_brightness


def detect_occupancy(frame_bgr: np.ndarray, slots: List[Slot]) -> List[DetectionResult]:
    if frame_bgr is None or len(frame_bgr.shape) < 2:
        return []

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Heuristic thresholds (tuned for typical 480-720p)
    EDGE_REF = 0.035  # higher => likely occupied
    TEX_REF = 220.0   # higher => likely occupied
    BRIGHT_MIN = 20.0 # too dark => low confidence

    results: List[DetectionResult] = []
    for slot in slots:
        mask = _mask_from_polygon(gray.shape, slot.polygon)
        edge_density, texture_var, mean_brightness = _compute_features(gray, mask)

        # Combine features into a score in [0,1]
        edge_score = np.clip((edge_density - (EDGE_REF * 0.6)) / (EDGE_REF * 0.8), 0.0, 1.0)
        tex_score = np.clip((texture_var - (TEX_REF * 0.5)) / (TEX_REF * 1.0), 0.0, 1.0)
        light_penalty = 0.0 if mean_brightness > BRIGHT_MIN else 0.3
        combined = 0.55 * edge_score + 0.45 * tex_score
        combined = float(np.clip(combined - light_penalty, 0.0, 1.0))

        occupied = combined > 0.45
        confidence = float(np.clip(0.3 + combined * 0.7, 0.0, 1.0))

        results.append(DetectionResult(slot_id=slot.id, occupied=bool(occupied), confidence=confidence))

    return results
