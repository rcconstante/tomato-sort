"""Color-based display contours, independent of model grading and tracking."""
from __future__ import annotations

import cv2
import numpy as np


def tomato_contour(frame: np.ndarray, box) -> np.ndarray | None:
    """Estimate a boundary within a detection; abstain on weak color masks.

    This is a display heuristic, not learned segmentation. Limit crop resolution
    to keep processing bounded even when the source frame is large.
    """
    h, w = frame.shape[:2]
    bx1, by1 = max(0, int(box[0])), max(0, int(box[1]))
    bx2, by2 = min(w, int(box[2])), min(h, int(box[3]))
    if bx2 - bx1 < 12 or by2 - by1 < 12:
        return None
    px, py = round((bx2 - bx1) * 0.18), round((by2 - by1) * 0.18)
    x1, y1 = max(0, bx1 - px), max(0, by1 - py)
    x2, y2 = min(w, bx2 + px), min(h, by2 + py)
    crop = frame[y1:y2, x1:x2]
    scale = min(1.0, 96 / max(crop.shape[:2]))
    small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else crop
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 55, 35), (95, 255, 255))
    mask |= cv2.inRange(hsv, (165, 55, 35), (179, 255, 255))
    if np.count_nonzero(mask) / mask.size < 0.1:
        return None
    sh, sw = mask.shape
    # GrabCut fits foreground/background appearance from this crop only.
    # Seed a small colored center; keep the crop border as background.
    gc = np.full(mask.shape, cv2.GC_PR_BGD, np.uint8)
    gc[mask > 0] = cv2.GC_PR_FGD
    center = np.zeros_like(mask)
    cv2.ellipse(center, (sw // 2, sh // 2), (max(1, sw // 9), max(1, sh // 9)), 0, 0, 360, 255, -1)
    gc[(center > 0) & (mask > 0)] = cv2.GC_FGD
    gc[0, :] = gc[-1, :] = cv2.GC_BGD
    gc[:, 0] = gc[:, -1] = cv2.GC_BGD
    if not np.any(gc == cv2.GC_FGD):
        return None
    try:
        cv2.grabCut(small, gc, None, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), 1, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sh, sw = mask.shape
    candidates = [c for c in contours if 0.15 <= cv2.contourArea(c) / (sh * sw) <= 0.95
                  and cv2.pointPolygonTest(c, (sw / 2, sh / 2), False) >= 0]
    if not candidates:
        return None
    contour = max(candidates, key=cv2.contourArea)
    contour = cv2.approxPolyDP(contour, 0.003 * cv2.arcLength(contour, True), True)
    points = contour.astype(np.float32)
    points[:, :, 0] = points[:, :, 0] * (x2 - x1) / sw + x1
    points[:, :, 1] = points[:, :, 1] * (y2 - y1) / sh + y1
    points = np.rint(points).astype(np.int32)
    points[:, :, 0] = np.clip(points[:, :, 0], x1, x2 - 1)
    points[:, :, 1] = np.clip(points[:, :, 1], y1, y2 - 1)
    return points
