"""Conservative visible-color assistance within an existing tomato detection."""
import cv2
import numpy as np


def color_ripeness(frame: np.ndarray, box) -> str | None:
    """Return a red/green appearance hint, never an independent detection."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    # Sample the fruit's center to reduce interference from stems and neighboring tomatoes.
    dx, dy = (x2 - x1) * 0.2, (y2 - y1) * 0.2
    x1, y1 = max(0, int(x1 + dx)), max(0, int(y1 + dy))
    x2, y2 = min(w, int(x2 - dx)), min(h, int(y2 - dy))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = frame[y1:y2, x1:x2]
    crop = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
    hue, saturation, value = cv2.split(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
    visible = (saturation >= 55) & (value >= 40)
    red = int(np.count_nonzero(visible & ((hue <= 20) | (hue >= 165))))
    green = int(np.count_nonzero(visible & (hue >= 25) & (hue <= 95)))
    colored = red + green
    if colored < hue.size * 0.25:
        return None
    if red / colored >= 0.8:
        return "ripe"
    if green / colored >= 0.8:
        return "unripe"
    return None
