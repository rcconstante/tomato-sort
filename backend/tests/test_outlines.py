import cv2
import numpy as np
import pytest

from app.outlines import tomato_contour


@pytest.mark.parametrize("color", [(0, 0, 220), (0, 180, 0), (0, 150, 240)])
def test_round_object_boundary_and_source_unchanged(color):
    frame = np.zeros((200, 220, 3), np.uint8)
    cv2.ellipse(frame, (110, 100), (45, 60), 0, 0, 360, color, -1)
    original = frame.copy()
    contour = tomato_contour(frame, (60, 35, 160, 165))
    assert contour is not None
    assert cv2.pointPolygonTest(contour, (110, 100), False) >= 0
    assert cv2.pointPolygonTest(contour, (62, 37), False) < 0
    assert np.array_equal(frame, original)
    assert 7000 < cv2.contourArea(contour) < 10000


@pytest.mark.parametrize("color", [(80, 80, 80), (220, 0, 0), (0, 0, 220)])
def test_unreliable_uniform_crop_abstains(color):
    frame = np.full((100, 100, 3), color, np.uint8)
    assert tomato_contour(frame, (0, 0, 100, 100)) is None


def test_tiny_or_out_of_frame_crop_abstains():
    frame = np.zeros((100, 100, 3), np.uint8)
    assert tomato_contour(frame, (5, 5, 8, 8)) is None
    assert tomato_contour(frame, (110, 110, 130, 130)) is None
