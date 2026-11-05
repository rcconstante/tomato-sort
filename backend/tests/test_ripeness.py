import numpy as np
import pytest

from app.ripeness import color_ripeness


@pytest.mark.parametrize("color,label", [((0, 0, 200), "ripe"), ((0, 180, 0), "unripe"),
                                       ((100, 100, 100), None), ((200, 0, 0), None)])
def test_clear_color_hints_and_abstention(color, label):
    frame = np.full((100, 100, 3), color, np.uint8)
    assert color_ripeness(frame, (0, 0, 100, 100)) == label


def test_mixed_colors_and_invalid_crops_abstain():
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[:, :50] = (0, 0, 200)
    frame[:, 50:] = (0, 180, 0)
    assert color_ripeness(frame, (0, 0, 100, 100)) is None
    assert color_ripeness(frame, (0, 0, 3, 3)) is None
