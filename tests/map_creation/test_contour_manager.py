import numpy as np

from deepaerialmapper.map_creation.contour import ContourManager


def test_split_sharp_corners():
    # Create some fake data
    contours = [
        np.asarray([[0, 0], [1, 0], [2, 0], [3, 0], [3, 1], [3, 2], [3, 3]]).reshape(
            [-1, 1, 2]
        )
    ]
    cm = ContourManager(contours)
    cm = cm.split_sharp_corners_v2(angle_threshold=50.0)

    assert len(cm) == 2
