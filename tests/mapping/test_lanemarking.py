import numpy as np

from deepaerialmapper.mapping import Lanemarking


def test_extend():
    dashed = [
        Lanemarking(
            [np.array([[0, 0], [1, 0]]).reshape([2, 1, 2])],
            Lanemarking.LanemarkingType.DASHED,
        )
    ]
    solid = [
        Lanemarking(
            [np.array([[10, 0], [11, 0]]).reshape([2, 1, 2])],
            Lanemarking.LanemarkingType.SOLID,
        ),
        Lanemarking(
            [np.array([[-10, 0], [-9, 0]]).reshape([2, 1, 2])],
            Lanemarking.LanemarkingType.SOLID,
        ),
    ]

    Lanemarking.extend_to(dashed, solid)
