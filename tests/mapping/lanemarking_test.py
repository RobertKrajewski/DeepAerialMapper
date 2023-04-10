import numpy as np

from deepaerialmapper.mapping import Lanemarking


def test_extend_to():
    dashed = [
        Lanemarking(
            [np.array([[5, 0], [6, 0], [7, 0]]).reshape([-1, 1, 2])],
            Lanemarking.LanemarkingType.DASHED,
        )
    ]
    solid = [
        Lanemarking(
            [np.array([[0, 0], [1, 0]]).reshape([-1, 1, 2])],
            Lanemarking.LanemarkingType.SOLID,
        ),
        Lanemarking(
            [np.array([[10, 0], [12, 0]]).reshape([-1, 1, 2])],
            Lanemarking.LanemarkingType.SOLID,
        ),
        # Distractor lanemarking
        Lanemarking(
            [np.array([[10, 3], [12, 3]]).reshape([-1, 1, 2])],
            Lanemarking.LanemarkingType.SOLID,
        ),
    ]

    Lanemarking.extend_to(
        dashed,
        solid,
        (100, 100),
        max_long_distance=10,
        max_lat_distance=3,
        check_border=False,
    )
    assert len(dashed) == 1
    np.testing.assert_equal(
        dashed[0].contour,
        np.array([[1, 0], [5, 0], [6, 0], [7, 0], [10, 0]]).reshape((-1, 1, 2)),
    )


def test_extend_to_check_border():
    dashed = [
        Lanemarking(
            [np.array([[5, 0], [6, 0], [7, 0]]).reshape([-1, 1, 2])],
            Lanemarking.LanemarkingType.DASHED,
        )
    ]
    solid = []

    Lanemarking.extend_to(
        dashed,
        solid,
        (100, 100),
        max_long_distance=10,
        max_lat_distance=3,
        check_border=True,
    )
    assert len(dashed) == 1
    np.testing.assert_equal(
        dashed[0].contour,
        np.array([[0, 0], [5, 0], [6, 0], [7, 0]]).reshape((-1, 1, 2)),
    )
