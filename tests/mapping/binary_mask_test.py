import cv2
import numpy as np
import pytest

from deepaerialmapper.mapping import BinaryMask, SemanticClass


@pytest.fixture
def dummy_mask() -> BinaryMask:
    mask = np.array([[0, 0, 0, 1, 1, 0, 1, 0]], dtype=bool)
    return BinaryMask(mask, [SemanticClass.ROAD])


@pytest.fixture
def other_dummy_mask() -> BinaryMask:
    mask = np.array([[1, 0, 0, 1, 1, 0, 0, 0]], dtype=bool)
    return BinaryMask(mask, [SemanticClass.ROAD])


def test_as_color_img(dummy_mask: BinaryMask) -> None:
    img = dummy_mask.as_color_img(scale_factor=255.0, dtype=np.uint8)
    ref_img = (
        np.asarray([0, 0, 0, 255, 255, 0, 255, 0]).reshape((1, -1, 1)).repeat(3, axis=2)
    )
    np.testing.assert_allclose(img, ref_img)


def test_median_blur(dummy_mask: BinaryMask) -> None:
    ref = np.asarray([[0, 0, 0, 1, 1, 1, 0, 0]], dtype=bool)
    np.testing.assert_equal(
        dummy_mask.median_blur(3, border_blur_size_divisor=0).mask, ref
    )


def test_median_blur_border() -> None:
    # Actual 2D mask is needed here due to border handling
    mask = np.array([[0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1]], dtype=bool).repeat(
        5, axis=0
    )
    dummy_mask = BinaryMask(mask, [SemanticClass.ROAD])
    ref = np.asarray([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
    np.testing.assert_equal(
        dummy_mask.median_blur(3, border_blur_size_divisor=2).mask[2, :], ref
    )


def test_close(dummy_mask: BinaryMask) -> None:
    ref = np.asarray([[0, 0, 0, 1, 1, 1, 1, 1]], dtype=bool)
    np.testing.assert_equal(dummy_mask.close(3).mask, ref)


def test_erode(dummy_mask: BinaryMask) -> None:
    np.testing.assert_equal(dummy_mask.erode(3).mask, False)


def test_dilate(dummy_mask: BinaryMask) -> None:
    ref = np.asarray([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=bool)
    np.testing.assert_equal(dummy_mask.dilate(3).mask, ref)


def test_thin() -> None:
    mask = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    thinned_mask = BinaryMask(mask, [SemanticClass.LANEMARKING]).thin()
    ref = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    np.testing.assert_equal(thinned_mask.mask, ref)


def test_union(dummy_mask: BinaryMask, other_dummy_mask: BinaryMask) -> None:
    ref = np.asarray([[1, 0, 0, 1, 1, 0, 1, 0]], dtype=bool)
    np.testing.assert_equal(dummy_mask.union(other_dummy_mask).mask, ref)


def test_intersection(dummy_mask: BinaryMask, other_dummy_mask: BinaryMask) -> None:
    ref = np.asarray([[0, 0, 0, 1, 1, 0, 0, 0]], dtype=bool)
    np.testing.assert_equal(dummy_mask.intersection(other_dummy_mask).mask, ref)


def test_subtraction(dummy_mask: BinaryMask, other_dummy_mask: BinaryMask) -> None:
    ref = np.asarray([[0, 0, 0, 0, 0, 0, 1, 0]], dtype=bool)
    np.testing.assert_equal(dummy_mask.subtraction(other_dummy_mask).mask, ref)


def test_contours() -> None:
    mask = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=bool,
    )
    contours = BinaryMask(mask, [SemanticClass.LANEMARKING]).contours(
        method=cv2.CHAIN_APPROX_SIMPLE
    )
    assert len(contours) == 2
    np.testing.assert_equal(contours[0], np.array([4, 4]).reshape((1, 1, 2)))
    np.testing.assert_equal(contours[1], np.array([[1, 2], [3, 2]]).reshape((2, 1, 2)))


def test__is_split_location_easy():
    contour_mask = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=bool,
    )
    assert BinaryMask._is_split_location(contour_mask, 2, 2) is False

    contour_mask = np.array(
        [
            [0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=bool,
    )
    assert BinaryMask._is_split_location(contour_mask, 3, 3) is True


def test__is_split_location_hard():
    contour_mask = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=bool,
    )
    assert BinaryMask._is_split_location(contour_mask, 2, 2) is False

    contour_mask = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=bool,
    )
    assert BinaryMask._is_split_location(contour_mask, 2, 2) is True

    contour_mask = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=bool,
    )
    assert BinaryMask._is_split_location(contour_mask, 2, 2) is False
    assert BinaryMask._is_split_location(contour_mask, 1, 2) is True


def test_find_split_points():
    contour_mask = np.array(
        [
            [0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=bool,
    )
    _, split_points = BinaryMask(
        contour_mask, [SemanticClass.ROAD]
    ).remove_split_points()
    assert len(split_points) == 1
    np.testing.assert_equal(split_points[0], [2, 2])

    contour_mask = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=bool,
    )
    _, split_points = BinaryMask(
        contour_mask, [SemanticClass.ROAD]
    ).remove_split_points()
    assert len(split_points) == 1
    np.testing.assert_equal(split_points[0], [2, 1])


def test_find_split_points_merging():
    contour_mask = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    _, split_points = BinaryMask(
        contour_mask, [SemanticClass.ROAD]
    ).remove_split_points(merging_distance=0)
    assert len(split_points) == 2
    np.testing.assert_equal(split_points[0], [2, 1])
    np.testing.assert_equal(split_points[1], [2, 3])

    _, split_points = BinaryMask(
        contour_mask, [SemanticClass.ROAD]
    ).remove_split_points(merging_distance=3)
    assert len(split_points) == 1
    np.testing.assert_equal(split_points[0], [2, 2])
