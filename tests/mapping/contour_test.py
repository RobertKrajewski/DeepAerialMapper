import math

import numpy as np
import pytest

from deepaerialmapper.mapping import ContourSegment


def test_from_coordinates():
    coordinates = np.array([[0, 5], [10, 5], [20, 5]]).reshape((-1, 1, 2))
    segment = ContourSegment.from_coordinates(coordinates)
    np.testing.assert_equal(segment.coordinates, coordinates)
    np.testing.assert_equal(segment.center, [10, 5])
    np.testing.assert_equal(segment.eigenvectors[0], [1, 0])
    np.testing.assert_equal(segment.eigenvectors[1], [0, 1])


def test_orientation():
    line = np.array([[0, 0], [0.5, 0.05], [1.0, 0]]).reshape((-1, 1, 2))

    _, orientation, _ = ContourSegment._pca_approximation(line)
    assert np.isclose(orientation, 0.0)

    _, orientation, _ = ContourSegment._pca_approximation(line[::-1])
    assert np.isclose(orientation, np.pi)

    _, orientation, _ = ContourSegment._pca_approximation(line[:, :, ::-1])
    assert np.isclose(orientation, np.pi / 2)


def test_abs_angle_difference():
    coordinates = np.array([[0, 5], [10, 5], [20, 5]]).reshape((-1, 1, 2))
    segment = ContourSegment.from_coordinates(coordinates)

    coordinates = np.array([[0, 0], [5, 5], [15, 15]]).reshape((-1, 1, 2))
    other_segment = ContourSegment.from_coordinates(coordinates)

    angle = segment.abs_angle_difference(other_segment)
    np.testing.assert_allclose(angle, np.pi / 4)


def test_min_distance_point():
    coordinates = np.array([[0, 5], [10, 5], [20, 5]]).reshape((-1, 1, 2))
    segment = ContourSegment.from_coordinates(coordinates)

    point = np.asarray([15, 0])
    assert segment.min_distance_point(point) == 5.0


def test_closest_point_empty():
    coordinates = np.array([[0, 5], [10, 5], [20, 5]]).reshape((-1, 1, 2))
    segment = ContourSegment.from_coordinates(coordinates)
    assert segment.closest_point([], 10, 10) is None


def test_closest_point():
    coordinates = np.array([[0, 5], [10, 5], [20, 5]]).reshape((-1, 1, 2))
    segment = ContourSegment.from_coordinates(coordinates)

    points = np.asarray([[30, 4], [35, -5]])
    # Too short matchign distance
    assert segment.closest_point(points, 10, 5) is None
    # Long enough matching distance
    np.testing.assert_equal(segment.closest_point(points, 25, 5), points[0])


def test_distance_center_to_point():
    coordinates = np.array([[0, 5], [10, 5], [20, 5]]).reshape((-1, 1, 2))
    segment = ContourSegment.from_coordinates(coordinates)

    point = np.asarray([10, 0])
    assert segment.distance_center_to_point(point) == 5.0


def test_distance_center_to_center():
    coordinates = np.array([[0, 5], [10, 5], [20, 5]]).reshape((-1, 1, 2))
    segment = ContourSegment.from_coordinates(coordinates)

    coordinates = np.array([[0, 0], [5, 5], [10, 10]]).reshape((-1, 1, 2))
    other_segment = ContourSegment.from_coordinates(coordinates)

    assert segment.distance_center_to_center(other_segment) == 5.0


def test_merge():
    coordinates = np.array([[0, 5], [10, 5], [20, 5]]).reshape((-1, 1, 2))
    segment = ContourSegment.from_coordinates(coordinates)

    coordinates = np.array([[0, 0], [5, 5], [10, 10]]).reshape((-1, 1, 2))
    other_segment = ContourSegment.from_coordinates(coordinates)

    merged_segment = segment.merge(other_segment)
    assert len(merged_segment.coordinates) == 6
    np.testing.assert_equal(merged_segment.coordinates[-3:], coordinates)


def test_group_by_length():
    short_segment_coordinates = np.asarray([[0, 0], [1, 0]]).reshape((-1, 1, 2))
    long_segment_coordinates = np.asarray([[0, 0], [10, 0]]).reshape((-1, 1, 2))
    contour = [
        ContourSegment.from_coordinates(short_segment_coordinates),
        ContourSegment.from_coordinates(long_segment_coordinates),
    ]
    long_segments, short_segments = ContourSegment.group_by_length(contour, 5.0)
    assert len(long_segments) == 1
    assert len(short_segments) == 1
    np.testing.assert_equal(long_segments[0].coordinates, long_segment_coordinates)
    np.testing.assert_equal(short_segments[0].coordinates, short_segment_coordinates)


def test_oriented_distance():
    coordinates = np.array([[0, 0], [1.0, 0]]).reshape((-1, 1, 2))
    segment = ContourSegment.from_coordinates(coordinates)

    coordinates = np.array([[2.0, 0], [3.0, 0]]).reshape((-1, 1, 2))
    other_segment = ContourSegment.from_coordinates(coordinates)

    long, lat = segment.oriented_distance(other_segment)
    assert long == 2.0
    assert lat == 0.0

    long, lat = other_segment.oriented_distance(segment)
    assert long == -2.0
    assert lat == 0.0


def test_endpoint_distance():
    coordinates = np.array([[0, 5], [10, 5], [20, 5]]).reshape((-1, 1, 2))
    segment = ContourSegment.from_coordinates(coordinates)

    coordinates = np.array([[0, 0], [5, 5], [10, 10]]).reshape((-1, 1, 2))
    other_segment = ContourSegment.from_coordinates(coordinates)

    distance = segment.endpoint_distance(other_segment)
    assert distance == math.sqrt(10**2 + 5**2)

    distance = other_segment.endpoint_distance(segment)
    assert distance == math.sqrt(10**2 + 5**2)


def test_oriented_distance_point():
    coordinates = np.array([[0, 5], [10, 5], [20, 5]]).reshape((-1, 1, 2))
    segment = ContourSegment.from_coordinates(coordinates)

    point = np.asarray([10, 0])
    np.testing.assert_equal(
        segment.oriented_distance_point(point, endpoint=False), [0.0, -5.0]
    )
    np.testing.assert_equal(
        segment.oriented_distance_point(point, endpoint=True), [-10.0, -5.0]
    )


def test_intersection():
    # Intersect (in first segment)
    line1 = ContourSegment.from_coordinates(
        np.asarray([0, 0, 0, 4096]).reshape((2, 1, 2))
    )
    line2 = ContourSegment.from_coordinates(
        np.asarray([10, 10, 5, 10]).reshape((2, 1, 2))
    )
    res = line1.intersection(line2)
    np.testing.assert_array_equal(res[0], [0, 10])
    assert 0 < res[1] < 1

    # Intersect (outside first segment)
    line2 = ContourSegment.from_coordinates(
        np.asarray([10, 5000, 5, 5000]).reshape((2, 1, 2))
    )
    res = line1.intersection(line2)
    np.testing.assert_array_equal(res[0], [0, 5000])
    assert res[1] > 1

    # Don't intersect
    line2 = ContourSegment.from_coordinates(
        np.asarray([10, 0, 10, 10]).reshape((2, 1, 2))
    )
    res = line1.intersection(line2)
    assert res is None


def test_intersection_image_border():
    segment = ContourSegment.from_coordinates(
        np.asarray([10, 10, 5, 10]).reshape((2, 1, 2))
    )
    intersection_point, long_distance = segment.intersection_image_border((2048, 4096))
    np.testing.assert_allclose(intersection_point, [0, 10])
    assert long_distance == 5.0

    segment = ContourSegment.from_coordinates(
        np.asarray([5, 10, 10, 10]).reshape((2, 1, 2))
    )
    intersection_point, long_distance = segment.intersection_image_border((2048, 4096))
    np.testing.assert_allclose(intersection_point, [4096.0, 10])
    assert long_distance == 4086.0

    segment = ContourSegment.from_coordinates(
        np.asarray([5, 5, 5, 10]).reshape((2, 1, 2))
    )
    intersection_point, long_distance = segment.intersection_image_border((2048, 4096))
    np.testing.assert_allclose(intersection_point, [5, 2048])
    assert long_distance == 2038.0
