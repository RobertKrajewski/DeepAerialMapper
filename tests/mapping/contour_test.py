import math

import numpy as np
import pytest

from deepaerialmapper.mapping import ContourSegment


def test_from_coordinates():
    coordinates = np.array([[0, 5], [10, 5], [20, 5]]).reshape((-1, 1, 2))
    segment = ContourSegment.from_coordinates(coordinates)
    np.testing.assert_equal(segment.coordinates, coordinates)
    np.testing.assert_equal(segment.center, [10, 5])
    np.testing.assert_equal(segment.first_component, [-1, 0])


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
    with pytest.raises(ValueError):
        segment.closest_point([], 10, 10)


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


def test_lines_intersection():
    # Intersect (in first segment)
    line1 = ContourSegment.from_coordinates(
        np.asarray([0, 0, 0, 4096]).reshape((2, 1, 2))
    )
    line2 = ContourSegment.from_coordinates(
        np.asarray([10, 10, 5, 10]).reshape((2, 1, 2))
    )
    res = line1.intersection(line2)
    assert np.all(res[0] == np.array([0, 10]))
    assert 0 < res[1] < 1

    # Intersect (outside first segment)
    line2 = ContourSegment.from_coordinates(
        np.asarray([10, 5000, 5, 5000]).reshape((2, 1, 2))
    )
    res = line1.intersection(line2)
    assert np.all(res[0] == np.array([0, 5000]))
    assert res[1] > 1

    # Don't intersect
    line2 = ContourSegment.from_coordinates(
        np.asarray([10, 0, 10, 10]).reshape((2, 1, 2))
    )
    res = line1.intersection(line2)
    assert res is None


def test_intersection_image_border():
    line = ContourSegment.from_coordinates(
        np.asarray([10, 10, 5, 10]).reshape((2, 1, 2))
    )
    res = line.intersection_image_border((2048, 4096))
    assert np.all(np.isclose(res[0], np.array([0, 10])))
    assert res[1] == (5.0, 0.0)

    line = ContourSegment.from_coordinates(
        np.asarray([5, 10, 10, 10]).reshape((2, 1, 2))
    )
    res = line.intersection_image_border((2048, 4096))
    assert np.all(np.isclose(res[0], np.array([0, 10])))
    assert res[1] == (-10.0, 0.0)

    line = ContourSegment.from_coordinates(np.asarray([5, 5, 5, 10]).reshape((2, 1, 2)))
    res = line.intersection_image_border((2048, 4096))
    assert np.all(np.isclose(res[0], np.array([5, 2048])))
    assert res[1] == (2038.0, 0.0)


def test_orientation():
    line = np.array([[0, 0], [0.5, 0.05], [1.0, 0]]).reshape((-1, 1, 2))

    segment = ContourSegment.from_coordinates(line)
    _, orientation, _ = segment._pca()
    assert np.isclose(orientation, 0.0)

    segment = ContourSegment.from_coordinates(line[::-1])
    _, orientation, _ = segment._pca()
    assert np.isclose(orientation, np.pi)

    segment = ContourSegment.from_coordinates(line[:, :, ::-1])
    _, orientation, _ = segment._pca()
    assert np.isclose(orientation, np.pi / 2)


def test_oriented_distance():
    line = np.array([[0, 0], [1.0, 0]]).reshape((-1, 1, 2))
    segment = ContourSegment.from_coordinates(line)

    other_line = np.array([[2.0, 0], [3.0, 0]]).reshape((-1, 1, 2))
    other_segment = ContourSegment.from_coordinates(other_line)

    long, lat = segment.oriented_distance(other_segment)
    assert long == 2.0
    assert lat == 0.0

    long, lat = other_segment.oriented_distance(segment)
    assert long == -2.0
    assert lat == 0.0


def test_contour_angle():
    ctr1 = ContourSegment.from_coordinates(np.array([0, 0, 5, 0]).reshape(-1, 1, 2))
    ctr2 = ContourSegment.from_coordinates(np.array([6, 0, 11, 1]).reshape(-1, 1, 2))
    ang12 = ctr1.abs_angle_difference(ctr2)
    print(f"ang12: {ang12} ")
    print(f"ang12: {math.degrees(ang12)} degree ")

    ctr3 = ContourSegment.from_coordinates(np.array([0, 0, 5, 0]).reshape(-1, 1, 2))
    ctr4 = ContourSegment.from_coordinates(np.array([6, 0, 6.1, 10]).reshape(-1, 1, 2))
    ang34 = ctr3.abs_angle_difference(ctr4)
    print(f"ang12: {ang34} ")
    print(f"ang34: {math.degrees(ang34)} degree ")
