import numpy as np

from deepaerialmapper.map_creation.contour import ContourSegment


def test_lines_intersection():
    # Intersect (in first segment)
    line1 = ContourSegment.from_coordinates(np.asarray([0, 0, 0, 4096]).reshape((2, 1, 2)))
    line2 = ContourSegment.from_coordinates(np.asarray([10, 10, 5, 10]).reshape((2, 1, 2)))
    res = line1.intersection(line2)
    assert np.all(res[0] == np.array([0, 10]))
    assert 0 < res[1] < 1

    # Intersect (outside first segment)
    line2 = ContourSegment.from_coordinates(np.asarray([10, 5000, 5, 5000]).reshape((2, 1, 2)))
    res = line1.intersection(line2)
    assert np.all(res[0] == np.array([0, 5000]))
    assert res[1] > 1

    # Don't intersect
    line2 = ContourSegment.from_coordinates(np.asarray([10, 0, 10, 10]).reshape((2, 1, 2)))
    res = line1.intersection(line2)
    assert res is None


def test_intersection_image_border():
    line = ContourSegment.from_coordinates(np.asarray([10, 10, 5, 10]).reshape((2, 1, 2)))
    res = line.intersection_image_border((2048, 4096))
    assert np.all(np.isclose(res[0], np.array([0, 10])))
    assert res[1] == (5.0, 0.0)

    line = ContourSegment.from_coordinates(np.asarray([5, 10, 10, 10]).reshape((2, 1, 2)))
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
    _, _, _, orientation, _, _ = segment.compute_pca()
    assert np.isclose(orientation, 0.0)

    segment = ContourSegment.from_coordinates(line[::-1])
    _, _, _, orientation, _, _ = segment.compute_pca()
    assert np.isclose(orientation, np.pi)


    segment = ContourSegment.from_coordinates(line[:, :, ::-1])
    _, _, _, orientation, _, _ = segment.compute_pca()
    assert np.isclose(orientation, np.pi/2)


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

