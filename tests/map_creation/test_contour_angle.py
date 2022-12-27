import numpy as np
import math

from deepaerialmapper.mapping.contour import ContourSegment


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


if __name__ == "__main__":
    test_contour_angle()
    # test_merge_contours()
