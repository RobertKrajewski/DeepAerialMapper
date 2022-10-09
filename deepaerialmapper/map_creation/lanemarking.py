from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from typing import List


@dataclass
class Lanemarking:
    class LanemarkingType(Enum):
        ROAD_BORDER = auto(),
        SOLID = auto(),
        DASHED = auto()

    elements: List[np.ndarray]
    type_: LanemarkingType

    @property
    def contour(self) -> np.ndarray:
        return np.concatenate(self.elements)

    @classmethod
    def extend_to(cls, lanemarkings: List["Lanemarking"], other_lanemarkings: List["Lanemarking"], img_shape,
                  check_border=False) -> List["Lanemarking"]:
        """Try to extend dashed contours to image border and solid contours"""
        max_long_distance = 300
        max_lat_distance = 30

        if not len(lanemarkings) or not len(other_lanemarkings):
            return lanemarkings

        all_solid_points = np.concatenate([o.contour for o in other_lanemarkings])

        from data_generation.mapping.contour import ContourSegment
        for lanemarking in lanemarkings:
            # Check end
            e = ContourSegment.from_coordinates(lanemarking.elements[-1])

            min_dist = 10000
            min_dist_point = None

            for i_p, p in enumerate(all_solid_points):
                long, lat = e.oriented_distance_point(p)

                if 0 < long < max_long_distance and abs(lat) < max_lat_distance and long < min_dist:
                    min_dist = long
                    min_dist_point = p

            if min_dist_point is not None:
                lanemarking.elements.append(min_dist_point[np.newaxis, :, :])
            elif check_border:
                res = e.intersection_image_border(img_shape)
                if res is not None and 0 < res[1][0] < max_long_distance:
                    new_point = np.round(res[0]).astype(int)
                    lanemarking.elements.append(new_point[np.newaxis, np.newaxis, :])

            # Check start
            s = ContourSegment.from_coordinates(lanemarking.elements[0][::-1])
            min_dist = 10000
            min_dist_point = None

            for i_p, p in enumerate(all_solid_points):
                long, lat = s.oriented_distance_point(p)

                if 0 < long < max_long_distance and abs(lat) < max_lat_distance and long < min_dist:
                    min_dist = long
                    min_dist_point = p

            if min_dist_point is not None:
                lanemarking.elements.insert(0, min_dist_point[np.newaxis, :, :])
            elif check_border:
                res = s.intersection_image_border(img_shape)
                if res is not None and 0 < res[1][0] < max_long_distance:
                    new_point = np.round(res[0]).astype(int)
                    lanemarking.elements.insert(0, new_point[np.newaxis, np.newaxis, :])

        return lanemarkings


def test_extend():
    dashed = [Lanemarking([np.array([[0, 0], [1, 0]]).reshape([2, 1, 2])], Lanemarking.LanemarkingType.DASHED)]
    solid = [Lanemarking([np.array([[10, 0], [11, 0]]).reshape([2, 1, 2])], Lanemarking.LanemarkingType.SOLID),
             Lanemarking([np.array([[-10, 0], [-9, 0]]).reshape([2, 1, 2])], Lanemarking.LanemarkingType.SOLID)]

    Lanemarking.extend_to(dashed, solid)
