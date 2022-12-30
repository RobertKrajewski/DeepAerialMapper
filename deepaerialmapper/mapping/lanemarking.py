from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

import numpy as np


@dataclass
class Lanemarking:
    class LanemarkingType(Enum):
        ROAD_BORDER = (auto(),)
        SOLID = (auto(),)
        DASHED = auto()

    elements: List[np.ndarray]
    type_: LanemarkingType

    @property
    def contour(self) -> np.ndarray:
        return np.concatenate(self.elements)

    @classmethod
    def extend_to(
        cls,
        lanemarkings: List["Lanemarking"],
        other_lanemarkings: List["Lanemarking"],
        img_shape: Tuple[int, int],
        max_long_distance: float = 300,
        max_lat_distance: float = 30,
        check_border=False,
    ) -> List["Lanemarking"]:
        """Try to extend dashed contours to image border and solid contours
        :param lanemarkings:
        :param other_lanemarkings:
        :param img_shape:
        :param max_long_distance:
        :param max_lat_distance:
        :param check_border:
        :return:
        """

        if not len(lanemarkings) or not len(other_lanemarkings):
            return lanemarkings

        other_lanemarkings_points = np.concatenate(
            [o.contour for o in other_lanemarkings]
        )

        from deepaerialmapper.mapping.contour import ContourSegment

        for lanemarking in lanemarkings:
            # Check end
            e = ContourSegment.from_coordinates(lanemarking.elements[-1])
            min_dist_point = e.closest_point(
                other_lanemarkings_points, max_long_distance, max_lat_distance
            )

            if min_dist_point is not None:
                lanemarking.elements.append(min_dist_point[np.newaxis, :, :])
            elif check_border:
                res = e.intersection_image_border(img_shape)
                if res is not None and 0 < res[1][0] < max_long_distance:
                    new_point = np.round(res[0]).astype(int)
                    lanemarking.elements.append(new_point[np.newaxis, np.newaxis, :])

            # Check start
            s = ContourSegment.from_coordinates(lanemarking.elements[0][::-1])
            min_dist_point = s.closest_point(
                other_lanemarkings_points, max_long_distance, max_lat_distance
            )

            if min_dist_point is not None:
                lanemarking.elements.insert(0, min_dist_point[np.newaxis, :, :])
            elif check_border:
                res = s.intersection_image_border(img_shape)
                if res is not None and 0 < res[1][0] < max_long_distance:
                    new_point = np.round(res[0]).astype(int)
                    lanemarking.elements.insert(0, new_point[np.newaxis, np.newaxis, :])

        return lanemarkings
