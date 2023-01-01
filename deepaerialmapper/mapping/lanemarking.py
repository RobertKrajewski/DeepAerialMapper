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
        check_border: bool = False,
    ) -> None:
        """Try to extend dashed contours to image border and solid contours
        :param lanemarkings: Lanemarkings to be extended (modified in-place)
        :param other_lanemarkings: Target lanemarkings for extending to
        :param img_shape: Size of the image (h,w), needed if check_border is activated
        :param max_long_distance: Maximum distance to extension target in longitudinal direction
        :param max_lat_distance: Maximum distance to extension target in lateral direction
        :param check_border: If True, lanemarkings close to image border are extended up to it
        """

        if not len(lanemarkings) or (not len(other_lanemarkings) and not check_border):
            return

        if len(other_lanemarkings):
            other_lanemarkings_points = np.concatenate(
                [o.contour for o in other_lanemarkings]
            )
        else:
            other_lanemarkings_points = []

        from deepaerialmapper.mapping.contour import ContourSegment

        for lanemarking in lanemarkings:
            # Check distance between end of lanemarking and any other lanemarking
            e = ContourSegment.from_coordinates(lanemarking.elements[-1])
            min_dist_point = e.closest_point(
                other_lanemarkings_points, max_long_distance, max_lat_distance
            )

            if min_dist_point is not None:
                lanemarking.elements.append(min_dist_point[np.newaxis, :, :])
            elif check_border:
                # Extend end of lanemarking up to image border
                intersection_point, distance = e.intersection_image_border(img_shape)
                if (
                    intersection_point is not None
                    and 0 < distance[0] < max_long_distance
                ):
                    new_point = np.round(intersection_point).astype(int)
                    lanemarking.elements.append(new_point[np.newaxis, np.newaxis, :])

            # Check distance between start of lanemarking and any other lanemarking
            s = ContourSegment.from_coordinates(lanemarking.elements[0][::-1])
            min_dist_point = s.closest_point(
                other_lanemarkings_points, max_long_distance, max_lat_distance
            )

            if min_dist_point is not None:
                lanemarking.elements.insert(0, min_dist_point[np.newaxis, :, :])
            elif check_border:
                # Extend start of lanemarking up to image border
                intersection_point, distance = s.intersection_image_border(img_shape)
                if (
                    intersection_point is not None
                    and 0 < distance[0] < max_long_distance
                ):
                    new_point = np.round(intersection_point).astype(int)
                    lanemarking.elements.insert(0, new_point[np.newaxis, np.newaxis, :])
