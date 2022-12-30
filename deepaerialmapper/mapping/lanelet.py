from collections import Counter
from typing import FrozenSet, Set

import cv2
import numpy as np
from loguru import logger

from deepaerialmapper.mapping.contour import ContourSegment
from deepaerialmapper.mapping.masks import SemanticClass


def derive_lanelets(img_ref, seg_mask, lanemarkings, px2m) -> Set[FrozenSet[int]]:
    valid_mask = seg_mask.class_mask(
        [SemanticClass.ROAD, SemanticClass.SYMBOL, SemanticClass.LANEMARKING]
    )

    min_center_dist = 1.5 / px2m
    max_center_dist = np.sqrt(3.5**2 + 1.5**2) / px2m
    max_angle_diff = np.deg2rad(15)
    lanelets: Set[FrozenSet[int]] = set()
    all_contours = lanemarkings
    for i_contour, lanemarking in enumerate(lanemarkings):
        # # Don't start with road borders
        # if lanemarking.type_ == Lanemarking.LanemarkingType.ROAD_BORDER:
        #     continue

        contour = lanemarking.contour
        logger.debug(f"Contour {i_contour}")
        left = []
        right = []
        img = np.copy(img_ref)

        # Iterate over each segment, find suitable other contour for each segment left and right, and remember those
        num_segments = contour.shape[0]
        for i_segment in range(num_segments - 1):
            segment = ContourSegment.from_coordinates(
                contour[i_segment : i_segment + 2]
            )

            left_costs = {}
            right_costs = {}

            for j_contour, other_lanemarking in enumerate(all_contours):
                # Don't match to itself
                if j_contour == i_contour:
                    continue

                other_contour = other_lanemarking.contour
                for j_segment in range(other_contour.shape[0] - 1):
                    other_segment = ContourSegment.from_coordinates(
                        other_contour[j_segment : j_segment + 2]
                    )

                    # Check if orientation and distance between centers is in allowed range
                    center_dist = segment.distance_center_to_center(other_segment)
                    angle_diff = segment.abs_angle_difference(other_segment)
                    if (
                        center_dist < min_center_dist
                        or center_dist > max_center_dist
                        or angle_diff > max_angle_diff
                    ):
                        continue

                    # Check if perpendicular distance is in appropriate range
                    perp_center_dist = segment.min_distance_point(other_segment.center)
                    if (
                        perp_center_dist < min_center_dist
                        or perp_center_dist > max_center_dist
                    ):
                        continue

                    # Check if in middle between both segments is also road and not e.g. grass
                    center = segment.center_between_centers(other_segment).astype(int)
                    if not valid_mask[center[1], center[0]]:
                        continue

                    if other_segment.center[1] < segment.center[1]:
                        left_costs[(j_contour, j_segment)] = (
                            perp_center_dist + 0.8 * center_dist
                        )
                    else:
                        right_costs[(j_contour, j_segment)] = (
                            perp_center_dist + 0.8 * center_dist
                        )

            if left_costs:
                left.append(min(left_costs, key=left_costs.get))
                j_contour, j_segment = left[-1]
                other_segment = ContourSegment.from_coordinates(
                    all_contours[j_contour].contour[j_segment : j_segment + 2]
                )
                cv2.line(
                    img,
                    segment.center.astype(int),
                    other_segment.center.astype(int),
                    (255, 0, 0),
                    thickness=3,
                )

            if right_costs:
                right.append(min(right_costs, key=right_costs.get))
                j_contour, j_segment = right[-1]
                other_segment = ContourSegment.from_coordinates(
                    all_contours[j_contour].contour[j_segment : j_segment + 2]
                )
                cv2.line(
                    img,
                    segment.center.astype(int),
                    other_segment.center.astype(int),
                    (0, 0, 255),
                    thickness=3,
                )

            cv2.circle(
                img, segment.center.astype(int), 3, (255, 128, 255), thickness=-1
            )

        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()

        if left:
            j_best_other_contour = Counter([m[0] for m in left]).most_common()[0][0]
            logger.info(
                f"Matched contour {i_contour} to left contour {j_best_other_contour}"
            )
            lanelets.update([frozenset([i_contour, j_best_other_contour])])

        if right:
            j_best_other_contour = Counter([m[0] for m in right]).most_common()[0][0]
            logger.info(
                f"Matched contour {i_contour} to right contour {j_best_other_contour}"
            )
            lanelets.update([frozenset([i_contour, j_best_other_contour])])

    return lanelets
