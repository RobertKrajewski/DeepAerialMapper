import itertools
import math
import pprint
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import scipy
from loguru import logger

from deepaerialmapper.mapping.lanemarking import Lanemarking
from deepaerialmapper.mapping.masks import ClassMask


@dataclass
class ContourSegment:
    """Class for keeping track of an item in inventory."""

    coordinates: np.ndarray
    center: np.ndarray
    slope: float
    b: float

    @staticmethod
    def from_coordinates(coordinates: np.ndarray):
        if len(coordinates) == 1:
            # There is no pca for line with single element
            return ContourSegment(
                coordinates=coordinates,
                center=np.mean(coordinates, axis=(0, 1)),
                slope=0.0,
                b=1.0,
            )

        from sklearn.decomposition import PCA

        r = PCA(n_components=2).fit(coordinates[:, 0, :2])

        return ContourSegment(
            coordinates=coordinates,
            center=np.mean(coordinates, axis=(0, 1)),
            slope=r.components_[0][0],
            b=r.components_[0][1],
        )

    def abs_angle_difference(self, other_segment: "ContourSegment") -> float:
        """Compute angle between two vectors"""
        vector_1 = np.asarray([self.slope, self.b])
        vector_2 = np.asarray([other_segment.slope, other_segment.b])
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        return np.minimum(
            np.abs(np.arccos(dot_product)), np.pi - np.abs(np.arccos(dot_product))
        )

    def min_distance_point(self, point: np.ndarray) -> float:
        """Distance between ax+by+c = 0 and (x,y)"""
        a, b, (c_x, c_y) = self.slope, self.b, self.center
        c = -a * c_x + b * c_y  # Calculate y-axis offset

        x, y = point
        return np.abs(
            (a * x - b * y + c) / (np.sqrt(a**2 + b**2))
        )  # Project point on orthogonal slope

    def distance_center_to_point(self, point: np.ndarray) -> float:
        return np.abs(np.linalg.norm(self.center - point))

    def distance_center_to_center(self, other_segment: "ContourSegment") -> float:
        return self.distance_center_to_point(other_segment.center)

    def center_between_centers(self, other_segment: "ContourSegment") -> np.ndarray:
        return (self.center + other_segment.center) / 2

    def merge(self, other_segment: "ContourSegment") -> "ContourSegment":
        all_coords = np.concatenate([self.coordinates, other_segment.coordinates])
        return self.from_coordinates(all_coords)

    @staticmethod
    def merge_list(segments: List["ContourSegment"]) -> "ContourSegment":
        all_coords = np.concatenate([s.coordinates for s in segments])
        return ContourSegment.from_coordinates(all_coords)

    @staticmethod
    def filter_by_length(
        contour: "Contour", threshold_length: float
    ) -> Tuple["Contour", "Contour"]:
        # TODO: Start/End point has effect on quality.
        long_segments: Contour = []
        short_segments: Contour = []
        for segment in contour:
            if cv2.arcLength(segment.coordinates, closed=False) > threshold_length:
                long_segments.append(segment)
            else:
                short_segments.append(segment)

        return long_segments, short_segments

    def touches_mask_border(self, shape: Tuple) -> bool:
        coords = self.coordinates
        if (
            any(coords[:, :, 0] == shape[1] - 1)
            or any(coords[:, :, 0] == 0)
            or any(coords[:, :, 1] == shape[0] - 1)
            or any(coords[:, :, 1] == 0)
        ):
            return True
        return False

    def compute_pca(self):
        """Compute the eigenvector and centroid position of a contour based on PCA analysis"""
        pts = self.coordinates

        mean, eigenvectors, eigenvalues = cv2.PCACompute2(
            np.reshape(pts.astype(float), (-1, 2)), np.empty(0)
        )
        # Find the end_point two points
        end_pointA = mean + eigenvectors[0] * np.max(np.linalg.norm(mean - pts, axis=2))
        end_pointB = mean - eigenvectors[0] * np.max(np.linalg.norm(mean - pts, axis=2))
        center = [mean[0, 0], mean[0, 1]]
        orientation = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])

        simple_orientation = np.arctan2(
            pts[-1, 0, 1] - pts[0, 0, 1], pts[-1, 0, 0] - pts[0, 0, 0]
        )

        # Make sure that orientation always is "away" from the first coordinate
        if abs(orientation - simple_orientation) > np.pi / 2:
            orientation += np.pi
            eigenvectors[0] *= -1

        return end_pointA, end_pointB, center, orientation, eigenvectors, eigenvalues

    def oriented_distance(self, other_segment: "ContourSegment") -> Tuple[float, float]:
        _, _, center, _, ev, _ = self.compute_pca()
        _, _, o_center, _, o_ev, _ = other_segment.compute_pca()
        center = np.asarray(center)
        o_center = np.asarray(o_center)

        if len(ev) < 2:
            return 10000, 10000

        # Project distance between centers to long and lat direction of current element
        d = o_center - center
        long_dist = d @ ev[0]
        lat_dist = d @ ev[1]

        return long_dist, lat_dist

    def endpoint_distance(self, other_segment: "ContourSegment") -> Tuple[float, float]:
        distances = []
        # Calculate L2 distance for every startpoint endpoint combination
        for idx_self, idx_other in itertools.product([-1], [0, -1]):
            distances.append(
                np.linalg.norm(
                    self.coordinates[idx_self] - other_segment.coordinates[idx_other]
                )
            )

        return min(distances), 0.0

    def oriented_distance_point(
        self, o_center: np.ndarray, endpoint: bool = False
    ) -> float:
        _, _, center, _, ev, _ = self.compute_pca()
        center = np.asarray(center)

        if endpoint:
            center = self.coordinates[-1, 0]

        if len(ev) < 2:
            return 10000, 10000

        # Project distance between centers to long and lat direction of current element
        d = o_center - center
        long_dist = d @ ev[0]
        lat_dist = d @ ev[1]

        return long_dist, lat_dist

    def intersection(self, other_segment: "ContourSegment") -> Tuple[np.ndarray, float]:
        """

        Approximates this and other contour by line (first and last coordinate)

        Args:
            other_segment:

        Returns:
            None, if no intersection exists
            Intersection coordinates and relative position for current segment

        """
        p = self.coordinates[0, 0]
        r = self.coordinates[-1, 0] - self.coordinates[0, 0]

        q = other_segment.coordinates[0, 0]
        s = other_segment.coordinates[-1, 0] - other_segment.coordinates[0, 0]

        denom = np.cross(r, s)
        if denom == 0.0:
            return None

        t = np.cross(q - p, s) / denom

        # This is the intersection point
        return p + t * r, t

    def intersection_image_border(self, img_shape: Tuple[int]):
        h, w = img_shape
        borders = [
            np.asarray([0, 0, 0, h]).reshape((2, 1, 2)),  # left
            np.asarray([0, h, w, h]).reshape((2, 1, 2)),  # bottom
            np.asarray([w, h, w, 0]).reshape((2, 1, 2)),  # right
            np.asarray([w, 0, 0, 0]).reshape((2, 1, 2)),  # top
        ]

        for border in borders:
            intersection = ContourSegment.from_coordinates(border).intersection(self)

            if intersection is not None and 0 < intersection[1] < 1:
                intersection_point = intersection[0]
                distance = self.oriented_distance_point(
                    np.asarray(intersection_point), endpoint=True
                )
                if distance[0] > 0:
                    return intersection_point, distance

        return None


Contour = List[ContourSegment]


class ContourManager:
    def __init__(self, contours: List[np.ndarray]):
        self.contours = contours

    @classmethod
    def from_mask(cls, mask: ClassMask) -> "ContourManager":
        contours = mask.contours(1)
        return cls(contours)

    def merge(self, other_manager: "ContourManager"):
        return ContourManager([*self.contours, *other_manager.contours])

    def unique_coordinates(self) -> "ContourManager":
        """Remove duplicate coordinates from all contours"""
        new_contours = []
        for c in self.contours:
            num_duplicate_coord = (
                np.unique(c, axis=0, return_counts=True)[1] == 2
            ).sum()
            if num_duplicate_coord > 0.15 * len(c):
                """if there is too many overlap, append only half segment of the contour"""
                new_contours.append(c[: len(c) // 2])
            else:
                new_contours.append(c)

        return ContourManager(new_contours)

    def group_at_split_points(self, split_ptrs):
        # Two end points are the most farest points in thin lane marking.
        end_A = [
            i[0].squeeze() for i in self.contours
        ]  # one end point of the thin contour
        end_B = [
            i[-1].squeeze() for i in self.contours
        ]  # another end point of the thin contour

        # If contour is too short, then use full contours.
        segments_A = [
            ContourSegment.from_coordinates(c[:2])
            if len(c) > 2
            else ContourSegment.from_coordinates(c)
            for c in self.contours
        ]  # First half
        segments_B = [
            ContourSegment.from_coordinates(c[-2:])
            if len(c) > 2
            else ContourSegment.from_coordinates(c)
            for c in self.contours
        ]  # Second half

        segments_full = [
            ContourSegment.from_coordinates(i)
            if len(i) <= 2
            else ContourSegment.from_coordinates(i)
            for i in self.contours
        ]

        idxes_close = []  # 3 indexes next to split point
        idx_separate = len(end_A)

        def compute_angle(idx_1: int, idx_2: int, mode="half") -> float:
            """
            half: Use first/second half of a segment
            full: Use full contour for angle calculation
            """

            if mode == "full":
                return segments_full[
                    close_idxs[idx_1] % idx_separate
                ].abs_angle_difference(segments_full[close_idxs[idx_2] % idx_separate])

            # mode == "half"
            first_idx = close_idxs[idx_1]
            second_idx = close_idxs[idx_2]

            first_segment = (
                segments_B[first_idx % idx_separate]
                if first_idx >= idx_separate
                else segments_A[first_idx]
            )
            second_segment = (
                segments_B[second_idx % idx_separate]
                if second_idx >= idx_separate
                else segments_A[second_idx]
            )
            angle = first_segment.abs_angle_difference(second_segment)
            return angle

        # Find 3 close contours around the split_points
        for i_split_ptr, split_ptr in enumerate(split_ptrs):
            logger.debug(f"Grouping around split point {i_split_ptr} at {split_ptr}")
            dist_2_A = scipy.linalg.norm(
                split_ptr - end_A, axis=1
            )  # distance btw segment A and split point
            dist_2_B = scipy.linalg.norm(
                split_ptr - end_B, axis=1
            )  # distance btw segment B and split point
            dist_2 = np.hstack((dist_2_A, dist_2_B))

            # Three closest ctrs
            close_idxs = np.argsort(dist_2)[:3]
            logger.debug(f"Closest endpoints: {close_idxs}")

            max_endpoint_distance = 6
            # Check that all selected end points are in proximity (<=5px distance)
            if np.all(dist_2[close_idxs] > max_endpoint_distance):
                logger.warning(
                    f"All endpoints are too far away from {i_split_ptr}th split point {split_ptr}: {np.array2string(dist_2[close_idxs], precision=2)}"
                )
                continue
            elif not np.all(dist_2[close_idxs] <= max_endpoint_distance):
                logger.warning(
                    f"At least one endpoint is too far away from {i_split_ptr}th split point {split_ptr}: {np.array2string(dist_2[close_idxs], precision=2)}"
                )

            mode = "half"
            angle_0_1 = compute_angle(0, 1, mode=mode)
            angle_0_2 = compute_angle(0, 2, mode=mode)
            angle_1_2 = compute_angle(1, 2, mode=mode)

            if np.isnan(angle_0_1):
                angle_0_1 = 1000
            if np.isnan(angle_0_2):
                angle_0_2 = 1000
            if np.isnan(angle_1_2):
                angle_1_2 = 1000

            # If the third element does not exist anymore (super short segment that was removed), increase weight to prevent selection
            if dist_2[close_idxs][0] > max_endpoint_distance:
                angle_0_1 += 1000
                angle_0_2 += 1000
            if dist_2[close_idxs][1] > max_endpoint_distance:
                angle_0_1 += 1000
                angle_1_2 += 1000
            if dist_2[close_idxs][2] > max_endpoint_distance:
                angle_0_2 += 1000
                angle_1_2 += 1000

            angles = (angle_0_1, angle_0_2, angle_1_2)
            if min(angles) >= 1000:
                logger.warning("Best assignment is too far away for split point")
                continue

            fit = np.argmin(angles)
            if fit == 0:  # connect line 0 and 1
                idxes_close.append(close_idxs[:2])
            elif fit == 1:  # connect line 0 and 2
                idxes_close.append(close_idxs[::2])
            else:  # connect line 1 and 2
                idxes_close.append(close_idxs[1:])

        # TODO: If same contour is in different idxes_close

        # Prevent merging lines with themselves
        filter_idxes_close = [
            i % idx_separate
            for i in idxes_close
            if np.unique(i % idx_separate).__len__() > 1
        ]
        if len(filter_idxes_close) < len(idxes_close):
            logger.debug(
                f"Removed {len(idxes_close) - len(filter_idxes_close)} merges of contours with themselves"
            )

        # Remove duplicate merges
        filter_idxes_close = np.unique(filter_idxes_close, axis=0)

        if len(filter_idxes_close) == 0:
            logger.debug("No contours were be merged at split points!")
            return self
        logger.debug(f"Merging contours \n{np.asarray(filter_idxes_close)}")

        grouped_idxes_close = self._group_contours(filter_idxes_close)

        logger.debug(
            f"Created {len(grouped_idxes_close)} merge groups:\n{pprint.pformat(grouped_idxes_close)}"
        )

        new_contours = []
        # merge contours around split points if they are parallel.
        new_contours.extend(self._merge_contours(self.contours, grouped_idxes_close))

        # include non split points related segments.
        grouped_idxes = sum(grouped_idxes_close, [])
        new_contours.extend(
            ctr for i_ctr, ctr in enumerate(self.contours) if i_ctr not in grouped_idxes
        )

        # Remove contours consisting of a single point only
        new_contours = [c for c in new_contours if len(c) > 1]

        return ContourManager(new_contours)

    def split_lanes(
        self, angle_threshold: float = 30.0, length_threshold: float = 100.0
    ) -> "ContourManager":

        new_contours = []
        for contour in self.contours:
            new_idx = 0

            for i in range(len(contour) - 3):

                current_segment = ContourSegment.from_coordinates(
                    np.concatenate([contour[new_idx : i + 2]], axis=0)
                )
                next_segment = ContourSegment.from_coordinates(
                    np.concatenate([contour[i + 1 : i + 3]], axis=0)
                )

                if current_segment.abs_angle_difference(next_segment) > math.radians(
                    angle_threshold
                ):  # this 5 should be adjusted!.
                    """segments exceed the threshold and current segment is not too short fraction"""
                    new_idx = i + 1

                if i == len(contour) - 4:
                    """Append the last point"""
                    new_contours.append(
                        np.ascontiguousarray(
                            np.concatenate(
                                [current_segment.merge(next_segment).coordinates],
                                axis=0,
                            )
                        )
                    )
                else:
                    new_contours.append(
                        np.ascontiguousarray(
                            np.concatenate([current_segment.coordinates], axis=0)
                        )
                    )

            if new_idx == 0:
                """no split in the current contour"""
                new_contours.append(contour)

        return ContourManager(new_contours)

    @staticmethod
    def _merge_contours(
        contours: List[np.ndarray], groups: List[List[int]]
    ) -> List[np.ndarray]:
        new_contours = []
        for i_group, group in enumerate(groups):
            logger.debug(f"Merging contour group {i_group} containing contours {group}")
            # Collect all the relevant contours
            group_contours = [c for i_c, c in enumerate(contours) if i_c in group]

            # Now merging necessary
            if len(group_contours) == 1:
                logger.debug(
                    "Group contains only single contour, no merging necessary!"
                )
                new_contours.append(group_contours[0])
                continue

            # Merge relevant contours
            group_contours = ContourManager(group_contours)
            grouped_group_contours, ungrouped_group_contours = group_contours.group(
                max_gap_size=15, distance_mode="endpoints"
            )

            if len(ungrouped_group_contours):
                logger.warning(
                    f"Shit happened. {len(ungrouped_group_contours)} contours were not grouped."
                )

            if len(grouped_group_contours) == 0:
                logger.error(
                    "Big shit happened. Contour could not be grouped at all :("
                )
                continue

            if len(grouped_group_contours) > 1:
                logger.warning(
                    f"Small shit happened. Contour was grouped into {len(grouped_group_contours)} contours :/"
                )

            # Add all new created groups
            new_contours.extend(cm.contour for cm in grouped_group_contours)

        return new_contours

    @staticmethod
    def _group_contours(groups: List[np.ndarray]) -> List[np.ndarray]:
        # Get the highest contour id
        max_contour_id = int(np.max(np.asarray(groups)))
        contour2group = [-1] * (max_contour_id + 1)

        next_group = 0
        for seg_a, seg_b in groups:
            higher_group_idx = max(contour2group[seg_a], contour2group[seg_b])
            lower_group_idx = min(contour2group[seg_a], contour2group[seg_b])

            if higher_group_idx == -1 and lower_group_idx == -1:
                # Start a new group
                contour2group[seg_a] = next_group
                contour2group[seg_b] = next_group
                next_group += 1
            elif higher_group_idx > -1 and lower_group_idx == -1:
                # Assign to existing group
                contour2group[seg_a] = higher_group_idx
                contour2group[seg_b] = higher_group_idx
            else:
                # Shit - we have to merge groups
                # Replace all occurrences of higher group idx by lower group idx
                contour2group = [
                    idx if idx != higher_group_idx else lower_group_idx
                    for idx in contour2group
                ]

        new_groups = []
        contour2group = np.asarray(contour2group)
        for group_idx in np.unique(contour2group):
            if group_idx == -1:
                # Skip group of non-grouped contours
                continue

            # For each group, get all assigned contours
            new_groups.append(
                [
                    i_c
                    for i_c in range(max_contour_id + 1)
                    if contour2group[i_c] == group_idx
                ]
            )

        return new_groups

    def split_sharp_corners(self, angle_threshold: float = 30.0) -> "ContourManager":
        new_contours = []
        for contour in self.contours:
            # Skip too short contours
            if len(contour) <= 2:
                new_contours.append(contour)
                continue

            split_points = []

            for i_point in range(1, len(contour) - 1):
                p = ContourSegment.from_coordinates(contour[i_point - 1 : i_point + 1])
                n = ContourSegment.from_coordinates(contour[i_point : i_point + 2])
                print(contour[i_point][0], p.abs_angle_difference(n))
                if p.abs_angle_difference(n) >= math.radians(angle_threshold):
                    split_points.append(i_point)

            if not split_points:
                new_contours.append(contour)
                continue

            split_points = [0, *split_points, len(split_points)]

            for i_point in range(1, len(split_points)):
                i_start = split_points[i_point - 1]
                i_end = split_points[i_point] + 1
                new_contours.append(contour[i_start:i_end])

        return ContourManager(new_contours)

    def subsample(self, factor: int, post_min_length: int = 2) -> "ContourManager":
        new_contours = []
        for contour in self.contours:
            coords = [contour[::factor]]
            if len(contour) % factor > factor / 3:
                coords.append(contour[-1:])  # Add final point
            elif len(coords):
                coords[-1][-1] = contour[-1]  # Replace last point
            new_contours.append(np.ascontiguousarray(np.concatenate(coords, axis=0)))

        # remove empty array
        new_contours = [x for x in new_contours if x.size > post_min_length]

        return ContourManager(new_contours)

    def split_at_border(self, mask_shape: Tuple[int]) -> "ContourManager":
        # Split contours at mask border
        new_contours = []
        for contour in self.contours:
            is_border = (
                (contour[:, 0, 0] == 0)
                | (contour[:, 0, 1] == 0)
                | (contour[:, 0, 0] == mask_shape[1] - 1)
                | (contour[:, 0, 1] == mask_shape[0] - 1)
            )

            groups = np.flatnonzero(
                np.diff(np.r_[0, np.invert(is_border), 0]) != 0
            ).reshape(-1, 2) - [0, 1]
            for g in groups:
                new_contour = contour[g[0] : g[1]]
                if len(new_contour):
                    new_contours.append(new_contour)
        return ContourManager(new_contours)

    def filter_by_length(
        self, max_length: int, min_length: int
    ) -> Tuple["ContourManager", "ContourManager"]:
        """Assumes sampled contours"""
        short_contours, long_contours = [], []
        for contour in self.contours:
            if contour.shape[0] > max_length:
                long_contours.append(contour)
            elif contour.shape[0] >= min_length:
                short_contours.append(contour)

        return ContourManager(short_contours), ContourManager(long_contours)

    def group(
        self,
        max_gap_size: float = 500.0,
        debug: bool = False,
        distance_mode="centers_oriented",
    ) -> Tuple["ContourManager", "ContourManager"]:
        """Assumes to be given short lanemarking contours (e.g. from `filter_by_length()`)"""
        contour_segments = [
            ContourSegment.from_coordinates(contour) for contour in self.contours
        ]

        # Calculate pairwise distances
        num_segments = len(contour_segments)
        pairwise_distances = np.zeros((num_segments, num_segments, 2))
        for i_contour, contour in enumerate(contour_segments):
            for j_contour, other_contour in enumerate(contour_segments):
                if i_contour == j_contour:
                    pairwise_distances[i_contour, j_contour] = np.array(
                        [np.inf, np.inf]
                    )
                    continue

                if distance_mode == "centers_oriented":
                    pairwise_distances[
                        i_contour, j_contour
                    ] = contour.oriented_distance(other_contour)
                else:
                    # Calculate distance between start and endpoints
                    pairwise_distances[
                        i_contour, j_contour
                    ] = contour.endpoint_distance(other_contour)

        # For each contour find best contour in front (to the right side in mage) by minimal longitudinal distance
        cost = pairwise_distances[..., 0] + np.abs(pairwise_distances[..., 1]) * 8

        # Prevent match to itself
        cost[cost == 0.0] = np.inf

        # Remove possible matches to the back
        cost[pairwise_distances[..., 0] < 0] = np.inf

        # Remove long distance assignments
        cost[pairwise_distances[..., 0] > 300] = np.inf

        # cost[(np.abs(pairwise_distances[..., 1]) + np.abs(
        #    pairwise_distances[..., 0]) / 10) > 1] = np.inf  # Remove high lat distance possible matches
        cost[
            np.abs(pairwise_distances[..., 1]) > 25
        ] = np.inf  # Remove high lat distance possible matches

        # TODO: Check orientation difference

        # Connection map:
        # row: i-th contour,
        # col 0: idx of prev contour
        # col 1: idx of next contour
        conn = np.zeros((len(contour_segments), 2), int) - 1
        for i_contour, _ in enumerate(contour_segments):
            logger.debug(f"Contour {i_contour}")

            # Find the closest other contour
            i_contour_best = np.argmin(cost[i_contour])
            if cost[i_contour, i_contour_best] > max_gap_size:
                # Too far away match
                logger.debug(
                    f"Did not match with {i_contour_best} due too large cost {cost[i_contour, i_contour_best]:.2f}"
                )
                continue

            conn[i_contour, 1] = i_contour_best  # Set next
            conn[i_contour_best, 0] = i_contour  # Set prev

        dashed_lanemarkings = []
        short_lanemarkings = []

        # show best connected contour
        if debug:
            import matplotlib.pyplot as plt

            for i_c, c in enumerate(contour_segments):
                plt.plot(
                    c.coordinates[:, 0, 0], c.coordinates[:, 0, 1], c="g", linewidth=3
                )
                plt.scatter(
                    c.coordinates[:1, 0, 0],
                    c.coordinates[:1, 0, 1],
                    c="k",
                    linewidths=1,
                )
                plt.text(c.center[0], c.center[1], str(i_c))

            for i_c, c in enumerate(conn):
                f = i_c
                s = c[1]
                if s == -1:
                    continue
                # plt.plot([contour_segments[f].center[0], contour_segments[s].center[0]],
                #          [contour_segments[f].center[1], contour_segments[s].center[1]], c='b', linewidth=3)
                plt.arrow(
                    contour_segments[f].center[0],
                    contour_segments[f].center[1],
                    contour_segments[s].center[0] - contour_segments[f].center[0],
                    contour_segments[s].center[1] - contour_segments[f].center[1],
                    color="b",
                    width=5,
                )

            plt.gca().axis("equal")
            plt.grid()
            plt.gca().invert_yaxis()
            plt.show()

        # Create groups
        for i_conn in range(conn.shape[0]):
            # Check if start point (== no previous segment)
            if conn[i_conn, 0] != -1:
                continue

            # Create a new group and iterate over all next segments
            new_group = [contour_segments[i_conn]]
            c = i_conn
            while conn[c, 1] >= 0:
                new_c = conn[c, 1]
                conn[c] = [-2, -2]  # Get next segment, set current to used
                c = new_c

                # last_point_current = contour_segments[-1].coordinates[-1]
                new_segment = contour_segments[c]
                # first_point_new = new_segment.coordinates[0]
                # last_point_new = new_segment.coordinates[-1]
                # if np.linalg.norm(first_point_new - last_point_current) < np.linalg.norm(last_point_new - last_point_current):
                #     new_segment.coordinates = new_segment.coordinates[::-1]
                new_group.append(new_segment)
            conn[c] = [-2, -2]

            # Depending on number of segments classify into short and dashed lanemarkings
            # TODO: Check if segments have roughly same size
            new_group = [c.coordinates for c in new_group]
            if len(new_group) == 1:
                short_lanemarkings.append(
                    Lanemarking(new_group, Lanemarking.LanemarkingType.DASHED)
                )
            else:
                # dashed_lanemarkings.append(ContourSegment.merge_list(new_group).coordinates)
                dashed_lanemarkings.append(
                    Lanemarking(new_group, Lanemarking.LanemarkingType.DASHED)
                )

        # short_lanemarkings = ContourManager(short_lanemarkings)
        # dashed_lanemarkings = ContourManager(dashed_lanemarkings)

        return dashed_lanemarkings, short_lanemarkings

    def __iter__(self):
        return self.contours.__iter__()

    def __getitem__(self, item):
        return self.contours.__getitem__(item)

    def __len__(self):
        return len(self.contours)

    def show(self, show=True):
        if not self.contours:
            return

        m = np.array([np.max(contour, axis=(0, 1)) for contour in self.contours])
        max_x, max_y = np.max(m[:, 0]), np.max(m[:, 1])

        img = np.zeros((max_y + 10, max_x + 10, 3), np.uint8)
        for i_contour, contour in enumerate(self.contours):
            img = cv2.polylines(
                img,
                [contour],
                isClosed=False,
                color=(
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ),
                thickness=2,
            )
            img = cv2.putText(
                img,
                str(i_contour),
                np.mean(contour, axis=(0, 1)).astype(int),
                cv2.FONT_HERSHEY_PLAIN,
                1.5,
                (255, 0, 0),
            )

        if show:
            import matplotlib.pyplot as plt

            plt.imshow(img)
            plt.show()

        return img
