import itertools
import pprint
from typing import List, Tuple

import cv2
import numpy as np
import scipy
from loguru import logger

from deepaerialmapper.mapping.binary_mask import BinaryMask
from deepaerialmapper.mapping.contour import ContourSegment
from deepaerialmapper.mapping.lanemarking import Lanemarking


class ContourManager:
    """Performs high-level actions on list of contours"""

    def __init__(self, contours: List[np.ndarray]):
        self.contours = contours

    @classmethod
    def from_mask(cls, mask: BinaryMask) -> "ContourManager":
        """Extract contours from a binary mask"""
        contours = mask.contours(cv2.CHAIN_APPROX_NONE)
        return cls(contours)

    def merge(self, other_manager: "ContourManager"):
        """Combine the contours of two contour managers"""
        return ContourManager([*self.contours, *other_manager.contours])

    def unique_coordinates(
        self, duplicates_ratio_threshold: float = 0.15
    ) -> "ContourManager":
        """Remove duplicate coordinates from all contours"""
        new_contours = []
        for c in self.contours:
            duplicates_count = (np.unique(c, axis=0, return_counts=True)[1] == 2).sum()
            if duplicates_count > duplicates_ratio_threshold * len(c):
                # If there are many duplicates, the second half is the reversed list of coordinates of the first half
                new_contours.append(c[: len(c) // 2])
            else:
                new_contours.append(c)

        return ContourManager(new_contours)

    def merge_at_split_points(
        self, split_points: List[np.ndarray], max_endpoint_distance: int = 6
    ) -> "ContourManager":
        """Find contours that meet at split points and merge those with a similar orientation.
        :param split_points:
        :param max_endpoint_distance:
        :return:
        """
        endpoints = [c[0, 0] for c in self.contours]  # First point of each contour
        endpoints.extend(c[-1, 0] for c in self.contours)  # Last point of each contour

        # First part of each contour
        contour_starts = [
            ContourSegment.from_coordinates(c[:2])
            if len(c) > 2
            else ContourSegment.from_coordinates(c)
            for c in self.contours
        ]
        # Last part of each contour
        contour_ends = [
            ContourSegment.from_coordinates(c[-2:])
            if len(c) > 2
            else ContourSegment.from_coordinates(c)
            for c in self.contours
        ]
        segments = [*contour_starts, *contour_ends]

        # Find three closest contours around each split point and try to pair two of them to form a new line
        idxs_paired_segments = []
        for i_split_point, split_point in enumerate(split_points):
            logger.debug(
                f"Grouping around split point {i_split_point} at {split_point}"
            )

            # Find three closest endpoints
            dists = scipy.linalg.norm(split_point - endpoints, axis=1)
            close_idxs = np.argsort(dists)[:3]
            logger.debug(f"Closest endpoints: {close_idxs}")

            # Check that all selected end points are in proximity (<=max endpoint distance)
            close_dists = dists[close_idxs]
            if np.all(close_dists > max_endpoint_distance):
                logger.warning(
                    f"All endpoints are too far away from {i_split_point}th split point {split_point}:"
                    f" {np.array2string(close_dists, precision=2)}"
                )
                continue
            elif not np.all(close_dists <= max_endpoint_distance):
                logger.warning(
                    f"At least one endpoint is too far away from {i_split_point}th split point {split_point}:"
                    f" {np.array2string(close_dists, precision=2)}"
                )

            costs = []
            candidate_pairs = []
            # Iterate over all combinations of found segments and derive cose for pairing them to a new contour
            for a, b in itertools.combinations(range(len(close_idxs)), 2):
                if max(close_dists[a], close_dists[b]) > max_endpoint_distance:
                    continue

                cost = segments[close_idxs[a]].abs_angle_difference(
                    segments[close_idxs[b]]
                )
                if np.isnan(cost):
                    continue

                costs.append(cost)
                candidate_pairs.append([a, b])

            if len(costs) == 0 or min(costs) == np.inf:
                logger.warning("Couldn't match segments at split point!")
                continue

            idxs_paired_segments.append(close_idxs[candidate_pairs[np.argmin(costs)]])

        # Remove segment pairs representing a single contour paired to itself
        num_segment_pairs = len(idxs_paired_segments)
        idxs_paired_contours = [
            idxs % len(self.contours)  # Map from segment idx to contour idx
            for idxs in idxs_paired_segments
            if len(np.unique(idxs % len(self.contours))) > 1
        ]
        if len(idxs_paired_contours) < num_segment_pairs:
            logger.debug(
                f"Removed {num_segment_pairs - len(idxs_paired_segments)} merges of contours with themselves"
            )

        # Remove duplicate pairs
        idxs_paired_contours = np.unique(idxs_paired_contours, axis=0).tolist()
        if len(idxs_paired_contours) == 0:
            logger.debug("No contours were be merged at split points!")
            return self
        logger.debug(f"Merging contours \n{np.asarray(idxs_paired_contours)}")

        # From the found pairs, find groups with >=2 contours
        contour_groups = self._group_contour_pairs(idxs_paired_contours)
        logger.debug(
            f"Created {len(contour_groups)} merge groups:\n{pprint.pformat(contour_groups)}"
        )

        # Merge the contours in the found groups
        new_contours = self._merge_contour_groups(self.contours, contour_groups)

        # Keep all contours that were not part of the merging process
        grouped_idxes = sum(contour_groups, [])
        new_contours.extend(
            ctr for i_ctr, ctr in enumerate(self.contours) if i_ctr not in grouped_idxes
        )

        # Remove contours consisting of a single point only
        new_contours = [c for c in new_contours if len(c) > 1]

        return ContourManager(new_contours)

    @staticmethod
    def _merge_contour_groups(
        contours: List[np.ndarray], groups: List[List[int]]
    ) -> List[np.ndarray]:
        """Given groups of contours, try to merge the contours in each group.

        :param contours: List of contours
        :param groups: List of groups. Each group is a list of idxs
        :return: List of contours after grouping
        """
        new_contours = []
        for i_group, group in enumerate(groups):
            logger.debug(f"Merging contour group {i_group} containing contours {group}")
            group_contours = [contours[i_c] for i_c in group]

            if len(group_contours) == 1:
                logger.debug(
                    f"Group {i_group} contains only single contour, no merging necessary!"
                )
                new_contours.append(group_contours[0])
                continue

            # Create a temporary ContourManager to use the group function
            group_contours = ContourManager(group_contours)

            # Use group function here as the given groups don't provide an order of the contours
            (
                grouped_group_contours,
                ungrouped_group_contours,
            ) = group_contours.group_dashed_contours(
                max_match_cost=15, distance_mode="endpoints"
            )

            if len(grouped_group_contours) == 0:
                logger.error("Contour could not be grouped at all")
                continue

            if len(ungrouped_group_contours):
                logger.warning(
                    f"{len(ungrouped_group_contours)} contours were not grouped."
                )

            if len(grouped_group_contours) > 1:
                logger.warning(
                    f"Contour was grouped into {len(grouped_group_contours)} contours"
                )

            # Add all new created groups
            new_contours.extend(cm.contour for cm in grouped_group_contours)

        return new_contours

    @staticmethod
    def _group_contour_pairs(idxs_paired: List[List[int]]) -> List[List[int]]:
        """Given contour pairs, find groups of contours.

        E.g. given pairs: [[0, 1], [2, 3], [1, 4]]
        can be grouped into two groups: [[0, 1, 4], [2, 3]] because contour 1 is connected to both contours 0 and 4

        :param idxs_paired: List of contour idxs pairs [(first_contour_idx, second_contour_idx),...]
        :return: List of contour groups. Each contour group is a list of contour idxs
        """
        # Get the highest contour id
        max_idx = int(np.max(np.asarray(idxs_paired)))
        contour2group = [-1] * (max_idx + 1)

        next_group = 0
        for idx_a, idx_b in idxs_paired:
            higher_group_idx = max(contour2group[idx_a], contour2group[idx_b])
            lower_group_idx = min(contour2group[idx_a], contour2group[idx_b])

            if higher_group_idx == -1 and lower_group_idx == -1:
                # Start a new group
                contour2group[idx_a] = next_group
                contour2group[idx_b] = next_group
                next_group += 1
            elif higher_group_idx > -1 and lower_group_idx == -1:
                # Assign to existing group
                contour2group[idx_a] = higher_group_idx
                contour2group[idx_b] = higher_group_idx
            else:
                # Two existing groups meet
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
            new_groups.append(np.flatnonzero(contour2group == group_idx).tolist())

        return new_groups

    def subsample(self, factor: int, post_min_length: int = 3) -> "ContourManager":
        """Subsample contours by keeping only every nth point.

        Special treatment of the last point of each contour to avoid to ensure more homogeneous distance between points.

        :param factor: Subsampling factor
        :param post_min_length: Keep only contours which have a minimum length after subsampling
        """
        new_contours = []
        for contour in self.contours:
            coords = [contour[::factor]]

            # Handle last point
            if len(contour) % factor > factor / 3:
                coords.append(contour[-1:])  # Add final point
            elif len(coords):
                coords[-1][-1] = contour[-1]  # Replace last point
            new_contours.append(np.ascontiguousarray(np.concatenate(coords, axis=0)))

        # Remove short segments
        new_contours = [x for x in new_contours if x.size >= post_min_length]

        return ContourManager(new_contours)

    def split_at_border(self, mask_shape: Tuple[int, int]) -> "ContourManager":
        """Split contours at the image/mask border.

        :param mask_shape: Size of the mask(height, width)
        """
        new_contours = []
        for contour in self.contours:
            # Check for every point if it is located on the image border
            is_border = (
                (contour[:, 0, 0] == 0)
                | (contour[:, 0, 1] == 0)
                | (contour[:, 0, 0] == mask_shape[1] - 1)
                | (contour[:, 0, 1] == mask_shape[0] - 1)
            )

            # Find sections/groups of coordinates by checking for transitions between is_border=True and is_border=False
            # Each group is represented by a tuple (start_index, end_index)
            groups = np.flatnonzero(
                np.diff(np.r_[0, np.invert(is_border), 0]) != 0
            ).reshape(-1, 2) - [0, 1]

            # Create a new contour for each group
            for g in groups:
                new_contour = contour[g[0] : g[1]]
                if len(new_contour):
                    # Avoid adding empty contours
                    new_contours.append(new_contour)
        return ContourManager(new_contours)

    def filter_and_group_by_length(
        self, max_length: int, min_length: int
    ) -> Tuple["ContourManager", "ContourManager"]:
        """Filter contours by number of coordinates.

        Remove too short contours with fewer than min_length coordinates.
        Group remaining contours by max_length into short and long contours.

        :returns: (short contours, long contours)
        """
        short_contours, long_contours = [], []
        for contour in self.contours:
            if contour.shape[0] > max_length:
                long_contours.append(contour)
            elif contour.shape[0] >= min_length:
                short_contours.append(contour)

        return ContourManager(short_contours), ContourManager(long_contours)

    def group_dashed_contours(
        self,
        max_match_cost: float = 500.0,
        lateral_distance_factor: float = 8.0,
        max_long_distance: float = 300.0,
        max_lat_distance: float = 25.0,
        distance_mode: str = "centers_oriented",
    ) -> Tuple[List[Lanemarking], List[Lanemarking]]:
        """Given short contours, try to group them to dashed lanemarkings.

        For grouping, for each possible pair of contours, an assignment cost is calculated that considers the
        longitudinal and lateral distances.

        :param max_match_cost: Max assignment cost. Can be interpreted as max assignment distance.
        :param lateral_distance_factor: Weight factor for lateral distance in cost calculation
        :param max_long_distance: Max allowed long distance for grouping contours
        :param max_lat_distance: Max allowed lat distance for grouping contours
        :param distance_mode: Method used to calculate the distance between two contours.
        :return: Tuple of Lanemarking lists
        """
        contour_segments = [
            ContourSegment.from_coordinates(contour) for contour in self.contours
        ]

        # Calculate pairwise long & lat distances
        num_segments = len(contour_segments)
        pairwise_distances = np.zeros((num_segments, num_segments, 2))
        for i_contour, contour in enumerate(contour_segments):
            for j_contour, other_contour in enumerate(contour_segments):

                # Avoid matching a contour to itself
                if i_contour == j_contour:
                    pairwise_distances[i_contour, j_contour] = np.array(
                        [np.inf, np.inf]
                    )
                    continue

                if distance_mode == "centers_oriented":
                    distances = contour.oriented_distance(other_contour)
                else:
                    distances = [contour.endpoint_distance(other_contour), 0.0]

                # If no valid distance could be calculated, avoid matching
                if distances[0] == -1.0:
                    distances = [np.inf, np.inf]

                pairwise_distances[i_contour, j_contour] = distances

        # Define matching cost by combination of longitudinal and lateral distance
        cost = (
            pairwise_distances[..., 0]
            + np.abs(pairwise_distances[..., 1]) * lateral_distance_factor
        )

        # Prevent match to itself
        cost[cost == 0.0] = np.inf

        # Only match in forward direction -> Avoid matches with negative long distance
        cost[pairwise_distances[..., 0] < 0] = np.inf

        # Avoid high long distance matches
        cost[pairwise_distances[..., 0] > max_long_distance] = np.inf

        # Avoid high lat distance matches
        cost[np.abs(pairwise_distances[..., 1]) > max_lat_distance] = np.inf

        # For each contour, find best next contour by min cost. Store result in a
        # Connection map:
        # row: i-th contour,
        # col 0: idx of prev contour
        # col 1: idx of next contour
        # (-1, -1) -> not matched
        # (-2, -2) -> invalidated (see below)
        conn = np.zeros((len(contour_segments), 2), int) - 1
        for i_contour, _ in enumerate(contour_segments):
            # Get contour with min cost
            i_contour_best = np.argmin(cost[i_contour])
            if cost[i_contour, i_contour_best] > max_match_cost:
                logger.debug(
                    f"Did not match contour {i_contour} with {i_contour_best} due too large cost "
                    f"{cost[i_contour, i_contour_best]:.2f}"
                )
                continue

            logger.debug(f"Matched contour {i_contour} with {i_contour_best}")
            conn[i_contour, 1] = i_contour_best  # Set next
            conn[i_contour_best, 0] = i_contour  # Set prev

        dashed_lanemarkings = []
        short_lanemarkings = []

        # Based on derived matches in connection map, group contours
        for i_contour in range(conn.shape[0]):
            # Check if start point (== no previous segment)
            if conn[i_contour, 0] != -1:
                continue

            # Create a new group and iterate over all next segments
            new_group = [contour_segments[i_contour]]
            i_current = i_contour
            while conn[i_current, 1] >= 0:  # While there is a valid next segment
                i_next = conn[i_current, 1]  # Get next segment,
                new_group.append(contour_segments[i_next])
                conn[i_current] = [-2, -2]  # Invalidate match to avoid being reused
                i_current = i_next
            conn[i_current] = [-2, -2]

            # Depending on number of segments group into short and dashed lanemarkings
            new_group = [c.coordinates for c in new_group]
            if len(new_group) == 1:
                short_lanemarkings.append(
                    Lanemarking(new_group, Lanemarking.LanemarkingType.DASHED)
                )
            else:
                dashed_lanemarkings.append(
                    Lanemarking(new_group, Lanemarking.LanemarkingType.DASHED)
                )

        return dashed_lanemarkings, short_lanemarkings

    def __iter__(self):
        return self.contours.__iter__()

    def __getitem__(self, item):
        return self.contours.__getitem__(item)

    def __len__(self):
        return len(self.contours)
