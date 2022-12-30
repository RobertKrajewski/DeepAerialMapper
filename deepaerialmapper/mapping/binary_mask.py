import itertools
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from loguru import logger
from scipy import linalg

IgnoreRegion = Dict[str, Any]


@dataclass
class BinaryMask:
    """Binary segmentation mask"""

    mask: np.ndarray
    class_names: List["SemanticClass"]

    @property
    def shape(self) -> Tuple:
        return self.mask.shape

    def astype(self, dtype: np.dtype) -> np.ndarray:
        return self.mask.astype(dtype)

    def __getitem__(self, item) -> np.ndarray:
        """Pixel access"""
        return self.mask.__getitem__(item)

    def as_color_img(
        self, scale_factor: float = 255.0, dtype: np.dtype = np.uint8
    ) -> np.ndarray:
        """Convert mask to color image."""
        scaled_mask = self.mask * scale_factor
        scaled_mask = scaled_mask.astype(dtype)
        scaled_mask = np.repeat(scaled_mask[:, :, np.newaxis], 3, axis=2)
        return scaled_mask

    def to_file(self, filepath: Path) -> None:
        """Store mask as image."""
        cv2.imwrite(str(filepath), self.mask.astype(np.uint8) * 255)

    def median_blur(
        self, blur_size: int, border_blur_size_divisor: int = 8
    ) -> "BinaryMask":
        """Apply median blur and closing to remove noise.

        Supports using a reduced filter size in border area to avoid side effects.

        :param blur_size: Blur filter size.
        :param border_blur_size_divisor: Blur size is reduced by this factor in border region. 0 to deactivate.
        :return: Resulting ClassMask
        """
        is_bool = self.mask.dtype == bool
        mask = self.mask.astype(np.uint8)

        if border_blur_size_divisor:
            edge_size = blur_size // 2

            mask_center = cv2.medianBlur(mask, blur_size, 0)
            mask = cv2.medianBlur(mask, blur_size // border_blur_size_divisor, 0)

            mask[edge_size:-edge_size, edge_size:-edge_size] = mask_center[
                edge_size:-edge_size, edge_size:-edge_size
            ]

        else:
            mask = cv2.medianBlur(mask, blur_size, 0)

        if is_bool:
            mask = mask.astype(bool)

        return BinaryMask(mask, self.class_names)

    def close(self, close_size: int) -> "BinaryMask":
        return self._morph(close_size, cv2.MORPH_CLOSE)

    def erode(self, erode_size: int) -> "BinaryMask":
        return self._morph(erode_size, cv2.MORPH_ERODE)

    def dilate(self, dilate_size: int) -> "BinaryMask":
        return self._morph(dilate_size, cv2.MORPH_DILATE)

    def _morph(self, size: int, operation: int) -> "BinaryMask":
        """Apply a morphological operation.

        :param size: Filter size
        :param operation: Operation type, e.g. cv.MORPH_ERODE
        :return: Resulting ClassMask
        """
        is_bool = self.mask.dtype == bool

        if is_bool:
            mask = self.mask.astype(np.uint8)

        kernel = np.ones((size, size), np.uint8)
        mask = cv2.morphologyEx(mask, operation, kernel)

        if is_bool:
            mask = mask.astype(bool)

        return BinaryMask(mask, self.class_names)

    def remove_regions(self, regions: List[IgnoreRegion]) -> "BinaryMask":
        """Remove rect and polygon regions from mask.
        :param regions: Via-style region annotations
        :return: Resulting ClassMask
        """
        if not regions:
            return BinaryMask(np.copy(self.mask), self.class_names)

        new_mask = np.copy(self.mask).astype(np.int32)

        logger.debug(f"Removing {len(regions)} regions")
        for region in regions:
            shape = region["shape_attributes"]
            if shape["name"] == "polygon":
                xcoords = shape["all_points_x"]
                ycoords = shape["all_points_y"]
            elif shape["name"] == "rect":
                xcoords = [
                    shape["x"] - shape["width"],
                    shape["x"] - shape["width"],
                    shape["x"] + shape["width"],
                    shape["x"] + shape["width"],
                ]
                ycoords = [
                    shape["y"] - shape["height"],
                    shape["y"] + shape["height"],
                    shape["y"] + shape["height"],
                    shape["y"] - shape["height"],
                ]

            polygon = (
                np.column_stack([xcoords, ycoords]).reshape((-1, 1, 2)).astype(np.int32)
            )

            cv2.fillPoly(new_mask, [polygon], 0)

        return BinaryMask(new_mask > 0, self.class_names)

    def thin(self) -> "BinaryMask":
        """Apply thinning to the mask to retrieve e.g. the skeleton of thick lines."""
        new_mask = cv2.ximgproc.thinning(self.mask.astype(np.uint8) * 255) == 255
        return BinaryMask(new_mask, self.class_names)

    def remove_split_points(
        self, merging_distance: int = 5, debug: bool = False
    ) -> Tuple["BinaryMask", List[np.ndarray]]:
        """Find and remove split points from mask.

        Split points typically occur when lanes start/end/split/merge as the number of required lanemarkings changes.
        This function finds these locations in the mask and removes them. This allows to describe the lanemarkings by
        more than one contour and lanemarking.

        Additionally allows to merge split points that are very close (<`merging_distance`).

        :param merging_distance: Distance threshold for merging split points. 0 to deactivate merging.
        :param debug: If true, an interactive plot windows shows the found split points.
        :return: Tuple of 1) Mask without (unmerged) split points, 2) List of split points after merging (x,y)
        """

        # Find all split points
        split_points = []
        for y, x in zip(*np.where(self.mask)):
            if self._is_split_location(self.mask, y, x):
                split_points.append(np.array([x, y]))

        # Remove split points (and their 3x3 neighbours) from contour mask
        new_mask = np.copy(self.mask)
        for split_x, split_y in split_points:
            for (x, y) in [
                (split_y + dy, split_x + dx)
                for (dy, dx) in itertools.product([-1, 0, 1], [-1, 0, 1])
            ]:
                if (y < 0) or (x < 0) or (y >= self.shape[0]) or (x >= self.shape[1]):
                    # Location out of image
                    continue
                new_mask[y, x] = 0

        if merging_distance > 0:
            # Detect groups of split points and keep only one per group
            grouped_split_points = []
            visited = [False] * len(split_points)
            for i_split_ptr, split_ptr in enumerate(split_points):
                if visited[i_split_ptr]:
                    continue

                close_ptrs = [split_ptr]

                for j_split_ptr in range(i_split_ptr, len(split_points)):
                    if visited[j_split_ptr]:
                        continue

                    other_point = split_points[j_split_ptr]
                    # Skip all points too close to current one
                    if linalg.norm(split_ptr - other_point) < merging_distance:
                        visited[j_split_ptr] = True
                        close_ptrs.append(other_point)

                grouped_split_points.append(
                    np.round(np.mean(close_ptrs, axis=0)).astype(int)
                )
                visited[i_split_ptr] = True

            logger.debug(
                f"Removed {len(split_points) - len(grouped_split_points)} split points due to proximity."
            )
        else:
            grouped_split_points = split_points
        logger.debug(
            f"Found {len(grouped_split_points)} (remaining split) points:\n{np.asarray(grouped_split_points)}"
        )

        # Debugging visualization
        if debug:
            drawing = cv2.cvtColor(new_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
            for x, y in split_points:
                drawing[y, x] = (0, 0, 255)

            for x, y in grouped_split_points:
                drawing[y, x] = (255, 0, 0)

            import matplotlib.pyplot as plt
            plt.imshow(drawing)
            plt.show()

        return BinaryMask(new_mask, self.class_names), grouped_split_points

    @staticmethod
    def _is_split_location(contour_mask: np.ndarray, y: int, x: int) -> bool:
        """Determine if contour splits at a given location.

        A contour splits at a given location, if more than two disconnected (1-connectivity) lines start
        in its 3x3 neighbourhood.

        Examples:
        At location X three disconnected groups of 1's start -> X is a fork location
        01100
        00X10
        11100

        At location X two disconnected groups of 1's start -> X is NOT a fork location
        01111
        00X10
        11100

        Note: This code was benchmark against skimage.measure.label(..., connectivity=1)

        :param contour_mask: Mask containing contours, e.g. of lanemarkings
        :param y: y-coordinate of the point to check
        :param x: x-coordinate of the point to check
        :return: True, if more than two independent branches start from this point
        """
        # Extract 3x3 contour around x,y
        if 1 <= x <= contour_mask.shape[1] and 1 <= y <= contour_mask.shape[0]:
            # Easy case: Not at the mask border
            contour_mask = contour_mask[y - 1 : y + 2, x - 1 : x + 2]
        else:
            # Complex case at mask border
            c = np.zeros((3, 3))
            # Check and copy each neighbour individually
            for dx, dy in itertools.product([-1, 0, 1], [-1, 0, 1]):
                if (
                    y + dy < 0
                    or x + dx < 0
                    or y + dy >= contour_mask.shape[0]
                    or x + dx >= contour_mask.shape[1]
                ):
                    continue
                c[1 + dy, 1 + dx] = contour_mask[y + dy, x + dx]
            contour_mask = c

        # Change x and y to center of the crop
        x = 1
        y = 1

        # For at least 3 lines to exist, we need more than 4 points in the mask
        if contour_mask.sum() <= 3:
            return False

        is_visited = np.zeros_like(
            contour_mask, bool
        )  # Remember which neighbours we already visited
        num_lines = 0  # How many lines start from this point

        # Hard coding all the neighbours that need to be checked
        locations2check = {
            # (dx, dy) of a neighbour  -> List[(dy, dy)] of neighbour's neighbours
            # dx and dy are always relative to the point of interest (x, y)
            (-1, 0): [(-1, -1), (-1, +1)],
            (+1, 0): [(+1, -1), (+1, +1)],
            (0, -1): [(-1, -1), (+1, -1)],
            (0, +1): [(-1, +1), (+1, +1)],
            (-1, -1): [(-1, 0), (0, -1)],
            (-1, +1): [(-1, 0), (0, +1)],
            (+1, -1): [(+1, 0), (0, -1)],
            (+1, +1): [(+1, 0), (0, +1)],
        }

        # Iterate over every direct neighbour
        for dx, dy in itertools.product([-1, 0, 1], [-1, 0, 1]):
            if dx == 0 and dy == 0:
                # Center point
                continue

            # Check if we are at border
            if (
                y + dy < 0
                or x + dx < 0
                or y + dy >= contour_mask.shape[0]
                or x + dx >= contour_mask.shape[1]
            ):
                continue

            # Visit every neighbour only once
            if is_visited[y + dy, x + dx]:
                continue

            is_visited[y + dy, x + dx] = True

            if contour_mask[y + dy, x + dx] == 0:
                # No new line starts here
                continue

            # We have found the start of a new line
            num_lines += 1

            # Recursively check the neighbours as they would belong to the same line
            def check_neighbours(dx, dy):
                locations = locations2check[(dx, dy)]
                for dx2, dy2 in locations:
                    new_x = x + dx2
                    new_y = y + dy2

                    if (
                        new_y < 0
                        or new_x < 0
                        or new_y >= contour_mask.shape[0]
                        or new_x >= contour_mask.shape[1]
                    ):
                        continue

                    if contour_mask[new_y, new_x] and not is_visited[new_y, new_x]:
                        is_visited[new_y, new_x] = True
                        check_neighbours(dx2, dy2)

            check_neighbours(dx, dy)

        return num_lines > 2

    def union(self, other_mask: "BinaryMask") -> "BinaryMask":
        mask = np.bitwise_or(self.mask, other_mask.mask)
        class_names = list(chain(self.class_names, other_mask.class_names))
        return BinaryMask(mask, class_names)

    def intersection(self, other_mask: "BinaryMask") -> "BinaryMask":
        mask = np.bitwise_and(self.mask, other_mask.mask)
        return BinaryMask(
            mask, self.class_names
        )  # TODO: class names don't make sense here anymore?

    def subtraction(self, other_mask: "BinaryMask") -> "BinaryMask":
        """Remove pixels of other mask from this mask"""
        mask = np.bitwise_and(self.mask, ~other_mask.mask)
        return BinaryMask(
            mask, self.class_names
        )  # TODO: class names don't make sense here anymore?

    def contours(self, method: int = cv2.CHAIN_APPROX_SIMPLE) -> List[np.ndarray]:
        """Retrieve all contours.

        :param method: OpenCV.findContours approximation method
        :return: List of OpenCV contours of shape (N,1,2)
        """
        contours, _ = cv2.findContours(self.astype(np.uint8), cv2.RETR_LIST, method)
        return contours
