import itertools
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
from loguru import logger
from scipy import linalg

from deepaerialmapper.mapping.lanemarking import Lanemarking


class SemanticClass(Enum):
    BLACK = 0
    VEGETATION = 2
    ROAD = 1
    TRAFFICISLAND = 3
    SIDEWALK = 4
    PARKING = 5
    SYMBOL = 6
    LANEMARKING = 7

    @classmethod
    def from_name(cls, name):
        for c in cls:
            if c.name == name:
                return name
        raise ValueError(f"Could not find SemanticClass with name {name}")

    @classmethod
    def from_id(cls, id):
        for c in cls:
            if c.value == id:
                return c
        raise ValueError(f"Could not find SemanticClass with id {id}")


palette_map = {
    SemanticClass.BLACK: [0, 0, 0],  # black
    SemanticClass.ROAD: [128, 128, 128],  # gray
    SemanticClass.SIDEWALK: [0, 0, 255],  # blue
    SemanticClass.TRAFFICISLAND: [153, 51, 255],  # purple
    SemanticClass.PARKING: [255, 255, 0],  # yellow
    SemanticClass.VEGETATION: [0, 255, 0],  # green
    SemanticClass.LANEMARKING: [0, 128, 128],  # cyan
    SemanticClass.SYMBOL: [255, 0, 0],  # red
}


@dataclass
class ClassMask:
    """Binary segmentation mask"""

    mask: np.ndarray
    class_names: List[SemanticClass]

    @property
    def shape(self) -> Tuple:
        return self.mask.shape

    def astype(self, dtype) -> np.ndarray:
        return self.mask.astype(dtype)

    def __getitem__(self, item):
        return self.mask.__getitem__(item)

    def as_color_img(self, scale_factor=255, dtype=np.uint8) -> np.ndarray:
        scaled_mask = self.mask * scale_factor
        scaled_mask = scaled_mask.astype(dtype)
        scaled_mask = np.repeat(scaled_mask[:, :, np.newaxis], 3, axis=2)
        return scaled_mask

    def to_file(self, filepath: Path) -> None:
        cv2.imwrite(str(filepath), self.mask.astype(np.uint8) * 255)

    def show_points(self, points):
        img = self.as_color_img()

        import matplotlib.pyplot as plt

        for point in points:
            # Draw line
            img = cv2.circle(img, (point[1], point[0]), 3, (255, 0, 0), thickness=-1)

        plt.imshow(img)
        plt.show()

    def show(
        self,
        contours=None,
        lanemarkings=None,
        background=None,
        show=True,
        window_name: str = "",
        random: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        from deepaerialmapper.mapping.contour import ContourManager

        if isinstance(contours, ContourManager):
            contours = contours.contours

        img = self.as_color_img()

        img_c = None
        img_overlay = None
        import matplotlib.pyplot as plt

        if show:
            plt.imshow(img)
        if contours is not None or lanemarkings is not None:
            if background is not None:
                img_c = background.copy()
            else:
                img_c = img.copy()

            if contours is not None:
                for contour in contours:

                    if random:
                        line_color = np.random.randint(256, size=3).tolist()
                    else:
                        line_color = (150, 255, 0)

                    # Draw line
                    img_c = cv2.polylines(
                        img_c,
                        [contour],
                        isClosed=False,
                        color=line_color,
                        thickness=19,
                        lineType=cv2.LINE_AA,
                    )

                    # Draw intermediate points
                    for i in range(contour.shape[0]):
                        if i == 0:
                            color = (255, 0, 0)  # red
                        elif i == contour.shape[0] - 1:
                            color = (0, 0, 0)  # black
                        else:
                            color = (0, 0, 255)  # blue

                        img_c = cv2.circle(img_c, contour[i, 0], 3, color, thickness=-1)

            if lanemarkings is not None:
                for lanemarking in lanemarkings:
                    contour = lanemarking.contour

                    if random:
                        color = np.random.randint(256, size=3).tolist()
                    elif lanemarking.type_ == Lanemarking.LanemarkingType.ROAD_BORDER:
                        color = (255, 0, 255)
                    elif lanemarking.type_ == Lanemarking.LanemarkingType.SOLID:
                        color = (0, 128, 255)
                    else:
                        color = (150, 255, 0)

                    # Draw line
                    img_c = cv2.polylines(
                        img_c,
                        [contour],
                        isClosed=False,
                        color=color,
                        thickness=3,
                        lineType=cv2.LINE_AA,
                    )

                    # Draw intermediate points
                    for i in range(contour.shape[0]):
                        if i == 0:
                            # start point
                            color = (255, 0, 0)  # red
                            size = 4
                        elif i == contour.shape[0] - 1:
                            # end point
                            color = (0, 0, 0)  # black
                            size = 4
                        else:
                            color = (0, 0, 255)  # blue
                            size = 3

                        img_c = cv2.circle(
                            img_c, contour[i, 0], size, color, thickness=-1
                        )

            img_overlay = cv2.addWeighted(img, 0.5, img_c, 0.5, 0.0)
            if show:
                plt.imshow(img_overlay)

        if show:
            figManager = plt.get_current_fig_manager()
            figManager.window.state("zoomed")
            if window_name:
                fig = plt.gcf()
                fig.canvas.set_window_title(window_name)
            plt.show()

        return img_c, img_overlay

    def blur_and_close(self, blur_size: int, border_effect: int = 8) -> "ClassMask":
        """
        blur_size : the size of kernel size of median filter
        border_effect : the value to decrease the aliasing on the edge of image.
                        if no value is given, no anti-aliasing happens.
        """
        is_bool = self.mask.dtype == bool
        mask = self.mask.astype(np.uint8)

        if border_effect:
            edge_size = blur_size // 2

            mask_center = cv2.medianBlur(mask, blur_size, 0)
            mask = cv2.medianBlur(mask, blur_size // border_effect, 0)

            # minimize the aliasing after blur
            mask[edge_size:-edge_size, edge_size:-edge_size] = mask_center[
                edge_size:-edge_size, edge_size:-edge_size
            ]

        else:
            mask = cv2.medianBlur(mask, blur_size, 0)

        # morphology: delete tiny dots inside
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        if is_bool:
            mask = mask.astype(bool)

        return ClassMask(mask, self.class_names)

    def erode(self, erode_size: int) -> "ClassMask":
        is_bool = self.mask.dtype == bool

        if is_bool:
            mask = self.mask.astype(np.uint8)

        # morphology: erode
        kernel = np.ones((erode_size, erode_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)

        if is_bool:
            mask = mask.astype(bool)

        return ClassMask(mask, self.class_names)

    def dilate(self, dilate_size: int) -> "ClassMask":
        is_bool = self.mask.dtype == bool

        if is_bool:
            mask = self.mask.astype(np.uint8)

        # morphology: erode
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        if is_bool:
            mask = mask.astype(bool)

        return ClassMask(mask, self.class_names)

    def remove(self, ignore_regions: List) -> "ClassMask":
        """

        Args:
            ignore_regions: List of regions in via-json style

        Returns:

        """
        new_mask = np.copy(self.mask).astype(np.int32)

        logger.debug(f"Removing {len(ignore_regions)} regions")

        for region in ignore_regions:
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

        return ClassMask(new_mask > 0, self.class_names)

    def thin(self) -> "ClassMask":
        new_mask = cv2.ximgproc.thinning(self.mask.astype(np.uint8) * 255) == 255
        return ClassMask(new_mask, self.class_names)

    def find_split_points(self, debug=False) -> Tuple["ClassMask", List[np.ndarray]]:
        """Eliminate splitpoint from mask"""

        # Find all split points
        split_ptrs = []
        idx_ys, idx_xs = np.where(self.mask)
        for idx_y, idx_x in zip(idx_ys, idx_xs):
            if self.is_split_point(self.mask, idx_y, idx_x):
                split_ptrs.append(np.array([idx_x, idx_y]))

        # Remove split points from contour mask
        new_mask = np.copy(self.mask)
        for idx_x, idx_y in split_ptrs:
            for (nb_idx_y, nb_idx_x) in [
                (idx_y + dy, idx_x + dx)
                for (dy, dx) in itertools.product([-1, 0, 1], [-1, 0, 1])
            ]:
                if (
                    (nb_idx_y < 0)
                    or (nb_idx_x < 0)
                    or (nb_idx_y >= self.shape[0])
                    or (nb_idx_x >= self.shape[1])
                ):
                    # if index is out of image, then skip
                    continue
                new_mask[nb_idx_y, nb_idx_x] = 0

        # Detect groups of split points and keep only one per group
        filtered_split_ptrs = []
        visited = [False] * len(split_ptrs)
        distance_threshold = 5
        for i_split_ptr, split_ptr in enumerate(split_ptrs):
            if visited[i_split_ptr]:
                continue

            close_ptrs = [split_ptr]

            for j_split_ptr in range(i_split_ptr, len(split_ptrs)):
                if visited[j_split_ptr]:
                    continue

                other_point = split_ptrs[j_split_ptr]
                # Skip all points too close to current one
                if linalg.norm(split_ptr - other_point) < distance_threshold:
                    visited[j_split_ptr] = True
                    close_ptrs.append(other_point)

            filtered_split_ptrs.append(
                np.round(np.mean(close_ptrs, axis=0)).astype(int)
            )
            visited[i_split_ptr] = True

        logger.debug(
            f"Removed {len(split_ptrs) - len(filtered_split_ptrs)} split points due to proximity."
        )
        logger.debug(
            f"Found {len(filtered_split_ptrs)} remaining split points:\n{np.asarray(filtered_split_ptrs)}"
        )

        # Visualize for debugging
        if debug:
            drawing = cv2.cvtColor(new_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)

            for idx_x, idx_y in filtered_split_ptrs:
                for (nb_idx_y, nb_idx_x) in [
                    (idx_y + dy, idx_x + dx)
                    for (dy, dx) in itertools.product([-1, 0, 1], [-1, 0, 1])
                ]:
                    if (
                        (nb_idx_y < 0)
                        or (nb_idx_x < 0)
                        or (nb_idx_y >= self.shape[0])
                        or (nb_idx_x >= self.shape[1])
                    ):
                        # if index is out of image, then skip
                        continue
                    drawing[nb_idx_y, nb_idx_x] = (255, 0, 0)

            import matplotlib.pyplot as plt

            plt.imshow(drawing)
            plt.show()

        return ClassMask(new_mask, self.class_names), filtered_split_ptrs

    @staticmethod
    def is_split_point(contour_mask: np.ndarray, y: int, x: int) -> bool:
        # Extract 3x3 contour around x,y
        # Easy case: Not at the mask border
        if 1 <= x <= contour_mask.shape[1] and 1 <= y <= contour_mask.shape[0]:
            contour_mask = contour_mask[y - 1 : y + 2, x - 1 : x + 2]
        else:
            c = np.zeros((3, 3))
            # Complex case at mask border
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
            # check_neighbours(locations2check, x, y, dx, dy, contour_mask, is_visited)

        return num_lines > 2

    def union(self, other_mask: "ClassMask") -> "ClassMask":
        mask = np.bitwise_or(self.mask, other_mask.mask)
        class_names = list(chain(self.class_names, other_mask.class_names))
        return ClassMask(mask, class_names)

    def intersection(self, other_mask: "ClassMask") -> "ClassMask":
        mask = np.bitwise_and(self.mask, other_mask.mask)
        return ClassMask(
            mask, self.class_names
        )  # TODO: class names don't make sense here anymore?

    def subtraction(self, other_mask: "ClassMask") -> "ClassMask":
        """Remove pixels of other mask from this mask"""
        mask = np.bitwise_and(self.mask, ~other_mask.mask)
        return ClassMask(
            mask, self.class_names
        )  # TODO: class names don't make sense here anymore?

    def contours(self, method=cv2.CHAIN_APPROX_SIMPLE) -> np.ndarray:
        """
        Args:
            method: 1 - no simplification
                    2 - approximate (default)
        """
        # contours, _ = cv2.findContours(self.astype(np.uint8), cv2.RETR_EXTERNAL, method)
        contours, _ = cv2.findContours(self.astype(np.uint8), cv2.RETR_LIST, method)
        return contours


@dataclass
class SegmentationMask:
    """Multi-Class segmentation mask"""

    mask: np.ndarray
    palette: Dict[SemanticClass, List[int]]

    @property
    def shape(self) -> Tuple:
        return self.mask.shape

    def __getitem__(self, item):
        return self.mask.__getitem__(item)

    def class_mask(
        self, class_names: Union[SemanticClass, List[SemanticClass]]
    ) -> ClassMask:
        """Returns a binary mask for one or more classes"""
        # Convert to list in case only a single class name is given
        if isinstance(class_names, SemanticClass):
            class_names = [class_names]

        output_mask = np.zeros(self.mask.shape[:2], dtype=bool)
        for class_name in class_names:
            output_mask = np.bitwise_or(
                output_mask, np.all(self.mask == self.palette[class_name], axis=2)
            )

        return ClassMask(output_mask, class_names)

    @staticmethod
    def from_file(filepath: Path, palette: Dict[str, List[int]]) -> "SegmentationMask":
        seg_mask_img = cv2.cvtColor(cv2.imread(str(filepath)), cv2.COLOR_BGR2RGB)
        return SegmentationMask(seg_mask_img, palette)

    def to_file(self, filepath: Path):
        cv2.imwrite(str(filepath), self.mask)

    def show(self):
        import matplotlib.pyplot as plt

        plt.imshow(self.mask)
        plt.show()
