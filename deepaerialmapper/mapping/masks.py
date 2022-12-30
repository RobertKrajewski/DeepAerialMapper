from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np

from deepaerialmapper.mapping import ClassMask

Palette = Dict["SemanticClass", Tuple[int, int, int]]


class SemanticClass(Enum):
    """Enum of all supported semantic segmentation classes."""

    BLACK = 0
    VEGETATION = 2
    ROAD = 1
    TRAFFICISLAND = 3
    SIDEWALK = 4
    PARKING = 5
    SYMBOL = 6
    LANEMARKING = 7

    @classmethod
    def from_name(cls, name: str) -> "SemanticClass":
        for c in cls:
            if c.name == name:
                return name
        raise ValueError(f"Could not find SemanticClass with name {name}")

    @classmethod
    def from_id(cls, id: int) -> "SemanticClass":
        for c in cls:
            if c.value == id:
                return c
        raise ValueError(f"Could not find SemanticClass with id {id}")


@dataclass
class SegmentationMask:
    """Multi-Class segmentation mask

    The segmentation mask is represented by an RGB image. Each class is by a unique color as described by the palette.
    """

    mask: np.ndarray
    palette: Palette

    @property
    def shape(self) -> Tuple:
        return self.mask.shape

    def __getitem__(self, item):
        """Pixel access"""
        return self.mask.__getitem__(item)

    def class_mask(
        self, class_names: Union[SemanticClass, List[SemanticClass]]
    ) -> ClassMask:
        """Create a binary mask of a selection of classes

        :param class_names: Single class name or list of class names to extract
        :return: ClassMask of the union of the selected class masks.
        """
        # Convert to list in case only a single class name is given
        if isinstance(class_names, SemanticClass):
            class_names = [class_names]

        output_mask = np.zeros(self.mask.shape[:2], dtype=bool)
        for class_name in class_names:
            output_mask = np.bitwise_or(
                output_mask, np.all(self.mask == self.palette[class_name], axis=2)
            )

        return ClassMask(output_mask, class_names)

    @classmethod
    def from_file(cls, filepath: Path, palette: Palette) -> "SegmentationMask":
        """Load semantic segmentation mask from RGB image.

        :param filepath: Path of image file to load.
        :param palette: Palette which maps between class and RGB colors.
        :return: Loaded segmentation mask.
        """
        seg_mask_img = cv2.cvtColor(cv2.imread(str(filepath)), cv2.COLOR_BGR2RGB)
        return cls(seg_mask_img, palette)

    def to_file(self, filepath: Path):
        cv2.imwrite(str(filepath), self.mask)

    def show(self):
        import matplotlib.pyplot as plt

        plt.imshow(self.mask)
        plt.show()
