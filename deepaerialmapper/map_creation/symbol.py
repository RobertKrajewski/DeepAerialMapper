import glob
from dataclasses import dataclass
import os

import cv2
import numpy as np
from typing import Optional, List

from deepaerialmapper.map_creation.masks import SemanticClass, SegmentationMask
from deepaerialmapper.classification.symbol import classify
from loguru import logger

import matplotlib.pyplot as plt


@dataclass
class Symbol:
    name: str
    contour: np.ndarray
    ref: List[List]
    box: Optional[np.ndarray] = None
    score: Optional[float] = None
    image: Optional[np.ndarray] = None

    def show(self):
        max_x, max_y = np.max(self.contour, axis=(0, 1))
        img = np.zeros((max_y + 10, max_x + 10), np.uint8)
        img = cv2.polylines(
            img, [self.contour], isClosed=True, color=(255, 255, 255), thickness=2
        )

        import matplotlib.pyplot as plt

        plt.imshow(img)
        plt.show()


class SymbolDetector:
    def __init__(self, pattern_type: List):

        self.patterns = pattern_type  # order of the list is important.
        logger.info(f"Loaded patterns: {self.patterns}")

    def _load_symbols(self):
        """Load symbols from disk"""

        symbols = []
        for pattern_type in self.pattern_types:

            filepaths = glob.glob(
                os.path.join(self.pattern_dir, pattern_type) + "/*.png"
            )  # if pattern is not png, needs to be changed.
            for filepath in filepaths:
                mask = cv2.imread(filepath)

                # Reduce the noise in the template
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(mask, 150, 255, 0)
                mask = cv2.resize(mask, dsize=None, fx=7.0 / 7.0, fy=7.0 / 7.0)

                # Smooth the contour of the template
                cnts, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                epsilon = 0.008 * cv2.arcLength(cnts[0], True)
                approx = cv2.approxPolyDP(cnts[0], epsilon, True)

                ref_point = self.extract_ref(approx)

                symbols.append(
                    Symbol(name=pattern_type, contour=approx, image=mask, ref=ref_point)
                )

        return symbols

    def extract_ref(self, cnts, debug=False):
        """Extract 2 reference points from contour"""

        epsilon = 0.008 * cv2.arcLength(cnts, True)
        approx = cv2.approxPolyDP(cnts, epsilon, True)

        far_dist = 0
        for i, point_i in enumerate(approx):
            for j, point_j in enumerate(approx):
                if i >= j:
                    continue

                dist_ij = np.linalg.norm(point_i - point_j)
                if dist_ij > far_dist:
                    far_dist = dist_ij
                    ref_points = [point_i, point_j]

        if debug:
            mask = np.zeros(
                np.append(np.squeeze(np.flip(np.amax(approx, axis=0), axis=None)), 3)
            )  # zeros(y, x, 3)
            cv2.drawContours(mask, [approx], -1, (0, 255, 0), 1)

            mask = cv2.circle(mask, ref_points[0][0], 2, (255, 0, 0), thickness=-1)
            mask = cv2.circle(mask, ref_points[1][0], 2, (255, 0, 0), thickness=-1)
            plt.imshow(mask, cmap="jet")
            plt.show()

        return ref_points

    def detect_patterns(
        self,
        seg_mask: SegmentationMask,
        cls_weight,
        min_area: int = 1000,
        max_area: int = 6000,
        debug: bool = False,
        dbg_rescale: float = 0.75,
    ) -> List[Symbol]:
        """
        using shape match to classify the detected symbol
        load symbol template and compare the detected ones with them, find the most likely one.
        """

        # Using image closing to merge the closely separated segments
        symbol_mask = seg_mask.class_mask([SemanticClass.SYMBOL]).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        symbol_mask = cv2.morphologyEx(symbol_mask, cv2.MORPH_CLOSE, kernel)

        if debug:
            symbol_mask = cv2.resize(symbol_mask, None, fx=dbg_rescale, fy=dbg_rescale)

        _, mask = cv2.threshold(symbol_mask, 150, 255, 0)

        symbol_cnts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for cnt in symbol_cnts:
            # determine the type of arrow by pattern matching
            scores = []

            epsilon = 0.001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            rect = cv2.minAreaRect(approx)  # Calculate rotated box with minimum area
            box = cv2.boxPoints(rect).astype(int)

            # Check for appropriate size
            area = cv2.contourArea(box)
            if area < min_area or area > max_area:
                continue

            ref_point = self.extract_ref(approx)

            # svm doesn't need to go through all patterns.
            best_name = classify(cnt, seg_mask, self.patterns, cls_weight)
            detections.append(
                Symbol(name=best_name, contour=approx, ref=ref_point, box=box)
            )

        return detections
