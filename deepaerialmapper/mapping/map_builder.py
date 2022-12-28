from dataclasses import dataclass
from typing import List

import cv2
from loguru import logger

from deepaerialmapper.mapping.contour import ContourManager
from deepaerialmapper.mapping.lanemarking import Lanemarking
from deepaerialmapper.mapping.masks import SemanticClass
from deepaerialmapper.visualization.mask_visualizer import MaskVisualizer
from deepaerialmapper.mapping import Map, SymbolDetector, Symbol


@dataclass
class ContourExtractor:
    subsample_factor: int = 30

    def from_mask(self, seg_mask, ignore_regions):
        # Find road borders
        road_mask = seg_mask.class_mask(
            [SemanticClass.ROAD, SemanticClass.SYMBOL, SemanticClass.LANEMARKING]
        ).blur_and_close(15)

        road_contours = (
            ContourManager.from_mask(road_mask)
            .split_at_border(road_mask.shape)
            .subsample(self.subsample_factor)
        )  # .split_sharp_corners_v2()   #.split_sharp_corners(angle_threshold=50, length_threshold=100)

        # Find lane markings
        lanemarking_mask = seg_mask.class_mask(SemanticClass.LANEMARKING)

        # Only allow lanemarkings far away from road borders
        road_trafficisland_mask = road_mask.union(
            seg_mask.class_mask(SemanticClass.TRAFFICISLAND)
        )
        lanemarking_mask = (
            lanemarking_mask.intersection(
                road_trafficisland_mask.blur_and_close(3, border_effect=1).erode(35)
            )
            .erode(3)
            .dilate(3)
        ).remove(
            ignore_regions
        )  # Extra erode and dilate to handle pointy lanemarkings after intersection

        # As lanemarkings are THICK, thin them to get a line only
        lanemarking_mask = lanemarking_mask.thin()
        lanemarking_mask, split_points = lanemarking_mask.find_split_points(debug=False)
        lanemarking_contours = (
            ContourManager.from_mask(lanemarking_mask)
            .unique_coordinates()
            .subsample(self.subsample_factor)
            .group_at_split_points(split_points)
        )  # .split_sharp_corners_v2() \
        # .split_sharp_corners(angle_threshold=50, length_threshold=10)

        return lanemarking_contours, lanemarking_mask, road_contours


class MapBuilder:
    def __init__(
        self,
        contour_extractor: ContourExtractor,
        symbol_detector: SymbolDetector,
        skip_lanelets=True,
        skip_symbols=False,
        debug_dir=None,
    ):
        self._contour_extractor = contour_extractor
        self._symbol_detector = symbol_detector

        self._skip_lanelets = skip_lanelets
        self._skip_symbols = skip_symbols
        self._debug_dir = debug_dir

    @staticmethod
    def derive_lanemarkings(
        road_borders: ContourManager, lanemarking_contours: ContourManager, img_shape
    ) -> List[Lanemarking]:
        # Filter lanemarkings by length, only merge short ones
        (
            short_lanemarking_contours,
            solid_lanemarking_contours,
        ) = lanemarking_contours.filter_by_length(max_length=5, min_length=2)
        # short_lanemarking_contours.show()

        # Create solid lanemarkings
        solid_lanemarkings = [
            Lanemarking([c], Lanemarking.LanemarkingType.SOLID)
            for c in solid_lanemarking_contours
        ]

        # Created dashed lanemarkings
        dashed_lanemarking, short_lanemarking = short_lanemarking_contours.group(
            debug=False
        )

        # Fill gaps between dashed and solid lanemarkings
        dashed_lanemarking = Lanemarking.extend_to(
            dashed_lanemarking, solid_lanemarkings, img_shape
        )

        road_borders = [
            Lanemarking([c], Lanemarking.LanemarkingType.ROAD_BORDER)
            for c in road_borders
        ]

        lanemarkings = [*road_borders, *solid_lanemarkings, *dashed_lanemarking]
        return lanemarkings

    def from_semantic_mask(
        self,
        seg_mask,
        ignore_regions: List,
        proj,
        origin,
        px2m,
        debug_prefix="",
    ) -> "Map":

        (
            lanemarking_contours,
            lanemarking_mask,
            road_contours,
        ) = self._contour_extractor.from_mask(seg_mask, ignore_regions)

        img_ref, img_overlay = MaskVisualizer().show(
            lanemarking_mask,
            lanemarking_contours.merge(road_contours),
            background=seg_mask.mask,
            show=False,
            random=True,
        )

        if self._debug_dir:
            cv2.imwrite(
                str(self._debug_dir / f"{debug_prefix}lanemarkings.png"),
                cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR),
            )

        lanemarkings = self.derive_lanemarkings(
            road_contours, lanemarking_contours, lanemarking_mask.shape
        )

        if self._debug_dir:
            _, img_overlay = lanemarking_mask.show(
                lanemarkings=lanemarkings,
                background=seg_mask.mask,
                show=False,
                window_name=debug_prefix,
                random=True,
            )
            cv2.imwrite(
                str(self._debug_dir / f"{debug_prefix}lanemarkings_post.png"),
                cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR),
            )

        # Derive lanelets
        if self._skip_lanelets:
            lanelets = set()
            logger.warning(f"Skipping lanelet matching")
        else:
            lanelets = self.derive_lanelets(img_ref, seg_mask, lanemarkings, px2m)
            logger.info(f"Found {len(lanelets)} lanelets")

        # Find all symbols
        if self._skip_symbols:
            symbols: List[Symbol] = []
            logger.warning(f"Skipping symbol detection!")
        else:
            symbols = self._symbol_detector.detect(seg_mask)
            logger.info(f"Detected {len(symbols)} symbols")

        logger.info("Map extraction finished!")
        return Map(lanemarkings, lanelets, symbols, proj, origin, px2m)
