from dataclasses import dataclass
from typing import List, Tuple

import cv2
from loguru import logger

from deepaerialmapper.mapping.binary_mask import BinaryMask, IgnoreRegion
from deepaerialmapper.mapping.contour_manager import ContourManager
from deepaerialmapper.mapping.lanelet import derive_lanelets
from deepaerialmapper.mapping.lanemarking import Lanemarking
from deepaerialmapper.mapping.map import Map
from deepaerialmapper.mapping.semantic_mask import SemanticClass, SemanticMask
from deepaerialmapper.mapping.symbol import Symbol, SymbolDetector
from deepaerialmapper.visualization.mask_visualizer import MaskVisualizer


@dataclass
class LanemarkingExtractor:
    filter_min_length: int = 2
    filter_max_length: int = 5
    dashed_max_match_cost: float = 500.0
    dashed_lateral_distance_factor: float = 8.0
    dashed_max_long_distance: float = 300.0
    dashed_max_lat_distance: float = 25.0
    extend_check_border: bool = False
    extend_max_long_distance: float = 300.0
    extend_max_lat_distance: float = 30.0

    def from_contours(
        self,
        road_borders: ContourManager,
        lanemarking_contours: ContourManager,
        img_shape: Tuple[int, int],
    ) -> List[Lanemarking]:
        """Create lanemarkings from road borders and lanemarking contours.

        Classify lanemarking contours by length into solid and dash lanemarking contours. Contours of dashed
        lanemarkings are grouped so that a single lanemarking includes all contours.

        :param road_borders: Road border contours
        :param lanemarking_contours: Lanemarking contours
        :param img_shape: Size of the satellite image used (h,w).
        :return: Extracted lanemarkings
        """
        # Filter lanemarkings by length, only merge short ones
        (
            short_lanemarking_contours,
            solid_lanemarking_contours,
        ) = lanemarking_contours.filter_and_group_by_length(
            max_length=self.filter_max_length, min_length=self.filter_min_length
        )

        # Create solid lanemarkings
        solid_lanemarkings = [
            Lanemarking([c], Lanemarking.LanemarkingType.SOLID)
            for c in solid_lanemarking_contours
        ]

        # Created dashed lanemarkings
        dashed_lanemarking, _ = short_lanemarking_contours.group_dashed_contours(
            max_match_cost=self.dashed_max_match_cost,
            lateral_distance_factor=self.dashed_lateral_distance_factor,
            max_long_distance=self.dashed_max_long_distance,
            max_lat_distance=self.dashed_max_lat_distance,
        )

        # Fill gaps between dashed and solid lanemarkings
        Lanemarking.extend_to(
            dashed_lanemarking,
            solid_lanemarkings,
            img_shape,
            check_border=self.extend_check_border,
            max_long_distance=self.extend_max_long_distance,
            max_lat_distance=self.extend_max_lat_distance,
        )

        road_borders = [
            Lanemarking([c], Lanemarking.LanemarkingType.ROAD_BORDER)
            for c in road_borders
        ]

        lanemarkings = [*road_borders, *solid_lanemarkings, *dashed_lanemarking]
        return lanemarkings


@dataclass
class ContourExtractor:
    subsample_factor: int = 30
    road_border_blur_size: int = 15
    road_border_border_size: int = 8
    lanemarking_blur_size: int = 3
    lanemarking_border_size: int = 1
    lanemarking_erode_size: int = 35

    def from_mask(
        self, seg_mask: SemanticMask, ignore_regions: List[IgnoreRegion]
    ) -> Tuple[ContourManager, BinaryMask, ContourManager]:
        """Extract contours of road borders and lanemarkings from a given semantic segmentation mask.

        Allows to manually remove mask regions from lanemarking contour extraction through ignore regions.

        :param seg_mask: Semantic segmentation of road section
        :param ignore_regions: List of regions ignored for lanemarking extraction
        :return: Tuple consisting of:
                 * Lanemarking contours
                 * Post-processed lanemarking segmentation mask
                 * Road border contours
        """
        # Find road borders
        road_mask = (
            seg_mask.class_mask(
                [SemanticClass.ROAD, SemanticClass.SYMBOL, SemanticClass.LANEMARKING]
            )
            .median_blur(self.road_border_blur_size, self.road_border_border_size)
            .close(3)
        )

        road_contours = (
            ContourManager.from_mask(road_mask)
            .split_at_border(road_mask.shape)
            .subsample(self.subsample_factor)
        )

        # Find lane markings
        lanemarking_mask = seg_mask.class_mask(SemanticClass.LANEMARKING)

        # Only allow lanemarkings far away from road borders
        road_trafficisland_mask = road_mask.union(
            seg_mask.class_mask(SemanticClass.TRAFFICISLAND)
        )
        road_trafficisland_mask = (
            road_trafficisland_mask.median_blur(
                self.lanemarking_blur_size, self.lanemarking_border_size
            )
            .close(3)
            .erode(self.lanemarking_erode_size)
        )
        lanemarking_mask = (
            lanemarking_mask.intersection(road_trafficisland_mask)
            # Extra erode and dilate to handle pointy lanemarkings after intersection
            .erode(3).dilate(3)
        ).remove_regions(ignore_regions)

        # As lanemarkings are THICK, thin them to get a line only
        lanemarking_mask = lanemarking_mask.thin()
        lanemarking_mask, split_points = lanemarking_mask.remove_split_points(
            debug=False
        )
        lanemarking_contours = (
            ContourManager.from_mask(lanemarking_mask)
            .unique_coordinates()
            .subsample(self.subsample_factor)
            .merge_at_split_points(split_points)
        )

        return lanemarking_contours, lanemarking_mask, road_contours


class MapBuilder:
    def __init__(
        self,
        contour_extractor: ContourExtractor,
        lanemarking_extractor: LanemarkingExtractor,
        symbol_detector: SymbolDetector,
        skip_lanelets=True,
        skip_symbols=False,
        debug_dir=None,
    ):
        """
        :param skip_lanelets: If true, lanelet association is deactivated.
        :param skip_symbols: If true, symbol detection is deactivated.
        :param debug_dir: If set, debug images are created during map creation process.
        """
        self._contour_extractor = contour_extractor
        self._lanemarking_extractor = lanemarking_extractor
        self._symbol_detector = symbol_detector

        self._skip_lanelets = skip_lanelets
        self._skip_symbols = skip_symbols
        self._debug_dir = debug_dir

    def from_semantic_mask(
        self,
        seg_mask: SemanticMask,
        ignore_regions: List[IgnoreRegion],
        proj: str,
        origin: Tuple[float, float],
        px2m: float,
        debug_prefix: str = "",
    ) -> "Map":
        """Creates a map from a semantic segmentation mask in a four-step process.

        Process steps:
        1) Extract contours from semantic segmentation mask
        2) Process contours to lanemarkings
        3) Group lanemarkings to lanelets
        4) Detect road symbols

        Creates debug images if `debug_dir` is set.

        :param seg_mask: Semantic segmentation of road section
        :param ignore_regions: List of regions ignored for lanemarking extraction
        :param proj: epsg code of the projection used, e.g. "epsg:25832"
        :param origin: Utm coordinates of the top left corner of the used satellite image.
        :param px2m: Conversion factor from pixel in satellite image to meters.
        :param debug_prefix: Prefix added to debug images.
        :return: Created map containing lanemarkings, lanelets and symbols.
        """
        (
            lanemarking_contours,
            lanemarking_mask,
            road_contours,
        ) = self._contour_extractor.from_mask(seg_mask, ignore_regions)

        img_ref, img_overlay = MaskVisualizer().show(
            lanemarking_mask,
            lanemarking_contours.merge(road_contours),
            background=seg_mask.mask,
            random=True,
        )

        if self._debug_dir:
            cv2.imwrite(
                str(self._debug_dir / f"{debug_prefix}lanemarkings.png"),
                cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR),
            )

        lanemarkings = self._lanemarking_extractor.from_contours(
            road_contours, lanemarking_contours, lanemarking_mask.shape
        )

        if self._debug_dir:
            _, img_overlay = MaskVisualizer().show(
                lanemarking_mask,
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

        if self._skip_lanelets:
            lanelets = set()
            logger.warning(f"Skipping lanelet matching!")
        else:
            lanelets = derive_lanelets(img_ref, seg_mask, lanemarkings, px2m)
            logger.info(f"Found {len(lanelets)} lanelets")

        if self._skip_symbols:
            symbols: List[Symbol] = []
            logger.warning(f"Skipping symbol detection!")
        else:
            symbols = self._symbol_detector.detect(seg_mask)
            logger.info(f"Detected {len(symbols)} symbols")

        logger.info("Map extraction finished!")
        return Map(lanemarkings, lanelets, symbols, proj, origin, px2m)
