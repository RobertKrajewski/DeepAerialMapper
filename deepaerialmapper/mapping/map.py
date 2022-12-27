import pickle
from typing import Tuple

from pyproj import Proj

from deepaerialmapper.mapping.symbol import Symbol

from collections import Counter
from pathlib import Path
from typing import FrozenSet, List, Set

import cv2
import numpy as np
from loguru import logger

from deepaerialmapper.mapping.contour import ContourManager, ContourSegment
from deepaerialmapper.mapping.lanemarking import Lanemarking
from deepaerialmapper.mapping.masks import SemanticClass
from deepaerialmapper.mapping.symbol import SymbolDetector
from deepaerialmapper.visualization.mask_visualizer import MaskVisualizer


class Map:
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

    @staticmethod
    def derive_lanelets(
        img_ref, seg_mask, lanemarkings, px2m
    ) -> Set[FrozenSet[int, int]]:
        valid_mask = seg_mask.class_mask(
            [SemanticClass.ROAD, SemanticClass.SYMBOL, SemanticClass.LANEMARKING]
        )

        min_center_dist = 1.5 / px2m
        max_center_dist = np.sqrt(3.5**2 + 1.5**2) / px2m
        max_angle_diff = np.deg2rad(15)
        lanelets: Set[FrozenSet[int, int]] = set()
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
                        perp_center_dist = segment.min_distance_point(
                            other_segment.center
                        )
                        if (
                            perp_center_dist < min_center_dist
                            or perp_center_dist > max_center_dist
                        ):
                            continue

                        # Check if in middle between both segments is also road and not e.g. grass
                        center = segment.center_between_centers(other_segment).astype(
                            int
                        )
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
                j_best_other_contour = Counter([m[0] for m in right]).most_common()[0][
                    0
                ]
                logger.info(
                    f"Matched contour {i_contour} to right contour {j_best_other_contour}"
                )
                lanelets.update([frozenset([i_contour, j_best_other_contour])])

        return lanelets

    @classmethod
    def from_semantic_mask(
        cls,
        seg_mask,
        ignore_regions: List,
        proj,
        origin,
        px2m,
        skip_lanelets=True,
        skip_symbols=False,
        debug_dir=None,
        debug_prefix="",
    ) -> "Map":

        # Find road borders
        road_mask = seg_mask.class_mask(
            [SemanticClass.ROAD, SemanticClass.SYMBOL, SemanticClass.LANEMARKING]
        ).blur_and_close(15)
        road_contours = (
            ContourManager.from_mask(road_mask)
            .split_at_border(road_mask.shape)
            .subsample(30)
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
        )  # .remove(ignore_regions)  # Extra erode and dilate to handle pointy lanemarkings after intersection
        # lanemarking_mask.show()

        # As lanemarkings are THICK, thin them to get a line only
        lanemarking_mask = lanemarking_mask.thin()
        # lanemarking_mask.show()
        lanemarking_mask, split_points = lanemarking_mask.find_split_points(debug=False)
        # lanemarking_mask.show()

        lanemarking_contours = (
            ContourManager.from_mask(lanemarking_mask)
            .unique_coordinates()
            .subsample(30)
            .group_at_split_points(split_points)
        )  # .split_sharp_corners_v2() \
        # .split_sharp_corners(angle_threshold=50, length_threshold=10)

        img_ref, img_overlay = MaskVisualizer().show(
            lanemarking_mask,
            lanemarking_contours.merge(road_contours),
            background=seg_mask.mask,
            show=False,
            random=True,
        )
        # img_ref, img_overlay = lanemarking_mask.show(lanemarking_contours.merge(road_contours), background=seg_mask.mask,
        #                                              show=True, random=True)
        if debug_dir:
            cv2.imwrite(
                str(debug_dir / f"{debug_prefix}lanemarkings.png"),
                cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR),
            )

        lanemarkings = cls.derive_lanemarkings(
            road_contours, lanemarking_contours, lanemarking_mask.shape
        )

        _, img_overlay = lanemarking_mask.show(
            lanemarkings=lanemarkings,
            background=seg_mask.mask,
            show=False,
            window_name=debug_prefix,
            random=True,
        )
        if debug_dir:
            cv2.imwrite(
                str(debug_dir / f"{debug_prefix}lanemarkings_post.png"),
                cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR),
            )

        # Derive lanelets
        if skip_lanelets:
            lanelets = set()
            logger.info(f"Skipping lanelet matching")
        else:
            lanelets = cls.derive_lanelets(img_ref, seg_mask, lanemarkings, px2m)
            logger.info(f"Found {len(lanelets)} symbols")

        # Find all symbols
        if skip_symbols:
            symbols = []
            logger.info(f"Skipping symbol detection")
        else:
            sym_patterns = [
                "Left",
                "Left_Right",
                "Right",
                "Straight",
                "Straight_Left",
                "Straight_Right",
                "Unknown",
            ]
            symbol_detector = SymbolDetector(sym_patterns)
            cls_weight = "configs/symbol_weights.pt"
            symbols = symbol_detector.detect_patterns(seg_mask, cls_weight)
            logger.info(f"Detected {len(symbols)} symbols")

        return cls(lanemarkings, lanelets, symbols, proj, origin, px2m)

    def __init__(
        self,
        lanemarkings,
        lanelets: Set[FrozenSet[int, int]],
        symbols: List[Symbol],
        proj: str,
        origin: Tuple[float, float],
        px2m: float,
    ) -> None:
        """Representation of a map including geo-referenced lanemarkings, symbols and lanelets.
        Implements export to lanelet2 format as well as storing lanemarkings for evaluation purposes.

        :param lanemarkings: List of lanemarkigns
        :param lanelets: Association of the lanemarkings to lanelets
        :param symbols: List of symbols, mainly arrows
        :param proj: epsg code of the projection used, e.g. "epsg:25832"
        :param origin: Utm coordinates of the top left corner of the used satellite image.
        :param px2m: Conversion factor from pixel in satellite image to meters.
        """
        self._lanemarkings = lanemarkings
        self._symbols = symbols
        self._lanelets = lanelets
        self._origin = origin
        self._px2m = px2m
        self._proj = proj

    def export_lanelet2(self, filepath: Path) -> None:
        """Export to lanelet2 format compatible with JOSM"""
        proj = Proj(self._proj)
        nodes: List[str] = []
        ways: List[str] = []
        relations: List[str] = []

        # Create nodes and ways for symbols
        for symbol in self._symbols:
            way_str = [f"<way id='{len(ways) + 1}' visible='true' version='1'>"]
            for ref in symbol.ref:
                # Convert from pixel coordinates to utm
                utm_x = self._origin[0] + self._px2m * ref[0, 0]
                utm_y = self._origin[1] - self._px2m * ref[0, 1]

                # Convert from utm to long/lat
                long, lat = proj(utm_x, utm_y, inverse=True)

                point_str = f"<node id='{len(nodes) + 1}' visible='true' version='1' lat='{lat}' lon='{long}' />"
                nodes.append(point_str)
                way_str.append(f"    <nd ref='{len(nodes) + 1}' />")
            way_str.append(f"    <tag k='subtype' v='{symbol.name.lower()}' />")
            way_str.append("    <tag k='type' v='arrow' />")
            way_str.append("</way>")
            ways.append("\n".join(way_str))

        # Create nodes and ways for lanemarkings
        for lanemarking in self._lanemarkings:
            way_str = [f"<way id='{len(ways) + 1}' visible='true' version='1'>"]
            for point in lanemarking.contour:
                utm_x = self._origin[0] + self._px2m * point[0, 0]
                utm_y = (
                    self._origin[1] - self._px2m * point[0, 1]
                )  # "-" as UTM has an inverted y-axis
                long, lat = proj(utm_x, utm_y, inverse=True)
                point_str = f"<node id='{len(nodes) + 1}' visible='true' version='1' lat='{lat}' lon='{long}' />"
                way_str.append(f"    <nd ref='{len(nodes) + 1}' />")
                nodes.append(point_str)
            if lanemarking.type_ == lanemarking.LanemarkingType.SOLID:
                way_str.append("    <tag k='type' v='line_thin' />")
                way_str.append("    <tag k='subtype' v='solid' />")
            elif lanemarking.type_ == lanemarking.LanemarkingType.DASHED:
                way_str.append("    <tag k='type' v='line_thin' />")
                way_str.append("    <tag k='subtype' v='dashed' />")
            elif lanemarking.type_ == lanemarking.LanemarkingType.ROAD_BORDER:
                way_str.append("    <tag k='type' v='road_border' />")
            way_str.append("</way>")
            ways.append("\n".join(way_str))

        # Create relations for lanelets
        for way_a, way_b in self._lanelets:
            lanelet_str = [
                f"<relation id='{len(relations) + 1}' visible='true' version='1'>",
                f"    <member type='way' ref='{way_a + 1}' role='left' />",
                f"    <member type='way' ref='{way_b + 1}' role='right' />",
                "    <tag k='location' v='urban' />",
                "    <tag k='one_way' v='no' />",
                "    <tag k='region' v='de' />",
                "    <tag k='subtype' v='road' />",
                "    <tag k='type' v='lanelet' />",
                "</relation>",
            ]
            relations.append("\n".join(lanelet_str))

        # Compose and write lanelet2 from individual components
        with filepath.open("w") as f:
            f.write(
                "<?xml version='1.0' encoding='UTF-8'?>\n"
                "<osm version='0.6' generator='JOSM'>"
            )
            f.write("\n".join(nodes))
            f.write("\n".join(ways))
            f.write("\n".join(relations))
            f.write("</osm>")

    def export_lanemarkings(self, filepath: Path) -> None:
        """Export lanemarkings only to a pickle file for accuracy evaluation."""
        contours = [
            l.contour[:, 0, :] for l in self._lanemarkings
        ]  # Remove obsolete axis 1

        with filepath.open("wb") as f:
            pickle.dump(contours, f)
