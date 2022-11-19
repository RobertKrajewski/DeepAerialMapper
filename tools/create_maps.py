import argparse
import shutil
from datetime import datetime
from collections import Counter
from pathlib import Path

import numpy as np
import cv2
from loguru import logger
from typing import Set, FrozenSet, List
import yaml

from deepaerialmapper.visualization.mask_visualizer import MaskVisualizer
from deepaerialmapper.map_creation.contour import ContourSegment, ContourManager
from deepaerialmapper.map_creation.lanemarking import Lanemarking
from deepaerialmapper.export.map import Lanelet2Map
from deepaerialmapper.map_creation.masks import SegmentationMask, SemanticClass
from deepaerialmapper.map_creation.symbol import SymbolDetector



def create_map_from_semantic_mask(seg_mask, origin, px2m, proj, ignore_regions: List, skip_lanelets=True,
                                  skip_symbols=False, debug_dir=None, debug_prefix="") -> Lanelet2Map:


    # Find road borders
    road_mask = seg_mask.class_mask(
        [SemanticClass.ROAD, SemanticClass.SYMBOL, SemanticClass.LANEMARKING]).blur_and_close(15)
    road_contours = ContourManager.from_mask(road_mask).split_at_border(road_mask.shape).subsample(
        30)#.split_sharp_corners_v2()   #.split_sharp_corners(angle_threshold=50, length_threshold=100)

    # Find lane markings
    lanemarking_mask = seg_mask.class_mask(SemanticClass.LANEMARKING)
    # Only allow lanemarkings far away from road borders
    road_trafficisland_mask = road_mask.union(seg_mask.class_mask(SemanticClass.TRAFFICISLAND))
    lanemarking_mask = lanemarking_mask.intersection(
        road_trafficisland_mask.blur_and_close(3, border_effect=1).erode(35)).erode(3).dilate(3) \
        # .remove(ignore_regions)  # Extra erode and dilate to handle pointy lanemarkings after intersection
    # lanemarking_mask.show()

    # As lanemarkings are THICK, thin them to get a line only
    lanemarking_mask = lanemarking_mask.thin()
    # lanemarking_mask.show()
    lanemarking_mask, split_points = lanemarking_mask.find_split_points(debug=False)
    # lanemarking_mask.show()

    lanemarking_contours = ContourManager.from_mask(lanemarking_mask).unique_coordinates().subsample(30) \
        .group_at_split_points(split_points)#.split_sharp_corners_v2() \
        # .split_sharp_corners(angle_threshold=50, length_threshold=10)

    img_ref, img_overlay = MaskVisualizer().show(lanemarking_mask, lanemarking_contours.merge(road_contours), background=seg_mask.mask,
                                                 show=False, random=True)
    # img_ref, img_overlay = lanemarking_mask.show(lanemarking_contours.merge(road_contours), background=seg_mask.mask,
    #                                              show=True, random=True)
    if debug_dir:
        cv2.imwrite(str(debug_dir / f"{debug_prefix}lanemarkings.png"), cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR))

    lanemarkings = derive_lanemarkings(road_contours, lanemarking_contours, lanemarking_mask.shape)

    _, img_overlay = lanemarking_mask.show(lanemarkings=lanemarkings, background=seg_mask.mask, show=False,
                                           window_name=debug_prefix, random=True)
    if debug_dir:
        cv2.imwrite(str(debug_dir / f"{debug_prefix}lanemarkings_post.png"),
                    cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR))

    # Derive lanelets
    if skip_lanelets:
        lanelets = set()
        logger.info(f"Skipping lanelet matching")
    else:
        lanelets = derive_lanelets(img_ref, seg_mask, lanemarkings, px2m)
        logger.info(f"Found {len(lanelets)} symbols")

    # Find all symbols
    if skip_symbols:
        symbols = []
        logger.info(f"Skipping symbol detection")
    else:
        sym_patterns = ['Left', 'Left_Right', 'Right', 'Straight', 'Straight_Left', 'Straight_Right', 'Unknown']
        symbol_detector = SymbolDetector(sym_patterns)
        cls_weight = "configs/symbol_weights.pt"
        symbols = symbol_detector.detect_patterns(seg_mask, cls_weight)
        logger.info(f"Detected {len(symbols)} symbols")

    return Lanelet2Map(lanemarkings, symbols, lanelets, origin, px2m, proj)


def derive_lanemarkings(road_borders: ContourManager, lanemarking_contours: ContourManager, img_shape) -> List[
    Lanemarking]:
    # Filter lanemarkings by length, only merge short ones
    short_lanemarking_contours, solid_lanemarking_contours = lanemarking_contours.filter_by_length(max_length=5,
                                                                                                   min_length=2)
    # short_lanemarking_contours.show()

    # Create solid lanemarkings
    solid_lanemarkings = [Lanemarking([c], Lanemarking.LanemarkingType.SOLID) for c in solid_lanemarking_contours]

    # Created dashed lanemarkings
    dashed_lanemarking, short_lanemarking = short_lanemarking_contours.group(debug=False)

    # Fill gaps between dashed and solid lanemarkings
    dashed_lanemarking = Lanemarking.extend_to(dashed_lanemarking, solid_lanemarkings, img_shape)

    road_borders = [Lanemarking([c], Lanemarking.LanemarkingType.ROAD_BORDER) for c in road_borders]

    lanemarkings = [*road_borders, *solid_lanemarkings, *dashed_lanemarking]
    return lanemarkings


def derive_lanelets(img_ref, seg_mask, lanemarkings, px2m) -> Set:
    valid_mask = seg_mask.class_mask([SemanticClass.ROAD, SemanticClass.SYMBOL, SemanticClass.LANEMARKING])

    min_center_dist = 1.5 / px2m
    max_center_dist = np.sqrt(3.5 ** 2 + 1.5 ** 2) / px2m
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
            segment = ContourSegment.from_coordinates(contour[i_segment:i_segment + 2])

            left_costs = {}
            right_costs = {}

            for j_contour, other_lanemarking in enumerate(all_contours):
                # Don't match to itself
                if j_contour == i_contour:
                    continue

                other_contour = other_lanemarking.contour
                for j_segment in range(other_contour.shape[0] - 1):
                    other_segment = ContourSegment.from_coordinates(other_contour[j_segment:j_segment + 2])

                    # Check if orientation and distance between centers is in allowed range
                    center_dist = segment.distance_center_to_center(other_segment)
                    angle_diff = segment.abs_angle_difference(other_segment)
                    if center_dist < min_center_dist or center_dist > max_center_dist or angle_diff > max_angle_diff:
                        continue

                    # Check if perpendicular distance is in appropriate range
                    perp_center_dist = segment.min_distance_point(other_segment.center)
                    if perp_center_dist < min_center_dist or perp_center_dist > max_center_dist:
                        continue

                    # Check if in middle between both segments is also road and not e.g. grass
                    center = segment.center_between_centers(other_segment).astype(int)
                    if not valid_mask[center[1], center[0]]:
                        continue

                    if other_segment.center[1] < segment.center[1]:
                        left_costs[(j_contour, j_segment)] = perp_center_dist + 0.8 * center_dist
                    else:
                        right_costs[(j_contour, j_segment)] = perp_center_dist + 0.8 * center_dist

            if left_costs:
                left.append(min(left_costs, key=left_costs.get))
                j_contour, j_segment = left[-1]
                other_segment = ContourSegment.from_coordinates(
                    all_contours[j_contour].contour[j_segment:j_segment + 2])
                cv2.line(img, segment.center.astype(int), other_segment.center.astype(int), (255, 0, 0), thickness=3)

            if right_costs:
                right.append(min(right_costs, key=right_costs.get))
                j_contour, j_segment = right[-1]
                other_segment = ContourSegment.from_coordinates(
                    all_contours[j_contour].contour[j_segment:j_segment + 2])
                cv2.line(img, segment.center.astype(int), other_segment.center.astype(int), (0, 0, 255), thickness=3)

            cv2.circle(img, segment.center.astype(int), 3, (255, 128, 255), thickness=-1)

        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()

        if left:
            j_best_other_contour = Counter([m[0] for m in left]).most_common()[0][0]
            logger.info(f"Matched contour {i_contour} to left contour {j_best_other_contour}")
            lanelets.update([frozenset([i_contour, j_best_other_contour])])

        if right:
            j_best_other_contour = Counter([m[0] for m in right]).most_common()[0][0]
            logger.info(f"Matched contour {i_contour} to right contour {j_best_other_contour}")
            lanelets.update([frozenset([i_contour, j_best_other_contour])])

    return lanelets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-process semantic segmentation to HDmap")
    parser.add_argument("input",
                        help="location of input segmentation folder.")
    parser.add_argument("--output", default="results/maps/<now>",
                        help="Location of the root output directory.")
    parser.add_argument("--start_map", default=0,
                        help="Index of the first image to process")

    args = parser.parse_args()

    input_dir = Path(args.input)
    segmentation_files = list(sorted(input_dir.glob("*.png")))
    logger.info(f"Found {len(segmentation_files)} segmentation mask(s) in {args.input}:\n{segmentation_files}")
    if not segmentation_files:
        exit()

    output_dir = args.output
    if "<now>" in output_dir:
        output_dir = output_dir.replace("<now>", datetime.now().strftime("%y%m%d_%H%M%S"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store logs to result dir
    logger.add(output_dir / "log.txt")
    logger.info(f"Writing results to {output_dir}")

    # Copy this script to result dir
    script_path = Path(__file__).resolve()
    shutil.copyfile(script_path, output_dir / script_path.name)

    config_dir = 'configs/config.yaml'
    with open(config_dir, 'r') as f:
        config = yaml.safe_load(f)

    meta = {} if config['meta'] is None else config['meta']
    ignore = {} if config['ignore'] is None else config['ignore']
    palette_map = {SemanticClass[i["type"]]: i['color'] for i in config['palette']}

    start_map = int(args.start_map)


    for i_segmentation_file, segmentation_file in enumerate(segmentation_files):
        if i_segmentation_file < start_map:
            logger.info(f"Skipping map {i_segmentation_file} as start_map is set to {start_map}!")
            continue
        logger.info(
            f'Processing segmentation {i_segmentation_file}/{len(segmentation_files)}: {segmentation_file}')
        seg_mask = SegmentationMask.from_file(segmentation_file, palette_map)
        filename = segmentation_file.stem

        if filename in meta:
            file_meta = meta[filename]
            origin = file_meta["origin"]
            px2m = file_meta["scale"][0]
            proj = file_meta["proj"]
        else:
            logger.warning("Could not find meta information! Using defaults!")
            origin = (293692.76, 5628231.55)
            px2m = 0.05
            proj = "epsg:25832"

        ignore_info = [value for key, value in ignore.items() if key.startswith(filename)]
        if len(ignore_info):
            ignore_info = ignore_info[0]["regions"]

        lanelet_map = create_map_from_semantic_mask(seg_mask, origin, px2m, proj, ignore_info, debug_dir=output_dir,
                                                    debug_prefix=filename)
        lanelet_map.export_lanelet2(output_dir / f"{segmentation_file.stem}.osm")
        lanelet_map.export_lanemarkings(output_dir / f"{segmentation_file.stem}.pkl")
