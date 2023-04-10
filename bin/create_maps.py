import json
import pprint
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import typer
import yaml
from loguru import logger

from deepaerialmapper.mapping import (
    ContourExtractor,
    IgnoreRegion,
    LanemarkingExtractor,
    MapBuilder,
    SemanticClass,
    SemanticMask,
    SymbolDetector,
)


def _load_configs_and_prepare(
    config_filepath: Path, output_dir: Path
) -> Tuple[List[Path], Dict, Dict, Dict]:
    shutil.copyfile(config_filepath, output_dir / config_filepath.name)
    with open(config_filepath, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_filepath}:\n{pprint.pformat(config)}")

    # Load data meta created by `extract_tif_meta.py`
    meta_filepath = Path(config.get("meta_filepath"))
    shutil.copyfile(meta_filepath, output_dir / meta_filepath.name)
    with open(meta_filepath, "r") as f:
        meta = yaml.safe_load(f)
    logger.info(f"Loaded meta from {meta_filepath}:\n{pprint.pformat(meta)}")

    # Load ignore regions manually annotated in via-annotator
    ignore_regions_filepath = Path(config.get("ignore_regions_filepath"))
    if ignore_regions_filepath.exists():
        shutil.copyfile(
            ignore_regions_filepath, output_dir / ignore_regions_filepath.name
        )
        with open(ignore_regions_filepath, "r") as f:
            ignore = json.load(f)

        # Remove file size from filenames for easier access
        ignore_areas_filenames = list(ignore.keys())
        for filename in ignore_areas_filenames:
            corrected_filename = Path(filename).stem
            ignore[corrected_filename] = ignore[filename]
            del ignore[filename]

        logger.info(
            f"Loaded ignore regions from {ignore_regions_filepath}:\n{pprint.pformat(ignore)}"
        )
    else:
        logger.warning("No ignore_regions.json file found!")
        ignore = {}
    masks_dir_path = Path(config.get("mask_dir_path"))
    segmentation_files = list(sorted(masks_dir_path.glob("*.png")))
    logger.info(
        f"Found {len(segmentation_files)} segmentation mask(s) in {masks_dir_path}:\n{pprint.pformat(segmentation_files)}"
    )
    return segmentation_files, meta, config, ignore


def create_maps(
    config_path: str,
    output_dir: str = "results/maps/<now>_<config>",
    start_map: int = 0,
) -> None:
    """Based on semantic segmentation of satellite images, derive lanemarkings and symbols in lanelet2 format.

    :param config_path: Directory containing semantic segmentation masks as png files as well as extra configuration information.
    :param output_dir: Directory to save results to.
    :param start_map: Skip the first n maps.
    """

    config_file_path = Path(config_path)

    # Prepare result directory
    if "<now>" in output_dir:
        output_dir = output_dir.replace(
            "<now>", datetime.now().strftime("%y%m%d_%H%M%S")
        )

    if "<config>" in output_dir:
        output_dir = output_dir.replace(
            "<config>", config_file_path.stem
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store logs to result dir
    logs_filepath = output_dir / "log.txt"
    logger.add(logs_filepath)
    logger.info(f"Writing results to {output_dir}")
    logger.info(f"Writing logs to {logs_filepath}")

    segmentation_files, meta, config, ignore = _load_configs_and_prepare(
        config_file_path, output_dir
    )
    palette_map = {SemanticClass[i["type"]]: i["color"] for i in config["palette"]}

    if not segmentation_files:
        raise ValueError("No segmentation masks given! Stopping!")

    contour_extractor_config = config.get("contour_extractor", {})
    contour_extractor = ContourExtractor(**contour_extractor_config)
    lanemarking_extractor_config = config.get("lanemarking_extractor", {})
    lanemarking_extractor = LanemarkingExtractor(**lanemarking_extractor_config)
    symbol_detector_config = config.get("symbol_detector", {})
    symbol_detector = SymbolDetector(**symbol_detector_config)
    builder = MapBuilder(
        contour_extractor,
        lanemarking_extractor,
        symbol_detector,
        skip_lanelets=False,
        skip_symbols=False,
        debug_dir=output_dir,
    )

    for i_segmentation_file, segmentation_file in enumerate(
        segmentation_files, start=start_map
    ):
        logger.info(
            f"Processing segmentation {i_segmentation_file}/{len(segmentation_files)}: {segmentation_file}"
        )
        seg_mask = SemanticMask.from_file(segmentation_file, palette_map)
        filename = segmentation_file.stem

        # Get matching meta
        if filename in meta:
            file_meta = meta[filename]
            origin = file_meta["origin"]
            px2m = file_meta["scale"][0]
            proj = file_meta["proj"]
        else:
            logger.warning("Could not find meta information! Using defaults!")
            origin = (294406.50, 5628828.23)
            px2m = 0.05
            proj = "epsg:25832"

        # Get matching ignore regions
        ignore_regions: List[IgnoreRegion] = []
        if filename in ignore:
            ignore_regions = ignore[filename]["regions"]
        logger.info(f"Found {len(ignore_regions)} ignore regions!")

        lanelet_map = builder.from_semantic_mask(
            seg_mask,
            ignore_regions,
            proj,
            origin,
            px2m,
            debug_prefix=filename,
        )

        # Store results to disk
        lanelet_map.export_lanelet2(output_dir / f"{segmentation_file.stem}.osm")
        lanelet_map.export_lanemarkings(output_dir / f"{segmentation_file.stem}.pkl")


if __name__ == "__main__":
    typer.run(create_maps)
