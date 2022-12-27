import json
import pickle
from pathlib import Path

import numpy as np
from loguru import logger
import typer
from typing import Dict, List

from eval import Polyline, LanemarkingEvaluator


def load_lanemarking_dataset(predictions_dir: str, groundtruth_dir: str) -> Dict[str, Dict[str, List[Polyline]]]:
    """Load manually annotated gt-lanemarkings and automatically generated lanemarkings from disk.

    :param predictions_dir: Path to directory containing `.pkl` lanemarking files. These files are automatically created
                            in `create_maps.py`.
    :param groundtruth_dir: Path to directory containing `.json` lanemarking files. These files are manually created in
                            via-annotator and exported in the via-format.
    :return: Dictionary containing lists of gt and predicted lanemarkings per map.
             Example:
             `{"<map1_name>": {"groundtruth": [...], "prediction": [...]}, "<map2_name>": {...}}`
    """
    dataset = {}
    for predictions_file in sorted(Path(predictions_dir).glob("*.pkl")):
        logger.info(f"Loading results from {predictions_file}")
        map_ = {"groundtruth": [], "prediction": []}

        # Load result created from `create_maps.py`
        with predictions_file.open("rb") as f:
            predictions = pickle.load(f)
        map_["prediction"] = predictions
        logger.info(f"Loaded {len(map_['prediction'])} predicted lanemarkings")

        # Get matching gt file manually created using the via annotator tool
        groundtruth_path = Path(groundtruth_dir) / (
            f"test_{predictions_file.stem}_json.json"
        )
        if not groundtruth_path.exists():
            logger.error(
                f"Found no matching groundtruth annotations for {predictions_file.name}. Skipping!"
            )
            continue

        logger.info(f"Loading groundtruth from {groundtruth_path}")
        with groundtruth_path.open() as f:
            groundtruth = json.load(f)

        # Convert all groundtruth via regions to numpy arrays (x,y) to match format of predictions
        regions = list(groundtruth.values())[0]["regions"]
        for region in regions:
            if region["shape_attributes"]["name"] != "polyline":
                logger.warning("Unexpected non-polyline region! Skipping!")
                continue

            x = np.asarray(region["shape_attributes"]["all_points_x"])
            y = np.asarray(region["shape_attributes"]["all_points_y"])
            map_["groundtruth"].append(np.c_[x, y])
        logger.info(f"Loaded {len(map_['groundtruth'])} groundtruth lanemarkings")

        dataset[predictions_file.stem] = map_
    logger.info(f"Loaded predictions and groundtruth for {len(dataset)} maps")
    return dataset


def evaluate_lanemarkings(predictions_dir: str, groundtruth_dir: str) -> Dict[str, float]:
    """Evaluate the quality of automatically created lanemarkings (including road borders).

    Compares a dataset of automatically create lanemarkings using this library against manually annotated groundtruth
    lanemarkings. Logs results including precision, accuracy and RMSE to console for each map as well as averages values

    :param predictions_dir: Path to directory containing `.pkl` lanemarking files. These files are automatically created
                            in `create_maps.py`.
    :param groundtruth_dir: Path to directory containing `.json` lanemarking files. These files are manually created in
                            via-annotator and exported in the via-format.
    :return: Evaluation results micro-averaged over all maps. Stores precision, recall and RMSE.
    """
    dataset = load_lanemarking_dataset(groundtruth_dir, predictions_dir)
    evaluator = LanemarkingEvaluator()
    results = evaluator.evaluate_dataset(dataset)
    logger.info(
        f"Precision: {results['precision']:.2f}, Recall: {results['recall']:.2f}, RMSE: {results['rmse']:.2f}"
    )
    return results


if __name__ == "__main__":
    typer.run(evaluate_lanemarkings)
