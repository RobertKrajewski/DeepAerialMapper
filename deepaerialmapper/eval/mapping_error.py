import json
import pickle
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Any
import typer


def resample_polygon(xy: np.ndarray, step_width: float = 4, add_final_point=True) -> np.ndarray:
    """Resample polygon to get points at equidistant positions along polygon"""

    # Cumulative Euclidean distance between successive polygon points.
    # This will be the "x" for interpolation
    d = np.cumsum(np.r_[0, np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1))])

    # get linearly spaced points along the cumulative Euclidean distance
    d_sampled = np.arange(0, d.max(), step_width)

    # interpolate x and y coordinates
    # Add final point only if last interpolated point is at least half the step size away, otherwise very close points
    # are generated causing false positives or negatives
    if add_final_point and d.max() - d_sampled[-1] >= step_width / 2:
        xy_interp = np.c_[
            np.r_[np.interp(d_sampled, d, xy[:, 0]), xy[-1, 0]],
            np.r_[np.interp(d_sampled, d, xy[:, 1]), xy[-1, 1]]
        ]
    else:
        xy_interp = np.c_[
            np.r_[np.interp(d_sampled, d, xy[:, 0])],
            np.r_[np.interp(d_sampled, d, xy[:, 1])]
        ]

    return xy_interp


def evaluate_map(groundtruth: List[np.ndarray], prediction: List[np.ndarray], resample_dist: int = 4,
                 max_match_distance: float = 15., debug: bool = False):
    # Interpolate all lines
    groundtruth = np.concatenate([resample_polygon(g, resample_dist, add_final_point=True) for g in groundtruth])
    groundtruth = np.unique(groundtruth, axis=0)
    prediction = np.concatenate([resample_polygon(p, resample_dist, add_final_point=True) for p in prediction])
    prediction = np.unique(prediction, axis=0)

    # rows = gt, cols = preds
    cost = np.sqrt(((groundtruth[:, :, None] - prediction[:, :, None].T) ** 2).sum(1))
    cost[cost > max_match_distance] = max_match_distance
    # cost: axis0: gt, axis1: pred

    gt_inds, p_inds = linear_sum_assignment(cost)

    if debug:
        import matplotlib.pyplot as plt
        plt.scatter(groundtruth[:, 0], groundtruth[:, 1])
        plt.scatter(prediction[:, 0], prediction[:, 1])

    tp = []
    fp = list(set(range(len(prediction))) - set(p_inds))
    fn = list(set(range(len(groundtruth))) - set(gt_inds))
    distances = []
    for i_gt, i_p in zip(gt_inds, p_inds):
        if cost[i_gt, i_p] >= max_match_distance:
            fp.append(i_p)
            fn.append(i_gt)
        else:
            tp.append((i_gt, i_p))
            distances.append(cost[i_gt, i_p])
            if debug:
                # Line connecting both points
                plt.plot([groundtruth[i_gt, 0], prediction[i_p, 0]], [groundtruth[i_gt, 1], prediction[i_p, 1]], c="k")

    if debug:
        plt.grid()
        plt.scatter(prediction[fp, 0], prediction[fp, 1])
        plt.scatter(groundtruth[fn, 0], groundtruth[fn, 1])
        plt.gca().axis('equal')

        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    return {
        "precision": len(tp) / (len(tp) + len(fp)),
        "recall": len(tp) / (len(tp) + len(fn)),
        "tp": len(tp),
        "fn": len(fn),
        "fp": len(fp),
        "distances": distances,
        "rmse": np.sqrt(np.mean(np.square(distances)))
    }


def evaluate_dataset(dataset: Dict[str, Dict[str, Any]]):
    map_results = []

    for name, map_ in list(dataset.items())[0:]:
        map_results.append(evaluate_map(map_["groundtruth"], map_["prediction"]))
        r = map_results[-1]
        logger.info(
            f"Results for {name}: Precision: {r['precision']:.2f}, Recall: {r['recall']:.2f}, RMSE: {r['rmse']:.2f}")

    tp = sum(r["tp"] for r in map_results)
    fp = sum(r["fp"] for r in map_results)
    fn = sum(r["fn"] for r in map_results)
    distances = np.concatenate([r["distances"] for r in map_results])
    rmse = np.sqrt(np.mean(np.square(distances)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "rmse": rmse,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


def main(predictions_dir: str, groundtruth_dir: str):
    dataset = {}

    for predictions_file in sorted(Path(predictions_dir).glob("*.pkl")):
        logger.info(f"Loading results from {predictions_file}")
        map_ = {
            "groundtruth": [],
            "prediction": [],
            "ignore": []
        }

        with open(predictions_file, "rb") as f:
            predictions = pickle.load(f)
        map_["prediction"] = predictions

        # Get matching gt file
        groundtruth_path = Path(groundtruth_dir) / (f"test_{predictions_file.stem}_json.json")
        if not groundtruth_path.exists():
            logger.error(f"Found no matching groundtruth annotations for {predictions_file.name}. Skipping!")
            continue

        with groundtruth_path.open() as f:
            groundtruth = json.load(f)

        # Convert all regions to numpy arrays
        regions = list(groundtruth.values())[0]["regions"]
        for region in regions:
            if region["shape_attributes"]["name"] != "polyline":
                continue

            x = np.asarray(region["shape_attributes"]["all_points_x"])
            y = np.asarray(region["shape_attributes"]["all_points_y"])
            map_["groundtruth"].append(np.c_[x, y])

        dataset[predictions_file.stem] = map_

    logger.info(f"Loaded data for {len(dataset)} files")

    results = evaluate_dataset(dataset)
    logger.info(f"Precision: {results['precision']:.2f}, Recall: {results['recall']:.2f}, RMSE: {results['rmse']:.2f}")


if __name__ == "__main__":
    typer.run(main)
