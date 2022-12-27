from typing import Dict, List, Union, Any

import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment
from nptyping import NDArray

Polyline = NDArray[(Any, 2), float]  # 2D polyline


def resample_polyline(
        polyline: Polyline, resample_dist: float = 4.0, add_final_point: bool = True
) -> Polyline:
    """Resample polyline to get equidistant support points.

    :param polyline: 2D polyline (x,y)
    :param resample_dist: Distance of resampled support points along original polyline.
    :param add_final_point: If true, original last point is added to resampled polyline if it is far away from the last
                            point of the resampled polyline.
    :return: Resampled polyline
    """

    # Cumulative euclidean distance between successive points on original polyline.
    dists = np.cumsum(np.r_[0, np.linalg.norm(np.diff(polyline, axis=0), axis=1)])

    # Resample distances to get equidistance support points.
    dists_resampled = np.arange(0, dists[-1], resample_dist)

    # Add final point only if last interpolated point is at least half the step size away, otherwise very close points
    # are generated causing false positives or negatives
    if add_final_point and dists.max() - dists_resampled[-1] >= resample_dist / 2:
        dists_resampled = np.concatenate((dists_resampled, dists[-1]))

    # Interpolate x and y separately.
    polyline_resampled = np.c_[
        np.r_[np.interp(dists_resampled, dists, polyline[:, 0])],
        np.r_[np.interp(dists_resampled, dists, polyline[:, 1])],
    ]

    return polyline_resampled


def _evaluate_map(
        groundtruth: List[Polyline],
        prediction: List[Polyline],
        resample_dist: int = 4,
        max_matching_dist: float = 15.0,
        debug: bool = False,
) -> Dict[str, Union[float, List[float]]]:
    """Evaluates lanemarkings of a single map.

    After resampling, this function matches optimally points on the predicted lanemarkings against points on groundtruth
    lanemarkings. Based on matches, metrics including matching precision and accuracy are derived. For matched points,
    the distance error is additionally derived.

    :param groundtruth: Ground truth polylines of lanemarkings
    :param prediction: Predicted polylines of lanemarkings
    :param resample_dist: Distance of resampled support points along original polyline. Smaller is better, but increases
                          computational requirements.
    :param max_matching_dist: Max distance between a point on gt and predicted polyline to be matched against.
    :param debug: If true, creates a debug plot that shows tp associations as well as fp and fn.
    :return: Dictionary containing results for lanemarkings of given map.
    """
    # Resample groundtruth lanemarkings
    groundtruth = np.concatenate(
        [resample_polyline(g, resample_dist, add_final_point=True) for g in groundtruth]
    )
    groundtruth = np.unique(groundtruth, axis=0)

    # Resample predicted lanemarkings
    prediction = np.concatenate(
        [resample_polyline(p, resample_dist, add_final_point=True) for p in prediction]
    )
    prediction = np.unique(prediction, axis=0)

    # Use euclidean distance as matching cost: axis0: gt, axis1: pred
    cost = np.sqrt(((groundtruth[:, :, None] - prediction[:, :, None].T) ** 2).sum(1))
    # Use maximum matching distance to avoid long distance matches
    cost[cost > max_matching_dist] = max_matching_dist
    # Derive optimal matching between supporting points of predicted and gt polylines.
    gt_inds, p_inds = linear_sum_assignment(cost)

    if debug:
        import matplotlib.pyplot as plt

        # Plot all predicted and groundtruth points
        plt.scatter(groundtruth[:, 0], groundtruth[:, 1])
        plt.scatter(prediction[:, 0], prediction[:, 1])

    # Derive true positive (tp) matches, as well as false positive (fp) and false negative (fn) points.
    tp = []
    fp = list(set(range(len(prediction))) - set(p_inds))
    fn = list(set(range(len(groundtruth))) - set(gt_inds))
    distances = []
    for i_gt, i_p in zip(gt_inds, p_inds):
        if cost[i_gt, i_p] >= max_matching_dist:
            # Remove match as it exceeds the allowed matching distance
            fp.append(i_p)
            fn.append(i_gt)
        else:
            # Confirm match
            tp.append((i_gt, i_p))
            distances.append(cost[i_gt, i_p])
            if debug:
                # Draw line connecting predicted and gt points
                plt.plot(
                    [groundtruth[i_gt, 0], prediction[i_p, 0]],
                    [groundtruth[i_gt, 1], prediction[i_p, 1]],
                    c="k",
                )

    if debug:
        # Show false positive and false negative associations
        plt.scatter(prediction[fp, 0], prediction[fp, 1])
        plt.scatter(groundtruth[fn, 0], groundtruth[fn, 1])

        # Prettify plot window
        plt.grid()
        plt.gca().axis("equal")
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
        "rmse": np.sqrt(np.mean(np.square(distances))),
    }


def evaluate_lanemarking_dataset(dataset: Dict[str, Dict[str, List[Polyline]]]):
    """Evaluate the quality of automatically created lanemarkings against groundtruth lanemarkings.

    Compares a dataset of automatically create lanemarkings using this library against manually annotated groundtruth
    lanemarkings. Creates results including precision, accuracy and RMSE for each map as well as average values

    :param dataset: Dictionary containing lists of gt and predicted lanemarkings per map.
                    Example:
                    `{"<map1_name>": {"groundtruth": [...], "prediction": [...]}, "<map2_name>": {...}}`
    :return: Evaluation results micro-averaged over all maps. Stores precision, recall and RMSE.
    """
    map_results = []
    # Evaluate each map individually
    for name, map_ in dataset.items():
        map_result = _evaluate_map(map_["groundtruth"], map_["prediction"])
        logger.info(
            f"Results for {name}: Precision: {map_result['precision']:.2f}, "
            f"Recall: {map_result['recall']:.2f}, RMSE: {map_result['rmse']:.2f}"
        )
        map_results.append(map_result)

    # Aggregate results
    tp = sum(r["tp"] for r in map_results)
    fp = sum(r["fp"] for r in map_results)
    fn = sum(r["fn"] for r in map_results)
    distances = np.concatenate([r["distances"] for r in map_results])

    # Calculate micro-averaged metrics
    rmse = np.sqrt(np.mean(np.square(distances)))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return {"precision": precision, "recall": recall, "rmse": rmse}
