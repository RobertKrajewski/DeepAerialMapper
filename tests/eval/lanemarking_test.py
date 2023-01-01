import numpy as np
import pytest

from deepaerialmapper.eval.lanemarking import LanemarkingEvaluator, resample_polyline


def test_resample_polygon_interpolation() -> None:
    polyline = np.array([[0, 0], [10.1, 0]])

    resampled_polyline = resample_polyline(
        polyline, resample_dist=2.0, add_final_point=False
    )
    np.testing.assert_equal(
        resampled_polyline, np.array([[0, 0], [2, 0], [4, 0], [6, 0], [8, 0], [10, 0]])
    )


def test_resample_polygon_final_point() -> None:
    polyline = np.array([[0, 0], [10, 0]])

    resampled_polyline = resample_polyline(
        polyline, resample_dist=3.0, add_final_point=True
    )
    np.testing.assert_equal(
        resampled_polyline, np.array([[0, 0], [3, 0], [6, 0], [9, 0]])
    )

    resampled_polyline = resample_polyline(
        polyline, resample_dist=4.0, add_final_point=True
    )
    np.testing.assert_equal(
        resampled_polyline, np.array([[0, 0], [4, 0], [8, 0], [10, 0]])
    )

    resampled_polyline = resample_polyline(
        polyline, resample_dist=4.0, add_final_point=False
    )
    np.testing.assert_equal(resampled_polyline, np.array([[0, 0], [4, 0], [8, 0]]))


def test_resample_polygon_illegal_resample_dist() -> None:
    polyline = np.array([[0, 0], [10, 0]])

    with pytest.raises(ValueError):
        resample_polyline(polyline, resample_dist=-1.0, add_final_point=False)


def test_lanemarking_evaluator_evaluate_map() -> None:
    gt = [np.array([[0, 0], [0, 10]]), np.array([[5, 0], [5, 10]])]

    pred = [np.array([[1, 0], [1, 10]]), np.array([[6, 0], [6, 5]])]

    evaluator = LanemarkingEvaluator(
        resample_dist=4.0, max_matching_dist=15.0, debug=False
    )
    results = evaluator._evaluate_map(gt, pred)

    assert results["precision"] == 1.0
    assert results["recall"] == 0.75
    assert results["tp"] == 6
    assert results["fn"] == 2
    assert results["fp"] == 0
    assert results["rmse"] == 1.0
    np.testing.assert_allclose(results["distances"], 1.0)


def test_evaluate_lanemarking_dataset() -> None:
    dataset = {
        "0": {
            "groundtruth": [np.array([[0, 0], [0, 10]]), np.array([[5, 0], [5, 10]])],
            "prediction": [np.array([[1, 0], [1, 10]]), np.array([[6, 0], [6, 5]])],
        },
        "1": {
            "groundtruth": [np.array([[0, 0], [0, 10]]), np.array([[5, 0], [5, 10]])],
            "prediction": [np.array([[1, 0], [1, 10]]), np.array([[6, 0], [6, 5]])],
        },
    }

    evaluator = LanemarkingEvaluator(
        resample_dist=4.0, max_matching_dist=15.0, debug=False
    )
    results = evaluator.evaluate_dataset(dataset)
    assert results == {"precision": 1.0, "recall": 0.75, "rmse": 1.0}
