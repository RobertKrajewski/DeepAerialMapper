import cv2
import numpy as np
from matplotlib import pyplot as plt

import sys

from typing import List, Tuple

import tqdm
import yaml

from tests.test_corner_split import find_split_points

sys.path.insert(0, r'.')
from deepaerialmapper.map_creation.contour import ContourSegment, ContourManager
from deepaerialmapper.eval.mapping_error import resample_polygon, evaluate_map, evaluate_dataset
from deepaerialmapper.map_creation.masks import SegmentationMask, SemanticClass, ClassMask
from deepaerialmapper.map_creation.symbol import SymbolDetector
from deepaerialmapper.map_creation.contour import compute_pca


def test_compute_pca():
    coords = np.arange(0, 10)
    coords = np.column_stack([coords, coords]).reshape((-1, 1, 2)).astype(float)

    print(compute_pca(coords))


def test_symbol_detector(image_path, *ori_path):
    config_dir = 'configs/mask/demo.yaml'
    with open(config_dir, 'r') as f:
        config = yaml.safe_load(f)
    palette_map = {SemanticClass[i["type"]]: i['palette'] for i in config['class']}

    # Load synthetic image with symbols on it
    # image_path = "data/synth/synthetic3.png"
    # image_path = "data\synth\synthetic3.png"
    seg_mask = SegmentationMask.from_file(image_path, palette_map)

    # Load symbol detector with arrow patterns
    cls_order = ['Left', 'Left_Right', 'Right', 'Straight', 'Straight_Left', 'Straight_Right', 'Unknown']
    cls_weight = "data_generation\mapping\symbol_cutout_crop.pt"
    detector = SymbolDetector(cls_order)

    # Show all loaded symbols
    # for symbol in detector.patterns:
    # symbol.show()

    symbols = detector.detect_patterns(seg_mask, cls_weight, debug=False,
                                       dbg_rescale=0.75)  # Scale too 100px size if debug=True

    # Draw and show results
    img = seg_mask.mask
    if ori_path != None:
        alpha = 0.5
        img = cv2.addWeighted(cv2.cvtColor(cv2.imread(ori_path[0]), cv2.COLOR_BGR2RGB), alpha, img, 1 - alpha, 0.0)
    for i_symbol, symbol in enumerate(symbols):
        # Draw contour of detected symbol
        img = cv2.polylines(img, [symbol.contour.reshape((-1, 1, 2))], isClosed=False, color=(40 * i_symbol, 0, 0),
                            thickness=1)
        # Put name above it
        img = cv2.putText(img, symbol.name, symbol.box[0] - [0, 20], cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                          thickness=2)

    # plt.imshow(img)
    # plt.show()
    return img


def test_oriented_distance():
    contour_a = np.array([[-1, 0], [1, 0]]).reshape((-1, 1, 2))
    contour_b = np.array([[9, 0], [11, 0]]).reshape((-1, 1, 2))

    contour_a = ContourSegment.from_coordinates(contour_a)
    contour_b = ContourSegment.from_coordinates(contour_b)

    assert contour_a.oriented_distance(contour_b) == (10, 0)

    contour_b = np.array([[-9, 0], [-11, 0]]).reshape((-1, 1, 2))
    contour_b = ContourSegment.from_coordinates(contour_b)
    assert contour_a.oriented_distance(contour_b) == (-10, 0)


def test_resample_polygon():
    # Straight line
    points = np.array([[0, 0], [0, 4], [0, 10]])
    resampled = resample_polygon(points, 4)
    assert np.all(resampled == np.array([[0, 0], [0, 4], [0, 8], [0, 10]]))

    # Double start point
    points = np.array([[0, 0], [0, 0], [0, 10]])
    resampled = resample_polygon(points, 4)
    assert np.all(resampled == np.array([[0, 0], [0, 4], [0, 8], [0, 10]]))


def test_evaluate_map():
    gt = [np.array([[0, 0], [5, 0], [20, 0]])]
    p = [np.array([[0, 1], [5, 2], [0, 20]])]
    r = evaluate_map(gt, p)
    assert r == {'precision': 0.5, 'recall': 0.42857142857142855, 'tp': 3, 'fn': 3, 'fp': 4}


def test_evaluate_dataset():
    gt = [np.array([[0, 0], [5, 0], [20, 0]])]
    p = [np.array([[0, 1], [5, 2], [0, 20]])]
    r = evaluate_dataset({"1": {"groundtruth": gt, "prediction": p}})
    assert r == {'precision': 0.5, 'recall': 0.42857142857142855, 'tp': 3, 'fn': 3, 'fp': 4}


def test_create_training_samples_symbols():
    filepath = r"C:\Users\samsung_\thesis\hdmap_segmentation\data\hdmap\ann_palette\1.png"
    segmask = SegmentationMask.from_file(filepath, palette_map)
    symbol_mask = segmask.class_mask(SemanticClass.SYMBOL)

    num_groups, group_mask, bboxes, centers = cv2.connectedComponentsWithStats(symbol_mask.astype(np.uint8),
                                                                               connectivity=8)

    for i in range(num_groups):
        single_mask = group_mask == i
        ...


def test_contour():
    lanemarking_mask = np.array([[0, 0, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 1, 0, 0],
                                 [0, 1, 1, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 1, 1, 1, 1, 0],
                                 [0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 1, 0],
                                 ], dtype=bool)

    lanemarking_mask = ClassMask(lanemarking_mask, class_names=[SemanticClass.LANEMARKING])

    points = find_split_points(lanemarking_mask.mask)
    assert len(points) == 2
    assert points[0] == (3, 3)

    lanemarking_contours = ContourManager.from_mask(lanemarking_mask)
    split_contours(lanemarking_contours, points)
    lanemarking_mask.show(ContourManager.from_mask(lanemarking_mask).unique_coordinates(),
                          show=True)
    print(lanemarking_contours)


def split_contours(contours: ContourManager, split_points: List[Tuple[int, int]]) -> ContourManager:
    new_contours: List[np.ndarray] = []

    for contour in contours:
        new_contours.extend(split_contour(contour, split_points))

    return ContourManager(new_contours)


def split_contour(contour: np.ndarray, split_points: List[Tuple[int, int]]) -> ContourManager:
    # Get all split points existing in current contour
    split_points = [p for p in split_points if np.any((contour[:, 0, 0] == p[1]) & (contour[:, 0, 1] == p[0]))]
    split_contours: List[np.ndarray] = []

    # For every split point (idx) get all the connected split_contours (idx)
    split_point_contours: List[List[int]] = []

    # Iterate over every split point and create segments
    for split_point in tqdm.tqdm(split_points):
        pass

    return ContourManager(split_contours)
