import itertools

import numpy as np
from typing import List, Tuple
# from numba import jit


def is_split_point(contour_mask: np.ndarray, y: int, x: int) -> bool:
    # Extract 3x3 contour around x,y
    # Easy case: Not at the mask border
    if 1 <= x <= contour_mask.shape[1] and 1 <= y <= contour_mask.shape[0]:
        contour_mask = contour_mask[y-1:y+2, x-1:x+2]
    else:
        c = np.zeros((3, 3))
        # Complex case at mask border
        for dx, dy in itertools.product([-1, 0, 1], [-1, 0, 1]):
            if y + dy < 0 or x + dx < 0 or y + dy >= contour_mask.shape[0] or x + dx >= contour_mask.shape[1]:
                continue
            c[1+dy, 1+dx] = contour_mask[y+dy, x+dx]
        contour_mask = c
    # Change x and y to center of the crop
    x = 1
    y = 1

    # For at least 3 lines to exist, we need more than 4 points in the mask
    if contour_mask.sum() <= 3:
        return False

    is_visited = np.zeros_like(contour_mask, bool)  # Remember which neighbours we already visited
    num_lines = 0  # How many lines start from this point

    # Hard coding all the neighbours that need to be checked
    locations2check = {
        # (dx, dy) of a neighbour  -> List[(dy, dy)] of neighbour's neighbours
        # dx and dy are always relative to the point of interest (x, y)
        (-1, 0): [(-1, -1), (-1, +1)],
        (+1, 0): [(+1, -1), (+1, +1)],
        (0, -1): [(-1, -1), (+1, -1)],
        (0, +1): [(-1, +1), (+1, +1)],
        (-1, -1): [(-1, 0), (0, -1)],
        (-1, +1): [(-1, 0), (0, +1)],
        (+1, -1): [(+1, 0), (0, -1)],
        (+1, +1): [(+1, 0), (0, +1)],
    }

    # Iterate over every direct neighbour
    for dx, dy in itertools.product([-1, 0, 1], [-1, 0, 1]):
        if dx == 0 and dy == 0:
            # Center point
            continue

        # Check if we are at border
        if y + dy < 0 or x + dx < 0 or y + dy >= contour_mask.shape[0] or x + dx >= contour_mask.shape[1]:
            continue

        # Visit every neighbour only once
        if is_visited[y + dy, x + dx]:
            continue

        is_visited[y + dy, x + dx] = True

        if contour_mask[y + dy, x + dx] == 0:
            # No new line starts here
            continue

        # We have found the start of a new line
        num_lines += 1

        # Recursively check the neighbours as they would belong to the same line
        def check_neighbours(dx, dy):
            locations = locations2check[(dx, dy)]
            for dx2, dy2 in locations:
                new_x = x + dx2
                new_y = y + dy2

                if new_y < 0 or new_x < 0 or new_y >= contour_mask.shape[0] or new_x >= contour_mask.shape[1]:
                    continue

                if contour_mask[new_y, new_x] and not is_visited[new_y, new_x]:
                    is_visited[new_y, new_x] = True
                    check_neighbours(dx2, dy2)

        check_neighbours(dx, dy)
        # check_neighbours(locations2check, x, y, dx, dy, contour_mask, is_visited)

    return num_lines > 2


def find_split_points(contour_mask: np.ndarray) -> List[Tuple[int, int]]:
    points = np.argwhere(contour_mask)
    split_points = [tuple(point) for point in points if is_split_point(contour_mask, *point)]
    return split_points


def test_split_point():
    # Easy case
    contour_mask = np.array([
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])

    assert is_split_point(contour_mask, 2, 2) is False
    contour_mask = np.array([
        [0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
    ])

    assert is_split_point(contour_mask, 3, 2) is True

    # Hard case
    contour_mask = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])

    assert is_split_point(contour_mask, 2, 2) is False

    contour_mask = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])

    assert is_split_point(contour_mask, 2, 2) is True

    # The case that surprised us
    contour_mask = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])

    assert is_split_point(contour_mask, 2, 2) is False
    assert is_split_point(contour_mask, 2, 1) is True


def test_find_split_points():
    contour_mask = np.array([
        [0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])

    split_points = find_split_points(contour_mask)
    assert len(split_points) == 1
    assert split_points[0] == (2, 2)

    contour_mask = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])
    split_points = find_split_points(contour_mask)
    assert len(split_points) == 1
    assert split_points[0] == (1, 2)

    contour_mask = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    split_points = find_split_points(contour_mask)
    assert len(split_points) == 2
    assert split_points[0] == (1, 2)
    assert split_points[1] == (3, 2)

    n = 400
    contour_mask = np.tile(contour_mask, (n, n))
    from time import time
    s = time()
    split_points = find_split_points(contour_mask)
    print(time()-s)
    assert len(split_points) == 2 * n * n

