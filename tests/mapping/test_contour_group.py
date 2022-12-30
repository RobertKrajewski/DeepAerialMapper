from typing import List

import numpy as np

from deepaerialmapper.mapping.contour import ContourManager


def group_contours(groups):

    # Get the highest contour id
    max_contour_id = int(np.max(np.array(groups)).flatten())  # -> 7
    contour2group = [-1] * (int(max_contour_id) + 1)

    next_group = 0
    for seg_a, seg_b in groups:
        higher_group_idx = max(contour2group[seg_a], contour2group[seg_b])
        lower_group_idx = min(contour2group[seg_a], contour2group[seg_b])

        if higher_group_idx == -1 and lower_group_idx == -1:
            # Start a new group
            contour2group[seg_a] = next_group
            contour2group[seg_b] = next_group
            next_group += 1
        elif higher_group_idx > -1 and lower_group_idx == -1:
            # Assign to existing group
            matching_group_idx = max(contour2group[seg_a], contour2group[seg_b])
            contour2group[seg_a] = matching_group_idx
            contour2group[seg_b] = matching_group_idx
        else:
            # Shit - we have to merge groups
            # Replace all occurences of higher group idx by lower group idx
            contour2group = [
                idx if idx != higher_group_idx else lower_group_idx
                for idx in contour2group
            ]

    new_groups = []
    for group_idx in np.unique(contour2group):
        if group_idx == -1:
            print("Shit happend")
            continue

        new_groups.append(
            [
                i_c
                for i_c in range(max_contour_id + 1)
                if contour2group[i_c] == group_idx
            ]
        )


def test_group_contours():

    groups = [[0, 1], [2, 3], [4, 8], [3, 4], [5, 6], [2, 7]]
    #          0  0    1  1    1  1    1  1    2  2    1  1
    result = [[0, 1], [2, 3, 4, 7, 8], [5, 6]]

    new_groups = group_contours(groups)

    print(new_groups)


def merge_contours(contours: ContourManager, groups: List[List[int]]) -> ContourManager:
    new_contours = []
    for group in groups:
        # Collect all the relevant contours
        group_contours = [c for i_c, c in enumerate(contours) if i_c in group]

        # Now merging necessary
        if len(group_contours) == 1:
            new_contours.append(group_contours[0])
            continue

        # Merge relevant contours
        group_contours = ContourManager(group_contours)
        grouped_group_contours, ungrouped_group_contours = group_contours.group(
            max_gap_size=5, distance_mode="endpoints"
        )

        if len(ungrouped_group_contours):
            print("Shit happened. Some contours were not grouped. Pls check.")

        if len(grouped_group_contours) == 0:
            print("Big shit happened. Contour could not be grouped at all :(")
            continue

        if len(grouped_group_contours) > 1:
            print("Small shit happened. Contour was grouped more than once :/")

        new_contours.append(grouped_group_contours[0].contour)

    return ContourManager(new_contours)


def test_merge_contours():
    contours = [
        np.array([0, 0, 4, 0]).reshape((-1, 1, 2)),
        np.array([5, 0, 10, 0]).reshape((-1, 1, 2)),
        np.array([11, 0, 20, 0]).reshape((-1, 1, 2)),
        np.array([21, 0, 30, 80]).reshape((-1, 1, 2)),
        np.array([100, 0, 120, 0]).reshape((-1, 1, 2)),
        np.array([121, 0, 140, 0]).reshape((-1, 1, 2)),
        np.array([1000, 0, 1200, 0]).reshape((-1, 1, 2)),
    ]

    groups = [[0, 1, 2, 3], [4, 5], [6]]

    merged_contours = merge_contours(contours, groups)
    print(merged_contours.contours)


if __name__ == "__main__":
    test_group_contours()
    # test_merge_contours()
