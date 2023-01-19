import itertools
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
from sklearn.decomposition import PCA

import cv2
import numpy as np


@dataclass
class ContourSegment:
    """Part (two or more support points) of a contour."""

    coordinates: np.ndarray  # List of coordinates (N, 1, 2[x,y])
    center: np.ndarray  # Mean of coordinates (2[x,y])
    first_component: np.ndarray  # First pca component of coordinates (2[x,y])

    @classmethod
    def from_coordinates(cls, coordinates: np.ndarray) -> "ContourSegment":
        """Create a contour segment from a list of coordinates by applying pca.

        :param coordinates: List of coordinates (N, 1, 2[x,y])
        """

        # Represent line by eigenvector of first principal component
        if len(coordinates) > 1:
            pca = PCA(n_components=2).fit(coordinates[:, 0, :2])
            first_component = np.asarray(pca.components_[0])
        else:
            # Set dummy values if only a single coordinate is given
            first_component = np.asarray([0.0, 1.0])

        return cls(
            coordinates=coordinates,
            center=np.mean(coordinates, axis=(0, 1)),
            first_component=first_component
        )

    def abs_angle_difference(self, other_segment: "ContourSegment") -> float:
        """Derive absolute angle between two vectors in rad"""
        first = self.first_component
        second = other_segment.first_component

        unit_first = first / np.linalg.norm(first)
        unit_second = second / np.linalg.norm(second)

        dot_product = np.dot(unit_first, unit_second)
        return np.minimum(
            np.abs(np.arccos(dot_product)), np.pi - np.abs(np.arccos(dot_product))
        )

    def min_distance_point(self, point: np.ndarray) -> float:
        """Minimal distance between segment and given point.

        Note: Segment is interpreted as line using first principal component and center (ax+by+c=0)"""
        a, b = self.first_component
        (c_x, c_y) = self.center
        c = -a * c_x + b * c_y  # Calculate y-axis offset

        x, y = point
        return np.abs(
            (a * x - b * y + c) / (np.sqrt(a**2 + b**2))
        )  # Project point on orthogonal slope

    def closest_point(
        self,
        points: List[np.ndarray],
        max_long_distance: float,
        max_lat_distance: float,
    ) -> np.ndarray:
        """Find the closest point in list of points.

        Points are first filtered by max longitudinal and lateral distance to segment.
        The closest point is the remaining point with the minimal positive longitudinal distance.
        Note: For longitudinal distance the orientation of the segment is considered and only points in forward
              direction are considered.

        :param points: List of points (2[x,y])
        :return: Coordinates of the closest point (2[x,y])
        """
        if not points:
            raise ValueError("List of points is empty!")

        min_dist_point = None
        min_dist = math.inf

        for point in points:
            long, lat = self.oriented_distance_point(point)

            if (
                    0 < long < max_long_distance
                    and abs(lat) < max_lat_distance
                    and long < min_dist
            ):
                min_dist = long
                min_dist_point = point
        return min_dist_point

    def distance_center_to_point(self, point: np.ndarray) -> float:
        return np.abs(np.linalg.norm(self.center - point))

    def distance_center_to_center(self, other_segment: "ContourSegment") -> float:
        return self.distance_center_to_point(other_segment.center)

    def center_between_centers(self, other_segment: "ContourSegment") -> np.ndarray:
        return (self.center + other_segment.center) / 2

    def merge(self, other_segment: "ContourSegment") -> "ContourSegment":
        """Merge two segments by appending the other segments coordinates"""
        all_coords = np.concatenate([self.coordinates, other_segment.coordinates])
        return self.from_coordinates(all_coords)

    @staticmethod
    def group_by_length(
        contour: "Contour", threshold_length: float
    ) -> Tuple["Contour", "Contour"]:
        """Group list of segments by arc length into short and long segments.

        :param contour: List of segments
        :return: List of short segments, list of long segments
        """
        long_segments: Contour = []
        short_segments: Contour = []
        for segment in contour:
            if cv2.arcLength(segment.coordinates, closed=False) > threshold_length:
                long_segments.append(segment)
            else:
                short_segments.append(segment)

        return long_segments, short_segments

    def compute_pca(self):
        """Compute the eigenvector and centroid position of a contour based on PCA analysis"""
        pts = self.coordinates

        mean, eigenvectors, eigenvalues = cv2.PCACompute2(
            np.reshape(pts.astype(float), (-1, 2)), np.empty(0)
        )
        # Find the end_point two points
        end_pointA = mean + eigenvectors[0] * np.max(np.linalg.norm(mean - pts, axis=2))
        end_pointB = mean - eigenvectors[0] * np.max(np.linalg.norm(mean - pts, axis=2))
        center = [mean[0, 0], mean[0, 1]]
        orientation = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])

        simple_orientation = np.arctan2(
            pts[-1, 0, 1] - pts[0, 0, 1], pts[-1, 0, 0] - pts[0, 0, 0]
        )

        # Make sure that orientation always is "away" from the first coordinate
        if abs(orientation - simple_orientation) > np.pi / 2:
            orientation += np.pi
            eigenvectors[0] *= -1

        return end_pointA, end_pointB, center, orientation, eigenvectors, eigenvalues

    def oriented_distance(self, other_segment: "ContourSegment") -> Tuple[float, float]:
        _, _, center, _, ev, _ = self.compute_pca()
        _, _, o_center, _, o_ev, _ = other_segment.compute_pca()
        center = np.asarray(center)
        o_center = np.asarray(o_center)

        if len(ev) < 2:
            return 10000, 10000

        # Project distance between centers to long and lat direction of current element
        d = o_center - center
        long_dist = d @ ev[0]
        lat_dist = d @ ev[1]

        return long_dist, lat_dist

    def endpoint_distance(self, other_segment: "ContourSegment") -> Tuple[float, float]:
        distances = []
        # Calculate L2 distance for every startpoint endpoint combination
        for idx_self, idx_other in itertools.product([-1], [0, -1]):
            distances.append(
                np.linalg.norm(
                    self.coordinates[idx_self] - other_segment.coordinates[idx_other]
                )
            )

        return min(distances), 0.0

    def oriented_distance_point(
        self, o_center: np.ndarray, endpoint: bool = False
    ) -> Tuple[float, float]:
        _, _, center, _, ev, _ = self.compute_pca()
        center = np.asarray(center)

        if endpoint:
            center = self.coordinates[-1, 0]

        if len(ev) < 2:
            return 10000, 10000

        # Project distance between centers to long and lat direction of current element
        d = o_center - center
        long_dist = d @ ev[0]
        lat_dist = d @ ev[1]

        return long_dist, lat_dist

    def intersection(self, other_segment: "ContourSegment") -> Tuple[np.ndarray, float]:
        """

        Approximates this and other contour by line (first and last coordinate)

        Args:
            other_segment:

        Returns:
            None, if no intersection exists
            Intersection coordinates and relative position for current segment

        """
        p = self.coordinates[0, 0]
        r = self.coordinates[-1, 0] - self.coordinates[0, 0]

        q = other_segment.coordinates[0, 0]
        s = other_segment.coordinates[-1, 0] - other_segment.coordinates[0, 0]

        denom = np.cross(r, s)
        if denom == 0.0:
            return None

        t = np.cross(q - p, s) / denom

        # This is the intersection point
        return p + t * r, t

    def intersection_image_border(self, img_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Check for intersection with image borders.

        :param img_shape: Image shape (h,w)
        :return: (intersection_point, distance_from_center) if intersection exists, (None, None) otherwise
        """
        h, w = img_shape
        borders = [
            np.asarray([0, 0, 0, h]).reshape((2, 1, 2)),  # left
            np.asarray([0, h, w, h]).reshape((2, 1, 2)),  # bottom
            np.asarray([w, h, w, 0]).reshape((2, 1, 2)),  # right
            np.asarray([w, 0, 0, 0]).reshape((2, 1, 2)),  # top
        ]

        for border in borders:
            intersection = ContourSegment.from_coordinates(border).intersection(self)
            if intersection is None:
                continue

            intersection_point, intersection_location = intersection
            # Check if intersection is along border and not before/after
            if 0 <= intersection_location <= 1:
                _, long_distance = self.oriented_distance_point(
                    np.asarray(intersection_point), endpoint=True
                )

                if long_distance > 0:
                    return intersection_point, long_distance

        return None, None


Contour = List[ContourSegment]
