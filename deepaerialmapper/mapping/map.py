import pickle
from pathlib import Path
from typing import FrozenSet, List, Set, Tuple

from pyproj import Proj

from deepaerialmapper.mapping.symbol import Symbol


class Map:
    def __init__(
        self,
        lanemarkings,
        lanelets: Set[FrozenSet[int]],
        symbols: List[Symbol],
        proj: str,
        origin: Tuple[float, float],
        px2m: float,
    ) -> None:
        """Representation of a map including geo-referenced lanemarkings, symbols and lanelets.
        Implements export to lanelet2 format as well as storing lanemarkings for evaluation purposes.

        :param lanemarkings: List of lanemarkigns
        :param lanelets: Association of the lanemarkings to lanelets
        :param symbols: List of symbols, mainly arrows
        :param proj: epsg code of the projection used, e.g. "epsg:25832"
        :param origin: Utm coordinates of the top left corner of the used satellite image.
        :param px2m: Conversion factor from pixel in satellite image to meters.
        """
        self._lanemarkings = lanemarkings
        self._symbols = symbols
        self._lanelets = lanelets
        self._origin = origin
        self._px2m = px2m
        self._proj = proj

    def export_lanelet2(self, filepath: Path) -> None:
        """Export to lanelet2 format compatible with JOSM"""
        proj = Proj(self._proj)
        nodes: List[str] = []
        ways: List[str] = []
        relations: List[str] = []

        # Create nodes and ways for symbols
        for symbol in self._symbols:
            way_str = [f"<way id='{len(ways) + 1}' visible='true' version='1'>"]
            for point in symbol.centerline:
                # Convert from pixel coordinates to utm
                utm_x = self._origin[0] + self._px2m * point[0]
                utm_y = self._origin[1] - self._px2m * point[1]

                # Convert from utm to long/lat
                long, lat = proj(utm_x, utm_y, inverse=True)

                point_str = f"<node id='{len(nodes) + 1}' visible='true' version='1' lat='{lat}' lon='{long}' />"
                nodes.append(point_str)
                way_str.append(f"    <nd ref='{len(nodes) + 1}' />")
            way_str.append(f"    <tag k='subtype' v='{symbol.name.lower()}' />")
            way_str.append("    <tag k='type' v='arrow' />")
            way_str.append("</way>")
            ways.append("\n".join(way_str))

        # Create nodes and ways for lanemarkings
        for lanemarking in self._lanemarkings:
            way_str = [f"<way id='{len(ways) + 1}' visible='true' version='1'>"]
            for point in lanemarking.contour:
                utm_x = self._origin[0] + self._px2m * point[0, 0]
                utm_y = (
                    self._origin[1] - self._px2m * point[0, 1]
                )  # "-" as UTM has an inverted y-axis
                long, lat = proj(utm_x, utm_y, inverse=True)
                point_str = f"<node id='{len(nodes) + 1}' visible='true' version='1' lat='{lat}' lon='{long}' />"
                way_str.append(f"    <nd ref='{len(nodes) + 1}' />")
                nodes.append(point_str)
            if lanemarking.type_ == lanemarking.LanemarkingType.SOLID:
                way_str.append("    <tag k='type' v='line_thin' />")
                way_str.append("    <tag k='subtype' v='solid' />")
            elif lanemarking.type_ == lanemarking.LanemarkingType.DASHED:
                way_str.append("    <tag k='type' v='line_thin' />")
                way_str.append("    <tag k='subtype' v='dashed' />")
            elif lanemarking.type_ == lanemarking.LanemarkingType.ROAD_BORDER:
                way_str.append("    <tag k='type' v='road_border' />")
            way_str.append("</way>")
            ways.append("\n".join(way_str))

        # Create relations for lanelets
        for way_a, way_b in self._lanelets:
            lanelet_str = [
                f"<relation id='{len(relations) + 1}' visible='true' version='1'>",
                f"    <member type='way' ref='{way_a + 1}' role='left' />",
                f"    <member type='way' ref='{way_b + 1}' role='right' />",
                "    <tag k='location' v='urban' />",
                "    <tag k='one_way' v='no' />",
                "    <tag k='region' v='de' />",
                "    <tag k='subtype' v='road' />",
                "    <tag k='type' v='lanelet' />",
                "</relation>",
            ]
            relations.append("\n".join(lanelet_str))

        # Compose and write lanelet2 from individual components
        with filepath.open("w") as f:
            f.write(
                "<?xml version='1.0' encoding='UTF-8'?>\n"
                "<osm version='0.6' generator='JOSM'>"
            )
            f.write("\n".join(nodes))
            f.write("\n".join(ways))
            f.write("\n".join(relations))
            f.write("</osm>")

    def export_lanemarkings(self, filepath: Path) -> None:
        """Export lanemarkings only to a pickle file for accuracy evaluation."""
        contours = [
            l.contour[:, 0, :] for l in self._lanemarkings
        ]  # Remove obsolete axis 1

        with filepath.open("wb") as f:
            pickle.dump(contours, f)
