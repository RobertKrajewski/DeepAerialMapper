import pickle
from pathlib import Path

from pyproj import Proj


class Lanelet2Map:
    def __init__(self, lanemarkings, symbols, lanelets, origin, px2m, proj):
        self.lanemarkings = lanemarkings
        self.symbols = symbols
        self.lanelets = lanelets
        self.origin = origin
        self.px2m = px2m
        self.proj = proj

    def export_lanelet2(self, filepath: Path):
        proj = Proj(self.proj)
        nodes = []
        ways = []
        relations = []

        # Write all symbols
        for symbol in self.symbols:
            way_str = [f"<way id='{len(ways)+1}' visible='true' version='1'>"]
            for ref in symbol.ref:
                utm_x = self.origin[0] + self.px2m * ref[0, 0]
                utm_y = self.origin[1] - self.px2m * ref[0, 1]
                long, lat = proj(utm_x, utm_y, inverse=True)
                point_str = f"<node id='{len(nodes)+1}' visible='true' version='1' lat='{lat}' lon='{long}' />"
                way_str.append(f"    <nd ref='{len(nodes)+1}' />")
                nodes.append(point_str)
            way_str.append(f"    <tag k='subtype' v='{symbol.name.lower()}' />")
            way_str.append("    <tag k='type' v='arrow' />")
            way_str.append("</way>")
            ways.append("\n".join(way_str))

        # Write all nodes and ways
        for lanemarking in self.lanemarkings:
            way_str = [f"<way id='{len(ways)+1}' visible='true' version='1'>"]
            for point in lanemarking.contour:
                utm_x = self.origin[0] + self.px2m * point[0, 0]
                utm_y = (
                    self.origin[1] - self.px2m * point[0, 1]
                )  # "-" as UTM has an inverted y-axis
                long, lat = proj(utm_x, utm_y, inverse=True)
                point_str = f"<node id='{len(nodes)+1}' visible='true' version='1' lat='{lat}' lon='{long}' />"
                way_str.append(f"    <nd ref='{len(nodes)+1}' />")
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

        # Write all lanelets
        for way_a, way_b in self.lanelets:
            lanelet_str = [
                f"<relation id='{len(relations)+1}' visible='true' version='1'>",
                f"    <member type='way' ref='{way_a+1}' role='left' />",
                f"    <member type='way' ref='{way_b+1}' role='right' />",
                "    <tag k='location' v='urban' />",
                "    <tag k='one_way' v='no' />",
                "    <tag k='region' v='de' />",
                "    <tag k='subtype' v='road' />",
                "    <tag k='type' v='lanelet' />",
                "</relation>",
            ]
            relations.append("\n".join(lanelet_str))

        with filepath.open("w") as f:
            f.write(
                "<?xml version='1.0' encoding='UTF-8'?>\n<osm version='0.6' generator='JOSM'>"
            )
            f.write("\n".join(nodes))
            f.write("\n".join(ways))
            f.write("\n".join(relations))
            f.write("</osm>")

    def export_lanemarkings(self, filepath: Path):
        contours = [l.contour[:, 0, :] for l in self.lanemarkings]
        with filepath.open("wb") as f:
            pickle.dump(contours, f)
