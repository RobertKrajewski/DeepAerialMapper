import numpy as np
import cv2
import yaml
from deepaerialmapper.map_creation.masks import SemanticClass


config_dir = "configs/mask/demo.yaml"
with open(config_dir, "r") as f:
    config = yaml.safe_load(f)
palette_map = {SemanticClass[i["type"]]: i["palette"] for i in config["class"]}


def show_img(img):
    import matplotlib.pyplot as plt

    plt.imshow(img, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def save_img(img, name):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    filename = f"../../data/synth/{name}.png"  # Save to data directory
    cv2.imwrite(filename, img)
    return filename


def create_straight_road_borders(add_lanemarkings=False):
    # Different road borders - straight
    img = np.zeros((720, 720, 3), np.uint8)
    vegetation_color = np.asarray(
        palette_map[SemanticClass.VEGETATION], np.uint8
    ).reshape((1, 1, 3))
    road_color = np.asarray(palette_map[SemanticClass.ROAD], np.uint8).reshape(
        (1, 1, 3)
    )
    lanemarking_color = np.asarray(
        palette_map[SemanticClass.LANEMARKING], np.uint8
    ).reshape((1, 1, 3))

    # Add grass
    img += vegetation_color

    lanemarking_width = 2 if add_lanemarkings else 0

    # Add a single lane road (3m == 60px)
    img = draw_horizontal_road(
        img, road_color, 60, 60, lanemarking_color, lanemarking_width
    )

    # Add double lane road (6m == 120px)
    img = draw_horizontal_road(
        img, road_color, 180, 120, lanemarking_color, lanemarking_width
    )
    img = draw_horizontal_lanemarking(img, 240, lanemarking_width, lanemarking_color)

    # Add triple lane road (9m == 180px)
    img = draw_horizontal_road(
        img, road_color, 360, 180, lanemarking_color, lanemarking_width
    )
    img = draw_horizontal_lanemarking(img, 420, lanemarking_width, lanemarking_color)
    img = draw_horizontal_lanemarking(img, 480, lanemarking_width, lanemarking_color)

    return img


def create_curved_road_borders():
    # Different road borders - curves
    img = np.zeros((720, 720, 3), np.uint8)
    vegetation_color = palette_map[SemanticClass.VEGETATION]
    road_color = palette_map[SemanticClass.ROAD]

    for i in reversed(range(15)):
        c = road_color if i % 2 else vegetation_color
        img = cv2.circle(img, (0, 0), 60 * (i + 1), c, thickness=-1)

    return img


def draw_horizontal_road(
    img, road_color, y_start, width, lanemarking_color, outer_lanemarking_width=0
):
    s = y_start
    e = y_start + width

    img[s:e] = road_color
    if outer_lanemarking_width > 0:
        img[s : (s + outer_lanemarking_width)] = lanemarking_color
        img[(e - outer_lanemarking_width) : e] = lanemarking_color

    return img


def draw_horizontal_lanemarking(
    img, y_center, width, color, segment_length=None, gap_length=0
):
    if width == 0:
        return img
    y_s = y_center - width // 2
    y_e = y_s + width

    max_x = img.shape[1]
    if segment_length is None:
        segment_length = max_x  # Draw full length
    x_s = 0
    while x_s < max_x:
        x_e = x_s + segment_length
        img[y_s:y_e, x_s:x_e] = color

        x_s += segment_length + gap_length

    return img


def crate_dashed_lanemarkings():
    # Create a large width road with different kind of road markings
    img = np.zeros((720, 720, 3), np.uint8)
    vegetation_color = np.asarray(
        palette_map[SemanticClass.VEGETATION], np.uint8
    ).reshape((1, 1, 3))
    road_color = np.asarray(palette_map[SemanticClass.ROAD], np.uint8).reshape(
        (1, 1, 3)
    )
    lanemarking_color = np.asarray(
        palette_map[SemanticClass.LANEMARKING], np.uint8
    ).reshape((1, 1, 3))

    # Add grass
    img += vegetation_color

    lanemarking_width = 2
    s = 60
    img = draw_horizontal_road(
        img, road_color, s, 180, lanemarking_color, lanemarking_width
    )
    img = draw_horizontal_lanemarking(img, s + 60, lanemarking_width, lanemarking_color)
    img = draw_horizontal_lanemarking(
        img, s + 120, lanemarking_width, lanemarking_color
    )

    lanemarking_width = 2
    s += 210
    img = draw_horizontal_road(
        img, road_color, s, 180, lanemarking_color, lanemarking_width
    )
    img = draw_horizontal_lanemarking(
        img,
        s + 60,
        lanemarking_width,
        lanemarking_color,
        segment_length=60,
        gap_length=100,
    )
    img = draw_horizontal_lanemarking(
        img,
        s + 120,
        lanemarking_width,
        lanemarking_color,
        segment_length=60,
        gap_length=100,
    )

    lanemarking_width = 4
    s += 210
    img = draw_horizontal_road(
        img, road_color, s, 180, lanemarking_color, lanemarking_width
    )
    img = draw_horizontal_lanemarking(
        img,
        s + 60,
        lanemarking_width,
        lanemarking_color,
        segment_length=60,
        gap_length=40,
    )
    img = draw_horizontal_lanemarking(
        img,
        s + 120,
        lanemarking_width,
        lanemarking_color,
        segment_length=60,
        gap_length=40,
    )

    return img


if __name__ == "__main__":
    save_img(create_straight_road_borders(), "0_straight")
    save_img(create_straight_road_borders(True), "1_straight_lanes")
    save_img(create_curved_road_borders(), "3_curves")
    save_img(crate_dashed_lanemarkings(), "2_markings")
