from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from deepaerialmapper.mapping.lanemarking import Lanemarking
from deepaerialmapper.mapping.masks import ClassMask


class MaskVisualizer:
    def __init__(self):
        self._idx_sel = -1

    def show(
        self,
        class_mask: ClassMask,
        contours=None,
        lanemarkings=None,
        background=None,
        show=False,
        window_name: str = "",
        random: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        from deepaerialmapper.mapping.contour import ContourManager

        if isinstance(contours, ContourManager):
            contours = contours.contours

        self._background = background
        self._contours = contours
        self._img = class_mask.as_color_img()
        self._lanemarkings = lanemarkings
        self._random = random
        img_c, img_overlay = self._draw()

        if show:
            self._fig, self._ax = plt.subplots()
            self._fig.canvas.mpl_connect("key_press_event", self._key_press)
            if img_overlay is None:
                self._ax.imshow(img_c)
            else:
                self._ax.imshow(img_overlay)

            # Maximise window
            figManager = plt.get_current_fig_manager()
            figManager.window.state("zoomed")

            # Add window title
            if window_name:
                self._fig.canvas.set_window_title(window_name)
            plt.show()

        return img_c, img_overlay

    def _draw(self):
        if self._background is not None:
            img_c = self._background.copy()
        else:
            img_c = self._img.copy()

        if self._contours is None and self._lanemarkings is None:
            return img_c, None

        if self._contours is not None:
            contours = self._contours
            if self._idx_sel >= 0 and self._idx_sel < len(contours):
                contours = contours[self._idx_sel : self._idx_sel + 1]
            img_c = self._draw_contours(contours, img_c, self._random)

        if self._lanemarkings is not None:
            lanemarkings = self._lanemarkings
            if self._idx_sel >= 0 and self._idx_sel < len(lanemarkings):
                lanemarkings = lanemarkings[self._idx_sel : self._idx_sel + 1]
            img_c = self._draw_lanemarkings(img_c, lanemarkings, self._random)

        img_overlay = cv2.addWeighted(self._img, 0.5, img_c, 0.5, 0.0)
        return img_c, img_overlay

    def _key_press(self, event):
        if event.key == "1":
            d = -5
        elif event.key == "2":
            d = -1
        elif event.key == "3":
            d = +1
        elif event.key == ",":
            d = +5
        else:
            d = 0
        self._idx_sel += d
        r = self._draw()
        self._ax.imshow(r[1])
        title = f"{self._idx_sel}"
        if self._contours:
            title += f"/{len(self._contours)} contours"
        if self._lanemarkings:
            title += f"/{len(self._lanemarkings)} lanemarkings"
        self._fig.canvas.set_window_title(title)
        self._fig.canvas.draw()

    @staticmethod
    def _draw_contours(contours, img_c, random):
        lines = []
        points = []
        texts = []

        for i_contour, contour in enumerate(contours):
            if random:
                line_color = np.random.randint(256, size=3).tolist()
            else:
                line_color = (150, 255, 0)
            lines.append((contour, line_color))

            # Draw intermediate points
            for i in range(contour.shape[0]):
                if i == 0:
                    color = (255, 0, 0)  # red
                elif i == contour.shape[0] - 1:
                    color = (0, 0, 0)  # black
                else:
                    color = (0, 0, 255)  # blue

                points.append((contour, i, 3, color))

            text_pos = np.array(contour[len(contour) // 2, 0]) + np.random.uniform(
                -3, 3, 2
            )
            text_pos = np.round(text_pos).astype(int)
            texts.append((f"{i_contour}", text_pos))

        # Draw all lines
        for contour, color in lines:
            img_c = cv2.polylines(
                img_c,
                [contour],
                isClosed=False,
                color=color,
                thickness=19,
                lineType=cv2.LINE_AA,
            )

        # Draw all points
        for contour, i, size, color in points:
            img_c = cv2.circle(img_c, contour[i, 0], size, color, thickness=-1)

        # Draw all texts
        for text, pos in texts:
            cv2.putText(
                img_c,
                text,
                pos,
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return img_c

    @staticmethod
    def _draw_lanemarkings(img_c, lanemarkings, random):
        lines = []
        points = []
        texts = []

        for i_lanemarking, lanemarking in enumerate(lanemarkings):
            contour = lanemarking.contour

            if random:
                color = np.random.randint(256, size=3).tolist()
            elif lanemarking.type_ == Lanemarking.LanemarkingType.ROAD_BORDER:
                color = (255, 0, 255)
            elif lanemarking.type_ == Lanemarking.LanemarkingType.SOLID:
                color = (0, 128, 255)
            else:
                color = (150, 255, 0)
            lines.append((contour, color))

            # Draw intermediate points
            for i in range(contour.shape[0]):
                if i == 0:
                    # start point
                    color = (255, 0, 0)  # red
                    size = 4
                elif i == contour.shape[0] - 1:
                    # end point
                    color = (0, 0, 0)  # black
                    size = 4
                else:
                    color = (0, 0, 255)  # blue
                    size = 3

                points.append((contour, i, size, color))

            texts.append((f"{i_lanemarking}", contour[len(contour) // 2, 0]))

        # Draw all lines
        for contour, color in lines:
            img_c = cv2.polylines(
                img_c,
                [contour],
                isClosed=False,
                color=color,
                thickness=3,
                lineType=cv2.LINE_AA,
            )

        # Draw all points
        for contour, i, size, color in points:
            img_c = cv2.circle(img_c, contour[i, 0], size, color, thickness=-1)

        # Draw all texts
        for text, pos in texts:
            cv2.putText(
                img_c,
                text,
                pos,
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return img_c
