import glob
import os
from dataclasses import dataclass
from typing import List, Optional

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from loguru import logger

from deepaerialmapper.mapping.semantic_mask import SemanticClass, SemanticMask


class Net(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_ch, out_channels=16, kernel_size=5, stride=2
        )
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(
            2304, 256
        )  # First channel size varies when you change parameter
        self.fc2 = nn.Linear(256, out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.dropout1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.dropout1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


def predict(
    model,
    image: np.ndarray,
    targets: list,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    test_transforms = A.Compose(
        [
            A.Resize(height=68, width=68, p=1),
            ToTensorV2(),
        ]
    )
    model.eval()
    if device.type == "cuda":
        model.cuda()

    with torch.no_grad():
        image_t = test_transforms(image=image)["image"]
        inp = torch.unsqueeze(image_t, 0)
        inp = inp.to(device, dtype=torch.float)
        output = model.forward(inp)
        index = output.data.cpu().numpy().argmax()
        return targets[index]


@dataclass
class Symbol:
    name: str
    contour: np.ndarray
    ref: List[List]
    box: Optional[np.ndarray] = None
    score: Optional[float] = None
    image: Optional[np.ndarray] = None

    def show(self):
        max_x, max_y = np.max(self.contour, axis=(0, 1))
        img = np.zeros((max_y + 10, max_x + 10), np.uint8)
        img = cv2.polylines(
            img, [self.contour], isClosed=True, color=(255, 255, 255), thickness=2
        )

        import matplotlib.pyplot as plt

        plt.imshow(img)
        plt.show()


class SymbolDetector:
    def __init__(self, symbols: List[str], weight_filepath: str) -> None:

        self._patterns = symbols  # order of the list is important.
        logger.info(f"Loaded patterns: {self._patterns}")

        self._cls_model = Net(in_ch=1, out_ch=len(symbols))
        self._cls_model.load_state_dict(
            torch.load(
                weight_filepath,
                map_location=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )
        )

    def _load_symbols(self):
        """Load symbols from disk"""

        symbols = []
        for pattern_type in self.pattern_types:

            filepaths = glob.glob(
                os.path.join(self.pattern_dir, pattern_type) + "/*.png"
            )  # if pattern is not png, needs to be changed.
            for filepath in filepaths:
                mask = cv2.imread(filepath)

                # Reduce the noise in the template
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(mask, 150, 255, 0)
                mask = cv2.resize(mask, dsize=None, fx=7.0 / 7.0, fy=7.0 / 7.0)

                # Smooth the contour of the template
                cnts, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                epsilon = 0.008 * cv2.arcLength(cnts[0], True)
                approx = cv2.approxPolyDP(cnts[0], epsilon, True)

                ref_point = self.extract_ref(approx)

                symbols.append(
                    Symbol(name=pattern_type, contour=approx, image=mask, ref=ref_point)
                )

        return symbols

    def extract_ref(self, cnts, debug=False):
        """Extract 2 reference points from contour"""

        epsilon = 0.008 * cv2.arcLength(cnts, True)
        approx = cv2.approxPolyDP(cnts, epsilon, True)

        far_dist = 0
        for i, point_i in enumerate(approx):
            for j, point_j in enumerate(approx):
                if i >= j:
                    continue

                dist_ij = np.linalg.norm(point_i - point_j)
                if dist_ij > far_dist:
                    far_dist = dist_ij
                    ref_points = [point_i, point_j]

        if debug:
            mask = np.zeros(
                np.append(np.squeeze(np.flip(np.amax(approx, axis=0), axis=None)), 3)
            )  # zeros(y, x, 3)
            cv2.drawContours(mask, [approx], -1, (0, 255, 0), 1)

            mask = cv2.circle(mask, ref_points[0][0], 2, (255, 0, 0), thickness=-1)
            mask = cv2.circle(mask, ref_points[1][0], 2, (255, 0, 0), thickness=-1)
            plt.imshow(mask, cmap="jet")
            plt.show()

        return ref_points

    def detect(
        self,
        seg_mask: SemanticMask,
        min_area: int = 1000,
        max_area: int = 6000,
        debug: bool = False,
        dbg_rescale: float = 0.75,
    ) -> List[Symbol]:
        """
        using shape match to classify the detected symbol
        load symbol template and compare the detected ones with them, find the most likely one.
        """

        # Using image closing to merge the closely separated segments
        symbol_mask = seg_mask.class_mask([SemanticClass.SYMBOL]).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        symbol_mask = cv2.morphologyEx(symbol_mask, cv2.MORPH_CLOSE, kernel)

        if debug:
            symbol_mask = cv2.resize(symbol_mask, None, fx=dbg_rescale, fy=dbg_rescale)

        _, mask = cv2.threshold(symbol_mask, 150, 255, 0)

        symbol_cnts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for cnt in symbol_cnts:
            # determine the type of arrow by pattern matching
            scores = []

            epsilon = 0.001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            rect = cv2.minAreaRect(approx)  # Calculate rotated box with minimum area
            box = cv2.boxPoints(rect).astype(int)

            # Check for appropriate size
            area = cv2.contourArea(box)
            if area < min_area or area > max_area:
                continue

            ref_point = self.extract_ref(approx)

            # svm doesn't need to go through all patterns.
            best_name = self.classify(cnt, seg_mask)
            detections.append(
                Symbol(name=best_name, contour=approx, ref=ref_point, box=box)
            )

        return detections

    def classify(self, symbol_cnt: List[np.ndarray], mask, scale=255):
        tx, ty, hori, verti = cv2.boundingRect(symbol_cnt)
        roi = ty, ty + verti, tx, tx + hori
        symbol_mask = mask.class_mask(SemanticClass.SYMBOL)
        cropped_mask = symbol_mask[roi[0] : roi[1], roi[2] : roi[3]]

        return predict(
            self._cls_model, cropped_mask.astype(np.uint8) * scale, self._patterns
        )
