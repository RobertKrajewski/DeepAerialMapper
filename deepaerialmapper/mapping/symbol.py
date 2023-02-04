import glob
import os
from dataclasses import dataclass
from typing import List, Optional

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from loguru import logger

from deepaerialmapper.mapping.semantic_mask import SemanticClass, SemanticMask


class Net(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        """
        Initialize the symbol classification network. The model is designed to optimize the computation effrt considering the relatively easy difficulty in sybol classification.

        :param in_ch: the number of input channel. It is 1 if the input is binary mask.
        :param out_ch: the number of output channel. It is equal to the number of object class.
        """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: input tenosr.
        :return:: output tensor as softmax.
        """
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


@dataclass
class Symbol:
    name: str
    contour: np.ndarray
    ref: List[List]
    box: Optional[np.ndarray] = None
    score: Optional[float] = None
    image: Optional[np.ndarray] = None

    def show(self):
        """
        Visualize symbols by drawing contours.
        """
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
        """
        Define Symbol Detector.

        :param symbols: List of symbols classes.
        :param weight_filepath: Path to symbol classification weight file.
        """

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

    def extract_ref(self, cnts: np.ndarray) -> List[int]:
        """
        Extract 2 reference points from symbol contour. These two points are used as reference points in lanelet.
        It consists of one center point and one of the farthest point.

        :param cnts: contours of symbol in numpy array.
        :return: two reference points of symbol contours.
        """

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

        return ref_points

    def detect(
        self, seg_mask: SemanticMask, min_area: int = 1000, max_area: int = 6000
    ) -> List[Symbol]:
        """
        Detect symbols from SemanticMask. If the symbol has appropriate size (between min and max area), then the type of symbol is classified.

        :param seg_mask: SemanticMask
        :param min_area: Allowed minimum area of symbol
        :param max_area: Allowed maximum area of symbol
        :return: List of detected symbols.
        """

        # Using image closing to merge the closely separated segments
        symbol_mask = seg_mask.class_mask([SemanticClass.SYMBOL]).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        symbol_mask = cv2.morphologyEx(symbol_mask, cv2.MORPH_CLOSE, kernel)

        _, mask = cv2.threshold(symbol_mask, 150, 255, 0)

        symbol_cnts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for cnt in symbol_cnts:
            # determine the type of arrow by pattern matching

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

    def classify(self, symbol_cnt: np.ndarray, mask: SemanticMask, scale=255) -> str:
        """
        Classify the type of symbol. It fetches the symbol mask and crop the relevant pixels.
        The relevant region is fed into classification model, and return the type of symbol.

        :param symbol_cnt: Symbol contours.
        :param mask: SemanticMask is loaded to use symbol mask.
        :param scale: the maximum scale of pixels to make binary mask to visible mask.
        :return: the predicted type of symbol.
        """
        tx, ty, hori, verti = cv2.boundingRect(symbol_cnt)
        roi = ty, ty + verti, tx, tx + hori
        symbol_mask = mask.class_mask(SemanticClass.SYMBOL)
        cropped_mask = symbol_mask[roi[0] : roi[1], roi[2] : roi[3]]

        test_transforms = A.Compose(
            [
                A.Resize(height=68, width=68, p=1),
                ToTensorV2(),
            ]
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._cls_model.eval()

        if device.type == "cuda":
            # if GPU is available, mount the model into GPU.
            self._cls_model.cuda()

        image = cropped_mask.astype(np.uint8) * scale

        with torch.no_grad():
            image_t = test_transforms(image=image)["image"]
            inp = torch.unsqueeze(image_t, 0)
            inp = inp.to(device, dtype=torch.float)
            output = self._cls_model.forward(inp)
            index = output.data.cpu().numpy().argmax()

        return self._patterns[index]
