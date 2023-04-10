from dataclasses import dataclass
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from loguru import logger

from deepaerialmapper.mapping.contour import ContourSegment
from deepaerialmapper.mapping.semantic_mask import (
    BinaryMask,
    SemanticClass,
    SemanticMask,
)


@dataclass
class Symbol:
    # Type of symbol, e.g. "arrow_right"
    name: str

    # Contour [N, 1, 2(x,y)]
    contour: np.ndarray

    # Centerline described by the two most distant points
    centerline: Tuple[np.ndarray, np.ndarray]

    def show(self):
        """Visualize symbols by drawing contours."""
        max_x, max_y = np.max(self.contour, axis=(0, 1))
        img = np.zeros((max_y + 10, max_x + 10), np.uint8)
        img = cv2.polylines(
            img, [self.contour], isClosed=True, color=(255, 255, 255), thickness=2
        )

        import matplotlib.pyplot as plt

        plt.imshow(img)
        plt.show()


class SymbolClassifier(nn.Module):
    def __init__(self, weight_filepath: str, num_img_channels: int, classes: List[str]):
        """Convolutional network to classify cropped binary masks of road symbols.

        :param num_img_channels: Number of channels of the input images (1 in case of binary masks).
        :param num_classes: Number of classes the network should be able to differentiate between.
        """

        super(SymbolClassifier, self).__init__()
        self.conv1 = nn.Conv2d(num_img_channels, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(2304, 256)
        self.fc2 = nn.Linear(256, len(classes))

        self._transforms = A.Compose(
            [
                A.Resize(height=68, width=68, p=1),
                ToTensorV2(),
            ]
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(weight_filepath, map_location=self._device))
        if self._device.type == "cuda":
            self.cuda()
            logger.info("Running symbol classification on GPU")
        else:
            logger.info("Running symbol classification on CPU")
        self.eval()

        self._classes = classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps image to soft-maxed classification output"""
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

    def classify(self, contour: np.ndarray, mask: BinaryMask) -> str:
        """Classify symbol mask.

        After cropping an image of  symbol from the mask using the given contour, apply the network.

        :param contour: Symbol contours.
        :param mask: SemanticMask is loaded to use symbol mask.
        :return: the predicted type of symbol.
        """
        x, y, width, height = cv2.boundingRect(contour)
        image = mask.as_bw_img()[y : (y + height), x : (x + width)]

        with torch.no_grad():
            # Prepare image
            input_image = self._transforms(image=image)["image"]
            input_image = torch.unsqueeze(input_image, 0)
            input_image = input_image.to(self._device, dtype=torch.float)

            # Classify
            output = self.forward(input_image)
            index = output.data.cpu().numpy().argmax()

        return self._classes[index]


class SymbolDetector:
    def __init__(
        self,
        symbols_names: List[str],
        classifier_weight_filepath: str,
        close_size: int = 5,
        min_area: int = 1000,
        max_area: int = 6000,
    ) -> None:
        """Detection and classification of symbols_names in binary masks using a neural network.

        :param symbols_names: List of symbols_names classes supported by the classifier.
        :param classifier_weight_filepath: Path to symbol classification weight file.
        :param min_area: Minimum area in px2 of rotated box enclosing found symbol contour
        :param max_area: Maximum area in px2 of rotated box enclosing found symbol contour
        """

        self._cls_model = SymbolClassifier(
            classifier_weight_filepath, num_img_channels=1, classes=symbols_names
        )
        self._close_size = close_size
        self._min_area = min_area
        self._max_area = max_area

    def detect(self, seg_mask: SemanticMask) -> List[Symbol]:
        """Detect and classify all symbols in a SemanticMask.

        Symbols are checked for an appropriate size.

        :param seg_mask: SemanticMask containing symbol class
        :return: List of detected symbols.
        """

        # Using image closing to merge the closely separated segments
        symbol_mask = seg_mask.class_mask([SemanticClass.SYMBOL])
        contours = symbol_mask.close(self._close_size).contours()
        logger.info(f"Found {len(contours)} symbol contours")

        symbols = []
        for contour in contours:
            # Check for appropriate size
            approx = cv2.approxPolyDP(
                contour, 0.001 * cv2.arcLength(contour, True), True
            )
            rect = cv2.minAreaRect(approx)  # Calculate rotated box with minimum area
            box = cv2.boxPoints(rect).astype(int)
            area = cv2.contourArea(box)
            if area < self._min_area or area > self._max_area:
                logger.info(
                    f"Found symbol contour outside of allowed area range: {area}px2"
                )
                continue

            centerline = ContourSegment.from_coordinates(approx).centerline()
            if centerline is None:
                logger.warning("Could not find centerline!")
                continue

            symbol_name = self._cls_model.classify(contour, symbol_mask)
            symbol = Symbol(name=symbol_name, contour=approx, centerline=centerline)
            symbols.append(symbol)
            logger.info(
                f'Found "{symbol_name}" symbol at {np.array2string(np.mean(box, axis=0))}'
            )

        return symbols
