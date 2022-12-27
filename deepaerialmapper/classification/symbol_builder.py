from torch.utils.data import Dataset
import cv2
import glob
import random

from loguru import logger


def flatten(t):
    return [item for sublist in t for item in sublist]


####################################################
#       Create Train and Test sets
####################################################
def set_files(train_data_path, test_data_path):
    train_image_paths = []  # to store image paths in list
    classes = []  # to store class values

    # 1.
    # get all the paths from train_data_path and append image paths and class to to respective lists
    for data_path in sorted(glob.glob(train_data_path + "/*")):
        classes.append(data_path.split("/")[-1])
        train_image_paths.append(glob.glob(data_path + "/*"))

    train_image_paths = list(flatten(train_image_paths))
    random.shuffle(train_image_paths)

    logger.info("train_image_path example: ", train_image_paths[0])
    logger.info(f"classes: {classes}")

    # 2.
    # create the test_image_paths
    test_image_paths = []
    for data_path in sorted(glob.glob(test_data_path + "/*")):
        test_image_paths.append(glob.glob(data_path + "/*"))

    test_image_paths = list(flatten(test_image_paths))

    logger.info(
        "Train size: {}, Test size: {}".format(
            len(train_image_paths), len(test_image_paths)
        )
    )

    #######################################################
    #      Create dictionary for class indexes
    #######################################################

    idx_to_class = {i: j for i, j in enumerate(classes)}
    class_to_idx = {value: key for key, value in idx_to_class.items()}

    return train_image_paths, test_image_paths, class_to_idx, idx_to_class


#######################################################
#               Define Dataset Class
#######################################################


class SymbolDataset(Dataset):
    def __init__(self, image_paths, class_to_idx, image_type, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.image_type = image_type

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        if self.image_type == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        label_name = image_filepath.split("/")[-2]
        label = self.class_to_idx[label_name]
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label
