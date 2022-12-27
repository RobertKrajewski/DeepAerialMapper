from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import cv2
import os
import numpy as np
from typing import List
from deepaerialmapper.map_creation.masks import SemanticClass

from deepaerialmapper.classification.symbol_builder import SymbolDataset, set_files
from loguru import logger


def classify(symbol_cnt: List[np.ndarray], mask, cls_order, cls_weight, scale=255):
    cls_model = Net(in_ch=1, out_ch=len(cls_order))
    cls_model.load_state_dict(
        torch.load(
            cls_weight,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    )

    tx, ty, hori, verti = cv2.boundingRect(symbol_cnt)
    roi = ty, ty + verti, tx, tx + hori
    symbol_mask = mask.class_mask(SemanticClass.SYMBOL)
    cropped_mask = symbol_mask[roi[0] : roi[1], roi[2] : roi[3]]

    return predict(cls_model, cropped_mask.astype(np.uint8) * scale, cls_order)


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


def train(
    args, model, device, train_loader, optimizer, epoch, train_losses, train_counter
):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        # zero the parameter gradients
        optimizer.zero_grad()

        data, target = data.to(device, dtype=torch.float), target.to(device)

        # forward
        output = model.forward(data)
        loss = F.nll_loss(output, target)

        # update weights
        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
        )
        if args.dry_run:
            break


def test(args, model, device, epoch, test_loader, test_losses):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float), target.to(device)
            output = model.forward(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    if epoch % args.log_interval == 0:
        logger.info(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--train_path", default="images/symbol/train", help="the dir to training images"
    )
    parser.add_argument(
        "--test_path", default="images/symbol/test", help="the dir to testing images"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=700,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        metavar="M",
        help="Learning rate step gamma (default: 0.9)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--check_point", default="checkpoints", help="the dir to save the checkpoint"
    )
    parser.add_argument("--shape", default="68", help="shape to resize input images")

    args = parser.parse_args()
    return args


def main(image_type, debug=False):
    # Training settings
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 4, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    #######################################################
    #               Define Transforms
    #######################################################

    # To define an augmentation pipeline, you need to create an instance of the Compose class.
    # As an argument to the Compose class, you need to pass a list of augmentations you want to apply.

    train_transforms = A.Compose(
        [
            A.SafeRotate(
                limit=[-180, 180],
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=1,
                p=1,
            ),
            # mask value could be 255. It should be verified.,
            A.Resize(height=68, width=68, p=1),
            # TODO: add random box
            ToTensorV2(),
        ]
    )

    test_transforms = A.Compose(
        [
            A.Resize(height=68, width=68, p=1),
            # TODO: apply rotate
            ToTensorV2(),
        ]
    )

    train_image_paths, test_image_paths, class_to_idx, idx_to_class = set_files(
        args.train_path, args.test_path
    )

    #######################################################
    #                  Create Dataset
    #######################################################

    train_dataset = SymbolDataset(
        train_image_paths, class_to_idx, image_type, train_transforms
    )
    test_dataset = SymbolDataset(
        test_image_paths, class_to_idx, image_type, test_transforms
    )

    logger.info(
        f"The shape of tensor for the first image in train dataset: {train_dataset[0][0].shape} "
    )
    logger.info(
        f"The label for the first image in train dataset: {idx_to_class[train_dataset[0][1]]}"
    )

    #######################################################
    #                  Define Dataloaders
    #######################################################

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = []

    if image_type == "RGB":
        in_ch = 3
    else:
        in_ch = 1

    num_class = len(class_to_idx)

    model = Net(in_ch, num_class).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)

    scheduler = StepLR(optimizer, step_size=500, gamma=args.gamma, verbose=False)
    for epoch in range(1, args.epochs + 1):
        test_counter.append(epoch * len(train_dataset))
        train(
            args,
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            train_losses,
            train_counter,
        )
        test(args, model, device, epoch, test_loader, test_losses)
        scheduler.step()

    if args.save_model:
        torch.save(model, os.path.join(args.check_point, "symbol.pt"))

    if debug:
        import matplotlib.pyplot as plt

        plt.plot(train_counter, train_losses, color="blue")
        plt.scatter(test_counter, test_losses, color="red")
        plt.legend(["Train Loss", "Test Loss"], loc="upper right")
        plt.xlabel("number of training examples seen")
        plt.ylabel("negative log likelihood loss in log scale")
        plt.yscale("log")
        plt.title("training loss")
        plt.show()
        visualize_augmentations(
            model,
            test_dataset,
            idx_to_class,
            np.random.randint(1, len(test_image_paths)),
            random_img=True,
        )


#######################################################
#                  Visualize Dataset
#         Images are plotted after augmentation
#######################################################


def visualize_augmentations(
    model, dataset, idx_to_class, samples=10, cols=5, random_img=False
):
    import matplotlib.pyplot as plt
    import copy

    dataset = copy.deepcopy(dataset)

    # we remove the normalize and tensor conversion from our augmentation pipeline
    dataset.transform = A.Compose(
        [t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))]
    )
    rows = samples // cols

    _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    for i in range(samples):
        if random_img:
            i = np.random.randint(1, dataset.__len__())
        image, lab = dataset[i]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
        ax.ravel()[i].set_title(predict(model, image, list(idx_to_class.values())))

    plt.tight_layout(pad=1)
    plt.show()


if __name__ == "__main__":
    main(image_type="Gray")  # RGB or Gray
