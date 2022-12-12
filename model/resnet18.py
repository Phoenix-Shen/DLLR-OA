import torch as t
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import relu
from utils import load_data_fashion_mnist, weight_init, create_folder, accuracy
import datetime
import os
import tensorboardX
import yaml


class Residual(nn.Module):
    """
    Residual block
    """

    def __init__(self, in_channels: int, num_channels: int, use_1x1conv=False, strides=1) -> None:
        """
        Parameters:
        -------
        in_channels: number of channels of the input features
        num_channels: number of channels of the output features,
        use_1x1conv: whether to use 1x1conv to adjust channel of the output features,
        strides: strides of the ALL convolution layers in the class.
        """
        super().__init__()
        # f(x) - x
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3,
                      padding=1, stride=strides),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels,
                      kernel_size=3, padding=1,),
            nn.BatchNorm2d(num_channels),
        )
        # x
        self.use_1x1conv = use_1x1conv
        if use_1x1conv:
            self.adj_conv = nn.Conv2d(
                in_channels, num_channels, kernel_size=1, stride=strides)

    def forward(self, x: Tensor) -> Tensor:
        """
        forward propagation
        -------
        Parameters:
            x: the input tensor, should be [b,c,h,w] shape.
        """
        Y = self.features.forward(x)
        if self.use_1x1conv:
            x = self.adj_conv.forward(x)
        Y = Y + x
        return relu(Y)


def resnet_block(input_channels: int, num_channels: int, num_residuals: int, first_block=False) -> list[Residual]:
    """
    construct resnet bottleneck
    ------
    Parameters:
        in_channels: the channel number of input data
        num_channels: the channel number of output data
        num_residuals: the number of residual blocks
        first_block: if this block is the first block
    Return:
        blk:list[Residual]
    """
    blk = []
    for i in range(num_residuals):
        # if this block is the first block, we should double the number of channels
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                       use_1x1conv=True, strides=2))
        # else we don't need to double the number of channels, and we don't need 1x1 conv.
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class Resnet18(nn.Module):
    """
    Resnet18, with 18 weight layers.
    """

    def __init__(self) -> None:
        super().__init__()
        b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        b2 = nn.Sequential(*resnet_block(64, 64, 2, True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))

        self.net = nn.Sequential(
            b1,
            b2,
            b3,
            b4,
            b5,
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net.forward(x)


class Resnet_FasionMNIST():
    def __init__(self, dataset_settings: dict, train_settings: dict) -> None:
        # save settings to member variables
        self.dataset_settings = dataset_settings
        self.train_settings = train_settings

        # init dataset
        self.train_loader, self.test_loader = load_data_fashion_mnist(
            dataset_settings["train_root"],
            dataset_settings["test_root"],
            dataset_settings["batch_size"],
            dataset_settings["resize"],
        )
        # create results folder
        now = datetime.datetime.now()
        folder_name = now.strftime("%Y-%m-%d %H-%M-%S")
        self.path = os.path.join("results", folder_name)
        create_folder(self.path)
        # save the train settings
        with open(os.path.join(self.path, "config.yaml"), "w") as fp:
            yaml.dump(
                {
                    "dataset": dataset_settings,
                    "train_settings": train_settings,
                },
                fp,
            )
        # device
        self.device = t.device(
            self.train_settings["device"] if t.cuda.is_available() else "cpu")
        # network
        self.net = Resnet18()
        self.net.apply(weight_init)
        self.net.to(self.device)
        # load if the state dict exists
        if self.train_settings["pretrained_parameters"] is not None and os.path.exists(
            self.train_settings["pretrained_parameters"]
        ):
            self.net.load_state_dict(
                t.load(self.train_settings["pretrained_parameters"]))
        # optimizer
        self.optimizer = t.optim.Adam(
            self.net.parameters(),
            lr=self.train_settings["lr"],
            weight_decay=self.train_settings["l2_penalty"])
        # loss function
        self.loss_func = nn.CrossEntropyLoss()
        # tensorboardX
        self.writer = tensorboardX.SummaryWriter(self.path)

    def train(self):
        """start the training procedure"""
        print("Training on", self.device)

        # get the number of the batches
        num_batches = len(self.train_loader)
        # start the training procedure (doubel for loop)
        train_epochs = self.train_settings["train_epoch"]
        for epoch in range(train_epochs):
            self.net.train()
            for i, (X, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.net.forward(X)
                l = self.loss_func(y_hat, y)
                l.backward()
                self.optimizer.step()
                with t.no_grad():
                    train_acc = accuracy(y_hat, y)
                self.writer.add_scalar(
                    "train_loss", l.item(), num_batches*epoch+i)
                self.writer.add_scalar(
                    "train_acc", train_acc, num_batches*epoch+i)

            # test
            avg_acc, avg_loss = self.test()
            self.writer.add_scalar("test_acc", avg_acc, epoch)
            self.writer.add_scalar("test_loss", avg_loss, epoch)
            # print results
            print(
                f"[{epoch}/{train_epochs:.3f}] train_loss: {l.item():.3f}, test_loss: {avg_loss:.3f}, train_acc: {train_acc*100:.3f}%, test_acc: {avg_acc*100:.3f}%")
        # save model
        print("training ended, saving parameters at {}".format(
            os.path.join(self.path, "model.pth")))
        t.save(self.net.parameters(), os.path.join(self.path, "model.pth"))

    def test(self, verbose=False):
        """
        start the test procedure,if verbose is True, then print the information
        """
        accmulate_acc = 0
        accmulate_loss = 0
        for i, (X, y) in enumerate(self.test_loader):
            X, y = X.to(self.device), y.to(self.device)
            y_hat = self.net.forward(X)
            l = self.loss_func(y_hat, y)
            test_acc = accuracy(y_hat, y)

            accmulate_acc += test_acc
            accmulate_loss += l.item()
            if verbose:
                print(
                    f"[{i}/{len(self.test_loader)}]- test_acc: {test_acc}, test_loss: {l.item()}")
        avg_acc = accmulate_acc / len(self.test_loader)
        avg_loss = accmulate_loss/len(self.test_loader)
        if verbose:
            print("avg_acc: {}, avg_loss: {}".format(avg_acc, avg_loss))
        return avg_acc, avg_loss
