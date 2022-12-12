import tensorboardX
import torch as t
import torch.nn as nn
from model.resnet18 import Resnet_FasionMNIST
import utils
import argparse


def train_resnet18():
    parser = argparse.ArgumentParser(
        prog="train_resnet18", description="load the resnet18 config file"
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="the config file path",
        default="configs\\resnet18.yaml",
    )

    parser.add_argument(
        "--operation",
        type=str,
        help="train or test",
        default="train",
    )

    config_arg = parser.parse_args()
    args = utils.load_config(config_arg.config_path)
    model = Resnet_FasionMNIST(
        args["dataset_settings"], args["train_settings"])
    if config_arg.operation == "train":
        model.train()
    else:
        model.test(True)


if __name__ == "__main__":
    train_resnet18()
