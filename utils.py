from torch.utils import data
from torchvision import transforms
import torchvision
import yaml
# %%
import os
import torch as t
from torch import Tensor
import yaml


def load_config(file_path: str) -> dict:
    """
    load the settings and return a dictionary
    ------
    Parameters:
        file_path: path to the configuration file
    Returns:
        dict: the settings dictionary
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found!")
    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, yaml.FullLoader)

    # print settings
    print("your config:")

    recursive_print_cfg(config, key="root")
    return config


def recursive_print_cfg(config: dict, key: str, level=0):
    if isinstance(config, dict):
        print("    "*level+key)
        for key in config.keys():
            # print(key+":")
            recursive_print_cfg(config[key], key, level=level+1)
    else:
        print("    "*level+f"[{key}]".ljust(25), "->", config)


def create_folder(folder_path: str) -> None:
    """
    Create the folder if not exisits
    ------
    parameters:
        folder_path: path to the folder
    Returns:
        None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_model_size(model: t.nn.Module) -> int:
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    # print("total size of the model is:{:.3f}MB".format(all_size))

    return all_size
# %%


def load_data_fashion_mnist(train_root, test_root, batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory.
    Defined in :numref:`sec_fashion_mnist`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=train_root, train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root=test_root, train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=0),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=0))


def weight_init(m: t.nn.Module):
    if type(m) == t.nn.Linear or type(m) == t.nn.Conv2d:
        t.nn.init.xavier_uniform_(m.weight)


def accuracy(y_hat: t.Tensor, y: t.Tensor) -> float:
    """
    Compute the number of correct predictions
    ------
    Parameters:
        y_hat: the predicted results
        y: the ground truth.
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = t.argmax(y_hat, dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(sum(cmp.type(y.dtype)))/y_hat.shape[0]
