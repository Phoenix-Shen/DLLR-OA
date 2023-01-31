from torch.utils import data
from torchvision import transforms
import torchvision
import yaml
# %%
import os
import torch as t
from torch import Tensor
import yaml
import torch.nn as nn
import numpy as np
from numpy import ndarray


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


def calculate_weight_norm(net: nn.Module, component_keys: list[str] = None):
    """
    Calculate the 2nd-norm of the weight layer of the network, will used in the 
    component choosing strategy and the power allocation coefficient computation
    ------
    Parameters:
        net: the given network
        component_keys: if None, return all components, else, return the 2nd-norm of the specified parameters
    Returns:
        dict: the sortd weight norm dictionary, if component_keys is not None, 
        return the 2nd-norm in order of the component_keys
    """

    weight_norm = {}
    with t.no_grad():
        if component_keys is not None:
            for name, param in net.named_parameters():
                if name in component_keys:
                    weight_norm[name] = t.norm(param).item()
        else:
            for name, param in net.named_parameters():
                if "weight" in name or "bias" in name:
                    weight_norm[name] = t.norm(param).item()

    if component_keys is None:
        weight_norm = sorted(weight_norm.items(),
                             key=lambda x: x[1], reverse=True)

    return weight_norm


def calculate_grad_norm(net: nn.Module):
    """
    calculate the 2nd-norm of the weight layer of the network
    ------
    Parameters:
        net: the given network
    Returns:
        dict: the sortd weight norm dictionary
    """
    grad_norm = {}
    for name, param in net.named_parameters():
        if "weight" in name or "bias" in name:
            grad_norm[name] = t.norm(param.grad).item()

    grad_norm = sorted(grad_norm.items(), key=lambda x: x[1], reverse=True)
    return grad_norm


def get_weight_num(net: nn.Module):
    """
    get the number of the weight layer
    ------
    Parameters:
        net: nn.Module, the given net
    return:
        int, the number of weight layers
    """
    num = 0
    for name, _ in net.named_parameters():
        if "weight" in name or "bias" in name:
            num += 1
    return num


def compute_power_coeff(E: float, W: float, channel_gain: ndarray, x: ndarray, pow_limit: bool):
    """
    compute the power allocation coefficient according to the channel conditions
    ------
    Parameters:
        E: float, the E_{ij} in equation (3)
        W: float, the W_{ij} in equation (3), W is the connectivity matrix
        channel_gain: the channel gain of the specified local device
        x: the component of each sub_carrier, i.e. the x_{ij}(k) in equation (3)
        pow_limit: whether the transmition power is limited
    Returns:
        ndarray: the power allocation coefficient of all channels
        float: xi
    """

    # first, calculate \xi^* according to equation (3)
    denominator = W**2 * np.sum(np.power(x, 2)/np.power(channel_gain, 2))
    xi = np.sqrt(E/denominator) if denominator != 0. else 0
    # second, caculate the b_{ij}^*(k)
    if pow_limit:
        b = xi*W/channel_gain
        # finally, return b
    else:
        b = W/channel_gain
    return b, xi


def compute_alpha(id: int, xi_neighbors: ndarray, weight_neighbors: ndarray = None, pow_limit=False):
    """
    compute alpha_i for local clients
    ------
    Parameters:
        id: int, the id of the local client
        xi_neighbors: ndarray
        weight_neighbors: ndarray
        pow_limit: whether the transmition power is limited
    Returns:
        the estimated alpha, if weight_neighbors is None, use equation (5) else use equation (6)
    """
    if pow_limit:
        xi_neighbors[id] = 0
        weight_neighbors[id] = 0
        # if weight_neighbors is none, use equation (5)
        if weight_neighbors is None:
            alpha = (xi_neighbors.shape[0]-1)/np.sum(xi_neighbors)
        # else use equation (6)
        else:
            alpha = np.sum(weight_neighbors) / \
                np.sum(weight_neighbors*xi_neighbors)
    else:
        alpha = 1.
    return alpha


def aggregation(model_dicts: list[dict], sigma: float):
    """
    aggregate the models through over-the-air computation,
    and add noise to each component of the model
    ------
    Parameters:
        model_dicts: the state dicts of the models
        sigma: the variance of the Gaussian noise
    Returns:
        the model after adding noise
    """
    processed_model = model_dicts[0]
    # begin aggregation
    for key in model_dicts[0].keys():
        for model in model_dicts[1:]:
            processed_model[key] += model[key]
        # add noise
        processed_model[key] += t.randn_like(processed_model[key])*sigma
    return processed_model


def gen_topo(num_clients: int, seed: int):
    """
    randomly generate a graph of the network
    ------
    Parameters:
        num_clients: the number of clients
        seed: the random seed
    Returns:
        the adjacency matrix
    """
    np.random.seed(seed)
    mat = np.random.randn(num_clients, num_clients)
    # should be modified later
    mat[mat <= 0] = 0.
    mat[mat > 0] = 1.
    # get the upper triangle
    mat_tri = np.triu(mat)
    # substract the diagonal matrix
    mat_tri -= np.diag(np.diagonal(mat_tri))
    # add the transpose
    mat_tri = mat_tri+mat_tri.T+np.eye(num_clients)
    return mat_tri


def init_w(adj_mat: ndarray,):
    """
    init the weight matrix of every client, it should be $\mathcal{R}^{n \times n}$
    ------
    Parameters:
        adj_mat: the adjacency matrix of the network
    Returns:
        the weight matrix of $\mathcal{R}^{n \times n}$ shape.
    """
    P = np.zeros_like(adj_mat)
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if i != j and adj_mat[i][j] != 0:
                P[i][j] = 1./np.max(
                    [np.sum(adj_mat[i]),
                     np.sum(adj_mat[j]),]
                )
        # in the end update P_{ij} when i=j
        P[i][i] = 1.0 - np.sum(P[i])
    return P


def calculate_E(W: float, x: ndarray, h: ndarray, beta: float):
    """
    Calculate E_{ij} of the certain device.
    ------
    Parameters:
        W: the weight of the device i to j
        x: the l2 norm of the model components
        h: the channel gains of the device i to j
        beta: the estimation factor
    Returns:
        the estimated E_{ij}
    """
    inner_term = W * x / h
    inner_term = np.power(inner_term, 2)
    inner_term = np.sum(inner_term)
    return inner_term*beta
