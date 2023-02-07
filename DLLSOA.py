import torch.nn as nn
import torch.utils.data as data
from torch.optim import Adam
import numpy as np
import torch as t
from utils import (calculate_grad_norm,
                   get_weight_num,
                   calculate_weight_norm,
                   compute_power_coeff,
                   compute_alpha,
                   aggregation,
                   weight_init,
                   init_w,
                   calculate_E,
                   gen_topo)
import random
from tensorboardX import SummaryWriter
from copy import deepcopy
from dataset import load_mnist
from model import resnet18
import random
from numpy import ndarray


class LocalClient(object):
    def __init__(self,
                 id: int,
                 local_model: nn.Module,
                 train_loader: data.DataLoader,
                 test_loader: data.DataLoader,
                 loss_func: nn.Module,
                 comp_strategy: str,
                 lr: float,
                 ep_num: int,
                 sub_carrier_num: int,
                 device: t.device,
                 channel_gain: ndarray,
                 pow_limit: bool):
        # save parameters as member variables
        self.id = id
        self.model = local_model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.comp_strategy = comp_strategy
        self.lr = lr
        self.ep_num = ep_num
        self.device = device
        self.sub_carrier_num = sub_carrier_num
        self.pow_limit = pow_limit
        # init optimizer and loss function
        self.optimizer = Adam(self.model.parameters(), self.lr)
        self.loss_func = loss_func
        self.model.apply(weight_init)
        # init channel gain
        self.channel_gain = channel_gain

    def rcv_params(self, model_params: dict, xi_neighbors: ndarray, weight_neighbors: ndarray, amendment_strategy: str):
        """
        receive the parameters from the neighbors
        ------
        Parameters:
            model_params:dict, the sum of neighboring clients information through over-the-air aggregation (already added noise)
            xi_neighbors: the xi of the neighbors
            weight_neighbors: the weight of the neighbors
        Returns:
            None
        """
        # compute alpha
        if amendment_strategy == "eq5":
            weight_neighbors = None
        elif amendment_strategy == "eq6":
            pass
        else:
            raise ValueError("Unsupported value, only support eq5 and eq6")
        alpha = compute_alpha(self.id, xi_neighbors,
                              weight_neighbors, self.pow_limit)
        if model_params is not None:
            # begin aggregation
            model_params = {k: v*alpha for k, v in model_params.items()}
            # add its' own parameters
            model_params = {
                k: model_params[k]+self.model.state_dict()[k] for k in model_params.keys()}
            # finally load the state dictionary
            self.model.load_state_dict(model_params, strict=False)

    def send_params(self, component_keys: list[str], W: float, channel_gain: ndarray, beta: float):
        """
        get the parameters according to the channel conditions
        ------
        Parameters:
            component_keys: the key of parameters, i.e. the mask in the paper
            W: float, the W_{ij} in equation (3), W is the connectivity matrix
            channel_gain: the channel gain of the specified local device
            beta: the estimation factor of the E
        Returns:
            model_param:the specified parameters after power coefficient adjustments
            xi: the computed xi
        """
        # start the training procedure

        #
        with t.no_grad():
            # compute the power allocation coefficients b_ij^t(k) first
            weight_norm = calculate_weight_norm(self.model, component_keys)
            weight_norm_val = np.array(list(weight_norm.values()))
            # compute E
            E = calculate_E(W, weight_norm_val, channel_gain, beta)
            b, xi = compute_power_coeff(
                E, W, channel_gain, weight_norm_val, self.pow_limit)
            # get the parameters according to the mask (here we use key to read the parameters)
            all_keys = self.model.state_dict().keys()
            model_param = {}
            for key in all_keys:
                if key in component_keys:
                    model_param[key] = self.model.state_dict()[key]
            # multiply the weight by the power allocation coefficients and the channel gain
            for idx, key in enumerate(component_keys):
                model_param[key] = model_param[key]*b[idx]*channel_gain[idx]
            # finally, return the processed parameters
            return model_param, xi

    def train(self,):
        """
        Train the local model using the given train_loader for a round
        -------
        Parameters:
            None
        Returns:
            the loss of the trainset.
        """
        for ep in range(self.ep_num):
            ep_loss = []
            for (X, y) in self.train_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                y_hat = self.model.forward(X)
                loss = self.loss_func.forward(y_hat, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                ep_loss.append(loss.item())

        return np.mean(ep_loss)

    def test(self,):
        """
        start the test procedure
        -------
        Parameters:
            None
        Returns:
            the loss and accuracy of the testset.
        """
        sum_loss = []
        correct = 0
        # start the iteration
        with t.no_grad():
            for (X, y) in self.test_loader:
                y_hat = self.model.forward(X)
                loss = self.loss_func.forward(y_hat, y)
                sum_loss.append(loss.item())
                y_prediction = y_hat.data.max(1, keepdim=True)[1]
                correct += y_prediction.eq(y.data.view_as(y_prediction)
                                           ).long().cpu().sum()

        return np.mean(sum_loss), 100.*correct / len(self.test_loader.dataset)

    def generate_mask(self):
        """
        generates the mask according to the subcarrier number
        ------
        Returns:
            a list of the parameters that should be chosen
        """
        if self.comp_strategy == "random":
            param_keys = []
            for name, _ in self.model.named_parameters():
                if "weight" in name or "bias" in name:
                    param_keys.append(name)
            random.shuffle(param_keys)
            return param_keys[:self.sub_carrier_num]
        elif self.comp_strategy == "weight":
            weight_dict = calculate_weight_norm(self.model)
            weight_keys = [item[0]
                           for item in weight_dict][:self.sub_carrier_num]
            return weight_keys
        elif self.comp_strategy == "grad":
            grad_dict = calculate_grad_norm(self.model)
            grad_keys = [item[0] for item in grad_dict][:self.sub_carrier_num]
            return grad_keys


class DLLSOA(object):
    def __init__(self, args: dict):
        # save arguments to member variables
        if args["model_name"] == "resnet-18":
            net = resnet18.Resnet18()
        elif args["model_name"] == "CNN":
            net = None  # not implemented yet
        else:
            raise ValueError(
                "only support resnet-18 and CNN, pls check the model_name")
        self.device = t.device(
            "cuda") if args["cuda"] and t.cuda.is_available() else t.device("cpu")
        self.num_clients = args["num_clients"]
        self.sigma = args["sigma"]
        self.amendment_strategy = args["amendment_strategy"]
        self.train_epoch = args["train_epoch"]
        self.beta = args["beta"]
        self.pow_limit = args["pow_limit"]
        # split datasets
        dataloader_allusr, train_loader, test_loader = load_mnist(
            args["iid"], args["num_clients"], args["batch_size"],)
        # get the subcarrier num
        weight_num = get_weight_num(net)
        if args["sub_carrier_strategy"] == "no-limit":
            sub_carrier_nums = [weight_num for _ in range(self.num_clients)]
        elif args["sub_carrier_strategy"] == "restricted-1":
            sub_carrier_nums = [weight_num for _ in range(self.num_clients/3)]
            sub_carrier_nums += [int(weight_num*0.8)
                                 for _ in range(self.num_clients/3)]
            sub_carrier_nums += [int(weight_num*0.6)
                                 for _ in range(self.num_clients/3)]
            random.shuffle(sub_carrier_nums)
        elif args["sub_carrier_strategy"] == "restricted-2":
            sub_carrier_nums = [weight_num for _ in range(self.num_clients/4)]
            sub_carrier_nums += [int(weight_num*0.8)
                                 for _ in range(self.num_clients/4)]
            sub_carrier_nums += [int(weight_num*0.7)
                                 for _ in range(self.num_clients/4)]
            sub_carrier_nums += [int(weight_num*0.6)
                                 for _ in range(self.num_clients/4)]
            random.shuffle(sub_carrier_nums)
        else:
            raise ValueError(
                "only support no-limit, restricted-1, restricted-2")

        # create clients
        self.clients = [
            LocalClient(
                i,
                deepcopy(net),
                dataloader_allusr[i],
                test_loader,
                nn.CrossEntropyLoss(),
                args["comp_strategy"],
                args["lr"],
                args["ep_num"],
                sub_carrier_nums[i],
                self.device,
                np.random.rayleigh(
                    1., size=(self.num_clients,
                              sub_carrier_nums[i])),
                self.pow_limit
            )
            for i in range(self.num_clients)
        ]
        # init some variables such as connetivity matrix
        topo = gen_topo(self.num_clients, args["seed"])
        self.W = init_w(topo)
        self.xi = np.zeros((self.num_clients, self.num_clients))

    def train(self):
        """
        begin the training procedure
        ------
        Parameters:
            None
        Returns:
            None
        """
        # here simply use for loop instead of multiprocessing
        # todo: use multiprocessing
        for ep in range(self.train_epoch):
            for i in range(self.num_clients):
                # 1. generate mask of components
                mask = self.clients[i].generate_mask()
                channel_gains = self.clients[i].channel_gain
                rcv_models = []
                # 2. send the mask to all neighbors and receive parameters
                for j in range(self.num_clients):
                    if self.W[i][j] != 0. and i != j:
                        model, self.xi[i][j] = self.clients[j].send_params(
                            mask, self.W[i][j], channel_gains[j], self.beta)
                        rcv_models.append(model)
                # 3. begin aggregation
                processed_model = aggregation(rcv_models, self.sigma)
                # 4. perform gradient descent and update parameters
                self.clients[i].rcv_params(
                    processed_model, self.xi[i], self.W[i], self.amendment_strategy)
            print(f"ep:[{ep}/{self.train_epoch}]")

    def test(self):
        """
        begin the test procedure
        """
