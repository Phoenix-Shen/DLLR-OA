# Communication Resources Limited Decentralized Learning with Privacy Guarantee through Over-the-Air Computation
- Here is the coded implementation of our paper `Communication Resources Limited Decentralized Learning with Privacy Guarantee through Over-the-Air Computation`, the original name is `DLLSOA` so the method name appeared in this repo is `DLLSOA`.
- Paper Link: https://dl.acm.org/doi/abs/10.1145/3565287.3610268.

## 1. Code Structure

- configs: stores the configuration of the experiment
- model: resnet18 neural network model
- shells: shell for executing the training process
- dataset.py: the methods of splitting the dataset
- DLLSOA.py: the main algorithm
- main.py: the entry of the whole program
- utils.py: some helper functions

## 2. How to run the experiment

It is very easy:

```shell
chmod +x shells/clean_logs.sh
chmod +x shells/run_cifar10.sh
chmod +x shells/run_mnist.sh

shells/run_cifar10.sh
```

For specific configuration, you can visit the `configs\dllsoa_template.yaml` to get related information and execute the command:

```shell
python main.py configs/dllsoa_template.yaml
```

All results will be stored in the `logs` directory, which will be reviewed by the tensorboard application.

## 3. Environment

We provide `requirements.txt` for you to install the same package. And your python version should not be below 3.9, or you will encounter some typing errors.

## 4. Citation
If you find this paper is helpful, please cite:
```bibtex
@inproceedings{10.1145/3565287.3610268,
author = {Qiao, Jing and Shen, Shikun and Chen, Shuzhen and Zhang, Xiao and Lan, Tian and Cheng, Xiuzhen and Yu, Dongxiao},
title = {Communication Resources Limited Decentralized Learning with Privacy Guarantee through Over-the-Air Computation},
year = {2023},
isbn = {9781450399265},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3565287.3610268},
doi = {10.1145/3565287.3610268},
abstract = {In this paper, we propose a novel decentralized learning algorithm, namely DLLR-OA, for resource-constrained over-the-air computation with formal privacy guarantee. Theoretically, we characterize how the limited resources induced model-components selection error and compound communication errors jointly impact decentralized learning, making the iterates of DLLR-OA converge to a contraction region centered around a scaled version of the errors. In particular, the convergence rate of the DLLR-OA algorithm in the error-free case [EQUATION] achieves the state-of-the-arts. Besides, we formulate a power control problem and decouple it into two sub-problems of transmitter and receiver to accelerate the convergence of the DLLR-OA algorithm. Furthermore, we provide quantitative privacy guarantee for the proposed over-the-air computation approach. Interestingly, we show that network noise can indeed enhance privacy of aggregated updates while over-the-air computation can further protect individual updates. Finally, the extensive experiments demonstrate that DLLR-OA performs well in the communication resources constrained setting. In particular, numerical results on CIFAR-10 dataset shows nearly 30\% communication cost reduction over state-of-the-art baselines with comparable learning accuracy even in resource constrained settings.},
booktitle = {Proceedings of the Twenty-Fourth International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing},
pages = {201–210},
numpages = {10},
keywords = {decentralized learning, over-the-air computation, resource allocation, privacy-preserving},
location = {Washington, DC, USA},
series = {MobiHoc '23}
}
```
