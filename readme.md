# Decentralized Learning with Limited Subcarriers through Over-the-Air Computation

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
