# %%
import torch as t
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(20*20*32, 784),
            nn.ReLU(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x: t.Tensor):
        return self.net.forward(x)
# %%
