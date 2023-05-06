from dataclasses import dataclass
import torch
import torch.nn as nn
import time
from nutils import NUtil


class TestNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lin1 = nn.Linear(10, 10)
        self.lin2 = nn.Linear(10, 10)
        self.lin3 = nn.Linear(10, 10)
        self.backbone = nn.Sequential(nn.Linear(10, 10), nn.Tanh())
        self.tanh = nn.Tanh()

    def forward(self, x):
        y = self.lin1(x)
        y = self.lin2(y)
        y = self.lin3(y)
        y = self.backbone(y)
        return self.tanh(y), True


def output_parser(x: torch.Tensor, boolean):
    return x.tolist(), boolean


if __name__ == "__main__":
    nn_util = NUtil()
    model = TestNet()
    nn_util.time(model, "TestNet")
    nn_util.capture_activation(model, "TestNet", output_parser)
    x = torch.randn((1, 10))
    model(x)
    # print(MODEL_METRICS)
    model(x)
    print(nn_util)
