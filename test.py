from dataclasses import dataclass
import torch
import torch.nn as nn
import time
from nutils import NUtil
import sys
import objsize
from tqdm import tqdm


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


def output_parser(module, inputs, outputs):
    return outputs[0].tolist(), outputs[1]

def main():
    nn_util = NUtil()
    model = TestNet()
    nn_util.time(model, "TestNet")
    nn_util.capture_activation(model, "TestNet", output_parser)
    print(nn_util)
    for i in tqdm(range(5)):
        x = torch.randn((1, 10))
        model(x)
        # nn_util.chunker_check(1024**2)
    print(nn_util)


if __name__ == "__main__":
    main()