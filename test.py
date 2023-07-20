from dataclasses import dataclass
import torch
import torch.nn as nn
import time
from nutils import NUtil
from mutualInfo.estimateMINE import EstimateMINE
# from mutualInfo.D_mine import EstimateMINE
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

def register_hook(module, inputs, outputs):
    print("Hook activated!")
    print(f"Inputs: {inputs}")
    print(f"Outputs: {outputs}")
    return inputs,outputs
    

def main():
    nn_util = NUtil()
    model = TestNet()
    nn_util.time(model, "TestNet")
    nn_util.capture_activation(model, "TestNet", output_parser)
    hook_handle = model.lin1.register_forward_hook(register_hook)
    print(hook_handle, end = "\n\n\n")
    x = torch.Tensor(1, 10).normal_()
    y = torch.Tensor(1, 10).normal_()
    # print(type(x),  "\n\n")
    input_dim = 10
    output_dim = 10

    mine_model = EstimateMINE(input_dim=input_dim, output_dim=output_dim, activation='tanh', device='cpu')
    # Estimate the layerwise mutual information using NUtil and EstimateMINE
    mi_estimations = nn_util.estimate_layerwise_mutual_information(mine_model, x, y, iters=100)

# Print the estimated mutual information for each layer
    for layer_id, mi in mi_estimations.items():
        print(f"Layer {layer_id}: {mi}")
    # print(nn_util)
    # for i in tqdm(range(5)):
    #     x = torch.randn((1, 10))
    #     model(x)
        # nn_util.chunker_check(1024**2)
    # print(nn_util)
    hook_handle.remove()


if __name__ == "__main__":
    main()

# def main():
#     nn_util = NUtil()
#     model = TestNet()
#     nn_util.time(model, "TestNet")
#     nn_util.capture_activation(model, "TestNet", output_parser)
#     x = torch.Tensor(1, 10).normal_()
#     y = torch.Tensor(1, 10).normal_()
#     print(type(x), "\n\n")
#     input_dim = 10
#     output_dim = 10

#     mine_model = EstimateMINE(input_dim=input_dim, output_dim=output_dim, activation='tanh', device='cpu')

#     # Register hooks for each specified class name with captured activations
#     class_names = ['Linear', 'Sequential']  # Add more class names as needed
#     mine_model.register_hooks(nn_util, class_names, model)  # Pass the TestNet model

#     # Trigger the hooks by passing x and y to the model
#     model(x)
#     model(y)

#     # Estimate the layerwise mutual information using NUtil and EstimateMINE
#     mi_estimations = nn_util.estimate_layerwise_mutual_information(mine_model, x, y, iters=100)

#     # Print the estimated mutual information for each layer
#     for layer_id, mi in mi_estimations.items():
#         print(f"Layer {layer_id}: {mi}")

#     # Print NUtil's summary
#     print(nn_util)

#     # Trigger hooks using different inputs
#     for i in tqdm(range(5)):
#         x = torch.randn((1, 10))
#         model(x)

#     # Print updated NUtil's summary
#     print(nn_util)

# if __name__ == "__main__":
#     main()

