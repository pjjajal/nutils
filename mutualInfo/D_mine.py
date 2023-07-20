import torch
import torch.nn as nn

class EstimateMINE(nn.Module):
    def __init__(self, input_dim, output_dim, activation='tanh', device='cpu'):
        super().__init__()
        self.device = device
        self.activation = activation

        self.T = nn.Sequential(
            nn.Linear(input_dim + output_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1)
        )

    def non_linear(self, x):
        if self.activation == 'tanh':
            return torch.tanh(x)
        elif self.activation == 'relu':
            return torch.relu(x)
        else:
            raise NotImplementedError

    def forward(self, x, z):
        # Concatenate x and z along the second dimension
        concatenated_input = torch.cat((x, z), dim=1)
        t = self.T(concatenated_input)
        return t.mean()

   

    def register_hooks(self, nn_util, layer_names, mine_module):
        def hook_fn(module, inputs, outputs):
            x, z = outputs[0], outputs[1]  # Unpack x and z from outputs
            mi = mine_module(x, z)  # Use mine_module to compute mutual information
            nn_util.data["estimations"][module_name].append(mi.item())

        for module_name, module in nn_util.model.named_modules():
            if type(module).__name__ in layer_names:
                handle = module.register_forward_hook(hook_fn)
                if(module_name in nn_util.handles):
                    nn_util.handles[module_name].append(handle)
                else:
                    nn_util.handles[module_name] = handle


