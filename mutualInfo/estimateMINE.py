
import torch
import torch.nn as nn
from torch.nn import functional as F
from mutualInfo.mututalInfo import Mine



class EstimateMINE(nn.Module):
    def __init__(self, input_dim, output_dim, activation='tanh', device='cpu'):
        super().__init__()
        self.device = device

        self.fc1 = nn.Linear(input_dim, 12)
        self.fc2 = nn.Linear(12, 10)
        self.fc3 = nn.Linear(10, 7)
        self.fc4 = nn.Linear(7, 5)
        self.fc5 = nn.Linear(5, 4)
        self.fc6 = nn.Linear(4, 3)
        self.fc7 = nn.Linear(3, output_dim)

        self.activation = activation
        self.softmax = nn.Softmax()

    def non_linear(self, x):
        if self.activation == 'tanh':
            return torch.tanh(x)
        elif self.activation == 'relu':
            return F.relu(x)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.get_layer_outputs(x)[-1]

    def get_layer_outputs(self, x):
        x1 = self.non_linear(self.fc1(x))
        x2 = self.non_linear(self.fc2(x1))
        x3 = self.non_linear(self.fc3(x2))
        x4 = self.non_linear(self.fc4(x3))
        x5 = self.non_linear(self.fc5(x4))
        x6 = self.non_linear(self.fc6(x5))
        out = self.fc7(x6)
        return [x1, x2, x3, x4, x5, x6, out]

    def estimate_layerwise_mutual_information(self, x, target, iters):
        n, input_dim = target.shape
        layer_outputs = self.get_layer_outputs(x)
        layer_outputs[-1] = F.softmax(layer_outputs[-1])
        to_return = dict()
        for layer_id, layer_output in enumerate(layer_outputs):

            _, layer_dim = layer_output.shape

            statistics_network = nn.Sequential(
                nn.Linear(input_dim + layer_dim, 400),
                nn.ReLU(),
                nn.Linear(400, 400),
                nn.ReLU(),
                nn.Linear(400, 1)
            )

            mi_estimator = Mine(T=statistics_network).to(self.device)
            
            mi = mi_estimator.optimize(
                target, layer_output.detach(), iters=iters, batch_size=n // 1, opt=None)

            to_return[layer_id] = mi.item()
        return to_return

    def calculate_information_plane(self, x, y, iters=100):
        info_x_t = self.estimate_layerwise_mutual_information(x, x, iters)
        info_y_t = self.estimate_layerwise_mutual_information(x, y, iters)

        return info_x_t, info_y_t
    
    def register_hooks(self, nn_util, layer_names, model):
        def hook_fn(module, inputs, outputs):
            activations = outputs[0] 
            mi = self.forward(activations)
            nn_util.data["estimations"][module_name].append(mi.item())

        for module_name, module in model.named_modules():
            if type(module).__name__ in layer_names:
                handle = module.register_forward_hook(hook_fn)
                if(module_name in nn_util.handles):
                    nn_util.handles[module_name].append(handle)