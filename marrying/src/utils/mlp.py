import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers=5, **kwargs):
        super(MLP, self).__init__()
        self.model = nn.Sequential()

        for _ in range(num_layers):
            self.model.append(nn.Linear(input_dim, hidden_dim))
            self.model.append(nn.LeakyReLU())
            input_dim = hidden_dim
        # append output layer
        self.model.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, inputs: torch.Tensor):
        # states [bs, ts, state_dim]
        return self.model(inputs)  # out: (bs, param_dim)
