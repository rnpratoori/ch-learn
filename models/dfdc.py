"""Neural network model for learning the free energy derivative df/dc."""

import torch
import torch.nn as nn


class FEDerivative(nn.Module):
    """Neural network to approximate the free energy derivative df/dc."""
    
    def __init__(self, hidden_size=50):
        super(FEDerivative, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, c):
        output = self.mlp(c)
        # Enforce zero mean on the output
        return output - torch.mean(output, dim=0, keepdim=True)
