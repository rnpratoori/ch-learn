"""Neural network model for learning the free energy derivative df/dc."""

import torch
import torch.nn as nn


class FEDerivative(nn.Module):
    """Neural network to approximate the free energy derivative df/dc."""
    
    def __init__(self, hidden_size=200):
        super(FEDerivative, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, c):
        output = self.mlp(c)
        # The CH equation only uses gradients of chemical potential, so adding
        # a constant to df/dc is unobservable.  Removing the batch mean fixes
        # that gauge freedom and stabilizes training.
        return output - torch.mean(output, dim=0, keepdim=True)
