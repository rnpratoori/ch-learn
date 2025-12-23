"""Neural network model for learning the free energy f(c)."""


import torch.nn as nn


class FEnergy(nn.Module):
    """Neural network to approximate the free energy f(c)."""
    
    def __init__(self, hidden_size=50):
        super(FEnergy, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, c):
        return self.mlp(c)
