import torch.nn as nn

class ClassicalHead(nn.Module):
    def __init__(self, in_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)
