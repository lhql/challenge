"""Standalone GAP Actor; intentionally no YOPO network, anchors or score head."""

import torch
from torch import nn
from ..dynamics import rotation


class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 16, 5, 2, 2),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 48, 3, 2, 1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 5)),
            nn.Flatten(),
            nn.Linear(720, 128),
            nn.SiLU(),
        )

    def forward(self, image):
        return self.net(image)


def state_features(x, goal):
    # Position enters only through relative goal: translation equivariance.
    R = rotation(x[:, 6:10])
    Rt = R.transpose(-1, -2)
    v = (Rt @ x[:, 3:6, None]).squeeze(-1)
    g = (Rt @ (goal - x[:, :3]).unsqueeze(-1)).squeeze(-1)
    q = x[:, 6:10] / x[:, 6:10].norm(dim=-1, keepdim=True).clamp_min(1e-8)
    q = torch.where(q[:, :1] < 0, -q, q)
    return torch.cat((v / 6, q, x[:, 10:13] / 3, g / 10), -1)


class Actor(nn.Module):
    def __init__(self, sensor="depth"):
        super().__init__()
        self.sensor = sensor
        self.encoder = Encoder(2)
        self.state = nn.Sequential(
            nn.Linear(13, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU()
        )
        self.head = nn.Sequential(nn.Linear(192, 128), nn.SiLU(), nn.Linear(128, 16))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("scale", torch.tensor([4.0, 4.0, 4.0, 1.5]))

    def forward(self, image, x, goal):
        z = torch.cat((self.encoder(image), self.state(state_features(x, goal))), -1)
        body = torch.tanh(self.head(z).reshape(-1, 4, 4)) * self.scale
        acceleration = torch.einsum("bij,bkj->bki", rotation(x[:, 6:10]), body[..., :3])
        return torch.cat((acceleration, body[..., 3:]), -1)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(2)
        self.net = nn.Sequential(nn.Linear(141, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, image, x, goal):
        return self.net(
            torch.cat((self.encoder(image), state_features(x, goal)), -1)
        ).squeeze(-1)
