import torch
import torch.nn as nn


class ODE_Pendulum(nn.Module):
    def __init__(self, c, l_pendulum, use_damping=True):
        super().__init__()

        self.use_damping = use_damping
        if use_damping:
            self.c = nn.Parameter(c, requires_grad=True)
        self.l_pendulum = nn.Parameter(torch.tensor(10.0), requires_grad=False)
        self.g = nn.Parameter(torch.tensor(2.28340227358), requires_grad=False)

    def forward(self, t, x):
        dx = torch.zeros_like(x)
        dx[0] = x[1]
        dx[1] = - torch.exp(self.g)/self.l_pendulum*torch.sin(x[0])
        if self.use_damping:
            dx[1] = dx[1] - self.c*x[1]
        return dx


class ODE_2ObjectsSpring(nn.Module):
    def __init__(self, k, eq_distance):
        super().__init__()

        self.k = nn.Parameter(k, requires_grad=True)
        self.eq_distance = nn.Parameter(eq_distance, requires_grad=True)

    def forward(self, t, x):
        dx = torch.zeros_like(x)
        p1 = x[:2]
        p2 = x[2:4]
        dx[:4] = x[4:]

        F = self.k*(p1 - p2) - self.eq_distance*(p1 - p2) / torch.norm(p1 - p2)

        # From other code
        norm = torch.norm(p1 - p2)
        direction = (p1 - p2)/norm
        F = self.k*(norm - 2*self.eq_distance)*direction

        dx[4:6] = -F
        dx[6:] = F
        return dx


class ODE_SlidingBlock(nn.Module):
    def __init__(self, mu=torch.tensor(0.), alpha=torch.tensor(0.)):
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(torch.pi / 13), requires_grad=False)
        self.mu = nn.Parameter(torch.tensor(0.2), requires_grad=False)
        self.g = nn.Parameter(torch.tensor(2.28340227358), requires_grad=False)

    def forward(self, t, x):
        dx = torch.zeros_like(x)
        dx[0] = x[1]
        dx[1] = torch.exp(self.g) * (torch.sin(self.alpha) - self.mu * torch.cos(self.alpha))
        return dx


class ODE_ThrownObject(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(2.28340227358), requires_grad=False)

    def forward(self, t, x):
        dx = torch.zeros_like(x)
        dx[:2] = x[2:]
        dx[3] = torch.exp(self.g)
        return dx


class ODE_BouncingBallDrop_CVDL(nn.Module):

    def __init__(self, elasticity=None):
        super().__init__()
        if elasticity is None:
            elasticity = torch.tensor(1.0)
        self.elasticity = nn.Parameter(torch.tensor(0.9), requires_grad=False)
        self.g = nn.Parameter(torch.tensor(2.28340227358), requires_grad=False)
        self.r = nn.Parameter(torch.tensor(3.0), requires_grad=False)

    def forward(self, t, x):
        dx = torch.zeros_like(x)
        dx[0] = x[2]
        dx[1] = x[3]
        dx[3] = torch.exp(self.g)
        if x[1] > 32.0 - self.r:
            # this is change in velocity, so need to first zero the velocity (-x[1])
            # and then apply the elasticity
            dx[3] = - self.elasticity * x[3] - x[3]
        return dx