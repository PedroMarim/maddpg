from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiscreteActor(nn.Module):
    """
    Actor for discrete actions:
    - input: obs (B, obs_dim_i)
    - output: logits (B, num_actions)
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dims: list[int] = [128, 128],
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.mlp = MLP(obs_dim, hidden_dims, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # logits
        return self.mlp(obs)


class CentralizedCritic(nn.Module):
    """
    Centralized critic for agent i:
    input: concat([x, a_all]) where
      x: (B, N*max_obs_dim)
      a_all: (B, N*num_actions) (one-hot or soft one-hot)
    output: Q-value (B, 1)
    """

    def __init__(
        self, x_dim: int, a_dim: int, hidden_dims: list[int] = [256, 256]
    ):
        super().__init__()
        self.in_dim = x_dim + a_dim
        self.q = MLP(self.in_dim, hidden_dims, out_dim=1)

    def forward(self, x: torch.Tensor, a_all: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, a_all], dim=-1)
        return self.q(z)


def gumbel_softmax_sample(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = False,
) -> torch.Tensor:
    """
    Differentiable discrete sampling.
    Returns a 'soft one-hot' (or hard one-hot if hard=True with straight-through estimator).
    Shape: same as logits (B, num_actions)
    """
    return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)


@torch.no_grad()
def select_discrete_action(
    actor: DiscreteActor,
    obs_1d: torch.Tensor,
    explore: bool = True,
) -> int:
    """
    Choose an integer action for environment interaction (single observation).
    obs_1d: (obs_dim,)
    """
    logits = actor(obs_1d.unsqueeze(0)).squeeze(0)  # (A,)
    if explore:
        # sample from categorical distribution
        probs = torch.softmax(logits, dim=-1)
        a = torch.multinomial(probs, num_samples=1).item()
    else:
        a = torch.argmax(logits).item()
    return int(a)
