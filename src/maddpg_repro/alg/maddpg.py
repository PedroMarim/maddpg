from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from maddpg_repro.alg.networks import DiscreteActor, CentralizedCritic
from maddpg_repro.alg.collate import pad_and_concat_obs, actions_to_onehot


@torch.no_grad()
def soft_update_(
    online: torch.nn.Module, target: torch.nn.Module, tau: float
) -> None:
    for p, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(1.0 - tau)
        p_t.data.add_(tau * p.data)


def _normalize_act_batch(act_obj: np.ndarray, num_agents: int) -> np.ndarray:
    """
    act_obj: (B, N) dtype=object from ReplayBuffer flexible mode.
    Each entry can be int, np.int64, or np.array([int]).
    Returns: int array (B, N)
    """
    B, N = act_obj.shape
    assert N == num_agents
    out = np.zeros((B, N), dtype=np.int64)
    for b in range(B):
        for i in range(N):
            a = act_obj[b, i]
            if isinstance(a, np.ndarray):
                # Often shape (1,) like array([2])
                out[b, i] = int(a.squeeze())
            else:
                out[b, i] = int(a)
    return out


def _obs_obj_to_tensors(
    obs_obj: np.ndarray, obs_dims: List[int], device: torch.device
) -> List[torch.Tensor]:
    """
    obs_obj: (B, N) dtype=object. obs_obj[b,i] is np.array(obs_dim_i,)
    Returns list of length N, each tensor shape (B, obs_dim_i)
    Assumes obs_dim_i is consistent for each fixed agent index i.
    """
    B, N = obs_obj.shape
    assert N == len(obs_dims)

    obs_tensors = []
    for i in range(N):
        arr = np.stack([obs_obj[b, i] for b in range(B)], axis=0).astype(
            np.float32
        )  # (B, obs_dim_i)
        assert arr.shape[1] == obs_dims[i], (arr.shape, obs_dims[i])
        obs_tensors.append(torch.from_numpy(arr).to(device))
    return obs_tensors


@dataclass
class MADDPGConfig:
    num_actions: int = 5
    gamma: float = 0.95
    tau: float = 0.01
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    gumbel_tau: float = 1.0
    hidden_actor: Tuple[int, int] = (128, 128)
    hidden_critic: Tuple[int, int] = (256, 256)


class MADDPG:
    """
    MADDPG for discrete actions (via Gumbel-Softmax for actor gradients).
    - One actor per agent (obs_dim may differ per agent)
    - One centralized critic per agent (shared input format)
    """

    def __init__(
        self,
        obs_dims: List[int],
        max_obs_dim: int,
        cfg: MADDPGConfig,
        device: str | torch.device = "cpu",
    ):
        self.obs_dims = list(obs_dims)
        self.N = len(obs_dims)
        self.max_obs_dim = int(max_obs_dim)
        self.cfg = cfg
        self.device = torch.device(device)

        # Dimensions for critic inputs
        self.x_dim = self.N * self.max_obs_dim
        self.a_dim = self.N * cfg.num_actions

        # Actors + target actors
        self.actors = torch.nn.ModuleList(
            [
                DiscreteActor(
                    obs_dim=d,
                    num_actions=cfg.num_actions,
                    hidden_dims=list(cfg.hidden_actor),
                ).to(self.device)
                for d in self.obs_dims
            ]
        )
        self.actors_targ = torch.nn.ModuleList(
            [
                DiscreteActor(
                    obs_dim=d,
                    num_actions=cfg.num_actions,
                    hidden_dims=list(cfg.hidden_actor),
                ).to(self.device)
                for d in self.obs_dims
            ]
        )

        # Critics + target critics (one per agent)
        self.critics = torch.nn.ModuleList(
            [
                CentralizedCritic(
                    x_dim=self.x_dim,
                    a_dim=self.a_dim,
                    hidden_dims=list(cfg.hidden_critic),
                ).to(self.device)
                for _ in range(self.N)
            ]
        )
        self.critics_targ = torch.nn.ModuleList(
            [
                CentralizedCritic(
                    x_dim=self.x_dim,
                    a_dim=self.a_dim,
                    hidden_dims=list(cfg.hidden_critic),
                ).to(self.device)
                for _ in range(self.N)
            ]
        )

        # Copy initial weights to targets
        self._hard_update_targets()

        # Optimizers
        self.actor_opts = [
            torch.optim.Adam(self.actors[i].parameters(), lr=cfg.actor_lr)
            for i in range(self.N)
        ]
        self.critic_opts = [
            torch.optim.Adam(self.critics[i].parameters(), lr=cfg.critic_lr)
            for i in range(self.N)
        ]

    @torch.no_grad()
    def _hard_update_targets(self):
        for i in range(self.N):
            self.actors_targ[i].load_state_dict(self.actors[i].state_dict())
            self.critics_targ[i].load_state_dict(self.critics[i].state_dict())

    def update(self, batch) -> dict:
        """
        batch is ReplayBuffer.Batch from flexible buffer:
          obs: (B,N) object
          act: (B,N) object
          rew: (B,N) float
          next_obs: (B,N) object
          done: (B,) float
        """
        cfg = self.cfg
        B = batch.rew.shape[0]

        # --- Build fixed critic inputs ---
        x_np = pad_and_concat_obs(
            batch.obs, max_obs_dim=self.max_obs_dim
        )  # (B, x_dim)
        x_next_np = pad_and_concat_obs(
            batch.next_obs, max_obs_dim=self.max_obs_dim
        )  # (B, x_dim)

        act_int_np = _normalize_act_batch(batch.act, self.N)  # (B, N)
        a_onehot_np = actions_to_onehot(
            act_int_np, num_actions=cfg.num_actions
        )  # (B, a_dim)

        x = torch.from_numpy(x_np).to(self.device)
        x_next = torch.from_numpy(x_next_np).to(self.device)
        a_onehot = torch.from_numpy(a_onehot_np).to(self.device)

        rew = torch.from_numpy(batch.rew.astype(np.float32)).to(
            self.device
        )  # (B, N)
        done = torch.from_numpy(batch.done.astype(np.float32)).to(
            self.device
        )  # (B,)

        # --- Prepare per-agent obs tensors for actors ---
        obs_list = _obs_obj_to_tensors(batch.obs, self.obs_dims, self.device)
        next_obs_list = _obs_obj_to_tensors(
            batch.next_obs, self.obs_dims, self.device
        )

        # =========================
        # 1) Critic updates
        # =========================
        critic_losses = []

        with torch.no_grad():
            # Next actions from target actors -> hard one-hot (no gradients)
            a_next_parts = []
            for j in range(self.N):
                logits_next = self.actors_targ[j](next_obs_list[j])  # (B, A)
                a_next_j = F.gumbel_softmax(
                    logits_next, tau=cfg.gumbel_tau, hard=True, dim=-1
                )  # (B, A)
                a_next_parts.append(a_next_j)
            a_next_all = torch.cat(a_next_parts, dim=-1)  # (B, N*A)

        for i in range(self.N):
            q = self.critics[i](x, a_onehot).squeeze(-1)  # (B,)

            with torch.no_grad():
                q_next = self.critics_targ[i](x_next, a_next_all).squeeze(
                    -1
                )  # (B,)
                y = rew[:, i] + cfg.gamma * (1.0 - done) * q_next

            loss_q = F.mse_loss(q, y)
            self.critic_opts[i].zero_grad()
            loss_q.backward()
            self.critic_opts[i].step()

            critic_losses.append(loss_q.item())

        # =========================
        # 2) Actor updates
        # =========================
        actor_losses = []

        # Freeze critics during actor updates (saves memory + prevents accidental grads)
        for c in self.critics:
            for p in c.parameters():
                p.requires_grad_(False)

        for i in range(self.N):
            # Build joint action vector where only agent i is differentiable
            a_parts = []
            for j in range(self.N):
                if j == i:
                    logits = self.actors[j](obs_list[j])  # (B, A)
                    a_j = F.gumbel_softmax(
                        logits, tau=cfg.gumbel_tau, hard=False, dim=-1
                    )  # soft for gradients
                else:
                    # Other agents: use replay actions as fixed one-hot
                    a_j = a_onehot[
                        :, j * cfg.num_actions : (j + 1) * cfg.num_actions
                    ].detach()
                a_parts.append(a_j)

            a_all = torch.cat(a_parts, dim=-1)  # (B, N*A)

            # Maximize Q_i -> minimize -Q_i
            q_i = self.critics[i](x, a_all).squeeze(-1)
            loss_pi = -q_i.mean()

            self.actor_opts[i].zero_grad()
            loss_pi.backward()
            self.actor_opts[i].step()

            actor_losses.append(loss_pi.item())

        # Unfreeze critics
        for c in self.critics:
            for p in c.parameters():
                p.requires_grad_(True)

        # =========================
        # 3) Target soft updates
        # =========================
        for i in range(self.N):
            soft_update_(self.actors[i], self.actors_targ[i], tau=cfg.tau)
            soft_update_(self.critics[i], self.critics_targ[i], tau=cfg.tau)

        return {
            "critic_loss": float(np.mean(critic_losses)),
            "actor_loss": float(np.mean(actor_losses)),
            "critic_losses": critic_losses,
            "actor_losses": actor_losses,
        }
