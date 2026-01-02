from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np


@dataclass
class Batch:
    obs: np.ndarray  # (B, N, obs_dim) if obs_dim fixed; else object array
    act: np.ndarray  # (B, N, act_dim) or object array
    rew: np.ndarray  # (B, N)
    next_obs: np.ndarray  # (B, N, obs_dim) or object array
    done: np.ndarray  # (B,) or (B, N)


class ReplayBuffer:
    """
    Joint replay buffer for multi-agent RL.
    Stores one transition containing all agents' data.
    """

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_dim: int | None = None,
        act_dim: int | None = None,
        store_done_per_agent: bool = False,
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.store_done_per_agent = store_done_per_agent

        self.rng = np.random.default_rng(seed)

        # If obs_dim/act_dim are known and fixed, we store dense arrays (fast).
        # Otherwise we store object arrays (more flexible, slower but safe).
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.dense = (obs_dim is not None) and (act_dim is not None)

        self._ptr = 0
        self._size = 0

        if self.dense:
            self.obs = np.zeros(
                (capacity, num_agents, obs_dim), dtype=np.float32
            )
            self.next_obs = np.zeros(
                (capacity, num_agents, obs_dim), dtype=np.float32
            )
            self.act = np.zeros(
                (capacity, num_agents, act_dim), dtype=np.float32
            )
        else:
            # Each entry holds a list/array per agent (variable dims supported)
            self.obs = np.empty((capacity, num_agents), dtype=object)
            self.next_obs = np.empty((capacity, num_agents), dtype=object)
            self.act = np.empty((capacity, num_agents), dtype=object)

        self.rew = np.zeros((capacity, num_agents), dtype=np.float32)

        if store_done_per_agent:
            self.done = np.zeros((capacity, num_agents), dtype=np.float32)
        else:
            self.done = np.zeros((capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs_n: List[np.ndarray] | np.ndarray,
        act_n: List[Any] | np.ndarray,
        rew_n: List[float] | np.ndarray,
        next_obs_n: List[np.ndarray] | np.ndarray,
        done_n: List[bool] | bool | np.ndarray,
    ) -> None:
        """
        Store one joint transition.
        - obs_n, next_obs_n: list of N arrays (obs_dim,)
        - act_n: list of N (could be int, array, etc.)
        - rew_n: list/array length N
        - done_n: list length N (from MPE) or single bool
        """
        i = self._ptr

        if self.dense:
            self.obs[i] = np.asarray(obs_n, dtype=np.float32)
            self.next_obs[i] = np.asarray(next_obs_n, dtype=np.float32)
            self.act[i] = np.asarray(act_n, dtype=np.float32)
        else:
            # Store per-agent objects
            for a in range(self.num_agents):
                self.obs[i, a] = np.asarray(obs_n[a], dtype=np.float32)
                self.next_obs[i, a] = np.asarray(
                    next_obs_n[a], dtype=np.float32
                )
                self.act[i, a] = act_n[a]  # keep original type for now

        self.rew[i] = np.asarray(rew_n, dtype=np.float32)

        if self.store_done_per_agent:
            if isinstance(done_n, (list, np.ndarray)):
                self.done[i] = np.asarray(done_n, dtype=np.float32)
            else:
                self.done[i] = float(done_n)  # broadcast if needed
        else:
            # Store a single done: "any agent done"
            if isinstance(done_n, (list, np.ndarray)):
                self.done[i] = float(np.any(done_n))
            else:
                self.done[i] = float(done_n)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        b = int(batch_size)
        idx = self.rng.integers(0, self._size, size=b)

        if self.dense:
            return Batch(
                obs=self.obs[idx],
                act=self.act[idx],
                rew=self.rew[idx],
                next_obs=self.next_obs[idx],
                done=self.done[idx],
            )
        else:
            # For object arrays, return object arrays; later we’ll collate/convert per env/agent.
            return Batch(
                obs=self.obs[idx].copy(),
                act=self.act[idx].copy(),
                rew=self.rew[idx].copy(),
                next_obs=self.next_obs[idx].copy(),
                done=self.done[idx].copy(),
            )
