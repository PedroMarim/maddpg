import numpy as np
from typing import List, Sequence


def pad_and_concat_obs(
    obs_batch: np.ndarray,
    max_obs_dim: int,
) -> np.ndarray:
    """
    Convert a batch of variable-length per-agent observations into
    a fixed-size concatenated representation.

    Parameters
    ----------
    obs_batch : np.ndarray
        Shape (B, N), dtype=object.
        obs_batch[b, i] is a 1D np.array of length obs_dim_i.

    max_obs_dim : int
        Maximum observation dimension across all agents.

    Returns
    -------
    x : np.ndarray
        Shape (B, N * max_obs_dim), dtype=float32.
    """
    B, N = obs_batch.shape
    x = np.zeros((B, N * max_obs_dim), dtype=np.float32)

    for b in range(B):
        for i in range(N):
            o = obs_batch[b, i]
            dim = len(o)
            start = i * max_obs_dim
            x[b, start : start + dim] = o

    return x


def actions_to_onehot(
    act_batch: np.ndarray,
    num_actions: int,
) -> np.ndarray:
    """
    Convert integer actions to one-hot vectors.

    Parameters
    ----------
    act_batch : np.ndarray
        Shape (B, N) or (B, N, 1), integers in [0, num_actions-1].

    num_actions : int
        Number of discrete actions (A).

    Returns
    -------
    a_onehot : np.ndarray
        Shape (B, N * num_actions), dtype=float32.
    """
    if act_batch.ndim == 3:
        act_batch = act_batch.squeeze(-1)

    B, N = act_batch.shape
    a = np.zeros((B, N, num_actions), dtype=np.float32)

    for b in range(B):
        for i in range(N):
            a[b, i, int(act_batch[b, i])] = 1.0

    return a.reshape(B, N * num_actions)
