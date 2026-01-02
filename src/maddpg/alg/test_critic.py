import numpy as np
import torch

from maddpg.alg.collate import pad_and_concat_obs, actions_to_onehot
from maddpg.alg.networks import CentralizedCritic


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # simple_adversary: N=3, obs dims are [8,10,10] -> max_obs_dim=10
    N = 3
    max_obs_dim = 10
    num_actions = 5

    B = 4  # batch size

    # Build a fake batch like ReplayBuffer gives us in flexible mode:
    obs_batch = np.empty((B, N), dtype=object)
    next_obs_batch = np.empty((B, N), dtype=object)

    for b in range(B):
        obs_batch[b, 0] = np.random.randn(8).astype(np.float32)
        obs_batch[b, 1] = np.random.randn(10).astype(np.float32)
        obs_batch[b, 2] = np.random.randn(10).astype(np.float32)

        next_obs_batch[b, 0] = np.random.randn(8).astype(np.float32)
        next_obs_batch[b, 1] = np.random.randn(10).astype(np.float32)
        next_obs_batch[b, 2] = np.random.randn(10).astype(np.float32)

    # Integer actions (B, N)
    act_int = np.random.randint(0, num_actions, size=(B, N))

    # Collate -> fixed shapes
    x = pad_and_concat_obs(
        obs_batch, max_obs_dim=max_obs_dim
    )  # (B, N*max_obs_dim)
    a_onehot = actions_to_onehot(
        act_int, num_actions=num_actions
    )  # (B, N*num_actions)

    # Torch tensors
    x_t = torch.from_numpy(x)
    a_t = torch.from_numpy(a_onehot)

    x_dim = N * max_obs_dim
    a_dim = N * num_actions

    critic = CentralizedCritic(x_dim=x_dim, a_dim=a_dim)

    q = critic(x_t, a_t)  # (B, 1)
    print("x shape:", x_t.shape)
    print("a shape:", a_t.shape)
    print("q shape:", q.shape)
    print("q sample:", q[:2].squeeze(-1).tolist())

    assert q.shape == (B, 1)
    print("critic test ok")


if __name__ == "__main__":
    main()
