import numpy as np
from maddpg.alg.collate import pad_and_concat_obs, actions_to_onehot


def main():
    B = 2
    N = 3
    max_obs_dim = 10
    num_actions = 5

    # Fake obs batch with heterogeneous dims
    obs_batch = np.empty((B, N), dtype=object)
    obs_batch[0, 0] = np.ones(8)
    obs_batch[0, 1] = np.ones(10)
    obs_batch[0, 2] = np.ones(10)
    obs_batch[1, 0] = np.ones(8) * 2
    obs_batch[1, 1] = np.ones(10) * 3
    obs_batch[1, 2] = np.ones(10) * 4

    x = pad_and_concat_obs(obs_batch, max_obs_dim)
    print("x shape:", x.shape)  # (B, N*max_obs_dim)
    assert x.shape == (B, N * max_obs_dim)

    # Fake discrete actions
    act_batch = np.array(
        [
            [0, 2, 4],
            [1, 3, 0],
        ]
    )

    a_onehot = actions_to_onehot(act_batch, num_actions)
    print("a_onehot shape:", a_onehot.shape)  # (B, N*num_actions)
    assert a_onehot.shape == (B, N * num_actions)

    print("collate utilities test passed")


if __name__ == "__main__":
    main()
