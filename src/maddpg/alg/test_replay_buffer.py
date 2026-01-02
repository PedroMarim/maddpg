import numpy as np
from maddpg.alg.replay_buffer import ReplayBuffer


def main():
    N = 3
    obs_dim = 18
    act_dim = 1  # for discrete actions we can store as 1-dim (e.g., [2])

    buf = ReplayBuffer(
        capacity=100, num_agents=N, obs_dim=obs_dim, act_dim=act_dim, seed=0
    )

    for _ in range(10):
        obs = np.random.randn(N, obs_dim).astype(np.float32)
        act = np.random.randint(0, 5, size=(N, 1)).astype(np.float32)
        rew = np.random.randn(N).astype(np.float32)
        next_obs = np.random.randn(N, obs_dim).astype(np.float32)
        done = False
        buf.add(obs, act, rew, next_obs, done)

    batch = buf.sample(4)
    print("obs", batch.obs.shape)
    print("act", batch.act.shape)
    print("rew", batch.rew.shape)
    print("next_obs", batch.next_obs.shape)
    print("done", batch.done.shape)

    print("Replay buffer test ok")


if __name__ == "__main__":
    main()
