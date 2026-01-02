import numpy as np

from maddpg_repro.envs.mpe_make import make_mpe_env
from maddpg_repro.alg.replay_buffer import ReplayBuffer
from maddpg_repro.alg.maddpg import MADDPG, MADDPGConfig


def main():
    scenario = "simple_adversary"
    env = make_mpe_env(scenario_name=scenario, discrete_action_input=True)

    # Infer per-agent obs dims
    obs_n = env.reset()
    obs_dims = [len(o) for o in obs_n]
    max_obs_dim = max(obs_dims)
    N = env.n

    print("obs_dims:", obs_dims, "max_obs_dim:", max_obs_dim, "N:", N)

    # Flexible buffer (handles heterogeneous obs dims)
    buf = ReplayBuffer(
        capacity=10_000, num_agents=N, obs_dim=None, act_dim=None, seed=0
    )

    # Collect a bit of random data
    steps = 300
    obs_n = env.reset()
    for _ in range(steps):
        act_n = [env.action_space[i].sample() for i in range(env.n)]
        next_obs_n, rew_n, done_n, _ = env.step(act_n)
        buf.add(obs_n, act_n, rew_n, next_obs_n, done_n)
        obs_n = next_obs_n

    print("buffer size:", len(buf))

    # Sample a batch
    batch = buf.sample(64)

    # Initialize MADDPG
    cfg = MADDPGConfig(
        num_actions=5,
        gamma=0.95,
        tau=0.01,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gumbel_tau=1.0,
    )
    algo = MADDPG(
        obs_dims=obs_dims, max_obs_dim=max_obs_dim, cfg=cfg, device="cpu"
    )

    # One update step
    stats = algo.update(batch)
    print("update stats:", stats)

    env.close()
    print("update step test done")


if __name__ == "__main__":
    main()
