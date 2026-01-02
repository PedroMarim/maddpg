import numpy as np

from maddpg_repro.envs.mpe_make import make_mpe_env
from maddpg_repro.alg.replay_buffer import ReplayBuffer


def sample_actions_discrete(env):
    # With env.discrete_action_input = True, action_space[i].sample() should work.
    return [env.action_space[i].sample() for i in range(env.n)]


def main():
    scenario = "simple_adversary"
    env = make_mpe_env(scenario_name=scenario, discrete_action_input=True)

    # Reset once to infer shapes
    obs_n = env.reset()
    n_agents = env.n

    buf = ReplayBuffer(
        capacity=50_000,
        num_agents=n_agents,
        obs_dim=None,
        act_dim=None,
        seed=0,
    )

    episodes = 5
    max_steps = 50

    for ep in range(episodes):
        obs_n = env.reset()
        ep_rew = np.zeros(n_agents, dtype=np.float32)

        for t in range(max_steps):
            act_n = sample_actions_discrete(env)

            next_obs_n, rew_n, done_n, info_n = env.step(act_n)

            buf.add(
                obs_n=obs_n,  # list of length N, each is np.array with its own dim
                act_n=act_n,  # keep original action objects for now
                rew_n=rew_n,  # list length N
                next_obs_n=next_obs_n,  # list length N
                done_n=done_n,
            )

            ep_rew += np.array(rew_n, dtype=np.float32)
            obs_n = next_obs_n

        print(f"Episode {ep}: total reward per agent = {ep_rew}")

    print(f"\nBuffer size: {len(buf)}")

    batch = buf.sample(8)
    print("Sampled batch shapes:")
    print("  obs      :", batch.obs.shape)
    print("  act      :", batch.act.shape)
    print("  rew      :", batch.rew.shape)
    print("  next_obs :", batch.next_obs.shape)
    print("  done     :", batch.done.shape)

    env.close()
    print("\n Random rollout collection OK")


if __name__ == "__main__":
    main()
