from maddpg.envs.mpe_make import make_mpe_env


def main():
    env = make_mpe_env("simple_spread", discrete_action_input=True)

    env.discrete_action_input = True
    obs_n = env.reset()

    print("Num agents:", env.n)
    print("Obs dims:", [len(o) for o in obs_n])

    for t in range(10):
        # Sample random actions for each agent
        act_n = [env.action_space[i].sample() for i in range(env.n)]
        obs_n, rew_n, done_n, info_n = env.step(act_n)
        print(f"t={t:02d} rewards={rew_n}")

    env.close()
    print("Smoke test done!")


if __name__ == "__main__":
    main()
