import os
import time
import json
import argparse
from pathlib import Path

import numpy as np
import torch

from maddpg_repro.envs.mpe_make import make_mpe_env
from maddpg_repro.alg.replay_buffer import ReplayBuffer
from maddpg_repro.alg.maddpg import MADDPG, MADDPGConfig


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_checkpoint(path: Path, algo: MADDPG, step: int, cfg: dict):
    payload = {
        "step": step,
        "cfg": cfg,
        "obs_dims": algo.obs_dims,
        "max_obs_dim": algo.max_obs_dim,
        "state": {
            "actors": [a.state_dict() for a in algo.actors],
            "critics": [c.state_dict() for c in algo.critics],
            "actors_targ": [a.state_dict() for a in algo.actors_targ],
            "critics_targ": [c.state_dict() for c in algo.critics_targ],
        },
    }
    torch.save(payload, path)


def sample_action_int_from_actor(
    actor, obs_i: np.ndarray, explore: bool, eps: float
) -> int:
    """
    Discrete integer action for env.
    - explore=True: epsilon-greedy on top of logits sampling
    """
    obs_t = torch.from_numpy(obs_i.astype(np.float32))
    with torch.no_grad():
        logits = actor(obs_t.unsqueeze(0)).squeeze(0)  # (A,)
        if explore and np.random.rand() < eps:
            # random action
            a = np.random.randint(0, logits.shape[-1])
            return int(a)

        probs = torch.softmax(logits, dim=-1)
        if explore:
            a = torch.multinomial(probs, num_samples=1).item()
        else:
            a = torch.argmax(probs).item()
    return int(a)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="simple_adversary")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--episode_len", type=int, default=25)
    parser.add_argument("--buffer_size", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--warmup", type=int, default=5_000
    )  # steps before updates
    parser.add_argument(
        "--update_every", type=int, default=1
    )  # update every env step
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=2_000)
    parser.add_argument("--save_every", type=int, default=20_000)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")  # "cpu" or "cuda"
    parser.add_argument("--eps_start", type=float, default=0.3)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay_steps", type=int, default=100_000)
    args = parser.parse_args()

    set_seed(args.seed)

    # Create run folder
    run_name = (
        args.run_name or f"{args.scenario}_seed{args.seed}_{int(time.time())}"
    )
    run_dir = Path("runs") / run_name
    ensure_dir(run_dir)
    ensure_dir(run_dir / "checkpoints")

    # Save config
    cfg_dict = vars(args)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2)

    # Env
    env = make_mpe_env(scenario_name=args.scenario, discrete_action_input=True)
    obs_n = env.reset()
    obs_dims = [len(o) for o in obs_n]
    max_obs_dim = max(obs_dims)
    N = env.n

    # Buffer: flexible mode (works for heterogeneous obs dims)
    buf = ReplayBuffer(
        capacity=args.buffer_size,
        num_agents=N,
        obs_dim=None,
        act_dim=None,
        seed=args.seed,
    )

    # MADDPG
    algo_cfg = MADDPGConfig(
        num_actions=5,
        gamma=0.95,
        tau=0.01,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gumbel_tau=1.0,
    )
    algo = MADDPG(
        obs_dims=obs_dims,
        max_obs_dim=max_obs_dim,
        cfg=algo_cfg,
        device=args.device,
    )

    # Training loop state
    step = 0
    episode = 0
    ep_rew = np.zeros(N, dtype=np.float32)
    obs_n = env.reset()

    # Simple logging to a jsonl file
    log_path = run_dir / "train_log.jsonl"

    start_time = time.time()

    while step < args.steps:
        # epsilon schedule
        frac = min(1.0, step / max(1, args.eps_decay_steps))
        eps = args.eps_start + frac * (args.eps_end - args.eps_start)

        # Select actions (int per agent)
        act_n = []
        for i in range(N):
            a_i = sample_action_int_from_actor(
                algo.actors[i], obs_n[i], explore=True, eps=eps
            )
            act_n.append(
                np.array([a_i], dtype=np.int64)
            )  # keep MPE MultiDiscrete style

        next_obs_n, rew_n, done_n, _ = env.step(act_n)

        # Store transition (keep objects, flexible)
        buf.add(obs_n, act_n, rew_n, next_obs_n, done_n)

        ep_rew += np.array(rew_n, dtype=np.float32)
        obs_n = next_obs_n
        step += 1

        # Episode boundary (fixed-length episodes typical in MPE papers)
        if step % args.episode_len == 0:
            # log episode reward
            record = {
                "step": step,
                "episode": episode,
                "eps": float(eps),
                "ep_rew": ep_rew.tolist(),
                "ep_rew_mean": float(np.mean(ep_rew)),
                "buffer_size": len(buf),
                "elapsed_sec": float(time.time() - start_time),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            # reset
            episode += 1
            ep_rew = np.zeros(N, dtype=np.float32)
            obs_n = env.reset()

        # Updates
        if len(buf) >= args.warmup and (step % args.update_every == 0):
            for _ in range(args.updates_per_step):
                batch = buf.sample(args.batch_size)
                stats = algo.update(batch)

            # occasional print
            if step % args.log_every == 0:
                print(
                    f"step={step} eps={eps:.3f} "
                    f"actor_loss={stats['actor_loss']:.4f} critic_loss={stats['critic_loss']:.4f} "
                    f"buf={len(buf)}"
                )

        # Save checkpoints
        if step % args.save_every == 0:
            ckpt_path = run_dir / "checkpoints" / f"ckpt_step{step}.pt"
            save_checkpoint(
                ckpt_path,
                algo,
                step,
                cfg={"train": cfg_dict, "algo": algo_cfg.__dict__},
            )
            print(f"Saved checkpoint: {ckpt_path}")

    env.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
