import argparse
import sys
import imageio
from pathlib import Path
import numpy as np
import torch

from maddpg_repro.envs.mpe_make import make_mpe_env
from maddpg_repro.alg.maddpg import MADDPG, MADDPGConfig


def load_checkpoint(path: Path, device: str):
    ckpt = torch.load(path, map_location=device)
    return ckpt


def select_action_deterministic(actor, obs_i: np.ndarray) -> int:
    obs_t = torch.from_numpy(obs_i.astype(np.float32))
    # FIX: Ensure tensor is on the same device as the actor
    device = next(actor.parameters()).device
    obs_t = obs_t.to(device)

    with torch.no_grad():
        logits = actor(obs_t.unsqueeze(0)).squeeze(0)
        return int(torch.argmax(logits).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--scenario", type=str, default="simple_adversary")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode_len", type=int, default=25)
    parser.add_argument("--device", type=str, default="cpu")
    # NEW ARGUMENTS
    parser.add_argument(
        "--save_gif",
        action="store_true",
        help="Save a gif of the first episode",
    )
    parser.add_argument("--gif_path", type=str, default="evaluation.gif")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    # Load checkpoint
    ckpt = load_checkpoint(ckpt_path, args.device)

    # Rebuild MADDPG
    algo_cfg = MADDPGConfig(**ckpt["cfg"]["algo"])
    algo = MADDPG(
        obs_dims=ckpt["obs_dims"],
        max_obs_dim=ckpt["max_obs_dim"],
        cfg=algo_cfg,
        device=args.device,
    )

    for i, a in enumerate(algo.actors):
        a.load_state_dict(ckpt["state"]["actors"][i])
    for i, c in enumerate(algo.critics):
        c.load_state_dict(ckpt["state"]["critics"][i])

    for a in algo.actors:
        a.eval()
    for c in algo.critics:
        c.eval()

    # Env
    env = make_mpe_env(scenario_name=args.scenario, discrete_action_input=True)
    N = env.n

    print(f"\nEvaluating checkpoint: {ckpt_path.name}")
    print(f"Scenario: {args.scenario}")
    print(f"Episodes: {args.episodes}\n")

    all_ep_rewards = []
    frames = []

    for ep in range(args.episodes):
        obs_n = env.reset()
        ep_rew = np.zeros(N, dtype=np.float32)

        for step_i in range(args.episode_len):
            # --- RENDER FRAME ---
            # We only record the first episode to save time/space
            if args.save_gif and ep == 0:
                # 'rgb_array' gets the pixel data
                frame = env.render(mode="rgb_array")
                # MPE sometimes returns a list of frames [frame], so we handle both
                if isinstance(frame, list):
                    frames.append(frame[0])
                else:
                    frames.append(frame)

            act_n = []
            for i in range(N):
                a_i = select_action_deterministic(algo.actors[i], obs_n[i])
                act_n.append(np.array([a_i], dtype=np.int64))

            obs_n, rew_n, done_n, _ = env.step(act_n)
            ep_rew += np.array(rew_n, dtype=np.float32)

        all_ep_rewards.append(ep_rew)
        print(f"Episode {ep}: rewards per agent = {ep_rew}")

    all_ep_rewards = np.stack(all_ep_rewards, axis=0)

    print("\n=== Evaluation summary ===")
    print("Mean reward per agent:", all_ep_rewards.mean(axis=0))
    print("Std  reward per agent:", all_ep_rewards.std(axis=0))
    print("Mean over agents:", all_ep_rewards.mean())

    env.close()

    # --- SAVE GIF ---
    if args.save_gif and len(frames) > 0:
        # Save at 10 FPS (frames per second) so you can see the movement clearly
        imageio.mimsave(args.gif_path, frames, fps=10)
        print(f"\nGIF saved successfully to: {args.gif_path}")
    elif args.save_gif:
        print("\nWarning: No frames were captured.")


if __name__ == "__main__":
    main()
