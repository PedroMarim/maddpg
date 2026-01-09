import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
MAX_STEPS = 300000  # Clip both runs to this point for a fair comparison


def load_log(run_dir: Path):
    """Loads step and reward data from a train_log.jsonl file."""
    log_path = run_dir / "train_log.jsonl"
    steps, rewards = [], []

    if not log_path.exists():
        print(f"Warning: No log found at {log_path}")
        return steps, rewards

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            step = data["step"]

            # --- CLIP DATA ---
            # If this step is beyond our limit, stop loading (or skip it)
            if step > MAX_STEPS:
                break

            steps.append(step)
            # ep_rew_mean is the average return (sum of rewards) per agent
            rewards.append(data["ep_rew_mean"])
    return steps, rewards


def smooth(y, box_pts=50):
    """Smooths the data for cleaner plotting."""
    if len(y) < box_pts:
        return y
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def main():
    # 1. Locate the Repository Root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent

    # 2. Define Run Paths (Relative to Repo Root)
    # Ensure these folder names match your actual 'runs' folder
    success_dir = repo_root / "runs" / "spread_3_agents_overnight"
    fail_dir = repo_root / "runs" / "spread_6_agents_overnight"

    print(f"Loading baseline from: {success_dir}")
    print(f"Loading stress test from: {fail_dir}")

    # 3. Load Data
    steps_3, rew_3 = load_log(success_dir)
    steps_6, rew_6 = load_log(fail_dir)

    if not steps_3:
        print(
            f"Error: Could not load data from {success_dir}. Check folder name."
        )
        return

    # 4. Plot
    plt.figure(figsize=(10, 6), dpi=100)

    # Plot 3 Agents (Success)
    plt.plot(
        steps_3,
        smooth(rew_3),
        label="3 Agents (Baseline)",
        color="green",
        linewidth=2,
    )

    # Plot 6 Agents (Fail)
    if steps_6:
        plt.plot(
            steps_6,
            smooth(rew_6),
            label="6 Agents (Stress Test)",
            color="red",
            linewidth=2,
            linestyle="--",
        )
    else:
        print("Warning: Could not load 6-agent data. Plotting only baseline.")

    plt.title("Scalability Analysis: MADDPG Performance", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Average Reward per Agent", fontsize=12)

    # Force the x-axis to look clean (optional, but nice)
    plt.xlim(0, MAX_STEPS)

    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot
    save_path = repo_root / "scalability_comparison.png"
    plt.savefig(save_path)
    print(f"\nSuccess! Plot saved to: {save_path}")
    # plt.show()


if __name__ == "__main__":
    main()
