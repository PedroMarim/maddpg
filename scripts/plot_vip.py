import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
# Name of your experiment folder (make sure this matches exactly)
EXPERIMENT_NAME = "vip_escort_final"


def load_log(run_dir: Path):
    log_path = run_dir / "train_log.jsonl"
    steps, rewards = [], []

    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        return [], []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            steps.append(data["step"])
            rewards.append(data["ep_rew_mean"])
    return steps, rewards


def smooth(y, box_pts=100):
    """Smooths the curve to make the trend visible."""
    if len(y) < box_pts:
        return y
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def main():
    # Locate the repo root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    run_dir = repo_root / "runs" / EXPERIMENT_NAME

    print(f"Loading data from: {run_dir}")
    steps, rewards = load_log(run_dir)

    if not steps:
        return

    # Plotting
    plt.figure(figsize=(10, 6), dpi=100)

    # Plot faint raw data
    plt.plot(steps, rewards, color="blue", alpha=0.15, label="Raw Data")
    # Plot smoothed trend
    plt.plot(
        steps,
        smooth(rewards),
        color="blue",
        linewidth=2,
        label="Smoothed Trend",
    )

    plt.title(f"Learning Curve: {EXPERIMENT_NAME}", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = repo_root / f"{EXPERIMENT_NAME}_plot.png"
    plt.savefig(save_path)
    print(f"\nPlot saved to: {save_path}")


if __name__ == "__main__":
    main()
