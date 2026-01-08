import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Set a clean scientific style
sns.set_theme(style="whitegrid")


def load_data(log_file):
    data = []
    # Read line-by-line to handle potential interruptions gracefully
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return pd.DataFrame(data)


def plot_curves(run_dir, window=100):
    run_path = Path(run_dir)
    log_file = run_path / "train_log.jsonl"

    if not log_file.exists():
        print(f"Error: Could not find {log_file}")
        return

    print(f"Loading data from {log_file}...")
    df = load_data(log_file)

    if df.empty:
        print("Log file is empty.")
        return

    # 1. Expand 'ep_rew' list into separate columns
    # In simple_adversary, ep_rew is [adv_reward, good_reward, good_reward]
    rewards_expanded = pd.DataFrame(df["ep_rew"].tolist(), index=df.index)
    num_agents = rewards_expanded.shape[1]

    # 2. Rename columns specifically for 'simple_adversary'
    # Agent 0 = Adversary, Agents 1+ = Good Agents
    col_names = []
    for i in range(num_agents):
        if i == 0:
            col_names.append("Adversary")
        else:
            col_names.append(f"Good_Agent_{i}")

    rewards_expanded.columns = col_names

    # Combine with original dataframe
    df = pd.concat([df, rewards_expanded], axis=1)

    # 3. Calculate Smoothed Averages (Rolling Mean)
    for col in col_names:
        df[f"{col}_smooth"] = (
            df[col].rolling(window=window, min_periods=1).mean()
        )

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))

    # Plot Adversary (Red)
    sns.lineplot(
        x=df["step"],
        y=df["Adversary_smooth"],
        label="Adversary (Agent 0)",
        color="#d62728",  # Standard Matplotlib Red
        linewidth=2.5,
    )

    # Plot Good Agents (Green)
    for i in range(1, num_agents):
        label = f"Good Agent {i}"
        # Use dashed/dotted lines to distinguish the good agents if needed
        linestyle = "--" if i == 1 else ":"

        sns.lineplot(
            x=df["step"],
            y=df[f"Good_Agent_{i}_smooth"],
            label=label,
            color="#2ca02c",  # Standard Matplotlib Green
            linestyle=linestyle,
            linewidth=2,
        )

    plt.title("MADDPG: Simple Adversary Reward Evolution", fontsize=16)
    plt.ylabel("Average Reward (Smoothed)", fontsize=12)
    plt.xlabel("Training Steps", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot inside the run folder
    save_path = run_path / "reward_curve_adversary.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot MADDPG training curves for simple_adversary"
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to the run folder (e.g. runs/simple_adversary_seed0...)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Smoothing window size (default: 100)",
    )
    args = parser.parse_args()

    plot_curves(args.run_dir, args.window)
