# MADDPG Reproduction (PyTorch)

This repository contains a PyTorch reproduction of the **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** algorithm, based on the paper *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments* by Lowe et al.

The code is designed to be modular and easy to run, with a focus on the **Simple Adversary** environment from the Multi-Agent Particle Environments (MPE).

## 📂 Project Structure

```text
├── src/
│   └── maddpg_repro/       # Core algorithm and environment wrappers
├── scripts/
│   └── plot_results.py     # Script to generate reward evolution curves
├── runs/                   # Checkpoints and logs (auto-generated)
├── train.py                # Main training entry point
└── requirements.txt        # Dependencies
```

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Usage

### 1. Training
To start training the agents in the `simple_adversary` scenario:

```bash
# Make sure 'src' is in your python path
export PYTHONPATH=src

# Run on CPU (default)
python src/maddpg_repro/train.py --scenario simple_adversary

# Run on GPU (Recommended)
python src/maddpg_repro/train.py --scenario simple_adversary --device cuda --steps 1000000
```

**Key Arguments:**
* `--scenario`: The MPE environment name (default: `simple_adversary`).
* `--device`: `cpu` or `cuda`.
* `--run_name`: Custom name for the output folder in `runs/`.

### 2. Visualization (Plotting Rewards)
After (or during) training, you can visualize the learning curve. This script specifically separates the **Adversary** from the **Good Agents** to show the competitive dynamic.

```bash
python scripts/plot_results.py runs/YOUR_RUN_FOLDER_NAME
```
*Output:* This will save a `reward_curve_adversary.png` inside the run folder.

## ☁️ Running on Google Colab
This repository is optimized for Colab usage to leverage free GPUs.

1. Clone the repo in a Colab cell.
2. Mount Google Drive to save checkpoints automatically (so you don't lose data if Colab disconnects).
3. Run the training command pointing to the Drive folder.

*(See the full Colab guide in the repository documentation/issues).*

## 📊 Results: Simple Adversary

In the `simple_adversary` environment:
* **Adversary (Red):** Wants to reach the target landmark.
* **Good Agents (Green):** Want to cover the target landmark while deceiving the adversary.

**Expected Behavior:**
As training progresses, you should observe a competitive tug-of-war where the Good Agents learn to split up to confuse the Adversary, and the Adversary learns to identify the real target based on the agents' movements.

![Reward Curve](path/to/your/image.png)
*(Add your generated plot here after your first run!)*

## 📚 References
* [MADDPG Paper](https://arxiv.org/abs/1706.02275)
* [Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs)