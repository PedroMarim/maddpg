import os
import sys
from pathlib import Path


def _add_mpe_to_path():
    # repo_root = .../maddpg
    repo_root = Path(__file__).resolve().parents[3]
    mpe_root = repo_root / "third_party" / "multiagent_particle_envs"
    sys.path.insert(0, str(mpe_root))


def make_mpe_env(scenario_name: str, discrete_action_input: bool = True):
    _add_mpe_to_path()

    # Optional: suppress the interactive prompt from the env repo
    os.environ.setdefault("SUPPRESS_MA_PROMPT", "1")

    from multiagent.environment import MultiAgentEnv
    from multiagent.scenarios import load

    scenario = load(f"{scenario_name}.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
    )

    env.discrete_action_input = discrete_action_input
    return env
