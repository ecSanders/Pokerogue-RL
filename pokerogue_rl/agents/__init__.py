import os
import random

from pokerogue_rl.agents.heuristic import HeuristicAgent


class RandomAgent:
    """Uniformly random legal action."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> int:
        return self.rng.choice([i for i, ok in enumerate(obs["legal"]) if ok])


def policy_agent():
    """Learned battle policy from the checkpoint at $POLICY_CHECKPOINT; heuristic for the rest."""
    import torch

    from pokerogue_rl.agents.policy_agent import PolicyAgent

    torch.set_num_threads(1)  # one agent per eval worker thread
    return PolicyAgent(os.environ["POLICY_CHECKPOINT"])


AGENTS = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
    "heuristic-lures": lambda: HeuristicAgent(lures="seek"),
    "policy": policy_agent,
}
