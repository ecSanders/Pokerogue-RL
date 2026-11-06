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
    """Learned policy from the checkpoint at $POLICY_CHECKPOINT; heuristic for the rest.

    $POLICY_RUN=1/0 forces whether the policy also plays run decisions (default: as trained).
    $POLICY_GREEDY=0 samples actions as in training instead of taking the most likely one (default 1)."""
    import torch

    from pokerogue_rl.agents.policy_agent import PolicyAgent

    torch.set_num_threads(1)  # one agent per eval worker thread
    run = os.environ.get("POLICY_RUN")
    return PolicyAgent(os.environ["POLICY_CHECKPOINT"], greedy=os.environ.get("POLICY_GREEDY", "1") != "0",
                       run_policy=None if run is None else run == "1")


AGENTS = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
    "heuristic-lures": lambda: HeuristicAgent(lures="seek"),
    "policy": policy_agent,
}
