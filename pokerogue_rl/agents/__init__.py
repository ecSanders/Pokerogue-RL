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


AGENTS = {"random": RandomAgent, "heuristic": HeuristicAgent}
