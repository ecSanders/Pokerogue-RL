"""Battle policy from a checkpoint for battle decisions; the heuristic for everything else."""

from __future__ import annotations

import torch

from pokerogue_rl import features as F
from pokerogue_rl.agents.heuristic import HeuristicAgent
from pokerogue_rl.policy import BattlePolicy, to_tensors


class PolicyAgent:
    def __init__(self, checkpoint: str, greedy: bool = True):
        self.policy = BattlePolicy()
        self.policy.load_state_dict(torch.load(checkpoint, map_location="cpu")["policy"])
        self.policy.eval()
        self.greedy = greedy
        self.helper = HeuristicAgent()

    def reset(self) -> None:
        self.helper.reset()

    def act(self, obs: dict) -> int:
        if obs["kind"] not in F.BATTLE_KINDS:
            return self.helper.act(obs)
        self.helper._update_carry(obs["party"])
        with torch.no_grad():
            logits, _ = self.policy(to_tensors(F.stack([F.encode(obs)]), torch.device("cpu")))
        if self.greedy:
            return int(logits.argmax(-1))
        return int(torch.distributions.Categorical(logits=logits).sample())
