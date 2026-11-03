"""Learned policy from a checkpoint; the heuristic plays whatever the policy wasn't trained on.

Battle decisions always come from the policy. Run decisions (rewards, item targets, moves, releases)
do too when the checkpoint was trained with --run-policy, or when `run_policy=True` is passed.
"""

from __future__ import annotations

import torch

from pokerogue_rl import features as F
from pokerogue_rl.agents.heuristic import HeuristicAgent
from pokerogue_rl.policy import AgentPolicy, encode_step, to_tensors
from pokerogue_rl.run_features import RUN_KINDS, TmMemory


class PolicyAgent:
    def __init__(self, checkpoint: str, greedy: bool = True, run_policy: bool | None = None):
        self.policy = AgentPolicy()
        ckpt = self.policy.load_checkpoint(checkpoint)
        self.policy.eval()
        if run_policy is None:
            run_policy = bool(ckpt.get("args", {}).get("run_policy", False))
        self.kinds = F.BATTLE_KINDS + (RUN_KINDS if run_policy else ())
        self.greedy = greedy
        self.helper = HeuristicAgent()
        self.tm_memory = TmMemory()

    def reset(self) -> None:
        self.helper.reset()
        self.tm_memory.reset()

    def act(self, obs: dict) -> int:
        if obs["kind"] not in self.kinds:
            return self.helper.act(obs)
        self.helper._update_carry(obs["party"])
        obs = F.restrict_boss_balls(self.tm_memory.filter(obs))
        with torch.no_grad():
            logits, _ = self.policy(to_tensors(F.stack([encode_step(obs)]), torch.device("cpu")))
        if self.greedy:
            action = int(logits.argmax(-1))
        else:
            action = int(torch.distributions.Categorical(logits=logits).sample())
        self.tm_memory.note(obs, action)
        return action
