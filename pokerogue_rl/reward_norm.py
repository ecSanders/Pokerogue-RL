"""Reward normalization for PPO (train_ppo.py --reward-norm).

Rewards are divided by the running standard deviation of each env's discounted return, so the critic's
targets keep a roughly constant scale however large the raw rewards are or grow as the policy improves
(raw-EXP rewards grow with the wave). Not mean-centred: shifting rewards would change what the agent
prefers, while scaling them only changes step sizes. The same scheme as Gym's NormalizeReward and the
"reward scaling" used by common PPO implementations.

Only the training rewards are scaled; episode returns in the log stay in the reward's own units.
"""

from __future__ import annotations

import numpy as np


class RewardNormalizer:
    def __init__(self, num_envs: int, gamma: float, eps: float = 1e-8):
        self.gamma = gamma
        self.eps = eps
        self.ret = np.zeros(num_envs, np.float64)  # each env's running discounted return
        self.count = 0.0
        self.mean = 0.0
        self.var = 1.0

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var + self.eps))

    def _update(self, x: np.ndarray) -> None:
        """Merge a batch into the running mean/variance (parallel Welford)."""
        n, mean, var = x.size, float(x.mean()), float(x.var())
        if n == 0:
            return
        delta, total = mean - self.mean, self.count + n
        self.var = (self.var * self.count + var * n + delta ** 2 * self.count * n / total) / total
        self.mean += delta * n / total
        self.count = total

    def __call__(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Scale a rollout's [T, N] rewards; `dones[t]` ends env n's episode after step t."""
        returns = np.empty_like(rewards, dtype=np.float64)
        for t in range(rewards.shape[0]):
            self.ret = self.ret * self.gamma + rewards[t]
            returns[t] = self.ret
            self.ret = np.where(dones[t] > 0, 0.0, self.ret)
        self._update(returns.ravel())
        return (rewards / self.std).astype(rewards.dtype)

    def state_dict(self) -> dict:
        return {"count": self.count, "mean": self.mean, "var": self.var}

    def load_state_dict(self, state: dict) -> None:
        self.count, self.mean, self.var = state["count"], state["mean"], state["var"]
