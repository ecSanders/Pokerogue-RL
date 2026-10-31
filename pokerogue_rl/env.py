"""Battle-only RL environment over the simulator.

The policy makes battle decisions (command, target, faint_switch). Everything else (rewards,
item targets, move learning, releases) is played by the heuristic agent in between.

Reward per policy step:
  + WAVE_REWARD per wave cleared                       (the objective)
  + TEAM_COEF * (gamma * team_health' - team_health)   (potential-based shaping: wear-down)
  + DAMAGE_COEF * enemy HP fraction removed this wave   (early learning signal)
team_health = sum of party HP fractions / 6, so fainted members count 0; 0 at game over.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from pokerogue_rl.agents.heuristic import HeuristicAgent
from pokerogue_rl.features import BATTLE_KINDS
from pokerogue_rl.sim_client import SimClient


@dataclass
class RewardConfig:
    gamma: float = 0.995
    wave_reward: float = 1.0
    team_coef: float = 1.0
    damage_coef: float = 0.2


def team_health(obs: dict) -> float:
    return sum(p["hp"] / p["maxHp"] for p in obs["party"] if p["maxHp"]) / 6


def enemy_health(obs: dict) -> float:
    return sum(e["hp"] / e["maxHp"] for e in obs["enemies"] if e["maxHp"])


class BattleEnv:
    def __init__(self, env_id: int, seed_prefix: str = "train", max_wave: int = 500,
                 reward: RewardConfig | None = None, sim_kwargs: dict | None = None):
        self.sim = SimClient.launch(request_timeout=120, **(sim_kwargs or {}))
        self.helper = HeuristicAgent()
        self.reward_cfg = reward or RewardConfig()
        self.max_wave = max_wave
        self.seeds = (f"{seed_prefix}-{env_id}-{i}" for i in itertools.count())
        self.obs: dict | None = None
        self.episode_return = 0.0
        self.episode_steps = 0

    def _play_until_battle(self, resp: dict) -> dict:
        """Let the heuristic answer non-battle decisions until the policy has a battle decision."""
        while not resp["done"] and resp["obs"]["kind"] not in BATTLE_KINDS:
            resp = self.sim.step(self.helper.act(resp["obs"]))
        return resp

    def reset(self) -> dict:
        self.helper.reset()
        resp = self._play_until_battle(self.sim.reset(seed=next(self.seeds), max_wave=self.max_wave))
        if resp["done"]:  # ended before any battle decision; try another seed
            return self.reset()
        self.obs = resp["obs"]
        self.episode_return = 0.0
        self.episode_steps = 0
        return self.obs

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        prev = self.obs
        self.helper._update_carry(prev["party"])  # the heuristic's item choices follow its carry
        resp = self._play_until_battle(self.sim.step(int(action)))
        obs, done = resp["obs"], resp["done"]

        cfg = self.reward_cfg
        reward = cfg.wave_reward * (obs["wave"] - prev["wave"])
        next_health = 0.0 if done and resp["info"]["reason"] == "game_over" else team_health(obs)
        reward += cfg.team_coef * (cfg.gamma * next_health - team_health(prev))
        if obs["wave"] == prev["wave"]:
            reward += cfg.damage_coef * max(0.0, enemy_health(prev) - enemy_health(obs))

        self.episode_return += reward
        self.episode_steps += 1
        info = {"invalid": resp["info"].get("invalidAction", False)}
        if done:
            info["episode"] = {"wave": obs["wave"], "reason": resp["info"]["reason"],
                               "return": self.episode_return, "steps": self.episode_steps}
            obs = self.reset()
        else:
            self.obs = obs
        return obs, reward, done, info

    def close(self) -> None:
        self.sim.close()


class VecBattleEnv:
    """N BattleEnvs stepped in parallel threads (each blocks on its own simulator process)."""

    def __init__(self, num_envs: int, **kwargs):
        with ThreadPoolExecutor(num_envs) as pool:
            self.envs = list(pool.map(lambda i: BattleEnv(i, **kwargs), range(num_envs)))
        self.pool = ThreadPoolExecutor(num_envs)

    def reset(self) -> list[dict]:
        return list(self.pool.map(lambda e: e.reset(), self.envs))

    def step(self, actions) -> tuple[list[dict], np.ndarray, np.ndarray, list[dict]]:
        results = list(self.pool.map(lambda ea: ea[0].step(ea[1]), zip(self.envs, actions)))
        obs, rewards, dones, infos = zip(*results)
        return list(obs), np.array(rewards, np.float32), np.array(dones, bool), list(infos)

    def close(self) -> None:
        list(self.pool.map(lambda e: e.close(), self.envs))
        self.pool.shutdown()
