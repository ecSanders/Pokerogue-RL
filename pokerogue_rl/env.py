"""RL environment over the simulator.

The policy makes the decisions in `policy_kinds`: by default the battle decisions (command, target,
faint_switch); add RUN_KINDS for rewards, item targets, move learning and releases. The heuristic
agent plays every other decision in between.

Reward per policy step:
  + WAVE_REWARD per wave cleared                       (the objective)
  + TEAM_COEF * (gamma * team_health' - team_health)   (potential-based shaping: wear-down)
  + DAMAGE_COEF * enemy HP fraction removed this wave   (early learning signal)
  + LEVEL_COEF * levels gained by kept Pokemon         (shaping: EXP now pays off at boss waves later)
  + CARRY_COEF * change in kept carry level            (shaping: one carry takes the EXP and items)
  + POKERUS_COEF * (pokerus' - pokerus)               (shaping: keep a carrier until all have it)
team_health = sum of party HP fractions / 6, so fainted members count 0; 0 at game over.
Level shaping rewards EXP: without it, catching (no EXP) clears wild waves as well as a KO does, and
the under-levelled team then dies to the wave-50 boss, beyond the discount horizon. Both level terms
count only Pokemon on the team before and after the step (matched by uid), so a catch or a release
never pays: summing party levels paid +0.1 * level for every catch, and the policy learned to catch
nearly every wild wave and churn its bench.
Level shaping is undiscounted: the gamma * potential form costs (1 - gamma) * levels every step,
which grows with the team and would reward ending waves in fewer turns, i.e. catching.
carry: highest level among those Pokemon (releasing the carry costs its level); pokerus = party members with Pokerus (1.5x EXP, spreads).
"""

from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from pokerogue_rl.agents.heuristic import HeuristicAgent
from pokerogue_rl.features import BATTLE_KINDS
from pokerogue_rl.run_features import TmMemory
from pokerogue_rl.sim_client import SimClient, SimError, launch_sims
from pokerogue_rl.strategy import StrategyTracker, carry_level, pokerus_count


@dataclass
class RewardConfig:
    gamma: float = 0.995
    wave_reward: float = 1.0
    team_coef: float = 1.0
    damage_coef: float = 0.2
    level_coef: float = 0.0
    carry_coef: float = 0.0
    pokerus_coef: float = 0.0


def team_health(obs: dict) -> float:
    return sum(p["hp"] / p["maxHp"] for p in obs["party"] if p["maxHp"]) / 6


def exp_levels(prev: list[dict], nxt: list[dict]) -> float:
    """Levels gained by Pokemon on the team both before and after."""
    before = {p["uid"]: p["level"] for p in prev}
    return float(sum(max(0, p["level"] - before[p["uid"]]) for p in nxt if p["uid"] in before))


def carry_gain(prev: list[dict], nxt: list[dict]) -> float:
    """Change in the highest level, counting only Pokemon already on the team."""
    before = {p["uid"] for p in prev}
    kept = [p["level"] for p in nxt if p["uid"] in before]
    return float(max(kept, default=0) - carry_level(prev))


def enemy_health(obs: dict) -> float:
    return sum(e["hp"] / e["maxHp"] for e in obs["enemies"] if e["maxHp"])


class BattleEnv:
    def __init__(self, env_id: int, seed_prefix: str = "train", max_wave: int = 500,
                 reward: RewardConfig | None = None, sim: SimClient | None = None,
                 policy_kinds: tuple[str, ...] = BATTLE_KINDS, double_chance: float | None = None):
        """`sim`: a client from a shared simulator group; launches its own simulator if omitted.
        `double_chance`: more double battles for practice (see SimClient.reset)."""
        self.double_chance = double_chance
        self.policy_kinds = tuple(policy_kinds)
        self.tm_memory = TmMemory()
        self.strategy = StrategyTracker()
        self.sim = sim or SimClient.launch(request_timeout=120)
        self.helper = HeuristicAgent()
        self.sim_errors = 0
        self.reward_cfg = reward or RewardConfig()
        self.max_wave = max_wave
        self.seeds = (f"{seed_prefix}-{env_id}-{i}" for i in itertools.count())
        self.obs: dict | None = None
        self.episode_return = 0.0
        self.episode_steps = 0
        # Draining before a recycle (see VecBattleEnv.start_drain): `pending` marks an episode that
        # must finish first; `tracked` says whether this episode counts in the episode stats.
        self.pending = False
        self.tracked = True
        self.track_new = True

    def _play_until_policy(self, resp: dict) -> dict:
        """Let the heuristic answer other decisions until the policy has one of its own."""
        while not resp["done"] and resp["obs"]["kind"] not in self.policy_kinds:
            action = self.helper.act(resp["obs"])
            self.strategy.note(resp["obs"], action)
            resp = self.sim.step(action)
        if not resp["done"]:
            resp = {**resp, "obs": self.tm_memory.filter(resp["obs"])}
        return resp

    def _restart_sim(self, error: Exception) -> None:
        """Replace a simulator that errored or hung, so one bad episode can't end a training run."""
        self.sim_errors += 1
        print(f"[env] simulator error ({error!r}); replacing it with a standalone one", flush=True)
        if self.sim.owns_proc:
            self.sim.proc.kill()
        self.sim.sock.close()  # a simulator in a shared group is abandoned, not killed
        self.sim = SimClient.launch(request_timeout=120)

    def reset(self) -> dict:
        try:
            return self._reset()
        except (SimError, OSError) as e:  # socket.timeout is an OSError
            self._restart_sim(e)
            return self._reset()

    def _reset(self) -> dict:
        self.helper.reset()
        self.tm_memory.reset()
        self.strategy.reset()
        resp = self._play_until_policy(self.sim.reset(seed=next(self.seeds), max_wave=self.max_wave,
                                                         double_chance=self.double_chance))
        if resp["done"]:  # ended before any battle decision; try another seed
            return self._reset()
        self.obs = resp["obs"]
        self.episode_return = 0.0
        self.episode_steps = 0
        self.tracked = self.track_new
        return self.obs

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        prev = self.obs
        self.helper._update_carry(prev["party"])  # the heuristic's item choices follow its carry
        self.tm_memory.note(prev, int(action))
        self.strategy.note(prev, int(action))
        try:
            resp = self._play_until_policy(self.sim.step(int(action)))
        except (SimError, OSError) as e:
            # End the episode with no reward; GAE stops at done, so nothing bootstraps across it.
            self._restart_sim(e)
            self.pending = False
            return self.reset(), 0.0, True, {"invalid": False, "sim_error": repr(e)}
        obs, done = resp["obs"], resp["done"]

        cfg = self.reward_cfg
        reward = cfg.wave_reward * (obs["wave"] - prev["wave"])
        next_health = 0.0 if done and resp["info"]["reason"] == "game_over" else team_health(obs)
        reward += cfg.team_coef * (cfg.gamma * next_health - team_health(prev))
        if cfg.level_coef:
            reward += cfg.level_coef * exp_levels(prev["party"], obs["party"])
        if cfg.carry_coef:
            reward += cfg.carry_coef * carry_gain(prev["party"], obs["party"])
        if cfg.pokerus_coef:
            reward += cfg.pokerus_coef * (pokerus_count(obs["party"]) - pokerus_count(prev["party"]))
        if obs["wave"] == prev["wave"]:
            reward += cfg.damage_coef * max(0.0, enemy_health(prev) - enemy_health(obs))

        self.episode_return += reward
        self.episode_steps += 1
        info = {"invalid": resp["info"].get("invalidAction", False)}
        if done:
            if self.tracked:
                info["episode"] = {"wave": obs["wave"], "reason": resp["info"]["reason"],
                                   "return": self.episode_return, "steps": self.episode_steps,
                                   **self.strategy.summary(obs)}
            self.pending = False
            obs = self.reset()
        else:
            self.obs = obs
        return obs, reward, done, info

    def close(self) -> None:
        self.sim.close()


class VecBattleEnv:
    """N BattleEnvs on one shared simulator process, stepped in parallel threads."""

    def __init__(self, num_envs: int, double_envs: int = 0, double_chance: float | None = None, **kwargs):
        """The first `double_envs` envs play with `double_chance` (a doubles curriculum)."""
        self.group = launch_sims(num_envs, request_timeout=120)
        self.envs = [BattleEnv(i, sim=client, double_chance=double_chance if i < double_envs else None, **kwargs)
                     for i, client in enumerate(self.group.clients)]
        self.pool = ThreadPoolExecutor(num_envs)
        self.draining = False

    def reset(self) -> list[dict]:
        return list(self.pool.map(lambda e: e.reset(), self.envs))

    def step(self, actions) -> tuple[list[dict], np.ndarray, np.ndarray, list[dict]]:
        results = list(self.pool.map(lambda ea: ea[0].step(ea[1]), zip(self.envs, actions)))
        obs, rewards, dones, infos = zip(*results)
        return list(obs), np.array(rewards, np.float32), np.array(dones, bool), list(infos)

    def start_drain(self) -> None:
        """Prepare to recycle without cutting episodes short.

        Recycling mid-episode drops the in-progress episodes, and long episodes are the likeliest to
        be in progress, so the stats would be biased low and late waves rarely played. Instead, every
        current episode is marked pending and plays on; envs that finish early start new episodes,
        which still train but are left out of the stats (those that finish before the recycle would
        be the short ones). Recycle once `drained`.
        """
        self.draining = True
        for env in self.envs:
            env.pending = True
            env.track_new = False

    @property
    def drained(self) -> bool:
        return not any(env.pending for env in self.envs)

    def recycle(self) -> int:
        """Restart all simulators (a slow per-episode leak remains) and start fresh episodes.

        Call between rollouts, ideally once `drained`. Returns how many tracked episodes were cut
        short (dropped, not reported as finished).
        """
        cut = sum(env.pending and env.tracked for env in self.envs)
        for env in self.envs:
            env.pending = False
            env.track_new = True
        self.draining = False
        list(self.pool.map(lambda e: e.close(), self.envs))
        self.group.close()
        self.group = launch_sims(len(self.envs), request_timeout=120)
        for env, client in zip(self.envs, self.group.clients):
            env.sim = client
        self.reset()
        return cut

    def close(self) -> None:
        list(self.pool.map(lambda e: e.close(), self.envs))
        self.pool.shutdown()
        self.group.close()
