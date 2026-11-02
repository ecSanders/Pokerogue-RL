"""Actor-critic network for battle decisions.

Species and move ids go through learned embeddings; per-Pokemon and per-move features are
concatenated with them and flattened with the global features into a shared MLP trunk.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from pokerogue_rl import features as F
from pokerogue_rl import run_features as RF


def _zero_linear(n_in: int, n_out: int) -> nn.Linear:
    """A bias-free linear layer that starts at zero: a new input that doesn't change the output yet."""
    layer = nn.Linear(n_in, n_out, bias=False)
    nn.init.zeros_(layer.weight)
    return layer


class BattlePolicy(nn.Module):
    def __init__(self, species_dim: int = 16, move_dim: int = 16, ability_dim: int = 8, hidden: int = 512):
        super().__init__()
        self.species_emb = nn.Embedding(F.NUM_SPECIES, species_dim)
        self.move_emb = nn.Embedding(F.NUM_MOVES, move_dim)
        # Abilities enter as a zero-initialised addition to the first layer, so checkpoints from
        # before they were encoded load and behave unchanged (see load_compatible).
        self.ability_emb = nn.Embedding(F.NUM_ABILITIES, ability_dim)
        self.ability_proj = _zero_linear(F.SLOTS * ability_dim, hidden)
        per_move = F.MOVE_DIM + move_dim
        per_mon = F.MON_DIM + species_dim + F.MOVES * per_move
        in_dim = F.SLOTS * per_mon + F.GLOBAL_DIM
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, F.NUM_ACTIONS)
        self.value_head = nn.Linear(hidden, 1)
        # Small initial policy logits so early behaviour is near-uniform over legal actions.
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        n = batch["species"].shape[0]
        moves = torch.cat([batch["move"], self.move_emb(batch["moves"])], dim=-1)  # [n, 8, 4, per_move]
        mons = torch.cat([batch["mon"], self.species_emb(batch["species"]), moves.flatten(2)], dim=-1)
        x = torch.cat([mons.flatten(1), batch["glob"]], dim=-1)
        x = self.trunk[0](x) + self.ability_proj(self.ability_emb(batch["ability"]).flatten(1))
        h = self.trunk[1:](x)
        logits = self.policy_head(h).masked_fill(~batch["mask"], -1e9)
        return logits, self.value_head(h).squeeze(-1)


def to_tensors(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}


class RunPolicy(nn.Module):
    """Scores the candidate actions of a run decision (see run_features) against the run state."""

    def __init__(self, species_dim: int = 16, move_dim: int = 16, item_dim: int = 16, ability_dim: int = 8,
                 mon_hidden: int = 128, hidden: int = 256):
        super().__init__()
        self.species_emb = nn.Embedding(F.NUM_SPECIES, species_dim)
        self.move_emb = nn.Embedding(F.NUM_MOVES, move_dim)
        self.item_emb = nn.Embedding(RF.ITEM_BUCKETS, item_dim)
        self.ability_emb = nn.Embedding(F.NUM_ABILITIES, ability_dim)
        self.ability_proj = _zero_linear(ability_dim, mon_hidden)
        per_mon = F.MON_DIM + species_dim + F.MOVES * (F.MOVE_DIM + move_dim)
        self.mon_enc = nn.Sequential(nn.Linear(per_mon, mon_hidden), nn.ReLU())
        state_in = RF.MON_SLOTS * mon_hidden + RF.GLOBAL_DIM + item_dim + move_dim + F.MOVE_DIM
        self.state = nn.Sequential(
            nn.Linear(state_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        cand_in = RF.CAND_DIM + item_dim + move_dim + mon_hidden
        self.cand_enc = nn.Sequential(nn.Linear(cand_in, hidden), nn.ReLU())
        self.score = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.value_head = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.score[-1].weight, gain=0.01)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        n = batch["species"].shape[0]
        moves = torch.cat([batch["move"], self.move_emb(batch["moves"])], dim=-1)  # [n, 7, 4, .]
        mons = torch.cat([batch["mon"], self.species_emb(batch["species"]), moves.flatten(2)], dim=-1)
        h_mon = self.mon_enc[1](self.mon_enc[0](mons) + self.ability_proj(self.ability_emb(batch["ability"])))
        state = self.state(torch.cat([
            h_mon.flatten(1), batch["glob"], self.item_emb(batch["ctx_item"]),
            self.move_emb(batch["ctx_move_id"]), batch["ctx_move"],
        ], dim=-1))
        # Candidate slot 0 = none; 1..7 = mon slot index + 1.
        slot_h = torch.cat([torch.zeros_like(h_mon[:, :1]), h_mon], dim=1)
        cand_mon = slot_h.gather(1, batch["cand_slot"].unsqueeze(-1).expand(-1, -1, h_mon.shape[-1]))
        c = self.cand_enc(torch.cat([
            batch["cand"], self.item_emb(batch["cand_item"]), self.move_emb(batch["cand_move"]), cand_mon,
        ], dim=-1))  # [n, C, hidden]
        logits = self.score(torch.cat([state.unsqueeze(1).expand_as(c), c], dim=-1)).squeeze(-1)
        return logits.masked_fill(~batch["mask"], -1e9), self.value_head(state).squeeze(-1)


NUM_JOINT_ACTIONS = max(F.NUM_ACTIONS, RF.MAX_CANDS)


_ZERO: dict[str, dict[str, np.ndarray]] = {}


def encode_step(obs: dict) -> dict[str, np.ndarray]:
    """Joint encoding for AgentPolicy: battle features under "b_", run features under "r_" (the
    unused half zeroed), and "is_run"."""
    is_run = obs["kind"] in RF.RUN_KINDS
    own = RF.encode(obs) if is_run else F.encode(obs)
    prefix, other = ("r_", "b_") if is_run else ("b_", "r_")
    if other not in _ZERO:
        _ZERO["b_"] = {k: np.zeros_like(v) for k, v in F.encode(_DUMMY_BATTLE).items()}
        _ZERO["r_"] = {k: np.zeros_like(v) for k, v in RF.encode(_DUMMY_RUN).items()}
    out = {prefix + k: v for k, v in own.items()}
    out.update({other + k: v for k, v in _ZERO[other].items()})
    out["is_run"] = np.bool_(is_run)
    return out


_DUMMY_BATTLE = {"kind": "command", "party": [], "enemies": [], "legal": [], "wave": 0, "turn": 0, "double": False,
                 "money": 0, "balls": [0] * F.NUM_BALLS}
_DUMMY_RUN = {"kind": "release", "party": [], "legal": [], "wave": 0, "money": 0, "balls": [0] * F.NUM_BALLS}


class AgentPolicy(nn.Module):
    """Battle policy plus run policy behind one action space of NUM_JOINT_ACTIONS logits.

    Each row of a batch goes through the network for its decision type (`is_run`); logits beyond a
    decision's own action count are masked out."""

    def __init__(self):
        super().__init__()
        self.battle = BattlePolicy()
        self.run = RunPolicy()

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        is_run = batch["is_run"]
        n = is_run.shape[0]
        some = next(iter(batch.values()))
        logits = torch.full((n, NUM_JOINT_ACTIONS), -1e9, device=some.device)
        value = torch.zeros(n, device=some.device)
        for mask, net, prefix, width in ((~is_run, self.battle, "b_", F.NUM_ACTIONS),
                                          (is_run, self.run, "r_", RF.MAX_CANDS)):
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=True)[0]
            sub = {k[2:]: v[idx] for k, v in batch.items() if k.startswith(prefix)}
            lg, val = net(sub)
            logits[idx, :width] = lg.to(logits.dtype)
            value[idx] = val.to(value.dtype)
        return logits, value

    def load_checkpoint(self, path: str, map_location="cpu") -> dict:
        """Load an AgentPolicy checkpoint, or a BattlePolicy one into the battle half."""
        ckpt = torch.load(path, map_location=map_location)
        state = ckpt["policy"]
        if any(k.startswith("battle.") for k in state):
            load_compatible(self, state)
        else:
            load_compatible(self.battle, state)
        return ckpt


# Parameters added after checkpoints already existed; they start neutral (see _zero_linear).
NEW_KEYS = ("ability_emb.", "ability_proj.")


def load_compatible(module: nn.Module, state: dict) -> None:
    """load_state_dict that tolerates only the NEW_KEYS parameters missing from older checkpoints."""
    missing, unexpected = module.load_state_dict(state, strict=False)
    bad = [k for k in missing if not any(n in k for n in NEW_KEYS)]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing {bad}, unexpected {unexpected}")
