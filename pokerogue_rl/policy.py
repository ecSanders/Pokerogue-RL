"""Actor-critic network for battle decisions.

Species and move ids go through learned embeddings; per-Pokemon and per-move features are
concatenated with them and flattened with the global features into a shared MLP trunk.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pokerogue_rl import features as F


class BattlePolicy(nn.Module):
    def __init__(self, species_dim: int = 16, move_dim: int = 16, hidden: int = 512):
        super().__init__()
        self.species_emb = nn.Embedding(F.NUM_SPECIES, species_dim)
        self.move_emb = nn.Embedding(F.NUM_MOVES, move_dim)
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
        h = self.trunk(x)
        logits = self.policy_head(h).masked_fill(~batch["mask"], -1e9)
        return logits, self.value_head(h).squeeze(-1)


def to_tensors(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
