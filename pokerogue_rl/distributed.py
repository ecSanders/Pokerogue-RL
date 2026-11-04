"""Multi-node data-parallel PPO: every rank (one per node) runs its own simulators and rollouts, and the
ranks average gradients each minibatch, so they train one shared model.

Off unless WORLD_SIZE (torchrun) or SLURM_NTASKS (srun) says there is more than one process.
MASTER_ADDR / MASTER_PORT must be set (hpc/job.sh does it). Ranks must use the same --num-envs and
--rollout, so they take the same number of minibatch steps per update.
"""

from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


class Dist:
    def __init__(self) -> None:
        env = os.environ
        self.world = int(env.get("WORLD_SIZE") or env.get("SLURM_NTASKS") or 1)
        self.rank = int(env.get("RANK") or env.get("SLURM_PROCID") or 0)
        self.local_rank = int(env.get("LOCAL_RANK") or env.get("SLURM_LOCALID") or 0)
        self.enabled = self.world > 1

    @property
    def main(self) -> bool:
        return self.rank == 0

    def device(self, requested: str) -> torch.device:
        dev = torch.device(requested)
        if self.enabled and dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", self.local_rank % torch.cuda.device_count())
        return dev

    def init(self, device: torch.device) -> None:
        if not self.enabled:
            return
        if device.type == "cuda":
            torch.cuda.set_device(device)
        # A long timeout: a rank restarting its simulators holds the others at the next all-reduce.
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", rank=self.rank,
                                world_size=self.world, timeout=datetime.timedelta(minutes=30))

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def broadcast_params(self, module: torch.nn.Module) -> None:
        """Start every rank from rank 0's weights."""
        if self.enabled:
            for t in module.state_dict().values():
                dist.broadcast(t, src=0)

    def average_grads(self, params: list[torch.nn.Parameter]) -> None:
        """Mean of the gradients over ranks, in one flat all-reduce. Parameters without a gradient on this
        rank (e.g. the run policy when the batch had no run decisions) count as zero."""
        if not self.enabled:
            return
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        flat = torch.cat([g.reshape(-1) for g in grads])
        dist.all_reduce(flat)
        flat /= self.world
        offset = 0
        for p in params:
            n = p.numel()
            p.grad = flat[offset:offset + n].view_as(p).clone()
            offset += n

    def gather(self, obj):
        """List of every rank's obj, on every rank."""
        if not self.enabled:
            return [obj]
        out = [None] * self.world
        dist.all_gather_object(out, obj)
        return out

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()
