"""Client for the headless PokeRogue simulator (sim/sim-server.ts).

The simulator speaks newline-delimited JSON over localhost TCP. `SimClient` connects to a
running server. `launch_sims(n)` starts one Vitest process hosting n simulators (~1.3GB shared plus
~0.7GB each, instead of ~2.1GB each as separate processes); `SimClient.launch()` starts one.
"""

from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import time
from pathlib import Path

from pokerogue_rl.install_sim import MAX_SIMS, install_sim, sim_test_file

POKEROGUE_DIR = Path(__file__).resolve().parent.parent / "pokerogue"


class SimError(RuntimeError):
    pass


def free_ports(n: int) -> int:
    """First of n consecutive free localhost ports."""
    for _ in range(100):
        base = random.randint(20000, 60000 - n)
        try:
            socks = []
            for port in range(base, base + n):
                s = socket.socket()
                socks.append(s)
                s.bind(("127.0.0.1", port))
            return base
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise SimError(f"no {n} consecutive free ports found")


class SimGroup:
    """One Vitest process hosting several simulators, each with its own client."""

    def __init__(self, proc: subprocess.Popen, clients: list[SimClient]):
        self.proc = proc
        self.clients = clients

    def close(self) -> None:
        for client in self.clients:
            client.close()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def launch_sims(n: int, log_path: str | os.PathLike | None = None,
                request_timeout: float | None = None) -> SimGroup:
    """Start one Vitest process running n simulators and connect a client to each."""
    if not 1 <= n <= MAX_SIMS:
        raise ValueError(f"can host 1-{MAX_SIMS} simulators per process, got {n}")
    install_sim()
    base = free_ports(n)
    log = open(log_path, "w") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(
        ["pnpm", "exec", "vitest", "run", "--silent", f"--maxWorkers={n}",
         *(f"test/rl/{sim_test_file(i)}" for i in range(n))],
        cwd=POKEROGUE_DIR,
        env={**os.environ, "RL_PORT": str(base), "RL_SIMS": str(n)},
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    clients = [SimClient(base + i, proc=proc, request_timeout=request_timeout) for i in range(n)]
    return SimGroup(proc, clients)


class SimClient:
    def __init__(self, port: int, host: str = "127.0.0.1", connect_timeout: float = 180.0,
                 proc: subprocess.Popen | None = None, request_timeout: float | None = None,
                 owns_proc: bool = False):
        self.proc = proc
        self.owns_proc = owns_proc
        deadline = time.monotonic() + connect_timeout
        while True:
            try:
                self.sock = socket.create_connection((host, port))
                break
            except OSError:
                if proc is not None and proc.poll() is not None:
                    raise SimError(f"simulator exited with code {proc.returncode} before accepting connections")
                if time.monotonic() > deadline:
                    raise SimError(f"could not connect to simulator on {host}:{port}")
                time.sleep(0.5)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Raises socket.timeout if the simulator takes longer than this to answer one request.
        self.sock.settimeout(request_timeout)
        self.reader = self.sock.makefile("r", encoding="utf-8")

    @classmethod
    def launch(cls, log_path: str | os.PathLike | None = None, request_timeout: float | None = None) -> SimClient:
        """Start a single-simulator process and connect to it; closing the client stops it."""
        client = launch_sims(1, log_path, request_timeout).clients[0]
        client.owns_proc = True
        return client

    def _request(self, msg: dict) -> dict:
        self.sock.sendall((json.dumps(msg) + "\n").encode())
        line = self.reader.readline()
        if not line:
            raise SimError("simulator closed the connection")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise SimError(resp.get("error", "unknown simulator error"))
        return resp

    def reset(self, seed: str | None = None, mode: str = "endless",
              starters: list[str] | None = None, max_wave: int | None = None,
              double_chance: float | None = None) -> dict:
        """`double_chance`: training curriculum, see ResetOpts.doubleChance in sim/sim-server.ts."""
        msg = {"cmd": "reset", "mode": mode}
        if double_chance is not None:
            msg["doubleChance"] = double_chance
        if seed is not None:
            msg["seed"] = seed
        if starters is not None:
            msg["starters"] = starters
        if max_wave is not None:
            msg["maxWave"] = max_wave
        return self._request(msg)

    def step(self, action: int) -> dict:
        return self._request({"cmd": "step", "action": int(action)})

    def close(self) -> None:
        """Disconnect; stop the simulator if it was launched for us (and its process, if we own it)."""
        try:
            self._request({"cmd": "shutdown" if self.proc is not None else "close"})
        except (SimError, OSError):
            pass
        self.sock.close()
        if self.owns_proc:
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self) -> SimClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
