"""Client for the headless PokeRogue simulator (pokerogue/test/rl/sim-server.test.ts).

The simulator speaks newline-delimited JSON over localhost TCP. `SimClient` connects to a
running server; `SimClient.launch()` starts one as a subprocess first.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

from pokerogue_rl.install_sim import install_sim

POKEROGUE_DIR = Path(__file__).resolve().parent.parent / "pokerogue"
NUM_ACTIONS = 10  # 0-3: move slots, 4+k: switch to party slot k


class SimError(RuntimeError):
    pass


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SimClient:
    def __init__(self, port: int, host: str = "127.0.0.1", connect_timeout: float = 120.0,
                 proc: subprocess.Popen | None = None, request_timeout: float | None = None):
        self.proc = proc
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
        """Start a simulator process on a free port and connect to it."""
        install_sim()
        port = free_port()
        log = open(log_path, "w") if log_path else subprocess.DEVNULL
        proc = subprocess.Popen(
            ["pnpm", "exec", "vitest", "run", "test/rl/sim-server.test.ts", "--silent"],
            cwd=POKEROGUE_DIR,
            env={**os.environ, "RL_PORT": str(port)},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        return cls(port, proc=proc, request_timeout=request_timeout)

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
              starters: list[str] | None = None, max_wave: int | None = None) -> dict:
        msg = {"cmd": "reset", "mode": mode}
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
        """Disconnect; also shut the server down if this client launched it."""
        try:
            self._request({"cmd": "shutdown" if self.proc is not None else "close"})
        except (SimError, OSError):
            pass
        self.sock.close()
        if self.proc is not None:
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self) -> SimClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
