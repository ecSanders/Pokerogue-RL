"""Check that PyTorch can use the GPU and compare it with the CPU on a policy-sized MLP.

Usage: python scripts/check_gpu.py
"""

import time

import torch
import torch.nn as nn


def make_policy() -> nn.Module:
    # Roughly the size we expect for the battle policy (~1-2M params).
    return nn.Sequential(
        nn.Linear(512, 1024), nn.ReLU(),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Linear(1024, 512), nn.ReLU(),
        nn.Linear(512, 16),
    )


def bench(device: torch.device, batch: int, iters: int = 50) -> float:
    """Return train steps/sec (forward + backward + optimizer step)."""
    model = make_policy().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    x = torch.randn(batch, 512, device=device)
    y = torch.randn(batch, 16, device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()

    for _ in range(5):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return iters / (time.perf_counter() - start)


def main() -> None:
    print(f"torch {torch.__version__}, HIP {torch.version.hip}")
    if not torch.cuda.is_available():
        raise SystemExit(
            "GPU not available to PyTorch. Try re-running setup with --override, "
            "and check that `rocminfo` lists your GPU."
        )

    gpu = torch.device("cuda")
    props = torch.cuda.get_device_properties(gpu)
    print(f"Device: {props.name} ({props.gcnArchName}), {props.total_memory / 2**30:.1f} GiB")

    # Correctness: GPU result should match CPU.
    a, b = torch.randn(256, 256), torch.randn(256, 256)
    err = (a @ b - (a.to(gpu) @ b.to(gpu)).cpu()).abs().max().item()
    print(f"Matmul max abs error vs CPU: {err:.2e}")
    if err > 1e-2:
        raise SystemExit("GPU results don't match CPU; the device is not working correctly.")

    print("\nTrain steps/sec on a ~2M-param MLP:")
    print(f"{'batch':>8} {'cpu':>10} {'gpu':>10} {'speedup':>9}")
    for batch in (256, 2048, 8192):
        cpu = bench(torch.device("cpu"), batch, iters=10)
        gpu_rate = bench(gpu, batch)
        print(f"{batch:>8} {cpu:>10.1f} {gpu_rate:>10.1f} {gpu_rate / cpu:>8.1f}x")


if __name__ == "__main__":
    main()
