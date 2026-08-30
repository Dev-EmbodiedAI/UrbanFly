#!/usr/bin/env python
"""Probe PPU PyTorch operator support while isolating native runtime aborts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time


CASES = (
    ("matmul", "float32"),
    ("conv2d", "float32"),
    ("matmul", "bfloat16"),
    ("conv2d", "bfloat16"),
    ("conv2d", "float16"),
    ("autocast_convstack", "bfloat16"),
)


def worker(device: str, operation: str, dtype_name: str) -> None:
    import torch

    dtype = getattr(torch, dtype_name)
    torch.manual_seed(123)
    started = time.perf_counter()
    if operation == "matmul":
        left = torch.randn(1024, 1024, device=device, dtype=dtype, requires_grad=True)
        right = torch.randn(1024, 1024, device=device, dtype=dtype)
        loss = (left @ right).float().square().mean()
        loss.backward()
        finite = bool(torch.isfinite(left.grad).all())
    elif operation == "conv2d":
        model = __import__("torch").nn.Conv2d(4, 16, 3, padding=1).to(device=device, dtype=dtype)
        inputs = torch.randn(2, 4, 64, 64, device=device, dtype=dtype)
        loss = model(inputs).float().square().mean()
        loss.backward()
        finite = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    elif operation == "autocast_convstack":
        model = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 4, 3, padding=1),
        ).to(device)
        inputs = torch.randn(8, 4, 128, 128, device=device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            loss = (model(inputs) - inputs).square().mean()
        loss.backward()
        finite = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    else:
        raise ValueError(operation)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    print(json.dumps({"ok": True, "device": device, "operation": operation, "dtype": dtype_name,
                      "loss": float(loss), "grad_finite": finite, "seconds": time.perf_counter() - started}))


def parent(device_prefix: str) -> int:
    import torch

    if device_prefix == "cuda":
        devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    else:
        devices = [device_prefix]
    results = []
    for device in devices:
        for operation, dtype_name in CASES:
            command = [sys.executable, __file__, "--worker", "--device", device,
                       "--operation", operation, "--dtype", dtype_name]
            completed = subprocess.run(command, text=True, capture_output=True)
            stdout = completed.stdout.strip().splitlines()
            item = {
                "device": device,
                "operation": operation,
                "dtype": dtype_name,
                "returncode": completed.returncode,
                "ok": completed.returncode == 0,
                "result": json.loads(stdout[-1]) if completed.returncode == 0 and stdout else None,
                "stderr_tail": "\n".join(completed.stderr.strip().splitlines()[-8:]),
            }
            results.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    if device_prefix == "cuda" and len(devices) > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(devices)}",
            __file__,
            "--distributed-worker",
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        item = {
            "device": "all",
            "operation": "pccl_all_reduce",
            "dtype": "float32",
            "returncode": completed.returncode,
            "ok": completed.returncode == 0,
            "result": completed.stdout.strip().splitlines()[-len(devices):] if completed.returncode == 0 else None,
            "stderr_tail": "\n".join(completed.stderr.strip().splitlines()[-12:]),
        }
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    summary = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_count": len(devices),
        "passed": sum(item["ok"] for item in results),
        "failed": sum(not item["ok"] for item in results),
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not any(not item["ok"] for item in results) else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-prefix", default="cuda")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--distributed-worker", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--operation", choices=("matmul", "conv2d", "autocast_convstack"), default="matmul")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    args = parser.parse_args()
    if args.worker:
        worker(args.device, args.operation, args.dtype)
        return
    if args.distributed_worker:
        import os
        import torch
        import torch.distributed as dist

        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl")
        value = torch.tensor([rank + 1.0], device=f"cuda:{rank}")
        dist.all_reduce(value)
        print(json.dumps({"rank": rank, "all_reduce": float(value), "ok": float(value) == 3.0}), flush=True)
        dist.destroy_process_group()
        return
    raise SystemExit(parent(args.device_prefix))


if __name__ == "__main__":
    main()
