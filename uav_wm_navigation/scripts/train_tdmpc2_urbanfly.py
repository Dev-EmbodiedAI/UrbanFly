from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uav_wm_navigation.data.world_model_dataset_v2 import (
    UrbanFlyTransitionDataset,
)
from uav_wm_navigation.world_models.tdmpc2_continuous import TDMPC2Network
from uav_wm_navigation.world_models.tdmpc2_training import TDMPC2Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dataset = UrbanFlyTransitionDataset(args.manifests)
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    model = TDMPC2Network().to(device)
    trainer = TDMPC2Trainer(model)
    iterator = iter(loader)
    latest = None
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        latest = trainer.train_step(batch)
        if step == 1 or step % 100 == 0:
            print(
                json.dumps({"step": step, **asdict(latest)}),
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    torch.save(
        {
            "model": model.state_dict(),
            "schema": "urbanfly-world-model-v2",
            "family": "tdmpc2_continuous",
            "training_steps": args.steps,
            "seed": args.seed,
            "manifests": [str(path.resolve()) for path in args.manifests],
            "final_loss": asdict(latest) if latest is not None else None,
        },
        temporary,
    )
    temporary.replace(args.output)
    print(json.dumps({"checkpoint": str(args.output.resolve())}))


if __name__ == "__main__":
    main(parse_args())
