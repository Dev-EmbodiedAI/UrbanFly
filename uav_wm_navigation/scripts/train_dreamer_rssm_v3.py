from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from uav_wm_navigation.data.world_model_dataset_v3 import UrbanFlyRGBDSequenceDataset
from uav_wm_navigation.world_models.dreamer_rssm_v3 import DreamerRSSMV3Network, DreamerRSSMV3Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the honest DreamerV3-style RGB-D RSSM candidate assistant")
    parser.add_argument("manifests", nargs="+", type=Path); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100_000); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--workers", type=int, default=4); parser.add_argument("--seed", type=int, default=101); parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    architecture = {"visual_dim": 384, "deterministic_dim": 768, "stochastic_groups": 32, "stochastic_classes": 16, "feature_dim": 512, "action_dim": 4}
    model = DreamerRSSMV3Network(**architecture).to(args.device); trainer = DreamerRSSMV3Trainer(model)
    dataset = UrbanFlyRGBDSequenceDataset(args.manifests, sequence_length=16, image_size=(128, 224), view="world_model_supervision", shuffle_shards=True, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=args.device.startswith("cuda")); iterator = iter(loader); history=[]; model.train()
    for step in range(1, args.steps + 1):
        try: batch = next(iterator)
        except StopIteration: iterator=iter(loader); batch=next(iterator)
        loss=trainer.train_step(batch)
        if step == 1 or step % args.log_every == 0:
            row={"step":step, **{name:float(getattr(loss,name)) for name in loss.__dataclass_fields__}}; history.append(row); print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True); payload={"schema":"urbanfly-world-model-v3","family":"dreamer_rssm_v3","honest_name":"DreamerV3-style RSSM assisted planning","training_steps":trainer.steps,"seed":args.seed,"architecture":architecture,"model":model.state_dict(),"dataset_manifests":[{"path":str(path.resolve()),"sha256":hashlib.sha256(path.resolve().read_bytes()).hexdigest()} for path in args.manifests],"actor_critic_included":False,"policy_inputs_exclude_privileged":True}
    temporary=args.output.with_suffix(args.output.suffix+".partial"); torch.save(payload,temporary); temporary.replace(args.output); args.output.with_suffix(".history.json").write_text(json.dumps(history,indent=2),encoding="utf-8"); print(json.dumps({"checkpoint":str(args.output.resolve()),"steps":trainer.steps}))


if __name__ == "__main__": main()
