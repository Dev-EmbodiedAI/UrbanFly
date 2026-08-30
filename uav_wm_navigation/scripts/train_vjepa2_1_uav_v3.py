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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from uav_wm_navigation.data.world_model_dataset_v3 import UrbanFlyRGBDSequenceDataset
from uav_wm_navigation.world_models.vjepa2_official import OfficialVJEPA21Encoder,UAVActionConditionedJEPAPredictor,vjepa_uav_training_loss


def main()->None:
    parser=argparse.ArgumentParser(description="Train the official V-JEPA 2.1 frozen/LoRA UAV action predictor")
    parser.add_argument("manifests",nargs="+",type=Path);parser.add_argument("--official-checkpoint",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--train-mode",choices=("frozen","lora"),default="frozen");parser.add_argument("--encoder-dim",type=int,default=1024);parser.add_argument("--steps",type=int,default=50_000);parser.add_argument("--batch-size",type=int,default=2);parser.add_argument("--workers",type=int,default=2);parser.add_argument("--seed",type=int,default=101);parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu");parser.add_argument("--log-every",type=int,default=50)
    args=parser.parse_args();random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    encoder=OfficialVJEPA21Encoder(args.official_checkpoint,train_mode=args.train_mode,history_frames=15);model=UAVActionConditionedJEPAPredictor(encoder,args.encoder_dim,latent_dim=256).to(args.device);parameters=[item for item in model.parameters() if item.requires_grad];optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=1e-4)
    dataset=UrbanFlyRGBDSequenceDataset(args.manifests,sequence_length=30,image_size=(192,320),view="world_model_supervision",shuffle_shards=True,seed=args.seed);loader=DataLoader(dataset,batch_size=args.batch_size,num_workers=args.workers,pin_memory=args.device.startswith("cuda"));iterator=iter(loader);history=[];model.train()
    for step in range(1,args.steps+1):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        total,losses=vjepa_uav_training_loss(model,batch);optimizer.zero_grad(set_to_none=True);total.backward();torch.nn.utils.clip_grad_norm_(parameters,20);optimizer.step()
        if step==1 or step%args.log_every==0:
            row={"step":step,"total":float(total.detach()),**{name:float(value.detach()) for name,value in losses.items()}};history.append(row);print(json.dumps(row),flush=True)
    state={name:value.cpu() for name,value in model.state_dict().items() if not name.startswith("encoder.encoder.") or ".a.weight" in name or ".b.weight" in name};payload={"schema":"urbanfly-world-model-v3","family":"vjepa2_1_uav","training_steps":args.steps,"seed":args.seed,"train_mode":args.train_mode,"encoder_dim":args.encoder_dim,"latent_dim":256,"official_model":"vjepa2_1_vit_large_384","official_checkpoint":str(args.official_checkpoint.resolve()),"official_checkpoint_sha256":hashlib.sha256(args.official_checkpoint.resolve().read_bytes()).hexdigest(),"model":state,"dataset_manifests":[{"path":str(path.resolve()),"sha256":hashlib.sha256(path.resolve().read_bytes()).hexdigest()} for path in args.manifests],"robot_ac_predictor_reused":False,"policy_inputs_exclude_privileged":True}
    args.output.parent.mkdir(parents=True,exist_ok=True);temporary=args.output.with_suffix(args.output.suffix+".partial");torch.save(payload,temporary);temporary.replace(args.output);args.output.with_suffix(".history.json").write_text(json.dumps(history,indent=2),encoding="utf-8");print(json.dumps({"checkpoint":str(args.output.resolve()),"steps":args.steps,"mode":args.train_mode}))


if __name__=="__main__":main()
