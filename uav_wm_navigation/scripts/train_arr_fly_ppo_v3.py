from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from uav_wm_navigation.world_models.arr_fly_policy import ARRFlyActorCritic,asymmetric_ppo_loss


REQUIRED=("depth_history","proprio","critic_privileged","action","old_log_probability","advantage","return")


def main()->None:
    parser=argparse.ArgumentParser(description="Train ARR-Fly asymmetric PPO from auditable on-policy rollout archives")
    parser.add_argument("rollouts",nargs="+",type=Path);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--updates",type=int,default=1,help="Must be one: collect a fresh archive before every PPO update");parser.add_argument("--batch-size",type=int,default=256);parser.add_argument("--epochs-per-update",type=int,default=4);parser.add_argument("--seed",type=int,default=101);parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args();random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if args.updates != 1: raise ValueError("PPO forbids replaying stale rollout archives; collect fresh on-policy data between updates")
    archives=[]
    for path in args.rollouts:
        payload=np.load(path)
        if any(name not in payload for name in REQUIRED):raise ValueError(f"{path} is not an ARR-Fly on-policy rollout archive")
        if payload["depth_history"].shape[1:]!=(15,6,34) or payload["critic_privileged"].shape[1:]!=(36,):raise ValueError(f"invalid ARR-Fly rollout tensor shapes in {path}")
        archives.append({name:payload[name] for name in REQUIRED})
    data={name:np.concatenate([item[name] for item in archives]) for name in REQUIRED};count=len(data["action"])
    model=ARRFlyActorCritic().to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4);rng=np.random.default_rng(args.seed);history=[];training_steps=0
    for update in range(1,args.updates+1):
        order=rng.permutation(count);losses=[]
        for _ in range(args.epochs_per_update):
            for start in range(0,count,args.batch_size):
                indices=order[start:start+args.batch_size];batch={name:torch.from_numpy(data[name][indices]).float().to(args.device) for name in REQUIRED};batch["advantage"]=(batch["advantage"]-batch["advantage"].mean())/(batch["advantage"].std()+1e-6)
                total,parts=asymmetric_ppo_loss(model,batch);optimizer.zero_grad(set_to_none=True);total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),0.5);optimizer.step();losses.append([float(total.detach()),float(parts["policy"].detach()),float(parts["value"].detach())]);training_steps+=1
        row={"update":update,"total":float(np.mean(losses,axis=0)[0]),"policy":float(np.mean(losses,axis=0)[1]),"value":float(np.mean(losses,axis=0)[2])};history.append(row);print(json.dumps(row),flush=True)
    payload={"schema":"urbanfly-world-model-v3","family":"arr_fly_ppo","training_steps":training_steps,"ppo_updates":args.updates,"seed":args.seed,"model":model.state_dict(),"rollout_archives":[{"path":str(path.resolve()),"sha256":hashlib.sha256(path.resolve().read_bytes()).hexdigest()} for path in args.rollouts],"actor_privileged_inputs":False,"critic_privileged_dim":36,"on_policy_archive_required":True}
    args.output.parent.mkdir(parents=True,exist_ok=True);temporary=args.output.with_suffix(args.output.suffix+".partial");torch.save(payload,temporary);temporary.replace(args.output);args.output.with_suffix(".history.json").write_text(json.dumps(history,indent=2),encoding="utf-8");print(json.dumps({"checkpoint":str(args.output.resolve()),"training_steps":training_steps}))


if __name__=="__main__":main()
