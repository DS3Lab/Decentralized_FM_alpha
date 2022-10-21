import os
import time
import random
import json
import numpy as np
import torch

from comm.comm_utils import *


def load_checkpoint(pipe, args):
    
    if os.path.isfile(os.path.join(args.checkpoint_path, 'latest')):
        with open(os.path.join(args.checkpoint_path, 'latest')) as f:
            latest_step = int(f.read())
    else:
        print('no checkpoint available, skipping')
        return
    
    checkpoint_step_path = os.path.join(args.checkpoint_path, f"checkpoint_{latest_step}")
    
    with open(os.path.join(checkpoint_step_path, 'meta.json')) as f:
        meta = json.load(f)
        
    pipe.global_step = latest_step
    
    pipe.model.model.load_state_dict(
        torch.load(
            os.path.join(
                checkpoint_step_path, f'prank_{get_pipeline_parallel_rank()}_checkpoint.pt'
            )
        )
    )
    
    pipe.optimizer.load_state_dict(
        torch.load(
            os.path.join(
                checkpoint_step_path, f'prank_{get_pipeline_parallel_rank()}_optimizer.pt'
            )
        )
    )
    
    pipe.scheduler.load_state_dict(
        torch.load(
            os.path.join(
                checkpoint_step_path, f'prank_{get_pipeline_parallel_rank()}_scheduler.pt'
            )
        )
    )
            
def save_checkpoint(pipe, args):
    
    latest_step = pipe.global_step
    checkpoint_step_path = os.path.join(args.checkpoint_path, f"checkpoint_{latest_step}")
    
    os.system(f"mkdir -p {checkpoint_step_path}")
    
    torch.save(
        pipe.model.model.state_dict(),
        os.path.join(
            checkpoint_step_path, f'prank_{get_pipeline_parallel_rank()}_checkpoint.pt'
        )
    )
    
    torch.save(
        pipe.optimizer.state_dict(),
        os.path.join(
            checkpoint_step_path, f'prank_{get_pipeline_parallel_rank()}_optimizer.pt'
        )
    )
    
    torch.save(
        pipe.scheduler.state_dict(),
        os.path.join(
            checkpoint_step_path, f'prank_{get_pipeline_parallel_rank()}_scheduler.pt'
        )
    )
    
    with open(os.path.join(checkpoint_step_path, 'meta.json'), 'w') as f:
        json.dump({
            'step': latest_step,
        }, f)
    
    with open(os.path.join(args.checkpoint_path, 'latest'), 'w') as f:
        f.write(f"{latest_step}")