import os
import subprocess

import torch
import torch.distributed as dist


def setup_distributed(backend='nccl'):
    """Initializes the distributed environment."""
    if not dist.is_available():
        print("Distributed training is not available.")
        return

    if not dist.is_initialized():
        # Environment variables like RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
        # are usually set by the launch utility (e.g., torchrun, Slurm)
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500") # Default port

        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://{master_addr}:{master_port}"
        )
        print(f"Process {rank}/{world_size} initialized.\n")
    return rank, world_size
