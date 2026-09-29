import os
import typing

import torch

CheckpointIO = str | os.PathLike | typing.BinaryIO | typing.IO[bytes]

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: CheckpointIO
)-> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: CheckpointIO,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    checkpint = torch.load(src, weights_only=True)

    model.load_state_dict(checkpint["model"])
    optimizer.load_state_dict(checkpint["optimizer"])

    return checkpint["iteration"]
    
    


