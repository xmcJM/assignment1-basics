import torch
from torch import Tensor

def softmax(x: Tensor, dim: int) -> Tensor:
    max_values = x.max(
        dim=dim,
        keepdim=True,
    ).values

    # 减去最大防溢出
    shifted_x = x - max_values

    exp_x = torch.exp(shifted_x)

    sum_exp_x = exp_x.sum(
        dim=dim,
        keepdim=True,
    )

    return exp_x / sum_exp_x