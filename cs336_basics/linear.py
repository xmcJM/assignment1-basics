import math

import torch
from einops import einsum
from torch import Tensor, nn

class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device | None=None,
            dtype: torch.dtype | None = None,
    )-> None:
        # 初始化父类
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # W形状 (dout, din)
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype
            )
        )
        std = math.sqrt(2/(in_features+out_features))
        # 原地初始化weight
        torch.nn.init.trunc_normal_(
            tensor=self.weight,
            mean=0, 
            std=std,
            a=-3*std,
            b=3*std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            x,
            self.weight,
            "... in_features, out_features in_features"
            "-> ... out_features",
        )


    


