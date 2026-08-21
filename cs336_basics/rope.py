import math

import torch
from torch import Tensor, nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: torch.device | None = None,
    ) -> None:
        # 初始化父类
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        if(d_k % 2 == 1):
            raise ValueError("d_k must be even")

        # i = 0, 1,..., d_k / 2 -1
        pair_indices = torch.arange(
            d_k // 2,
            device=device,
            dtype=torch.float32,
        )

        # freq_i = theta^(-2i / d_k)
        frequencies = theta ** (
            -2 * pair_indices / d_k
        )

        # position = 0,1...max_seq_len-1
        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32
        )

        # (max_seq_len, 1) * (1, d_k / 2)
        # -> (max_seq_len, d_k / 2)
        angles = (
            positions[:,None] * frequencies[None, :]
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
         return self.weight[token_ids]
        
    

