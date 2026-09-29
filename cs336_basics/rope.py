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

        # cos 和 sin 不参与训练
        self.register_buffer(
            "cos_cache",
            torch.cos(angles)
        )
        self.register_buffer(
            "sin_cache",
            torch.sin(angles)
        )

    def forward(
            self, 
            token_ids: torch.Tensor, 
            token_positions: torch.Tensor) -> torch.Tensor:
         if x.shape[-1] != self.d_k:
            raise ValueError(
                f"Expected last dimension {self.d_k}, "
                f"got {x.shape[-1]}"
            )

        input_dtype = x.dtype

        cos = self.cos_cache[token_positions].to(input_dtype)
        sin = self.sin_cache[token_positions].to(input_dtype)

        x_even = [..., 0::2]
        x_odd = [..., 1::2]

        rotated_even = x_even*cos - x_odd*sin
        rotated_odd = x_even*sin + x_odd*cos

        result = torch.stack(
            (rotated_even, rotated_odd),
            dim=-1
        ).flatten(start_dim=-2)
        return result.to(input_dtype)
    

