import torch
from torch import Tensor, nn
from einops import rearrange

from linear import Linear
from rsnorm import RMSNorm
from multihead import MultiHeadSelfAttention
from glu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None, 
        ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.norm1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )

        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
        )

        self.norm2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff= d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        # pre-norm
        n1 = self.norm1(x)
        attention_output = MultiHeadSelfAttention(n1)
        y  = x + attention_output

        # 第二个子层swiglu
        n2 = self.norm2(y)
        ffn_output = SwiGLU(n2)
        z = y + ffn_output

        return z

