import torch
from torch import Tensor, nn
from einops import rearrange

from linear import Linear

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.k_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.v_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.output_proj = Linear(
                    d_model,
                    d_model,
                    device=device,
                    dtype=dtype,
                )

        def forward(self, x: Tensor) -> Tensor:
            # x:(..., seq_len, d_model)
            seq_len = x.shape[-2]
            #(...,seq_len, d_model)
            queries = self.q_proj(x)
            keys = self.k_proj(x)
            values = self.v_proj(x)

            #(...,seq_len, d_model) 
            #->(..., seq_len, num_heads, head_dim)

            queries = rearrange(
                queries,
                "... seq (heads heads_dim) -> ... heads seq heads heads dim",
                heads = self.heads,
            )

            keys = rearrange(
                 keys,
                "... seq (heads head_dim) -> ... heads seq head_dim",
                 heads=self.num_heads,
            )
            values = rearrange(
                values,
                "... seq (heads head_dim) -> ... heads seq head_dim",
                heads=self.num_heads,
            )
            # 下三角因果 mask
            # shape: (seq_len, seq_len)
            causal_mask = torch.tril(
                torch.ones(
                    seq_len,
                    seq_len,
                    dtype=torch.bool,
                    device=x.device,
                )
            )
            # (..., num_heads, seq_len, head_dim)
            # -> (..., seq_len, d_model)
            attention_output = rearrange(
                attention_output,
                "... num_heads seq_len head_dim -> ... seq (heads head_dim)" 
            )


            return self.output_proj(attention_output)
