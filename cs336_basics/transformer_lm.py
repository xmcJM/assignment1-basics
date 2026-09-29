
import torch
from torch import Tensor, nn
from einops import rearrange

from linear import Linear
from rsnorm import RMSNorm
from multihead import MultiHeadSelfAttention
from glu import SwiGLU
from transform_block import TransformerBlock
from embd import Embedding

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        # 每次循环都会创建一个新的、权重独立的 Block
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=theta,
                max_seq_len=context_length,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: (batch_size, seq_len)
        seq_len = token_ids.shape[-1]

        if seq_len > self.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"context length {self.context_length}"
            )

        # (seq_len,) 供rope
        token_positions = torch.arange(
            seq_len,
            device=token_ids.device,
            dtype=torch.long,
        )

        # (batch, seq_len)
        # (batch, seq_len, d_model)
        x = self.token_embeddings(token_ids)

        # 每一层接收上一层输出
        for layer in self.layers:
            x = layer(
                x,
                token_positions = token_positions,
            )

        # 最终RMSNorm  (batch, seq_len, d_model)
        x = self.final_norm(x)

        # -> (batch, seq_len, vocab_size)
        logits = self.lm_head(x)

        return logits
        




