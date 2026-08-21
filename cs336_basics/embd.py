import math

import torch
from torch import Tensor, nn

class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        # 初始化父类
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype
            )
        )

         # 原地初始化weight
        torch.nn.init.trunc_normal_(
            tensor=self.weight,
            mean=0.0, 
            std=1.0,
            a=-3,
            b=3,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
         return self.weight[token_ids]
        
    

