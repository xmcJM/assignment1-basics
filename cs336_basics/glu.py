import torch
from torch import Tensor, nn
from linear import Linear

class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device | None=None,
            dtype: torch.dtype | None = None, 
            )->None:
        
        super().__init__()

        if d_ff is None:
            d_ff = 64* math.ceil(
                (8 /3 * d_model) / 64
            )

        self.d_model = d_model
        self.d_ff = d_ff

        # w1 w3并行分支

        self.w1 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype
        )

        self.w3 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype
        )

        # w2中间维度投影回模型维度
        self.w3 = Linear(
            in_features=d_ff,
            out_features=d_model,
            device=device,
            dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        w1_output = self.w1(x)
        w3_output = self.w3(x)

        #SiLU(x) = x * sigmod(x)
        sliu_output = w1_output * torch.sigmoid(w1_output)

        # 逐元相乘
        gated = sliu_output * w3_output

        return self.w2(gated)