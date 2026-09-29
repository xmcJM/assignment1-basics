import torch
from torch import Tensor

def cross_entropy(
    logits: Tensor,
    targets: Tensor,
) -> Tensor:
   # logits:  (batch_size, m, vocab_size)
   # targets: (batch_size, m)

   # 1. 沿vocab_size取最大值
   # shape: (..., 1)
   max_logits = logits.max(
      dim=-1,
      keepdim=True,
   ).values

   # 2. 平移logits，避免exp溢出
   shifted_logits = logits - max_logits

   # 3. 计算稳定的log-sum-exp
   # shape: (..., 1)
   log_sum_exp = (
      torch.log(
         torch.exp(shifted_logits).sum(
            dim=-1,
            keepdim=True,
         )
      )
      + max_logits
   )

   # 4. 取出每个位置真实 token 对应的 logit
   # targets[..., None] 将形状从 (...) 变成 (..., 1)
   target_logits = logits.gather(
      dim=-1,
      index=targets[..., None],
   )

   # 5. 每个位置的交叉熵
   # # shape: (..., 1)
   losses = log_sum_exp - target_logits

   # 6. 对所有位置求平均，返回标量
   return losses.mean()