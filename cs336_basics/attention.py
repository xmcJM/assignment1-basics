import math

import torch
from einops import einsum
from torch import Tensor
from softmax import softmax

def scaled_dot_product_attention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    d_k = queries.shape[-1]
    # (..., query_seq, d_k) * (..., keys_seq, d_k)
    # (..., query_seq, key_seq)
    anttention_sorces = einsum(
        queries,
        keys,
        "... query d_k, ... key d_k -> ... query key",
    )

    # 缩放点积
    anttention_sorces = anttention_sorces /  math.sqrt(d_k)

    if mask is not None:
        anttention_sorces = anttention_sorces.masked_fill(
            ~mask,
            float("-inf"),
        )

    # 每个 query 在所有 key 上形成概率分布
    anttention_probilities = softmax(
        anttention_sorces,
        dim = -1,
    )

    # (..., query_seq, key_seq) * (..., key_seq, d_v)
    # (..., query_seq, d_v)
    output = einsum(
            anttention_probilities,
            values,
            "... query key, ... key d_v -> ... query d_v",
        )
    return output