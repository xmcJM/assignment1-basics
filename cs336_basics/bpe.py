import os
import regex
from collections import Counter


GPT2_PATTERN = regex.compile(
    r"'(?:[sdmt]|ll|ve|re)"
    r"| ?\p{L}+"
    r"| ?\p{N}+"
    r"| ?[^\s\p{L}\p{N}]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. 初始化字节词表
    vocab: dict[int, bytes] = {
        token_id: bytes([token_id])
        for token_id in range(256)
    }

    # 2. 按传入顺序加入特殊 token
    for token_id, special_token in enumerate(
        special_tokens,
        start=256,
    ):
        vocab[token_id] = special_token.encode("utf-8")

    # 检查初始词表大小
    if len(vocab) > vocab_size:
        raise ValueError(
            f"vocab_size 太小：至少需要 {len(vocab)}，"
            f"但传入的是 {vocab_size}"
        )

    # 3. 读取训练文本
    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read()

    # 4. 隔离特殊 token
    special_set = set(special_tokens)

    special_pattern = "|".join(
        regex.escape(token)
        for token in sorted(
            special_tokens,
            key=lambda token: (-len(token), token),
        )
    )

    if special_pattern:
        parts = regex.split(
            f"({special_pattern})",
            text,
        )
    else:
        parts = [text]

    # 5. 统计 pre-token 出现次数
    pretoken_counts: Counter[str] = Counter()

    for part in parts:
        # 跳过空片段和特殊 token
        if not part or part in special_set:
            continue

        for match in GPT2_PATTERN.finditer(part):
            pretoken = match.group(0)
            pretoken_counts[pretoken] += 1

    # 6. 将 pre-token 转成单字节 token 序列
    word_frequencies: Counter[tuple[bytes, ...]] = Counter()

    for pretoken, count in pretoken_counts.items():
        encoded = pretoken.encode("utf-8")

        byte_tokens = tuple(
            bytes([byte_value])
            for byte_value in encoded
        )

        word_frequencies[byte_tokens] += count

    # 7. 逐轮进行 BPE 合并
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        # 每一轮都重新统计 pair，不能放在循环外
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()

        for token_sequence, count in word_frequencies.items():
            if len(token_sequence) < 2:
                continue

            for first_token, second_token in zip(
                token_sequence,
                token_sequence[1:],
            ):
                pair_counts[(first_token, second_token)] += count

        # 没有可合并的 pair
        if not pair_counts:
            break

        # 频次最高优先；
        # 频次相同时选择字典序更大的 pair
        best_pair = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], pair),
        )

        first_token, second_token = best_pair
        new_token = first_token + second_token

        # 记录 merge 规则
        merges.append(best_pair)

        # 加入新 token
        vocab[len(vocab)] = new_token

        # 合并所有 token sequence
        new_word_frequencies: Counter[tuple[bytes, ...]] = Counter()

        for token_sequence, count in word_frequencies.items():
            merged_sequence: list[bytes] = []

            index = 0

            while index < len(token_sequence):
                if (
                    index + 1 < len(token_sequence)
                    and token_sequence[index] == first_token
                    and token_sequence[index + 1] == second_token
                ):
                    # 合并两个 token
                    merged_sequence.append(new_token)
                    index += 2
                else:
                    # 保留当前 token
                    merged_sequence.append(token_sequence[index])
                    index += 1

            # 如果合并后序列相同，Counter 会自动累加频次
            new_word_frequencies[
                tuple(merged_sequence)
            ] += count

        word_frequencies = new_word_frequencies

    return vocab, merges