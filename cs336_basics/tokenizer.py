from collections.abc import Iterable, Iterator
from typing import Self
import regex
import pickle

GPT2_PATTERN = regex.compile(
    r"'(?:[sdmt]|ll|ve|re)"
    r"| ?\p{L}+"
    r"| ?\p{N}+"
    r"| ?[^\s\p{L}\p{N}]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int,bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens)

        self.token_to_id: dict[bytes, int] = {}

        for token_id, token_bytes in self.vocab.items():
            # 如果bytes重复，保留第一次的id
            if token_bytes not in self.token_to_id:
                self.token_to_id[token_bytes] = token_id

        # merge pair -> 越早出现优先级越高
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {}

        for priority, pair in enumerate(self.merges):
            self.merge_ranks.setdefault(pair, priority)

        # 特殊token -> token id
        self.special_token_to_id: dict[str, int] = {}

        # 当前最大tokenid +1
        next_token_id = max(self.vocab, default=-1) +1

        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")
            # 如果存在使用旧id
            if special_bytes in self.token_to_id:
                token_id = self.token_to_id[special_bytes]
            else:
                token_id = next_token_id
                next_token_id+=1

                self.vocab[token_id] = special_bytes 
                self.token_to_id[special_bytes] = token_id
            self.special_token_to_id[special_token] = token_id

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Self:
        # 读取 vocab
        with open(vocab_filepath, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)

        # 读取 merges
        with open(merges_filepath, "rb") as merges_file:
            merges = pickle.load(merges_file)

        # 使用读取出的数据构造 Tokenizer
        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )

    def _encode_pretoken(self, pretoken: str)->list[int]:
        # 1. 编码成utt8
        encoded = pretoken.encode("utf-8")

        # 2. 每个字节转为单字节token
        tokens: list[bytes] = [
            bytes([byte_value])
            for byte_value in encoded
        ]

        
        while len(tokens) >=2 :
            # 3. 查看当前序列所有相邻pair
            available_pairs: list[tuple[int, tuple[bytes,bytes]]] =[]
            for first_token, second_token in zip(
                tokens,
                tokens[1:],
            ):
                pair = (first_token, second_token)
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    available_pairs.append((rank,pair))
            # 4. 没有可用的停止合并
            if not available_pairs:
                break
            # 5. 选择rank最小的pair
            _, best_pair = min(
                available_pairs,
                key=lambda item: item[0],
            )

            first_token,second_token = best_pair
            merged_token = first_token + second_token

            # 6. 从左到右，非重叠全部合并
            new_tokens: list[bytes] = []
            idx = 0;
            while idx < len(tokens):
                if (
                    idx+1< len(tokens)
                    and tokens[idx] == first_token
                    and tokens[idx+1] == second_token
                ):
                    new_tokens.append(merged_token)
                    idx+=2
                else:
                    new_tokens.append(tokens[idx])
                    idx+=1
            tokens = new_tokens
        # 7. 将最终bytes token转为整数id
        return [
            self.token_to_id[token]
             for token in tokens
        ]


    def encode(self, text: str) -> list[int]:
        special_pattern = "|".join(
            regex.escape(token)
            for token in sorted(
                self.special_tokens,
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

        token_ids: list[int] = []

        for part in parts:
            if not part:
                continue

            # 当前片段是特殊 token
            if part in self.special_token_to_id:
                token_ids.append(
                    self.special_token_to_id[part]
                )
                continue

            # 普通文本使用 GPT-2 regex
            for match in GPT2_PATTERN.finditer(part):
                pretoken = match.group(0)

                # 将多个 ID 加入同一个列表
                token_ids.extend(
                    self._encode_pretoken(pretoken)
                )

        # 必须返回编码结果
        return token_ids


    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        for text_part in iterable:
            token_ids = self.encode(text_part)
            for token_id in token_ids:
                yield token_id


    def decode(self, ids: list[int]) -> str:
        encoded = b"".join(
            self.vocab[token_id]
            for token_id in ids
        )
        return encoded.decode(
            "utf-8",
            errors="replace"
        )