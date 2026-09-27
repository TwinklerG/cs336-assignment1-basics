import os
import json
from collections.abc import Iterable, Iterator
import regex as re

from cs336_basics.utils import PAT


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

    def from_files(
        self,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        self.vocab = json.load(open(vocab_filepath))
        self.merges = []
        with open(merges_filepath) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                a, b = line.strip().split()
                self.merges.append((a.encode(), b.encode()))
        self.special_tokens = special_tokens or []

    def encode(self, text: str) -> list[int]:
        pattern = re.compile(
            "(" + "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)) + ")"
        )
        ret: list[int] = []
        vocab = {v: k for k, v in self.vocab.items()}
        segments = pattern.split(text) if self.special_tokens else [text]
        for mth in segments:
            if mth in self.special_tokens:
                ret.append(vocab[mth.encode()])
                continue
            for chunk in re.finditer(PAT, mth):
                byte_list = [bytes([b]) for b in chunk.group(0).encode()]
                for merge in self.merges:
                    i = 0
                    new_byte_list: list[bytes] = []
                    while i < len(byte_list):
                        if i < len(byte_list) - 1 and byte_list[i] == merge[0] and byte_list[i + 1] == merge[1]:
                            new_byte_list.append(merge[0] + merge[1])
                            i += 2
                        else:
                            new_byte_list.append(byte_list[i])
                            i += 1
                    byte_list = new_byte_list
                ret.extend([vocab[b] for b in byte_list])
        return ret

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[i] for i in ids]).decode(errors="replace")
