import os
from tqdm import tqdm
import regex as re
from collections import defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries
from joblib import Parallel, delayed

from cs336_basics.utils import PAT
from tests.common import FIXTURES_PATH


def _build_word2cnt(
    input_path: str | os.PathLike, special_tokens: list[str], num_workers: int = 4
) -> dict[tuple[bytes], int]:
    word2cnt: dict[tuple[bytes], int] = defaultdict(int)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 4, b"<|endoftext|>")
    tasks = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    def _count_chunk(task: tuple[str | os.PathLike, int, int, list[str]]) -> dict[tuple[bytes], int]:
        input_path, start, end, special_tokens = task
        word2cnt: dict[tuple[bytes], int] = defaultdict(int)
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks = re.compile("|".join(re.escape(token) for token in special_tokens)).split(chunk)
            for chunk in chunks:
                for match in re.finditer(PAT, chunk):
                    bts = match.group(0).encode()
                    word2cnt[tuple([bytes([b]) for b in bts])] += 1
        return word2cnt

    partial_counts = Parallel(n_jobs=num_workers)(delayed(_count_chunk)(task) for task in tasks)

    for partial_count in partial_counts:
        for word, count in partial_count.items():
            word2cnt[word] += count
    return word2cnt


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = defaultdict(bytes)
    for i in range(256):
        vocab[i] = bytes([i])
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    word2cnt: dict[tuple[bytes], int] = _build_word2cnt(input_path, special_tokens)

    merges: list[tuple[bytes, bytes]] = []

    for _ in tqdm(range(vocab_size - len(vocab)), desc="Training BPE"):
        pairs: dict[tuple[bytes, bytes], int] = defaultdict(int)
        for word, freq in word2cnt.items():
            symbols = word
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq

        to_merge = max(pairs, key=lambda pair: (pairs[pair], pair))
        merges.append((to_merge[0], to_merge[1]))
        vocab[len(vocab)] = to_merge[0] + to_merge[1]
        new_word2cnt: dict[tuple[bytes], int] = defaultdict(int)
        for word, freq in word2cnt.items():
            symbols = list(word)
            i = 0
            new_symbols: list[bytes] = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == to_merge[0] and symbols[i + 1] == to_merge[1]:
                    new_symbols.append(to_merge[0] + to_merge[1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_word2cnt[tuple(new_symbols)] += freq
        word2cnt = new_word2cnt

    return (vocab, merges)


if __name__ == "__main__":
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size=10000, special_tokens=special_tokens)
