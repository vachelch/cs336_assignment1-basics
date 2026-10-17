import regex as re
import functools
import pickle, os
from typing import Iterable, Iterator
from cs336_basics.utils import log_time
from cs336_basics.train_bpe import find_chunk_boundaries
import numpy as np

class Tokenizer:
    """
    A BPE tokenizer. Given a vocabulary and a list of merges,
    encodes text into integer IDs and decodes integer IDs into text.
    """
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """
        Construct a tokenizer from a given vocabulary, list of merges,
        and (optionally) a list of special tokens
        """
        self.re_special = None
        self.special_token_set = set(special_tokens) if special_tokens else set()
        if special_tokens:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            pat = "(" + "|".join([re.escape(tok) for tok in special_tokens]) + ")"
            self.re_special = re.compile(pat)
        self.re_pre_tokenize = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.vocab: dict[int, bytes] = vocab
        self.pair2rank: dict[tuple[bytes, bytes], int] = {pair:i for i, pair in enumerate(merges)}
        self.bytes2id: dict[bytes, int] = {bs:i for i, bs in vocab.items()}

    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary
        and list of merges (in the same format that your BPE training code output) and
        (optionally) a list of special tokens.
        """
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    @functools.lru_cache(maxsize=50000)
    def _bpe(self, word: str) -> list[int]:
        tokens = [bytes([b]) for b in word.encode('utf-8')]

        while len(tokens) > 1:
            # find best pair to merge
            best, best_rk = None, None
            for pair in zip(tokens[:-1], tokens[1:]):
                rk = self.pair2rank.get(pair)
                if rk is None: continue
                if best_rk is None or rk < best_rk:
                    best, best_rk = pair, rk
            if best is None: break

            # merge pair get new tokens
            new_tokens = []
            new_tok = best[0] + best[1]
            i = 0
            while i < len(tokens):
                if i == len(tokens)-1 or (tokens[i], tokens[i+1]) != best:
                    new_tokens.append(tokens[i])
                    i += 1
                else:
                    new_tokens.append(new_tok)
                    i += 2
            tokens = new_tokens
        return [self.bytes2id[tok] for tok in tokens]

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        segments = self.re_special.split(text) if self.re_special else [text]

        ids = []
        for segment in segments:
            if not segment: continue
            if segment in self.special_token_set:
                ids.append(self.bytes2id[segment.encode('utf-8')])
                continue
            for match in self.re_pre_tokenize.finditer(segment):
                ids.extend(self._bpe(match.group()))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs. 
        
        This is required for memory-efficient tokenization of large files 
        that we cannot directly load into memory.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        return b"".join([self.vocab[i] for i in ids]).decode("utf-8", errors = "replace")



def sample_docs(
    path: str, 
    special_token: str = "<|endoftext|>",
    read_bytes: int = 20 * 1024 * 1024, 
    sample_num: int = 20
) -> list[str]:
    import random
    SEED = 0

    with open(path, 'rb') as f:
        text = f.read(read_bytes).decode("utf-8", errors = "replace")
    docs = text.split(special_token)[:-1]
    docs = [d for d in docs if d]

    return random.Random(SEED).sample(docs, sample_num)

def compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> float:
    n_bytes = sum([len(doc.encode('utf-8')) for doc in docs])
    n_tokens =  sum([len(tokenizer.encode(doc)) for doc in docs])
    return n_bytes / n_tokens

def run_throughput(tokenizer: Tokenizer, path: str, n_bytes:int = 10 * 2**20) -> None:
    import time

    # show encode & decode case
    original_text = "Hello, world!<|endoftext|>"
    token_ids = tokenizer.encode(original_text)
    print("original_text:", original_text)
    print("Token IDs:", token_ids)
    print("Tokens:", [tokenizer.vocab[i] for i in token_ids])
    print("Decode Text:", tokenizer.decode(token_ids))

    # throughput
    with open(path, 'rb') as f:
        text = f.read(n_bytes).decode('utf-8', errors = "replace")
    n_bytes = len(text.encode('utf-8'))

    t0 = time.perf_counter()
    ids = tokenizer.encode(text)
    t1 = time.perf_counter()
    tput = n_bytes / (t1 - t0)

    pile_gb = 825
    pile_days = pile_gb * 1e9 / tput / (24*3600)
    print(f"throughput {tput / 1e6:.2f} MB/s -> Pile({pile_gb}GB) = {pile_days:.2f} days (single process)")

_tok = None
def _init_worker(tokenizer: Tokenizer) -> None:
    global _tok
    _tok = tokenizer

def _worker(args) -> tuple[int, bytes]:
    in_file, l, r = args
    with open(in_file, "rb") as f:
        f.seek(l)
        text = f.read(r-l).decode('utf-8', errors = "replace")
    ids = _tok.encode(text)
    ids_bytes = np.asarray(ids, dtype=np.uint16).tobytes()
    return r-l, ids_bytes

@log_time
def run_encode_and_save(
    tokenizer: Tokenizer,
    in_file: str, 
    out_file: str, 
    special_token: str = "<|endoftext|>",
    num_procs: int | None = None,
) -> None:
    assert max(tokenizer.vocab) < 2**16

    import tqdm
    from multiprocessing import Pool

    with open(in_file, 'rb') as f:
        bounds = find_chunk_boundaries(f, special_token.encode('utf-8'), desired_chunk_size=5*1024*1024)
    tasks = [(in_file, l, r) for l, r in zip(bounds[:-1], bounds[1:])]
    num_procs = min(num_procs or os.cpu_count(), len(tasks))
    total_bytes = os.path.getsize(in_file)

    with open(out_file, "wb") as out, \
        Pool(num_procs, initializer = _init_worker, initargs = (tokenizer,)) as pool, \
        tqdm.tqdm(total = total_bytes, unit = "B", unit_scale = True, desc = "Encoding") as pbar:

        for nbytes, ids_bytes in pool.imap(_worker, tasks, chunksize=1):
            out.write(ids_bytes)
            pbar.update(nbytes)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level = logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        )

    # vocab_filepath = "data/vocab/owt-vocab.pkl"
    # merges_filepath = "data/vocab/owt-merges.pkl"
    # in_file = "data/owt_train.txt"
    # out_file = "data/tokenized/owt_train_ids.bin"

    vocab_filepath = "data/vocab/TinyStoriesV2-vocab.pkl"
    merges_filepath = "data/vocab/TinyStoriesV2-merges.pkl"
    in_file = "data/TinyStoriesV2-GPT4-train.txt"
    out_file = "data/tokenized/TinyStoriesV2_train_ids.bin"

    tokenizer = Tokenizer.from_files(
        vocab_filepath = vocab_filepath,
        merges_filepath = merges_filepath,
        special_tokens = ["<|endoftext|>"],
    )

    docs = sample_docs(in_file)
    ratio = compression_ratio(tokenizer, docs)
    print(f"compression_ratio: {ratio:.2f} bytes/token")

    run_throughput(tokenizer, in_file)

    run_encode_and_save(tokenizer, in_file, out_file)