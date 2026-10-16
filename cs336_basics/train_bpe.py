import os, heapq
import regex as re
from typing import BinaryIO
from collections import Counter, defaultdict
from multiprocessing import Pool
from cs336_basics.utils import log_time

def find_chunk_boundaries(
    file: BinaryIO,
    desired_chunk_size: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # Initial guesses for chunk boundary locations, spaced by desired_chunk_size
    # Chunks start on previous index, don't include last index
    chunk_boundaries = list(range(0, file_size, desired_chunk_size)) + [file_size]

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, chunks may be larger than desired_chunk_size
    return sorted(set(chunk_boundaries))

def _worker(args: tuple[str, int, int, list[str]]) -> Counter[str]:
    input_path, start, end, special_tokens = args
    special_pattern = "|".join([re.escape(tok) for tok in sorted(special_tokens, key = len, reverse = True)])
    pre_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    with open(input_path, 'rb') as f:
        f.seek(start)
        text = f.read(end - start).decode('utf-8', errors = "ignore")
        parts = re.split(special_pattern, text) if len(special_tokens) > 0 else [text]
        word_cnts = Counter()
        for part in parts:
            for match in re.finditer(pre_pattern, part):
                word = match.group(0)
                word_cnts[word] += 1
    return word_cnts

@log_time
def pre_tokenization(
        input_path: str, 
        special_tokens: list[str],
        num_pros: int | None = None,
        chunk_size: int = 10*1024*1024
) -> Counter[str]:
    with open(input_path, 'rb') as f:
        split_special_token = '\n'
        if len(special_tokens) > 0:
            split_special_token = special_tokens[0]
        bounds = find_chunk_boundaries(f, chunk_size, split_special_token.encode('utf-8'))
    
    tasks = [(input_path, l, r, special_tokens) for l, r in zip(bounds[:-1], bounds[1:])]
    num_pros = min(num_pros or os.cpu_count(), len(tasks))
    word_cnts: Counter[str] = Counter()
    with Pool(num_pros) as pool:
        for cnter in pool.imap_unordered(_worker, tasks):
            word_cnts.update(cnter)
    return word_cnts

class _RevPair:
    __slots__ = ("pair",)
    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair
    def __lt__(self, other: "_RevPair"):
        return self.pair > other.pair
    def __eq__(self, other: "_RevPair"):
        return self.pair == other.pair

@log_time
def merge_bpe(
        word_cnts: Counter[str],
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # init merge
    vocab = [tok.encode('utf-8') for tok in special_tokens] + [bytes([i]) for i in range(256)]
    merges: list[tuple[bytes, bytes]] = []

    pair_cnt: Counter[tuple[bytes, bytes]] = Counter()
    pair2words: defaultdict[tuple[bytes, bytes], set[str]] = defaultdict(set)
    word2info:  dict[str, tuple[list[bytes], int]] = {}

    for word_str, freq in word_cnts.items():
        tokens = [bytes([i]) for i in word_str.encode('utf-8')]
        word2info[word_str] = (tokens, freq)
        for pair in zip(tokens[:-1], tokens[1:]):
            pair_cnt[pair] += freq
            pair2words[pair].add(word_str)

    # merge
    hq = [(-c, _RevPair(p)) for p, c in pair_cnt.items()]
    heapq.heapify(hq)
    while len(vocab) < vocab_size:
        # get next pair to merge
        # best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        # optimize to heapq
        best = None
        while hq:
            neg_freq, rp = heapq.heappop(hq)
            if pair_cnt.get(rp.pair, 0) == -neg_freq:
                best = rp.pair
                break
        if best is None: break

        new_tok = best[0] + best[1]
        merges.append(best)
        vocab.append(new_tok)

        # combine, update affected
        affected_words = pair2words.get(best, set()).copy()
        cnt_changed = set()
        for word_str in affected_words:
            tokens, freq = word2info[word_str]

            # remove old tokens related
            for pair in zip(tokens[:-1], tokens[1:]):
                cnt_changed.add(pair)
                pair_cnt[pair] -= freq
                pair2words[pair].discard(word_str)
                if pair_cnt[pair] == 0:
                    del pair_cnt[pair]
                    del pair2words[pair]

            # combine best pair get new token
            i = 0
            new_toks = []
            while i < len(tokens):
                if i < len(tokens)-1 and best == (tokens[i], tokens[i+1]):
                    new_toks.append(new_tok)
                    i += 2
                else:
                    new_toks.append(tokens[i])
                    i += 1

            # update new tokens related
            word2info[word_str] = (new_toks, freq)
            for pair in zip(new_toks[:-1], new_toks[1:]):
                cnt_changed.add(pair)
                pair_cnt[pair] += freq
                pair2words[pair].add(word_str)
        
        # update hq
        for pair in cnt_changed:
            c = pair_cnt.get(pair, 0)
            if c > 0:
                heapq.heappush(hq, (-c, _RevPair(pair)))
    
    return {i: tok for i, tok in enumerate(vocab)}, merges

@log_time
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    word_cnts = pre_tokenization(input_path, special_tokens)
    vocab, merges = merge_bpe(word_cnts, vocab_size, special_tokens)
    return vocab, merges



if __name__ == "__main__":
    import logging, pickle

    logging.basicConfig(
        level = logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        )
    print(f"cpu_count: {os.cpu_count()}")

    vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    with open('data/vocab/TinyStoriesV2-vocab.txt', 'w', encoding = 'utf-8') as f:
        for i, tok in vocab.items():
            f.write(f"{i}\t{tok}\n")
    with open('data/vocab/TinyStoriesV2-merges.txt', 'w', encoding = 'utf-8') as f:
        for l, r in merges:
            f.write(f"{l}\t{r}\n")
    with open('data/vocab/TinyStoriesV2-vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open('data/vocab/TinyStoriesV2-merges.pkl', 'wb') as f:
        pickle.dump(merges, f)

    # vocab, merges = train_bpe("data/owt_train.txt", 32000, ["<|endoftext|>"])
    # with open('data/vocab/owt-vocab.txt', 'w', encoding = 'utf-8') as f:
    #     for i, tok in vocab.items():
    #         f.write(f"{i}\t{tok}\n")
    # with open('data/vocab/owt-merges.txt', 'w', encoding = 'utf-8') as f:
    #     for l, r in merges:
    #         f.write(f"{l}\t{r}\n")
    # with open('data/vocab/owt-vocab.pkl', 'wb') as f:
    #     pickle.dump(vocab, f)
    # with open('data/vocab/owt-merges.pkl', 'wb') as f:
    #     pickle.dump(merges, f)

    