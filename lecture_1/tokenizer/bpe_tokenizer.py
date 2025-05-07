import regex as re
import collections
import psutil
import time
import os
from datasets import load_dataset

# Regex Pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Token helpers
def get_pair_freqs(tokenized_texts):
    pair_freqs = collections.Counter()
    for token in tokenized_texts:
        for i in range(len(token)-1):
            pair = (token[i], token[i+1])
            pair_freqs[pair] += 1
    return pair_freqs

# Pre-tokenization
def pre_tokenize_chunk(text, pattern):
    return [list(match.group().encode('utf-8')) for match in re.finditer(pattern, text)]

# def initialize_pair_tracking(tokenized_text):
#     """
#     Initialiazed the frequency and position pair tracking
#     :param tokenized_text:
#     :return:
#     """
#     pair_freqs = collections.Counter()
#     pair_positions = collections.defaultdict(list)
#
#     for idx, tokens in enumerate(tokenized_text):
#         for i in range(len(tokens)-1):
#             pair = (tokens[i], tokens[i+1])
#             pair_freqs[pair] += 1
#             pair_positions[pair].append((idx, i))
#
#     return pair_freqs, pair_positions



def train_bpe_tokenizer(input_path: str, vocab_size: int, special_tokens: list):
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
    # Read the input text file
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Remove and save special token
    escaped = [re.escape(token) for token in special_tokens]
    text_chunks = re.split('|'.join(escaped), text)

    # Pre-tokenize
    tokenized_texts = []
    for chunk in text_chunks:
        tokenized_texts.extend(pre_tokenize_chunk(chunk, PAT))

    # Initialize Vocabulary
    byte_vocab = {i: bytes([i]) for i in range(len(tokenized_texts))}
    for token in special_tokens:
        byte_vocab[max(byte_vocab.keys()) + 1] = byte_vocab[token]

    merges = []
    current_vocab_size = len(byte_vocab)

    while current_vocab_size < vocab_size:
        # Tracking pair
        pair_freqs = get_pair_freqs(tokenized_texts)
        if not pair_freqs:
            break # if there is no more pairs to merge

        # Get most frequent adjacent token pair
        most_common_pair, _ = max(pair_freqs.items(), key=lambda x: (x[1],x[0]))[0]
        # Form a new token by merging the pair
        new_token = most_common_pair[0] + most_common_pair[1]
        # Add pair into merges
        merges.append(most_common_pair)

        byte_vocab[max(byte_vocab.keys()) + 1] = new_token
        current_vocab_size += 1

    # Return updated vocabulary and merges
    return (byte_vocab,), merges

def main():
    # Initialize time and memory calculation
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024
    start_time = time.time()

    # Save dataset as file
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    input_file = "tinystories.txt"
    with open(input_file, 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(example['text'] + '\n')

    vocab_size = 10000
    special_tokens = ["<endoftext>"]

    vocab, merges = train_bpe_tokenizer(input_file, vocab_size, special_tokens)

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024

    # Outputting results
    print(f"Running time: {end_time - start_time: .2f} seconds")
    print(f"Memory usage: {mem_after - mem_before: .2f} MB")

    print("Vocab size: {}".format(len(vocab)))
    print("Sample vocab: ")
    for k, v in list(vocab[0].items())[:10]:
        print(f"ID: {k}, Token: {v}")

if __name__ == '__main__':
    main()
