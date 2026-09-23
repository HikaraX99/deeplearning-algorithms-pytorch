import collections
import re

from d2l import torch as d2l


# Time Machine dataset
d2l.DATA_HUB['time_machine'] = (
    d2l.DATA_URL + 'timemachine.txt',
    '090b5e7e70c295757f55df93cb0a180b9691891a'
)


def read_time_machine():
    """Load and preprocess the Time Machine dataset."""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()

    return [
        re.sub('[^A-Za-z]', ' ', line).strip().lower()
        for line in lines
    ]


def count_corpus(tokens):
    """Count token frequencies."""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a single token list.
        tokens = [token for line in tokens for token in line]

    return collections.Counter(tokens)


class Vocab:
    """Vocabulary for mapping tokens to indices."""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []

        if reserved_tokens is None:
            reserved_tokens = []

        # Count tokens and sort them by frequency.
        counter = count_corpus(tokens)
        self._token_freqs = sorted(
            counter.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Reserve index 0 for the unknown token.
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {
            token: idx
            for idx, token in enumerate(self.idx_to_token)
        }

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break

            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)

        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]

        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs