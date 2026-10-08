import collections
import random
import re
import math

import torch
from torch import nn
from torch.nn import functional as F
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

def tokenize(lines, token='word'):
    """Split text lines into word or character tokens."""
    if token == 'word':
        return [line.split() for line in lines]
    if token == 'char':
        return [list(line) for line in lines]
    raise ValueError(f'Unknown token type: {token}')


def load_corpus_time_machine(max_tokens=-1):
    """Return token indices and the vocabulary of the Time Machine dataset."""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)

    # Flatten all lines into one token sequence.
    corpus = [vocab[token] for line in tokens for token in line]

    if max_tokens > 0:
        corpus = corpus[:max_tokens]

    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_steps):
    """Generate minibatches by random sampling."""
    corpus = corpus[random.randint(0, num_steps - 1):]

    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(
        range(0, num_subseqs * num_steps, num_steps)
    )
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size

    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """Generate minibatches by sequential partitioning."""
    offset = random.randint(0, num_steps)

    num_tokens = (
        (len(corpus) - offset - 1) // batch_size
    ) * batch_size

    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])

    Xs = Xs.reshape(batch_size, -1)
    Ys = Ys.reshape(batch_size, -1)

    num_batches = Xs.shape[1] // num_steps

    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


class SeqDataLoader:
    """Iterator for loading sequence data."""
    def __init__(
        self,
        batch_size,
        num_steps,
        use_random_iter=False,
        max_tokens=10000
    ):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential

        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size = batch_size
        self.num_steps = num_steps

    def __iter__(self):
        return self.data_iter_fn(
            self.corpus,
            self.batch_size,
            self.num_steps
        )


def load_data_time_machine(
    batch_size,
    num_steps,
    use_random_iter=False,
    max_tokens=10000
):
    """Return the Time Machine data iterator and vocabulary."""
    data_iter = SeqDataLoader(
        batch_size,
        num_steps,
        use_random_iter,
        max_tokens
    )
    return data_iter, data_iter.vocab


class RNNModelScratch:
    """A recurrent neural network model implemented from scratch."""

    def __init__(
        self,
        vocab_size,
        num_hiddens,
        device,
        get_params,
        init_state,
        forward_fn
    ):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state = init_state
        self.forward_fn = forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


# RNN utilities

def predict_ch8(prefix, num_preds, net, vocab, device):
    """Generate new characters following prefix."""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]

    def get_input():
        return torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    # Warm up the hidden state using the supplied prefix.
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])

    # Generate one character at a time.
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))

    return ''.join(vocab.idx_to_token[i] for i in outputs)


def grad_clipping(net, theta):
    """Clip gradients so their global L2 norm does not exceed theta."""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params

    params_with_grad = [p for p in params if p.grad is not None]
    if not params_with_grad:
        return

    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params_with_grad))
    if norm > theta:
        for param in params_with_grad:
            param.grad[:] *= theta / norm


def train_epoch_ch8(net, train_iter, loss, updater, device,
                    use_random_iter=False):
    """Train a recurrent model for one epoch."""
    state = None
    timer = d2l.Timer()
    metric = d2l.Accumulator(2)  # sum of losses, number of tokens

    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # Detach hidden state from the previous minibatch graph.
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()

        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()

        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)

        metric.add(l * y.numel(), y.numel())

    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """Train an RNN/GRU/LSTM language model from Chapter 8 of D2L."""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(
        xlabel='epoch', ylabel='perplexity',
        legend=['train'], xlim=[10, num_epochs]
    )

    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr=lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)

    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)

    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter
        )
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])

    print(f'perplexity {ppl:.1f}, {speed:.1f} token/s {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
