import html
import re
from collections import Counter

import torch
from torch import nn

PAD, UNK = "<pad>", "<unk>"
LABELS = {"negative": 0, "positive": 1}


def tokenize(text: str) -> list[str]:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.findall(r"[a-z']+", text.lower())


def build_vocab(texts: list[str], min_freq: int = 1) -> dict[str, int]:
    counts = Counter(token for text in texts for token in tokenize(text))
    return {PAD: 0, UNK: 1, **{word: i + 2 for i, (word, count) in enumerate(
        (item for item in counts.most_common() if item[1] >= min_freq)
    )}}


def encode(text: str, vocab: dict[str, int], max_length: int = 200) -> tuple[list[str], list[int]]:
    tokens = tokenize(text)[:max_length]
    ids = [vocab.get(token, vocab[UNK]) for token in tokens]
    return tokens, ids


class AttentionClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 48, hidden_dim: int = 48):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False)
        )
        self.classifier = nn.Linear(hidden_dim * 2, 2)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = token_ids.ne(0)
        encoded, _ = self.encoder(self.embedding(token_ids))
        scores = self.attention(encoded).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled), weights



