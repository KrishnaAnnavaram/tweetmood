"""Frozen-encoder hybrids (requires torch): token states from a FROZEN embedder, a trained head on top.

The prototype froze BERT but trained the new heads with lr 2e-5 (a fine-tuning rate) for 3 epochs,
so the heads were under-trained. Here the encoder states are computed once (the encoder is frozen,
so they never change), and only the head trains, with a head learning rate (1e-3), up to 15 epochs
and early stopping on validation macro-F1.

Embedders:

* ``hash`` - a fixed random vector per token (from a hash of the token). Offline, no context.
* ``hf``   - the last hidden states of a frozen Hugging Face model (needs ``transformers``).
"""

from __future__ import annotations

import copy
import json
import zlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..evaluate import macro_f1
from ..text.normalize import Normalizer
from .base import write_meta


class HashEmbedder:
    def __init__(self, dim: int = 64, max_len: int = 64):
        self.dim = dim
        self.max_len = max_len
        self._cache: dict[str, np.ndarray] = {}

    def spec(self) -> dict:
        return {"kind": "hash", "dim": self.dim, "max_len": self.max_len}

    def _vec(self, tok: str) -> np.ndarray:
        if tok not in self._cache:
            rng = np.random.default_rng(zlib.crc32(tok.encode("utf-8")))
            self._cache[tok] = rng.normal(0, 1 / np.sqrt(self.dim), self.dim).astype(np.float32)
        return self._cache[tok]

    def embed(self, texts) -> list[torch.Tensor]:
        out = []
        for t in texts:
            toks = t.split()[: self.max_len] or ["<empty>"]
            out.append(torch.from_numpy(np.stack([self._vec(tok) for tok in toks])))
        return out


class HFEmbedder:
    def __init__(self, model_name: str, max_len: int = 64, local_files_only: bool = False, device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer

        self.model_name, self.max_len, self.device = model_name, max_len, device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only).to(device).eval()
        self.dim = int(self.model.config.hidden_size)

    def spec(self) -> dict:
        return {"kind": "hf", "model_name": self.model_name, "max_len": self.max_len}

    @torch.no_grad()
    def embed(self, texts, batch_size: int = 32) -> list[torch.Tensor]:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(list(texts[i : i + batch_size]), truncation=True, max_length=self.max_len,
                                 padding=True, return_tensors="pt").to(self.device)
            states = self.model(**enc).last_hidden_state.cpu()
            for row, n in zip(states, enc["attention_mask"].sum(dim=1).tolist()):
                out.append(row[:n].clone())
        return out


def make_embedder(spec: dict, device: str = "cpu"):
    if spec["kind"] == "hash":
        return HashEmbedder(dim=spec["dim"], max_len=spec["max_len"])
    if spec["kind"] == "hf":
        return HFEmbedder(spec["model_name"], max_len=spec["max_len"], local_files_only=spec.get("offline", False),
                          device=device)
    raise ValueError(f"unknown embedder {spec['kind']!r}")


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(1) / m.sum(1).clamp(min=1)


class BiLSTMHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, 2)

    def forward(self, x, mask):
        lengths = mask.sum(1).clamp(min=1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        y, _ = self.lstm(packed)
        y, _ = pad_packed_sequence(y, batch_first=True, total_length=x.size(1))
        return self.out(self.drop(_masked_mean(y, mask)))


class TransformerHead(nn.Module):
    def __init__(self, dim: int, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        heads = 8 if dim % 8 == 0 else 1
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=2 * dim, dropout=dropout, batch_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(dim, 2)

    def forward(self, x, mask):
        return self.out(_masked_mean(self.body(x, src_key_padding_mask=~mask), mask))


HEADS = {"bilstm": BiLSTMHead, "transformer": TransformerHead}


def _batch(states: list[torch.Tensor], idx) -> tuple[torch.Tensor, torch.Tensor]:
    chunk = [states[i] for i in idx]
    width = max(s.size(0) for s in chunk)
    x = torch.zeros(len(chunk), width, chunk[0].size(1))
    mask = torch.zeros(len(chunk), width, dtype=torch.bool)
    for r, s in enumerate(chunk):
        x[r, : s.size(0)] = s
        mask[r, : s.size(0)] = True
    return x, mask


class HybridModel:
    def __init__(self, head: str = "bilstm", seed: int = 42, embedder: dict | None = None, lr: float = 1e-3,
                 epochs: int = 15, patience: int = 3, batch_size: int = 64, normalizer: Normalizer | None = None,
                 device: str = "cpu"):
        if head not in HEADS:
            raise ValueError(f"unknown head {head!r}")
        self.name = f"hybrid-{head}"
        self.head_kind, self.seed, self.lr = head, seed, lr
        self.epochs, self.patience, self.batch_size, self.device = epochs, patience, batch_size, device
        self.embedder_spec = embedder or {"kind": "hash", "dim": 64, "max_len": 64}
        self.normalizer = normalizer or Normalizer()
        self._embedder = None
        self.head: nn.Module | None = None
        self.history: list[float] = []
        self.best_epoch = -1

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = make_embedder(self.embedder_spec, self.device)
        return self._embedder

    def _states(self, texts):
        return self.embedder.embed([self.normalizer(t) for t in texts])

    def _proba(self, states) -> np.ndarray:
        self.head.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(states), 256):
                x, m = _batch(states, range(i, min(i + 256, len(states))))
                out.append(torch.softmax(self.head(x.to(self.device), m.to(self.device)), -1).cpu().numpy())
        return np.concatenate(out)

    def fit(self, train_texts, train_y, val_texts, val_y) -> "HybridModel":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        tr, va = self._states(list(train_texts)), self._states(list(val_texts))
        y_tr = torch.tensor(np.asarray(train_y), dtype=torch.long)
        y_va = np.asarray(val_y)
        self.head = HEADS[self.head_kind](self.embedder.dim).to(self.device)
        opt = torch.optim.Adam(self.head.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        best_score, best_state, bad = -1.0, None, 0
        for epoch in range(self.epochs):
            self.head.train()
            order = rng.permutation(len(tr))
            for i in range(0, len(order), self.batch_size):
                idx = order[i : i + self.batch_size]
                x, m = _batch(tr, idx)
                opt.zero_grad()
                loss = loss_fn(self.head(x.to(self.device), m.to(self.device)), y_tr[idx].to(self.device))
                loss.backward()
                nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
                opt.step()
            score = macro_f1(y_va, self._proba(va).argmax(1))
            self.history.append(score)
            if score > best_score:
                best_score, best_state, bad, self.best_epoch = score, copy.deepcopy(self.head.state_dict()), 0, epoch
            else:
                bad += 1
                if bad > self.patience:
                    break
        self.head.load_state_dict(best_state)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        if self.head is None:
            raise RuntimeError("fit the model first")
        return self._proba(self._states(list(texts)))

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        write_meta(folder, self.name, {"head": self.head_kind, "seed": self.seed, "embedder": self.embedder_spec,
                                       "lr": self.lr, "history": self.history, "best_epoch": self.best_epoch,
                                       "dim": self.embedder.dim})
        torch.save(self.head.state_dict(), folder / "head.pt")

    @classmethod
    def load(cls, folder: Path) -> "HybridModel":
        meta = json.loads((Path(folder) / "model.json").read_text(encoding="utf-8"))
        model = cls(head=meta["head"], seed=meta["seed"], embedder=meta["embedder"], lr=meta["lr"])
        model.head = HEADS[meta["head"]](meta["dim"])
        model.head.load_state_dict(torch.load(Path(folder) / "head.pt", map_location="cpu", weights_only=True))
        model.history, model.best_epoch = meta["history"], meta["best_epoch"]
        return model
