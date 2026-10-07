"""Full fine-tuning of a Twitter-domain transformer (requires torch and transformers).

Default model: ``cardiffnlp/twitter-roberta-base`` (set ``TWEETMOOD_HF_MODEL`` for BERTweet or another).
All weights train with lr 2e-5, linear decay, early stopping on validation macro-F1 and a deep copy
of the best state. The text goes through the shared normalizer with emoji naming OFF (the tokenizer
of a Twitter model knows emoji) and the slang map ON.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from ..evaluate import macro_f1
from ..text.normalize import Normalizer
from .base import write_meta


class FineTuneModel:
    name = "finetune"

    def __init__(self, seed: int = 42, model_name: str = "cardiffnlp/twitter-roberta-base", lr: float = 2e-5,
                 epochs: int = 3, patience: int = 1, batch_size: int = 32, max_len: int = 64, device: str = "cpu",
                 local_files_only: bool = False):
        try:
            import transformers  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError("install the 'hf' extra: pip install -e '.[hf]'") from exc
        self.seed, self.model_name, self.lr, self.epochs = seed, model_name, lr, epochs
        self.patience, self.batch_size, self.max_len, self.device = patience, batch_size, max_len, device
        self.local_files_only = local_files_only
        self.normalizer = Normalizer(emoji=False, emoticons=True, slang=True)
        self.model = None
        self.tokenizer = None
        self.history: list[float] = []

    def _load(self, source: str) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=self.local_files_only)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            source, num_labels=2, local_files_only=self.local_files_only).to(self.device)

    def _enc(self, texts):
        return self.tokenizer([self.normalizer(t) for t in texts], truncation=True, max_length=self.max_len,
                              padding=True, return_tensors="pt")

    @torch.no_grad()
    def predict_proba(self, texts) -> np.ndarray:
        self.model.eval()
        texts, out = list(texts), []
        for i in range(0, len(texts), 64):
            enc = self._enc(texts[i : i + 64]).to(self.device)
            out.append(torch.softmax(self.model(**enc).logits, -1).cpu().numpy())
        return np.concatenate(out)

    def fit(self, train_texts, train_y, val_texts, val_y) -> "FineTuneModel":
        torch.manual_seed(self.seed)
        self._load(self.model_name)
        texts, y = list(train_texts), np.asarray(train_y)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        steps = self.epochs * ((len(texts) + self.batch_size - 1) // self.batch_size)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / max(1, steps)))
        rng = np.random.default_rng(self.seed)
        best, best_state, bad = -1.0, None, 0
        for _ in range(self.epochs):
            self.model.train()
            order = rng.permutation(len(texts))
            for i in range(0, len(order), self.batch_size):
                idx = order[i : i + self.batch_size]
                enc = self._enc([texts[j] for j in idx]).to(self.device)
                loss = self.model(**enc, labels=torch.tensor(y[idx], device=self.device)).loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
            score = macro_f1(np.asarray(val_y), self.predict_proba(val_texts).argmax(1))
            self.history.append(score)
            if score > best:
                best, best_state, bad = score, copy.deepcopy(self.model.state_dict()), 0
            else:
                bad += 1
                if bad > self.patience:
                    break
        self.model.load_state_dict(best_state)
        return self

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        write_meta(folder, self.name, {"seed": self.seed, "base_model": self.model_name, "history": self.history,
                                       "max_len": self.max_len})
        self.model.save_pretrained(folder / "hf")
        self.tokenizer.save_pretrained(folder / "hf")

    @classmethod
    def load(cls, folder: Path) -> "FineTuneModel":
        meta = json.loads((Path(folder) / "model.json").read_text(encoding="utf-8"))
        model = cls(seed=meta["seed"], model_name=meta["base_model"], max_len=meta["max_len"], local_files_only=True)
        model._load(str(Path(folder) / "hf"))
        model.history = meta["history"]
        return model
