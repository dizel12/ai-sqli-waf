from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

from ml.models.base import Detector, register

_MODEL = "distilbert-base-uncased"
_MAXLEN = 192


@register
class DistilBertDetector(Detector):
    name = "distilbert"

    def __init__(self) -> None:
        torch.manual_seed(13)
        self.tok = AutoTokenizer.from_pretrained(_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            _MODEL, num_labels=2)

    def _encode(self, texts: list[str]) -> dict:
        return self.tok(texts, truncation=True, padding="max_length",
                        max_length=_MAXLEN, return_tensors="pt")

    def fit(self, tt, tl, vt, vl) -> None:
        enc = self._encode(tt)
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"],
                           torch.tensor(tl))
        dl = DataLoader(ds, batch_size=16, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        self.model.train()
        for _epoch in range(2):
            for ids, mask, y in dl:
                opt.zero_grad()
                out = self.model(input_ids=ids, attention_mask=mask, labels=y)
                out.loss.backward()
                opt.step()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        self.model.eval()
        enc = self._encode(texts)
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=1)[:, 1]
        return probs.cpu().numpy().astype(float)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)

    @classmethod
    def load(cls, path: Path) -> "DistilBertDetector":
        obj = cls.__new__(cls)
        Detector.__init__(obj)
        obj.tok = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path)
        obj.model.eval()
        return obj
