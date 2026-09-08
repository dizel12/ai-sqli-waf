from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ml.models.base import Detector, register

MAX_LEN = 256
EMB = 64
CH = 128
KERNELS = (3, 5, 7)


def _encode(texts: list[str]) -> torch.Tensor:
    arr = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    for i, t in enumerate(texts):
        b = t.encode("utf-8", "ignore")[:MAX_LEN]
        arr[i, :len(b)] = [c + 1 for c in b]  # 0 = pad
    return torch.from_numpy(arr)


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(257, EMB, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(EMB, CH, k, padding=k // 2) for k in KERNELS])
        self.fc = nn.Linear(CH * len(KERNELS), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.emb(x).transpose(1, 2)
        feats = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(torch.cat(feats, dim=1)).squeeze(1)


@register
class CNNDetector(Detector):
    name = "cnn"

    def __init__(self) -> None:
        torch.manual_seed(13)
        self.net = _Net()

    def fit(self, tt, tl, vt, vl) -> None:
        self.net.train()
        pos = max(sum(tl), 1)
        neg = max(len(tl) - sum(tl), 1)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)

        xt, yt = _encode(tt), torch.tensor(tl, dtype=torch.float)
        xv, yv = _encode(vt), torch.tensor(vl, dtype=torch.float)
        best, bad = float("inf"), 0
        for _epoch in range(8):
            perm = torch.randperm(len(xt))
            for i in range(0, len(xt), 128):
                idx = perm[i:i + 128]
                opt.zero_grad()
                out = self.net(xt[idx])
                loss = loss_fn(out, yt[idx])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vloss = loss_fn(self.net(xv), yv).item()
            self.net.train()
            if vloss < best - 1e-4:
                best, bad = vloss, 0
            else:
                bad += 1
                if bad >= 2:
                    break

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            logits = self.net(_encode(texts))
            return torch.sigmoid(logits).cpu().numpy().astype(float)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "cnn.pt")
        (path / "cnn_meta.json").write_text(json.dumps(
            {"max_len": MAX_LEN, "emb": EMB, "ch": CH, "kernels": list(KERNELS)}))

    @classmethod
    def load(cls, path: Path) -> "CNNDetector":
        obj = cls()
        obj.net.load_state_dict(torch.load(path / "cnn.pt", map_location="cpu"))
        obj.net.eval()
        return obj
