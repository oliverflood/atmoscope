from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from sklearn.metrics import f1_score, average_precision_score

@dataclass
class MultiLabelMetrics:
    classes: List[str]
    thresholds: Optional[np.ndarray] = None

    def __post_init__(self):
        self.reset()

    def reset(self):
        self._y_true: List[np.ndarray] = []
        self._y_prob: List[np.ndarray] = []

    @torch.no_grad()
    def update(self, y_true: torch.Tensor, logits: torch.Tensor):
        # y_true: [B, C] floats from {0,1}
        # logits: [B, C] raw scores
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        self._y_prob.append(probs)
        self._y_true.append(y_true.detach().cpu().numpy())

    def _stack(self):
        y_true = np.vstack(self._y_true) if self._y_true else np.zeros((0, len(self.classes)), dtype=np.float32)
        y_prob = np.vstack(self._y_prob) if self._y_prob else np.zeros_like(y_true)
        return y_true, y_prob

    def compute(self) -> Dict[str, Any]:
        y_true, y_prob = self._stack()
        if y_true.size == 0:
            return {"micro_f1": 0.0, "macro_f1": 0.0, "mAP": 0.0, "per_class_f1": {}, "support": {}}

        th = self.thresholds if self.thresholds is not None else 0.5 * np.ones(len(self.classes))
        y_pred = (y_prob >= th.reshape(1, -1)).astype(np.float32)

        micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        per_f1 = np.array(f1_score(y_true, y_pred, average=None, zero_division=0))
        support = y_true.sum(axis=0).astype(int).tolist()


        support = y_true.sum(axis=0)
        per_ap = []
        for c in range(y_true.shape[1]):
            if support[c] == 0:
                per_ap.append(float('nan'))
            else:
                per_ap.append(average_precision_score(y_true[:, c], y_prob[:, c]))
        macro_AP = (np.nanmean(per_ap) if np.any(np.isfinite(per_ap)) else float('nan'))
        micro_AP = (average_precision_score(y_true.ravel(), y_prob.ravel())
                    if y_true.sum() > 0 else float('nan'))

        return {
            "micro_f1": float(micro),
            "macro_f1": float(macro),
            "micro_AP": micro_AP,
            "macro_AP": macro_AP,
            "per_class_f1": {c: float(f) for c, f in zip(self.classes, per_f1)},
            "support": {c: s for c, s in zip(self.classes, support)},
        }

    def tune_thresholds_for_f1(self, max_points: int = 19) -> np.ndarray:
        y_true, y_prob = self._stack()
        if y_true.size == 0:
            return np.full(len(self.classes), 0.5, dtype=np.float32)
        
        ts = np.linspace(0.05, 0.95, max_points)
        best = np.zeros(len(self.classes), dtype=np.float32)

        for c in range(len(self.classes)):
            f1s = [f1_score(y_true[:, c], (y_prob[:, c] >= t).astype(np.float32), zero_division=0) for t in ts]
            best[c] = ts[int(np.argmax(np.array(f1s)))]

        self.thresholds = best
        return best
