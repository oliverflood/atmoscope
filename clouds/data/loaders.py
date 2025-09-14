from abc import ABC, abstractmethod
import pandas as pd
import hashlib

class SourceLoader(ABC):
    @abstractmethod
    def load(self):
        pass

# The job of this loader is just to get a nice and clean pd dataframe
# into the hands of the future ImageDataset class
class GazeLoader(SourceLoader):
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
    
    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)

        DIRECTIONS = ["North", "East", "South", "West", "Up"]

        LABEL_COL_MAP = {
            "Clearsky": "clearsky",
            "Cirrus/Cirrostratus": "cirrus_cirrostratus",
            "Cirrocumulus/Altocumulus": "cirrocumulus_altocumulus",
            "Altostratus/Stratus": "altostratus_stratus",
            "Stratocumulus": "stratocumulus",
            "Cumulus": "cumulus",
            "Cumulonimbus": "cumulonimbus",
            "Contrails": "contrails",
            "Smoke/Haze": "smoke_haze",
            "Dust": "dust"
        }

        PER_IMAGE_META = {
            "Agreement": "agreement",
            "Classification Count": "classification_count",
            "Retirement": "retirement"
        }

        GLOBAL_META = {
            "Observation Number": "observation_number",
            "Measurement Date (UTC)": "measurement_date_utc",
            "Measurement Time (UTC)": "measurement_time_utc",
            "Observation Latitude": "latitude",
            "Observation Longitude": "longitude"
        }

        # This is really doing the job of a "transform" thing but whatever can be refactored later
        out_rows = []
        for direction in DIRECTIONS:
            dir_labels = {f"{direction} {k}": v for k, v in LABEL_COL_MAP.items()} # {"North Cumulus": "cumulus" ...}
            per_image_labels = {f"{direction} {k}": v for k, v in PER_IMAGE_META.items()}
            url_col = f"{direction} Image URL"
            
            for _, row in df.iterrows():
                class_labels = {v: row.get(k) for k, v in dir_labels.items()}
                
                if all(val == 5 for val in class_labels.values()):
                    for key in class_labels:
                        class_labels[key] = 0
                    class_labels["not_classified"] = 1
                else:
                    class_labels["not_classified"] = 0

                image_metadata = {v: row.get(k) for k, v in per_image_labels.items()}
                row_metadata = {v: row.get(k) for k, v in GLOBAL_META.items()}

                out_row = {
                    "photo_url": row.get(url_col),
                    "direction": direction.lower()
                }
                for dict in [class_labels, image_metadata, row_metadata]:
                    out_row.update(dict)
                out_rows.append(out_row)
        
        flat = pd.DataFrame(out_rows)
        flat["source"] = "gaze"
        flat["local_path"] = None

        return flat



from abc import ABC
from dataclasses import dataclass

@dataclass
class LabelSpace:
    name: str
    classes: list[str]

COARSE_8 = LabelSpace("coarse8", [
    "altostratus_stratus", 
    "cirrocumulus_altocumulus",
    "cirrus_cirrostratus", 
    "clearsky",
    "cumulonimbus", 
    "cumulus", 
    "not_classified",
    "stratocumulus"
])

from ..config import DATA_INTERIM
gazeLoader = GazeLoader(f"{DATA_INTERIM}/gaze_raw.csv")
df = gazeLoader.load()

# col_sum = df[COARSE_8.classes].sum(axis=0)
# for col, total in col_sum.items():
#     print(f"{col}: {total}")


import numpy as np

class TaskAdapter(ABC):
    @property
    @abstractmethod
    def classes(self):
        pass
    
    @abstractmethod
    def map_row(self, row) -> np.ndarray:
        pass

class GazeToCoarse(TaskAdapter):
    def __init__(self):
        self.label_space = COARSE_8
    
    @property
    def classes(self):
        return self.label_space.classes

    def map_row(self, row: pd.Series) -> np.ndarray:
        y = np.zeros(len(self.classes))

        for i, col in enumerate(self.classes):
            y[i] = row.get(col, 0)
        
        return y

from ..config import DATA_IMAGES
from pathlib import Path

from PIL import Image
from io import BytesIO
import os, requests


class ImageStore:
    def __init__(self):
        self.cache_dir = Path(DATA_IMAGES)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash_url(url: str) -> str:
        h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return f"{h}.jpg"
    
    def load(self, row):
        local_path = row.get("local_path")
        if local_path and os.path.exists(str(local_path)):
            return Image.open(local_path).convert("RGB")

        url = row.get("photo_url")
        if not url:
            raise FileNotFoundError("No local_path or photo_url available for this row")
        
        fname = self._hash_url(url)
        fpath = self.cache_dir / fname
        
        if not fpath.exists():
            r = requests.get(url, timeout=12)
            r.raise_for_status()
            fpath.write_bytes(r.content)
        
        return Image.open(fpath).convert("RGB")
    

import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, table: pd.DataFrame, adapter: TaskAdapter, store: ImageStore, transforms=None):
        self.df = table.reset_index(drop=True)
        self.adapter = adapter
        self.store = store
        self.transforms = transforms
        self.classes = adapter.classes
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = self.store.load(row)
        if self.transforms:
            img = self.transforms(img)
        y_vec = self.adapter.map_row(row)
        # y = int(y_vec.argmax())
        # y = torch.tensor(y, dtype=torch.long)
        y = torch.from_numpy(y_vec).float()
        meta = {
            "id": row.get("id"),
            "source": row.get("source"),
            # "observation_group": row.get("observation_group"),
            "direction": row.get("direction"),
        }

        # IGNORING METADATA (for now)
        meta = {}
        return img, y, meta
    



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

        # mAP = average_precision_score(y_true, y_prob, average="macro")
        mAP = 0

        return {
            "micro_f1": float(micro),
            "macro_f1": float(macro),
            "mAP": float(mAP),
            "per_class_f1": {c: float(f) for c, f in zip(self.classes, per_f1)},
            "support":     {c: s for c, s in zip(self.classes, support)},
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






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from ..config import DATA_INTERIM
csv_path = f"{DATA_INTERIM}/gaze_raw.csv"
df = GazeLoader(csv_path).load()

from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
groups = np.array(df["observation_number"].astype(str).values)
train_idx, val_idx = next(gss.split(df, groups=groups))


from torchvision import transforms as T
tfm = T.Compose([
    T.Resize(256), 
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

store = ImageStore()
adapter = GazeToCoarse()
full_ds = ImageDataset(df, adapter, store, transforms=tfm)
val_ds = ImageDataset(df, adapter, store, transforms=tfm)

tr_ds = torch.utils.data.Subset(full_ds, train_idx)
va_ds = torch.utils.data.Subset(val_ds,  val_idx)

from torch.utils.data import DataLoader
train_loader = DataLoader(tr_ds, batch_size=32, shuffle=True,  num_workers=2, pin_memory=True)
val_loader = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

from torchvision import models
import torch.nn as nn
num_classes = len(adapter.classes)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
in_feats = model.fc.in_features
model.fc = nn.Linear(in_feats, num_classes)
model = model.to(device)

train_label_mat = df.iloc[train_idx][adapter.classes].values.astype("float64")
n_pos = train_label_mat.sum(axis=0)
N = len(train_idx)
pos_weight_np = (N - n_pos) / (n_pos + 1e-8)
pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

from tqdm.auto import tqdm
import torch

def run_epoch(dl, metrics: MultiLabelMetrics, train=True, epoch=1):
    phase = "Train" if train else "Val"
    model.train() if train else model.eval()
    metrics.reset()
    total_loss, total = 0.0, 0

    pbar = tqdm(dl, desc=f"{phase} {epoch}", unit="batch", leave=False)
    for imgs, y, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(imgs)
            loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total += bs

        metrics.update(y, logits)
        cur = metrics.compute()
        pbar.set_postfix({"loss": f"{(total_loss/max(total,1)):.4f}", "μF1": f"{cur['micro_f1']:.3f}"})

    epoch_stats = metrics.compute()
    return total_loss / max(total, 1), epoch_stats


metrics_train = MultiLabelMetrics(adapter.classes)
metrics_val = MultiLabelMetrics(adapter.classes)

EPOCHS = 10
best_micro = 0.0
for ep in range(1, EPOCHS + 1):
    tr_loss, tr = run_epoch(train_loader, metrics_train, train=True, epoch=ep)
    va_loss, va = run_epoch(val_loader, metrics_val, train=False, epoch=ep)

    if va["micro_f1"] > best_micro:
        best_micro = va["micro_f1"]
        torch.save(model.state_dict(), "best.pt")

    tqdm.write(
        f"Epoch {ep:02d} | "
        f"train_loss {tr_loss:.4f}  microF1 {tr['micro_f1']:.3f}  macroF1 {tr['macro_f1']:.3f}  mAP {tr['mAP']:.3f} | "
        f"val_loss {va_loss:.4f}  microF1 {va['micro_f1']:.3f}  macroF1 {va['macro_f1']:.3f}  mAP {va['mAP']:.3f}"
    )

from ..config import MODELS_DIR
best_thr = metrics_val.tune_thresholds_for_f1()
np.save(f"{MODELS_DIR}/best_thresholds.npy", best_thr)