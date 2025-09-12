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

col_sum = df[COARSE_8.classes].sum(axis=0)
for col, total in col_sum.items():
    print(f"{col}: {total}")


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
        y = int(y_vec.argmax())
        y = torch.tensor(y, dtype=torch.long)
        meta = {
            "id": row.get("id"),
            "source": row.get("source"),
            # "observation_group": row.get("observation_group"),
            "direction": row.get("direction"),
        }

        # IGNORING METADATA (for now)
        meta = {}
        return img, y, meta
    


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
    T.Resize(256), T.CenterCrop(224),
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

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)


from tqdm.auto import tqdm
scaler = torch.cuda.amp.GradScaler()
amp_enabled = torch.cuda.is_available()

def run_epoch(dl, train=True, epoch=1):
    phase = "Train" if train else "Val"
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(dl, desc=f"{phase} {epoch}", unit="batch", leave=False)
    for imgs, y, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", enabled=amp_enabled), torch.set_grad_enabled(train):
            logits = model(imgs)
            loss = criterion(logits, y)

        if train:
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total += bs
        correct += (logits.argmax(dim=1) == y).sum().item()

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        postfix = {"loss": f"{avg_loss:.4f}", "acc": f"{acc:.3f}"}
        if train:
            postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
        pbar.set_postfix(postfix)

    return total_loss / max(total, 1), correct / max(total, 1)

EPOCHS = 10
for ep in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True, epoch=ep)
    va_loss, va_acc = run_epoch(val_loader, train=False, epoch=ep)
    tqdm.write(f"Epoch {ep:02d} | "
               f"train_loss {tr_loss:.4f}  train_acc {tr_acc:.3f} | "
               f"val_loss {va_loss:.4f}  val_acc {va_acc:.3f}")
