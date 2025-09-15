import torch
import numpy as np
from clouds import (
    GazeLoader, 
    ImageDataset, 
    ImageStore, 
    GazeToCoarse, 
    TrainConfig, 
    Trainer,
    MultiLabelMetrics
)

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

from ..config import MODELS_DIR
# ckpt_path = f"{MODELS_DIR}/resnet18_coarse7_best.pt"
# ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
# model.load_state_dict(ckpt)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    classes=adapter.classes,
    metrics_factory=lambda cls: MultiLabelMetrics(cls),
)

config = TrainConfig(epochs=1, ckpt_path=f"{MODELS_DIR}/resnet18_coarse7_best.pt")
best_val = trainer.fit(train_loader, val_loader, config)

val_stats = trainer.evaluate(val_loader)
print(val_stats)
