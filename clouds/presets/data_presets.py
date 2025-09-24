import torch, numpy as np
from sklearn.model_selection import GroupShuffleSplit
from torchvision import transforms as T
from torch.utils.data import DataLoader, Subset
from typing import Optional
from clouds.config import DATA_INTERIM
from clouds.data.loaders import GazeLoader
from clouds.data.adapters import GazeToCoarse
from clouds.data.image_store import ImageStore
from clouds.data.datasets import ImageDataset
from clouds.presets.transforms import TFMS_BASIC
from .bundles import DataBundle
from .registry import register_data

@register_data("gaze")
def make_gaze_data(
    csv_path: Optional[str] = None,
    batch_size: int = 32,
    val_size: float = 0.15,
    seed: int = 42,
    num_workers: int = 2,
    device: Optional[torch.device] = None,
    transforms: T.Compose = TFMS_BASIC,
):
    csv_path = csv_path or f"{DATA_INTERIM}/gaze_raw.csv"
    df = GazeLoader(csv_path).load()

    adapter = GazeToCoarse()
    classes = adapter.classes

    # grouped split by observation_number
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    groups = np.array(df["observation_number"].astype(str).values)
    train_idx, val_idx = next(gss.split(df, groups=groups))

    store = ImageStore()
    full = ImageDataset(df, adapter, store, transforms=transforms)
    tr_ds = Subset(full, train_idx)
    va_ds = Subset(full, val_idx)

    train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # compute pos_weight on train split only
    train_label_mat = df.iloc[train_idx][classes].values.astype("float64")
    n_pos = train_label_mat.sum(axis=0)
    N = len(train_idx)
    pos_weight_np = (N - n_pos) / (n_pos + 1e-8)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32)
    if device is not None:
        pos_weight = pos_weight.to(device)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        classes=classes,
        pos_weight=pos_weight,
        df=df, train_idx=train_idx, val_idx=val_idx
    )


@register_data("gaze_tiny")
def make_gaze_tiny_data(
    csv_path: Optional[str] = None,
    batch_size: int = 32,
    val_size: float = 0.15,
    seed: int = 42,
    num_workers: int = 2,
    device: Optional[torch.device] = None,
    transforms: T.Compose = TFMS_BASIC,
):
    csv_path = csv_path or f"{DATA_INTERIM}/gaze_raw.csv"
    df = GazeLoader(csv_path).load()

    # crop dataframe size
    df = df.head(100)

    adapter = GazeToCoarse()
    classes = adapter.classes

    # grouped split by observation_number
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    groups = np.array(df["observation_number"].astype(str).values)
    train_idx, val_idx = next(gss.split(df, groups=groups))

    store = ImageStore()
    full = ImageDataset(df, adapter, store, transforms=transforms)
    tr_ds = Subset(full, train_idx)
    va_ds = Subset(full, val_idx)

    train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # compute pos_weight on train split only
    train_label_mat = df.iloc[train_idx][classes].values.astype("float64")
    n_pos = train_label_mat.sum(axis=0)
    N = len(train_idx)
    pos_weight_np = (N - n_pos) / (n_pos + 1e-8)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32)
    if device is not None:
        pos_weight = pos_weight.to(device)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        classes=classes,
        pos_weight=pos_weight,
        df=df, train_idx=train_idx, val_idx=val_idx
    )