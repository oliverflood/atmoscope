from dataclasses import dataclass
from typing import List, Optional
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    classes: List[str]
    pos_weight: torch.Tensor
    df: Optional[pd.DataFrame] = None
    train_idx: Optional[np.ndarray] = None
    val_idx: Optional[np.ndarray] = None

@dataclass
class ModelBundle:
    model: torch.nn.Module
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
