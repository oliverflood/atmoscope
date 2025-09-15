import torch
from torch.utils.data import Dataset
import pandas as pd
from .adapters import TaskAdapter
from .image_store import ImageStore

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

        y = torch.from_numpy(y_vec).float()
        meta = {
            "id": row.get("id"),
            "source": row.get("source"),
            "direction": row.get("direction"),
        }

        # IGNORING METADATA (for now)
        meta = {}
        return img, y, meta