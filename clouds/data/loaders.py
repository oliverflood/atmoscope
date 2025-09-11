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
    def __init__(self, label_space: LabelSpace):
        self.label_space = label_space
    def map_row(self, row):
        pass

class GazeToCoarse(TaskAdapter):
    def __init__(self):
        self.classes = COARSE_8.classes

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
            r.raise_for_status() # error if we get unsuccessful code
            fpath.write_bytes(r.content)
        
        return Image.open(fpath).convert("RGB")
    
store = ImageStore()
store.load(df.iloc[0])

adapter = GazeToCoarse()
print(adapter.map_row(df.iloc[0]))

