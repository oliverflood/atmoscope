from abc import ABC, abstractmethod
import pandas as pd

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

            for i, row in df.iterrows():
                out_row = {v: row.get(k) for k, v in dir_labels.items()}
                out_row.update({v: row.get(k) for k, v in per_image_labels.items()})
                out_row.update({v: row.get(k) for k, v in GLOBAL_META.items()})

                out_rows.append(out_row)

        return pd.DataFrame(out_rows)


from ..config import DATA_INTERIM
gazeLoader = GazeLoader(f"{DATA_INTERIM}/gaze_flat.csv")
df = gazeLoader.load()
print(df.head())

