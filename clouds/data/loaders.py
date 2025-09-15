from abc import ABC, abstractmethod
import pandas as pd

class SourceLoader(ABC):
    @abstractmethod
    def load(self):
        pass

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
