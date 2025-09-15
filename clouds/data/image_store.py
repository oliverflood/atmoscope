from pathlib import Path
from PIL import Image
import requests
import hashlib
import os
from ..config import DATA_IMAGES

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
