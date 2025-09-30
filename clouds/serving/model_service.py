from typing import Optional, Tuple, List
import torch, torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.models import convnext_tiny
from clouds.presets.transforms import TFMS_BASIC

class CloudClassifier:
    def __init__(self, bundle_path: str, device: Optional[torch.device] = None):
        b = torch.load(bundle_path, map_location="cpu")
        self.classes: List[str] = b["classes"]
        self.arch: str = b.get("arch", "convnext_tiny")
        self.thresholds = b.get("thresholds")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # extend later
        if self.arch != "convnext_tiny":
            raise ValueError(f"Unsupported arch in bundle: {self.arch}")

        m = convnext_tiny(weights=None)
        in_feats = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_feats, len(self.classes)) # type: ignore
        m.load_state_dict(b["state_dict"])

        self.model = m.eval().to(self.device)
        self.tfms = TFMS_BASIC

    @torch.no_grad()
    def predict(self, img: Image.Image, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        x = self.tfms(img).unsqueeze(0).to(self.device)  # type: ignore
        probs = torch.sigmoid(self.model(x)).squeeze(0).cpu().numpy()
        thr = self.thresholds if self.thresholds is not None else threshold
        if np.isscalar(thr):
            decisions = (probs >= float(thr)).astype(np.uint8) # type: ignore
        else:
            decisions = (probs >= np.asarray(thr, dtype=np.float32)).astype(np.uint8)
        return probs, decisions