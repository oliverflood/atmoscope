import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
from clouds.data.label_spaces import COARSE_7

CLASSES = COARSE_7.classes

CKPT = "models/convtiny_coarse7_best.pt"
OUT = "models/clouds_bundle.pt"
ARCH = "convnext_tiny"

def main():
    device = "cpu"

    # build convnext_tiny with the correct head size
    model = convnext_tiny(weights=None)
    in_feats = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_feats, len(CLASSES)) # type: ignore

    # load checkpoint
    raw = torch.load(CKPT, map_location=device)
    if isinstance(raw, dict) and any(isinstance(v, torch.Tensor) for v in raw.values()):
        state_dict = raw
    elif isinstance(raw, dict):
        state_dict = raw.get("state_dict") or raw.get("model_state_dict")
        if state_dict is None and hasattr(raw, "state_dict"):
            state_dict = raw.state_dict() # type: ignore
        if state_dict is None:
            raise RuntimeError("Could not find a state_dict in checkpoint dict.")
    else:
        state_dict = raw.state_dict()

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("[warn] strict=False load")
        if missing: 
            print("  missing:", missing)
        if unexpected: 
            print("  unexpected:", unexpected)

    bundle = {
        "arch": ARCH,
        "state_dict": model.state_dict(),
        "classes": CLASSES,
        "meta": {"format": "clouds_bundle_v1"},
    }
    torch.save(bundle, OUT)
    print(f"Wrote {OUT} with {len(CLASSES)} classes.")

if __name__ == "__main__":
    main()
