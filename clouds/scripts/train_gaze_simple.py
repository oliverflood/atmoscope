from clouds.experiments.runner import run_experiment
from clouds.config import MODELS_DIR
import torch

stats = run_experiment(
    data="gaze",
    model="resnet18",
    epochs=10,
    batch_size=32,
    lr=3e-4,
    weight_decay=1e-2,
    ckpt_path=f"{MODELS_DIR}/resnet18_coarse7_best.pt",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
print(stats)
