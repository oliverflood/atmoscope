import torch
from clouds.experiments.runner import run_experiment
from clouds.config import MODELS_DIR
from clouds.trackers.mlflow_tracker import MLflowTracker
from clouds.presets.transforms import TFMS_MEDIUM

tracker = MLflowTracker(
    experiment="test",
    run_name="convtiny_coarse7_seed42",
    tags={"project":"clouds-ml","label_space":"coarse7"}
)

run_experiment(
    data="gaze",
    model="convnext_tiny",
    epochs=15,
    batch_size=32,
    lr=3e-4,
    weight_decay=1e-2,
    val_size=0.2,
    ckpt_path=f"{MODELS_DIR}/convtiny_coarse7_best.pt",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    transforms=TFMS_MEDIUM,
    tracker=tracker
)

# tracker = MLflowTracker(
#     experiment="test",
#     run_name="resnet18_coarse7_seed42",
#     tags={"project":"clouds-ml","label_space":"coarse7"}
# )

# run_experiment(
#     data="gaze_tiny",
#     model="resnet18",
#     epochs=5,
#     batch_size=32,
#     lr=3e-4,
#     weight_decay=1e-2,
#     ckpt_path=f"{MODELS_DIR}/resnet18_coarse7tiny_best.pt",
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#     tracker=tracker
# )
