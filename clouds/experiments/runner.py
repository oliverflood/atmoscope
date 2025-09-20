import torch
from typing import Optional, Dict, Any
from clouds.metrics.metrics import MultiLabelMetrics
from clouds.trackers.mlflow_tracker import MLflowTracker
from clouds.train.trainer import Trainer, TrainConfig
from clouds.presets.registry import get_data, get_model

def run_experiment(
    data: str = "gaze",
    model: str = "resnet18",
    *,
    epochs: int = 10,
    batch_size: int = 32,
    val_size: float = 0.15,
    split_seed: int = 42,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    ckpt_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    csv_path: Optional[str] = None,
    transforms=None,
    tracker: MLflowTracker
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_fn = get_data(data)
    db = data_fn(csv_path=csv_path, 
                 batch_size=batch_size, 
                 val_size=val_size,
                 seed=split_seed, 
                 device=device, 
                 transforms=transforms)

    model_fn = get_model(model)
    mb = model_fn(classes=db.classes, 
                  lr=lr, 
                  weight_decay=weight_decay,
                  device=device, 
                  pos_weight=db.pos_weight)
    
    tracker.start()
    tracker.log_params({
        # data
        "data": data,
        "csv_path": csv_path,
        "val_size": val_size,
        "split_seed": split_seed,  

        # model
        "model": model,
        "num_classes": len(db.classes),
        "ckpt_path": ckpt_path, 

        # training
        "epochs": epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "lr": lr, 
    })

    trainer = Trainer(
        model=mb.model,
        optimizer=mb.optimizer,
        criterion=mb.criterion,
        device=device,
        classes=db.classes,
        metrics_factory=lambda cls: MultiLabelMetrics(cls),
        tracker=tracker
    )
    cfg = TrainConfig(epochs=epochs, ckpt_path=ckpt_path)
    trainer.fit(db.train_loader, db.val_loader, cfg)

    if hasattr(tracker, "log_pytorch_model"):
        tracker.log_model(trainer.model, artifact_path="model")

    tracker.end()
