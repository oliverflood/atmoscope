import torch
from typing import Optional, Dict, Any
from clouds.metrics.metrics import MultiLabelMetrics
from clouds.train.trainer import Trainer, TrainConfig
from clouds.presets.registry import get_data, get_model

def run_experiment(
    data: str = "gaze",
    model: str = "resnet18",
    *,
    epochs: int = 10,
    batch_size: int = 32,
    val_size: float = 0.15,
    seed: int = 42,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    ckpt_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    csv_path: Optional[str] = None,
    transforms=None,
) -> Dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_fn = get_data(data)
    db = data_fn(csv_path=csv_path, 
                 batch_size=batch_size, 
                 val_size=val_size,
                 seed=seed, 
                 device=device, 
                 transforms=transforms)

    model_fn = get_model(model)
    mb = model_fn(classes=db.classes, 
                  lr=lr, 
                  weight_decay=weight_decay,
                  device=device, 
                  pos_weight=db.pos_weight)

    trainer = Trainer(
        model=mb.model,
        optimizer=mb.optimizer,
        criterion=mb.criterion,
        device=device,
        classes=db.classes,
        metrics_factory=lambda cls: MultiLabelMetrics(cls),
    )
    cfg = TrainConfig(epochs=epochs, ckpt_path=ckpt_path)
    best_val = trainer.fit(db.train_loader, db.val_loader, cfg)
    final_val = trainer.evaluate(db.val_loader)
    return {"best": best_val, "final": final_val}
