import torch
import torch.nn as nn
from torchvision import models
from typing import List, Optional
from .bundles import ModelBundle
from .registry import register_model

@register_model("resnet18")
def build_resnet18(
    classes: List[str],
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    device: Optional[torch.device] = None,
    pos_weight: Optional[torch.Tensor] = None,
    pretrained: bool = True,
):
    num_classes = len(classes)
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT if pretrained else None
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if device is not None:
        model = model.to(device)

    if pos_weight is not None and device is not None:
        pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    return ModelBundle(model=model, criterion=criterion, optimizer=optimizer)
