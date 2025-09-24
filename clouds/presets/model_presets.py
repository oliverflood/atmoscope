import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from typing import List, Optional
from .bundles import ModelBundle
from .registry import register_model


@register_model("resnet18")
def build_resnet18(
    classes: List[str],
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    device: Optional[torch.device] = None,
    pos_weight: Optional[torch.Tensor] = None
):
    num_classes = len(classes)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if device is not None:
        model = model.to(device)

    if pos_weight is not None and device is not None:
        pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    return ModelBundle(model=model, criterion=criterion, optimizer=optimizer)


@register_model("convnext_tiny")
def build_convnext_tiny(
    classes: List[str],
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    device: Optional[torch.device] = None,
    pos_weight: Optional[torch.Tensor] = None,
    pretrained: bool = True,
):
    num_classes = len(classes)
    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)

    in_feats = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_feats, num_classes) # type: ignore

    if device is not None:
        model = model.to(device)

    if pos_weight is not None and device is not None:
        pos_weight = pos_weight.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    return ModelBundle(model=model, criterion=criterion, optimizer=optimizer)
