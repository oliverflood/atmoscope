from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List
import numpy as np
import torch
from contextlib import nullcontext
from tqdm.auto import tqdm
from clouds.trackers.base import BaseTracker, NullTracker

@dataclass
class TrainConfig:
    epochs: int = 10
    ckpt_path: Optional[str] = None

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        classes: List[str],
        metrics_factory: Callable[[List[str]], Any], #lambda cls: MultiLabelMetrics(cls)
        tracker: Optional[BaseTracker] = None,
        use_cuda_amp: Optional[bool] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.classes = classes
        self.metrics_factory = metrics_factory
        self.tracker = tracker or NullTracker()

        if use_cuda_amp is None:
            use_cuda_amp = torch.cuda.is_available()
        self.use_cuda_amp = use_cuda_amp
        self.scaler = torch.GradScaler("cuda") if use_cuda_amp else None

    def _amp_ctx(self):
        return torch.autocast(device_type="cuda") if self.use_cuda_amp else nullcontext()

    def _run_epoch(self, loader, train: bool, epoch: int) -> tuple[float, Dict[str, Any]]:
        phase = "Train" if train else "Val"
        self.model.train() if train else self.model.eval()

        metrics = self.metrics_factory(self.classes)
        metrics.reset()
        total_loss, total = 0.0, 0

        pbar = tqdm(loader, desc=f"{phase} {epoch}", unit="batch", leave=False)
        for imgs, y, _ in pbar:
            imgs = imgs.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True).float()

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with self._amp_ctx(), torch.set_grad_enabled(train):
                logits = self.model(imgs)
                loss = self.criterion(logits, y)

            if train:
                if self.use_cuda_amp:
                    self.scaler.scale(loss).backward() # type: ignore
                    self.scaler.step(self.optimizer) # type: ignore
                    self.scaler.update() # type: ignore
                else:
                    loss.backward()
                    self.optimizer.step()

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total += bs

            metrics.update(y, logits)
            cur = metrics.compute()

            post = {"loss": f"{(total_loss/max(total,1)):.4f}", "μF1": f"{cur['micro_f1']:.3f}"}
            pbar.set_postfix(post)

        return total_loss / max(total, 1), metrics.compute()

    def fit(self, train_loader, val_loader, cfg: TrainConfig):
        best_metric = -float("inf")

        for ep in range(1, cfg.epochs + 1):
            tr_loss, tr = self._run_epoch(train_loader, train=True,  epoch=ep)
            va_loss, va = self._run_epoch(val_loader,   train=False, epoch=ep)

            self.tracker.log_metrics(
                {"train/loss": tr_loss, "train/micro_f1": tr["micro_f1"], "train/macro_f1": tr["macro_f1"]},
                step=ep
            )
            self.tracker.log_metrics(
                {"val/loss": va_loss, "val/micro_f1": va["micro_f1"], "val/macro_f1": va["macro_f1"]},
                step=ep
            )

            selected = va["micro_f1"]
            tqdm.write(
                f"Epoch {ep:02d} | "
                f"train_loss {tr_loss:.4f}  μF1 {tr['micro_f1']:.3f}  M-F1 {tr['macro_f1']:.3f}"
                f"\n  μAP {tr.get('micro_AP', float('nan')):.3f}  M-AP {tr.get('macro_AP', float('nan')):.3f} | "
                f"val_loss {va_loss:.4f}  μF1 {va['micro_f1']:.3f}  M-F1 {va['macro_f1']:.3f}"
                f"\n  μAP {va.get('micro_AP', float('nan')):.3f}  M-AP {va.get('macro_AP', float('nan')):.3f}"
            )

            if selected > best_metric:
                best_metric = selected
                if cfg.ckpt_path:
                    torch.save(self.model.state_dict(), cfg.ckpt_path)
                    self.tracker.log_artifact(cfg.ckpt_path)
    
    @torch.no_grad()
    def evaluate(self, loader, thresholds: Optional[np.ndarray] = None) -> Dict[str, Any]:
        self.model.eval()
        metrics = self.metrics_factory(self.classes)
        metrics.reset()

        for imgs, y, _ in tqdm(loader, desc="Eval", unit="batch", leave=False):
            imgs = imgs.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True).float()
            logits = self.model(imgs)
            if thresholds is not None:
                metrics.thresholds = thresholds
            metrics.update(y, logits)

        return metrics.compute()
