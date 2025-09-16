from __future__ import annotations
from typing import Optional, Dict, Any
from .base import BaseTracker
import mlflow
import mlflow.pytorch as mlflow_pytorch

class MLflowTracker(BaseTracker):
    def __init__(self, 
            experiment: str = "clouds", 
            run_name: Optional[str] = None,
            tracking_uri: Optional[str] = None, 
            tags: Optional[Dict[str, str]] = None,
            register_model: Optional[str] = None):
        
        self.experiment = experiment
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.tags = tags or {}
        self.register_model = register_model
        self._active = False

    def start(self):
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)
        mlflow.start_run(run_name=self.run_name)
        if self.tags:
            mlflow.set_tags(self.tags)
        self._active = True

    def end(self):
        if self._active:
            mlflow.end_run()
            self._active = False

    def log_params(self, params: Dict[str, Any]):
        # flatten nested dicts a bit
        flat = {}
        for k, v in params.items():
            if isinstance(v, (list, tuple, set)):
                v = ",".join(map(str, v))
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}.{kk}"] = vv
                continue
            flat[k] = v
        mlflow.log_params(flat)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        mlflow.log_metrics(metrics, step=step) if step is not None else mlflow.log_metrics(metrics)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None):
        if artifact_path:
            mlflow.log_artifact(path, artifact_path)
        else:
            mlflow.log_artifact(path)

    def log_dict(self, d: Dict[str, Any], artifact_file: str):
        mlflow.log_dict(d, artifact_file)

    def log_pytorch_model(self, model, artifact_path: str = "model"):
        mlflow_pytorch.log_model(model, artifact_path=artifact_path,
                                 registered_model_name=self.register_model)
