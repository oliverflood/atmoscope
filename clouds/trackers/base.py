from __future__ import annotations
from typing import Optional, Dict, Any

class BaseTracker:
    def start(self): pass
    def end(self): pass
    def log_params(self, params: Dict[str, Any]): pass
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None): pass
    def log_artifact(self, path: str, artifact_path: Optional[str] = None): pass
    def set_tags(self, tags: Dict[str, str]): pass
    def log_dict(self, d: Dict[str, Any], artifact_file: str): pass
    def log_pytorch_model(self, model, artifact_path: str): pass

class NullTracker(BaseTracker):
    pass
