from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from .label_spaces import COARSE_7

class TaskAdapter(ABC):
    @property
    @abstractmethod
    def classes(self):
        pass
    
    @abstractmethod
    def map_row(self, row) -> np.ndarray:
        pass

class GazeToCoarse(TaskAdapter):
    def __init__(self):
        self.label_space = COARSE_7
    
    @property
    def classes(self):
        return self.label_space.classes

    def map_row(self, row: pd.Series) -> np.ndarray:
        y = np.zeros(len(self.classes))

        for i, col in enumerate(self.classes):
            y[i] = row.get(col, 0)
        
        return y