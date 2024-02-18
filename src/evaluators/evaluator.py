import torch
import numpy as np


class MeanEvaluator:
    '''
        
    '''
    def __init__(self, name: str):
        super(MeanEvaluator, self).__init__()
        self.name = name
        self.values = []
        self.result = None

    def get_result(self) -> np.float32:
        if self.result is None:
            self.result = np.mean(self.values)
        return self.result

    def calculate_value(self, logits: torch.Tensor, labels: torch.Tensor) -> np.float32:
        raise NotImplementedError('This method must be implemented in the derived class.')

    def append(self, logits: torch.Tensor, labels: torch.Tensor):
        value = self.calculate_value(logits=logits, labels=labels)
        self.values.append(value)
        self.result = None

    def zero(self):
        self.values.clear()

    def __getitem__(self, index) -> np.float32:
        return self.values[index]