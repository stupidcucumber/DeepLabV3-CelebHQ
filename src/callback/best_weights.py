from typing import Any
import pathlib
import numpy as np
import torch
from .base import Callback


class BestWeightsCallback(Callback):
    def __init__(self, output: pathlib.Path) -> None:
        super(BestWeightsCallback, self).__init__()
        self.output = output
        if not output.exists():
            output.mkdir()
        self.last_best_accuracy = None

    def _calculate_mAP(self, accuracies: dict[str, float]) -> float:
        return np.mean([value for value in accuracies.values() if value != 1.0])
    
    def epoch_end(self, data: Any) -> None:
        mAP = self._calculate_mAP(accuracies=data['extra_val']['accuracy'])
        if self.last_best_accuracy is None \
            or self.last_best_accuracy <= mAP:
            self.last_best_accuracy = mAP
            weights_path = self.output.joinpath('best_weights.pt')
            model = data['model']
            torch.save(model, weights_path)