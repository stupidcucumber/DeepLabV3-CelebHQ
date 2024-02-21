import numpy as np
from torch import Tensor
import torch
from .evaluator import MeanEvaluator
from ..utils import decode


class AccuracyMeanEvaluator(MeanEvaluator):
    def __init__(self, name: str = 'accuracy', mapping: dict | None = None):
        super(AccuracyMeanEvaluator, self).__init__(name=name, mapping=mapping)

    def _calculate_accuracy_layer(self, index: int, layer_predicted, layer_true) -> np.float32:
        intersection = torch.sum(layer_predicted[layer_true == index])
        union = torch.sum(layer_predicted == 1) + torch.sum(layer_true == index)
        if union > 0:
            accuracy = intersection / union
            return accuracy
        return 1.0

    def calculate_value(self, logits: Tensor, labels: Tensor) -> dict | np.float32:
        _batch_num, _layer_num = logits.shape[:2]
        accuracies = np.zeros(shape=(_layer_num, _batch_num))
        for batch_index, (sample_logits, sample_label) in enumerate(zip(logits, labels)):
            decoded_logits = decode(sample_logits, dim=0)
            for layer_index, predicted_layer in enumerate(decoded_logits):
                accuracies[layer_index, batch_index] \
                    = self._calculate_accuracy_layer(index=layer_index, layer_predicted=predicted_layer, layer_true=sample_label)
        if self.mapping:
            result_accuracies = accuracies.mean(axis=1)
            result = {self.mapping[index]: result_accuracies[index] for index in range(_layer_num)}
        else:
            result = accuracies.mean(axis=0)
        return result