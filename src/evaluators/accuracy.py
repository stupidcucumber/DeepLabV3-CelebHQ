import numpy as np
from torch import Tensor
from .evaluator import MeanEvaluator
from ..utils import decode


class AccuracyMeanEvaluator(MeanEvaluator):
    def __init__(self, name: str):
        super(AccuracyMeanEvaluator, self).__init__(name=name)

    def calculate_value(self, logits: Tensor, labels: Tensor) -> np.float32:
        _batch_num, _layer_num = logits.shape[:2]
        accuracies = np.zeros(shape=(_layer_num, _batch_num))
        for batch_index, (sample_logits, sample_label) in enumerate(zip(logits, labels)):
            decoded_logits = decode(sample_logits, dim=0)
            for layer_index, (predicted_layer, true_layer) in enumerate(zip(decoded_logits, sample_label)):
                _mp = np.multiply(predicted_layer, true_layer)
                intersection = np.sum(_mp)
                union = np.sum(np.asarray([predicted_layer, true_layer])) - intersection
                _accuracy = intersection / union
                accuracies[layer_index, batch_index] = _accuracy
        return accuracies.mean(axis=1)