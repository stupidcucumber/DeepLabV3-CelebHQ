import numpy as np
from torch import Tensor
from .evaluator import MeanEvaluator
from ..utils import decode


class AccuracyMeanEvaluator(MeanEvaluator):
    def __init__(self, name: str):
        super(AccuracyMeanEvaluator, self).__init__(name=name)
    
    def _calculate_accuracy_layer(self, layer_predicted, layer_true) -> np.float32:
        _mp = np.multiply(layer_predicted, layer_true)
        intersection = np.sum(_mp)
        union = np.sum(np.asarray([layer_predicted, layer_true])) - intersection
        accuracy = intersection / union
        return accuracy

    def calculate_value(self, logits: Tensor, labels: Tensor) -> dict | np.float32:
        _batch_num, _layer_num = logits.shape[:2]
        accuracies = np.zeros(shape=(_layer_num, _batch_num))
        for batch_index, (sample_logits, sample_label) in enumerate(zip(logits, labels)):
            decoded_logits = decode(sample_logits, dim=0)
            for layer_index, (predicted_layer, true_layer) in enumerate(zip(decoded_logits, sample_label)):
                accuracies[layer_index, batch_index] \
                    = self._calculate_accuracy_layer(layer_predicted=predicted_layer, layer_true=true_layer)
        if self.mapping:
            result_accuracies = accuracies.mean(axis=1)
            result = {self.mapping[index]: result_accuracies[index] for index in range(_layer_num)}
        else:
            result = accuracies.mean(axis=0)
        return result