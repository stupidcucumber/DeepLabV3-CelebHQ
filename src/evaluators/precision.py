import numpy as np
import torch
from torch import Tensor
from .evaluator import MeanEvaluator
from ..utils import decode


class PrecisionMeanEvaluator(MeanEvaluator):
    def __init__(self, name: str='precision', mapping: dict | None = None):
        super(PrecisionMeanEvaluator, self).__init__(name=name, mapping=mapping)

    def _calculate_precision_layer(self, index: int, layer_predicted: Tensor, layer_true: Tensor) -> np.float32:
        true = torch.sum(layer_predicted[layer_true == index])
        predicted = torch.sum(layer_predicted == 1)
        if predicted > 0:
            precision = true / predicted
        else:
            precision = 1.0
        return precision

    def calculate_value(self, logits: Tensor, labels: Tensor) -> dict[str, np.float32] | np.float32:
        _batch_num, _layer_num = logits.shape[:2]
        precisions = np.zeros(shape=(_layer_num, _batch_num))
        for batch_index, (sample_logits, sample_label) in enumerate(zip(logits, labels)):
            decoded_logits = decode(sample_logits, dim=0)
            for layer_index, predicted_layer in enumerate(decoded_logits):
                precisions[layer_index, batch_index] \
                    = self._calculate_precision_layer(index=layer_index, layer_predicted=predicted_layer, layer_true=sample_label)
        if self.mapping:
            result_precisions = precisions.mean(axis=1)
            result = {self.mapping[index]: result_precisions[index] for index in range(_layer_num)}
        else:
            result = precisions.mean(axis=0)
        return result
