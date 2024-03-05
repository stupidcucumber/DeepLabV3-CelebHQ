import numpy as np
from torch import Tensor
from .evaluator import MeanEvaluator
from ..utils import decode


class RecallMeanEvaluator(MeanEvaluator):
    def __init__(self, name: str='recall', mapping: dict | None = None):
        super(RecallMeanEvaluator, self).__init__(name=name, mapping=mapping)

    def _calculate_recall_layer(index: int, layer_predicted: Tensor, layer_true: Tensor) -> np.float32:
        pass

    def calculate_value(self, logits: Tensor, labels: Tensor) -> dict[str, np.float32] | np.float32:
        _batch_num, _layer_num = logits.shape[:2]
        recalls = np.zeros(shape=(_layer_num, _batch_num))
        for batch_index, (sample_logits, sample_label) in enumerate(zip(logits, labels)):
            decoded_logits = decode(sample_logits, dim=0)
            for layer_index, predicted_layer in enumerate(decoded_logits):
                recalls[layer_index, batch_index] \
                    = self._calculate_recall_layer(index=layer_index, layer_predicted=predicted_layer, layer_true=sample_label)
        if self.mapping:
            result_recalls = recalls.mean(axis=1)
            result = {self.mapping[index]: result_recalls[index] for index in range(_layer_num)}
        else:
            result = recalls.mean(axis=0)
        return result