import numpy as np
from .evaluator import MeanEvaluator


class AccuracyMeanEvaluator(MeanEvaluator):
    def __init__(self, name: str):
        super(AccuracyMeanEvaluator, self).__init__(name=name)

    def get_result(self) -> np.float32:
        return super().get_result()