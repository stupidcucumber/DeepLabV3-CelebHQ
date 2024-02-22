from typing import Any
from .base import Callback


class BestWeightsCallback(Callback):
    def __init__(self) -> None:
        super(BestWeightsCallback, self).__init__()
    
    def epoch_end(self, data: Any) -> None:
        return super().epoch_end(data)