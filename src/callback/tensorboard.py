from typing import Any
import pathlib
from torch.utils.tensorboard import SummaryWriter
from .base import Callback


class TensorboardCallback(Callback):
    def __init__(self, log_dir: str | pathlib.Path = 'logs') -> None:
        super(TensorboardCallback, self).__init__()
        self.writer = SummaryWriter(log_dir=log_dir)

    def epoch_end(self, data: Any) -> None:
        return super().epoch_end(data)