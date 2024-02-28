from typing import Any
import pathlib
from torch.utils.tensorboard import SummaryWriter
from .base import Callback


class TensorboardCallback(Callback):
    def __init__(self, log_dir: str | pathlib.Path = 'logs') -> None:
        super(TensorboardCallback, self).__init__()
        self.writer = SummaryWriter(log_dir=log_dir)
        self.iteration = 0

    def batch_end(self, data: Any) -> None:
        for key, value in data.items():
            if not str(key).startswith('extra'):
                self.writer.add_scalar(key, value, global_step=self.iteration)
            else:
                partition = key.split('_')[1]
                for evaluator, evaluation in data[key].items():
                    for inner_key, inner_value in evaluation.items():
                        self.writer.add_scalar(
                            '_'.join([partition, evaluator, inner_key]), 
                            inner_value, 
                            global_step=self.iteration
                        )
        self.iteration += 1