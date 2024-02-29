from typing import Any
import pathlib
from torch.utils.tensorboard import SummaryWriter
from .base import Callback


class TensorboardCallback(Callback):
    def __init__(self, log_dir: str | pathlib.Path = 'logs') -> None:
        super(TensorboardCallback, self).__init__()
        self.writer = SummaryWriter(log_dir=log_dir)
        self.iterations = dict()

    def _get_iteration_num(self, key) -> int:
        if self.iterations.get(key, None) is None:
            self.iterations[key] = 0
        else:
            self.iterations[key] += 1
        return self.iterations[key]

    def batch_end(self, data: Any) -> None:
        for key, value in data.items():
            if not str(key).startswith('extra'):
                partition = key.split('_')[0]
                iteration = self._get_iteration_num(key=key)
                self.writer.add_scalar(
                    key, 
                    value, 
                    global_step=iteration)
            else:
                partition = key.split('_')[1]
                for evaluator, evaluation in data[key].items():
                    for inner_key, inner_value in evaluation.items():
                        compound_key = '_'.join([partition, evaluator, inner_key])
                        iteration = self._get_iteration_num(key=compound_key)
                        self.writer.add_scalar(
                            compound_key,
                            inner_value, 
                            global_step=iteration
                        )