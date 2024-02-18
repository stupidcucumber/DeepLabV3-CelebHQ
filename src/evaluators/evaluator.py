import torch
import numpy as np


class MeanEvaluator:
    '''
        This is the abstract class for the evaluators.
    '''
    def __init__(self, name: str, mapping: dict | None = None):
        '''
            :param name: The name of the evaluator
            :param mapping: The mapping name for layer, it comes as a list of dicts with keys - index: name. E.g.: {0: 'background', ...},
                the number of dicts must be the same as the size of the first non-batch dimention.
            :return: Returns nothing.
        '''
        super(MeanEvaluator, self).__init__()
        self.name = name
        self.mapping = mapping
        if self.mapping:
            self.values = {key: [] for key in self.mapping.values()}
        else:
            self.values = []
        self.result = None

    def get_result(self) -> dict[str, np.float32] | np.float32:
        if self.result is None:
            if self.mapping:
                self.result = {key: np.mean(value) for key, value in self.values.items()}
            else:
                self.result = np.mean(self.values)
        return self.result

    def calculate_value(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, np.float32] | np.float32:
        '''
            Function that calculates value of the metrics. If dict mapping is being passed then the result returned is in format
        {'layer_name': metric, ...}. This is only virtual function and therefore must be defined in the derived class.
        
        :param logits: Logits of the model.
        :param labels:  True Labels of the model.
        :return: Either dict[str, np.float32] or np.float32
        '''
        raise NotImplementedError('This method must be implemented in the derived class.')

    def append(self, logits: torch.Tensor, labels: torch.Tensor):
        if self.mapping:
            _dict = self.calculate_value(logits=logits, labels=labels)
            for key in self.values.keys():
                self.values[key].append(_dict[key])
        else:
            value = self.calculate_value(logits=logits, labels=labels)
            self.values.append(value)
        self.result = None

    def zero(self):
        '''
            This function zeros the calculator, so it can calculate new values on the new epoch.
        '''
        if self.mapping:
            for key in self.mapping.keys():
                self.values[key].clear()
        else:
            self.values.clear()

    def __getitem__(self, index) -> np.float32:
        if self.mapping:
            return {key: value[index] for key, value in self.values.items}
        return self.values[index]