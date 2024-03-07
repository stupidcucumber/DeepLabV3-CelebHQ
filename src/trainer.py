from torch.utils.data import DataLoader
from torch.nn import Module
import torch
import logging
from robustprinter import Printer
from robustprinter.formatter import DefaultFormatter
from .evaluators import MeanEvaluator
from .callback import Callback

logger = logging.getLogger('train')

class Trainer:
    def __init__(self, model: Module, 
                 loss_fn,
                 optimizer: torch.optim.Optimizer,
                 evaluators: list[MeanEvaluator],
                 callbacks: list[Callback],
                 device: str = 'cpu'):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.evaluators = evaluators
        self.callbacks = callbacks
        self.device = device
        self.rprinter = Printer(formatter=DefaultFormatter(max_columns=4))

    def _unroll_metrics(self, data: dict) -> dict:
        result = dict()
        for key, value in data.items():
            if isinstance(value, dict):
                result.update({
                    ((key + '_') if not key.startswith('extra') else '') + inner_key: inner_value 
                    for inner_key, inner_value in self._unroll_metrics(data=value).items()
                })
            else:
                result[key] = value
        return result

    def _construct_data(self, epoch: int, partition: str, step: int, max_steps: int, extras: dict):
        data = dict()
        data['epoch'] = epoch
        data['partition'] = partition
        data['step'] = step
        data['max_steps'] = max_steps
        data['metrics'] = self._unroll_metrics(data=extras)
        return data

    def _move_to_device(self, inputs: torch.Tensor, labels: torch.Tensor) -> list[torch.Tensor, torch.Tensor]:
        return inputs.to(self.device), labels.to(self.device)

    def _train_step(self, logits, labels) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        loss_1 = self.loss_fn(logits['out'], labels)
        loss_2 = self.loss_fn(logits['aux'], labels)
        loss = loss_1 + loss_2
        loss.backward()
        self.optimizer.step()
        return loss

    def _val_step(self, logits, labels) -> float:
        self.model.eval()
        loss_1 = self.loss_fn(logits['out'], labels)
        loss_2 = self.loss_fn(logits['aux'], labels)
        loss = loss_1 + loss_2
        return loss
    
    def _evaluate(self, data: dict, logits: torch.Tensor, labels: torch.Tensor, partition: str = 'train'):
        for evaluator in self.evaluators:
            evaluator.append(logits=logits, labels=labels)
            data['extra_%s' % partition][evaluator.name] = evaluator.get_result()
    
    def _compute_epoch(self, epoch: int, data: dict, loader: DataLoader, partition: str = 'train'):
        losses = []
        for step, (inputs, labels) in enumerate(loader):
            for callback in self.callbacks:
                callback.batch_start(data=data)
            inputs, labels = self._move_to_device(inputs=inputs, labels=labels)
            output = self.model(inputs)
            loss = self._train_step(logits=output, labels=labels) if partition == 'train' \
                else self._val_step(logits=output, labels=labels)
            logits = output['out'].float()
            losses.append(loss)
            average_loss = torch.mean(torch.as_tensor(losses, dtype=torch.float32))
            data['%s_loss' % partition] = loss
            self._evaluate(data=data, logits=logits, labels=labels, partition=partition)
            self.rprinter.print(
                data=self._construct_data(
                    epoch=epoch, 
                    partition=partition, 
                    step=step,
                    max_steps=len(loader),
                    extras=data
                )
            )
            for callback in self.callbacks:
                callback.batch_end(data=data)
        return average_loss

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
                 epochs: int):
        logger.info('Start fitting the model.')
        self.rprinter.start()
        for epoch in range(epochs):
            data = {
                'extra_train': dict(),
                'extra_val' : dict()
            }
            callback_data = data.copy()
            callback_data.update({'model': self.model})
            for callback in self.callbacks:
                callback.epoch_start(data=callback_data)
            
            average_loss = self._compute_epoch(epoch=epoch, data=data, loader=train_loader, partition='train')
            self.rprinter.break_loop()

            logger.info('training', extra={'epoch': epoch, 'average_loss': average_loss})
            data.pop('train_loss')
            data.pop('extra_train')
            
            with torch.no_grad():
                average_loss = self._compute_epoch(epoch=epoch, data=data, loader=val_loader, partition='val')
                self.rprinter.break_loop()
                logger.info('validating', extra={'epoch': epoch, 'average_loss': average_loss})
            
            callback_data = data.copy()
            callback_data.update({'model': self.model})
            for callback in self.callbacks:
                callback.epoch_end(data=callback_data)

        logger.info('Fitting has been ended.')