from torch.utils.data import DataLoader
from torch.nn import Module
import torch
import logging
from .evaluators import MeanEvaluator


class Trainer:
    def __init__(self, model: Module, 
                 loss_fn,
                 optimizer: torch.optim.Optimizer,
                 logger: logging.Logger,
                 evaluators: list[MeanEvaluator]):
        self.model = model
        self.logger = logger
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.evaluators = evaluators

    def train_step(self, logits, labels) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.loss_fn(logits, labels)
        loss.backward()
        self.optimizer.step()
        return loss

    def val_step(self, logits, labels) -> float:
        self.model.eval()
        loss = self.loss_fn(logits, labels)
        return loss

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
                 epochs: int):
        self.logger.info('Start fitting the model.')

        for epoch in range(epochs):
            losses = []
            for inputs, labels in train_loader:
                logits = self.model(inputs)['out'].float()
                loss = self.train_step(logits=logits, labels=labels)
                losses.append(loss)
                average_loss = torch.mean(torch.as_tensor(losses, dtype=torch.float32))
                print('Loss is: ', average_loss)
                for evaluator in self.evaluators:
                    evaluator.append(logits=logits, labels=labels)
                    print('Results of %s' % evaluator.name, evaluator.get_result())
            self.logger.info('training', extra={'epoch': epoch, 'average_loss': average_loss})

            losses.clear()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    logits = self.model(inputs)
                    loss = self.val_step(logits=logits, labels=labels)
                    losses.append(loss)
                    average_loss = torch.mean(torch.as_tensor(losses, dtype=torch.float32))
                    print('Loss is: ', average_loss)
                    for evaluator in self.evaluators:
                        evaluator.append(logits=logits, labels=labels)
                        print('Results of %s' % evaluator.name, evaluator.get_result())
                self.logger.info('validating', extra={'epoch': epoch, 'average_loss': average_loss})

        self.logger.info('Fitting has been ended.')