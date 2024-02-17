from torch.utils.data import DataLoader
from torch.nn import Module
import torch
import logging


class Trainer:
    def __init__(self, model: Module, 
                 loss_fn: function,
                 optimizer: torch.optim.Optimizer,
                 logger: logging.Logger):
        self.model = model
        self.logger = logger
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_step(self, logits, labels) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.loss_fn(logits, labels)
        loss.backward()
        self.optimizer.step()
        return loss.numpy()

    def val_step(self, logits, labels) -> float:
        self.model.eval()
        loss = self.loss_fn(logits, labels)
        return loss

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
                 epochs: int):
        self.logger.info('Start fitting the model.')

        for epoch in epochs:
            losses = []
            for inputs, labels in train_loader:
                logits = self.model(inputs)
                loss = self.train_step(logits=logits, labels=labels)
                losses.append(loss)
                average_loss = torch.mean(torch.as_tensor(losses, dtype=torch.float32))
                print('Loss is: ', average_loss)
            self.logger.info('training', extra={'epoch': epoch, 'average_loss': average_loss})

            losses.clear()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    logits = self.model(inputs)
                    loss = self.val_step(logits=logits, labels=labels)
                    losses.append(loss)
                    average_loss = torch.mean(torch.as_tensor(losses, dtype=torch.float32))
                    print('Loss is: ', )
                self.logger.info('validating', extra={'epoch': epoch, 'average_loss': average_loss})