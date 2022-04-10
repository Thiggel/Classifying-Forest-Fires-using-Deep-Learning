"""
Author: Filipe Laitenberger
"""

from pytorch_lightning import LightningModule
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities.types import LRSchedulerType
from torch import Tensor, save, load
from torch.nn.functional import cross_entropy
from os.path import isfile
from typing import Tuple, List
from torchmetrics import Accuracy


class TreeClassificationModel(LightningModule):
    def __init__(self, learning_rate: float = 0.05, filename: str = 'model.pt') -> None:
        """
        This class serves as an abstract class that implements methods
        both models in this study use, like the training/validation/testing steps
        and configurations options such as optimizers and schedulers
        :param learning_rate: The learning rate of the model
        :param filename: The filename under which it will be saved after each epoch
        """
        super().__init__()

        self.filename = filename
        self.learning_rate = learning_rate
        self.accuracy = Accuracy()

    def forward(self, x: Tensor) -> Tensor:
        pass

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRSchedulerType]]:
        """
        This methods specifies the optimizer and learning rate scheduler.
        We use ADAM and a ReduceOnPlateau learning rate scheduler that
        multiplies the learning rate by 0.1 if the training loss doesn't
        decrease for three epochs
        :return: the optimizer and learning rate scheduler
        """
        optimizer = Adam(self.parameters(), lr=self.learning_rate)

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, patience=3),
            'monitor': 'train_loss'
        }

        return [optimizer], [scheduler]

    def training_step(self, batch: Tensor, _) -> Tensor:
        """
        A training step
        :param batch: the batch tensor
        :return: the loss of the batch under the current model
        """
        # get columns of batch
        images, targets = batch

        predicted = self.forward(images)
        loss = cross_entropy(predicted, targets)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch: Tensor, _) -> Tensor:
        """
        A validation step
        :param batch: the batch tensor
        :return: the loss of the batch under the current model
        """
        # get columns of batch
        images, targets = batch

        predicted = self.forward(images)
        loss = cross_entropy(predicted, targets)
        accuracy = self.accuracy(predicted, targets)

        self.log('val_loss', loss)
        self.log('val_acc', accuracy)

        print('Validation Loss: ', loss)
        print('Validation Accuracy: ', accuracy)

        return loss

    def test_step(self, batch: Tensor, _) -> Tensor:
        """
        A test step
        :param batch: the batch tensor
        :return: the loss of the batch under the current model
        """
        # get columns of batch
        images, targets = batch

        predicted = self.forward(images)
        loss = cross_entropy(predicted, targets)
        accuracy = self.accuracy(predicted, targets)

        self.log('test_loss', loss)
        self.log('test_acc', accuracy)

        print('Test Loss: ', loss)
        print('Test Accuracy: ', accuracy)

        return loss

    def training_epoch_end(self, _) -> None:
        """
        At the end of each epoch we save the model
        """
        self.save()

    def save(self) -> None:
        """
        Save model under specified filename
        """
        print("Saving model at: " + self.filename)
        save(self.state_dict(), self.filename)

    def load(self) -> None:
        """
        Load model from the filename provided
        """
        if isfile(self.filename):
            print("Loading model from: " + self.filename)
            self.load_state_dict(load(self.filename))