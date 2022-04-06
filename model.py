import timm
from pytorch_lightning import LightningModule
from torch.nn import Linear, Dropout, Softmax
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities.types import LRSchedulerType
from torch import Tensor, save, load
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy
from os.path import isfile


class Model(LightningModule):
    def __init__(self, num_classes: int, learning_rate: float = 0.05, dropout: float = 0.1, filename: str = 'model.pt') -> None:
        """
        This model is an image classifier inspired by the 'Xception' model
        :param num_classes: the number of classes in the dataset
        """
        super().__init__()

        self.filename = filename

        # we use a predefined model 'Xception' as it is
        # one of the state of the art networks
        self.model = timm.create_model('xception')

        # change the last layer of the network to map to the
        # classes of our dataset
        self.model.fc = Linear(2048, num_classes)

        self.dropout = Dropout(dropout)

        self.softmax = Softmax(dim=1)

        self.learning_rate = learning_rate

        self.accuracy = Accuracy()

    def forward(self, x):
        """
        feed images through the model
        :param x: The images
        :return: A tensor of class probabilities
        """
        return self.softmax(self.dropout(self.model(x)))

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRSchedulerType]]:
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