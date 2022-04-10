"""
Author: Filipe Laitenberger, Thijs van der Laan
"""

import timm
from torch.nn import Linear, Softmax

from TreeClassificationModel import TreeClassificationModel


class Xception(TreeClassificationModel):
    def __init__(self, num_classes: int, learning_rate: float = 0.05, filename: str = 'model.pt') -> None:
        """
        This model is an image classifier inspired by the 'Xception' model
        :param num_classes: The number of classes in the dataset
        :param learning_rate: The learning rate for performing gradient descent
        :param filename: The filename under which the model is saved after every epoch
        """
        super().__init__(learning_rate, filename)

        # we use a predefined model 'Xception' as it is
        # one of the state of the art networks
        self.model = timm.create_model('xception')

        # change the last layer of the network to map to the
        # classes of our dataset
        self.model.fc = Linear(2048, num_classes)

        self.softmax = Softmax(dim=1)

    def forward(self, x):
        """
        feed images through the model
        :param x: The images
        :return: A tensor of class probabilities
        """
        return self.softmax(self.model(x))
