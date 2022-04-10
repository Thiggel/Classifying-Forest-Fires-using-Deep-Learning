"""
Author: Mohamed Gamil, Thijs van der Laan
"""

import torch.nn as nn
from torch import Tensor

from TreeClassificationModel import TreeClassificationModel


class LeNet5(TreeClassificationModel):
    def __init__(self, num_classes: int, learning_rate: float = 0.05, filename: str = 'model.pt') -> None:
        """
        This class implements the Lenet-5 model proposed by Yann LeCunn in 1998
        :param num_classes: The number of classes in the dataset
        :param learning_rate: The learning rate for performing gradient descent
        :param filename: The filename under which the model is saved after every epoch
        """
        super().__init__(learning_rate, filename)

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1), padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1), padding=0),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()

        # fully connected
        self.fc = nn.Linear(9720, 84)
        self.relu = nn.ReLU()

        # output layer
        self.fc2 = nn.Linear(84, num_classes)

        self.softmax = nn.Softmax()

    def forward(self, x: Tensor) -> Tensor:
        """
        feed images through the model
        :param x: The images
        :return: A tensor of class probabilities
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.flatten(out)
        out = self.relu(self.fc(out))
        out = self.softmax(self.fc2(out))

        return out
