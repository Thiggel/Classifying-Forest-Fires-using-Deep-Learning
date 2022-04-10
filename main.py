"""
Author: Filipe Laitenberger
"""

from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from torch import cuda

from dataset.TreeSpeciesClassificationDataModule import TreeSpeciesClassificationDataModule
from Xception import Xception
from Lenet5 import LeNet5


def run_model(model: LightningModule, data_module: LightningDataModule) -> None:
    """
    Run a model out of the two used in this experiment (xception or lenet5)
    and do some repetetive steps such as calculating the learning rate
    :param model: A pytorch-lightning model
    """
    # load the model if it exists already
    model.load()

    trainer = Trainer(
        max_epochs=100,
        # if GPUs are available, use all of them
        gpus=(-1 if cuda.is_available() else 0)
    )

    # calculate optimal learning rate
    lr_finder = trainer.tuner.lr_find(model, datamodule=data_module)
    model.learning_rate = lr_finder.suggestion()

    # train the network
    trainer.fit(model, data_module)

    # test the network
    trainer.test(model, data_module)


if __name__ == '__main__':
    """
    Run the experiment by training and testing the two models
    (Xception and LeNet-5)
    """

    data_module = TreeSpeciesClassificationDataModule()

    models = [
        Xception(num_classes=data_module.num_classes, filename='xception.pt'),
        LeNet5(num_classes=data_module.num_classes, filename='lenet5.pt')
    ]

    for model in models:
        run_model(model, data_module)

