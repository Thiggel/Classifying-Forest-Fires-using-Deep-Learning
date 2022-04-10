from dataset.TreeSpeciesClassificationDataModule import TreeSpeciesClassificationDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from torch import cuda

from model import Model


if __name__ == '__main__':
    data_module = TreeSpeciesClassificationDataModule()

    model = Model(num_classes=data_module.num_classes)

    # load the model if it exists already
    model.load()

    trainer = Trainer(
        max_epochs=300,
        # if GPUs are available, use all of them
        gpus=(-1 if cuda.is_available() else 0)
    )

    # train the network
    trainer.fit(model, data_module)

    # test the network
    trainer.test(model, data_module)