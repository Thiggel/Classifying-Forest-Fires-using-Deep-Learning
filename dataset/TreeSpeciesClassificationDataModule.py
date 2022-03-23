from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, random_split, Subset, DataLoader
from typing import Sequence

from TreeSpeciesClassification import TreeSpeciesClassification


class TreeSpeciesClassificationDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 32) -> None:
        """
        This dataloader splits the dataset into a training set, a validation set,
        and a test set, and provides data loaders for all of them
        :param batch_size: The size of each batch of image/target pairs
        used for stochastic gradient decent (or one of its variants)
        """
        super().__init__()

        self.batch_size = batch_size

        self.train, self.test, self.val = self.split_dataset(
            TreeSpeciesClassification(image_dir='dataset/species_classification')
        )

    @staticmethod
    def split_dataset(dataset: Dataset) -> list[Subset[Dataset]]:
        """
        Split a dataset into a training set, a validation set,
        and a test set
        :param dataset: the dataset that is to be split
        :return: a list of three subsets of the original dataset (train/test/val)
        """
        size = dataset.__len__()

        # get 70% for the train set
        train_size = int(size // 1.25)

        # 20% for test set
        test_size = int(size // 5)

        # get 10% for val set
        val_size = int(size - train_size - test_size)

        lengths: Sequence = [train_size, test_size, val_size]

        return random_split(dataset, lengths)

    def train_dataloader(self) -> DataLoader:
        """
        Get a data loader that shuffles and provides batches of the training set
        :return: the training data loader
        """
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=12)

    def val_dataloader(self) -> DataLoader:
        """
        Get a data loader that shuffles and provides batches of the validation set
        :return: the validation data loader
        """
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=12)

    def test_dataloader(self) -> DataLoader:
        """
        Get a data loader that shuffles and provides batches of the test set
        :return: the test data loader
        """
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=12)
