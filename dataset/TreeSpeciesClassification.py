"""
Author: Filipe Laitenberger
        Thijs van der Laan
"""

from torch.utils.data import Dataset
from torch import Tensor
from os import walk
from os.path import join
from PIL.Image import open
from torchvision.transforms import PILToTensor, Compose, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter
from typing import Tuple


class TreeSpeciesClassification(Dataset):
    def __init__(self, image_dir: str) -> None:
        """
        This dataset provides pictures of tree tops (from above) and their
        corresponding labels.
        It also performs Data Augmentation on the dataset
        :param image_dir: the path to the folder that contains the dataset's images
        """
        super().__init__()

        # create an array that contains all image paths
        # in the dataset, by flattening the directory structure
        self.images = [join(path, name) for path, _, files in walk(image_dir) for name in files]

        # The images are transformed to tensors and augmented
        self.transform = Compose([
            # transform the image to a tensor
            PILToTensor(),

            # randomly flip the image horizontally, and vertically
            # we don't rotate the images since that would leave
            # some pixels black which could inhibit learning
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.5),

            # randomly change the brightness, contrast, and
            # saturation of the images
            ColorJitter(
                brightness=(0.5, 1.5),
                contrast=(0.5, 1.5),
                saturation=(0.5, 1.5)
            )
        ])

        self.labels = {
            'Bi': 'Birch',
            'Bu': 'Beech',
            'Dgl': 'Douglas fir',
            'Ei': 'Oak',
            'Eis': 'Damaged Oak',
            'Erl': 'Alder',
            'Fi': 'Spruce',
            'Ki': 'Pine',
            'La': 'Larch',
            'Sch': 'Shadow / background'
        }

    def load_image(self, index: int) -> Tensor:
        """
        Load the index'th image from the dataset. The images are traversed such that
        the image with index 0 is the first image from the first sub folder of the dataset,
        and if, for instance, the first folder contains 1000 images, then the index 1000
        corresponds to the first image of the second sub folder, and so on.
        The image is then transformed to a tensor and augmented (randomly flipped both horizontally and vertically,
        and the contrast/brightness/saturation is randomly changed)
        :param index: the index of the image in the flattened file structure
        :return: the transformed image
        """
        image = open(self.images[index]).convert("RGB")

        return self.transform(image).float()

    def load_target(self, index: int) -> int:
        """
        Load the index of the target class
        :param index: the index of the image in the flattened
        folder structure
        :return: the target class's index
        """
        label = self.images[index].split('/')[-2]
        return list(self.labels.keys()).index(label)

    def __getitem__(self, index) -> Tuple[Tensor, int]:
        """
        Load a data point
        :param index: the index of the image in the flattened
        folder structure
        :return: a tuple containing the transformed image
        and the target class's index
        """
        return self.load_image(index), self.load_target(index)

    @property
    def num_classes(self) -> int:
        """
        Get the number of classes in the dataset
        :return: the number of classes (10)
        """
        return len(self.labels)

    def __len__(self) -> int:
        """
        Get the number of images in the dataset.
        This doesn't take into account that images will
        be randomly augmented (flipped + contrast/saturation/brightness changed),
        but those transforms are partly performed within continuous ranges,
        so that they will create infinitely many variations of the images.
        Hence, only the original number of images is returned here
        :return: the number of images in the dataset
        """
        return len(self.images)
