import timm
from pytorch_lightning import LightningModule
from torch.nn import Linear


class Model(LightningModule):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        # we use a predefined model 'Xception' as it is
        # one of the state of the art networks
        self.model = timm.create_model('xception')

        # change the last layer of the network to map to the
        # classes of our dataset
        self.model.fc = Linear(2048, num_classes)

        print(self.model)


model = Model(10)
