import typing as tp

import pytorch_lightning as pl


class ModelWrapper(pl.LightningModule):
    def __init__(
        self,
        model: pl.LightningModule,
        classes: tp.List[str],
        img_size: int,
        threshold: float,
    ):
        """
        Create a simple wrapper for model before converting to TorchScript.

        Parameters
        ----------
        model : torch.nn.Module
            Model to wrap.
        classes : List[str]
            List with classes labels of model.
        img_size : int
            Required size of input images
        threshold : float
            Threshold to classify images by probabilities.
        """
        super().__init__()
        self.model = model
        self.classes = classes
        self.img_size = img_size
        self.threshold = threshold

    def forward(self, image):
        return self.model.forward(image)
