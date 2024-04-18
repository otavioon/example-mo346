import lightning as L
import torch
from models.mlp import MultiLayerPerceptron
from data_modules.har import HarDataModule
from base import Configuration
from models.linear_classifier import LinearClassifier


class HAR_PreTrain(Configuration):
    def model(
        self,
        input_features: int = 360,
        hidden_size: int = 64,
        num_classes: int = 6,
        learning_rate: float = 0.001,
    ) -> L.LightningModule:
        """
        Create and return a LightningModule for the HAR model.

        Parameters
        ----------
        input_features : int, optional
            The number of input features, by default 360
        hidden_size : int, optional
            The size of the hidden layer, by default 64
        num_classes : int, optional
            The number of output classes, by default 6
        learning_rate : float, optional
            The learning rate for the optimizer, by default 0.001

        Returns
        -------
        L.LightningModule
            The created LightningModule for the HAR model.
        """
        return MultiLayerPerceptron(
            input_features=input_features,
            hidden_size=hidden_size,
            num_classes=num_classes,
            learning_rate=learning_rate,
        )

    def data_module(self, path, *args, **kwargs) -> L.LightningDataModule:
        """Create a LightningDataModule for the HAR model.

        This method creates and returns a LightningDataModule object for the HAR (Human Activity Recognition) model.
        The LightningDataModule is responsible for preparing the data for training, validation, and testing.

        Parameters
        ----------
        path : str
            The path to the dataset.

        Returns
        -------
        L.LightningDataModule
            The LightningDataModule object for the HAR model.
        """
        return HarDataModule(path, flatten=True)


class HAR_Downstream(Configuration):
    def model(
        self,
        backbone_ckpt_path: str = None,
        input_features: int = 360,
        hidden_size: int = 64,
        num_classes: int = 6,
        learning_rate: float = 0.001,
    ) -> L.LightningModule:
        """
        Create a model for human activity recognition.

        Parameters
        ----------
        backbone_ckpt_path : str, optional
            Path to the checkpoint file for the backbone model. If provided, 
            the weights of the backbone model will be loaded from the 
            checkpoint, by default None.
        input_features : int, optional
            Number of input features, by default 360.
        hidden_size : int, optional
            Size of the hidden layer in the backbone model, by default 64.
        num_classes : int, optional
            Number of output classes, by default 6.
        learning_rate : float, optional
            Learning rate for the optimizer, by default 0.001.

        Returns
        -------
        L.LightningModule
            The created model for human activity recognition.

        Notes
        -----
        This method creates a model for human activity recognition using a 
        backbone model and a linear classifier.

        The backbone model is a MultiLayerPerceptron, which consists of a series of fully connected layers.
        The number of input features, hidden size, and number of output classes can be customized.

        If a `backbone_ckpt_path` is provided, the weights of the backbone model will be loaded from the checkpoint.

        The backbone model is modified by removing the last layer, and a new linear layer is added as the head of the model.

        The created model is returned as a `L.LightningModule`, which is a PyTorch Lightning module that can be used for training and evaluation.
        """
        backbone = MultiLayerPerceptron(
            input_features=input_features,
            hidden_size=hidden_size,
            num_classes=num_classes,
            learning_rate=learning_rate,
        )
        if backbone_ckpt_path is not None:
            ckpt = torch.load(backbone_ckpt_path)
            backbone.load_state_dict(ckpt["state_dict"])

        backbone.block = torch.nn.Sequential(
            *list(backbone.block.children())[:-1]
        )
        head = torch.nn.Linear(hidden_size, num_classes)
        return LinearClassifier(backbone, head)

    def data_module(self, path, *args, **kwargs) -> L.LightningDataModule:
        return HarDataModule(path, flatten=True)
