import glob
import os
import urllib.request
import zipfile

import lightning as L
import numpy as np
import tifffile as tiff
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


class SeismicDataset(Dataset):
    """
    A custom dataset class for seismic data.

    Parameters
    ----------
        data_dir: str
            The directory path where the data files are located.
        labels_dir: str
            The directory path where the label files are located.
        transform: callable, optional
            A function/transform that takes in a sample and returns a transformed version.
            Default is None.
    """

    def __init__(self, data_dir, labels_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = glob.glob(data_dir + "/*.tif")
        self.labels_dir = labels_dir

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns
        -------
            int
                The total number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index.

        Parameters
        ----------
            idx: int
                The index of the sample to retrieve.

        Returns
        -------
            tuple
                A tuple containing the image and label of the sample.
        """
        img = tiff.imread(self.data[idx])
        img_base_name = os.path.basename(self.data[idx]).split(".")[0]

        label = np.array(
            Image.open(os.path.join(self.labels_dir, img_base_name + ".png"))
        )
        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        img = self.pad(img, [255, 701])
        label = self.pad(label, [255, 701])

        return img, label

    def pad(self, x, target_size):
        h, w = x.shape[:2]
        pad_h = max(0, target_size[0] - h)
        pad_w = max(0, target_size[1] - w)
        if len(x.shape) == 2:
            padded = np.pad(x, ((0, pad_h), (0, pad_w)), mode="reflect")
            padded = np.expand_dims(padded, axis=2)
        else:
            padded = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            padded = padded.astype(float)

        padded = np.transpose(padded, (2, 0, 1))

        return padded


class SeismicDataModule(L.LightningDataModule):
    def __init__(self, root_dir, batch_size=32):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.setup()

    def setup(self):
        # check if root dir exists
        if not os.path.exists(self.root_dir):
            # download data
            print("Downloading data")
            # TODO replace url_to_data with the actual URL
            url = "url_to_data"
            urllib.request.urlretrieve(url, self.root_dir + "/f3.zip")
            # extract data
            with zipfile.ZipFile(self.root_dir + "/f3.zip", "r") as zip_ref:
                zip_ref.extractall(self.root_dir)
            print("Data downloaded and extracted")

        self.data_dir = os.path.join(self.root_dir, "images")
        self.labels_dir = os.path.join(self.root_dir, "annotations")

        train_dataset = SeismicDataset(
            self.data_dir + "/train", self.labels_dir + "/train"
        )
        val_dataset = SeismicDataset(self.data_dir + "/val", self.labels_dir + "/val")
        test_dataset = SeismicDataset(
            self.data_dir + "/test", self.labels_dir + "/test"
        )

        self.train_dl = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_dl = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_dl = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl
