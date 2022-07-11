import os
import numpy as np

import torch

from abc import ABC, abstractmethod

#from .download_utils import download_dataset


class Dataset(ABC):
    """
    Abstract Dataset Base Class
    All subclasses must define __getitem__() and __len__()
    """
    def __init__(self, root, download_url=None, force_download=False, verbose=False):
        self.root_path = root
        # The actual archive name should be all the text of the url after the
        # last '/'.
        #if download_url is not None:
        #    dataset_zip_name = download_url[download_url.rfind('/')+1:].split('?')[0]
        #    self.dataset_zip_name = dataset_zip_name
        #    download_dataset(
        #        url=download_url,
        #        data_dir=root,
        #        dataset_zip_name=dataset_zip_name,
        #        force_download=force_download,
        #        verbose=verbose,
        #    )

    @abstractmethod
    def __getitem__(self, index):
        """Return data sample at given index"""

    @abstractmethod
    def __len__(self):
        """Return size of the dataset"""

class ImageFolderDataset(Dataset):
    """MNIST Dataset Class borrowed from i2dl"""

    def __init__(self, *args,
                 root=None,
                 images=None,
                 labels=None,
                 transform=None,
                 download_url="https://i2dl.dvl.in.tum.de/downloads/mnist.zip",
                 **kwargs):
        super().__init__(*args,
                         download_url=None,
                         root=root,
                         **kwargs)

        self.images = torch.load(os.path.join(root, images))
        if labels is not None:
            self.labels = torch.load(os.path.join(root, labels))
        else:
            self.labels = None
        # transform function that we will apply later for data preprocessing
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is not None:
            return image, self.labels[index]
        else:
            return image
