import numpy as np

from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch import is_tensor, from_numpy
from torchvision import transforms
from torch import cuda
from torch.backends import cudnn

from pyTsetlinMachine.tools import Binarizer
from pyTsetlinMachine.tm import MultiClassTsetlinMachine

from taimatsu.utils import to_categorical
from taimatsu.utils.options import get_device
from taimatsu.resnet import ResNet18
from taimatsu.model import Model

import logging

from sklearn import datasets as skl_datasets
from torchvision import datasets as tv_datasets
from sklearn.model_selection import StratifiedShuffleSplit

device = get_device()


class DataLoaderFetcher:
    def __init__(self, name: str = "Iris", model_name: str = "ResNet18"):
        self.name = name
        self.model_name = model_name

    # TODO: Handle error in case user passes an unsupported dataset name
    def train_loader(self) -> DataLoader:

        # TODO: make Iris the default
        if self.name == "Iris" or self.name == "Wine":
            return DataLoader(
                self.dataset(train=True, transform=None),
                batch_size=len(self.dataset(train=True, transform=None)),  # batch size is the entire dataset
                shuffle=True,
                drop_last=False,
            )


        elif self.name == "MNIST":
            transform_train = transforms.Compose(
                [
                    # Image Transformations suitable for MNIST dataset(handwritten digits)
                    transforms.RandomRotation(30),
                    transforms.RandomAffine(
                        degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)
                    ),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    # Mean and Std deviation values of MNIST dataset
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            return DataLoader(
                self.dataset(train=True, transform=transform_train),
                batch_size=125,
                shuffle=True,
                num_workers=2,
            )

    def test_loader(self) -> DataLoader:
        if self.name == "Iris" or self.name == "Wine":
            return DataLoader(
                self.dataset(train=False),
                batch_size=len(self.dataset(train=False)), # batch size is the entire dataset
                shuffle=True,
                drop_last=False,
            )


        elif self.name == "MNIST":
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            return DataLoader(
                self.dataset(train=False, transform=transform_test),
                batch_size=100,
                shuffle=False,
                num_workers=2,
            )

    def dataset(self, train=True, transform=None):
        if self.model_name == "TsetlinMachine":
            binarize = True
            categorical = False
        else:
            binarize = False
            categorical = True
        if self.name == "Iris":
            return TabularDataSet(train=train, dataset=skl_datasets.load_iris, binarize=binarize, categorical=categorical)
        elif self.name == "Wine":
            return TabularDataSet(train=train, dataset=skl_datasets.load_wine, binarize=binarize, categorical=categorical)
        elif self.name == "MNIST":
            return tv_datasets.MNIST(
                root="./data", train=train, download=True, transform=transform
            )

    def model(self):
        if self.model_name == "ResNet18":
            if self.name in ["Iris", "Wine"]:
                model = Model(
                    n_features=self.dataset().number_of_predictors(),
                    n_neurons=10,
                    n_out=self.dataset().number_of_categories(),
                )
            elif self.name == "MNIST":
                model = ResNet18(in_channels=1, num_classes=10)

            model = model.to(device)
            if cuda.is_available():
                cudnn.benchmark = True
                model = DataParallel(model)

            return model
        elif self.model_name == "TsetlinMachine":
            if self.name == "Iris":
                number_of_clauses = 300
                number_of_state_bits = 100

                model = MultiClassTsetlinMachine(
                    number_of_clauses=number_of_clauses,
                    number_of_state_bits=number_of_state_bits,
                    s=3.0,
                    T=10,
                    boost_true_positive_feedback=1
                )
                return model
        else:
            raise ValueError(f"Model '{self.model_name}' not supported for dataset '{self.name}'")




class TabularDataSet(Dataset):
    def __init__(self, train=True, dataset=None, binarize=False, categorical=True):

        self.binarize_bool = binarize
        if dataset is None:
            raise RuntimeError("Dataset not provided")

        data_bundle = dataset()

        x, y = data_bundle.data, data_bundle.target

        if self.binarize_bool:
            x = self.binarize_data(x)
        if categorical:
            y_formatted = to_categorical(y)
            self._number_of_categories = np.shape(y_formatted)[1]
        else:
            
            y_formatted = np.array(y)
            # change to 1D array of arrays
            y_formatted = np.reshape(y_formatted, (len(y_formatted), 1))
            self._number_of_categories = np.shape(np.unique(y))[0]
            print(f"Number of categories: {self._number_of_categories}")

        self._number_of_predictors = np.shape(x)[1]

        stratified_shuffle_split = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=123
        )

        for train_index, test_index in stratified_shuffle_split.split(X=x, y=y):
            x_train_array = x[train_index]
            x_test_array = x[test_index]
            y_train_array = y_formatted[train_index]
            y_test_array = y_formatted[test_index]

        if train:

            self.data = x_train_array
            self.target = y_train_array
        else:

            self.data = x_test_array
            self.target = y_test_array

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        # TODO: may be repetition of from_numpy here

        predictors = from_numpy(self.data[idx, :]).float().to(device)

        categories = from_numpy(self.target[idx]).to(device)
        return predictors, categories

    def number_of_predictors(self):
        return self._number_of_predictors

    def number_of_categories(self):
        return self._number_of_categories

    @staticmethod
    def binarize_data(data: np.array) -> np.array:
        """Binarize the data using pyTsetlinMachine.tools.Binarizer."""
        logging.debug(f"feature shape: {data.shape}")
        logging.debug(f"feature sample: \n {data[:5]}")
        binarizer = Binarizer(max_bits_per_feature=4)
        binarizer.fit(data)
        binarized_data_array = binarizer.transform(data)
        logging.debug(f"Binarized data shape: {binarized_data_array.shape}")
        logging.debug(f"Binarized data sample: \n {binarized_data_array[:5]}")

        return binarized_data_array
    
         

    def __len__(self):
        return len(self.data)



