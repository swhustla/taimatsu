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

import pandas as pd

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
        if self.name == "Iris":
            if self.model_name == "ResNet18":
                return DataLoader(
                    self.dataset(train=True),
                    batch_size=40,
                    shuffle=True,
                    drop_last=False,
                )
            elif self.model_name == "TsetlinMachine":
                return DataBinarizer(DataLoader(
                    self.dataset(train=True),
                    batch_size=40,
                    shuffle=True,
                    drop_last=False,
                ))

        elif self.name == "Wine":
            return DataLoader(
                self.dataset(train=True),
                batch_size=40,
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
        if self.name == "Iris":
            return DataLoader(
                self.dataset(train=False),
                batch_size=10,
                shuffle=True,
                drop_last=False,
            )

        elif self.name == "Wine":
            return DataLoader(
                self.dataset(train=False),
                batch_size=10,
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
        if self.name == "Iris":
            return TabularDataSet(train=train, dataset=skl_datasets.load_iris)
        elif self.name == "Wine":
            return TabularDataSet(train=train, dataset=skl_datasets.load_wine)
        elif self.name == "MNIST":
            return tv_datasets.MNIST(
                root="./data", train=train, download=True, transform=transform
            )

    def model(self):
        if self.model_name == "ResNet18":
            if self.name == "Iris":
                model = Model(
                    n_features=self.dataset().number_of_predictors(),
                    n_neurons=10,
                    n_out=self.dataset().number_of_categories(),
                )
            elif self.name == "Wine":
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
                # TODO: Use torch.nn.parallel.DistributedDataParallel
                model = DataParallel(model)
            return model
        elif self.model_name == "TsetlinMachine":
            if self.name == "Iris":
                threshold_large_t = 10  # Tsetlin Machine threshold
                boost_factor_s = 3.0 # Tsetlin Machine boost factor
                number_of_clauses = 300 # Number of clauses per TM
                number_of_states = 100 # Number of states per TM

                model = MultiClassTsetlinMachine(
                    self.dataset().number_of_categories(),
                    number_of_clauses,
                    self.dataset().number_of_predictors(),
                    number_of_states,    
                    boost_factor_s,
                    threshold_large_t,
                    boost_true_positive_feedback = 1
                )
                return model
        else:
            raise RuntimeError(f"Model {self.model_name} not supported with dataset {self.name}")



class TabularDataSet(Dataset):
    def __init__(self, train=True, dataset=None):
        if dataset is None:
            raise RuntimeError("Dataset not provided")

        data_bundle = dataset()

        x, y = data_bundle.data, data_bundle.target
        y_categorical = to_categorical(y)

        self._number_of_predictors = np.shape(x)[1]
        self._number_of_categories = np.shape(y_categorical)[1]

        stratified_shuffle_split = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=123
        )

        for train_index, test_index in stratified_shuffle_split.split(X=x, y=y):
            x_train_array = x[train_index]
            x_test_array = x[test_index]
            y_train_array = y_categorical[train_index]
            y_test_array = y_categorical[test_index]

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

    def __len__(self):
        return len(self.data)





class DataBinarizer(DataLoader):
    """A class to binarize the data, aimed at the Tsetlin Machine."""
    
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        super().__init__(dataset, batch_size, shuffle, num_workers)

    def __iter__(self):
        for predictors, categories in super().__iter__():
            binarized_predictors = self.__binarize_data(predictors)
            yield binarized_predictors, categories

    @staticmethod
    def __binarize_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Binarize the data.

        The Tsetlin Machine requires the data to be binarized,
        i.e. to be represented as a set of 0s and 1s, due to the way that it works.
        """
        binarizer = Binarizer(max_bits_per_feature=10)
        np_array_version_of_data = np.array(data)
        binarizer.fit(np_array_version_of_data)
        binarized_data_array = binarizer.transform(np_array_version_of_data)
        new_binarized_columns = [
            f"binarized_{i}" for i in range(binarized_data_array.shape[1])
        ]
        return pd.DataFrame(
            binarized_data_array, columns=new_binarized_columns
        )
    
    
