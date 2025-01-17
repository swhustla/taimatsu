import numpy as np

from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch import is_tensor, from_numpy
from torchvision import transforms
from torch import cuda
from torch.backends import cudnn

from adaswarm.utils import to_categorical
from adaswarm.utils.options import get_device
from adaswarm.resnet import ResNet18
from adaswarm.model import Model

from sklearn import datasets as skl_datasets
from torchvision import datasets as tv_datasets
from sklearn.model_selection import StratifiedShuffleSplit

device = get_device()


class DataLoaderFetcher:
    def __init__(self, name: str = "Iris"):
        self.name = name

    # TODO: Handle error in case user passes an unsupported dataset name
    def train_loader(self) -> DataLoader:

        # TODO: make Iris the default
        if self.name == "Iris":
            return DataLoader(
                self.dataset(train=True),
                batch_size=40,
                shuffle=True,
                drop_last=False,
            )

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
