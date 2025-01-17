"""
    A module for reporting metrics for training and
    testing runs
"""
import os
import time
import datetime
import pandas as pd
import numpy as np
from adaswarm.utils.options import get_device
from platform import platform


class Metrics:
    """
    Metrics class to capture
    * Time taken to run
    * Accuracy
    """

    class Stats:
        """
        Class to hold the value of accuracy
        """

        def __init__(self):
            self.best_training_accuracy = 0.0
            self.best_test_accuracy = 0.0
            self.best_training_loss = None
            self.best_test_loss = None
            self.number_of_epochs = 0
            self.epoch_train_losses = []
            self.epoch_test_losses = []
            self.epoch_train_accuracies = []
            self.epoch_test_accuracies = []

        def update_batch_training_accuracy(self, value):
            """
            Compare and store the best training accuracy
            value during the batch in progress
            """

            # TODO: Take epoch as an argument and store the best accuracy
            # and the epoch when it was achieved
            if value > self.best_training_accuracy:
                self.best_training_accuracy = value

        def update_test_accuracy(self, value):
            """
            Compare and store the best test accuracy
            value
            """
            if value > self.best_test_accuracy:
                self.best_test_accuracy = value

        def update_training_loss(self, value):
            """
            Compare and store the best training loss
            value
            """
            if (self.best_training_loss == None) or (value < self.best_training_loss):
                self.best_training_loss = value

        def update_test_loss(self, value):
            """
            Compare and store the best test loss
            value
            """
            if (self.best_test_loss == None) or (value < self.best_test_loss):
                self.best_test_loss = value

        def current_epoch(self, value):
            self.number_of_epochs = value

        def add_epoch_train_loss(self, value):
            self.epoch_train_losses.append(value)

        def add_epoch_train_accuracy(self, value):
            self.epoch_train_accuracies.append(value)

        def add_epoch_test_loss(self, value):
            self.epoch_test_losses.append(value)

        def add_epoch_test_accuracy(self, value):
            self.epoch_test_accuracies.append(value)

        def current_epoch_loss(self):
            return np.round(self.epoch_train_losses[-1:], 3)

        def run_data(self):
            return (
                self.epoch_train_losses,
                self.epoch_train_accuracies,
                self.epoch_test_losses,
                self.epoch_test_accuracies,
            )

    def __init__(
        self,
        name: str = "Default run",
        csv_filepath: str = os.path.join("report", "summary.csv"),
        dataset: str = "Not set",
    ):
        self.csv_filepath = csv_filepath
        self.name = name
        self.stats = Metrics.Stats()
        self.tstart = time.time()
        self.dataset = dataset

    def __enter__(self):
        return self.stats

    def __write_summary_report(self):
        time_taken = time.time() - self.tstart
        this_summary_dataframe = pd.DataFrame(
            {
                "Start time": datetime.datetime.fromtimestamp(self.tstart).strftime(
                    "%d-%m-%y %H:%M:%S"
                ),
                "Name": self.name,
                "Device": get_device().type,
                "Platform": platform(),
                "Dataset": self.dataset,
                "Epochs": self.stats.number_of_epochs,
                "Elapsed (s)": time_taken,
                "Training Acc %": np.round(
                    100.0 * self.stats.best_training_accuracy, 2
                ),
                "Training Loss": self.stats.best_training_loss
                if self.stats.best_training_loss != None
                else "Not set",
                "Test Acc %": np.round(100.0 * self.stats.best_test_accuracy, 2),
                "Test Loss": self.stats.best_test_loss
                if self.stats.best_test_loss != None
                else "Not set",
            },
            index=[0],
        )
        os.makedirs(os.path.dirname(self.csv_filepath), exist_ok=True)

        if os.path.exists(self.csv_filepath):
            previous_summary_dataframe = pd.read_csv(self.csv_filepath, index_col=None)
        else:
            previous_summary_dataframe = pd.DataFrame()

        this_summary_dataframe = pd.concat(
            [previous_summary_dataframe, this_summary_dataframe], axis=0
        )

        this_summary_dataframe.to_csv(self.csv_filepath, index=False)

    def __exit__(self, *args):
        self.__write_summary_report()
