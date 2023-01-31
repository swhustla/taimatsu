#!/usr/bin/env python3

"""An example that uses Tsetlin Machine on MNIST and other datasets"""

import argparse
import logging
import os
import sys
import time

import numpy as np



dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, ".."))


# pylint: disable=C0411, E0401, C0413
from taimatsu.utils import progress_bar, Metrics
from taimatsu.data import DataLoaderFetcher

from taimatsu.utils.options import (
    # number_of_epochs,
    dataset_name,
    get_device,
    log_level,
)

logging.basicConfig(level=log_level())

# pylint: disable=R0914,R0915,C0116,C0413


def run():
    with Metrics(name="Tsetlin", dataset=dataset_name()) as metrics:

        number_of_epochs = 500
        logging.debug("in run function")
        device = get_device()
        # Data
        logging.info("==> Preparing data..")

        fetcher = DataLoaderFetcher(name=dataset_name(), model_name="TsetlinMachine")
        print(f"dataset name: {dataset_name()}")
        print(f"Type of fetcher: {type(fetcher)}")
        train_loader = fetcher.train_loader()
        test_loader = fetcher.test_loader()

        num_batches_train = int(len(train_loader.dataset) / train_loader.batch_size)
        num_batches_test = int(len(test_loader.dataset) / test_loader.batch_size)

        logging.info(f"Training set size: {len(train_loader.dataset)}")
        logging.info(f"Test set size: {len(test_loader.dataset)}")

        # Model
        logging.info("==> Building model..")
        model = fetcher.model()

        # Training
        logging.info(f"==> Training Tsetlin machine model for {number_of_epochs} epochs..")
        
        def train():

            batch_accuracies = []

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                tic = time.monotonic()

                if dataset_name() in ["Iris", "Wine", "BreastCancer"]:
                    # convert to numpy array with 1 dimension
                    targets = np.array(targets).reshape(-1)
                    inputs = inputs.numpy()

                model.fit(X=inputs, Y=targets, epochs=number_of_epochs)

                predictions = model.predict(inputs)

                # if dataset_name() in ["Iris", "Wine", "BreastCancer"]:
                #     predictions = np.array(predictions).reshape(-1,1)
                #     # convert predictions to a triple column, via one-hot encoding (for Iris, Wine, BreastCancer)
                #     predictions = np.eye(3)[predictions]
                #     # convert predicitons to a single column (for Iris, Wine, BreastCancer)
                #     predictions = np.array(predictions).reshape(-1)


                # compare predictions to targets avoiding IndexError: list assignment index out of range

                accuracy = np.sum(predictions == targets) / targets.shape[0]
                # print(f"accuracy: {accuracy}")
                batch_accuracies.append(accuracy)
                toc = time.monotonic()
                logging.info(f"Batch {batch_idx} of {num_batches_train} had accuracy {100*batch_accuracies[batch_idx]:.1f}% | Time: {toc-tic:.2f}s")
                metrics.update_batch_training_accuracy(batch_accuracies[batch_idx])
                metrics.update_training_loss(0)
            logging.info(f"Average accuracy on training set: {100*np.mean(batch_accuracies):.1f}% +/- {100*np.std(batch_accuracies):.1f}%")
            metrics.add_epoch_train_accuracy(100 * sum(batch_accuracies) / len(batch_accuracies))
            metrics.add_epoch_train_loss(0)

        def test():
            batch_accuracies = []
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                if dataset_name() in ["Iris", "Wine", "BreastCancer"]:
                    # convert to numpy array with 1 dimension
                    targets = np.array(targets).reshape(-1)
                    inputs = inputs.numpy()
                predictions = model.predict(inputs)


                # if dataset_name() in ["Iris", "Wine", "BreastCancer"]:
                #     predictions = np.array(predictions).reshape(-1,1)
                #     # convert predictions to a triple column, via one-hot encoding (for Iris, Wine, BreastCancer)
                #     predictions = np.eye(3)[predictions]
                #     # convert predicitons to a single column (for Iris, Wine, BreastCancer)
                #     predictions = np.array(predictions).reshape(-1)


                batch_accuracies.append(np.sum(predictions == targets) / targets.shape[0])
                logging.info(f"Batch {batch_idx} of {num_batches_test} had accuracy {100*batch_accuracies[batch_idx]:.1f}%")
                metrics.update_test_accuracy(batch_accuracies[batch_idx])
                metrics.update_test_loss(0)
            logging.info(f"Average accuracy on test set: {100*np.mean(batch_accuracies):.1f}% +/- {100*np.std(batch_accuracies):.1f}%")
            metrics.add_epoch_test_accuracy(100 * sum(batch_accuracies) / len(batch_accuracies))
            metrics.add_epoch_test_loss(0)

        train()
        test()

        return metrics

if __name__ == "__main__":
    run()





