"""
Original by Michael Nielsen.
Modified by Joseph Palermo.

A library to load the MNIST image data.
"""

import cPickle
import gzip
import numpy as np
from DataSet import DataSet

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    """
    f = gzip.open('data/mnist_data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Convert MNIST data into a format convenient for consumption by a machine
    learning algorithm."""
    # load data as tuples of features and labels
    tr_d, va_d, te_d = load_data()
    # extract features and labels, and convert labels to one-hot format
    tr_features, tr_labels = tr_d[0], convert_labels_to_one_hot(tr_d[1])
    va_features, va_labels = va_d[0], convert_labels_to_one_hot(va_d[1])
    te_features, te_labels = te_d[0], convert_labels_to_one_hot(te_d[1])
    # wrap data in DataSet
    training_data = DataSet(tr_features, tr_labels)
    validation_data = DataSet(va_features, va_labels)
    test_data = DataSet(te_features, te_labels)
    return training_data, validation_data, test_data

# return a one-hot 1-d np array corresponding to the integer argument
def vectorized_result(j):
    e = np.zeros(10)
    e[j] = 1.0
    return e

# convert a 1-d numpy array into a 2-d array with one-hot entries
def convert_labels_to_one_hot(labels, max_label=10):
    one_hot_labels = np.zeros((labels.shape[0], max_label))
    for i, label in enumerate(labels):
        one_hot_labels[i] = vectorized_result(label)
    return one_hot_labels
