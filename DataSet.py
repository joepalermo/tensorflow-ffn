"""This code is inspired by the DataSet class which is part of the tensorflow
MNIST tutorial."""

import numpy as np

class DataSet(object):
    """This is a utility class to represent a dataset before consumption by a
    batch-based machine learning algorithm"""

    def __init__(self, features, labels, training=True):
        self.features = features
        self.labels = labels
        self.training = training
        self.num_examples = features.shape[0]
        self.example_index = 0

    # shuffle the data
    def _shuffle(self):
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self.features = self.features[perm]
        self.labels = self.labels[perm]

    def next_batch(self, batch_size):
        """Returns a mini-batch of size 'batch_size' as a tuple containing a
        numpy array of features and labels, respectively. If the DataSet is for
        training (as opposed to validation or testing) then it will be shuffled
        before returning the first mini-batch of an epoch."""
        # if this is the start of a training epoch, shuffle the training data
        if self.training and self.example_index == 0:
            self._shuffle()
        batch_xs = self.features[self.example_index : self.example_index + batch_size]
        batch_ys = self.labels[self.example_index : self.example_index + batch_size]
        self.example_index += batch_size
        # if this is the end of an epoch, reset example index
        if self.example_index + batch_size > self.num_examples:
            self.example_index = 0
        return batch_xs, batch_ys

    def epoch_generator(self, batch_size):
        """A generator that returns all batches of a given size. This method
        doesn't perform any shuffling and is intended for validation and test
        epochs, not for training."""
        num_batches = self.num_examples // batch_size
        for batch_i in xrange(num_batches):
            batch_xs = self.features[batch_i * batch_size: (batch_i + 1) * batch_size]
            batch_ys = self.labels[batch_i * batch_size: (batch_i + 1) * batch_size]
            yield (batch_xs, batch_ys)
