import sys
import tensorflow as tf
from tf_network3 import Network
from tf_network3 import FullyConnectedLayer, SoftmaxLayer
from mnist_loader import load_data_wrapper
from DataSet import DataSet

def main(_):

    # fetch the training data
    (training_data, validation_data, test_data) = load_data_wrapper()

    # hyperparameters
    epochs = 5
    mini_batch_size = 10
    eta = 4

    # constrct the network
    layers = [FullyConnectedLayer(784, 30), SoftmaxLayer(30, 10)]
    net = Network(layers, mini_batch_size, classifier=True)

    # train
    net.SGD(training_data, epochs, mini_batch_size, eta, validation_data, test_data)


if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])
