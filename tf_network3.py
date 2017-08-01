"""This code is a Tensorflow clone of the Michael Nielsen's network3.py"""

import numpy as np
import tensorflow as tf

class Network(object):

    def __init__(self, layers, mini_batch_size, classifier=True):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = tf.placeholder(tf.float32, [None, layers[0].n_in]) # None means any size
        self.y = tf.placeholder(tf.float32, [None, layers[-1].n_out])
        self.classifier = classifier
        # construct the computation graph up to the network output
        inpt = self.x
        for layer in self.layers:
            layer.set_inpt(inpt, mini_batch_size)
            inpt = layer.output
        self.output = self.layers[-1].output

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data):
        """Perform stochastic gradient descent."""
        # compute the number of mini-batches per epoch
        num_training_batches = training_data.num_examples // mini_batch_size
        num_validation_batches = validation_data.num_examples // mini_batch_size
        num_test_batches = test_data.num_examples // mini_batch_size

        # define the cost function and optimization procedure
        cost = self.layers[-1].cost(self)
        train_step = tf.train.GradientDescentOptimizer(eta).minimize(cost)

        # if the network is intended to be a classifier then it can use accuracy
        # as a validation/test metric
        if self.classifier:
            correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            test_metric = accuracy
        else:
            test_metric = cost

        # perform training and validation/test
        best_validation_result = 0.0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch_i in range(epochs):
                for mini_batch_i in range(num_training_batches):
                    iteration = num_training_batches * epoch_i + mini_batch_i
                    if iteration % 1000 == 0:
                        print("Training mini-batch number " + str(mini_batch_i))
                    batch_xs, batch_ys = training_data.next_batch(mini_batch_size)
                    # train
                    train_step.run(feed_dict= {self.x: batch_xs, self.y: batch_ys})
                    # validation/test phase
                    if (iteration + 1) % num_training_batches == 0:
                        validation_result = np.mean([sess.run(test_metric, feed_dict={self.x: batch_xs, self.y: batch_ys})
                                            for (batch_xs, batch_ys) in validation_data.epoch_generator(mini_batch_size)])
                        print("Epoch " + str(epoch_i) + ": validation metric " + str(validation_result))
                        if validation_result >= best_validation_result:
                            print("This is the best validation metric to date.")
                            best_validation_result = validation_result
                            best_iteration = iteration
                            if test_data:
                                test_result = np.mean([sess.run(test_metric, feed_dict={self.x: batch_xs, self.y: batch_ys})
                                              for (batch_xs, batch_ys) in test_data.epoch_generator(mini_batch_size)])
                                print('The corresponding test metric is {0:.2}'.format(
                                    test_result))
            print("Finished training network.")
            print("Best validation metric of {0:.2} obtained at iteration {1}".format(
                best_validation_result, best_iteration))
            print("Corresponding test metric of {0:.2}".format(test_result))


class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=tf.nn.sigmoid):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.w = tf.Variable(tf.random_normal([n_in, n_out]))
        self.b = tf.Variable(tf.random_normal([n_out]))
        self.params = [self.w, self.b]

    # construct the computation graph from the layer input to the layer output
    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = tf.reshape(inpt, (mini_batch_size, self.n_in))
        self.output = self.activation_fn(tf.matmul(self.inpt, self.w) + self.b)
        self.y_out = tf.argmax(self.output, axis=1)

    def cost(self, net):
        """MSE cost function."""
        return 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(self.output - net.y), axis=1))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.w = tf.Variable(tf.random_normal([n_in, n_out]))
        self.b = tf.Variable(tf.random_normal([n_out]))
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = tf.reshape(inpt, (mini_batch_size, self.n_in))
        self.output = tf.matmul(self.inpt, self.w) + self.b
        self.y_out = tf.argmax(self.output, axis=1)

    # numerically stable
    def cost(self, net):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=net.y, logits=self.output))

    # may be numerically unstable
    def cost2(self, net):
        return -tf.reduce_mean(tf.reduce_sum(net.y * tf.log(tf.nn.softmax(self.output)), reduction_indices=[1]))
