"""
TODO: The logits weights and bias are always removed right now. We should add a
      flag to disable that during prediction. ~ Tim
      In inception_resnet_v2.py I also replaced the size of the AuxLogits layer
      with a hardcoded 1001 classes because we won't be able to retrain that
      layer. Instead, we only change the shape of the last logits layer. ~ Tim
      I also added stop_gradient to inception_resnet_v2.py. ~ Tim
"""

import tensorflow as tf
from core.inception_resnet_v2 import *
import numpy as np
from scipy.misc import imresize
from scipy.ndimage import imread
from lib.model import Model

class VisionModel(Model):
    def __init__(self, num_classes, learning_rate = 0.01):
        self.num_classes   = num_classes
        self.learning_rate = learning_rate

        self._create_placeholders()
        self._build_model()

        Model.__init__(self)

    def _create_placeholders(self):
        self.image_data = tf.placeholder(tf.float32,
                                         shape = (None, 299, 299, 3))
        self.labels     = tf.placeholder(tf.float32,
                                         shape = (None, self.num_classes))

    def _build_model(self):
        scope = inception_resnet_v2_arg_scope()
        with tf.contrib.slim.arg_scope(scope):
            self.logits, self.end_points = \
            inception_resnet_v2(self.image_data, is_training = False,
                                num_classes = self.num_classes)

        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope = "InceptionResnetV2")

        ce = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits,
                                                     labels = self.labels)

        self.cost = tf.reduce_mean(ce)
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.optimizer = optimizer.minimize(self.cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(self.logits, 1),
                                tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def restore(self, session, path):
        # Remove the logits weights and bias.
        variables = self.variables[0 : -2]

        saver = tf.train.Saver(var_list = variables)
        saver.restore(session, path)

        # Collect variables that remain uninitialized.
        variables = tf.global_variables()
        initialized_variables = session.run([
            tf.is_variable_initialized(variable)
            for variable in variables
        ])

        # See which variables are uninitialized and initialize those.
        variables = [
            variable
            for i, variable in enumerate(variables)
            if not initialized_variables[i]
        ]
        session.run(tf.variables_initializer(variables))

    """
    This function loads and preprocesses the image at the given path by first
    resizing it to 299x299 pixels (3 channels), which is the shape accepted by
    Inception-ResNet v2, and subsequently transforming the range to [-1, 1],
    from [0, 256].
    """
    def _preprocess(self, path):
        image = imread(path, mode = 'RGB')

        image = imresize(image, (299, 299))
        image = image / 256.0
        image = image - 0.5
        image = image * 2.0

        return image

    """
    This function returns class (0-999), features (1x1536), confidence (0-1).
    """
    def predict(self, session, path):
        predict_values, features, logit_values = session.run([
            self.end_points['Predictions'],
            self.end_points['PreLogitsFlatten'],
            self.logits
        ], feed_dict = {
            self.image_data: [
                self._preprocess(path)
            ]
        })

        return np.argmax(predict_values), features[0], np.max(predict_values)

    def _one_hot(self, index):
        return [ 1.0 if index == i else 0.0 for i in range(self.num_classes) ]

    """
    This function trains the underlying model on the given set of images and
    labels. The number of items in both arrays should be the same: that is the
    batch size. The input items should be paths to images that are loaded into
    memory when needed.
    """
    def train(self, session, paths, labels):
        session.run(self.optimizer, feed_dict = {
            self.image_data: [
                self._preprocess(path)
                for path in paths
            ],
            self.labels: [
                self._one_hot(label)
                for label in labels
            ]
        })

    """
    This function evaluates the given batch and returns the loss and accuracy.
    """
    def evaluate(self, session, paths, labels):
        acc, loss = session.run([ self.accuracy, self.cost ], feed_dict = {
            self.image_data: [
                self._preprocess(path)
                for path in paths
            ],
            self.labels: [
                self._one_hot(label)
                for label in labels
            ]
        })

        return acc, loss
