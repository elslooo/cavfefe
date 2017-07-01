from lib.lm.core import MultiLSTMCell, dynamic_rnn
from lib.model import Model

import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.distributions import Categorical
from decoder import Decoder

class LanguageModel(Model):
    def __init__(self, max_length, embedding_size, feature_size, num_hidden,
                 learning_rate = 0.01):
        self.max_length     = max_length
        self.embedding_size = embedding_size
        self.feature_size   = feature_size
        self.num_hidden     = num_hidden
        self.learning_rate  = learning_rate

        with tf.variable_scope("LanguageModel"):
            self._create_placeholders()
            self._create_weights()
            self._build_model()

        Model.__init__(self)

    def _create_placeholders(self):
        # X is a batch of sentences. The sentences should be padded to n_steps.
        self.x = tf.placeholder("float", [ None, self.max_length,
                                           self.embedding_size ], name = "x")

        # F is a batch of feature vectors: one for each training instance. These
        # features are retrieved from the image classification model (resnet).
        self.f = tf.placeholder("float", [ None,
                                           self.feature_size ], name = "f")

        # This is the sequence length for each sentence in the batch. This does
        # include the start-of-sentence marker, but does not include the
        # end-of-sentence marker.
        self.seqlen = tf.placeholder(tf.int32, [ None ], name = "seqlen")

        # This is an embedding of the next word (that followes the partial
        # sentence in X), again in batch.
        self.y = tf.placeholder("float", [ None, self.embedding_size ],
                                name = "y")

    def _create_weights(self):
        # These weights are used to get from the output layer of the first LSTM
        # / input layer of the second LSTM to the output layer of the second
        # LSTM. Note that in our implementation, this does not include the
        # feature embedding size.
        self.out_W = tf.Variable(tf.random_uniform([
            self.num_hidden,
            self.embedding_size
        ], -1, 1))

        self.out_b = tf.Variable(tf.random_uniform([
            self.embedding_size
        ], -1, 1))

    def _build_model(self):
        self.decoder = Decoder(self.x, self.f, self.seqlen, self.label, self.embedding_size)
        self.cost = self.decoder.loss

        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)

        self.optimizer = optimizer.minimize(self.cost)

        # Evaluate model
        # correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        # self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        self.accuracy = self.decoder.accuracy

        tf.summary.scalar("accuracy", self.accuracy)
        tf.summary.scalar("cost", self.cost)
        tf.summary.histogram("out_W", self.out_W)
        tf.summary.histogram("out_b", self.out_b)

        self.summary = tf.summary.merge_all()

    """
    This function trains the language model on a batch of sentences.
    """
    def train(self, session, x, f, y, label, sequence_lengths):
        return session.run([ self.optimizer, self.summary ], feed_dict = {
            self.x: x,
            self.f: f,
            self.y: y,
            self.label: label,
            self.seqlen: sequence_lengths
        })[1]

    """  
    This function evaluates the accuracy and loss of the language model on the
    given batch of sentences.
    """
    def evaluate(self, session, x, f, y, sequence_lengths):
        acc, loss = session.run([ self.accuracy, self.cost ], feed_dict = {
            self.x: x,
            self.f: f,
            self.y: y,
            self.seqlen: sequence_lengths
        })

        return acc, loss

    def generate(self, session, features):
        results = session.run(self.decoder.sampler, {
            self.f: features
        })

        return results
        # sos = [ 1 ] + [ 0 ] * (self.embedding_size - 1)
        # eos = [ 0, 1 ] + [ 0 ] * (self.embedding_size - 2)
        #
        # sentence = [ sos ]
        #
        # for i in range(self.max_length - 1):
        #     input    = sentence[:]
        #
        #     while len(input) < self.max_length:
        #         input = input + [ eos ]
        #
        #     input = np.array([ input ])
        #
        #     output = session.run(self.pred_softmax, feed_dict = {
        #         self.x: input,
        #         self.f: [ features ],
        #         self.seqlen: [ len(sentence) ]
        #     })[0]
        #
        #     word_index = np.argmax(output)
        #
        #     sentence.append(output)
        #
        # return sentence
