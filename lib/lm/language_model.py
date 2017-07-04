from lib.lm.core import MultiLSTMCell, dynamic_rnn
from lib.model import Model

import numpy as np
import os
import tensorflow as tf
from lib.sc import SentenceClassifier
from tensorflow.contrib.distributions import Categorical
from decoder import Decoder

class LanguageModel(Model):
    def __init__(self, max_length, embedding_size, feature_size, num_hidden,
                 num_classes, learning_rate = 0.01):
        self.max_length     = max_length
        self.embedding_size = embedding_size
        self.feature_size   = feature_size
        self.num_hidden     = num_hidden
        self.num_classes    = num_classes
        self.learning_rate  = learning_rate

        with tf.variable_scope("LanguageModel"):
            self._create_placeholders()
            self.decoder = Decoder(self.x, self.f, self.seqlen,
                                   self.embedding_size)

        self._build_relevance_loss()
        self._build_discriminative_loss()

        with tf.variable_scope("LanguageModel"):
            self._build_model()

        Model.__init__(self)

    def _create_placeholders(self):
        # X is a batch of sentences. The sentences should be padded to n_steps.
        self.x = tf.placeholder(tf.float32, [ None, self.max_length + 1,
                                              self.embedding_size ], name = "x")

        # F is a batch of feature vectors: one for each training instance. These
        # features are retrieved from the image classification model (resnet).
        self.f = tf.placeholder(tf.float32, [ None, self.feature_size ],
                                name = "f")

        # This is the sequence length for each sentence in the batch. This does
        # include the start-of-sentence marker, but does not include the
        # end-of-sentence marker.
        self.seqlen = tf.placeholder(tf.int32, [ None ], name = "seqlen")

        # These are indices that indicate the class that a sentence belongs to.
        # This is used for the discriminative loss.
        self.y = tf.placeholder(tf.int32, [ None ], name = "y")

    def _build_model(self):
        self.cost = self.rel_loss + 0.25 * self.dis_loss

        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)

        self.optimizer = optimizer.minimize(self.cost)

        # Evaluate model
        # correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        # self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        self.accuracy = self.decoder.accuracy

        tf.summary.scalar("accuracy", self.accuracy)
        tf.summary.scalar("cost", self.cost)
        tf.summary.histogram("out_W", self.decoder.out_W)
        tf.summary.histogram("out_b", self.decoder.out_b)

        self.summary = tf.summary.merge_all()

    def _build_relevance_loss(self):
        self.rel_loss = self.decoder.loss

    def _build_discriminative_loss(self):
        samples = self.decoder.training_sampler
        samples = tf.one_hot(samples, self.embedding_size)

        self.sentence_classifier = \
        SentenceClassifier(self.max_length, self.embedding_size,
                           self.num_hidden, self.num_classes,
                           sentences = samples, sequence_lengths = self.seqlen)

        pred = self.sentence_classifier.pred

        y = tf.one_hot(self.y, self.num_classes)

      	loss = tf.nn.softmax_cross_entropy_with_logits(logits = pred,
                                                       labels = y)
        self.dis_loss = tf.reduce_sum(loss) / float(self.num_classes)

    """
    This function trains the language model on a batch of sentences.
    """
    def train(self, session, x, f, y, sequence_lengths):
        return session.run([ self.optimizer, self.summary ], feed_dict = {
            self.x: x,
            self.f: f,
            self.y: y,
            self.seqlen: sequence_lengths
        })[1]

    """
    This function evaluates the accuracy and loss of the language model on the
    given batch of sentences.
    """
    def evaluate(self, session, x, f, y, sequence_lengths):
        acc, loss, rel, dis = session.run([ self.accuracy, self.cost,
                                            self.rel_loss, self.dis_loss ],
                                          feed_dict = {
            self.x: x,
            self.f: f,
            self.y: y,
            self.seqlen: sequence_lengths
        })

        return acc, loss, rel, dis

    def generate(self, session, features):
        results = session.run(self.decoder.sampler, {
            self.f: features
        })

        return results
