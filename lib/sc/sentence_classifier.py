import tensorflow as tf
import numpy as np
from lib.model import Model
from lib.sc.lstm import LSTM

class SentenceClassifier(Model):
    def __init__(self, max_length, embedding_size, num_hidden, num_classes,
                 learning_rate = 0.01, sentences = None,
                 sequence_lengths = None):
        self.max_length     = max_length
        self.embedding_size = embedding_size
        self.num_hidden     = num_hidden
        self.num_classes    = num_classes
        self.learning_rate  = learning_rate

        with tf.variable_scope("SentenceClassifier"):
            self._create_placeholders()
            self._create_weights()
            self._build_model(sentences, sequence_lengths)

        Model.__init__(self)

    def _create_placeholders(self):
        # X is a batch of sentences. The sentences should be padded to n_steps.
        self.x = tf.placeholder("float", [ None, self.max_length,
                                           self.embedding_size ], name = "x")

        # This is the sequence length for each sentence in the batch. This does
        # include the start-of-sentence marker, but does not include the
        # end-of-sentence marker.
        self.seqlen = tf.placeholder(tf.int32, [ None ], name = "seqlen")

        # This is the class prediction distribution
        self.y = tf.placeholder("float", [ None, self.num_classes],
                                name = "y")

    def _create_weights(self):
        self.out_W = tf.Variable(tf.random_normal([
            self.num_hidden,
            self.num_classes
        ]), name = "out_W")

        self.out_b = tf.Variable(tf.random_normal([
            self.num_classes
        ]), name = "out_b")

    def _build_model(self, sentences = None, sequence_lengths = None):
        if sentences is None or sequence_lengths is None:
            self.lstm = LSTM(self.x, self.seqlen, self.num_hidden,
                             max_length = self.max_length)
        else:
            self.lstm = LSTM(sentences, sequence_lengths, self.num_hidden,
                             max_length = self.max_length)

        self.hidden = self.lstm.output

        self.pred = tf.matmul(self.hidden, self.out_W) + self.out_b
        self.pred_softmax = tf.nn.softmax(self.pred)

        if sentences is not None and sequence_lengths is not None:
            self.pred_softmax = tf.stop_gradient(self.pred_softmax)
            return

        ce = tf.nn.softmax_cross_entropy_with_logits(logits = self.pred,
                                                     labels = self.y)
        self.cost = tf.reduce_mean(ce)
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.optimizer = optimizer.minimize(self.cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    """
    This function trains the language model on a batch of sentences.
    """
    def train(self, x, y, sequence_lengths, session):
        session.run(self.optimizer, feed_dict = {
            self.x: x,
            self.y: y,
            self.seqlen: sequence_lengths
        })

    """
    This function extracts the activations of the hidden layer for the given
    batch of sentences.
    """
    def extract(self, session, x, y, sequence_lengths):
        return session.run(self.hidden, feed_dict = {
            self.x: x,
            self.y: y,
            self.seqlen: sequence_lengths
        })

    def evaluate(self, x, y, sequence_lengths, session):
        acc, loss = session.run([ self.accuracy, self.cost ], feed_dict = {
            self.x: x,
            self.y: y,
            self.seqlen: sequence_lengths
        })

        return acc, loss
