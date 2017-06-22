import tensorflow as tf
import numpy as np
from lib.model import Model

class SentenceClassifier(Model):
    def __init__(self, max_length, embedding_size, num_hidden, num_classes,
                 learning_rate = 0.01):
        self.max_length     = max_length
        self.embedding_size = embedding_size
        self.num_hidden     = num_hidden
        self.num_classes    = num_classes
        self.learning_rate  = learning_rate

        self._create_placeholders()
        self._create_weights()
        self._build_model()

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
        ]))

        self.out_b = tf.Variable(tf.random_normal([
            self.num_classes
        ]))

    def _build_model(self):
        lstm = tf.contrib.rnn.BasicLSTMCell(self.num_hidden, forget_bias = 1.0)

        outputs, final_state = tf.nn.dynamic_rnn(lstm, self.x,
                                                 dtype = tf.float32)

        self.pred = tf.matmul(outputs[:, -1], self.out_W) + self.out_b
        self.pred_softmax = tf.nn.softmax(self.pred)

        self.cost = \
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.pred,
                                                               labels = self.y))
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

    def evaluate(self, x, y, sequence_lengths, session):
        acc, loss = session.run([ self.accuracy, self.cost ], feed_dict = {
            self.x: x,
            self.y: y,
            self.seqlen: sequence_lengths
        })

        return acc, loss
