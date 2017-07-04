import tensorflow as tf
from tensorflow.contrib.distributions import Categorical

class Decoder:
    """
    The sentences (with the <SOS> marker) represented by a sequence of one-hot
    vectors. Their shape is [batch_size, max_length, embedding_size]
    """
    def __init__(self, sentences, features, sequence_length,
                 embedding_size, max_length = 30):
        self.sentences       = sentences
        self.features        = features
        self.sequence_length = sequence_length
        # self.batch_size      = sentences.shape[0]
        self.num_steps       = max_length
        self.num_hidden      = 512
        self.embedding_size  = embedding_size

        self._create_weights()
        self._create_cells()
        self._build_model()
        self._build_sampler()

    def _create_weights(self):
        self.out_W = tf.Variable(tf.random_uniform([
            self.num_hidden,
            self.embedding_size
        ], -1, 1))

        self.out_b = tf.Variable(tf.random_uniform([
            self.embedding_size
        ], -1, 1))

    def _create_cells(self):
        self.cells = [
            tf.contrib.rnn.BasicLSTMCell(self.num_hidden),
            tf.contrib.rnn.BasicLSTMCell(self.num_hidden)
        ]

        self.states = [
            self.cells[0].zero_state(128, tf.float32),
            self.cells[1].zero_state(128, tf.float32)
        ]

    def _build_model(self):
        # The input sentences are limited to the first T steps (therefore
        # efectively dropping the last word).
        x = self.sentences[:, : self.num_steps, :]

        # The output sentences start at the 2nd word. This effectively means that
        # x and y have the same shape.
        y = self.sentences[:, 1 : self.num_steps + 1, :]

        # The size of features is [batch_size, feature_size]
        f = self.features[:, :]

        mask = tf.sequence_mask(self.sequence_length - 1,
                                maxlen = self.num_steps,
                                dtype = tf.float32)

        # We retrieve the initial states of both LSTMs.
        c_0, h_0 = self.states[0]
        c_1, h_1 = self.states[1]

        loss = 0.0
        acc  = 0.0

        training_sampler = []

        for t in range(self.num_steps):
            # Run the first LSTM.
            with tf.variable_scope('first_lstm', reuse = t > 0):
                output, (c_0, h_0) = self.cells[0](inputs = x[:, t, :],
                                                   state  = [ c_0, h_0 ])

            # Concatenate the output of the first LSTM with the features.
            input = tf.concat([ output, f ], axis = 1)

            with tf.variable_scope('second_lstm', reuse = t > 0):
                output, (c_1, h_1) = self.cells[1](inputs = input,
                                                   state  = [ c_1, h_1 ])

            logits = tf.nn.xw_plus_b(output, self.out_W, self.out_b)

            targets = y[:, t]

            ce = tf.nn.softmax_cross_entropy_with_logits(logits = logits,
                                                         labels = targets)

            loss += tf.reduce_sum(ce * mask[:, t])

            correct_pred = tf.equal(tf.argmax(logits,  axis = 1),
                                    tf.argmax(targets, axis = 1))
            correct_pred = tf.cast(correct_pred, tf.float32)
            correct_pred = correct_pred * mask[:, t] + 1 * (1 - mask[:, t])
            acc += tf.reduce_mean(correct_pred)

            training_sampler.append(tf.argmax(logits, axis = 1))

        self.accuracy = acc  / float(self.num_steps)
        self.loss     = loss / 128.0 / float(self.num_steps)

        self.training_sampler = tf.stack(training_sampler, axis = 1)

    def _build_sampler(self):
        # We start with the <SOS> token.
        x = tf.constant([
            [ 1.0 ] + [ 0.0 ] * (self.embedding_size - 1)
            for _ in range(128)
        ])
        # x[:, 0] = 1

        sampler = []

        # The size of features is [batch_size, feature_size]
        f = self.features[:, :]

        # We retrieve the initial states of both LSTMs.
        c_0, h_0 = self.states[0]
        c_1, h_1 = self.states[1]

        for t in range(self.num_steps):
            # Run the first LSTM.
            with tf.variable_scope('first_lstm', reuse = True):
                output, (c_0, h_0) = self.cells[0](inputs = x,
                                                   state  = [ c_0, h_0 ])

            # Concatenate the output of the first LSTM with the features.
            input = tf.concat([ output, f ], axis = 1)

            with tf.variable_scope('second_lstm', reuse = True):
                output, (c_1, h_1) = self.cells[1](inputs = input,
                                                   state  = [ c_1, h_1 ])

            logits = tf.nn.xw_plus_b(output, self.out_W, self.out_b)

            x = tf.nn.softmax(logits)
            word = tf.argmax(logits, axis = 1)

            sampler.append(word)

        self.sampler = tf.transpose(sampler)
