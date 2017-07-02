import tensorflow as tf
from tensorflow.contrib.distributions import Categorical

class LSTM:
    """
    The sentences (with the <SOS> marker) represented by a sequence of one-hot
    vectors. Their shape is [batch_size, max_length, embedding_size]
    """
    def __init__(self, sentences, sequence_length, num_hidden, max_length = 30):
        self.sentences       = sentences
        self.sequence_length = sequence_length
        self.num_steps       = max_length
        self.num_hidden      = num_hidden

        self._create_cells()
        self._build_model()

    def _create_cells(self):
        self.cell  = tf.contrib.rnn.BasicLSTMCell(self.num_hidden)
        self.state = self.cell.zero_state(128, tf.float32)

    def _build_model(self):
        x = self.sentences[:, : self.num_steps, :]

        mask = tf.sequence_mask(self.sequence_length, maxlen = self.num_steps,
                                dtype = tf.float32)

        c_0, h_0 = self.state

        curr_output = tf.zeros((self.num_hidden, 128), dtype = tf.float32)

        for t in range(self.num_steps):
            with tf.variable_scope('lstm', reuse = t > 0):
                next_output, (c_0, h_0) = self.cell(inputs = x[:, t, :],
                                                    state  = [ c_0, h_0 ])
                next_output = tf.transpose(next_output)

                curr_output = curr_output * (1.0 - mask[:, t]) + \
                              next_output * (0.0 + mask[:, t])


        self.output = tf.transpose(curr_output)
