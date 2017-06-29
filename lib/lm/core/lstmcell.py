from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import variable_scope as vs


class MultiLSTMCell(tf.contrib.rnn.MultiRNNCell):

    def __init__(self, cells, state_is_tuple=True):

        self._cells = cells
        self._state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        if self._state_is_tuple:
            return tuple(cell.state_size for cell in self._cells)
        else:
            return sum([cell.state_size for cell in self._cells])

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def __call__(self, inputs, im_features, state, scope=None):
        """Run this multi-layer cell on inputs, starting from state. ASSUMING STATE_IS_TUPLE=TRUE"""
        cur_inp = inputs
        new_states = []

        # Manually define first and second layer of LSTMs
        with vs.variable_scope("cell_0"):
            cur_state = state[0]
            cur_inp, new_state = self._cells[0](cur_inp, cur_state)
            new_states.append(new_state)
        with vs.variable_scope("cell_1"):
            cur_state = state[1]

            # Concatenate here
            # Original input is 128 x 200 in this case
            cur_inp = tf.concat([cur_inp, im_features], 1)

            cur_inp, new_state = self._cells[1](cur_inp, cur_state)
            new_states.append(new_state)

        new_states = tuple(new_states)

        return cur_inp, new_states
