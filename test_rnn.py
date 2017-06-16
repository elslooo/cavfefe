import tensorflow as tf 
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

batch_size = 128
# Network Parameters
n_input = 28  # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# Placeholder for the inputs in a given iteration	
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, 10])

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0) for _ in range(2)])


x2 = tf.unstack(x, n_steps, 1)
# Initial state of the LSTM memory.
init_state = lstm.zero_state(batch_size, tf.float32)
outputs, final_state = tf.contrib.rnn.static_rnn(lstm, x2, initial_state=init_state)

# CHANGE INTO DYNAMIC
# initial_state = state = lstm.zero_state(tf.random_normal([batch_size]))
# outputs, final_state = tf.nn.dynamic_rnn(cell, x2, initial_state=init_state)

prediction = tf.matmul(outputs[-1], weights['out']) + biases['out']

optimizer = tf.nn.softmax(prediction)

# C-E LOSS
# cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	batch_x, batch_y = mnist.train.next_batch(batch_size)
	batch_x = batch_x.reshape((batch_size, n_steps, n_input))
	sess.run(optimizer, feed_dict={x: batch_x})
	print outputs


 

