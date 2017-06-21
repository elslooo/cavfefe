from lib.lstmcell import MultiLSTMCell
from lib.rnn import dynamic_rnn, static_rnn
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.001
training_iters = 100000
display_step = 10

# Generate image features
batch_size = 128
nr_features = 2
im_features = tf.random_normal([batch_size, nr_features])

# Network Parameters
n_input = 28  # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 200 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# Placeholder for the inputs in a given iteration
x = tf.placeholder("float", [None, None, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

lstm = MultiLSTMCell([tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0) for _ in range(2)])

	
# x2 = tf.unstack(x, n_steps, 1)

# Initial state of the LSTM memory.
init_state = lstm.zero_state(batch_size, tf.float32)
# outputs, final_state = static_rnn(lstm, x2, im_features, initial_state=init_state)
outputs, final_state = dynamic_rnn(lstm, x, im_features, initial_state=init_state)

pred = tf.matmul(outputs[:, -1], weights['out']) + biases['out']

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# INCLUDE DROPOUT

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        # print sess.run(outputs, feed_dict={x:batch_x, y:batch_y})[:,-1].shape
        # print outputs.eval().shape
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))


