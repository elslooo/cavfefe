import tensorflow as tf
import numpy as np
from idx2word import idx2sentence, embedding_size

filename_queue = tf.train.string_input_producer(["data/produced_sentences.csv"])

reader = tf.TextLineReader(skip_header_lines = True)

def get_batch(batch_size):
    key, value = reader.read_up_to(filename_queue, batch_size)
    label, instance, sentences, words = tf.decode_csv(value, [ [''], [''], [''], [''] ])

    sentences = tf.map_fn(lambda s: tf.string_split([ s ], delimiter = "|").values, sentences)
    sentences = tf.string_to_number(sentences, out_type = tf.int32)
    sentences = tf.one_hot(sentences, embedding_size)

    words = tf.string_to_number(words, out_type = tf.int32)
    words = tf.one_hot(words, embedding_size)

    return sentences.eval(), words.eval()

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord = coord)
#     # print(sess.run(sentences))
#
# exit(0)

learning_rate = 0.001
training_iters = 100000
batch_size = 128
epochs     = int(942878 / batch_size)
display_step = 10
# Network Parameters
n_input = embedding_size  # MNIST data input (img shape: 28*28)
n_steps = 10 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = embedding_size # MNIST total classes (0-9 digits)

# Placeholder for the inputs in a given iteration
x = tf.placeholder("float", [None, n_steps, n_input], name = "x")
y = tf.placeholder("float", [None, n_classes], name = "y")

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

pred= tf.matmul(outputs[-1], weights['out']) + biases['out']

optimizer = tf.nn.softmax(pred)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# C-E LOSS
# cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

# LEKKER GEKOPIEERD DIT

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)

    for step in range(1, epochs):
        batch_x, batch_y = get_batch(batch_size = batch_size)

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        # if step % display_step == 0:
        # Calculate batch accuracy
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        # Calculate batch loss
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print("Iter " + str(step) + " / " + str(epochs) + \
              ", Minibatch Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))

        if step % 10 == 0:
            sos = tf.one_hot([0], embedding_size).eval()[0]
            eos = tf.one_hot([1], embedding_size).eval()[0]

            sentence = [ sos ]

            for i in range(n_steps):
                input    = sentence[:]

                while len(input) < n_steps:
                    input.append(eos)

                input = np.array([input for i in range(128)])

                output = sess.run(pred, feed_dict = { x: input })[0]

                sentence.append(output)

            print(idx2sentence([ np.argmax(word) for word in sentence ]))

    print("Optimization Finished!")

    # # Calculate accuracy for 128 mnist test images
    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    # test_label = mnist.test.labels[:test_len]
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
