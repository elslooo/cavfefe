from lib.lstmcell import MultiLSTMCell
from lib.static_rnn import static_rnn
# from tensorflow.contrib.rnn import static_rnn
import tensorflow as tf
import numpy as np
from idx2word import idx2sentence, embedding_size

max_length = 6

filename_queue = tf.train.string_input_producer(["data/produced_sentences.csv"])

reader = tf.TextLineReader(skip_header_lines = True)

def get_batch(batch_size):
    key, value = reader.read_up_to(filename_queue, batch_size)
    label, instance, sentences, words, length = tf.decode_csv(value, [ [''], [''], [''], [''], [''] ])

    sentences = sentences.eval()
    sentences = [ sentence.split("|") for sentence in sentences ]
    sentences = [ [ int(word) for word in sentence] for sentence in sentences ]
    sentences = tf.one_hot(sentences, embedding_size, name = "word_one_hot")

    words = words.eval()
    words = [ int(word) for word in words ]
    words = tf.one_hot(words, embedding_size, name = "target_one_hot")

    length = tf.string_to_number(length, out_type = tf.int32, name = "sentence_length")

    label    = tf.string_to_number(label, out_type = tf.int32, name = "label")
    features = tf.one_hot(label, 100, name = "features")

    return sentences.eval(), words.eval(), length.eval(), features.eval()

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord = coord)
#     # print(sess.run(sentences))
#
# exit(0)

learning_rate = 0.01
training_iters = 100000
batch_size = 128
epochs     = int(942878 / batch_size)
# Network Parameters
n_input = embedding_size  # MNIST data input (img shape: 28*28)
n_steps = max_length # timesteps
n_hidden = 512 # hidden layer num of features
n_classes = embedding_size # MNIST total classes (0-9 digits)

# Placeholder for the inputs in a given iteration
x = tf.placeholder("float", [None, n_steps, n_input], name = "x")
f = tf.placeholder("float", [None, 100], name = "f")
seqlen = tf.placeholder(tf.int32, [None], name = "seqlen")
y = tf.placeholder("float", [None, n_classes], name = "y")

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

lstm = MultiLSTMCell([
    tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
    for _ in range(2)
])

x2 = tf.unstack(x, n_steps, 1)

# Initial state of the LSTM memory.
init_state = lstm.zero_state(batch_size, tf.float32)
outputs, final_state = static_rnn(lstm, x2, f, dtype = tf.float32)
                                        #  sequence_length = seqlen)

# CHANGE INTO DYNAMIC
# initial_state = state = lstm.zero_state(tf.random_normal([batch_size]))
# outputs, final_state = tf.nn.dynamic_rnn(cell, x2, initial_state=init_state)

pred = tf.matmul(outputs[-1], weights['out']) + biases['out']

# print(outputs.shape)

pred_softmax = tf.nn.softmax(pred)

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

    sos = tf.one_hot([0], embedding_size).eval()[0]
    eos = tf.one_hot([1], embedding_size).eval()[0]

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)

    for step in range(1, epochs):
        batch_x, batch_y, batch_len, batch_f = get_batch(batch_size = batch_size)

        # print(batch_f)

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={
            x: batch_x,
            f: batch_f,
            y: batch_y,
            seqlen: batch_len
        })

        # Calculate batch accuracy and loss
        acc, loss = sess.run([ accuracy, cost ], feed_dict={
            x: batch_x,
            f: batch_f,
            y: batch_y,
            seqlen: batch_len
        })

        print("Iter " + str(step) + " / " + str(epochs) + \
              ", Minibatch Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))

        if step % 10 == 0:
            sentence = [ sos ]

            for i in range(n_steps - 1):
                input    = sentence[:]

                while len(input) < n_steps:
                    input = input + [ eos ]

                features = [ 0 ] * 5 + [ 1 ] + [ 0 ] * 94

                input = np.array([input for i in range(batch_size)])

                output = sess.run(pred_softmax, feed_dict = {
                    x: input,
                    f: [ features for i in range(batch_size) ],
                    seqlen: [ len(sentence) for i in range(batch_size) ]
                })[0]

                word_index = np.argmax(output)

                sentence.append(output)

            print(idx2sentence([ np.argmax(word) for word in sentence ]))

    print("Optimization Finished!")

    # # Calculate accuracy for 128 mnist test images
    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    # test_label = mnist.test.labels[:test_len]
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
