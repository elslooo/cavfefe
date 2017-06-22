import tensorflow as tf
import numpy as np
from idx2word import idx2sentence, embedding_size
import lib.lm as lm
from lib.lm.sentenceclass import SentenceClassifier
from lib.lm.class_reader import SentenceClassReader
import lib.etc as etc

max_length = 15
nr_classes = 200

reader = SentenceClassReader("data/produced_sentences_classifier.csv",
                        nr_classes, embedding_size = embedding_size)

def get_batch(batch_size):
    return reader.read(lines = batch_size)

batch_size = 128
epochs     = int(942878 / batch_size)

# Network Parameters
num_hidden = 512 # hidden layer num of features

model = SentenceClassifier(max_length, embedding_size, num_hidden, nr_classes)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step, pi in etc.range(epochs):
        # Get a batch of training instances.
        batch_x, batch_y, batch_len = get_batch(batch_size)

        # Run optimization op (backprop)
        model.train(batch_x, batch_y, batch_len, sess)

        # Calculate batch accuracy and loss
        acc, loss = model.evaluate(batch_x, batch_y, batch_len, sess)

        print("Iter " + str(1 + step) + " / " + str(epochs) + \
              ", Minibatch Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc) + ", Time Remaining= " + \
              etc.format_seconds(pi.time_remaining()))

        # Generate a sample sentence after each 10 iterations.
        # if (1 + step) % 10 == 0:
        #     features = [ 0 ] * 5 + [ 1 ] + [ 0 ] * 94

            # sentence = model.generate(sess)

            # print(idx2sentence([ np.argmax(word) for word in sentence ]))

    print("Optimization Finished!")
