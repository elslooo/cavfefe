from __future__ import print_function
import tensorflow as tf
import numpy as np
from lib.ds import Vocabulary
import lib.lm as lm
import lib.etc as etc
import sys

def lm_train():
    max_length = 30

    vocabulary     = Vocabulary().restore("data/ds/vocabulary.csv")
    embedding_size = len(vocabulary)

    reader = lm.SentenceReader("data/lm/training.csv",
                               embedding_size = embedding_size)

    batch_size = 128
    epochs     = 100

    # Network Parameters
    num_hidden = 512 # hidden layer num of features

    model = lm.LanguageModel(max_length, embedding_size, num_hidden)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for step, pi in etc.range(epochs):
            # Get a batch of training instances.
            batch = reader.read(lines = batch_size)

            # Run optimization op (backprop)
            model.train(sess, *batch)

            # Calculate batch accuracy and loss
            acc, loss = model.evaluate(sess, *batch)

            print("Iter " + str(1 + step) + " / " + str(epochs) + \
                  ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc) + ", Time Remaining= " + \
                  etc.format_seconds(pi.time_remaining()), file = sys.stderr)

            # Generate a sample sentence after each 10 iterations.
            if (1 + step) % 10 == 0:
                features = [ 0 ] * 5 + [ 1 ] + [ 0 ] * 94

                sentence = model.generate(sess, features)

                print(vocabulary.sentence([
                    np.argmax(word) for word in sentence
                ]), file = sys.stderr)

                model.save(sess, step)

        print("Optimization Finished!", file = sys.stderr)
