from __future__ import print_function
import tensorflow as tf
import numpy as np
from lib.ds import Vocabulary
from lib.sc import SentenceClassifier, SentenceReader
import lib.etc as etc
import sys

def sc_evaluate():
    max_length     = 30
    num_classes    = 200
    embedding_size = len(Vocabulary().restore("data/ds/vocabulary.csv"))

    reader = SentenceReader("data/sc/testing.csv",
                            num_classes, embedding_size = embedding_size)

    batch_size = 128
    epochs     = len(reader) / batch_size

    # Network Parameters
    num_hidden = 512 # hidden layer num of features

    tf.reset_default_graph()
    model = SentenceClassifier(max_length, embedding_size,
                               num_hidden, num_classes)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        model.restore(sess, "pretrained/sc/SentenceClassifier")

        total_acc = 0.0
        total_cnt = 0.0

        for step, pi in etc.range(epochs):
            # Get a batch of training instances.
            batch_x, batch_y, batch_len = reader.read(lines = batch_size)

            # Calculate batch accuracy and loss
            acc, loss = model.evaluate(batch_x, batch_y, batch_len, sess)

            total_acc += acc
            total_cnt += 1.0

            print("Iter " + str(1 + step) + " / " + str(epochs) + \
                  ", Validation Accuracy= " + \
                  "{:.5f}".format(total_acc / total_cnt) + \
                  ", Time Remaining= " + \
                  etc.format_seconds(pi.time_remaining()), file = sys.stderr)
