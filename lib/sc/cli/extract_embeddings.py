from __future__ import print_function
import lib.etc as etc
import sys
import tensorflow as tf
from lib.ds import Vocabulary
from lib.sc import SentenceClassifier, SentenceReader

def sc_extract_embeddings():
    path = 'checkpoints/SentenceClassifier-10'
    batch_size = 1024

    max_length  = 30
    num_classes = 200

    embedding_size = len(Vocabulary().restore("data/ds/vocabulary.csv"))

    reader = SentenceReader("data/produced_sentences_classifier.csv",
                            num_classes, embedding_size = embedding_size)

    sess  = tf.Session()
    # sess.run(tf.global_variables_initializer())

    num_hidden = 512 # hidden layer num of features

    model = SentenceClassifier(max_length, embedding_size, num_hidden, num_classes)

    model.restore(sess, path)

    # Get a batch of training instances.
    x, y, seqlen = reader.read(lines = 10)

    # Calculate batch accuracy and loss
    embeddings = model.extract(sess, x, y, seqlen)

    print(embeddings)
    print(embeddings.shape)
