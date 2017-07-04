from __future__ import print_function
import tensorflow as tf
import numpy as np
from lib.ds import Dataset, Vocabulary
from lib.cv import FeatureCache
from lib.sc import EmbeddingCache
import lib.lm as lm
import lib.etc as etc
import sys
import os
import json

mode = "best" # best | worst | random

def experiment(labels):
    if mode == "best":
        return labels[-1]
    elif mode == "worst":
        return labels[0]
    else:
        return np.random.random_integers(1, 200)

def lm_generate():
    max_length = 30

    dataset = Dataset()

    vocabulary     = Vocabulary().restore("data/ds/vocabulary.csv")
    embedding_size = len(vocabulary)
    feature_size   = 512 + 1536

    reader = lm.SentenceReader("data/lm/training.csv",
                               embedding_size = embedding_size)

    batch_size = 128
    epochs     = len(reader) / batch_size

    # Network Parameters
    num_hidden = 512 # hidden layer num of features

    model = lm.LanguageModel(max_length     = max_length,
                             embedding_size = embedding_size,
                             feature_size   = feature_size,
                             num_hidden     = num_hidden,
                             num_classes    = 200)

    init = tf.global_variables_initializer()

    embedding_cache = EmbeddingCache()
    embedding_cache.restore("data/sc/embeddings.csv")

    feature_cache = FeatureCache()
    feature_cache.restore("data/cv/features.csv")

    try:
        os.makedirs("logs")
    except:
        pass

    writer = tf.summary.FileWriter("logs", graph = tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(init)

        model.restore(sess, "pretrained/lm/LanguageModel")

        results = []

        for step, pi in etc.range(epochs):
            # Get a batch of training instances.
            instances, labels, sentences, lengths = \
            reader.read(lines = batch_size)

            features = [
                feature_cache.get(index)
                for index in instances
            ]

            labels = [ label for label, feature in features ]

            features = [
                np.concatenate([ embedding_cache.get(experiment(label)), feature ])
                for label, feature in features
            ]

            sentences = model.generate(sess, features)

            print("Iter " + str(1 + step) + " / " + str(epochs) + \
                  ", Time Remaining= " + \
                  etc.format_seconds(pi.time_remaining()), file = sys.stderr)

            for i, sentence in enumerate(sentences):
                results.append({
                    "image_id": dataset.example(instances[i]).path + ".jpg",
                    "caption": vocabulary.sentence([
                        word for word in sentence
                    ], limit = True)
                })

            if step % 10 == 0:
                try:
                    os.makedirs("products")
                except:
                    pass

                with open('products/sentences.best.json', 'w') as file:
                    json.dump(results, file)

        print("Optimization Finished!", file = sys.stderr)
