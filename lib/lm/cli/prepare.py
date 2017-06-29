import numpy as np
from random import shuffle
import os
import csv
from lib.ds import Dataset, Vocabulary

def lm_prepare():
    dataset = Dataset()

    max_length = 30

    training   = []
    testing    = []
    vocabulary = Vocabulary()
    vocabulary.restore("data/ds/vocabulary.csv")

    sos = vocabulary.index("<SOS>")
    eos = vocabulary.index("<EOS>")

    def compute_in_out(words):
        pairs = []

        words = words[0 : max_length]

        if len(words) == max_length:
            words[-1] = eos

        for i in range(len(words) - 1):
            x = words[0 : i + 1]
            x = np.pad(x, (0, max_length - len(x)), 'constant',
                       constant_values = eos)
            y = words[i + 1]

            pairs.append((x, y, i + 1))

        return pairs

    for example in dataset.examples():
        data = example.sentences()

        for sentence in data:
            words = sentence

            words  = [ sos ] + \
                     [ vocabulary.index(word) for word in words ] + \
                     [ eos ]
            subsets = compute_in_out(words)

            # for pair in subsets:
            pair = subsets[-1]
            if example.is_training:
                training.append([ example.species, example.id,
                                  pair[0], pair[1], pair[2] ])
            else:
                testing.append([ example.species, example.id,
                                 pair[0], pair[1], pair[2] ])

    try:
        os.makedirs("data/lm")
    except:
        pass

    with open('data/lm/training.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow([
            'class', 'instance', 'sentence', 'next_word', 'length'
        ])

        shuffle(training)
        for species, text, pair_in, pair_out, pair_len in training:
            writer.writerow([ species, text,
                              '|'.join([ str(idx) for idx in pair_in]),
                              pair_out, pair_len ])

    with open('data/lm/testing.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow([
            'class', 'instance', 'sentence', 'next_word', 'length'
        ])

        shuffle(testing)
        for species, text, pair_in, pair_out, pair_len in testing:
            writer.writerow([ species, text,
                              '|'.join([ str(idx) for idx in pair_in]),
                              pair_out, pair_len ])
