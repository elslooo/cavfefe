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

    for example in dataset.examples():
        data = example.sentences()

        for sentence in data:
            words = sentence

            words  = [ sos ] + \
                     [ vocabulary.index(word) for word in words ] + \
                     [ eos ]

            # Add padding.
            while len(words) < max_length + 1:
                words.append(eos)

            if len(words) > max_length + 1:
                words = words[0 : max_length] + [ eos ]

            # for pair in subsets:
            if example.is_training:
                training.append([ example.species, example.id,
                                  words, min(len(sentence), max_length - 1) + 2 ])
            else:
                testing.append([ example.species, example.id,
                                 words, min(len(sentence), max_length - 1) + 2 ])

    try:
        os.makedirs("data/lm")
    except:
        pass

    with open('data/lm/training.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow([
            'class', 'instance', 'sentence', 'length'
        ])

        shuffle(training)
        for species, text, pair_in, pair_len in training:
            writer.writerow([ species, text,
                              '|'.join([ str(idx) for idx in pair_in]),
                              pair_len ])

    with open('data/lm/testing.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow([
            'class', 'instance', 'sentence', 'length'
        ])

        shuffle(testing)
        for species, text, pair_in, pair_len in testing:
            writer.writerow([ species, text,
                              '|'.join([ str(idx) for idx in pair_in]),
                              pair_len ])
