import csv
import glob
import pprint
import numpy as np

pp = pprint.PrettyPrinter()

# Retrieve a list of species.
labels = glob.glob('data/cvpr2016_cub/text_c10/*/')

max_length = 10

sentences  = []
vocabulary = []
instances  = []
voc_lookup = dict()
ins_lookup = dict()

def tokenize(sentence):
    sentence = ''.join([ c if c.isalpha() else ' ' for c in sentence ])
    words    = sentence.split(' ')
    words    = [ word for word in words if len(word) > 0 ]

    return words

def word2idx(word):
    if word in voc_lookup:
        return voc_lookup[word]

    voc_lookup[word] = len(vocabulary)
    vocabulary.append(word)

    return voc_lookup[word]

def path2idx(path):
    if path in ins_lookup:
        return ins_lookup[path]

    ins_lookup[path] = len(instances)
    instances.append(path)

    return ins_lookup[path]

sos = word2idx("<SOS>")
eos = word2idx("<EOS>")

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

for label in labels:
    # Retrieve a list of text files with sentences for that species.
    texts = glob.glob(label + '*.txt')

    for text in texts:
        species = int(text.split('/')[3].split('.')[0])
        text_idx = path2idx(text)

        with open(text, 'r') as file:
            data = file.readlines()

            for sentence in data:
                words = [
                    word
                    for word in tokenize(sentence.rstrip())
                ]

                words = [ sos ] + [ word2idx(word) for word in words ] + [ eos ]

                for pair in compute_in_out(words):
                    sentences.append([ species, text_idx, pair[0], pair[1], pair[2] ])

with open('data/produced_idx2word.csv', 'w') as file:
    writer = csv.writer(file)

    writer.writerow([ 'index', 'word' ])

    for idx, word in enumerate(vocabulary):
        writer.writerow([ idx, word ])

with open('data/produced_idx2instance.csv', 'w') as file:
    writer = csv.writer(file)

    writer.writerow([ 'index', 'path' ])

    for idx, path in enumerate(instances):
        writer.writerow([ idx, path ])

with open('data/produced_sentences.csv', 'w') as file:
    writer = csv.writer(file)

    writer.writerow([ 'class', 'instance', 'sentence' ])

    for species, text, pair_in, pair_out, pair_len in sentences:
        writer.writerow([ species, text,
                          '|'.join([ str(idx) for idx in pair_in]), pair_out, pair_len ])
