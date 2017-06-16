import csv
import glob
import pprint

pp = pprint.PrettyPrinter()

# Retrieve a list of species.
labels = glob.glob('data/cvpr2016_cub/text_c10/*/')

sentences  = []
vocabulary = []
voc_lookup = dict()
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

for label in labels:
    # Retrieve a list of text files with sentences for that species.
    texts = glob.glob(label + '*.txt')

    for text in texts:
        species = int(text.split('/')[3].split('.')[0])

        with open(text, 'r') as file:
            data = file.readlines()

            for sentence in data:
                words = [
                    word
                    for word in tokenize(sentence.rstrip())
                ]

                words = [ word2idx(word) for word in words ]

                sentences.append([ species, text, words ])

with open('data/produced_idx2word.csv', 'w') as file:
    writer = csv.writer(file)

    writer.writerow([ 'index', 'word' ])

    for idx, word in enumerate(vocabulary):
        writer.writerow([ idx, word ])

with open('data/produced_sentences.csv', 'w') as file:
    writer = csv.writer(file)

    writer.writerow([ 'class', 'instance', 'sentence' ])

    for species, text, words in sentences:
        writer.writerow([ species, text, '|'.join([ str(idx) for idx in words]) ])
