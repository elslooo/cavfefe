from sentence_reader import SentenceReader
from random_reader import RandomReader

class SentenceClassReader(SentenceReader):
    def __init__(self, path, nr_classes, embedding_size=10):
        SentenceReader.__init__(self, path, embedding_size)
        self.nr_classes = nr_classes

    def read(self, lines = 10):
        results = RandomReader.read(self, lines = lines)

        results = [ result.split(',') for result in results ]

        labels, instance, sentences, length = \
        list(map(list, zip(*results)))

        sentences = [ sentence.split("|") for sentence in sentences ]
        sentences = [ self.string_to_number(sentence)
                      for sentence in sentences ]
        sentences = [ self.one_hot(sentence, self.embedding_size)
                      for sentence in sentences ]

        length = self.string_to_number(length)

        labels    = self.string_to_number(labels)
        labels = self.one_hot(labels, self.nr_classes)

        return (sentences, labels, length)