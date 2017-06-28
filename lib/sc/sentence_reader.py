from ..random_reader import RandomReader

class SentenceReader(RandomReader):
    def __init__(self, path, num_classes, embedding_size = 10):
        RandomReader.__init__(self, path)

        self.num_classes    = num_classes
        self.embedding_size = embedding_size

    def one_hot(self, indices, num_classes):
        return [
            [ 1 if index == i else 0 for i in range(num_classes) ]
            for index in indices
        ]

    def string_to_number(self, strings):
        return [
            int(string) for string in strings
        ]

    def _process(self, results):
        labels, instance, sentences, length = \
        list(map(list, zip(*results)))

        sentences = [ sentence.split("|") for sentence in sentences ]
        sentences = [ self.string_to_number(sentence)
                      for sentence in sentences ]
        sentences = [ self.one_hot(sentence, self.embedding_size)
                      for sentence in sentences ]

        length = self.string_to_number(length)

        labels = self.string_to_number(labels)
        labels = self.one_hot(labels, self.num_classes)

        return (sentences, labels, length)

    def query(self, label):
        results = []

        for line in self.cache:
            line = line.split(',')

            if int(line[0]) == label:
                results.append(line)

        return self._process(results)

    def read(self, lines = 10):
        results = RandomReader.read(self, lines = lines)
        results = [ result.split(',') for result in results ]

        return self._process(results)
