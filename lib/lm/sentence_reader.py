from ..random_reader import RandomReader

class SentenceReader(RandomReader):
    def __init__(self, path, embedding_size = 10):
        RandomReader.__init__(self, path)

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

    def read(self, lines = 10):
        results = RandomReader.read(self, lines = lines)

        results = [ result.split(',') for result in results ]

        label, instances, sentences, words, length = \
        list(map(list, zip(*results)))

        sentences = [ sentence.split("|") for sentence in sentences ]
        sentences = [ self.string_to_number(sentence)
                      for sentence in sentences ]
        sentences = [ self.one_hot(sentence, self.embedding_size)
                      for sentence in sentences ]

        instances = [ int(instance) for instance in instances ]

        words = [ int(word) for word in words ]
        words = self.one_hot(words, self.embedding_size)

        length = self.string_to_number(length)

        label    = self.string_to_number(label)

        return (instances, label, sentences, words, length)
