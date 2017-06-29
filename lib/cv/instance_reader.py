from ..random_reader import RandomReader

class InstanceReader(RandomReader):
    def __init__(self, path):
        RandomReader.__init__(self, path)

    def _preprocess(self, id, name, label):
        return (id, 'data/cv/images/' + name, int(label))

    def read(self, lines = 10):
        results = RandomReader.read(self, lines = lines)

        results = [
            self._preprocess(*result.split(','))
            for result in results
        ]

        return list(map(list, zip(*results)))
