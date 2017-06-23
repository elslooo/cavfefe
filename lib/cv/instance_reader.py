from ..random_reader import RandomReader

class InstanceReader(RandomReader):
    def __init__(self, path):
        RandomReader.__init__(self, path)

    def _preprocess(self, path):
        relevant = path.split('/')[-2 :]
        species  = int(relevant[0].split('.')[0]) - 1
        path     = 'data/CUB_200_2011/CUB_200_2011/images/' + \
                   relevant[0] + '/' + relevant[1].split('.')[0] + '.jpg'

        return (path, species)

    def read(self, lines = 10):
        results = RandomReader.read(self, lines = lines)

        results = [
            self._preprocess(result.split(',')[1])
            for result in results
        ]

        return list(map(list, zip(*results)))
