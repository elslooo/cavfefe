from example import Example
from species import Species

class Dataset:
    def __init__(self, path = 'data'):
        self.path = path

    def _readmap(self, name):
        path = self.path + "/CUB_200_2011/CUB_200_2011/" + name + ".txt"

        map = dict()

        with open(path, 'r') as file:
            while True:
                line = file.readline()

                if line is None or line.rstrip() == '':
                    break

                parts = line.rstrip().split(' ')

                map[parts[0]] = parts[1]

        return map

    def species(self):
        map     = self._readmap('classes')
        results = []

        for id in map:
            results.append(Species(id = int(id), name = map[id]))

        return sorted(results, key = lambda x: x.id)

    def examples(self):
        class_labels     = self._readmap("image_class_labels")
        images           = self._readmap("images")
        train_test_split = self._readmap("train_test_split")

        results = []

        for id in images:
            results.append(Example(id          = int(id),
                                   path        = images[id][: -4],
                                   species     = int(class_labels[id]),
                                   is_training = int(train_test_split[id]) == 1,
                                   datadir     = self.path))

        return results
