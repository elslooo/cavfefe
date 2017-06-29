class Example:
    def __init__(self, id, path, species, is_training, datadir):
        self.id          = id
        self.path        = path
        self.species     = species
        self.is_training = is_training
        self.datadir     = datadir

    def __str__(self):
        return "<Example id={} path={} is_training={}>" \
               .format(self.id, self.path, self.is_training)

    def __repr__(self):
        return self.__str__(self)

    def _tokenize(self, sentence):
        sentence = ''.join([ c if c.isalpha() else ' ' for c in sentence ])
        words    = sentence.split(' ')
        words    = [ word for word in words if len(word) > 0 ]

        return words

    def image_path(self):
        return self.datadir + '/CUB_200_2011/CUB_200_2011/images/' + \
               self.path + '.jpg'

    def sentences(self):
        path = self.datadir + "/cvpr2016_cub/text_c10/" + self.path + ".txt"

        results = []

        with open(path, 'r') as file:
            while True:
                line = file.readline()

                if not line or line.rstrip() == '':
                    break

                results.append(self._tokenize(line.rstrip()))

        return results
