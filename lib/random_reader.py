import numpy as np

class RandomReader:
    def __init__(self, path, skip_header_lines = True):
        self.file              = open(path, 'r')
        self.skip_header_lines = skip_header_lines

        if self.skip_header_lines:
            self.file.readline()

        self.cache = []

        while True:
            line = self.file.readline()

            if not line:
                break

            self.cache.append(line.rstrip())

        self.length = len(self.cache)
        self.order  = np.random.permutation(np.arange(self.length))
        self.cache  = np.array(self.cache)[self.order]
        self.cursor = 0

    def _readline(self):
        self.cursor %= self.length
        self.cursor += 1

        return self.cache[self.cursor - 1]

    def read(self, lines = 10):
        return [
            self._readline()
            for line in range(lines)
        ]

    def __len__(self):
        return self.length
