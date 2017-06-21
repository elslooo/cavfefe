class RandomReader:
    def __init__(self, path, skip_header_lines = True):
        self.file = open(path, 'r')

        if skip_header_lines:
            self.file.readline()

    def read(self, lines = 10):
        return [
            self.file.readline().rstrip()
            for line in range(lines)
        ]
