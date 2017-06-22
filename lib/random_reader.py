class RandomReader:
    def __init__(self, path, skip_header_lines = True):
        self.file              = open(path, 'r')
        self.skip_header_lines = skip_header_lines

        if self.skip_header_lines:
            self.file.readline()

    def _readline(self):
        line = self.file.readline()

        if not line:
            self.file.seek(0, 0)

            if self.skip_header_lines:
                self.file.readline()

            line = self.file.readline()

        return line.rstrip()

    def read(self, lines = 10):
        return [
            self._readline()
            for line in range(lines)
        ]
