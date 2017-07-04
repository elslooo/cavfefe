import csv
import os

class Vocabulary:
    def __init__(self):
        self.words          = []
        self.inverted_index = dict()

    def index(self, word):
        return self.inverted_index[word]

    def word(self, index):
        return self.words[index]

    def sentence(self, indices, limit = False):
        if limit == True and 1 in indices:
            indices = indices[: indices.index(1)]
            
        return " ".join([ self.word(index) for index in indices ])

    def add(self, word):
        if word in self.inverted_index:
            return self.inverted_index[word]
        else:
            self.words.append(word)
            self.inverted_index[word] = len(self.words) - 1
            return len(self.words) - 1

    def save(self, path):
        try:
            os.makedirs(os.path.dirname(path))
        except:
            pass

        with open(path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow([ "index", "word" ])

            for index, word in enumerate(self.words):
                writer.writerow([ str(index), word ])

    def restore(self, path):
        with open(path, 'r') as file:
            file.readline()

            reader = csv.reader(file)

            for row in reader:
                assert(int(row[0]) == self.add(row[1]))

        return self

    def __len__(self):
        return len(self.words)
