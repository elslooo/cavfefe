import csv
import os

class EmbeddingCache:
    def __init__(self):
        self.embeddings = dict()

    def set(self, label, embedding):
        self.embeddings[label] = embedding

    def get(self, label):
        return self.embeddings[label]

    def restore(self, path):
        with open(path, 'r') as file:
            reader = csv.reader(file)
            reader.read()

    def save(self, path):
        try:
            os.makedirs(os.path.dirname(path))
        except:
            pass

        with open(path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow([ 'label', 'embedding' ])

            for label in self.embeddings:
                writer.writerow([
                    str(label),
                    '|'.join([
                        str(x) for x in self.embeddings[label]
                    ])
                ])
