import csv
import numpy as np
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
            file.readline()

            reader = csv.reader(file)

            for row in reader:
                label     = int(row[0])
                embedding = np.array([
                    float(x)
                    for x in row[1].split('|')
                ])

                self.set(label, embedding)

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
