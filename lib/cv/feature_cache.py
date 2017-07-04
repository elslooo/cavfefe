import csv
import numpy as np
import os

class FeatureCache:
    def __init__(self):
        self.labels   = dict()
        self.features = dict()

    def set(self, id, label, features):
        self.labels[id]   = label
        self.features[id] = features

    def get(self, id):
        return (self.labels[id], self.features[id])

    def restore(self, path):
        with open(path, 'r') as file:
            file.readline()

            reader = csv.reader(file)

            for row in reader:
                id      = int(row[0])
                label   = np.array([
                    int(x)
                    for x in row[1].split('|')
                ])
                feature = np.array([
                    float(x)
                    for x in row[2].split('|')
                ])

                self.set(id, label, feature)

    def save(self, path):
        try:
            os.makedirs(os.path.dirname(path))
        except:
            pass

        with open(path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow([ 'id', 'label', 'features' ])

            for id in self.labels:
                writer.writerow([
                    id,
                    '|'.join([
                        str(x) for x in self.labels[id]
                    ]),
                    '|'.join([
                        str(x) for x in self.features[id]
                    ])
                ])
