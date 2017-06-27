import csv
import os

class FeatureCache:
    def __init__(self):
        self.labels   = dict()
        self.features = dict()

    def set(self, path, label, features):
        self.labels[path]   = label
        self.features[path] = features

    def get(self, path):
        return (self.labels[path], self.features[path])

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
            writer.writerow([ 'path', 'label', 'features' ])

            for path in self.labels:
                writer.writerow([
                    path,
                    str(self.labels[path]),
                    '|'.join([
                        str(x) for x in self.features[path]
                    ])
                ])
