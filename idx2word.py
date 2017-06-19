import csv

words = []

with open('data/produced_idx2word.csv', 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        if row[1] == 'word':
            continue

        words.append(row[1])

def idx2sentence(indices):
    return ' '.join([ words[idx] if idx < len(words) else '<NULL>' for idx in indices  ])

embedding_size = len(words)
