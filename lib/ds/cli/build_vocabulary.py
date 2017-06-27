from lib.ds import Dataset, Vocabulary

def ds_build_vocabulary():
    vocabulary = Vocabulary()
    dataset    = Dataset()

    vocabulary.add("<SOS>")
    vocabulary.add("<EOS>")

    for example in dataset.examples():
        for sentence in example.sentences():
            for word in sentence:
                vocabulary.add(word)

    vocabulary.save('data/ds/vocabulary.csv')
