# Cavfefe

## TODOs

- If you download a checkpoint with a BasicLSTMCell (such as the language model
  and the sentence classifier) from the DAS, it will not be accepted by our
  local Tensorflow installations because the variable names were changed after
  Tensorflow 1.0. Therefore, run `fix_basic_lstm.sh` with the path to the
  checkpoint (e.g. `pretrained/sc/SentenceClassifier`) to replace the old names
  with the new ones.

## Preparing the Dataset

### Building a Vocabulary

For reasons of efficiency, we start by mapping words (literals) to indices
(integers), and save the resulting vocabulary to disk so we can use it from
several implementations (the sentence classifier and the language model).

```
python cavfefe.py --ds-build-vocabulary
```

## Computer Vision Model

### Preparing the Dataset

The vision model accepts only 299x299 images with 3 channels. In addition, we
also need to split the training and the testset.

```
python cavfefe.py --cv-prepare
```

### Training

Once we have prepared the dataset, we can train the model by running the command
below. Note that this will only train the logits layer (the last layer), which
means that even the features that will be extracted later on, remain unchanged.

```
python cavfefe.py --cv-train
```

### Evaluation

After training, it is important to verify that the model is behaving correctly
before proceeding to integrating the entire project. The accuracy of the CNN can
be computed in this intermediate stage by running the command below.

```
python cavfefe.py --cv-evaluate
```

### Feature Extraction

We want to extract the features from the vision model so we can concatenate
those in the language model. The command below extracts both features of images in the training set as well as the test set and saves both to disk.

```
python cavfefe.py --cv-extract-features
```

## Sentence Classifier

### Preparing the Dataset

Again, we first need to preprocess the dataset. In this case, we need to pad the
sentences to a fixed maximum length (default: 30). Running the command below requires that the vocabulary has already been built (step #1).

```
python cavfefe.py --sc-prepare
```

### Training

The sentence classifier can be trained by running the following command. In our
experience, the classifier takes a bit longer than usual before beginning to
converge. There is also a significant variance between the accuracy after
convergence: we suspect that it gets more easily stuck in local optima than
other models.

```
python cavfefe.py --sc-train
```

### Evaluation

Again, we need to evaluate the accuracy of the sentence classifier before
proceeding with the rest of the project. The authors of the paper mention they
achieved 22% classification error on their validation set. We managed to get
21.754% which is close enough. Note that our model uses 512 hidden units
(instead of 1000).

```
python cavfefe.py --sc-evaluate
```

### Embedding Extraction

In addition to the features we retrieve from the computer vision model, we also
want to concatenate a class embedding to the input of the language model. The
authors of the paper take the average over all activations of sentences in the
training set per class. For each class, we have 30 examples with 10 sentences
each, so for each class we take the average over 300 512-dimensional vectors and
end up with 200 512-dimensional vectors, one for each class. This command writes
the embeddings to disk.

```
python cavfefe.py --sc-extract-embeddings
```
