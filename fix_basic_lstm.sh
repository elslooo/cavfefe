#!/bin/bash

path="pretrained/sc/SentenceClassifier"

python tensorflow_rename_variables.py \
       --checkpoint_dir="$path" \
       --replace_from=weights \
       --replace_to=kernel

python tensorflow_rename_variables.py \
       --checkpoint_dir="$path" \
       --replace_from=biases \
       --replace_to=bias
