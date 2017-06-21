# Language Model

This directory contains all code that runs the language model.

## TODOs

- The Random Reader is not yet random.
- This is probably solved once we completed the previous todo but still: make
  sure the reader loops over all lines (e.g. when moving past the last line),
  this currently is not the case.

## Random Reader

The random reader opens a specified file and reads the lines in random sequence. Note that if you specify the number of lines, these lines are not necessarily contiguous.

## Sentence Reader

This reader opens the prepared sentences file. It subsequently splits the
columns (delimited by ,) and the sentences column itself (delimited by |).
Furthermore, it converts words to one-hot embeddings.
