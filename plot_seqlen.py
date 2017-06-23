import numpy as np
import matplotlib.pyplot as plt

with open('data/procued_sentence_lengths.np', 'r') as file:
    data = np.load(file)

plt.hist(data, bins = 50)
plt.show()
