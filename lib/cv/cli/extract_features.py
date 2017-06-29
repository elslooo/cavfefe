from __future__ import print_function
import lib.etc as etc
import sys
import tensorflow as tf
from lib.cv import VisionModel, InstanceReader, FeatureCache

def cv_extract_features():
    path = 'pretrained/cv/VisionModel'

    reader     = InstanceReader("data/cv/all.csv")
    batch_size = 1
    epochs     = reader.length / batch_size

    sess  = tf.Session()
    model = VisionModel(num_classes = 200)

    model.restore(sess, path)

    cache = FeatureCache()

    for step, pi in etc.range(epochs):
        # Get a batch of testing instances.
        batch_ids, batch_images, _ = reader.read(lines = batch_size)

        # Retrieve the first path from the batch.
        path = batch_images[0]

        # Calculate batch accuracy and loss
        label, features, confidence = model.predict(sess, path)

        cache.set(batch_ids[0], label, features)

        print("Iter " + str(1 + step) + " / " + str(epochs) +
              ", Time Remaining= " + \
              etc.format_seconds(pi.time_remaining()), file = sys.stderr)

    cache.save("data/cv/features.csv")
