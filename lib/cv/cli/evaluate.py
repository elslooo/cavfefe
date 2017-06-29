from __future__ import print_function
import lib.etc as etc
import sys
import tensorflow as tf
from lib.cv import VisionModel, InstanceReader

def cv_evaluate():
    path = 'pretrained/cv/VisionModel'

    reader     = InstanceReader("data/cv/testing.csv")
    batch_size = 64
    epochs     = reader.length / batch_size

    sess  = tf.Session()
    model = VisionModel(num_classes = 200)

    model.restore(sess, path)

    total_acc = 0.0
    total_cnt = 0.0

    for step, pi in etc.range(epochs):
        # Get a batch of training instances.
        batch_images, batch_labels = reader.read(lines = batch_size)

        # Calculate batch accuracy and loss
        acc, loss = model.evaluate(sess, batch_images, batch_labels)

        total_acc += acc
        total_cnt += 1.0

        print("Iter " + str(1 + step) + " / " + str(epochs) + \
              ", Validation Accuracy= " + \
              "{:.5f}".format(total_acc / total_cnt) + ", Time Remaining= " + \
              etc.format_seconds(pi.time_remaining()), file = sys.stderr)
