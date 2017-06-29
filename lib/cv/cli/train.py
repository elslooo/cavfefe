from __future__ import print_function
import lib.etc as etc
import sys
import tensorflow as tf
from lib.cv import VisionModel, InstanceReader

def cv_train():
    path = 'data/inception_resnet_v2_2016_08_30.ckpt'
    batch_size = 128

    reader = InstanceReader("data/cv/training.csv")
    epochs = reader.length / 128 * 10

    def get_batch(batch_size):
        return reader.read(lines = batch_size)

    sess  = tf.Session()
    model = VisionModel(num_classes = 200)

    model.restore(sess, path, last_layer = False)

    for step, pi in etc.range(epochs):
        # Get a batch of training instances.
        batch_ids, batch_images, batch_labels = get_batch(batch_size)
        model.train(sess, batch_images, batch_labels)

        # Calculate batch accuracy and loss
        acc, loss = model.evaluate(sess, batch_images, batch_labels)

        print("Iter " + str(1 + step) + " / " + str(epochs) + \
              ", Minibatch Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc) + ", Time Remaining= " + \
              etc.format_seconds(pi.time_remaining()), file = sys.stderr)

        model.save(sess, 1 + step)

    print("Optimization Finished!", file = sys.stderr)
