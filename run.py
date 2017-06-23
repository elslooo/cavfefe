from __future__ import print_function
import lib.etc as etc
import sys
import tensorflow as tf
from lib.cv import VisionModel, InstanceReader

path = 'data/inception_resnet_v2_2016_08_30.ckpt'
epochs = 2000
batch_size = 128

reader = InstanceReader("data/produced_cv_instances.csv")

def get_batch(batch_size):
    return reader.read(lines = batch_size)

sess  = tf.Session()
model = VisionModel(num_classes = 2)

model.restore(sess, path)

for step, pi in etc.range(epochs):
    # Get a batch of training instances.
    batch_images, batch_labels = get_batch(batch_size)
    model.train(sess, batch_images, batch_labels)

    # Calculate batch accuracy and loss
    acc, loss = model.evaluate(sess, batch_images, batch_labels)

    print("Iter " + str(1 + step) + " / " + str(epochs) + \
          ", Minibatch Loss= " + \
          "{:.6f}".format(loss) + ", Training Accuracy= " + \
          "{:.5f}".format(acc) + ", Time Remaining= " + \
          etc.format_seconds(pi.time_remaining()), file = sys.stderr)

    if (1 + step) % 10 == 0:
        model.save(sess, 1 + step)

print("Optimization Finished!", file = sys.stderr)
