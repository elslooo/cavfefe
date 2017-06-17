import tensorflow as tf
from inception_resnet_v2 import *
import numpy as np

class Resnet:
    def __init__(self, session, ckpt_name):
        scope = inception_resnet_v2_arg_scope()


        self.decode_jpeg_data = tf.placeholder(tf.string)
        decode_jpeg = tf.image.decode_jpeg(self.decode_jpeg_data, channels=3)

        if decode_jpeg.dtype != tf.float32:
            decode_jpeg = tf.image.convert_image_dtype(decode_jpeg, dtype=tf.float32)

        image = tf.expand_dims(decode_jpeg, 0)
        image = tf.image.resize_bilinear(image, [299,299], align_corners=False)

        scaled_input_tensor = tf.subtract(image, 0.5)
        scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)

        with tf.contrib.slim.arg_scope(scope):
            self.logits, self.end_points = \
            inception_resnet_v2(scaled_input_tensor, is_training = False)

        saver = tf.train.Saver()
        saver.restore(session, ckpt_name)

        self.session = session

    """
    This function returns class (0-999), features (1x1536), confidence (0-1).
    """
    def predict(self, image):
        predict_values, features, logit_values = self.session.run([
            self.end_points['Predictions'],
            self.end_points['PreLogitsFlatten'],
            self.logits
        ], feed_dict={ self.decode_jpeg_data: open(image, 'rb').read() })

        return np.argmax(predict_values), features[0], np.max(predict_values)
