import tensorflow as tf
import os

class Model:
    def __init__(self):
        self.saver = tf.train.Saver()

        try:
            os.makedirs("checkpoints")
        except:
            pass

    def restore(self, session, path, scope = None):
        if scope is None:
            scope = self.__class__.__name__

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      scope = scope)

        print([ var.name for var in variables ])

        saver = tf.train.Saver()
        saver.restore(session, path)

    def save(self, session, epoch):
        self.saver.save(session, 'checkpoints/' + self.__class__.__name__,
                        global_step = epoch)
