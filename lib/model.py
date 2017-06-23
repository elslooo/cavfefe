import tensorflow as tf
import os

class Model:
    def __init__(self):
        self.saver = tf.train.Saver()

        try:
            os.makedirs("checkpoints")
        except:
            pass

    def save(self, session, epoch):
        self.saver.save(session, 'checkpoints/' + self.__class__.__name__,
                        global_step = epoch)
