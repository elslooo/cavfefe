import tensorflow as tf
from lib.resnet import Resnet

ckpt_name = 'inception_resnet_v2_2016_08_30.ckpt'

session = tf.Session()
resnet = Resnet(session, ckpt_name)

sample_images = ['dog_1.jpg', 'panda_1.jpg', 'panda_2.jpg', 'dog_2.jpg']

print("Printing tuples of (class, features, confidence)")
print("[!] TIP: look at the webpage below to match classes to labels.")
print("... https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a")

for image in sample_images:
    print(resnet.predict(image))
