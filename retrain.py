import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

from lib.resnet import Resnet
from lib.inception_resnet_v2 import *
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.contrib.framework import assign_from_checkpoint_fn
from lib import inception_preprocessing


# parameter settings
NUM_CLASSES = 200
NUM_EPOCHS = 2
NUM_READERS = 4
NUM_THREADS = 4
BATCH_SIZE = 1
IMG_SIZE = 299

tfrc_dir = './CUB_200_2011/images/'
ckpt_name = './data/inception_resnet_v2_2016_08_30.ckpt'
labels_filename = './CUB_200_2011/images/labels.txt'
log_dir = './log'

file_initials = 'cub200_2011_tfrc_%s_*.tfrecord'
split_name = 'train'

# params for the learning rate , need to be tuned
lr_initial = 0.0003
lr_decay_factor = 0.8
lr_decay_step = 100

# tf session configurations
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)
resnet = Resnet(session, ckpt_name)
tf.logging.set_verbosity(tf.logging.INFO)


def get_split(split_name, tfrc_dir, file_initials):

    data_source_path = os.path.join(tfrc_dir, file_initials % (split_name))
    tfrc_filenames = 'cub200_2011_tfrc_' + split_name
    tfrc_names = [os.path.join(tfrc_dir, f) for f in os.listdir(tfrc_dir) if
                  f.startswith(tfrc_filenames)]

    # get the number of records in a tfrecord file
    num_samples = 0
    for tfrc in tfrc_names:
        for record in tf.python_io.tf_record_iterator(tfrc):
            num_samples += 1

    class_to_name_dict = {}
    with open(labels_filename, 'r') as f:
        for line in f:
            label, name = line.rstrip().split(':')
            class_to_name_dict[int(label)] = name

    # key to feature dict for the decoder
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    # items to different tf class handlers dict for the decoder
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    # item descriptions, required for the Dataset class
    items_to_descriptions = {
        'image': 'rgb images',
        'label': 'associated class'
    }

    # decode the binary tfrecord file
    tfrc_decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    tfrc_reader = tf.TFRecordReader

    # dataset object, containing the information to actually retrieve the images
    dataset = slim.dataset.Dataset(
        data_sources=data_source_path,
        decoder=tfrc_decoder,
        reader=tfrc_reader,
        num_readers=NUM_READERS,
        num_samples=num_samples,
        num_classes=NUM_CLASSES,
        labels_to_name=class_to_name_dict,
        items_to_descriptions=items_to_descriptions)

    return dataset


def batch_loading(dataset, batch_size, image_size, is_training=True):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity=batch_size*NUM_THREADS*4,
        common_queue_min=batch_size*NUM_THREADS*2)

    raw_image, label = data_provider.get(['image', 'label'])
    image = inception_preprocessing.preprocess_image(raw_image, image_size, image_size, is_training)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=NUM_THREADS,
        capacity=NUM_THREADS*batch_size,  # capacity of the queue
        allow_smaller_final_batch=True)

    return images, labels


def retrain():
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    with tf.Graph().as_default() as graph:
        dataset = get_split('train', tfrc_dir, file_initials=file_initials)
        images, labels = batch_loading(dataset, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
        epoch_size = dataset.num_samples / BATCH_SIZE

        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(
                images, num_classes=dataset.num_classes, is_training=True)

        variables_to_restore = slim.get_variables_to_restore(
            exclude=['InceptionResnetV2/Logits',
                     'InceptionResnetV2/AuxLogits']
        )

        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
        tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
        total_loss = tf.losses.get_total_loss()
        global_step = get_or_create_global_step()

        lr = tf.train.exponential_decay(
            learning_rate=lr_initial,
            global_step=global_step,
            decay_steps=lr_decay_step,
            decay_rate=lr_decay_factor,
            staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = slim.learning.create_train_op(total_loss, optimizer)
        init_fn = assign_from_checkpoint_fn(ckpt_name, variables_to_restore)

        max_step = epoch_size * NUM_EPOCHS
        saver = tf.train.Saver(max_to_keep=2)

        final_loss = slim.learning.train(train_op, log_dir, init_fn=init_fn,
                                         number_of_steps=max_step, saver=saver)
        print 'Final loss: %.3f' % final_loss


retrain()
