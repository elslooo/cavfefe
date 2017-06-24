import tensorflow as tf
from utility import retrieve_names_classes
from utility import img_class_to_name

from dataset_utils import write_label_file, _convert_dataset


flags = tf.app.flags
flags.DEFINE_string('dataset_dir', './data/CUB_200_2011/CUB_200_2011/images', 'str, dataset directory')
flags.DEFINE_string('metadata_dir', './data/CUB_200_2011/CUB_200_2011/', 'str, metadata directory')
flags.DEFINE_integer('num_shards', 10, 'int, number of split in tfrecord')
flags.DEFINE_string('tfrecord_filename', './cub200_2011_tfrc', 'str, output filename')

FLAGS = flags.FLAGS

# print FLAGS.dataset_dir
# print FLAGS.metadata_dir

train_names, train_labels, train_classes = retrieve_names_classes(FLAGS.metadata_dir, train=True)
val_names, val_labels, val_classes = retrieve_names_classes(FLAGS.metadata_dir, train=False)

class_name_dict = img_class_to_name()
name_class_dict = {v: k for k, v in class_name_dict.iteritems()}

_convert_dataset('train', train_names, name_class_dict,
                 dataset_dir=FLAGS.dataset_dir,
                 tfrecord_filename=FLAGS.tfrecord_filename,
                 _NUM_SHARDS=FLAGS.num_shards)

_convert_dataset('validation', val_names, name_class_dict,
                 dataset_dir=FLAGS.dataset_dir,
                 tfrecord_filename=FLAGS.tfrecord_filename,
                 _NUM_SHARDS=FLAGS.num_shards)

write_label_file(class_name_dict, FLAGS.dataset_dir)

