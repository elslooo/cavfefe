import numpy as np
import tensorflow as tf
import glob, os, math, sys


flags = tf.app.flags
flags.DEFINE_string('data_dir', './data/cvpr2016_cub/text_c10', 'str, dataset directory')
flags.DEFINE_integer('num_shards', 2, 'int, number of split in tfrecord')
flags.DEFINE_string('tfrecord_sent_filename', 'cvpr2016_sent', 'str, output filename')
FLAGS = flags.FLAGS

# sentences, labels, lengths


def retrieve_names_labels(data_dir):
    all_classes = glob.glob(os.path.join(data_dir, '*/'))
    sent_file_paths = list()
    sent_labels = list()
    name2class = dict()
    class2name = dict()

    # traverse through the directory
    for one_class in all_classes:
        # get a list of files in a subdirectory
        class_file_names = glob.glob(one_class + '*.txt')

        # # build the dictionary for name to class and class to name
        one_class = one_class.split('/')[-2]
        class_id, class_name = one_class.split('.')
        name2class[str(class_name)] = int(class_id)
        class2name[str(class_id)] = str(class_name)

        for class_file_name in class_file_names:
            sent_file_paths.append(class_file_name)
            sent_labels.append(int(class_id))

    return sent_file_paths, sent_labels, class2name, name2class


sentences, labels, c2n, n2c = retrieve_names_labels(FLAGS.data_dir)
# print len(sentences), len(labels)


def   int64_feature(values):
  """Returns a TF-Feature of int64s.
  Args:
    values: A scalar or list of values.
  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.
  Args:
    values: A string.
  Returns:
    a TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def txt_to_tfexample(image_data, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'txt/encoded': bytes_feature(image_data),
      'txt/class/label': int64_feature(class_id)
  }))


def convert_dataset_txt(filenames, name_to_class, data_dir, tfrecord_filename, num_shards):
    num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))

    with tf.Graph().as_default():
        with tf.Session('') as sess:

            for shard_id in range(num_shards):
                output_filename = '%s_%05d-of-%05d.tfrecord' % (tfrecord_filename, shard_id, num_shards)
                output_file_path = os.path.join(data_dir, output_filename)

                with tf.python_io.TFRecordWriter(output_file_path) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        input_data = tf.gfile.FastGFile(filenames[i], 'r').read()

                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_name = str(class_name.split('.')[1])
                        class_id = name_to_class[class_name]

                        example = txt_to_tfexample(input_data, class_id)

                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


convert_dataset_txt(sentences, name_to_class=n2c, data_dir=FLAGS.data_dir,
                    tfrecord_filename=FLAGS.tfrecord_sent_filename, num_shards=FLAGS.num_shards)
