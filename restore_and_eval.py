import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

from lib.resnet import Resnet
from lib.inception_resnet_v2 import *
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.contrib.framework import assign_from_checkpoint_fn
from lib import inception_preprocessing
from retrain import get_split, batch_loading
from tensorflow.python.tools.inspect_checkpoint import _get_checkpoint_filename

# load data pipeline

# first get the splits
split_name = 'validation'

tfrc_dir = './CUB_200_2011/images/'
ckpt_name = './data/inception_resnet_v2_2016_08_30.ckpt'
labels_filename = './CUB_200_2011/images/labels.txt'
log_dir = './log'

file_initials = 'cub200_2011_tfrc_%s_*.tfrecord'

layer_to_grab = "InceptionResnetV2/Logits/AvgPool_1a_8x8/AvgPool:0"


def validate_resnet():
    # get the trained network
    scope = inception_resnet_v2_arg_scope()

    with tf.Graph().as_default() as graph:
        # get checkpoint name
        #checkpoint_file = _get_checkpoint_filename(log_dir)
        #if checkpoint_file is None:
        #    raise ValueError("Couldn't find 'checkpoint' file or checkpoints in "
        #            " given directory %s" % log_dir)
        # this could also be the latest:
        checkpoint_file = tf.train.latest_checkpoint(log_dir)

        # get the data
        dataset = get_split(split_name, tfrc_dir, file_initials=file_initials)
        images, labels = batch_loading(dataset, batch_size=BATCH_SIZE)
        epoch_size = dataset.num_samples / BATCH_SIZE

        # create graph
        with slim.arg_scope(scope):
            logits, end_points = inception_resnet_v2(images, num_classes=dataset.num_samples, is_training=False)

        # get the variables to restore
        var_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(var_to_restore)

        with tf.Session() as sess:
            # restore variables
            saver.restore(sess, checkpoint_file)

        # grab the layer before the fully convolutional one
        # this are the weights/features
        feature_tensor = graph.get_tensor_by_name(layer_to_grab)

        # global step, manually increment
        #global_step = get_or_create_global_step()
        #global_step_op = tf.assign(global_step, global_step+1)

        # metrics
        #predictions = tf.argmax(end_points['Predictions'], 1)
        #accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        #metrics_op = tf.group(accuracy_update)


        # Choose the metrics to compute:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'accuracy': slim.metrics.accuracy(predictions, labels),
            'precision': slim.metrics.precision(predictions, labels),
            'recall': slim.metrics.recall(mean_relative_errors, 0.3),
            })

        # Create the summary ops such that they also print out to std output:
        summary_ops = []
        for metric_name, metric_value in names_to_values.iteritems():
            op = tf.summary.scalar(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)

        num_examples = dataset.num_samples
        batch_size = BATCH_SIZE
        num_batches = epoch_size

        # Setup the global step.
        slim.get_or_create_global_step()

        output_dir = log_dir # Where the summaries are stored.
        eval_interval_secs = 1 # How often to run the evaluation.

        slim.evaluation.evaluation_loop(
                'local',
                checkpoint_dir,
                log_dir,
                num_evals=epoch_size,
                eval_op=names_to_updates.values(),
                summary_op=tf.summary.merge(summary_ops),
                eval_interval_secs=eval_interval_secs)
