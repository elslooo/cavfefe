import tensorflow as tf
import numpy as np
import os, re, sys
from scipy import ndimage, misc
import math

# sys.path.append('../tensorflow-resnet/')
# from synset import *


def retrieve_names_classes(meta_dir, train=True):
    """
    :param meta_dir: directory to the meta-data level
    :param train: determine if retrieving training or testing data
    :return: img_names: list of image paths from the project root
             img_labels: list of image labels
             class_names: list of class names corresponding to the list 'img_names'
    """
    train_test_split = 'train_test_split.txt'
    image_list = 'images.txt'
    image_class_labels = 'image_class_labels.txt'

    # specify training image or test image
    train_indicator = 0
    if train:
        train_indicator = 1

    # get a list of image ids from the definition file 'train_test_split.txt'
    img_ids = list()
    with open(meta_dir+train_test_split, 'r') as f:
        for line in f:
            data = line.rstrip().split(' ')
            if int(data[1]) == train_indicator:
                img_ids.append(int(data[0]))

    # get the file names of the images from the definition file 'images.txt'
    img_names = list()
    with open(meta_dir+image_list, 'r') as f:
        for line in f:
            data = line.rstrip().split(' ')
            if int(data[0]) in img_ids:
                # find the image name and form the path
                img_names.append(meta_dir + 'images/' + str(data[1]))

    # get the labels for the images
    img_labels = list()
    with open(meta_dir + image_class_labels, 'r') as f:
        for line in f:
            data = line.rstrip().split(' ')
            if int(data[0]) in img_ids:
                img_labels.append(int(data[1]))

    class_names = img_class_to_name(img_labels)

    return img_names, img_labels, class_names


def img_class_to_name(img_class=None):
    """
    :param img_class: class number, can be int, or a list of int, if None, return the dictionary
    :return: a list of class names
    """
    data_dir = './CUB_200_2011/'
    class_names_file = 'classes.txt'

    # create a dictionary for the class name and id conversion
    class_name_dict = dict()
    with open(data_dir+class_names_file, 'r') as f:
        for line in f:
            im_id, name = line.rstrip().split(' ')
            class_name_dict[int(im_id)] = str(name)

    if img_class is None:
        return class_name_dict

    img_class_list = list()
    assert isinstance(img_class, (int, list)), 'img_class_to_name: invalid input, int required'
    if isinstance(img_class, int):
        img_class_list.append(img_class)
    else:
        img_class_list = [i for i in img_class if isinstance(i, int)]
        assert (len(img_class_list) > 0), 'img_class_to_name: invalid input'

    # convert the list of input ids to the list of class names
    class_names = list()
    for cid in img_class_list:
        class_names.append(class_name_dict[cid])

    return class_names


# def print_prob(prob):
#     pred = np.argsort(prob)[::-1]
#
#     # Get top1 label
#     top1 = synset[pred[0]]
#     print "Top1: ", top1
#     # Get top5 label
#     top5 = [synset[pred[i]] for i in range(5)]
#     print "Top5: ", top5
#     return top1


def print_operations(sess):
    ops = sess.graph.get_operations()
    for i in ops:
        print i.name, i.values()
