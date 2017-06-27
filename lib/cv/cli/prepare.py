from scipy.misc import imread, imresize, imsave
import progressbar
import os
import csv

def cv_prepare():
    images = open('data/CUB_200_2011/CUB_200_2011/images.txt', 'r')
    splits = open('data/CUB_200_2011/CUB_200_2011/train_test_split.txt', 'r')
    labels = open('data/CUB_200_2011/CUB_200_2011/image_class_labels.txt', 'r')

    try:
        os.makedirs('data/cv')
    except:
        pass

    all      = csv.writer(open('data/cv/all.csv', 'w'))
    training = csv.writer(open('data/cv/training.csv', 'w'))
    testing  = csv.writer(open('data/cv/testing.csv', 'w'))

    training.writerow([ "path", "label" ])
    testing.writerow([ "path", "label" ])

    with progressbar.ProgressBar(max_value = 11788) as bar:
        i = 0

        while True:
            line = images.readline()

            if line is None or line.rstrip() == '':
                break

            line  = line.rstrip()
            name  = line.split(' ')[1]

            label = int(labels.readline().rstrip().split(' ')[1])
            skip  = int(splits.readline().rstrip().split(' ')[1])

            image = imread('data/CUB_200_2011/CUB_200_2011/images/' + name,
                           mode = 'RGB')
            image = imresize(image, (299, 299))
            image = image / 256.0
            image = image - 0.5
            image = image * 2.0

            try:
                os.makedirs('data/cv/images/' + os.path.dirname(name))
            except:
                pass

            imsave('data/cv/images/' + name, image)

            if skip:
                testing.writerow([ name, str(label) ])
            else:
                training.writerow([ name, str(label) ])

            all.writerow([ name, str(label) ])

            i += 1

            bar.update(i)
