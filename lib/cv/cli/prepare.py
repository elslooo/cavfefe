from scipy.misc import imread, imresize, imsave
import progressbar
import os
import csv
from lib.ds import Dataset

def cv_prepare():
    dataset = Dataset()

    try:
        os.makedirs('data/cv')
    except:
        pass

    all      = csv.writer(open('data/cv/all.csv', 'w'))
    training = csv.writer(open('data/cv/training.csv', 'w'))
    testing  = csv.writer(open('data/cv/testing.csv', 'w'))

    all.writerow([ "id", "path", "label" ])
    training.writerow([ "id", "path", "label" ])
    testing.writerow([ "id", "path", "label" ])

    with progressbar.ProgressBar(max_value = 11788) as bar:
        i = 0

        for example in dataset.examples():
            image = imread(example.image_path(), mode = 'RGB')
            image = imresize(image, (299, 299))
            image = image / 256.0
            image = image - 0.5
            image = image * 2.0

            try:
                os.makedirs('data/cv/images/' + os.path.dirname(example.path))
            except:
                pass

            imsave('data/cv/images/' + example.path + '.jpg', image)

            if example.is_training:
                training.writerow([
                    example.id,
                    example.path + '.jpg',
                    str(example.species)
                ])
            else:
                testing.writerow([
                    example.id,
                    example.path + '.jpg',
                    str(example.species)
                ])

            all.writerow([
                example.id,
                example.path + '.jpg',
                str(example.species)
            ])

            i += 1

            bar.update(i)
