import os, json


# Get a dictionary mapping the image id to its path, input=images.txt
def get_id2path_dict(image_id2path):
    img_id2path_dict = dict()
    with open(image_id2path, 'r') as f:
        for line in f:
            img_id, img_path = line.rstrip().split()
            img_id2path_dict[int(img_id)] = str(img_path)

    return img_id2path_dict


# Get a list of image ids for the training/testing set, input=train_test_split.txt
def get_image_ids(meta_data, train=True):
    train_ind = 1 if train else 0
    img_ids = list()

    with open(meta_data, 'r') as f:
        for line in f:
            img_id, is_train = line.rstrip().split()
            if int(is_train) == train_ind:
                img_ids.append(int(img_id))
    return img_ids


def cap2json(train=True):
    train_test_split = './data/CUB_200_2011/CUB_200_2011/train_test_split.txt'
    image_id2path = './data/CUB_200_2011/CUB_200_2011/images.txt'
    data_dir = './data/cvpr2016_cub/text_c10'

    image_ids = get_image_ids(train_test_split, train=train)
    image_id2path_dict = get_id2path_dict(image_id2path)

    master_data = dict()
    master_data['images'] = list()
    master_data['type'] = 'captions'
    master_data['annotations'] = list()

    for image_id in image_ids:
        image_name = image_id2path_dict[int(image_id)]
        image_path = os.path.join(data_dir, image_name)
        image_path = os.path.splitext(image_path)[0] + '.txt'

        master_data['images'].append({
            'file_name': image_name,
            'id': image_name
        })

        with open(image_path, 'r') as f:
            for line in f:
                master_data['annotations'].append({
                    'caption': str(line.rstrip()),
                    'id': str(image_id),
                    'image_id': str(image_name)
                })

    outfile_name = 'caption_train.json' if train else 'caption_test.json'
    with open(outfile_name, 'w') as f:
        json.dump(master_data, f)

cap2json(train=False)
