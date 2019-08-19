import numpy as np
import os
from scipy.misc import imread

from d00_utils.db_utils import dbReadWriteViews


io_views = dbReadWriteViews()
df = io_views.get_table("instances_w_labels")


def get_onehot_vec(filename_str):
    """ 
    Returns a one-hot vector corresponding to 23-class view classification model
    @params filename_str: name of .jpg file, e.g. a_113150_2WD5N6AS_53.jpg
    """
    filename = filename_str.split("_")[2].split(".")[0]
    df_row = df[df["filename"] == filename]
    label = df_row["view"].tolist()[0]

    vec = [0] * 23  # note: hard-coded for 23-class model
    mapping = {"plax": 1, "a2c": 8, "a4c": 14}
    idx = mapping[label]
    vec[idx] = 1

    return vec


def load_data(config, val_split):
    """
    Preprocesses data and loads them into numpy arrays
    Returns
        x_train, x_test (training and validation images)
        y_train, y_test (training and validation labels)
    Images have dimensions N x im_size x im_size x feature_dim
    Labels have dimensions N x im_size x im_size x label_dim 
        N is number of images
        im_size is dimension of image
        feature_dim is number of channels in image
        label_dim is number of unique labels

    @params config: dictionary of hyperparemeters
    """
    # TODO: soft code paths
    directory_train = "/home/ubuntu/data/02_intermediate/train_split100_downsampleby20/"
    directory_test = "/home/ubuntu/data/02_intermediate/test_downsampleby5/"
    directories = [directory_train, directory_test]

    img_train, img_test, label_train, label_test, filenames = ([] for i in range(5))

    for directory in directories:
        for filename in os.listdir(directory):
            if filename not in filenames:
                filenames.append(filename)
                label_vec = get_onehot_vec(filename)
                np_img = imread(directory + filename)
                if directory == directory_train:
                    img_train.append(np_img)
                    label_train.append(label_vec)
                elif directory == directory_test:
                    img_test.append(np_img)
                    label_test.append(label_vec)

    x_train = np.array(img_train).reshape(
        (len(img_train), config.image_size, config.image_size, config.feature_dim)
    )
    x_test = np.array(img_test).reshape(
        (len(img_test), config.image_size, config.image_size, config.feature_dim)
    )
    y_train = np.array(label_train).reshape((len(label_train), config.label_dim))
    y_test = np.array(label_test).reshape((len(label_test), config.label_dim))

    return x_train, x_test, y_train, y_test
