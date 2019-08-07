import numpy as np
import matplotlib.pyplot as plt
import os, sys
import random

sys.path.append("../../src")
from util import * ## e.g. from echocv/utils. do we have this in our repo?
                    # may not need it, as they only import 1 function (I think), i.e. split_data()
from scipy.misc import imread

from d00_utils.db_utils import dbReadWriteViews

io_views = dbReadWriteViews()
df = io_views.get_table('instances_w_labels')

def get_onehot_vector(filename_str):
    ''' 
    Returns a one-hot vector corresponding to 
    '''
    filename = filename_str.split('_')[2].split('.')[0]
    #filename = '8Z0BWX0M' # TO DO: delete this line!
    df_row = df[df['filename']==filename]
    label = df_row['view'].tolist()[0]

    vec = [0] * 23
    mapping = {'plax': 1, 'a2c': 8, 'a4c': 14}
    idx = mapping[label]
    vec[idx] = 1
    
    return vec

## TO DO:
# Use this file as a template for getting data in the same format
# Look at d00_utils/dcm_utils, which has preprocessing scripts improved from Zhang
# We already split data into train/test. May not need to process like this

def load_data(config, val_split): ## may be able
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

    # config.data = 'm1_usal', config.label_dim = 2, config.feature_dim = 1, config.image_size=224
    
    # TO DO: soft code this later, make sure it's correct with our actual data location
    directory = '~/data/02_intermediate/test'
    directory_train = '~/data/02_intermediate/train_downsampleby5'
    directory_test = '~/data/02_intermediate/test_downsampleby5'  
    directories = [directory_train, directory_test]

    #data_index = [0, 1]
    img_train = []
    img_test = []
    labels = []
    filenames = []

    # WHAT I WANT: two separate numpy arrays x_train, x_test
    # Getting them from separate directories
    # Note: they import util from echocv/util.py
    # Note: to get y_train, y_test, will need to use study/instance and look up value in table
    #       first figure out how we want to load files. stripping strings? meh

    for directory in directories:
        for filename in os.listdir(directory):
            if filename not in filenames:
                filenames.append(filename)
                if directory == directory_train:
                    img_train.append(np.load(directory + filename).astype("uint8")[:, :, :])
                elif directory == directory_test:
                    img_test.append(np.load(directory + filename).astype("uint8")[:, :, :])
                #label = [0] * config.label_dim
                #label[data_index[i]] = 1
                #labels.append(label)
    
    x_train = np.array(img_train).reshape(
        (len(x_train), config.image_size, config.image_size, config.feature_dim)
    )

    x_test = np.array(img_test).reshape(
        (len(x_test), config.image_size, config.image_size, config.feature_dim)
    )

       
    # what do we actually want to load in as our y? Currently we have strings in a db
    # they used config.label_dim to get dim two, i.e. case vs control
    # I assume we want a four (or 23) dim vector one-hot encoded by view
    y_train = np.array(y_train).reshape((len(y_train), config.label_dim))
    y_test = np.array(y_test).reshape((len(y_test), config.label_dim))
    return x_train, x_test, y_train, y_test


## ALL THIS DOES is make a list of all the filenames in the directory. why do we need this?
## (note: redundant with code above)
def load_filenames():
    directories = [
        "/media/deoraid03/deeplearning/hcm_case_control_pklfiles/a4c/ctl/systdiastpairs/",
        "/media/deoraid03/deeplearning/hcm_case_control_pklfiles/a4c/case/systdiastpairs/",
    ]
    data_index = [0, 1]
    filenames = []
    for i, directory in enumerate(directories):
        for filename in os.listdir(directory):
            if filename not in filenames:
                filenames.append(filename)

    return filenames
