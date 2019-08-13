import numpy as np
import matplotlib.pyplot as plt
import os, sys
import random
#from util import * ## e.g. from echocv/utils. do we have this in our repo?
                    # may not need it, as they only import 1 function (I think), i.e. split_data()
from scipy.misc import imread

sys.path.append('/home/ubuntu/dvv/usal_echo/src/')
#print(sys.path)
from d00_utils.db_utils import dbReadWriteViews


io_views = dbReadWriteViews()
df = io_views.get_table('instances_w_labels')


def get_onehot_vector(filename_str):
    ''' 
    Returns a one-hot vector corresponding to 23-class view classification model
    @params filename_str: name of .jpg file, e.g. a_113150_2WD5N6AS_53.jpg
    '''
    filename = filename_str.split('_')[2].split('.')[0]
    #filename = '8Z0BWX0M' # TO DO: delete this line!
    df_row = df[df['filename']==filename]
    label = df_row['view'].tolist()[0]

    vec = [0] * 23 # note: hard-coded for 23-class model
    mapping = {'plax': 1, 'a2c': 8, 'a4c': 14}
    idx = mapping[label]
    vec[idx] = 1
    
    return vec


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
    # config.data = 'm1_usal', config.label_dim = 23, config.feature_dim = 1, config.image_size=224
    
    # TO DO: soft code this later, make sure it's correct with our actual data location
    directory = '~/data/02_intermediate/test'
    directory_train = '~/data/02_intermediate/train_downsampleby5'
    directory_test = '~/data/02_intermediate/train_split100_downsampleby20'  
    directories = [directory_train, directory_test]

    img_train, img_test, label_train, label_test, filenames = ([] for i in range(5))

    # Getting x_train, x_test from separate directories
    # Note: they import util from echocv/util.py

    for directory in directories:
        for filename in os.listdir(directory):
            if filename not in filenames:
                filenames.append(filename)
                label_vec = get_onehot_vec(filename)
                if directory == directory_train:
                    img_train.append(np.load(directory + filename).astype("uint8")[:, :, :])
                    label_train.append(label_vec)
                elif directory == directory_test:
                    img_test.append(np.load(directory + filename).astype("uint8")[:, :, :])
                    label_test.append(label_vec)
    
    x_train = np.array(img_train).reshape(
        (len(x_train), config.image_size, config.image_size, config.feature_dim)
    )
    x_test = np.array(img_test).reshape(
        (len(x_test), config.image_size, config.image_size, config.feature_dim)
    )
    y_train = np.array(label_train).reshape((len(y_train), config.label_dim))
    y_test = np.array(label_test).reshape((len(y_test), config.label_dim))
    
    return x_train, x_test, y_train, y_test

'''
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
'''
