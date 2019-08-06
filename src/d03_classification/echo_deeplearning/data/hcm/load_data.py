import numpy as np
import matplotlib.pyplot as plt
import os, sys
import random
sys.path.append('src/')
from util import *
from scipy.misc import imread



def load_data(config, val_split):
    '''
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
    @val_split: split data into training and test based on validation split
    '''
    #######################################################################
    # REPLACE WITH YOUR OWN PREPROCESSING METHODS THAT RETURN SAME FORMAT #
    #######################################################################

    directories = ['/media/deoraid03/deeplearning/hcm_case_control_pklfiles/a4c/ctl/systdiastpairs/',
                   '/media/deoraid03/deeplearning/hcm_case_control_pklfiles/a4c/case/systdiastpairs/']
    data_index = [0,1]
    images = []
    labels = []
    filenames = []

    for i,directory in enumerate(directories):
        for filename in os.listdir(directory):
            if filename not in filenames:
                images.append(np.load(directory + filename).astype('uint8')[:,:,:])
                filenames.append(filename)
                label = [0] * config.label_dim
                label[data_index[i]] = 1
                labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    train_lst = np.load('data/' + config.data + '/splits/train_lst_' + str(val_split) + '.npy')
    val_lst = np.load('data/' + config.data + '/splits/val_lst_' + str(val_split) + '.npy')

    x_train, y_train, x_test, y_test = split_data(filenames, images, labels, train_lst, val_lst)

    x_train = np.array(x_train).reshape((len(x_train),config.image_size,config.image_size,config.feature_dim))
    y_train = np.array(y_train).reshape((len(y_train),config.label_dim))
    x_test = np.array(x_test).reshape((len(x_test),config.image_size,config.image_size,config.feature_dim))
    y_test = np.array(y_test).reshape((len(y_test),config.label_dim))
    return x_train, x_test, y_train, y_test

def load_filenames():
    directories = ['/media/deoraid03/deeplearning/hcm_case_control_pklfiles/a4c/ctl/systdiastpairs/',
                   '/media/deoraid03/deeplearning/hcm_case_control_pklfiles/a4c/case/systdiastpairs/']
    data_index = [0,1]
    filenames = []
    for i,directory in enumerate(directories):
        for filename in os.listdir(directory):
            if filename not in filenames:
                filenames.append(filename)

    return filenames