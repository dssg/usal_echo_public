import numpy as np
import matplotlib.pyplot as plt
import os,sys
import random
sys.path.append('src/')
from util import *
from scipy.misc import imread, imresize

def load_data(config, val_split=0):
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
    directory = 'data/' + config.data + '/seg_data/'
    filenames = []
    images = []
    segmentations = []
    im_size = config.image_size

    # Load and resize images and labels to im_size 
    for folder in os.listdir(directory):
        image = imresize(imread(directory+folder+'/'+folder  + '.jpg', flatten = True),(im_size,im_size))

        images.append(image)
        filenames.append(folder)

        seg = np.load(directory+folder+'/seg.npy')
        temp = np.zeros((im_size,im_size,config.label_dim))
        for i in range(config.label_dim):
            temp[:,:,i] = imresize(seg[:,:,i],(im_size,im_size), interp='nearest')/255.0
        segmentations.append(temp)
                
    images = np.array(images)
    segmentations = np.round(np.array(segmentations)).astype('uint8')

    # Load in cross-validation split
    train_lst = np.load('data/' + config.data + '/splits/train_lst_' + str(val_split) + '.npy')
    val_lst = np.load('data/' + config.data + '/splits/val_lst_' + str(val_split) + '.npy')

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # Separate training and validation images/labels using predefined splits
    for i in range(len(filenames)):
        filename = filenames[i]
        study = '_'.join(filename.split('_')[:4])
        if study in train_lst:
            x_train.append(images[i])
            y_train.append(segmentations[i])
        else:
            x_test.append(images[i])
            y_test.append(segmentations[i])
                    
    x_train = np.array(x_train).reshape((len(x_train),im_size,im_size,1))
    x_test = np.array(x_test).reshape((len(x_test),im_size,im_size,1))
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train, x_test, y_train, y_test

def load_filenames():
    directory = 'data/a4c/seg_data/'
    filenames = []
    for folder in os.listdir(directory):
        filenames.append(folder)

    return filenames
