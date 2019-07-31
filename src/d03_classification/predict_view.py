# coding: utf-8

import random
import sys

import os
import subprocess
import time
import cv2
import pydicom
from optparse import OptionParser
from shutil import rmtree

import tensorflow as tf
import numpy as np
from scipy.misc import imread

from d00_utils.dcm_utils import extract_imgs_from_dicom

sys.path.append('./funcs/')
sys.path.append('./nets/')

# # Hyperparams
parser=OptionParser()
parser.add_option("-d", "--dicomdir", dest="dicomdir", default = "dicomsample", help = "dicomdir")
parser.add_option("-g", "--gpu", dest="gpu", default = "0", help = "cuda device to use")
parser.add_option("-M", "--modeldir", dest="modeldir", default = "models", help = "modeldir")
parser.add_option("-m", "--model", dest="model")
params, args = parser.parse_args()
dicomdir = params.dicomdir
modeldir = params.modeldir
model = params.model

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu


def classify(directory, feature_dim, label_dim, model_name):
    """
    Classifies echo images in given directory

    @param directory: folder with jpg echo images for classification
    """
    imagedict = {}
    predictions = {}
    for filename in os.listdir(directory):
        if "jpg" in filename:
            image = imread(directory + filename, flatten = True).astype('uint8')
            imagedict[filename] = [image.reshape((224,224,1))]

    tf.reset_default_graph()
    sess = tf.Session()
    model = vgg.Network(0.0, 0.0, feature_dim, label_dim, False)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, model_name)

    for filename in imagedict:
        predictions[filename] =np.around(model.probabilities(sess, \
                  imagedict[filename]), decimals = 3)
    
    return predictions


def main():
    
    # To use dicomdir option set in global scope.
    global dicomdir, modeldir #TODO should not be setting global parameters
    
    # Create directories for saving images
    results_dir = '/home/ubuntu/data/03_classification/results' #TODO this shouldn't be hardcoded    
    os.makedirs(results_dir, exist_ok=True)
    temp_image_dir = os.path.join(dicomdir, 'image/')
    os.makedirs(temp_image_dir, exist_ok=True)
    
    model = "view_23_e5_class_11-Mar-2018"
    model_name = os.path.join(modeldir, model)

    infile = open("d03_classification/viewclasses_" + model + ".txt")
    infile = infile.readlines()
    views = [i.rstrip() for i in infile]

    feature_dim = 1
    label_dim = len(views)
   
    # In case dicomdir is path with more than one part.
    dicomdir_basename = os.path.basename(dicomdir)
    out = open(results_dir + model + "_" + dicomdir_basename + "_probabilities.txt", 'w')
    out.write("study\timage")

    for j in views:
        out.write("\t" + j)
    out.write('\n')
    x = time.time()

    extract_jpgs_from_dcmdir(dicomdir, temp_image_dir)
    predictions = classify(temp_image_dir, feature_dim, label_dim, model_name)
    predictprobdict = {}

    for image in list(predictions.keys()):
        prefix = image.split(".dcm")[0] + ".dcm"
        if prefix not in predictprobdict:
            predictprobdict[prefix] = []
        predictprobdict[prefix].append(predictions[image][0])

    for prefix in list(predictprobdict.keys()):
        predictprobmean =  np.mean(predictprobdict[prefix], axis = 0)
        out.write(dicomdir + "\t" + prefix)
        #for (i,k) in zip(predictprobmean, views):
            #out.write("\n" + "prob_" + k + " :" + str(i))

        for i in predictprobmean:
            out.write("\t" + str(i))
            out.write( "\n")
    y = time.time()
    print("time:  " +str(y - x) + " seconds for " +  str(len(list(predictprobdict.keys())))  + " videos")
    #rmtree(temp_image_directory)
    out.close()
    

if __name__ == '__main__':
    main()
