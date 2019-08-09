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

from d02_intermediate.dcm_utils import dcmdir_to_jpgs_for_classification
from d00_utils.log_utils import *
logger = setup_logging(__name__, "d03_classification")

from d03_classification import vgg

sys.path.append("./funcs/")
sys.path.append("./nets/")

# Hyperparams --DELETE
parser = OptionParser()
parser.add_option(
    "-d", "--dicomdir", dest="dicomdir", default="dicomsample", help="dicomdir")
parser.add_option("-g", "--gpu", dest="gpu", default="0", help="cuda device to use")
parser.add_option(
    "-M", "--modeldir", dest="modeldir", default="models", help="modeldir")
parser.add_option("-m", "--model", dest="model")
params, args = parser.parse_args()
dicomdir = params.dicomdir
modeldir = params.modeldir
model = params.model

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu


def classify(directory, feature_dim, label_dim, model_name):
    """
    Classifies echo images in given directory

    :param directory: folder with jpg echo images for classification
    
    """
    imagedict = {}
    predictions = {}
    for filename in os.listdir(directory):
        if "jpg" in filename:
            image = imread(directory + filename, flatten=True).astype("uint8")
            imagedict[filename] = [image.reshape((224, 224, 1))]

    tf.reset_default_graph()
    sess = tf.Session()
    model = vgg.Network(0.0, 0.0, feature_dim, label_dim, False)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, model_name)

    for filename in imagedict:
        predictions[filename] = np.around(
            model.probabilities(sess, imagedict[filename]), decimals=3
        )

    return predictions


def run_classify():

    # To use dicomdir option set in global scope.
    global dicomdir, modeldir  # TODO should not be setting global parameters

    # Create directories for saving images
    results_dir = (
        "/home/ubuntu/data/03_classification/results/"
    )  # TODO this shouldn't be hardcoded

    os.makedirs(results_dir, exist_ok=True)
    #temp_image_dir = os.path.join(dicomdir, "image/")
    temp_image_dir = '/home/ubuntu/data/02_intermediate/test_downsampleby5/'

    #model_name = os.path.join(modeldir, model)
    model = 'view_23_e5_class_11-Mar-2018'
    model_name = '/home/ubuntu/models/view_23_e5_class_11-Mar-2018'

    # TODO: soft-code this
    infile = open('/home/ubuntu/dvv/usal_echo/src/d03_classification/viewclasses_' + model + '.txt')
    infile = infile.readlines()
    views = [i.rstrip() for i in infile]

    feature_dim = 1
    label_dim = len(views)

    # In case dicomdir is path with more than one part.
    dicomdir_basename = os.path.basename(dicomdir)
    out = open(
        results_dir + model + "_" + dicomdir_basename + "_probabilities.txt", "w"
    )
    out.write("study\timage")

    for j in views:
        out.write("\t" + j)
    out.write("\n")
    x = time.time()

    #dcmdir_to_jpgs_for_classification(dicomdir, temp_image_dir)
    predictions = classify(temp_image_dir, feature_dim, label_dim, model_name)
    predictprobdict = {}

    for image in list(predictions.keys()):
        prefix = image.split(".dcm")[0] + ".dcm"
        if prefix not in predictprobdict:
            predictprobdict[prefix] = []
        predictprobdict[prefix].append(predictions[image][0])

    for prefix in list(predictprobdict.keys()):
        predictprobmean = np.mean(predictprobdict[prefix], axis=0)
        out.write(dicomdir + "\t" + prefix)

    for i in predictprobmean:
        out.write("\t" + str(i))
        out.write("\n")
    
    y = time.time()

    print(
        "time:  "
        + str(y - x)
        + " seconds for "
        + str(len(list(predictprobdict.keys())))
        + " videos"
    )
    # rmtree(temp_image_directory)
    out.close()


#if __name__ == "__main__":
#    run_classify(model="view_23_e5_class_11-Mar-2018")
