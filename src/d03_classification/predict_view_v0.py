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
import vgg as network
import numpy as np
from scipy.misc import imread

from d00_utils.dcm_utils_v0 import output_imgdict

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

<<<<<<< HEAD
=======
import d03_classification.vgg as network

>>>>>>> 011273296ebb6d1eb7c126549ae8531ca851e7ad
os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu


def read_dicom(out_directory, filename, counter):
    """
    
    """
    if counter < 50:
        outrawfilename = filename + "_raw"
        print(out_directory, filename, counter, "trying")
        if os.path.exists(os.path.join(out_directory, outrawfilename)):
            time.sleep(2)
            try:
                ds = pydicom.read_file(os.path.join(out_directory, outrawfilename),
                                 force=True)
                framedict = output_imgdict(ds)
                y = len(list(framedict.keys())) - 1
                if y > 10:
                    m = random.sample(list(range(0, y)), 10)
                    for n in m:
                        targetimage = framedict[n][:]
                        outfile = os.path.join(out_directory, filename) + str(n) + '.jpg'
                        cv2.imwrite(outfile, \
                                cv2.resize(targetimage, (224, 224)), [cv2.IMWRITE_JPEG_QUALITY, 95])
                        counter = 50
            except (IOError, EOFError, KeyError) as e:
                print(out_directory + "\t" + outrawfilename + "\t" +
                      "error", counter, e)
        else:
            counter = counter + 1
            time.sleep(3)
            read_dicom(out_directory, filename, counter)
    return counter


def extract_imgs_from_dicom(directory, out_directory):
    """
    Extracts jpg images from DCM files in the given directory

    @param directory: folder with DCM files of echos
    @param out_directory: destination folder to where converted jpg files are placed
    @param target: destination folder to where converted jpg files are placed
    """
    allfiles = os.listdir(directory)

    for filename in allfiles[:]:
        # Prefix differs for sample files and our files.
        global dicomdir
        prefix = "Image" if 'dicomsample' in dicomdir else "a"
        if filename.startswith(prefix):
            ds = pydicom.read_file(os.path.join(directory, filename),
                                 force=True)
            if ("NumberOfFrames" in  dir(ds)) and (ds.NumberOfFrames>1):
                outrawfilename = filename + "_raw"
                command = 'gdcmconv -w ' + os.path.join(directory, filename) + " " + os.path.join(out_directory, outrawfilename)
                subprocess.Popen(command, shell=True)
                filesize = os.stat(os.path.join(directory, filename)).st_size
                time.sleep(3)
                counter = 0
                while counter < 5:
                    counter = read_dicom(out_directory, filename, counter)
                    counter = counter + 1
    return 1


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
    model = network.Network(0.0, 0.0, feature_dim, label_dim, False)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, model_name)

    for filename in imagedict:
        predictions[filename] =np.around(model.probabilities(sess, \
                  imagedict[filename]), decimals = 3)
    
    return predictions


def main():
    model = "view_23_e5_class_11-Mar-2018"
    # To use dicomdir option set in global scope.
    global dicomdir, modeldir
    model_name = os.path.join(modeldir, model)

    infile = open("d03_classification/viewclasses_" + model + ".txt")
    infile = infile.readlines()
    views = [i.rstrip() for i in infile]

    feature_dim = 1
    label_dim = len(views)

    probabilities_dir = 'probabilities/'
    #probabilities_dir = '/home/ubuntu/data/d03_classification/probabilities'
    if not os.path.exists(probabilities_dir):
        os.makedirs(probabilities_dir)
    # In case dicomdir is path with more than one part.
    dicomdir_basename = os.path.basename(dicomdir)
    out = open(probabilities_dir + model + "_" + dicomdir_basename + "_probabilities.txt", 'w')
    out.write("study\timage")
    for j in views:
        out.write("\t" + j)
    out.write('\n')
    x = time.time()
    temp_image_directory = os.path.join(dicomdir, 'image/')
    #if os.path.exists(temp_image_directory):
        #rmtree(temp_image_directory)
    if not os.path.exists(temp_image_directory):
        os.makedirs(temp_image_directory)
    extract_imgs_from_dicom(dicomdir, temp_image_directory)
    predictions = classify(temp_image_directory, feature_dim, label_dim, model_name)
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
