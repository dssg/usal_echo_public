import time
from optparse import OptionParser

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.misc import imresize
from datetime import datetime
import hashlib

#from d00_utils.echocv_utils_v0 import *
from d02_intermediate.download_dcm import dcm_to_segmentation_arrays
#from d00_utils.dcm_utils import dcm_to_segmentation_arrays
from d00_utils.db_utils import dbReadWriteViews, dbReadWriteClassification, dbReadWriteSegmentation
from d00_utils.echocv_utils_v0 import *
#from d02_intermediate.dcm_utils import dcm_to_segmentation_arrays
from d04_segmentation.model_unet import Unet


## Set environment parameters
#parser = OptionParser()
#parser.add_option(
#    "-d", "--dicomdir", dest="dicomdir", default="dicomsample", help="dicomdir"
#)
#parser.add_option("-g", "--gpu", dest="gpu", default="0", help="cuda device to use")
#parser.add_option("-M", "--modeldir", default="models", dest="modeldir")
#params, args = parser.parse_args()
#dicomdir = params.dicomdir
#modeldir = params.modeldir

#os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu

os.environ["CUDA_VISIBLE_DEVICES"]="0"


def segmentChamber(videofile, dicomdir, view, model_path):
    """
    
    """
    # TODO: Need to put some error handling in here for when the file is not found
    mean = 24
    weight_decay = 1e-12
    learning_rate = 1e-4
    maxout = False
#    sesses = []
#    models = []
    modeldir = model_path

    print(videofile, dicomdir)

    if view == "a4c":
        g_1 = tf.Graph()
        with g_1.as_default():
            label_dim = 6  # a4c
            sess1 = tf.Session()
            model1 = Unet(mean, weight_decay, learning_rate, label_dim, maxout=maxout)
            sess1.run(tf.local_variables_initializer())
            sess = sess1
            model = model1
        with g_1.as_default():
            saver = tf.train.Saver()
            saver.restore(
                sess1, os.path.join(modeldir, "a4c_45_20_all_model.ckpt-9000")
            )
    elif view == "a2c":
        g_2 = tf.Graph()
        with g_2.as_default():
            label_dim = 4
            sess2 = tf.Session()
            model2 = Unet(mean, weight_decay, learning_rate, label_dim, maxout=maxout)
            sess2.run(tf.local_variables_initializer())
            sess = sess2
            model = model2
        with g_2.as_default():
            saver = tf.train.Saver()
            saver.restore(
                sess2, os.path.join(modeldir, "a2c_45_20_all_model.ckpt-10600")
            )
    elif view == "a3c":
        g_3 = tf.Graph()
        with g_3.as_default():
            label_dim = 4
            sess3 = tf.Session()
            model3 = Unet(mean, weight_decay, learning_rate, label_dim, maxout=maxout)
            sess3.run(tf.local_variables_initializer())
            sess = sess3
            model = model3
        with g_3.as_default():
            saver.restore(
                sess3, os.path.join(modeldir, "a3c_45_20_all_model.ckpt-10500")
            )
    elif view == "psax":
        g_4 = tf.Graph()
        with g_4.as_default():
            label_dim = 4
            sess4 = tf.Session()
            model4 = Unet(mean, weight_decay, learning_rate, label_dim, maxout=maxout)
            sess4.run(tf.local_variables_initializer())
            sess = sess4
        model = model4
        with g_4.as_default():
            saver = tf.train.Saver()
            saver.restore(
                sess4, os.path.join(modeldir, "psax_45_20_all_model.ckpt-9300")
            )
    elif view == "plax":
        g_5 = tf.Graph()
        with g_5.as_default():
            label_dim = 7
            sess5 = tf.Session()
            model5 = Unet(mean, weight_decay, learning_rate, label_dim, maxout=maxout)
            sess5.run(tf.local_variables_initializer())
            sess = sess5
            model = model5
        with g_5.as_default():
            saver = tf.train.Saver()
            saver.restore(
                sess5, os.path.join(modeldir, "plax_45_20_all_model.ckpt-9600")
            )
    outpath = "/home/ubuntu/data/04_segmentation/" + view + "/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    images, orig_images = dcm_to_segmentation_arrays(dicomdir, videofile)
    np_arrays_x3 = []
    images_uuid_x3 = []
    if view == "a4c":
        a4c_lv_segs, a4c_la_segs, a4c_lvo_segs, preds = extract_segs(
            images, orig_images, model, sess, 2, 4, 1
        )
        np_arrays_x3.append(np.array(a4c_lv_segs).astype("uint8"))
        np_arrays_x3.append(np.array(a4c_la_segs).astype("uint8"))
        np_arrays_x3.append(np.array(a4c_lvo_segs).astype("uint8"))
        number_frames = (np.array(a4c_lvo_segs).astype("uint8").shape)[0]
        model_name = "a4c_45_20_all_model.ckpt-9000"
    elif view == "a2c":
        a2c_lv_segs, a2c_la_segs, a2c_lvo_segs, preds = extract_segs(
            images, orig_images, model, sess, 2, 3, 1
        )
        np_arrays_x3.append(np.array(a2c_lv_segs).astype("uint8"))
        np_arrays_x3.append(np.array(a2c_la_segs).astype("uint8"))
        np_arrays_x3.append(np.array(a2c_lvo_segs).astype("uint8"))
        number_frames = (np.array(a2c_lvo_segs).astype("uint8").shape)[0]
        model_name = "a2c_45_20_all_model.ckpt-10600"

    j = 0
    nrow = orig_images[0].shape[0]
    ncol = orig_images[0].shape[1]
    print(nrow, ncol)
    plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.imshow(imresize(preds, (nrow, ncol)))
    plt.savefig(outpath + "/" + videofile + "_" + str(j) + "_" + "segmentation.png")
    images_uuid_x3.append(
        hashlib.md5(
            (
                outpath + "/" + videofile + "_" + str(j) + "_" + "segmentation.png"
            ).encode()
        ).hexdigest()
    )
    plt.close()
    plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.imshow(orig_images[0])
    plt.savefig(outpath + "/" + videofile + "_" + str(j) + "_" + "originalimage.png")
    images_uuid_x3.append(
        hashlib.md5(
            (
                outpath + "/" + videofile + "_" + str(j) + "_" + "originalimage.png"
            ).encode()
        ).hexdigest()
    )
    plt.close()
    background = Image.open(
        outpath + "/" + videofile + "_" + str(j) + "_" + "originalimage.png"
    )
    overlay = Image.open(
        outpath + "/" + videofile + "_" + str(j) + "_" + "segmentation.png"
    )
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    outImage = Image.blend(background, overlay, 0.5)
    outImage.save(outpath + "/" + videofile + "_" + str(j) + "_" + "overlay.png", "PNG")
    images_uuid_x3.append(
        hashlib.md5(
            (outpath + "/" + videofile + "_" + str(j) + "_" + "overlay.png").encode()
        ).hexdigest()
    )
    # return 1
    return [number_frames, model_name, np_arrays_x3, images_uuid_x3]


def segmentstudy(viewlist_a2c, viewlist_a4c, dcm_path, model_path):
    
    # set up for writing to segmentation schema
    io_views = dbReadWriteViews()
    io_segmentation = dbReadWriteSegmentation()
    instances_unique_master_list = io_views.get_table("instances_unique_master_list")
    # below cleans the filename field
    instances_unique_master_list["instancefilename"] = instances_unique_master_list[
        "instancefilename"
    ].apply(lambda x: str(x).strip())
    #Columns names are:prediction_id	study_id	instance_id	file_name	
        #num_frames	model_name	date_run	output_np_lv	output_np_la	
        #output_np_lvo	output_image_seg	output_image_orig	output_image_overlay
    column_names = [
            "study_id",
            "instance_id",
            "file_name",
            "num_frames",
            "model_name",
            "date_run",
            "output_np_lv",
            "output_np_la",
            "output_np_lvo",
            "output_image_seg",
            "output_image_orig",
            "output_image_overlay",            
        ]

    for video in viewlist_a4c:
        print(video)
        print('for a4c')
        [number_frames, model_name, np_arrays_x3, images_uuid_x3] = segmentChamber(video, dcm_path, "a4c", model_path)
        instancefilename = video.split("_")[2].split(".")[
            0
        ]  # split from 'a_63712_45TXWHPP.dcm' to '45TXWHPP'
        studyidk = int(video.split("_")[1])
        # below filters to just the record of interest
        df = instances_unique_master_list.loc[
            (instances_unique_master_list["instancefilename"] == instancefilename)
            & (instances_unique_master_list["studyidk"] == studyidk)
        ]
        df = df.reset_index()
        instance_id = df.at[0, "instanceidk"]
        #Columns names are:prediction_id	study_id	instance_id	file_name	
        #num_frames	model_name	date_run	output_np_lv	output_np_la	
        #output_np_lvo	output_image_seg	output_image_orig	output_image_overlay
        d = [studyidk,
            instance_id,
            str(video),
            number_frames,
            model_name,
            str(datetime.now()),
            np_arrays_x3[0],
            np_arrays_x3[1],
            np_arrays_x3[2],
            images_uuid_x3[0],
            images_uuid_x3[1],
            images_uuid_x3[2]]
        io_segmentation.save_prediction_numpy_array_to_db(d, column_names)


    for video in viewlist_a2c:
        [number_frames, model_name, np_arrays_x3, images_uuid_x3] = segmentChamber(video, dicomdir, "a2c", model_path)
        instancefilename = video.split("_")[2].split(".")[
            0
        ]  # split from 'a_63712_45TXWHPP.dcm' to '45TXWHPP'
        studyidk = int(video.split("_")[1])
        # below filters to just the record of interest
        df = instances_unique_master_list.loc[
            (instances_unique_master_list["instancefilename"] == instancefilename)
            & (instances_unique_master_list["studyidk"] == studyidk)
        ]
        df = df.reset_index()
        instance_id = df.at[0, "instanceidk"]
        d = [studyidk,
             instance_id,
             str(video),
             number_frames,
             model_name,
             str(datetime.now()),
             np_arrays_x3[0],
             np_arrays_x3[1],
             np_arrays_x3[2],
             images_uuid_x3[0],
             images_uuid_x3[1],
             images_uuid_x3[2]]
        io_segmentation.save_prediction_numpy_array_to_db(d, column_names)

    return 1


def create_seg(output, label):
    output = output.copy()
    output[output != label] = -1
    output[output == label] = 1
    output[output == -1] = 0
    return output


def extract_segs(images, orig_images, model, sess, lv_label, la_label, lvo_label):
    segs = []
    preds = np.argmax(model.predict(sess, images[0:1])[0, :, :, :], 2)
    label_all = list(range(1, 8))
    label_good = [lv_label, la_label, lvo_label]
    for i in label_all:
        if not i in label_good:
            preds[preds == i] = 0
    for i in range(len(images)):
        seg = np.argmax(model.predict(sess, images[i : i + 1])[0, :, :, :], 2)
        segs.append(seg)
    lv_segs = []
    lvo_segs = []
    la_segs = []
    for seg in segs:
        la_seg = create_seg(seg, la_label)
        lvo_seg = create_seg(seg, lvo_label)
        lv_seg = create_seg(seg, lv_label)
        lv_segs.append(lv_seg)
        lvo_segs.append(lvo_seg)
        la_segs.append(la_seg)
    return lv_segs, la_segs, lvo_segs, preds


def run_segment(dcm_path, model_path):
    # To use dicomdir option set in global scope.
    #global dicomdir
    
    # In case dicomdir is path with more than one part.
    # dicomdir_basename = os.path.basename(dicomdir)
    #viewfile = "/home/ubuntu/courtney/usal_echo/data/d04_segmentation/view_probabilities_test2019-08-14.txt"
    # viewfile = '/home/ubuntu/courtney/usal_echo/data/d04_segmentation/view_23_e5_class_11-Mar-2018_dcm_sample_labelled_probabilities.txt'
    
    
        
    viewlist_a2c = []
    viewlist_a4c = []

    infile = open(
        "/home/ubuntu/courtney/usal_echo/src/d03_classification/viewclasses_view_23_e5_class_11-Mar-2018.txt"
    )
    infile = infile.readlines()
    infile = [i.rstrip() for i in infile]

    viewdict = {}

    for i in range(len(infile)):
        viewdict[infile[i]] = i + 2
    
    path = dcm_path

    file_path = []
    filenames = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.dcm' in file:
                file_path.append(os.path.join(r, file))
                fullfilename = os.path.basename(os.path.join(r, file))
                #print(str(fullfilename).split('.')[0])
                filenames.append(str(fullfilename).split('.')[0])
    print("Number of files in the directory: {}".format(len(file_path)))
    io_class = dbReadWriteClassification()
    predictions = io_class.get_table('predictions')
    filename_df = pd.DataFrame(filenames)
    #print(filename_df.head())
    
    file_predictions = pd.merge(filename_df, predictions, how='inner', left_on =[0], right_on = ['file_name'])
    print("Number of files successfully matched with predictions: {}".format(file_predictions.shape[0]))

    
    start = time.time()
    
    #for idx, row in predictions.iterrows():
    for idx, row in file_predictions.iterrows():
        pred_filename = row[0]
        if row[8] == 'a4c': 
            viewlist_a4c.append(str(pred_filename) + '.dcm')
            print(" {} appended to a4c list".format(str(pred_filename)))
        elif row[8] == 'a2c':
            viewlist_a2c.append(str(pred_filename) + '.dcm')
            print(" {} appended to a2c list ".format(str(pred_filename)))
    segmentstudy(viewlist_a2c, viewlist_a4c, dcm_path, model_path)
    end = time.time()
    viewlist = viewlist_a2c + viewlist_a4c
    print(
        "time:  " + str(end - start) + " seconds for " + str(len(viewlist)) + " videos"
    )

