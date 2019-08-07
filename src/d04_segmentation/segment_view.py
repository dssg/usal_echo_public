import time
from optparse import OptionParser

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.misc import imresize
from skimage.color import rgb2gray, gray2rgb
from datetime import datetime
import pandas as pd
import hashlib

from d00_utils.echocv_utils_v0 import *
from d00_utils.dcm_utils import dcm_to_segmentation_arrays
from d00_utils.db_utils import dbReadWriteViews, dbReadWriteSegmentation
from d04_segmentation.model_unet import Unet


## Set environment parameters
parser = OptionParser()
parser.add_option(
    "-d", "--dicomdir", dest="dicomdir", default="dicomsample", help="dicomdir"
)
parser.add_option("-g", "--gpu", dest="gpu", default="0", help="cuda device to use")
parser.add_option("-M", "--modeldir", default="models", dest="modeldir")
params, args = parser.parse_args()
dicomdir = params.dicomdir
modeldir = params.modeldir

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu


def segmentChamber(videofile, dicomdir, view):
    """
    
    """
    # TODO: Need to put some error handling in here for when the file is not found 
    mean = 24
    weight_decay = 1e-12
    learning_rate = 1e-4
    maxout = False
    sesses = []
    models = []
    global modeldir

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
    outpath = "/home/ubuntu/courtney/usal_echo/data/segment/" + view + "/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    images, orig_images = dcm_to_segmentation_arrays(dicomdir, videofile)
    np_arrays_x3 = [];
    images_uuid_x3 = [];
    if view == "a4c":
        a4c_lv_segs, a4c_la_segs, a4c_lvo_segs, preds = extract_segs(
            images, orig_images, model, sess, 2, 4, 1)
        #np.save(outpath + "/" + videofile + "_lv", np.array(a4c_lv_segs).astype("uint8"))
        np_arrays_x3.append(np.array(a4c_lv_segs).astype("uint8"))
        #np.save(outpath + "/" + videofile + "_la", np.array(a4c_la_segs).astype("uint8"))
        np_arrays_x3.append(np.array(a4c_la_segs).astype("uint8"))
        #np.save(outpath + "/" + videofile + "_lvo", np.array(a4c_lvo_segs).astype("uint8"))
        np_arrays_x3.append(np.array(a4c_lvo_segs).astype("uint8"))
    elif view == "a2c":
        a2c_lv_segs, a2c_la_segs, a2c_lvo_segs, preds = extract_segs(
            images, orig_images, model, sess, 2, 3, 1)
        #np.save(outpath + "/" + videofile + "_lv", np.array(a2c_lv_segs).astype("uint8"))
        np_arrays_x3.append(np.array(a2c_lv_segs).astype("uint8"))
        #np.save(outpath + "/" + videofile + "_la", np.array(a2c_la_segs).astype("uint8"))
        np_arrays_x3.append(np.array(a2c_la_segs).astype("uint8"))
        #np.save(outpath + "/" + videofile + "_lvo", np.array(a2c_lvo_segs).astype("uint8"))
        np_arrays_x3.append(np.array(a2c_lvo_segs).astype("uint8"))
    '''
    elif view == "psax":
        psax_lv_segs, psax_lvo_segs, psax_rv_segs, preds = extract_segs(
            images, orig_images, model, sess, 2, 1, 3
        )
        np.save(
            outpath + "/" + videofile + "_lv", np.array(psax_lv_segs).astype("uint8")
        )
        np.save(
            outpath + "/" + videofile + "_lvo", np.array(psax_lvo_segs).astype("uint8")
        )
    elif view == "a3c":
        a3c_lv_segs, a3c_la_segs, a3c_lvo_segs, preds = extract_segs(
            images, orig_images, model, sess, 2, 3, 1
        )
        np.save(
            outpath + "/" + videofile + "_lvo", np.array(a3c_lvo_segs).astype("uint8")
        )
        np.save(
            outpath + "/" + videofile + "_lv", np.array(a3c_lv_segs).astype("uint8")
        )
        np.save(
            outpath + "/" + videofile + "_la", np.array(a3c_la_segs).astype("uint8")
        )
    elif view == "plax":
        plax_lv_segs, plax_la_segs, plax_ao_segs, preds = extract_segs(
            images, orig_images, model, sess, 1, 5, 3
        )
        np.save(
            outpath + "/" + videofile + "_lv", np.array(plax_lv_segs).astype("uint8")
        )
        np.save(
            outpath + "/" + videofile + "_la", np.array(plax_la_segs).astype("uint8")
        )
    '''
    j = 0
    nrow = orig_images[0].shape[0]
    ncol = orig_images[0].shape[1]
    print(nrow, ncol)
    plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.imshow(imresize(preds, (nrow, ncol)))
    plt.savefig(outpath + "/" + videofile + "_" + str(j) + "_" + "segmentation.png")
    images_uuid_x3.append(str(outpath + "/" + videofile + "_" + str(j) + "_" + "segmentation.png"))
    plt.close()
    plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.imshow(orig_images[0])
    plt.savefig(outpath + "/" + videofile + "_" + str(j) + "_" + "originalimage.png")
    images_uuid_x3.append(str(outpath + "/" + videofile + "_" + str(j) + "_" + "originalimage.png"))
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
    images_uuid_x3.append(str(outpath + "/" + videofile + "_" + str(j) + "_" + "overlay.png"))
    #return 1
    return [np_arrays_x3, images_uuid_x3]


def segmentstudy(viewlist_a2c, viewlist_a4c, viewlist_psax, viewlist_plax, dicomdir):
    #set up for writing to segmentation schema
    io_views = dbReadWriteViews()
    io_segmentation = dbReadWriteSegmentation()
    instances_unique_master_list = io_views.get_table('instances_unique_master_list')
    #below cleans the filename field
    instances_unique_master_list['instancefilename'] = instances_unique_master_list['instancefilename'].apply(lambda x: str(x).strip())
    #instance_id_dict = instances_unique_master_list.set_index('instancefilename')['instanceidk'].to_dict()
    #study_id_dict = instances_unique_master_list.set_index('instancefilename')['studyidk'].to_dict()
    
    for video in viewlist_a4c:
        [np_arrays_x3, images_uuid_x3] = segmentChamber(video, dicomdir, "a4c")
        instancefilename = video.split('_')[2].split('.')[0] #split from 'a_63712_45TXWHPP.dcm' to '45TXWHPP'
        studyidk = int(video.split('_')[1])
        #below filters to just the record of interest
        df = instances_unique_master_list.loc[(instances_unique_master_list['instancefilename']==instancefilename) & (instances_unique_master_list['studyidk']==studyidk)]
        df = df.reset_index()
        instance_id = df.at[0, 'instanceidk']
        d = [instance_id, studyidk, "a4c", np_arrays_x3[0], np_arrays_x3[1], np_arrays_x3[2], 
             images_uuid_x3[0], images_uuid_x3[1], images_uuid_x3[2], str(datetime.now()), 
             video]
        print(d)
        io_segmentation.save_numpy_array_to_db(d, 'predictions')
        '''
        d = {'instance_id': instance_id,
             'study_id': studyidk,
             'view_name': "a4c",
             'output_np': None,
             'output_image': images_uuid_x3,
             'date_run': datetime.now(),
             'file_name': video}
        df = pd.DataFrame(data=d)
        print(df)
        d_np = {'output_np':np_arrays_x3}
        df_np = pd.DataFrame(data=d_np).to_numpy()
        io_segmentation.save_numpy_array_to_db(df_np, 'predictions', 'output_np')
        io_segmentation.save_to_db(df, 'predictions')
        '''
    for video in viewlist_a2c:
        np_arrays_x3, images_uuid_x3 = segmentChamber(video, dicomdir, "a2c")
        instancefilename = video.split('_')[2].split('.')[0] #split from 'a_63712_45TXWHPP.dcm' to '45TXWHPP'
        studyidk = int(video.split('_')[1])
        #below filters to just the record of interest
        df = instances_unique_master_list.loc[(instances_unique_master_list['instancefilename']==instancefilename) & (instances_unique_master_list['studyidk']==studyidk)]
        df = df.reset_index()
        instance_id = df.at[0, 'instanceidk']
        d = [instance_id, studyidk, "a2c", np_arrays_x3[0], np_arrays_x3[1], np_arrays_x3[2], 
             images_uuid_x3[0], images_uuid_x3[1], images_uuid_x3[2], str(datetime.now()), 
             video]
        print(d)
        io_segmentation.save_numpy_array_to_db(d, 'predictions')
        '''
        d = {'instance_id': instance_id,
             'study_id': studyidk,
             'view_name': "a2c",
             'output_np': None,
             'output_image': images_uuid_x3,
             'date_run': datetime.now(),
             'file_name': video}
        df = pd.DataFrame(data=d)
        print(df)
        d_np = {'output_np':np_arrays_x3}
        df_np = pd.DataFrame(data=d_np).to_numpy()
        io_segmentation.save_numpy_array_to_db(df_np, 'predictions', 'output_np')
        io_segmentation.save_to_db(df, 'predictions')
        #io_segmentation.save_numpy_array_to_db(df.to_numpy(), 'predictions')
        '''
    return 1
    
'''
    for video in viewlist_psax:
        np_arrays_x3, images_uuid_x3 = segmentChamber(video, dicomdir, "psax")
        io_segmentation.save_to_db(instance_id_dict.get(video), 
                                   study_id_dict.get(video), 
                                   "psax",
                                   np_arrays_x3,
                                   images_uuid_x3,
                                   datetime.now(),
                                   video)
    for video in viewlist_plax:
        np_arrays_x3, images_uuid_x3 = segmentChamber(video, dicomdir, "plax")
        io_segmentation.save_to_db(instance_id_dict.get(video), 
                                   study_id_dict.get(video), 
                                   "plax",
                                   np_arrays_x3,
                                   images_uuid_x3,
                                   datetime.now(),
                                   video)
    
'''

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


def main():
    # To use dicomdir option set in global scope.
    global dicomdir
    # In case dicomdir is path with more than one part.
    # dicomdir_basename = os.path.basename(dicomdir)
    viewfile = "/home/ubuntu/courtney/usal_echo/data/d04_segmentation/view_probabilities_test2019-08-07.txt"
    # viewfile = '/home/ubuntu/courtney/usal_echo/data/d04_segmentation/view_23_e5_class_11-Mar-2018_dcm_sample_labelled_probabilities.txt'
    viewlist_a2c = []
    viewlist_a3c = []
    viewlist_a4c = []
    viewlist_plax = []
    viewlist_psax = []

    infile = open(
        "/home/ubuntu/courtney/usal_echo/src/d03_classification/viewclasses_view_23_e5_class_11-Mar-2018.txt"
    )
    infile = infile.readlines()
    infile = [i.rstrip() for i in infile]

    viewdict = {}

    for i in range(len(infile)):
        viewdict[infile[i]] = i + 2

    probthresh = (
        0.5
    )  # arbitrary choice of "probability" threshold for view classification

    infile = open(viewfile)
    infile = infile.readlines()
    infile = [i.rstrip() for i in infile]
    infile = [i.split("\t") for i in infile]

    start = time.time()
    for i in infile[1:]:
        print(i)
        dicomdir = i[0]
        filename = i[1]
        print(filename)
        if eval(i[viewdict["psax_pap"]]) > probthresh:
            viewlist_psax.append(filename)
        elif eval(i[viewdict["a4c"]]) > probthresh:
            viewlist_a4c.append(filename)
        elif eval(i[viewdict["a2c"]]) > probthresh:
            viewlist_a2c.append(filename)
        elif eval(i[viewdict["a3c"]]) > probthresh:
            viewlist_a3c.append(filename)
        elif eval(i[viewdict["plax_plax"]]) > probthresh:
            viewlist_plax.append(filename)
    print(viewlist_a2c, viewlist_a4c, viewlist_a3c, viewlist_psax, viewlist_plax)
    segmentstudy(viewlist_a2c, viewlist_a4c, viewlist_psax, viewlist_plax, dicomdir)
    tempdir = os.path.join(dicomdir, "image")
    end = time.time()
    viewlist = viewlist_a2c + viewlist_a4c + viewlist_psax + viewlist_plax
    print(
        "time:  " + str(end - start) + " seconds for " + str(len(viewlist)) + " videos"
    )
 

if __name__ == "__main__":
    main()
