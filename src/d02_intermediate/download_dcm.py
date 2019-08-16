# coding: utf-8

import random
import os
import numpy as np
import subprocess

from scipy.misc import imresize
import cv2
import pydicom
from skimage.color import rgb2gray

from d00_utils.db_utils import dbReadWriteViews
from d00_utils.s3_utils import download_s3_objects
from d00_utils.log_utils import setup_logging

logger = setup_logging(__name__, __name__)


def _ybr2gray(y, u, v):
    r = y + 1.402 * (v - 128)
    g = y - 0.34414 * (u - 128) - 0.71414 * (v - 128)
    b = y + 1.772 * (u - 128)
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return np.array(gray, dtype="int8")


def _decompress_dcm(dcm_filepath, dcmraw_filepath):

    dcm_dir = os.path.dirname(dcmraw_filepath)
    os.makedirs(dcm_dir, exist_ok=True)

    command = "gdcmconv -w " + dcm_filepath + " " + dcmraw_filepath

    try:
        subprocess.run(command, shell=True, check=True)
        logger.info(
            "{} decompressed".format(os.path.basename(dcmraw_filepath).split(".")[0])
        )

    except subprocess.CalledProcessError as e:
        logger.error(
            "{} FAILED to decompress - {}".format(
                os.path.basename(dcmraw_filepath).split(".")[0], e
            )
        )

        
def _split_train_test(ratio, table_name):
    """Split views.table_name into train/test. 
    
    :param ratio (float): ratio for splitting into train/test
                 e.g. if 0.8, will take 80% as train set and 20% as test set
    :param table_name (str): name of views.table with master instances

    """
    io_views = dbReadWriteViews()
    df = io_views.get_table(table_name)

    np.random.seed(0)
    msk = np.random.rand(len(df)) < ratio
    df_train = df[msk].reset_index(drop=True)
    df_test = df[~msk].reset_index(drop=True)

    perc_trn = int(100 * ratio)
    perc_tst = 100 - perc_trn
    logger.info(
        "{} split into {}% train, {}% test".format(table_name, perc_trn, perc_tst)
    )

    return df_train, df_test


def _downsample_train_test(downsample_ratio, train_test_ratio, table_name):
    """Downsamples views.table_name train/test dataset by a factor of X.

    :param downsample_ratio (float): percentage by which to downsample dataset
                         e.g. if ratio=0.1, will downsample by a factor of 10
    :param train_test_ratio (float): ratio for splitting into train/test
    :param table_name (str): name of views.table with master instancest

    """
    df_train, df_test = _split_train_test(train_test_ratio, table_name)

    np.random.seed(0)
    msk_train = np.random.rand(len(df_train)) < downsample_ratio
    msk_test = np.random.rand(len(df_test)) < downsample_ratio

    df_train_downsampled = df_train[msk_train].reset_index(drop=True)
    df_test_downsampled = df_test[msk_test].reset_index(drop=True)

    inv_ratio = int(1 / downsample_ratio)
    logger.info("{} downsampled by a factor of {}".format(table_name, inv_ratio))

    return df_train_downsampled, df_test_downsampled


def s3_download_decomp_dcm(train_test_ratio, downsample_ratio, dcm_dir, table_name='instances_w_labels', train=False, bucket='cibercv'):
    """Downloads and decompresses test/train dicoms from s3.
    
    :param downsample_ratio (float): percentage by which to downsample dataset
                         e.g. if ratio=0.1, will downsample by a factor of 10
    :param train_test_ratio (float): ratio for splitting into train/test
    :param table_name (str): name of views.table with master instancest
    :param train (bool): download train set instead of test set, default=False
    
    """
    
    df_train, df_test = _downsample_train_test(
        downsample_ratio, train_test_ratio, table_name
    )

    if train is True:
        instances = df_train
        dir_name = "train_split{}_downsampleby{}".format(
            int(100 * train_test_ratio), int(1 / downsample_ratio)
        )
    else:
        instances = df_test
        dir_name = "test_split{}_downsampleby{}".format(
            int(100 - 100 * train_test_ratio), int(1 / downsample_ratio)
        )

    prefix = (
        instances["studyidk"].astype(str) + "/a_" + instances["filename"].astype(str)
    )
    filenames = (
        "a_"
        + instances["studyidk"].astype(str)
        + "_"
        + instances["filename"].astype(str)
        + ".dcm"
    )

    download_dict = dict(zip(prefix, filenames))

    datadir = os.path.join(dcm_dir, dir_name)
    raw_datadir = os.path.join(datadir, "raw")

    for p in list(download_dict.keys()):
        dcm_filepath = os.path.join(datadir, download_dict[p])
        # TODO: consider downloading to a main image directory and only creating a
        # symlink to experiment directory. Could save a lot of downloading time.
        if not os.path.isfile(dcm_filepath):
            download_s3_objects(
                bucket, outfile=dcm_filepath, prefix=p, suffix=".dcm"
            )

        dcm_rawfilepath = os.path.join(raw_datadir, download_dict[p] + "_raw")
        if not os.path.isfile(dcm_rawfilepath):
            _decompress_dcm(dcm_filepath, dcm_rawfilepath)

    logger.info(
        "Downloaded {} [train/test split = {}, downsample ratio = {}]".format(
            table_name, train_test_ratio, downsample_ratio
        )
    )

    return dir_name


def _read_dcmraw(dcmraw_filepath):
    
    try:
        ds = pydicom.dcmread(dcmraw_filepath, force=True)
        if ("NumberOfFrames" in dir(ds)) and (ds.NumberOfFrames > 1):
            return ds
        else:
            logger.debug("{} is a single frame".format(os.path.basename(dcmraw_filepath)))
    except IOError:
        print('file {} not found'.format(dcmraw_filepath))


def _dcmraw_to_np(dcmraw_obj):
    """Converts frames of decompressed dicom object to dictionary of numpy arrays.

    :param dcmraw_obj (pydicom): pydicom.dcmread() object

    """
    # pydicom reads ds.pixel array as (nframes, nrow, ncol, nchannels)
    # pxl_array is a copy of ds.pixel_array with dicom's format
    pxl_array = np.moveaxis(dcmraw_obj.pixel_array, -1, 0)

    if len(pxl_array.shape) == 4:  # format 3, nframes, nrow, ncol
        nframes = pxl_array.shape[1]
        maxframes = nframes * 3
    elif len(pxl_array.shape) == 3:  # format nframes, nrow, ncol
        nframes = pxl_array.shape[0]
        maxframes = nframes * 1

    nrow = int(dcmraw_obj.Rows)
    ncol = int(dcmraw_obj.Columns)
    ArrayDicom = np.zeros((nrow, ncol), dtype=pxl_array.dtype)
    framedict = {}

    for counter in range(0, maxframes, 3):  # iterate through all subframes
        k = counter % nframes
        j = (counter) // nframes
        m = (counter + 1) % nframes
        l = (counter + 1) // nframes
        o = (counter + 2) % nframes
        n = (counter + 2) // nframes

        if len(pxl_array.shape) == 4:
            a = pxl_array[j, k, :, :]
            b = pxl_array[l, m, :, :]
            c = pxl_array[n, o, :, :]
            ArrayDicom[:, :] = _ybr2gray(a, b, c)
            ArrayDicom[0 : int(nrow / 10), 0 : int(ncol)] = 0  # blanks out name
            counter = counter + 1
            ArrayDicom.clip(0)
            nrowout = nrow
            ncolout = ncol
            x = int(counter / 3)
            framedict[x] = imresize(ArrayDicom, (nrowout, ncolout))
        elif len(pxl_array.shape) == 3:
            ArrayDicom[:, :] = pxl_array[counter, :, :]
            ArrayDicom[0 : int(nrow / 10), 0 : int(ncol)] = 0  # blanks out name
            counter = counter + 1
            ArrayDicom.clip(0)
            nrowout = nrow
            ncolout = ncol
            x = int(counter / 3)
            framedict[x] = imresize(ArrayDicom, (nrowout, ncolout))

    return framedict


def _extract_framedict_from_dcmraw(dcmraw_filepath):
    """Extracts dicom frames as dictionary of numpy arrays.

    :param dcmraw_filepath: path to decompressed dicom file

    """
    dcmraw_obj = _read_dcmraw(dcmraw_filepath)
    framedict = _dcmraw_to_np(dcmraw_obj)

    return framedict


def _dcmraw_to_10_jpgs(dcmraw_filepath, img_dir):
    """Selects 10 frames from dicom image and saves them as jpg files.

    :param dcmraw_filepath: path to dicom file
    :param img_dir: directory for storing image files

    """
    os.makedirs(img_dir, exist_ok=True)
    filename = dcmraw_filepath.split("/")[-1].split(".")[0]

    all_imgs = os.listdir(img_dir)
    count_imgs = [s for s in all_imgs if filename in s]

    if len(count_imgs) < 10:
        framedict = _extract_framedict_from_dcmraw(dcmraw_filepath)
        y = len(list(framedict.keys())) - 1
        if y > 10:
            m = random.sample(list(range(0, y)), 10)
            for n in m:
                targetimage = framedict[n][:]
                outfile = os.path.join(img_dir, filename) + "_" + str(n) + ".jpg"
                cv2.imwrite(
                    outfile,
                    cv2.resize(targetimage, (224, 224)),
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )

        logger.info("{} 10 random frames extracted".format(filename))

    else:
        logger.info("{} frames exist".format(filename))

    return


def dcmdir_to_jpgs_for_classification(dcm_dir, img_dir):
    """Creates jpg images for all files in dcm_dir.

    The following processing steps are performed:
    0. Checks if raw dicom file exists.
    1. Decompresses and saves dicom video at dcm_filepath adding suffix '_raw'.

    :param dcm_dir: directory with dicom files
    :param img_dir: directory for storing image files

    """
    dcmraw_dir = os.path.join(dcm_dir, "raw")

    for filename in os.listdir(dcm_dir):

        if filename.endswith(".dcm"):
            dcm_filepath = os.path.join(dcm_dir, filename)
            dcmraw_filepath = os.path.join(dcmraw_dir, filename + "_raw")

            if not os.path.isfile(dcmraw_filepath):
                _decompress_dcm(dcm_filepath, dcmraw_filepath)
            try:
                dcm_filepath = os.path.join(dcm_dir, filename)
                _dcmraw_to_10_jpgs(dcm_filepath, img_dir)
            except AttributeError:
                logger.error("{} could not save images".format(filename))


def dcm_to_segmentation_arrays(dcm_dir, filename):
    """Creates a numpy array of all frames for filename in dcm_dir.
    
    :param dcm_dir: directory with dicom files
    :param filename: path to dicom file

    """

    dcmraw_dir = os.path.join(dcm_dir, "raw")
    dcm_filepath = os.path.join(dcm_dir, filename)
    dcmraw_filepath = os.path.join(dcmraw_dir, filename + "_raw")

    if not os.path.isfile(dcmraw_filepath):
        _decompress_dcm(dcm_filepath, dcmraw_filepath)

    try:
        framedict = _extract_framedict_from_dcmraw(dcmraw_filepath)
        images = []
        orig_images = []

        for key in list(framedict.keys()):
            image = np.zeros((384, 384))
            image[:, :] = imresize(rgb2gray(framedict[key]), (384, 384, 1))
            images.append(image)
            orig_images.append(framedict[key])

        images = np.array(images).reshape((len(images), 384, 384, 1))

        return images, orig_images

    except AttributeError:
        logger.error("{} could not return dict".format(filename))
