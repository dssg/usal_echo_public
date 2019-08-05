# coding: utf-8

import random
import os
import subprocess

import numpy as np
from scipy.misc import imresize
import cv2
import pydicom
from skimage.color import rgb2gray

from src.d00_utils.log_utils import *
logger = setup_logging(__name__, 'download_decompress_dcm')


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
    subprocess.Popen(command, shell=True)
    logger.info('{} decompressed'.format(os.path.basename(dcm_filepath)))

    return


def _read_dcmraw(dcmraw_filepath):

    ds = pydicom.dcmread(dcmraw_filepath, force=True)
    if ("NumberOfFrames" in dir(ds)) and (ds.NumberOfFrames > 1):
        return ds
    else:
        logger.debug("{} is a single frame".format(os.path.basename(dcmraw_filepath)))


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
        nframes = pxl_array.shape[1]
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
            d = np.vstack((a, b))
            e = np.vstack((d, c))
            g = e.reshape(3 * nrow * ncol, 1)
            y = g[::3]
            u = g[1::3]
            v = g[2::3]
            y = y.reshape(nrow, ncol)
            u = u.reshape(nrow, ncol)
            v = v.reshape(nrow, ncol)
            ArrayDicom[:, :] = _ybr2gray(y, u, v)
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


def extract_framedict_from_dcmraw(dcmraw_filepath):
    """Extracts dicom frames as dictionary of numpy arrays.

    The following processing steps are performed:
    1. Reads in the raw dicom file.
    2. Creates a dictionary of numpy arrays for all frames in the file.

    :param dcmraw_filepath: path to decompressed dicom file

    """
    dcmraw_obj = _read_dcmraw(dcmraw_filepath)
    framedict = _dcmraw_to_np(dcmraw_obj)

    return framedict


def dcmraw_to_10_jpgs(dcmraw_filepath, img_dir):
    """Selects 10 frames from dicom image and saves them as jpg files.

    :param dcmraw_filepath: path to dicom file
    :param img_dir: directory for storing image files

    """
    os.makedirs(img_dir, exist_ok=True)
    filename = dcmraw_filepath.split("/")[-1].split('.')[0]
    framedict = extract_framedict_from_dcmraw(dcmraw_filepath)

    y = len(list(framedict.keys())) - 1
    if y > 10:
        m = random.sample(list(range(0, y)), 10)
        for n in m:
            targetimage = framedict[n][:]
            outfile = os.path.join(img_dir, filename) + str(n) + ".jpg"
            cv2.imwrite(
                outfile,
                cv2.resize(targetimage, (224, 224)),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

    logger.info('{} 10 random frames extracted'.format(filename))

    return


def dcmdir_to_jpgs_for_classification(dcm_dir, img_dir):
    """Creates jpg images for all files in dcm_dir.

    The following processing steps are performed:
    0. Checks if raw dicom file exists.
    1. Decompresses and saves dicom video at dcm_filepath adding suffix '_raw'.

    :param dcm_dir: directory with dicom files
    :param img_dir: directory for storing image files

    """
    dcmraw_dir = os.path.join(dcm_dir, 'raw')

    for filename in os.listdir(dcm_dir):
        
        if filename.endswith(".dcm"):
            dcm_filepath = os.path.join(dcm_dir, filename)
            dcmraw_filepath = os.path.join(dcmraw_dir, filename + "_raw")
            
            if not os.path.isfile(dcmraw_filepath):
                _decompress_dcm(dcm_filepath, dcmraw_filepath)
            try:
                dcm_filepath = os.path.join(dcm_dir, filename)
                dcmraw_to_10_jpgs(dcm_filepath, img_dir)
            except AttributeError:
                logger.error('{} could not save images'.format(filename))

    return


def s3_to_jpgs_for_classification():

    # TODO

    return


def dcm_to_segmentation_arrays(dcm_dir, filename):
    """Creates a numpy array of all frames for filename in dcm_dir.
    
    :param dcm_dir: directory with dicom files
    :param filename: path to dicom file

    """

    dcmraw_dir = os.path.join(dcm_dir, 'raw')
    dcm_filepath = os.path.join(dcm_dir, filename)
    dcmraw_filepath = os.path.join(dcmraw_dir, filename + "_raw")

    if not os.path.isfile(dcmraw_filepath):
        _decompress_dcm(dcm_filepath, dcmraw_filepath)

    try:
        framedict = extract_framedict_from_dcmraw(dcmraw_filepath)
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
        logger.error('{} could not return dict'.format(filename))
