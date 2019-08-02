# coding: utf-8

import random
import sys
import os
import subprocess
from subprocess import Popen, PIPE
import time
from optparse import OptionParser
from shutil import rmtree

import numpy as np
from scipy.misc import imread, imresize
import cv2
import pydicom


def _ybr2gray(y, u, v):
    r = y + 1.402 * (v - 128)
    g = y - 0.34414 * (u - 128) - 0.71414 * (v - 128)
    b = y + 1.772 * (u - 128)
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return np.array(gray, dtype="int8")


def _decompress_dcm(dcm_filepath):
    """Decompresses and saves dicom videos at dcm_filepath with suffix '_raw'.
    
    """
    dcmraw_filepath = dcm_filepath + "_raw"
    command = "gdcmconv -w " + dcm_filepath + " " + dcmraw_filepath
    subprocess.Popen(command, shell=True)

    return


def _read_dcmraw(dcm_dir, filename):

    dcmraw_filepath = os.path.join(dcm_dir, filename + "_raw")
    ds = pydicom.read_file(dcmraw_filepath, force=True)

    if ("NumberOfFrames" in dir(ds)) and (ds.NumberOfFrames > 1):
        return dcmraw_obj

    else:
        print(filename, "is a single frame")


def _dcmraw_to_np(dcmraw_obj):
    """Converts frames of decompressed dicom object to dictionary of numpy arrays.
    
    :param dcmraw_obj (pydicom): pydicom.read_file() object
    
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
    imgdict = {}

    for counter in range(
        0, maxframes, 3
    ):  # this will iterate through all subframes for a loop
        k = counter % nframes
        j = (counter) // nframes
        m = (counter + 1) % nframes
        l = (counter + 1) // nframes
        o = (counter + 2) % nframes
        n = (counter + 2) // nframes
        # print("j", j, "k", k, "l", l, "m", m, "n", n, "o", o)

        if len(pxl_array.shape) == 4:
            a = pxl_array[j, k, :, :]
            b = pxl_array[l, m, :, :]
            c = pxl_array[n, o, :, :]
            d = np.vstack((a, b))
            e = np.vstack((d, c))
            # print(e.shape)
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
            imgdict[x] = imresize(ArrayDicom, (nrowout, ncolout))
        elif len(pxl_array.shape) == 3:
            ArrayDicom[:, :] = pxl_array[counter, :, :]
            ArrayDicom[0 : int(nrow / 10), 0 : int(ncol)] = 0  # blanks out name
            counter = counter + 1
            ArrayDicom.clip(0)
            nrowout = nrow
            ncolout = ncol
            x = int(counter / 3)
            imgdict[x] = imresize(ArrayDicom, (nrowout, ncolout))

    return imgdict


def _dcmraw_to_10_jpgs(dcm_dir, img_dir, filename):
    """
    For CLASSIFICATION.
    
    """
    dcmraw_obj = _read_dcmraw(dcm_dir, filename)
    framedict = _dcmraw_to_np(dcmraw_obj)

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

    return


def extract_jpgs_from_dcmdir(dcm_dir, img_dir):
    """Extracts jpg images from DCM files in the given directory for CLASSIFICATION.

    :param directory: directory with DCM files of echos
    :param out_directory: destination folder to where converted jpg files are placed
    
    """
    for filename in os.listdir(dcm_dir):
        if file.endswith(".dcm"):
            dcm_filepath = os.path.join(dcm_dir, filename)
            if os.path.isfile(dcm_filepath + "_raw"):
                continue
            else:
                _decompress_dcm(dcm_filepath)

            _dcmraw_to_10_jpgs(img_dir, filename)

    return


######################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%_________________
# SEGMENTATION


def create_imgdict_from_dicom(directory, filename):
    """
    Convert compressed DICOM format into numpy array for SEGMENTATION.
    
    """
    temp_directory = os.path.join(directory, "image")
    os.makedirs(temp_directory, exist_ok=True)

    targetfile = os.path.join(directory, filename)
    ds = pydicom.read_file(targetfile, force=True)

    if ("NumberOfFrames" in dir(ds)) and (ds.NumberOfFrames > 1):
        out_raw_filepath = os.path.join(temp_directory, filename + "_raw")
        command = (
            "gdcmconv -w " + os.path.join(directory, filename) + " " + out_raw_filepath
        )
        subprocess.Popen(command, shell=True)

        if os.path.exists(out_raw_filepath):
            ds = pydicom.read_file(out_raw_filepath)
            imgdict = _dcmraw_to_np(ds)
        else:
            print(out_raw_filepath, "missing")

    return imgdict


def extract_images(framedict):
    """
    Used for SEGMENTATION.
    """
    images = []
    orig_images = []

    for key in list(framedict.keys()):
        image = np.zeros((384, 384))
        image[:, :] = imresize(rgb2gray(framedict[key]), (384, 384, 1))
        images.append(image)
        orig_images.append(framedict[key])

    images = np.array(images).reshape((len(images), 384, 384, 1))

    return images, orig_images
