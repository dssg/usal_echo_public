# coding: utf-8
import sys
import os
import dicom
import time
sys.path.append("/home/rdeo/anaconda/lib/python2.7/site-packages/")
import numpy as np
import subprocess
from subprocess import Popen, PIPE
from scipy.misc import imresize
import cv2

def computehr_gdcm(data):
    hr = "None"
    for i in data:
        i = i.lstrip()
        if i.split(" ")[0] == '(0018,1088)':
            hr = eval(i.split("[")[1].split("]")[0])
            print("heart rate found")
    return hr

def computexy_gdcm(data):
    for i in data:
        i = i.lstrip()
        if i.split(" ")[0] == '(0028,0010)':
            rows = i.split(" ")[2]
        elif i.split(" ")[0] == '(0028,0011)':
            cols = i.split(" ")[2]
    return int(rows), int(cols)

def computebsa_gdcm(data):
    '''
    dubois, height in m, weight in kg
    :param data: 
    :return: 
    '''
    for i in data:
        i = i.lstrip()
        if i.split(" ")[0] == '(0010,1020)':
            h = i.split("[")[1].split("]")[0]
        elif i.split(" ")[0] == '(0010,1030)':
            w = i.split("[")[1].split("]")[0]
    return 0.20247 * (eval(h)**0.725) * (eval(w)**0.425)

def computedeltaxy_gdcm(data):
    '''
    the unit is the number of cm per pixel 
    '''
    xlist = []
    ylist = []
    for i in data:
        i = i.lstrip()
        if i.split(" ")[0] == '(0018,602c)':
            deltax = i.split(" ")[2]
            if np.abs(eval(deltax)) > 0.012:
                xlist.append(np.abs(eval(deltax)))
        if i.split(" ")[0] == '(0018,602e)':
            deltay = i.split(" ")[2]
            if np.abs(eval(deltax)) > 0.012:
                ylist.append(np.abs(eval(deltay)))
    return np.min(xlist), np.min(ylist)

def remove_periphery(imgs):
    imgs_ret = []
    for img in imgs:
        image = img.astype('uint8').copy()
        fullsize = image.shape[0] * image.shape[1]
        image[image > 0 ] = 255
        image = cv2.bilateralFilter(image, 11, 17, 17)
        thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = cnts[1]
        areas = []
        for i in range(0, len(contours)):
            areas.append(cv2.contourArea(contours[i]))

        if len(areas) == 0:
            imgs_ret.append(img)
        else:
            select = np.argmax(areas)
            roi_corners_clean = []
            roi_corners = np.array(contours[select], dtype = np.int32)
            for i in roi_corners:
                roi_corners_clean.append(i[0])
            hull = cv2.convexHull(np.array([roi_corners_clean], dtype = np.int32))
            mask = np.zeros(image.shape, dtype=np.uint8)
            mask = cv2.fillConvexPoly(mask, hull, 1)
            imgs_ret.append(img*mask)
    return np.array(imgs_ret)

def computeft_gdcm(video, study, appdir):
    videodir = appdir + "static/studies/" + study.file
    command = 'gdcmdump ' + videodir + "/" + video.file + "| grep Frame"
    pipe = Popen(command, stdout=PIPE, stderr=None, shell=True)
    text = pipe.communicate()[0]
    data = text.split("\n")
    defaultframerate = 30
    counter = 0
    for i in data:
        if i.split(" ")[0] == '(0018,1063)':
            frametime = i.split(" ")[2][1:-1]
            counter = 1
        elif i.split(" ")[0] == '(0018,0040)':
            framerate = i.split("[")[1].split(" ")[0][:-1]
            frametime = str(1000 / eval(framerate))
            counter = 1
        elif i.split(" ")[0] == '(7fdf,1074)':
            framerate = i.split(" ")[3]
            frametime = str(1000 / eval(framerate))
            counter = 1
    if not counter == 1:
        print("missing framerate")
        framerate = defaultframerate
        frametime = str(1000 / framerate)
    ft = eval(frametime)
    return ft

def computeft_gdcm_strain(data):
    defaultframerate = None
    counter = 0
    for i in data:
        if i.split(" ")[0] == '(0018,1063)':
            frametime = i.split(" ")[2][1:-1]
            counter = 1
        elif i.split(" ")[0] == '(0018,0040)':
            framerate = i.split("[")[1].split(" ")[0][:-1]
            frametime = str(1000 / eval(framerate))
            counter = 1
        elif i.split(" ")[0] == '(7fdf,1074)':
            framerate = i.split(" ")[3]
            frametime = str(1000 / eval(framerate))
            counter = 1
    if not counter == 1:
        print("missing framerate")
        framerate = defaultframerate
        frametime = str(1000 / framerate)
    ft = eval(frametime)
    return ft

def output_imgdict(imagefile):
    '''
    converts raw dicom to numpy arrays
    '''
    try:
        ds = imagefile
        if len(ds.pixel_array.shape) == 4: #format 3, nframes, nrow, ncol
            nframes = ds.pixel_array.shape[1]
            maxframes = nframes * 3
        elif len(ds.pixel_array.shape) == 3: #format nframes, nrow, ncol
            nframes = ds.pixel_array.shape[0]
            maxframes = nframes * 1
        #print("nframes", nframes)
        nrow = int(ds.Rows)
        ncol = int(ds.Columns)
        ArrayDicom = np.zeros((nrow, ncol), dtype=ds.pixel_array.dtype)
        imgdict = {}
        for counter in range(0, maxframes, 3):  # this will iterate through all subframes for a loop
            k = counter % nframes
            j = (counter) // nframes
            m = (counter + 1) % nframes
            l = (counter + 1) // nframes
            o = (counter + 2) % nframes
            n = (counter + 2) // nframes
            #print("j", j, "k", k, "l", l, "m", m, "n", n, "o", o)
            if len(ds.pixel_array.shape) == 4:
                a = ds.pixel_array[j, k, :, :]
                b = ds.pixel_array[l, m, :, :]
                c = ds.pixel_array[n, o, :, :]
                d = np.vstack((a, b))
                e = np.vstack((d, c))
                #print(e.shape)
                g = e.reshape(3 * nrow * ncol, 1)
                y = g[::3]
                u = g[1::3]
                v = g[2::3]
                y = y.reshape(nrow, ncol)
                u = u.reshape(nrow, ncol)
                v = v.reshape(nrow, ncol)
                ArrayDicom[:, :] = ybr2gray(y, u, v)
                ArrayDicom[0:int(nrow / 10), 0:int(ncol)] = 0  # blanks out name
                counter = counter + 1
                ArrayDicom.clip(0)
                nrowout = nrow
                ncolout = ncol
                x = int(counter / 3)
                imgdict[x] = imresize(ArrayDicom, (nrowout, ncolout))
            elif len(ds.pixel_array.shape) == 3:
                ArrayDicom[:, :] = ds.pixel_array[counter, :, :]
                ArrayDicom[0:int(nrow / 10), 0:int(ncol)] = 0  # blanks out name
                counter = counter + 1
                ArrayDicom.clip(0)
                nrowout = nrow
                ncolout = ncol
                x = int(counter / 3)
                imgdict[x] = imresize(ArrayDicom, (nrowout, ncolout))
        return imgdict
    except:
        return None


def create_mask(imgs):
    '''
    removes static burned in pixels in image; will use for disease diagnosis
    '''
    from scipy.ndimage.filters import gaussian_filter
    diffs = []
    for i in range(len(imgs) - 1):
        temp = np.abs(imgs[i] - imgs[i + 1])
        temp = gaussian_filter(temp, 10)
        temp[temp <= 50] = 0
        temp[temp > 50] = 1

        diffs.append(temp)

    diff = np.mean(np.array(diffs), axis=0)
    diff[diff >= 0.5] = 1
    diff[diff < 0.5] = 0
    return diff

def ybr2gray(y, u, v):
    r = y + 1.402 * (v - 128)
    g = y - 0.34414 * (u - 128) - 0.71414 * (v - 128)
    b = y + 1.772 * (u - 128)
    # print r, g, b
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
    return np.array(gray, dtype="int8")


def create_imgdict_from_dicom(directory, filename):
    """
    convert compressed DICOM format into numpy array
    """
    targetfile = os.path.join(directory, filename)
    temp_directory = os.path.join(directory, "image")
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)
    ds = dicom.read_file(targetfile, force = True)
    if ("NumberOfFrames" in  dir(ds)) and (ds.NumberOfFrames>1):
        outrawfile = os.path.join(temp_directory, filename + "_raw")
        command = 'gdcmconv -w ' + os.path.join(directory, filename) + " "  + outrawfile
        subprocess.Popen(command, shell=True)
        time.sleep(10)
        if os.path.exists(outrawfile):
            ds = dicom.read_file(outrawfile, force = True)
            imgdict = output_imgdict(ds)
        else:
            print(outrawfile, "missing")
    return imgdict


