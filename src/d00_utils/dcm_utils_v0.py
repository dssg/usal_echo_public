# coding: utf-8
import sys
import os
import pydicom
import time
import numpy as np
import subprocess
from subprocess import Popen, PIPE
from scipy.misc import imresize
import cv2


def computehr_gdcm(data):
    """
    
    """
    hr = "None"
    for i in data:
        i = i.lstrip()
        if i.split(" ")[0] == '(0018,1088)':
            hr = eval(i.split("[")[1].split("]")[0])
            print("heart rate found")
    return hr


def computexy_gdcm(data):
    """
    
    """
    for i in data:
        i = i.lstrip()
        if i.split(" ")[0] == '(0028,0010)':
            rows = i.split(" ")[2]
        elif i.split(" ")[0] == '(0028,0011)':
            cols = i.split(" ")[2]
    return int(rows), int(cols)


def computebsa_gdcm(data):
    """
    dubois, height in m, weight in kg
    :param data: 
    :return: 
    
    """
    for i in data:
        i = i.lstrip()
        if i.split(" ")[0] == '(0010,1020)':
            h = i.split("[")[1].split("]")[0]
        elif i.split(" ")[0] == '(0010,1030)':
            w = i.split("[")[1].split("]")[0]
    return 0.20247 * (eval(h)**0.725) * (eval(w)**0.425)


def computedeltaxy_gdcm(data):
    """
    the unit is the number of cm per pixel 
    
    """
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
    """
    
    """
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
    """
    
    """
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
    """
    
    """
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


def create_mask(imgs):
    """
    removes static burned in pixels in image; will use for disease diagnosis
    
    """
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