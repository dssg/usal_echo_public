# coding: utf-8
import os
import time

import numpy as np
import pandas as pd

import matplotlib as mpl

mpl.use("Agg")

import pickle

APICAL_4_CHAMBER = "a4c"
APICAL_2_CHAMBER = "a2c"


def point_distance(point1, point2):
    point1 = np.array(point1).astype(float)
    point2 = np.array(point2).astype(float)
    return np.sqrt(np.sum((point1 - point2) ** 2))


def L(seg, x, y, x_la, y_la):
    fit = np.polyfit(x, y, 1)
    fit_fn = np.poly1d(fit)
    line_points = fit_fn(x_la)
    x_min = x_la[np.argmin(line_points)]
    y_min = np.min(line_points)
    x_max = x_la[np.argmax(line_points)]
    y_max = np.max(line_points)
    l = point_distance([x_min, y_min], [x_max, y_max])
    return l


def extract_area_l_scaled(
    video,
    directory,
    lv_segs,
    la_segs,
    la_areas,
    lv_areas,
    x_scale,
    y_scale,
    nrow,
    ncol,
    hr,
):
    rows, cols = nrow, ncol
    # left atrium analysis
    la_seg = la_segs[np.argsort(la_areas)[int(0.80 * len(la_segs))]]
    lv_seg = lv_segs[np.argsort(la_areas)[int(0.80 * len(la_segs))]]
    seg = lv_seg
    seg = imresize(seg.copy(), (rows, cols), interp="nearest")
    la_seg = imresize(la_seg.copy(), (rows, cols), interp="nearest")
    x, y = np.where(seg > 0)
    x_la, y_la = np.where(la_seg > 0)
    l_la = L(la_seg, x, y, x_la, y_la)

    # Within window, 90% and 10% of areas as LV end-diastolic/systolic areas.
    lveda_seg = lv_segs[np.argsort(lv_areas)[int(0.90 * len(lv_segs))]]
    lvesa_seg = lv_segs[np.argsort(lv_areas)[int(0.10 * len(lv_segs))]]
    la_seg = la_segs[np.argsort(la_areas)[int(0.90 * len(lv_segs))]]

    # left ventricular diastolic volume analysis
    seg = la_seg.copy()
    seg = imresize(seg.copy(), (rows, cols), interp="nearest")
    lveda_seg = imresize(lveda_seg.copy(), (rows, cols), interp="nearest")
    x, y = np.where(seg > 0)
    x_lveda, y_lveda = np.where(lveda_seg > 0)
    l_lveda = L(lveda_seg, x, y, x_lveda, y_lveda)

    # left ventricular systolic volume analysis
    seg = la_seg.copy()
    seg = imresize(seg.copy(), (rows, cols), interp="nearest")
    lvesa_seg = imresize(lvesa_seg.copy(), (rows, cols), interp="nearest")
    x, y = np.where(seg > 0)
    x_lvesa, y_lvesa = np.where(lvesa_seg > 0)
    l_lvesa = L(lvesa_seg, x, y, x_lvesa, y_lvesa)

    return (
        len(x_la) * x_scale ** 2,
        l_la * x_scale,
        len(x_lveda) * x_scale ** 2,
        l_lveda * x_scale,
        len(x_lvesa) * x_scale ** 2,
        l_lvesa * x_scale,
        hr,
    )


def computevolume_AL(area, length):
    volume = 0.85 * area ** 2 / length
    return volume


def computediastole(lv_areas_window, ft):
    windowlength = len(lv_areas_window)
    minarea = np.min(lv_areas_window[: int(0.6 * windowlength)])
    maxarea = np.max(lv_areas_window)
    minindex = lv_areas_window.index(minarea)
    maxindex = lv_areas_window.index(minarea)
    half = 0.5 * (minarea + maxarea)
    halfarea = [i for i in lv_areas_window[maxindex:] if i > half]
    if not halfarea == []:
        halfareachoice = halfarea[0]
        halfindex = lv_areas_window.index(halfareachoice)
        diasttime = ft * (halfindex - minindex)
        return diasttime
    else:
        return np.nan


def apply_rolling_window(areas):
    areas = (
        pd.DataFrame(areas)[0]
        .rolling(window=4, center=True)
        .median()
        .fillna(method="bfill")
        .fillna(method="ffill")
        .tolist()
    )
    return areas


def extract_areas(segs):
    areas = []
    for seg in segs:
        area = len(np.where(seg > 0)[0])
        areas.append(area)
    return areas


def remove_periphery(imgs):
    """
    
    """
    imgs_ret = []
    for img in imgs:
        image = img.astype("uint8").copy()
        fullsize = image.shape[0] * image.shape[1]
        image[image > 0] = 255
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
            roi_corners = np.array(contours[select], dtype=np.int32)
            for i in roi_corners:
                roi_corners_clean.append(i[0])
            hull = cv2.convexHull(np.array([roi_corners_clean], dtype=np.int32))
            mask = np.zeros(image.shape, dtype=np.uint8)
            mask = cv2.fillConvexPoly(mask, hull, 1)
            imgs_ret.append(img * mask)
    return np.array(imgs_ret)


def compute_la_lv_volume(
    dicomDir, videofile, hr, ft, window, x_scale, y_scale, nrow, ncol, view
):
    npydir = "/home/ubuntu/data/d04_segmentation/" + view
    la_segs = np.load(npydir + "/" + videofile + "_la.npy")
    lv_segs = np.load(npydir + "/" + videofile + "_lv.npy")

    la_segs = remove_periphery(la_segs)
    lv_segs = remove_periphery(lv_segs)

    la_areas = extract_areas(la_segs)
    lv_areas = extract_areas(lv_segs)

    la_areas = apply_rolling_window(la_areas)
    lv_areas = apply_rolling_window(lv_areas)

    efmax, efmin = 0.8, 0.1
    lvesvmax = 600
    lvedvmin, lvedvmax = 20, 600
    lavolmin, lavolmax = 10, 300
    diastmin, diastmax = 100, 400

    eflist = []
    lvesvlist = []
    lvedvlist = []
    lavollist = []
    lveda_l_list = []
    diastlist = []

    # Sliding window with step size of half a cardiac cycle for multiple measurements.
    for i in range(0, len(la_areas), int(window / 2)):
        start = i
        # Window length of 90% of cardiac cycle to avoid end-systole/diastole twice.
        end = np.min((i + int(0.9 * window), len(la_areas)))

        if (end - start) > int(0.8 * window):
            la_segs_window = la_segs[start:end]
            lv_segs_window = lv_segs[start:end]

            la_areas_window = la_areas[start:end]
            lv_areas_window = lv_areas[start:end]

            try:
                la_a, la_l, lveda_a, lveda_l, lvesa_a, lvesa_l, hr = extract_area_l_scaled(
                    dicomDir,
                    videofile,
                    lv_segs_window,
                    la_segs_window,
                    la_areas_window,
                    lv_areas_window,
                    x_scale,
                    y_scale,
                    nrow,
                    ncol,
                    hr,
                )

                lavol = computevolume_AL(la_a, la_l)
                # Derived LVEDV and LVESV using the area-length formula.
                lvedv = computevolume_AL(lveda_a, lveda_l)
                lvesv = computevolume_AL(lvesa_a, lvesa_l)
                # Used LVEDV and LVESV to compute an EF for cycle.
                ef = (lvedv - lvesv) / lvedv

                if ef > efmin and ef < efmax:
                    eflist.append(ef)
                if lvesv < lvesvmax:
                    lvesvlist.append(lvesv)
                if lvedv < lvedvmax and lvedv > lvedvmin:
                    lvedvlist.append(lvedv)
                if lavol < lavolmax and lavol > lavolmin:
                    lavollist.append(lavol)
                lveda_l_list.append(lveda_l)
            except Exception as e:
                print(e, "la, lv calculation")

            diasttime = computediastole(lv_areas_window, ft)
            if diasttime < diastmax and diasttime > diastmin:
                diastlist.append(diasttime)

    # First percentile cutoff, for multiple measurements within one video.
    # TODO: 25% percentile values for LAVOL?
    lavol = np.nan if lavollist == [] else np.nanpercentile(lavollist, 75)
    # TODO: 50% percentile values for EF?
    ef = np.nan if eflist == [] else np.nanpercentile(eflist, 90)
    lvedv = np.nan if lvedvlist == [] else np.nanpercentile(lvedvlist, 90)

    lvesv = np.nan if lvesvlist == [] else np.nanpercentile(lvesvlist, 50)
    lveda_l = np.nan if lveda_l_list == [] else np.nanpercentile(lveda_l_list, 50)
    diasttime = np.nan if diastlist == [] else np.nanpercentile(diastlist, 50)

    return lavol, lvedv, lvesv, ef, diasttime, lveda_l


def get_window(hr, ft):
    """
    Estimate duration of cardiac cycle with heart rate and frame time.
    
    (seconds/beat) / (seconds/frame) = frames/beat
    
    """
    window = int(((60 / hr) / (ft / 1000)))
    return window


def extractmetadata(dicomDir, videofile):
    command = "gdcmdump " + dicomDir + "/" + videofile
    pipe = subprocess.Popen(command, stdout=PIPE, stderr=None, shell=True)
    text = pipe.communicate()[0]
    data = text.split("\n")
    # Note: for *_scale, min([frame.delta for frame in frames if |delta| > 0.012])
    a = computedeltaxy_gdcm(data)
    x_scale, y_scale = (None, None) if a == None else a
    hr = computehr_gdcm(data)
    b = computexy_gdcm(data)
    nrow, ncol = (None, None) if b == None else b
    # Note: returns frame_time (msec/frame) or 1000/cine_rate (frames/sec)
    ft = computeft_gdcm_strain(data)
    if hr < 40:
        print(hr, "problem heart rate")
        hr = 70
    return ft, hr, nrow, ncol, x_scale, y_scale


def get_views_to_indices(model):
    infile = open(f"d03_classification/viewclasses_{model}.txt")
    views = [i.rstrip() for i in infile.readlines()]

    views_to_indices = {}
    for i, view in enumerate(views):
        # Skip two indices for "study" and "image" in probabilities file.
        viewdict[view] = i + 2

    return views_to_indices


def get_viewprob_lists(model, dicomdir_basename):
    viewfile = f"/home/ubuntu/data/03_classification/probabilities/{model}_{dicomdir_basename}_probabilities.txt"
    infile = open(viewfile)
    viewprob_lists = [i.rstrip().split("\t") for i in infile.readlines()]

    return viewprob_lists


def get_viewlists(viewprob_lists, viewdict, probthresh=0.5):
    viewlist_a2c = []
    viewlist_a4c = []

    # Skip header row
    for viewprobs in viewprob_lists[1:]:
        dicomdir = viewprobs[0]
        filename = viewprobs[1]
        if float(viewprobs[viewdict[APICAL_4_CHAMBER]]) > probthresh:
            viewlist_a4c.append(filename)
        elif float(viewprobs[viewdict[APICAL_2_CHAMBER]]) > probthresh:
            viewlist_a2c.append(filename)

    return viewlist_a2c, viewlist_a4c


def main():
    """
    
    We compute chamber dimensions and ejection fraction from segmentations.
    We rely on variation in ventricular area to identify end-systole/diastole.
    We emphasize averaging over many cardiac cycles, within/across video(s).
    We used all videos with the unoccluded chambers of interest.
    We selected two percentiles/metric, from multiple cycles within/across videos.
    We selected first percentile based on how humans choose images, avoiding min/max.
    We selected second percentile to minimize auto/manual difference, default median.
    
    """

    model = "view_23_e5_class_11-Mar-2018"
    # TODO: change from hardcoded value
    dicomdir = "/home/ubuntu/data/01_raw/dcm_sample_labelled"
    dicomdir_basename = os.path.basename(dicomdir)

    views_to_indices = get_views_to_indices(model)
    viewprob_lists = get_viewprob_lists(model, dicomdir_basename)
    viewlist_a2c, viewlist_a4c = get_viewlists(viewprob_lists, views_to_indices)
    print(f"Apical 2 Chamber video files: {viewlist_a2c}")
    print(f"Apical 4 Chamber video files: {viewlist_a4c}")

    measuredict = {}
    for videofile in viewlist_a4c + viewlist_a2c:
        measuredict[videofile] = {}

        ft, hr, nrow, ncol, x_scale, y_scale = extractmetadata(dicomdir, videofile)
        window = get_window(hr, ft)
        view = APICAL_4_CHAMBER if videofile in viewlist_a4c else APICAL_2_CHAMBER

        lavol, lvedv, lvesv, ef, diasttime, lveda_l = compute_la_lv_volume(
            dicomdir, videofile, hr, ft, window, x_scale, y_scale, nrow, ncol, view
        )

        measuredict[videofile]["lavol"] = lavol
        measuredict[videofile]["lvedv"] = lvedv
        measuredict[videofile]["lvesv"] = lvesv
        measuredict[videofile]["ef"] = ef
        measuredict[videofile]["diasttime"] = diasttime
        measuredict[videofile]["lveda_l"] = lveda_l

    print(f"Video files: {list(measuredict.items())}")

    # TODO: second cutoff (across multiple videos in a study)?
    # 50% for LVEDV, 25% for LVESV, 75% for LVEF, 25% for LAVOL
    # Exclude measurements from videos where LAVOL/LVEDV < 30%

    # TODO: write to database
    out = open(
        "/home/ubuntu/data/04_segmentation/"
        + dicomdir_basename
        + "_measurements_dict.txt",
        "w",
    )
    pickle.dump(measuredict, out)
    out.close()


if __name__ == "__main__":
    main()
