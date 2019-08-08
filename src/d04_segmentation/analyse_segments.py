# coding: utf-8
import os
import time
import pickle
import numpy as np
import pandas as pd

from d00_utils.dcm_utils import *
from d00_utils.output_utils import *
from d00_utils.log_utils import *

logger = setup_logging(__name__, "analyse_segments")


def get_window(hr, ft):
    """
    Estimate duration of cardiac cycle with heart rate and frame time.

    (seconds/beat) / (seconds/frame) = frames/beat

    """
    window = int(((60 / hr) / (ft / 1000)))
    return window


def compute_la_lv_volume(
    dicomDir, videofile, hr, ft, window, x_scale, y_scale, nrow, ncol, view
):
    npydir = "~/data/04_segmentation/results/" + view
    la_segs = np.load(npydir + "/" + videofile + "_la.npy")
    lv_segs = np.load(npydir + "/" + videofile + "_lv.npy")

    la_segs = remove_periphery(la_segs)
    lv_segs = remove_periphery(lv_segs)

    la_areas = extract_areas(la_segs)
    lv_areas = extract_areas(lv_segs)

    la_areas = apply_rolling_window(la_areas)
    lv_areas = apply_rolling_window(lv_areas)

    lavolmin, lavolmax = 10, 300
    lvedvmin, lvedvmax = 20, 600
    lvesvmax = 600
    efmax, efmin = 0.8, 0.1
    diastmin, diastmax = 100, 400

    lavollist = []
    lvedvlist = []
    lvesvlist = []
    eflist = []
    lveda_l_list = []
    diastlist = []

    # Sliding window with step size of half a cardiac cycle for multiple measurements.
    for start in range(0, len(la_areas), int(window / 2)):
        # Window length of 90% of cardiac cycle to avoid end-systole/diastole twice.
        end = np.min((start + int(0.9 * window), len(la_areas)))

        # Why 0.8?
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

                lavol = compute_volume_AL(la_a, la_l)
                # Derived LVEDV and LVESV using the area-length formula.
                lvedv = compute_volume_AL(lveda_a, lveda_l)
                lvesv = compute_volume_AL(lvesa_a, lvesa_l)
                # Used LVEDV and LVESV to compute EF for cycle.
                ef = (lvedv - lvesv) / lvedv

                if lavol < lavolmax and lavol > lavolmin:
                    lavollist.append(lavol)
                if lvedv < lvedvmax and lvedv > lvedvmin:
                    lvedvlist.append(lvedv)
                if lvesv < lvesvmax:
                    lvesvlist.append(lvesv)
                if ef > efmin and ef < efmax:
                    eflist.append(ef)
                lveda_l_list.append(lveda_l)
            except Exception as e:
                logger.error(e, "la, lv calculation")

            diasttime = compute_diastole(lv_areas_window, ft)
            if diasttime < diastmax and diasttime > diastmin:
                diastlist.append(diasttime)

    # First percentile cutoff, for multiple measurements within one video.
    # Supplementary materials says 25% percentile values for LAVOL?
    lavol = np.nan if lavollist == [] else np.nanpercentile(lavollist, 75)
    lvedv = np.nan if lvedvlist == [] else np.nanpercentile(lvedvlist, 90)
    lvesv = np.nan if lvesvlist == [] else np.nanpercentile(lvesvlist, 50)
    # Supplementary materials says 50% percentile values for EF?
    ef = np.nan if eflist == [] else np.nanpercentile(eflist, 90)
    lveda_l = np.nan if lveda_l_list == [] else np.nanpercentile(lveda_l_list, 50)
    diasttime = np.nan if diastlist == [] else np.nanpercentile(diastlist, 50)

    return {
        "lavol": lavol,
        "lvedv": lvedv,
        "lvesv": lvesv,
        "ef": ef,
        "lveda_l": lveda_l,
        "diasttime": diasttime
    }


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


def extract_areas(segs):
    areas = []
    for seg in segs:
        area = len(np.where(seg > 0)[0])
        areas.append(area)
    return areas


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


def extract_area_l_scaled(
    video,
    directory,
    lv_segs,
    la_segs,
    la_areas,
    lv_areas,
    x_scale,
    y_scale,
    rows,
    cols,
    hr,
):
    # Left atrium analysis.
    # Why 0.80?
    la_seg = la_segs[np.argsort(la_areas)[int(0.80 * len(la_segs))]]
    lv_seg = lv_segs[np.argsort(la_areas)[int(0.80 * len(la_segs))]]
    seg = lv_seg
    seg = imresize(seg.copy(), (rows, cols), interp="nearest")
    la_seg = imresize(la_seg.copy(), (rows, cols), interp="nearest")
    x, y = np.where(seg > 0)
    x_la, _ = np.where(la_seg > 0)
    l_la = L(x, y, x_la)

    # Within window, 90% and 10% of areas as LV end-diastolic/systolic areas.
    lveda_seg = lv_segs[np.argsort(lv_areas)[int(0.90 * len(lv_segs))]]
    lvesa_seg = lv_segs[np.argsort(lv_areas)[int(0.10 * len(lv_segs))]]
    la_seg = la_segs[np.argsort(la_areas)[int(0.90 * len(lv_segs))]]

    # Left ventricular diastolic volume analysis.
    seg = la_seg.copy()
    seg = imresize(seg.copy(), (rows, cols), interp="nearest")
    lveda_seg = imresize(lveda_seg.copy(), (rows, cols), interp="nearest")
    x, y = np.where(seg > 0)
    x_lveda, _ = np.where(lveda_seg > 0)
    l_lveda = L(x, y, x_lveda)

    # Left ventricular systolic volume analysis.
    seg = la_seg.copy()
    seg = imresize(seg.copy(), (rows, cols), interp="nearest")
    lvesa_seg = imresize(lvesa_seg.copy(), (rows, cols), interp="nearest")
    x, y = np.where(seg > 0)
    x_lvesa, _ = np.where(lvesa_seg > 0)
    l_lvesa = L(x, y, x_lvesa)

    return (
        len(x_la) * x_scale ** 2,
        l_la * x_scale,
        len(x_lveda) * x_scale ** 2,
        l_lveda * x_scale,
        len(x_lvesa) * x_scale ** 2,
        l_lvesa * x_scale,
        hr,
    )


def L(x, y, x_la):
    fit = np.polyfit(x, y, 1)
    fit_fn = np.poly1d(fit)
    line_points = fit_fn(x_la)
    x_min = x_la[np.argmin(line_points)]
    y_min = np.min(line_points)
    x_max = x_la[np.argmax(line_points)]
    y_max = np.max(line_points)
    l = point_distance([x_min, y_min], [x_max, y_max])
    return l


def point_distance(point1, point2):
    point1 = np.array(point1).astype(float)
    point2 = np.array(point2).astype(float)
    return np.sqrt(np.sum((point1 - point2) ** 2))


def compute_volume_AL(area, length):
    volume = 0.85 * area ** 2 / length
    return volume


def compute_diastole(lv_areas_window, ft):
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


def calculate_measurements(folder="dcm_sample_labelled"):
    """
    Write pickle of dictionary with calculated measurements.
    
    All the functions involved were extracted and adapted from Zhang et al. code.

    We compute chamber dimensions and ejection fraction from segmentations.
    We rely on variation in ventricular area to identify end-systole/diastole.
    We emphasize averaging over many cardiac cycles, within/across video(s).
    We use all videos with the unoccluded chambers of interest.
    We selected two percentiles/measurement, for multiple cycles within/across videos.
    We selected first percentile based on how humans choose images: avoid min/max.
    We selected second percentile to minimize auto/manual difference: default median.

    """

    model = "view_23_e5_class_11-Mar-2018"
    dicomdir = f"~/data/01_raw/{folder}"
    dicomdir_basename = os.path.basename(dicomdir)

    views_to_indices = get_views_to_indices(model)
    viewprob_lists = get_viewprob_lists(model, dicomdir_basename)
    viewlist_a2c, viewlist_a4c = get_viewlists(viewprob_lists, views_to_indices)
    logger.info(f"Apical 2 Chamber video files: {viewlist_a2c}")
    logger.info(f"Apical 4 Chamber video files: {viewlist_a4c}")

    study_measure_dict = {}
    for videofile in viewlist_a4c + viewlist_a2c:
        study_measure_dict[videofile] = {}

        ft, hr, nrow, ncol, x_scale, y_scale = extract_metadata_for_measurements(
            dicomdir, videofile
        )
        window = get_window(hr, ft)
        view = "a4c" if videofile in viewlist_a4c else "a2c"

        video_measure_dict = compute_la_lv_volume(
            dicomdir, videofile, hr, ft, window, x_scale, y_scale, nrow, ncol, view
        )

        study_measure_dict[videofile] = video_measure_dict

    logger.info(f"Results: {study_measure_dict}")

    # TODO: in future, aggregate measurements across multiple videos in a study
    # Exclude measurements from videos where LAVOL/LVEDV < 30%, in case occluded
    # Percentiles: 50% for LVEDV, 25% for LVESV, 75% for LVEF, 25% for LAVOL

    # TODO: write to database
    out = open(
        "~/data/04_segmentation/"
        + dicomdir_basename
        + "_measurements_dict.pickle",
        "wb",
    )
    pickle.dump(study_measure_dict, out)
    out.close()


if __name__ == "__main__":
    calculate_measurements()
