import os
import json
import numpy as np
import pandas as pd

from subprocess import Popen, PIPE


def extract_metadata_for_measurements(dicomdir, videofile):
    """Get DICOM metadata using GDCM utility."""
    command = "gdcmdump " + dicomdir + "/" + videofile
    pipe = Popen(command, stdout=PIPE, shell=True, universal_newlines=True)
    text = pipe.communicate()[0]
    lines = text.split("\n")
    dicom_tags = json.load(open("/home/ubuntu/src/d02_intermediate/dicom_tags.json"))
    # Convert ["<tag1>", "<tag2>"] format to "(<tag1>, <tag2>)" GDCM output format.
    dicom_tags = {
        k: str(tuple(v)).replace("'", "").replace(" ", "")
        for k, v in dicom_tags.items()
    }
    # Note: *_scale = min([|frame.delta| for frame in frames if |frame.delta| > 0.012])
    x_scale, y_scale = _extract_delta_xy_from_gdcm_str(lines, dicom_tags) or (
        None,
        None,
    )
    hr = _extract_hr_from_gdcm_str(lines, dicom_tags)
    nrow, ncol = _extract_xy_from_gdcm_str(lines, dicom_tags) or (None, None)
    # Note: returns frame_time (msec/frame) or 1000/cine_rate (frames/sec)
    ft = _extract_ft_from_gdcm_str(lines, dicom_tags)
    if hr < 40:
        logger.debug(f"problem heart rate: {hr}")
        hr = 70
    return ft, hr, nrow, ncol, x_scale, y_scale


def _extract_delta_xy_from_gdcm_str(lines, dicom_tags):
    """Get x_scale, y_scale from gdcmdump output."""
    xlist = []
    ylist = []
    for line in lines:
        line = line.lstrip()
        tag = line.split(" ")[0]
        if tag == dicom_tags["physical_delta_x_direction"]:
            deltax = line.split(" ")[2]
            deltax = np.abs(float(deltax))
            if deltax > 0.012:
                xlist.append(deltax)
        if tag == dicom_tags["physical_delta_y_direction"]:
            deltay = line.split(" ")[2]
            deltay = np.abs(float(deltay))
            if deltay > 0.012:
                ylist.append(deltay)
    return np.min(xlist), np.min(ylist)


def _extract_hr_from_gdcm_str(lines, dicom_tags):
    """Get heart rate from gdcmdump output."""
    hr = "None"
    for line in lines:
        line = line.lstrip()
        tag = line.split(" ")[0]
        if tag == dicom_tags["heart_rate"]:
            hr = int(line.split("[")[1].split("]")[0])
    return hr


def _extract_xy_from_gdcm_str(lines, dicom_tags):
    """Get rows, columns from gdcmdump output."""
    for line in lines:
        line = line.lstrip()
        tag = line.split(" ")[0]
        if tag == dicom_tags["rows"]:
            rows = line.split(" ")[2]
        elif tag == dicom_tags["columns"]:
            cols = line.split(" ")[2]
    return int(rows), int(cols)


def _extract_ft_from_gdcm_str(lines, dicom_tags):
    """Get frame time from gdcmdump output."""
    default_framerate = 30
    is_framerate = False
    for line in lines:
        tag = line.split(" ")[0]
        if tag == dicom_tags["frame_time"]:
            frametime = line.split("[")[1].split("]")[0]
            is_framerate = True
        elif tag == dicom_tags["cine_rate"]:
            framerate = line.split("[")[1].split("]")[0]
            frametime = 1000 / float(framerate)
            is_framerate = True
    if not is_framerate:
        logger.debug("missing framerate")
        framerate = defaultframerate
        frametime = 1000 / framerate
    ft = float(frametime)
    return ft


def get_window(hr, ft):
    """Estimate duration of cardiac cycle with heart rate and frame time.

    (seconds/beat) / (seconds/frame) = frames/beat

    """
    window = int(((60 / hr) / (ft / 1000)))
    return window


def compute_la_lv_volume(
    dicomDir, videofile, hr, ft, window, x_scale, y_scale, nrow, ncol, view
):
    """Return measurement dictionary for video."""
    npydir = f"{os.path.expanduser('~')}/data/04_segmentation/results/{view}"
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
        "diasttime": diasttime,
    }


def remove_periphery(imgs):
    """Clean segmentations (adapted from Zhang et al code)."""
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
    """Get area for segmentation (adapted from Zhang et al code)."""
    areas = []
    for seg in segs:
        area = len(np.where(seg > 0)[0])
        areas.append(area)
    return areas


def apply_rolling_window(areas):
    """Clean areas (adapted from Zhang et al code)."""
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
    """Get areas/lengths scaled by metadata (adapted from Zhang et al.)"""
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
    """Get Euclidean distance from points fit by function (adapted from Zhang et al.)"""
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
    """Get Euclidean distance between two points (adapted from Zhang et al.)"""
    point1 = np.array(point1).astype(float)
    point2 = np.array(point2).astype(float)
    return np.sqrt(np.sum((point1 - point2) ** 2))


def compute_volume_AL(area, length):
    """Calculate volume with area-length formula (adapted from Zhang et al.)"""
    volume = 0.85 * area ** 2 / length
    return volume


def compute_diastole(lv_areas_window, ft):
    """Compute diastole time (adapted from Zhang et al.)"""
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
