# coding: utf-8
import time
from optparse import OptionParser

import matplotlib as mpl

mpl.use("Agg")

import pickle
import pandas as pd
import scipy.fftpack as fft

from d00_utils.dcm_utils_v0 import *


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


def smooth_fft(displist, cutoff):
    x = np.arange(len(displist))
    N = len(displist)
    y = np.array(displist)

    w = fft.rfft(y)
    f = fft.rfftfreq(N, x[1] - x[0])
    spectrum = w ** 2

    cutoff_idx = spectrum < (spectrum.max() / cutoff)
    w2 = w.copy()
    w2[cutoff_idx] = 0

    y2 = fft.irfft(w2)
    return x, y2


def extract_areas(segs):
    areas = []
    for seg in segs:
        area = len(np.where(seg > 0)[0])
        areas.append(area)
    return areas


def extract_area_psax(
    video, study, outer_segs, inner_segs, psax_areas, x_scale, y_scale, nrow, ncol
):
    x_area_scale, y_area_scale = x_scale, y_scale
    area_rows, area_cols = nrow, ncol
    outer_seg = outer_segs[np.argsort(psax_areas)[int(0.9 * len(outer_segs))]]
    inner_seg = inner_segs[np.argsort(psax_areas)[int(0.9 * len(inner_segs))]]
    outer_seg = imresize(outer_seg.copy(), (area_rows, area_cols), interp="nearest")
    inner_seg = imresize(inner_seg.copy(), (area_rows, area_cols), interp="nearest")
    x_outer, y_outer = np.where(outer_seg > 0)
    x_inner, y_inner = np.where(inner_seg > 0)
    return len(x_outer) * x_area_scale ** 2, len(x_inner) * x_area_scale ** 2


def compute_lvmi(
    dicomDir, videofile, lvlength, hr, ft, window, x_scale, y_scale, nrow, ncol
):
    l = lvlength
    npydir = "./segment/psax/"
    psax_outer_segs = np.load(npydir + "/" + videofile + "_lvo.npy")
    psax_outer_segs = remove_periphery(psax_outer_segs)
    psax_inner_segs = np.load(npydir + "/" + videofile + "_lv.npy")
    psax_inner_segs = remove_periphery(psax_inner_segs)
    psax_outer_areas = extract_areas(psax_outer_segs)
    psax_inner_areas = extract_areas(psax_inner_segs)
    outer_area = (
        pd.DataFrame(psax_outer_areas)[0]
        .rolling(window=4, center=True)
        .median()
        .fillna(method="bfill")
        .fillna(method="ffill")
        .tolist()
    )
    inner_area = (
        pd.DataFrame(psax_inner_areas)[0]
        .rolling(window=4, center=True)
        .median()
        .fillna(method="bfill")
        .fillna(method="ffill")
        .tolist()
    )
    massmin = 25
    massmax = 500
    masslist = []
    for i in range(0, len(psax_outer_areas), int(window / 2)):
        start = i
        end = np.min((i + int(0.9 * window), len(psax_outer_areas)))
        if (end - start) > int(0.8 * window):
            psax_outer_segs_window = psax_outer_segs[start:end]
            psax_inner_segs_window = psax_inner_segs[start:end]
            inner_area_window = inner_area[start:end]
            outer_area_final, inner_area_final = extract_area_psax(
                videofile,
                dicomDir,
                psax_outer_segs_window,
                psax_inner_segs_window,
                inner_area_window,
                x_scale,
                y_scale,
                nrow,
                ncol,
            )
            t = np.sqrt((outer_area_final + inner_area_final) / np.pi) - np.sqrt(
                (inner_area_final) / np.pi
            )
            mass = 1.05 * (
                (5 / 6) * (outer_area_final + inner_area_final) * (l + t)
                - (5 / 6) * (inner_area_final) * (l)
            )
            masslist.append(mass)
    if not masslist == []:
        massfinal = np.nanpercentile(masslist, 50)
    else:
        massfinal = np.nan
    return massfinal


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
    # left ventricular diastolic volume analysis
    lveda_seg = lv_segs[np.argsort(lv_areas)[int(0.90 * len(lv_segs))]]
    lvesa_seg = lv_segs[np.argsort(lv_areas)[int(0.10 * len(lv_segs))]]
    la_seg = la_segs[np.argsort(la_areas)[int(0.90 * len(lv_segs))]]
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


def compute_la_lv_volume(
    dicomDir, videofile, hr, ft, window, x_scale, y_scale, nrow, ncol, view
):
    npydir = "./segment/" + view
    la_segs = np.load(npydir + "/" + videofile + "_la.npy")
    la_segs = remove_periphery(la_segs)
    lv_segs = np.load(npydir + "/" + videofile + "_lv.npy")
    lv_segs = remove_periphery(lv_segs)
    lv_areas = extract_areas(lv_segs)
    lv_areas = (
        pd.DataFrame(lv_areas)[0]
        .rolling(window=4, center=True)
        .median()
        .fillna(method="bfill")
        .fillna(method="ffill")
        .tolist()
    )
    la_areas = extract_areas(la_segs)
    la_areas = (
        pd.DataFrame(la_areas)[0]
        .rolling(window=4, center=True)
        .median()
        .fillna(method="bfill")
        .fillna(method="ffill")
        .tolist()
    )
    eflist = []
    efmax, efmin = 0.8, 0.1
    lvedvmin, lvedvmax = 20, 600
    lvesvmax = 600
    lavolmin, lavolmax = 10, 300
    diastmin, diastmax = 100, 400
    lvesvlist = []
    lvedvlist = []
    lavollist = []
    diastlist = []
    lveda_l_list = []
    for i in range(0, len(la_areas), int(window / 2)):
        start = i
        end = np.min((i + int(0.9 * window), len(la_areas)))
        if (end - start) > int(0.8 * window):
            lv_segs_window = lv_segs[start:end]
            la_segs_window = la_segs[start:end]
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
                lvedv = computevolume_AL(lveda_a, lveda_l)
                lvesv = computevolume_AL(lvesa_a, lvesa_l)
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
    if not lavollist == []:
        lavol = np.nanpercentile(lavollist, 75)
    else:
        lavol = np.nan
    if not eflist == []:
        ef = np.nanpercentile(eflist, 90)
    else:
        ef = np.nan
    if not lvedvlist == []:
        lvedv = np.nanpercentile(lvedvlist, 90)
    else:
        lvedv = np.nan
    if not lvesvlist == []:
        lvesv = np.nanpercentile(lvesvlist, 50)
    else:
        lvesv = np.nan
    if not lveda_l_list == []:
        lveda_l = np.nanpercentile(lveda_l_list, 50)
    else:
        lveda_l = np.nan
    if not diastlist == []:
        diasttime = np.nanpercentile(diastlist, 50)
    else:
        diasttime = np.nan
    return lavol, lvedv, lvesv, ef, diasttime, lveda_l


def extractmetadata(dicomDir, videofile):
    command = "gdcmdump " + dicomDir + "/" + videofile
    pipe = subprocess.Popen(command, stdout=PIPE, stderr=None, shell=True)
    text = pipe.communicate()[0]
    data = text.split("\n")
    a = computedeltaxy_gdcm(data)
    if not a == None:
        x_scale, y_scale = a
    else:
        x_scale, y_scale = None, None
    hr = computehr_gdcm(data)
    b = computexy_gdcm(data)
    if not b == None:
        nrow, ncol = b
    else:
        nrow, ncol = None, None
    ft = computeft_gdcm_strain(data)
    if hr < 40:
        print(hr, "problem heart rate")
        hr = 70
    return ft, hr, nrow, ncol, x_scale, y_scale


def main():
    viewfile = "view_23_e5_class_11-Mar-2018_dicomsample_probabilities.txt"
    dicomdir = "dicomsample"
    viewlist_a2c = []
    viewlist_a4c = []
    viewlist_psax = []

    infile = open("viewclasses_view_23_e5_class_11-Mar-2018.txt")
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

    for i in infile[1:]:
        dicomdir = i[0]
        filename = i[1]
        if eval(i[viewdict["psax_pap"]]) > probthresh:
            viewlist_psax.append(filename)
        elif eval(i[viewdict["a4c"]]) > probthresh:
            viewlist_a4c.append(filename)
        elif eval(i[viewdict["a2c"]]) > probthresh:
            viewlist_a2c.append(filename)
    print(viewlist_a2c, viewlist_a4c, viewlist_psax)
    measuredict = {}
    lvlengthlist = []
    for videofile in viewlist_a4c + viewlist_a2c:
        if videofile in viewlist_a4c:
            view = "a4c"
        elif videofile in viewlist_a2c:
            view = "a2c"
        measuredict[videofile] = {}
        ft, hr, nrow, ncol, x_scale, y_scale = extractmetadata(dicomdir, videofile)
        window = int(((60 / hr) / (ft / 1000)))
        lavol, lvedv, lvesv, ef, diasttime, lveda_l = compute_la_lv_volume(
            dicomdir, videofile, hr, ft, window, x_scale, y_scale, nrow, ncol, view
        )
        lvlengthlist.append(lveda_l)
        measuredict[videofile]["lavol"] = lavol
        measuredict[videofile]["lvedv"] = lvedv
        measuredict[videofile]["lvesv"] = lvesv
        measuredict[videofile]["ef"] = ef
        measuredict[videofile]["diasttime"] = diasttime
        measuredict[videofile]["lveda_l"] = lveda_l
    lvlength = np.median(lvlengthlist)
    for videofile in viewlist_psax:
        measuredict[videofile] = {}
        ft, hr, nrow, ncol, x_scale, y_scale = extractmetadata(dicomdir, videofile)
        window = int(((60 / hr) / (ft / 1000)))
        lvmass = compute_lvmi(
            dicomdir, videofile, lvlength, hr, ft, window, x_scale, y_scale, nrow, ncol
        )
        measuredict[videofile]["lvmass"] = lvmass
    print(list(measuredict.items()))
    out = open(dicomdir + "_measurements_dict.txt", "w")
    pickle.dump(measuredict, out)
    out.close()


if __name__ == "__main__":
    main()
