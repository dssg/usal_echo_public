# coding: utf-8
import os
import pickle

from d00_utils.output_utils import *
from d04_segmentation.meas_utils import *
from d00_utils.log_utils import *

logger = setup_logging(__name__, "analyse_segments")


def calculate_measurements(folder="dcm_sample_labelled"):
    """Write pickle of dictionary with calculated measurements.
    
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
    dicomdir = f"{os.path.expanduser('~')}/data/01_raw/{folder}"
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

    # TODO: in future, aggregate measurements across multiple videos in a study?
    # Exclude measurements from videos where LAVOL/LVEDV < 30%, in case occluded
    # Percentiles: 50% for LVEDV, 25% for LVESV, 75% for LVEF, 25% for LAVOL

    # TODO: write to database
    out = open(
        f"{os.path.expanduser('~')}/data/04_segmentation/{dicomdir_basename}_measurements_dict.pickle",
        "wb",
    )
    pickle.dump(study_measure_dict, out)
    out.close()


if __name__ == "__main__":
    calculate_measurements()
