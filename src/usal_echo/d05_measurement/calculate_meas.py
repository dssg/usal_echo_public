#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml

from datetime import datetime
from tqdm import tqdm

from usal_echo import usr_dir
from usal_echo.d05_measurement.meas_utils import *
from usal_echo.d00_utils.log_utils import setup_logging
from usal_echo.d00_utils.db_utils import dbReadWriteSegmentation, dbReadWriteMeasurement

logger = setup_logging(__name__, __name__)


def calculate_meas(folder):
    """Write calculated volumes, ejection fractions, and recommendations.
    
    All the functions involved were extracted and adapted from Zhang et al. code.

    We compute chamber dimensions and ejection fraction from segmentations.
    We rely on variation in ventricular area to identify end-systole/diastole.
    We emphasize averaging over many cardiac cycles, within/across video(s).
    We use all videos with the unoccluded chambers of interest.
    We selected two percentiles/measurement, for multiple cycles within/across videos.
    We selected first percentile based on how humans choose images: avoid min/max.
    We selected second percentile to minimize auto/manual difference: default median.

    """
    io_segmentation = dbReadWriteSegmentation()
    io_measurement = dbReadWriteMeasurement()

    # Get files in specified folder.
    with open(os.path.join(usr_dir,"conf","path_parameters.yml")) as f:
        paths = yaml.safe_load(f)
    dicomdir = f"{os.path.expanduser(paths['dcm_dir'])}/{folder}/raw"

    file_names_dcm = [
        file_name.replace("_raw", "") for file_name in os.listdir(dicomdir)
    ]

    # Initialize mapping of filename to measurement dictionary.
    folder_measure_dict = {}

    # Can only read a small number of segmentation rows at a time due to Numpy arrays.
    step = 10
    for start in tqdm(range(0, len(file_names_dcm), step)):
        # Get small number of rows.
        small_file_names_dcm = file_names_dcm[start : start + step]
        small_df = io_segmentation.get_segmentation_rows_for_files(
            "predictions", tuple(small_file_names_dcm)
        )
        for _, row in small_df.iterrows():
            # Get relevant info.
            study_id = row["study_id"]
            instance_id = row["instance_id"]
            file_name = row["file_name"].split(".")[0]

            # Calculate window.
            videofile = f"{file_name}.dcm_raw"
            ft, hr, nrow, ncol, x_scale, y_scale = extract_metadata_for_measurements(
                dicomdir, videofile
            )
            window = get_window(hr, ft)

            # Get back buffers.
            output_np_la = row["output_np_la"]
            output_np_lv = row["output_np_lv"]

            # Read buffers into Numpy.
            output_np_la = np.frombuffer(output_np_la, dtype="uint8")
            output_np_lv = np.frombuffer(output_np_lv, dtype="uint8")

            # Correct Numpy shape.
            output_np_la = np.reshape(output_np_la, (-1, 384, 384))
            output_np_lv = np.reshape(output_np_lv, (-1, 384, 384))

            # Flip segmentations.
            output_np_la = np.flipud(output_np_la)
            output_np_lv = np.flipud(output_np_lv)

            # Get dictionary of measurements.
            la_segs = output_np_la
            lv_segs = output_np_lv
            video_measure_dict = compute_la_lv_volume(
                dicomdir,
                videofile,
                hr,
                ft,
                window,
                x_scale,
                y_scale,
                nrow,
                ncol,
                la_segs,
                lv_segs,
            )

            video_measure_dict["study_id"] = study_id
            video_measure_dict["instance_id"] = instance_id
            video_measure_dict["file_name"] = file_name
            folder_measure_dict[file_name] = video_measure_dict

    # TODO: in future, aggregate measurements across multiple videos in a study?
    # Exclude measurements from videos where LAVOL/LVEDV < 30%, in case occluded
    # Percentiles: 50% for LVEDV, 25% for LVESV, 75% for LVEF, 25% for LAVOL

    # Get measurement names and units for writing to a table.
    # For a new measurement, you would need to specify the name and unit here.
    all_measurement_names = [
        "VTD(MDD-ps4)",
        "VTS(MDD-ps4)",
        "FE(MDD-ps4)",
        "recommendation",
    ]
    all_measurement_units = ["mL", "mL", "%", ""]
    num_meas = len(all_measurement_names)

    # Get relevant info for filenames that are keys in the dictionary.
    file_names = list(folder_measure_dict.keys())

    # Repeat the instance information for each measurement.
    study_ids = np.repeat(
        [folder_measure_dict[file_name]["study_id"] for file_name in file_names],
        num_meas,
    )
    instance_ids = np.repeat(
        [folder_measure_dict[file_name]["instance_id"] for file_name in file_names],
        num_meas,
    )

    # Get list of lists, which will later be flattened.
    measurement_names = [all_measurement_names for file_name in file_names]
    measurement_units = [all_measurement_units for file_name in file_names]

    # Get list of each measurement for all files.
    lvedv_values = [folder_measure_dict[file_name]["lvedv"] for file_name in file_names]
    lvesv_values = [folder_measure_dict[file_name]["lvesv"] for file_name in file_names]
    ef_values = [folder_measure_dict[file_name]["ef"] for file_name in file_names]
    rec_values = [
        "normal"
        if ef >= 60
        else "abnormal"
        if ef < 40
        else np.nan
        if np.isnan(ef)
        else "greyzone"
        for ef in ef_values
    ]

    # Get one list of all measurements for all files.
    measurement_values = [
        list(l) for l in zip(lvedv_values, lvesv_values, ef_values, rec_values)
    ]

    # Produce final dataframe to write to table, flattening measurement info.
    date_run = datetime.now()
    calculations_df = pd.DataFrame.from_dict(
        {
            "study_id": study_ids,
            "instance_id": instance_ids,
            "file_name": np.repeat(file_names, num_meas),
            "date_run": date_run,
            "measurement_name": pd.Series(measurement_names).explode(),
            "measurement_unit": pd.Series(measurement_units).explode(),
            "measurement_value": pd.Series(measurement_values).explode(),
        }
    )

    # Write calculations to schema.
    # Add serial id.
    old_calculations_df = io_measurement.get_table("calculations")
    start = len(old_calculations_df)
    calculation_id = pd.Series(start + calculations_df.index)
    calculations_df.insert(0, "calculation_id", calculation_id)
    all_calculations_df = old_calculations_df.append(calculations_df)
    io_measurement.save_to_db(all_calculations_df, "calculations")
    logger.info("Successfully calculated measurements")
