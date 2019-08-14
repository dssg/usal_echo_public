# coding: utf-8
import os
from datetime import datetime

from d00_utils.output_utils import *
from d05_measurement.meas_utils import *
from d00_utils.log_utils import *
from d00_utils.db_utils import dbReadWriteSegmentation, dbReadWriteMeasurement

logger = setup_logging(__name__, "analyse_segments")


def calculate_measurements(folder="test_downsampleby5", calculation_source="predictions"):
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
    
    rootdir = os.path.expanduser('~')
    dicomdir = f"{rootdir}/data/01_raw/{folder}"
    file_names = [fn.split(".")[0] for fn in os.listdir(dicomdir)]

    io_segmentation = dbReadWriteSegmentation()
    segmentation_df = io_segmentation.get_table(calculation_source)
    filtered_segmentation_df = segmentation_df[segmentation_df['file_name'].isin(file_names)]
    
    # TODO: if calculation_source="ground_truths", group df and get output_np_la and output_np_lv

    folder_measure_dict = {}
    for _, row in filtered_segmentation_df.iterrows():
        study_id = row['study_id']
        instance_id = row['instance_id']
        file_name = row['file_name']
        
        videofile = f"a_{study_id}_{file_name}.dcm"
        ft, hr, nrow, ncol, x_scale, y_scale = extract_metadata_for_measurements(
            dicomdir, videofile
        )
        window = get_window(hr, ft)

        la_segs = row['output_np_la']
        lv_segs = row['output_np_lv']
        video_measure_dict = compute_la_lv_volume(
            dicomdir, videofile, hr, ft, window, x_scale, y_scale, nrow, ncol, la_segs, lv_segs
        )

        video_measure_dict['study_id'] = study_id
        video_measure_dict['instance_id'] = instance_id
        video_measure_dict['file_name'] = file_name
        folder_measure_dict[file_name] = video_measure_dict

    # TODO: in future, aggregate measurements across multiple videos in a study?
    # Exclude measurements from videos where LAVOL/LVEDV < 30%, in case occluded
    # Percentiles: 50% for LVEDV, 25% for LVESV, 75% for LVEF, 25% for LAVOL

    # Write to database.
    study_ids = []
    instance_ids = []
    file_names = []
    dates_run = []
    measurement_names = []
    measurement_units = []
    measurement_values = []
    all_measurements = ["lvedv", "lvesv", "ef", "recommendation"]
    all_measurement_names = ["VTD(MDD-ps4)", "VTS(MDD-ps4)", "FE(MDD-ps4)", "recommendation"]
    all_measurement_units = ["mL", "mL", "%", ""]
    
    all_measurement_names = []
    for file_name, video_measure_dict in folder_measure_dict.items():
        study_id = video_measure_dict['study_id']
        instance_id = video_measure_dict['instance_id']
        ef = video_measure_dict['ef']
        
        for i, measurement in enumerate(all_measurements):
            study_ids.append(study_id)
            instance_ids.append(instance_id)
            file_names.append(file_name)
            
            measuremnt_name = all_measurement_names[i]
            measurement_unit = all_measurement_units[i]
            if measurement_name == 'recommendation':
                measurement_value = 'normal' if ef >= 60 else 'abnormal' if ef < 40 else "greyzone"
            else:
                measurement_value = video_measure_dict[measurement]
            
            measurement_names.append(measurement_name)
            measurement_units.append(measurement_unit)
            measurement_values.append(measurement_value)
    
    calculation_df = pd.DataFrame.from_dict(
        {'study_id': study_ids, 'instance_id': instance_ids, 'file_name': file_names, 'date_run': datetime.now(), 'measurement_name': measurement_names, 'measurement_unit': measurement_units, 'measurement_value': measurement_values, 'calculation_source': calculation_source})
    # TODO: write to 
    return calculation_df


if __name__ == "__main__":
    calculate_measurements()
