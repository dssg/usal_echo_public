# coding: utf-8
import os

from datetime import datetime
from tqdm import tqdm

from d00_utils.output_utils import *
from d05_measurement.meas_utils import *
from d00_utils.log_utils import *
from d00_utils.db_utils import dbReadWriteSegmentation, dbReadWriteMeasurement

logger = setup_logging(__name__, "analyse_segments")


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
    
    rootdir = os.path.expanduser('~')
    dicomdir = f"{rootdir}/data/01_raw/{folder}/raw"
    # TODO: change to `file_names` when they are written without suffix in segmentation
    file_names_dcm = [file_name.replace('_raw', '') for file_name in os.listdir(dicomdir)]
    
    # Can only read a small number of segmentation rows at a time due to Numpy arrays.
    folder_measure_dict = {}
    step = 10
    for i in tqdm(range(0, len(file_names_dcm), step)):
        small_file_names_dcm = file_names_dcm[i*10:(i+1)*10]
        small_df = io_segmentation.get_segmentation_rows_for_files('predictions', tuple(small_file_names_dcm))
        for _, row in small_df.iterrows():
            study_id = row['study_id']
            instance_id = row['instance_id']

            # TODO: change when they are written without suffix in segmentation
            file_name = row['file_name'].split('.')[0]

            videofile = f'{file_name}.dcm_raw'
            ft, hr, nrow, ncol, x_scale, y_scale = extract_metadata_for_measurements(
                dicomdir, videofile
            )
            window = get_window(hr, ft)

            # Get back buffers, flipped.
            output_np_la = row['output_np_la']
            output_np_lv = row['output_np_lv']

            # Read buffers into Numpy.
            output_np_la = np.frombuffer(output_np_la, dtype='uint8')
            output_np_lv = np.frombuffer(output_np_lv, dtype='uint8')

            # Correct Numpy shape.
            output_np_la = np.reshape(output_np_la, (-1, 384, 384))
            output_np_lv = np.reshape(output_np_lv, (-1, 384, 384))

            # TODO: delete when written as flipped in segmentation
            output_np_la = np.flipud(output_np_la)
            output_np_lv = np.flipud(output_np_lv)

            la_segs = output_np_la
            lv_segs = output_np_lv
            video_measure_dict = compute_la_lv_volume(
                dicomdir, videofile, hr, ft, window, x_scale, y_scale, nrow, ncol, la_segs, lv_segs
            )

            video_measure_dict['study_id'] = study_id
            video_measure_dict['instance_id'] = instance_id
            video_measure_dict['file_name'] = file_name
            folder_measure_dict[file_name] = video_measure_dict
        break

    # TODO: in future, aggregate measurements across multiple videos in a study?
    # Exclude measurements from videos where LAVOL/LVEDV < 30%, in case occluded
    # Percentiles: 50% for LVEDV, 25% for LVESV, 75% for LVEF, 25% for LAVOL

    # Write to database.
    file_names = list(folder_measure_dict.keys())
    study_ids = np.repeat([folder_measure_dict[file_name]['study_id'] for file_name in file_names], 4)
    instance_ids = np.repeat([folder_measure_dict[file_name]['instance_id'] for file_name in file_names], 4)
    all_measurement_names = ["VTD(MDD-ps4)", "VTS(MDD-ps4)", "FE(MDD-ps4)", "recommendation"]
    all_measurement_units = ["mL", "mL", "%", ""]
    measurement_names = [all_measurement_names for file_name in file_names]
    measurement_units = [all_measurement_units for file_name in file_names]
    lvedv_values = [folder_measure_dict[file_name]['lvedv'] for file_name in file_names]
    lvesv_values = [folder_measure_dict[file_name]['lvesv'] for file_name in file_names]
    ef_values = [folder_measure_dict[file_name]['ef'] for file_name in file_names]
    rec_values = ['normal' if ef >= 60 else 'abnormal' if ef < 40 else np.nan if np.isnan(ef) else "greyzone" for ef in ef_values]
    measurement_values = [list(l) for l in zip(lvedv_values, lvesv_values, ef_values, rec_values)]
    
    date_run = datetime.now()
    calculation_df = pd.DataFrame.from_dict({'study_id': study_ids, 'instance_id': instance_ids, 'file_name': np.repeat(file_names, 4), 'date_run': date_run, 'measurement_name': pd.Series(measurement_names).explode(), 'measurement_unit': pd.Series(measurement_units).explode(), 'measurement_value': pd.Series(measurement_values).explode()})

    return calculation_df


if __name__ == "__main__":
    calculate_measurements()
