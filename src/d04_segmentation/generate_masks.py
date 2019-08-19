from time import time

import os
import numpy as np
import pandas as pd

from shapely.geometry import Polygon
from skimage.draw import polygon
from scipy.misc import imresize

from d00_utils.log_utils import setup_logging
from d00_utils.db_utils import dbReadWriteViews, dbReadWriteSegmentation
from d05_measurement.meas_utils import extract_metadata_for_measurements


logger = setup_logging(__name__, __name__)

def generate_masks(dcm_path):
    #io_views = dbReadWriteViews()
    io_segmentation = dbReadWriteSegmentation()
    
    #Instances to write masks for
    #instances_w_labels_test_downsampleby5_df = io_views.get_table('instances_w_labels_test_downsampleby5')  

    masks_df = create_masks(dcm_path)
    
    #ground_truth_id	study_id	instance_id	file_name	frame	chamber	view_name	numpy_array
    
    gt_table_column_names = ['study_id', 'instance_id', 'file_name', 
                    'frame', 'chamber', 'view_name', 'numpy_array']

    for index, mask in masks_df.iterrows():
        resized_mask = (imresize(mask['mask'], (384, 384)))
        d = [int(mask['studyidk']), mask['instanceidk'], mask['instancefilename'], 
             int(mask['frame']), mask['chamber'], mask['view'], resized_mask]
        
        io_segmentation.save_ground_truth_numpy_array_to_db(d, gt_table_column_names)
    
    logger.info('{} ground truth masks written to the segmentation.ground_truths table'.format(masks_df.shape[0]))
    

def get_lines(row):
    """Get lines from start and end coordinates.
    
    Output format: [((x1, y1), (x2, y2)), ...]
    
    :param row: measurement segments grouped by instance and index
    :return: line coordinates
    """

    x1s, y1s, x2s, y2s = (
        row["x1coordinate"],
        row["y1coordinate"],
        row["x2coordinate"],
        row["y2coordinate"],
    )
    start_points, end_points = tuple(zip(x1s, y1s)), tuple(zip(x2s, y2s))
    lines = list(zip(start_points, end_points))
    return lines


# TODO: Join outer points using spline curves?
def get_points(row):
    """Get outer points of polygon from chord lines.
    
    Output format: [(x1, y1), (x2, y2), ...]
    
    The "Eje largo" chord line is first and perpendicular to the "Disco" chord lines.
    
    Start at one end of "Eje largo" line, cycle through one end of "Disco" lines,
    then cycle back through other end of "Eje largo"/"Disco" lines, ending at start point.
    
    :param: measurement segments grouped by instance and index
    :return: point coordinates
    """

    eje_largo_line = row["lines"][0]
    discos_lines = row["lines"][1:]

    eje_largo_start = eje_largo_line[0]
    eje_largo_end = eje_largo_line[1]

    discos_starts = [line[0] for line in discos_lines]
    discos_ends = [line[1] for line in discos_lines]

    points = [eje_largo_start]
    points += discos_starts
    points += [eje_largo_end]
    points += reversed(discos_ends)
    points += [eje_largo_start]
    return points


def get_mask(row):
    """Get Numpy mask from polygon specified by points.
    
    :param: measurement segments grouped by instance and index
    :return: Numpy mask
    """

    #poly = Polygon(row["points"])
    #exterior = list(poly.exterior.coords)

    # https://scikit-image.org/docs/dev/api/skimage.draw.html#skimage.draw.polygon
    c = [pt[0] for pt in row['points']]#exterior]
    r = [pt[1] for pt in row['points']] #exterior]
    rr, cc = polygon(r, c)

 
    proper_file_name = 'a_' + str(int(row['studyidk'])) + '_' + row['instancefilename'] +'.dcm_raw'
    _, _, nrow, ncol, _, _ = extract_metadata_for_measurements(row['file_path'], proper_file_name)
    
    if nrow == 0:
        nrow = 600
    if ncol == 0:
        ncol = 800
        
    SHAPE = (nrow, ncol)
    try:
        img = np.zeros(SHAPE, dtype=np.uint8)
        img[rr, cc] = 1
    except:
        img = None
    
    return img


def create_masks(dcm_path):
    """Convert measurement segments to Numpy masks.
    
    :return: updated DataFrame
    """

    io_views = dbReadWriteViews()

    chords_by_volume_mask_df = io_views.get_table("chords_by_volume_mask")
    #instances_w_labels_test_downsampleby5_df = io_views.get_table('instances_w_labels_test_downsampleby5')    
    #chords_by_volume_mask_df.loc[chords_by_volume_mask_df["view_name"].str.contains('ven'), "chamber"] = "lv"
    #chords_by_volume_mask_df.loc[chords_by_volume_mask_df["view_name"].str.contains('atr'), "chamber"] = "la"
    
    #merge_df = pd.merge(instances_w_labels_test_downsampleby5_df, chords_by_volume_mask_df, 
    #                  how='inner', on=['studyidk', 'instanceidk'])
    
    #merge_df = chords_by_volume_mask_df[chords_by_volume_mask_df['instanceidk'].isin(instance_ids)]

    start = time()
    group_df = chords_by_volume_mask_df.groupby(["studyidk", "instanceidk", "indexinmglist"]).agg(
        {
            "x1coordinate": list,
            "y1coordinate": list,
            "x2coordinate": list,
            "y2coordinate": list,
            "chamber": pd.Series.unique,
            "frame": pd.Series.unique,
            "view" : pd.Series.unique,
            "instancefilename" : pd.Series.unique,
        }
    )
    end = time()
    
    logger.info(f"{int(end-start)} seconds to group {len(group_df)} rows")
    
    path = dcm_path

    file_paths = []
    filenames = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.dcm_raw' in file:
                file_paths.append(os.path.join(r, file))
                fullfilename = os.path.basename(os.path.join(r, file))
                #print(str(fullfilename).split('.')[0])
                f = str(fullfilename).split('.')[0]
                f = str(f).split('_')[2]
                #f = str(fullfilename).split('.')[0]
                filenames.append(f)
                
    logger.info("Number of files in the directory: {}".format(len(file_paths)))
    #print(filenames)
    filename_df = pd.DataFrame({'file_name': filenames})
    filename_df['file_path'] = os.path.join(dcm_path, 'raw')

    group_df = group_df.reset_index()

    file_gt_masks = pd.merge(filename_df, group_df, how='inner', left_on =['file_name'], right_on = ['instancefilename'])
    logger.info("Number of files successfully matched with ground truth masks: {}".format(file_gt_masks.shape[0]))
    
    #merge_df = pd.merge(instances_w_labels_test_downsampleby5_df, group_df, 
    #                  how='left', on=['studyidk', 'instanceidk'])
    #print('{} rows on which to generate masks'.format(merge_df.shape[0]))
    
    group_df = file_gt_masks

    start = time()
    group_df["lines"] = group_df.apply(get_lines, axis=1)
    group_df["points"] = group_df.apply(get_points, axis=1)
    group_df["mask"] = group_df.apply(get_mask, axis=1)
    group_df = group_df.reset_index()
    end = time()
    
    logger.info(f"{int(end-start)} seconds to apply {len(group_df)} rows")

    return group_df
