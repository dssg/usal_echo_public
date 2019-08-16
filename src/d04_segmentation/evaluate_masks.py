# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 13:46:14 2019

@author: court
"""

from d00_utils.db_utils import dbReadWriteSegmentation
import numpy as np
from d00_utils.log_utils import setup_logging

logger = setup_logging(__name__, __name__)

def evaluate_masks():
    #Go through the ground truth table and write IOUS
    
    # Prediction Table: "instance_id","study_id", "view_name", "frame", "output_np_lv", "output_np_la",
    #        "output_np_lvo","output_image_seg", "output_image_orig", "output_image_overlay", "date_run",
    #        "file_name"
    # Ground truth table: ground_truth_id, instance_id, frame, chamber, study_id, view_name, numpy_array
    # Evaluation Table: evaluation_id, instance_id, frame, chamber, study_id, score_type, score_value
    
    io_segmentation = dbReadWriteSegmentation()
    ground_truths = io_segmentation.get_segmentation_table('ground_truths')
    
    #instance_id_list = ground_truths.instance_id.unique() 
    #instance_id_list = instance_id_list.astype(str)
    #predictions = io_segmentation.get_instances_from_segementation_table('predictions', instance_id_list)

    
    #Go through the ground truth table and write IOUS
        
    for index, gt in ground_truths.iterrows():
        #match the gt to the predictin table
        gt_instance_id = gt['instance_id']
        gt_study_id = gt['study_id']
        gt_chamber = gt['chamber']
        gt_view_name = gt['view_name']
        gt_frame_no = gt['frame']
        
        pred = io_segmentation.get_instance_from_segementation_table('predictions', gt_instance_id)
        pred = pred.reset_index()
        logger.info('got predictions details for instance {}'.format(gt_instance_id))
        
        if len(pred.index) > 0:
            pred_view_name = gt['view_name']
            #retrieve gt numpy array
            gt_numpy_array = io_segmentation.convert_to_np(gt['numpy_array'], 1)#frame = 1, as it wants number of frames in np array, not frame number
            if gt_numpy_array == None:
                continue
            #retrive relevant pred numpy array
            if gt_chamber == 'la':
                pred_numpy_array = io_segmentation.convert_to_np(pred['output_np_la'][0], pred['num_frames'][0])            
            elif gt_chamber == 'lv':
                pred_numpy_array = io_segmentation.convert_to_np(pred['output_np_lv'][0], pred['num_frames'][0])
            elif gt_chamber == 'lvo':
                pred_numpy_array = io_segmentation.convert_to_np(pred['output_np_lvo'][0], pred['num_frames'][0])
            else:
                logger.error('invalid chamber') 
            
            #get the frame of the prediction, that corresponds to the frame of the ground thruth
            pred_numpy_array_frame = pred_numpy_array[gt_frame_no, :, :]
            
            #calculate iou
            reported_iou = iou(gt_numpy_array, pred_numpy_array_frame)
            logger.info('IOU of: {}'.format(reported_iou))
        else:
            logger.error('No record exists for study id {} & instance id {}'.format(gt_study_id, gt_instance_id))
        
        #write to db
        # Evaluation Table: evaluation_id, instance_id, frame, chamber, study_id, score_type, score_value
        d_columns = ['instance_id', 'frame', 'chamber', 'study_id', 'score_type', 
                     'score_value', 'gt_view_name', 'pred_view_name']
        d = [gt_instance_id, gt['frame'], gt_chamber, gt_study_id, 'iou', 
             reported_iou, gt_view_name, pred_view_name]
        #df = pd.DataFrame(data=d, columns=d_columns)
        io_segmentation.save_seg_evaluation_to_db(d, d_columns)
    

def iou(gt, pred):
    gt_bool = np.array(gt, dtype=bool)
    pred_bool = np.array(pred, dtype=bool)

    overlap = gt_bool * pred_bool  # Logical AND
    union = gt_bool + pred_bool  # Logical OR

    IOU = float(overlap.sum()) / float(union.sum())

    return IOU
