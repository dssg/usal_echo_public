# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 13:46:14 2019

@author: court
"""

from d00_utils.db_utils import dbReadWriteSegmentation
import numpy as np
import pandas as pd

def main():
    #Go through the ground truth table and write IOUS
    
    # Prediction Table: "instance_id","study_id", "view_name", "frame", "output_np_lv", "output_np_la",
    #        "output_np_lvo","output_image_seg", "output_image_orig", "output_image_overlay", "date_run",
    #        "file_name"
    # Ground truth table: ground_truth_id, instance_id, frame, chamber, study_id, view_name, numpy_array
    # Evaluation Table: evaluation_id, instance_id, frame, chamber, study_id, score_type, score_value
 
    print('hello world x')
    
    io_segmentation = dbReadWriteSegmentation()
    print(io_segmentation.cursor)
    print('cursor obtained')
    ground_truths = io_segmentation.get_segmentation_table('ground_truths')
    print('ground truth obtained')
    
    instance_id_list = ground_truths.instance_id.unique() 
    
    predictions = io_segmentation.get_instances_from_segementation_table('predictions', instance_id_list)
    print('prediction tables obtained')
    
    #Go through the ground truth table and write IOUS
        
    for index, gt in ground_truths.iterrows():
        #match the gt to the prediction table
        gt_instance_id = gt['instance_id']
        gt_study_id = gt['study_id']
        gt_chamber = gt['chamber']
        
        pred = predictions.loc[(predictions['study_id'] == gt_study_id) & 
                               (predictions['instance_id'] == gt_instance_id)]
        pred = pred.reset_index()
        
        if len(pred.index) > 0:
            #retrieve gt numpy array
            gt_numpy_array = io_segmentation.convert_to_np(gt['numpy_array'], 1) 
                        #frame = 1, as it wants number of frames in np array, not frame number
            print('hello lili')
            #retrive relevant pred numpy array
            if gt_chamber == 'la':
                pred_numpy_array = io_segmentation.convert_to_np(pred['output_np_la'][0], pred['frame'][0])            
            elif gt_chamber == 'lv':
                pred_numpy_array = io_segmentation.convert_to_np(pred['output_np_lv'][0], pred['frame'][0])
            elif gt_chamber == 'lvo':
                pred_numpy_array = io_segmentation.convert_to_np(pred['output_np_lvo'][0], pred['frame'][0])
            else:
                print('invalid chamber') #TODO: log this
            
            #calculate iou
            reported_iou = iou(gt_numpy_array, pred_numpy_array)
            print('IOU of: {}'.format(reported_iou))
        else:
            print('No record exists for study id {} & instance id {}'.format(gt_study_id, gt_instance_id))
        
        #write to db
        # Evaluation Table: evaluation_id, instance_id, frame, chamber, study_id, score_type, score_value
        d_columns = ['instance_id', 'frame', 'chamber', 'study_id', 'score_type', 'score_value']
        d = [gt_instance_id, gt['frame'], gt_chamber, gt_study_id, 'iou', reported_iou]
        #df = pd.DataFrame(data=d, columns=d_columns)
        io_segmentation.save_seg_evaluation_to_db(d, d_columns)
        print('saved to db')
    

def iou(gt, pred):
    gt_bool = np.array(gt, dtype=bool)
    pred_bool = np.array(pred, dtype=bool)

    overlap = gt_bool * pred_bool  # Logical AND
    union = gt_bool + pred_bool  # Logical OR

    IOU = float(overlap.sum()) / float(union.sum())

    return IOU

if __name__ == "__main__":
    main()