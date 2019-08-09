# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 13:46:14 2019

@author: court
"""

from d00_utils.db_utils import dbReadWriteSegmentation
import numpy as np

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
    predictions = io_segmentation.get_segmentation_table('predictions')
    print('hello again xx')
    ground_truths = io_segmentation.get_segmentation_table('ground_truths')
    
    print('tables obtained')
    
    #Go through the ground truth table and write IOUS
        
    for index, gt in ground_truths.iterrows():
        #match the gt to the prediction table
        gt_instance_id = gt['instance_id']
        gt_study_id = gt['study_id']
        gt_chamber = gt['chamber']
        
        pred = predictions.loc[(predictions['study_id'] == gt_study_id) & 
                               (predictions['instance_id'] == gt_instance_id)]
        
        #retrieve gt numpy array
        gt_numpy_array = io_segmentation.convert_to_np(gt.at[index, 'numpy_array'], 1) 
                    #frame = 1, as it wants number of frames in np array, not frame number
        
        #retrive relevant pred numpy array
        if gt_chamber == 'la':
            pred_numpy_array = io_segmentation.convert_to_np(pred['output_np_la'], pred['frame'])            
        elif gt_chamber == 'lv':
            pred_numpy_array = io_segmentation.convert_to_np(pred['output_np_lv'], pred['frame'])
        elif gt_chamber == 'lvo':
            pred_numpy_array = io_segmentation.convert_to_np(pred['output_np_lvo'], pred['frame'])
        else:
            print('invalid chamber') #TODO: log this
        
        #calculate iou
        reported_iou = iou(gt_numpy_array, pred_numpy_array)
        print('IOU of: {}'.format(reported_iou))
        
        #write to db
        # Evaluation Table: evaluation_id, instance_id, frame, chamber, study_id, score_type, score_value
        df = {'instance_id' : gt_instance_id, 'frame' : gt['frame'], 
              'chamber' : gt_chamber, 'study_id': gt_study_id, 
              'score_type' : 'iou', 'score_value' : reported_iou}
        io_segmentation.save_seg_evaluation_to_db(df)
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