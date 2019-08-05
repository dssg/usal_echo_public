# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 13:46:14 2019

@author: court
"""

from src.d00_utils.db_utils import dbReadWriteViews
import os

def calculate_ious(instance_id_list):
    """
    Produces a dataframe with IOU detail for segmentation model testing. 
    
    :param: instanceids: a list of instance ids
    :return df
    """
    def iou(gt, pred):
        gt_bool = np.array(gt, dtype=bool)
        pred_bool = np.array(pred, dtype=bool)
        
        overlap = gt_bool*pred_bool # Logical AND
        union = gt_bool + pred_bool # Logical OR
        
        IOU = float(overlap.sum())/float(union.sum())
        
        return IOU
    
    
    
    
    
    