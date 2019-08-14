import os
from os import listdir
from os.path import isfile, join
import sys
import pandas as pd

from d00_utils.db_utils import dbReadWriteViews, dbReadWriteClassification


def _groundtruth_views():

    # Get ground truth labels via views.instances_w_labels table
    io_views = dbReadWriteViews()
    io_class = dbReadWriteClassification()
    
    groundtruth = io_views.get_table('instances_w_labels')
    groundtruth.rename(columns={'filename': 'file_name'}, inplace=True)
    predictions = io_class.get_table('predictions')

    # Merge tables df_new and labels_df
    df = predictions.merge(ground_truth, on='file_name')   
    
    return df

def evaluate_views(view_mapping):
    
    groundtruth = _groundtruth_views()
    groundtruth.rename(columns={'view': 'view_true'}, inplace=True)
    df['correct'] = df[view_mapping] == df['view_true'] 
    
    return df
