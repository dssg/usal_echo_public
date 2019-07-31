import pandas as pd
import sys
import numpy as np

from d00_utils.db_utils import dbReadWriteViews

def downsample_df():
    ''' Split dataset into train/test from db table views.instances_with_labels '''

    io_views = dbReadWriteViews()

    df = io_views.get_table('instances_with_labels')

    msk = np.random.rand(len(df)) < 0.8

    df_train = df[msk]
    df_test = df[~msk]

    io_views.save_to_db(df_train, 'instances_w_labels_train')
    io_views.save_to_db(df_test, 'instances_w_labels_test')


def downsample_train_test():
    
    io_views = dbReadWriteViews()

    df_inst_w_labels = io_views.get_table('instances_with_labels')

    ratio = 0.25
