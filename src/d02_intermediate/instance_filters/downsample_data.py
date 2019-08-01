import pandas as pd
import numpy as np

from d00_utils.db_utils import dbReadWriteViews


def downsample_df(df, ratio=0.1):
    """ Downsample any dataframe by a given ratio """

    msk = np.random.rand(len(df)) < ratio

    return df[msk]


def downsample_train_test():

    io_views = dbReadWriteViews()

    # df_inst_w_labels = io_views.get_table('instances_with_labels')
    df_train = io_views.get_table("instances_w_labels_train")
    df_test = io_views.get_table("instances_w_labels_test")

    ratio = 0.1
    inv_ratio = int(1 / ratio)

    df_train_downsampled = downsample_df(df_train, ratio)
    df_test_downsampled = downsample_df(df_test, ratio)

    io_views.save_to_db(
        df_train_downsampled, "instances_w_labels_train_downsamp{0}".format(inv_ratio)
    )
    io_views.save_to_db(
        df_test_downsampled, "instances_w_labels_test_downsamp{0}".format(inv_ratio)
    )
