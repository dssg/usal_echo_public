import numpy as np

from usal_echo.d00_utils.db_utils import dbReadWriteViews
from usal_echo.d00_utils.log_utils import *


def downsample_df(df, ratio):
    """ Downsample any dataframe by a given ratio """

    msk = np.random.rand(len(df)) < ratio

    return df[msk]


def downsample_train_test(ratio=0.1):

    logger = setup_logging(__name__, "downsample_data.py")

    io_views = dbReadWriteViews()

    # df_inst_w_labels = io_views.get_table('instances_with_labels')
    df_train = io_views.get_table("instances_w_labels_train")
    df_test = io_views.get_table("instances_w_labels_test")

    inv_ratio = int(1 / ratio)

    df_train_downsampled = downsample_df(df_train, ratio)
    df_test_downsampled = downsample_df(df_test, ratio)

    io_views.save_to_db(
        df_train_downsampled,
        "instances_w_labels_train_downsampleby{0}".format(inv_ratio),
    )
    io_views.save_to_db(
        df_test_downsampled, "instances_w_labels_test_downsampleby{0}".format(inv_ratio)
    )
    logger.info("Dataset downsampled by a factor of {0}".format(inv_ratio))
