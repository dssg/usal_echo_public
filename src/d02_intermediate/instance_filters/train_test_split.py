import numpy as np

from d00_utils.db_utils import dbReadWriteViews
from d00_utils.log_utils import *


def split_train_test(ratio=0.5):
    """ 
    Split dataset into train/test from db table views.instances_with_labels 
    
    :param ratio: ratio for splitting into train/test
                 e.g. if 0.8, will take 80% as train set and 20% as test set
    """

    logger = setup_loggin(__name__, "train_test_split.py")

    io_views = dbReadWriteViews()

    df = io_views.get_table("instances_with_labels")

    msk = np.random.rand(len(df)) < ratio

    df_train = df[msk]
    df_test = df[~msk]

    io_views.save_to_db(df_train, "instances_w_labels_train")
    io_views.save_to_db(df_test, "instances_w_labels_test")

    perc_trn = int(100 * ratio)
    perc_tst = 100 - perc_trn
    logger.info("Dataset split into {0}% train, {1}% test".format(perc_trn, perc_tst))
