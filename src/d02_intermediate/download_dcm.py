# coding: utf-8

import random
import os
import numpy as np

from d00_utils.db_utils import dbReadWriteViews
from d00_utils.s3_utils import download_s3_objects
from d02_intermediate.dcm_utils import decompress_dcm
from d00_utils.log_utils import setup_logging
logger = setup_logging(__name__, "d02_intermediate")


def split_train_test(ratio, table_name):
    """Split views.table_name into train/test. 
    
    :param ratio (float): ratio for splitting into train/test
                 e.g. if 0.8, will take 80% as train set and 20% as test set
    :param table_name (str): name of views.table with master instances

    """
    io_views = dbReadWriteViews()
    df = io_views.get_table(table_name)

    np.random.seed(0)
    msk = np.random.rand(len(df)) < ratio
    df_train = df[msk].reset_index(drop=True)
    df_test = df[~msk].reset_index(drop=True)

    perc_trn = int(100 * ratio)
    perc_tst = 100 - perc_trn
    logger.info("{} split into {}% train, {}% test".format(table_name, perc_trn, perc_tst))
    
    return df_train, df_test


def downsample_train_test(downsample_ratio, train_test_ratio, table_name):
    """Downsamples views.table_name train/test dataset by a factor of X.

    :param downsample_ratio (float): percentage by which to downsample dataset
                         e.g. if ratio=0.1, will downsample by a factor of 10
    :param train_test_ratio (float): ratio for splitting into train/test
    :param table_name (str): name of views.table with master instancest

    """
    df_train, df_test = split_train_test(train_test_ratio, table_name)
    
    np.random.seed(0)
    msk_train = np.random.rand(len(df_train)) < downsample_ratio
    msk_test = np.random.rand(len(df_test)) < downsample_ratio

    df_train_downsampled = df_train[msk_train].reset_index(drop=True)
    df_test_downsampled = df_test[msk_test].reset_index(drop=True)
    
    inv_ratio = int(1/downsample_ratio)
    logger.info("{} downsampled by a factor of {}".format(table_name, inv_ratio))

    return df_train_downsampled, df_test_downsampled


def s3_download_decomp_dcm(downsample_ratio, train_test_ratio, table_name, train):
    """Downloads and decompresses test/train dicoms from s3.
    
    :param downsample_ratio (float): percentage by which to downsample dataset
                         e.g. if ratio=0.1, will downsample by a factor of 10
    :param train_test_ratio (float): ratio for splitting into train/test
    :param table_name (str): name of views.table with master instancest
    :param train (bool): download train set instead of test set, default=False
    
    """
    df_train, df_test = downsample_train_test(downsample_ratio, train_test_ratio, table_name)
    
    if train is True:
        instances = df_train
        dir_name = 'train_split{}_downsampleby{}'.format(int(100*train_test_ratio), int(1/downsample_ratio))
    else:
        instances = df_test
        dir_name = 'test_split{}_downsampleby{}'.format(int(100-100*train_test_ratio), int(1/downsample_ratio))
        
    prefix = instances['studyidk'].astype(str) + '/a_' + instances['filename'].astype(str)
    filenames = 'a_' + instances['studyidk'].astype(str) + '_' + instances['filename'].astype(str) + '.dcm'

    download_dict = dict(zip(prefix, filenames))
    
    datadir = os.path.expanduser('/home/ubuntu/data/01_raw/'+dir_name)
    raw_datadir = os.path.join(datadir, 'raw')

    for p in list(download_dict.keys()):
        dcm_filepath=os.path.join(datadir, download_dict[p])
        #TODO: consider downloading to a main image directory and only creating a 
        #symlink to experiment directory. Could save a lot of downloading time.
        if not os.path.isfile(dcm_filepath):
            download_s3_objects('cibercv', outfile=dcm_filepath, prefix=p, suffix='.dcm') 

        dcm_rawfilepath=os.path.join(raw_datadir, download_dict[p]+'_raw')
        if not os.path.isfile(dcm_rawfilepath):
            decompress_dcm(dcm_filepath, dcm_rawfilepath)
            
    return dir_name
