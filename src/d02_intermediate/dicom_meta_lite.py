#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:32:40 2019

@author: wiebket
"""

import pandas as pd

from ..d00_utils.db_utils import save_to_db

def get_meta_lite(dicom_tags, metadata_path, to_db=False, credentials_file=None, db_table=None):

    """Select subset of dicom tags and save to database.
    
    :param dicom_tags (dict): dict of dicom tag descriptions and tag tuple
    :param metadata_path (str): path to dicom metadata file
    :param save_to_db (bool): if True saves tag subset to postgres database; default=False
    :param credentials_file (str): path to credentials file; required if save_to_db=True
    :param db_table (str): name of database table to write to; required if save_to_db=True
    :return: pandas dataframe with filtered metadata
    
    """    
    datalist = []
    # Read metadata in chunks to avoid kernel crashing due to large data volume.
    for chunk in pd.read_csv(metadata_path, chunksize=1000000,
                             dtype={'dirname':'category','filename':'category',
                                    'tag1':'category','tag2':'category'}):
        chunk['tags'] = list(zip(chunk['tag1'],chunk['tag2']))
        filtered_chunk = chunk[chunk['tags'].isin(dicom_tags.values())]
        if to_db is True:
            try:
                save_to_db(filtered_chunk, db_table, credentials_file)
            except:
                raise
            print('saved chunks to db')
        datalist.append(filtered_chunk)
    
    meta = pd.concat(datalist)
    
    return meta