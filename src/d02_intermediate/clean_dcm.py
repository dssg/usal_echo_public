#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 4 14:32:40 2019

@author: wiebket
"""

import pandas as pd
from json import load

from d00_utils.db_utils import dbReadWriteClean


def clean_dcm(metadata_path, to_db=False, credentials_file=None, db_table=None):
    """Select subset of dicom tags and save to database.
    
    This function selects a subset of dicom metadata tags and saves it.
    
    **Requirements:
    json formatted config file with dicom tag descriptions and values in dicom_tags.json
    
    :param metadata_path (str): path to dicom metadata file
    :param save_to_db (bool): if True saves tag subset to postgres database; default=False
    :param credentials_file (str): path to credentials file; required if save_to_db=True
    :param db_table (str): name of database table to write to; required if save_to_db=True
    :return: pandas dataframe with filtered metadata
    
    """
    # Get dicom tags and metadata path from config file(s)
    with open("dicom_tags.json") as f:
        dicom_tags = load(f)
    for k, v in dicom_tags.items():
        dicom_tags[k] = tuple(v)

    io_clean = dbReadWriteClean()

    datalist = []
    # Read metadata in chunks to avoid kernel crashing due to large data volume.
    for chunk in pd.read_csv(
        metadata_path,
        chunksize=1000000,
        dtype={
            "dirname": "category",
            "filename": "category",
            "tag1": "category",
            "tag2": "category",
        },
    ):
        chunk["tags"] = list(zip(chunk["tag1"], chunk["tag2"]))
        filtered_chunk = chunk[chunk["tags"].isin(dicom_tags.values())]
        if to_db is True:
            try:
                io_clean.save_to_db(filtered_chunk, db_table, if_exists="append")
            except:
                raise
            print("saved chunks to db")
        datalist.append(filtered_chunk)

    meta = pd.concat(datalist)

    return meta
