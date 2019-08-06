#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 4 14:32:40 2019

@author: wiebket
"""

import pandas as pd
from json import load
import os
from pathlib import Path

from d00_utils.db_utils import dbReadWriteRaw, dbReadWriteClean
from d00_utils.log_utils import *
logger = setup_logging(__name__, "d02_intermediate")

dcm_tags = os.path.join(Path(__file__).parents[0], "dicom_tags.json")

def clean_dcm_meta():
    """Selects a subset of dicom metadata tags and saves them to postgres.
    
    **Requirements:
    json formatted config file with dicom tag descriptions and values 
    in d02_intermediate/dicom_tags.json

    """
    with open(dcm_tags) as f:
        dicom_tags = load(f)
    for k, v in dicom_tags.items():
        dicom_tags[k] = tuple(v)

    io_raw = dbReadWriteRaw()
    io_clean = dbReadWriteClean()
    metadata = io_raw.get_table("metadata")

    metadata["tags"] = list(zip(metadata["tag1"], metadata["tag2"]))
    meta_lite = metadata[metadata["tags"].isin(dicom_tags.values())]

    io_clean.save_to_db(meta_lite, "meta_lite")
    logger.info("Metadata filtered.")


def clean_dcm(metadata_path, to_db=False, credentials_file=None, db_table=None):
    """Select subset of dicom tags and save to database.
    
    This function selects a subset of dicom metadata tags and saves it.
        
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
