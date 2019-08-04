#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 2019

@author: wiebket
"""

import pandas as pd
import os
import boto3
import tempfile

from d00_utils.db_utils import dbReadWriteRaw
from d00_utils.s3_utils import get_matching_s3_keys


def get_dicom_metadata(bucket, file_path, description=False):

    """Get all dicom tags for file in file_path.

    This function uses gdcmdump to retrieve the metadata tags of the file in object_bath.
    The tags are as a pandas dataframe.
    
    ** Requirements 
    libgdcm: 
        Unix install with `sudo apt-get install libgdcm-tools`
        Mac install with `brew install gdcm`
    .aws/credentials file with s3 access details saved as default profile
    
    :param bucket (str): s3 bucket
    :param object_path (str): path to dicom file
    :param description (bool): include dicom tag descriptions instead of 
        values, default=False        
    :return: pandas DataFrame object with columns=['dirname','filename','tag1','tag2','value']
    
    """
    s3 = boto3.client("s3")
    tmp = tempfile.NamedTemporaryFile()

    # Dump metadata of file to temp file
    s3.download_file(bucket, file_path, tmp.name)
    os.system("gdcmdump " + tmp.name + " > temp.txt")

    dir_name = file_path.split("/")[0]
    file_name = file_path.split("/")[1].split(".")[0]

    # Parse temp.txt file to extract tags
    temp_file = "temp.txt"
    meta = []
    with open(temp_file, "r") as f:
        line_meta = []
        for one_line in f:
            try:
                clean_line = one_line.replace("]", "").strip()
                if not clean_line:  # ignore empty lines
                    continue
                elif not clean_line.startswith("#"):  # ignore comment lines:
                    tag1 = clean_line[1:5]
                    tag2 = clean_line[6:10]
                    if description == False:
                        value = (
                            clean_line[15 : clean_line.find("#")]
                            .strip()
                            .replace("[", "")
                        )
                    elif description == True:
                        value = clean_line[clean_line.find("#") + 2 :].strip()
                    line_meta = [dir_name, file_name, tag1, tag2, value]
                    meta.append(line_meta)
            except IndexError:
                break

    df = pd.DataFrame.from_records(
        meta, columns=["dirname", "filename", "tag1", "tag2", "value"]
    )
    df_dedup = df.drop_duplicates(keep="first")
    df_dedup_goodvals = df_dedup[~df_dedup.value.str.contains("no value")]
    df_dedup_goodvals_short = df_dedup_goodvals[
        (df_dedup_goodvals["value"].str.len() > 0)
        & (df_dedup_goodvals["value"].str.len() < 50)
    ]
    df_out = df_dedup_goodvals_short.replace({"value": {r"\\": "--"}}, regex=True)

    return df_out


def write_dicom_metadata_csv(df, metadata_file_suffix=None):
    """Write the output of 'get_dicom_metadata()' to a csv file.
    
    :param df (pandas.DataFrame): output of 'get_dicom_metadata()'
    :param metadata_file_suffix (str): string to append to metadata file name 
        'dicom_metadata.csv', default=None
    :return: csv file; saves to ~/data_usal/01_raw/dicom_metadata.csv
    
    """
    data_path = os.path.join(os.path.expanduser("~"), "data_usal", "01_raw")
    os.makedirs(os.path.expanduser(data_path), exist_ok=True)
    if metadata_file_suffix is None:
        dicom_meta_path = os.path.join(data_path, "dicom_metadata.csv")
    else:
        dicom_meta_path = os.path.join(
            data_path, "dicom_metadata_" + str(metadata_file_suffix) + ".csv"
        )
    if not os.path.isfile(dicom_meta_path):  # create new file if it does not exist
        print("Creating new metadata file")
        df.to_csv(dicom_meta_path, index=False)
    else:  # if file exists append
        df.to_csv(dicom_meta_path, mode="a", index=False, header=False)

    print(
        "dicom metadata saved for study {}, instance {}".format(
            df.iloc[0, 0], df.iloc[0, 1]
        )
    )


def write_dicom_metadata_postgres(df, db_table):
    """Write the output of 'get_dicom_metadata()' to a postgres table.
    
    :param df (pandas.DataFrame): output of 'get_dicom_metadata()'
    :param db_table (str): database table to create and write to
    :return: db_table written to schema 'raw'
    
    """
    io_raw = dbReadWriteRaw()
    io_raw.save_to_db(df, db_table, "append")

    print("study: {}, instance: {}".format(df.iloc[0, 0], df.iloc[0, 1]))


def ingest_dcm(write_to="postgres", prefix=""):
    """
    Retrieve all dicom metadata from s3 and save to dicom_metadata.csv file.
    
    NB: prefix='' retrieves all metadata, which takes ~60hrs 
    
    :param write_to (str): method for writing data, can be 'postgres' (default) or 'csv'
    :param prefix (str): study for which to retrieve metadata, default='' (ie. all)
    :return: (none) dicom metadata from s3/cibercv retrieved and stored
    
    """
    if write_to == "postgres":
        func = 'write_dicom_metadata_postgres(df, "metadata")'
    elif write_to == "csv":
        func = "write_dicom_metadata_csv(df)"

    for key in get_matching_s3_keys("cibercv", prefix, ".dcm"):
        df = get_dicom_metadata("cibercv", key)
        eval(func)
    os.remove("temp.txt")

    return "All dicom metadata has been retrieved."
