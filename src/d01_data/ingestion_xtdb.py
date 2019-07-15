#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:26:44 2019

@author: wiebket
"""
import boto3
import tempfile
import pandas as pd

from ..d00_utils.s3_utils import get_matching_s3_keys
from ..d00_utils.db_utils import dbReadWriteRaw


def ingest_xtdb():
    """Retrieve all Xcelera_tablas csv files from s3 and save to postgres database.
    
    """
    io_raw = dbReadWriteRaw()
    tmp = tempfile.NamedTemporaryFile()
    
    for file in get_matching_s3_keys('cibercv','0.DATABASE','.csv'):
        s3 = boto3.client('s3')
        s3.download_file('cibercv', file, tmp.name)
       
        tbl = pd.read_csv(tmp.name, encoding='iso-8859-2', na_values='', decimal=',')
        tbl.columns = [t.lower() for t in tbl.columns]
        tbl.dropna(how='all', inplace=True)
        tbl_name = file.split('/')[-1].split('.')[0].lower()
        tbl.fillna('', inplace=True)
        
        io_raw.save_to_db(tbl, tbl_name)
        print('Created table `'+tbl_name+'` in schema '+io_raw.schema)
       
       
if __name__ == '__main__':
    ingest_xtdb()        