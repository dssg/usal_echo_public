#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:26:44 2019

@author: saintlyvi
"""
import boto3
import tempfile
import pandas as pd
import os

from .d00_utils.s3_utils import get_matching_s3_keys
from .d00_utils.db_utils import dbReadWriteRaw, dbReadWriteEncode



def download_xtdb():
    
    raw_data = dbReadWriteRaw()
    tmp = tempfile.NamedTemporaryFile()
    
    for file in get_matching_s3_keys('cibercv','0.DATABASE','.csv'):
       s3 = boto3.client('s3')
       s3.download_file('cibercv', file, tmp.name)
       
       tbl = pd.read_csv(tmp.name)
       tbl_name = file.split('/')[-1].split('.')[0]
       
       raw_data.save_to_db(tbl, tbl_name)
       print('Created table `'+tbl_name+'` in schema '+raw_data.schema)


       
def encode_xtdb():
    
    raw_data = dbReadWriteRaw()
    encode_data = dbReadWriteEncode()
    
    raw_tables = raw_data.list_tables()
    
    for t in raw_tables:
        tbl = raw_data.get_table(t)
        # encode table in utf8
        encode_tbl = ()
        encode_data.save_to_db(encode_tbl, t)
        

        
def clean_xtdb():
    
    # do some stuff
        
    