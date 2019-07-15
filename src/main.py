'''
Script for running full pipeline
'''

import os

#from d00_utils import db_utils, s3_utils
#from d01_data import ingestion_dcm, ingestion_xtdb
#from d02_intermediate import clean_dcm, clean_xtdb
#from d03_classification import filter_views
#from d07_luigi import run_luigi


if __name__ == '__main__':
    os.system('luigi --module main --of Pipeline') # path issues? i.e. Pipeline() defined in d07
