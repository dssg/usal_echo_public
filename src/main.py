from d00_utils import db_utils, s3_utils
from d01_data import ingestion_dcm, ingestion_xtdb, master_list
from d02_intermediate import clean_dcm, clean_xtdb
from d03_classification import filter_views, unlabeled_instances

#from d07_luigi import run_luigi

if __name__ == '__main__':
    filter_views.filter_by_views()
    #master_list.create_master_instance_list()
    #unlabeled_instances.create_unlabeled_instances()
