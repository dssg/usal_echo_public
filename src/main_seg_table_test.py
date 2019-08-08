from d00_utils import db_utils, s3_utils
from d01_data import ingestion_dcm, ingestion_xtdb #, master_list
from d02_intermediate import clean_dcm, clean_xtdb
from d03_classification import predict_view_v0
from d04_segmentation import segment_view

#from d02_intermediate import train_test_split


if __name__ == "__main__":
    io_segmentation = dbReadWriteSegmentation()
    table = io_segmentation.get_segmentation_table('predictions')
    print(table)
