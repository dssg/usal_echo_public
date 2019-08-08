#, s3_utils
#from d01_data import ingestion_dcm, ingestion_xtdb #, master_list
#from d02_intermediate import clean_dcm, clean_xtdb
#from d03_classification import predict_view_v0
from d04_segmentation import generate_masks

#from d02_intermediate import train_test_split


if __name__ == "__main__":
    generate_masks.write_masks()