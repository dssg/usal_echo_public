from d00_utils import db_utils, s3_utils
from d01_data import ingestion_dcm, ingestion_xtdb#, master_list
from d02_intermediate import clean_dcm, clean_xtdb
from d03_classification import predict_view_v0

from d02_intermediate import create_dataset

if __name__ == "__main__":
    #predict_view_v0.main()
	create_dataset.run_all()
