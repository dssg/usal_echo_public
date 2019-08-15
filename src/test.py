'''
Function for prototyping individual scripts
'''

from d00_utils import db_utils, s3_utils
from d01_data import ingestion_dcm, ingestion_xtdb#, master_listM
from d02_intermediate import clean_dcm, clean_xtdb, download_dcm, dcm_utils
from d02_intermediate import clean_dcm, clean_xtdb, download_dcm

from d03_classification import classificn_results_to_db

if __name__ == "__main__":
    #predict_view_v0.main()
	#filter_instances.run_all()
    #filter_views.filter_by_views()
    #classificn_results_to_db.results_txt_to_db()
    #predict_view.run_classify()
    #download_dcm.s3_download_decomp_dcm(0.05, 1.0, 'instances_w_labels_train')
    
    
    #dcm_dir = '/home/ubuntu/data/01_raw/train_split100_downsampleby20/'
    #img_dir = '/home/ubuntu/data/02_intermediate/test_split100_downsampleby20/'
    #dcm_utils.dcmdir_to_jpgs_for_classification(dcm_dir, img_dir)
