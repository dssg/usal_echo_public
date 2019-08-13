'''
Function for prototyping individual scripts
'''

from d00_utils import db_utils, s3_utils
from d01_data import ingestion_dcm, ingestion_xtdb#, master_list
from d02_intermediate import clean_dcm, clean_xtdb, download_dcm
from d03_classification import predict_view_v0, classificn_results_to_db, predict_view

#from d02_intermediate import filter_instances
#from d02_intermediate.instance_filters import filter_views

if __name__ == "__main__":
    #predict_view_v0.main()
	  #filter_instances.run_all()
    #filter_views.filter_by_views()
    #classificn_results_to_db.results_txt_to_db()
    #predict_view.run_classify()
    download_dcm.s3_download_decomp_dcm(0.05, 1.0, 'instances_w_labels_train')
