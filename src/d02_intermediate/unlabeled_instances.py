import pandas

from d00_utils.db_utils import dbReadWriteViews

'''
This script contains create_unlabeled_instances(), which creates
the table views.instances_labeled_other with the following columns:
    sopinstanceuid, instancefilename, instanceidk, studyidk

Unlabeled instances refer to the instances that can be used 
    for the "other" class when classifying views
That is, these instances meet the following criteria:
    1) from studies which contain labeled instances
    2) are not labeled as plax, a4c, a2c, or combinations thereof
'''

def create_unlabeled_instances():
    io_views = dbReadWriteViews()

    all_inst_df = io_views.get_table('instances_unique_master_list')
    label_inst_df = io_views.get_table('instances_with_labels')
    confl_inst_df = io_views.get_table('instances_with_conflicts')

    # get studies for which labels exist
    studies_w_labels = list(set(label_inst_df['studyidk'].tolist()))

    # get instances w labels - to be removed from all instances
    inst_w_labels = list(set(label_inst_df['instanceidk'].tolist()))

    # get instances w conflicts - to be removed from all instances
        # conflict i.e. instances s.t. is_multiview==True
    inst_w_conflicts = list(set(confl_inst_df['instanceidk'].tolist()))

    # get all instanceidk's from studies which have measurements
    poss_inst_df = all_inst_df[all_inst_df['studyidk'].isin(studies_w_labels)]

    # remove instances which have labels
    poss_inst_df = poss_inst_df[~poss_inst_df['instanceidk'].isin(inst_w_labels)]

    # remove instances which have conflicts
    poss_inst_df = poss_inst_df[~poss_inst_df['instanceidk'].isin(inst_w_conflicts)]

    #io_views.save_to_db(poss_inst_df, 'instances_labeled_other')

    # Filter out instances from old machines
    new_machines_df = io_views.get_table('machines_new_bmi')
    studies_new_machines = list(set(new_machines_df['studyidk'].tolist()))
    oth_inst_new_df = poss_inst_df[poss_inst_df['studyidk'].isin(studies_new_machines)]

    oth_inst_new_df['view'] = 'other' # add 'other' column for consistency with view table
    
    io_views.save_to_db(oth_inst_new_df, 'instances_labeled_other')
