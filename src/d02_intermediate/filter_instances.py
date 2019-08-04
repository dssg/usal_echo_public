import sys

from d02_intermediate.instance_filters.filter_machines import filter_by_machine_type
from d02_intermediate.instance_filters.master_list import create_master_instance_list
from d02_intermediate.instance_filters.filter_views import filter_by_views
from d02_intermediate.instance_filters.conflict_instances import find_table_conflicts
from d02_intermediate.instance_filters.unlabeled_instances import create_unlabeled_instances
from d02_intermediate.instance_filters.train_test_split import split_train_test
from d02_intermediate.instance_filters.downsample_data import downsample_train_test


'''
This is a high-level script which runs all filtering of instances needed to create the dataset.
Summaries of each filtering steps are included below. Further details in each original script.
'''
def run_all():

    ''' 
    Creates two tables:
		views.machines_all_bmi: list of all studies in db; columns: studyidk, machine type and bmi
		views.machines_new_bmi: same as machines_all_bmi, but only includes studies with new machines, 
                                i.e. machine types ECOEPIQ2, EPIQ7-1, ECOIE33, AFFINITI_1, AFFINITI_2
    '''
	filter_by_machine_type()


	'''
	Creates views.instances_unique_master_list, a list of unique instances in the database. 
    Unique means that instances with naming conflicts (e.g. duplicate instanceidk's) have been removed
	'''
	create_master_instance_list()


	'''
	Creates many tables:
		views.frames_w_labels: all frames with labels plax, a4c, a2c
		views.instances_w_labels: all instances which are labeled plax, a4c, a2c
			Assumption: if a frame has a view label, other frames within that instance correspond 
                        to the same view. This discludes instances which has >1 frames with 
                        conflicting labels
		views.frames_sorted_by_views_temp: intermediate table; used by other scripts
	'''
    filter_by_views()


	'''
	Creates views.instances_w_conflicts, i.e. instances to avoid
	'''
	find_table_conflicts()

	'''
	Creates views.instances_labeled_other: all instances which are fair game for classifying as 
    others, i.e. instances which are from studies that contain labels + no conflicts. 
    The true views of these instances are not known; we only know that they haven't been 
    explicitly labeled plax, a4c, a2c
	'''
	create_unlabeled_instances()


	'''
	Creates views.instances_w_labels_{train/test} from views.instances_with_labels 
    based on a hard-coded split ratio
    
    :param ratio(float): ratio for splitting into train/test
                         e.g. if 0.8, will take 80% as train set and 20% as test set
	'''
	split_train_test(ratio=0.5)

	'''
	Creates views.instances_w_labels_{train/test}_downsampX, i.e. versions of 
    views.instances_w_labels_{train/test} that have been downsampled by a factor of X
	
    :param ratio(float): percentage by which to downsample dataset
                         e.g. if ratio=0.1, will downsample by a factor of 10
    '''
	downsample_train_test(ratio=0.1)
	
