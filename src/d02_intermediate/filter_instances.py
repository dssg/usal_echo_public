import sys

from d02_intermediate.instance_filters import filter_machines
from d02_intermediate.instance_filters import master_list
from d02_intermediate.instance_filters import filter_views
from d02_intermediate.instance_filters import conflict_instances
from d02_intermediate.instance_filters import unlabeled_instances
from d02_intermediate.instance_filters import train_test_split
from d02_intermediate.instance_filters import downsample_data


"""
This is a high-level script which runs all filtering of instances needed to create the dataset.
Summaries of each filtering steps are included below. Further details in each original script.
"""


def filter_all():

    """ 
    Creates two tables:
        views.machines_all_bmi: list of all studies in db; columns: studyidk, machine type and bmi
        views.machines_new_bmi: same as machines_all_bmi, but only includes studies with new machines, 
                                i.e. machine types ECOEPIQ2, EPIQ7-1, ECOIE33, AFFINITI_1, AFFINITI_2
    """
    filter_machines.filter_by_machine_type()

    """
    Creates views.instances_unique_master_list, a list of unique instances in the database. 
    Unique means that instances with naming conflicts (e.g. duplicate instanceidk's) have been removed
    """
    master_list.create_master_instance_list()

    """
    Creates many tables:
        views.frames_w_labels: all frames with labels plax, a4c, a2c
        views.instances_w_labels: all instances which are labeled plax, a4c, a2c
            Assumption: if a frame has a view label, other frames within that instance correspond 
                        to the same view. This discludes instances which has >1 frames with 
                        conflicting labels
        views.frames_sorted_by_views_temp: intermediate table; used by other scripts
    """
    filter_views.filter_by_views()

    """
    Creates views.instances_w_conflicts, i.e. instances to avoid
    """
    conflict_instances.find_table_conflicts()

    """
    Creates views.instances_unlabeled: all instances which are fair game for classifying as 
    others, i.e. instances which are from studies that contain labels + no conflicts. 
    The true views of these instances are not known; we only know that they haven't been 
    explicitly labeled plax, a4c, a2c
    """
    unlabeled_instances.create_unlabeled_instances()

    ##########################################################################
    ##TO DO, WIEBKE: integrate below scripts further down pipe, delete here
    """
    Creates views.instances_w_labels_{train/test} from views.instances_with_labels 
    based on a hard-coded split ratio
    
    :param ratio(float): ratio for splitting into train/test
                         e.g. if 0.8, will take 80% as train set and 20% as test set
    """
    # train_test_split.split_train_test(ratio=0.5)

    """
    Creates views.instances_w_labels_{train/test}_downsampX, i.e. versions of 
    views.instances_w_labels_{train/test} that have been downsampled by a factor of X
    
    :param ratio(float): percentage by which to downsample dataset
                         e.g. if ratio=0.1, will downsample by a factor of 10
    """
    # downsample_data.downsample_train_test(ratio=0.1)
