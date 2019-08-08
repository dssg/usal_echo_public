"""
This is a high-level script which runs all filtering of instances needed to create the dataset.
Summaries of each filtering steps are included below. Further details in each original script.

"""

import sys

from d02_intermediate.instance_filters import filter_machines
from d02_intermediate.instance_filters import master_list
from d02_intermediate.instance_filters import filter_views
from d02_intermediate.instance_filters import conflict_instances


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
