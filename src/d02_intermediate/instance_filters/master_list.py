from d00_utils.db_utils import dbReadWriteRaw, dbReadWriteClean, dbReadWriteViews
from d00_utils.log_utils import *

"""
This script contains create_master_instance_list(), which creates 
the table views.instances_unique_master_list with the following columns:
    instanceidk, studyidk, instancefilename, sopinstanceuid

This master list is created in the following way:
    1) Take instanceidk from A_instance
    2) Join A_instance and instance_filename on SOPinstanceUID. Take filename from intance_filename
    3) Join A_instance and A_studyseries on studyseriesidk and take studyiinstanceidk from A_studyseries
"""


def create_master_instance_list():

    logger = setup_logging(__name__, "master_list.py")

    io_raw = dbReadWriteRaw()
    io_clean = dbReadWriteClean()
    io_views = dbReadWriteViews()

    # Get tables into df
    filename_list_df = io_clean.get_table("instance_filename")
    A_instance_df = io_raw.get_table("a_instance")
    A_study_series_df = io_raw.get_table("a_studyseries")

    # Drop unnecessary columns
    filename_list_df.drop(labels="seriesdbkey", axis=1, inplace=True)
    A_instance_df.drop(labels="sopclassuid", axis=1, inplace=True)
    A_study_series_df.drop(labels=["studyseriesuid", "modality"], axis=1, inplace=True)

    # Drop all rows which have duplicate values in specified columns
    cols_drop_filename = ["sopinstanceuid", "instanceidk", "instancefilename"]
    for col in cols_drop_filename:
        filename_list_df.drop_duplicates(subset=col, keep=False, inplace=True)
    cols_drop_A_instance = ["sopinstanceuid", "instanceidk"]
    for col in cols_drop_A_instance:
        A_instance_df.drop_duplicates(subset=col, keep=False, inplace=True)

    # Convert to int
    A_instance_df["instanceidk"] = A_instance_df["instanceidk"].astype("int")

    # First merge, i.e. step 2 in script overview
    merge_df = A_instance_df.merge(filename_list_df, on="sopinstanceuid")
    merge_df = merge_df[merge_df["instanceidk_x"] == merge_df["instanceidk_y"]]
    merge_df["instanceidk"] = merge_df["instanceidk_x"]
    cols_to_drop = ["instanceidk_x", "instanceidk_y"]
    merge_df.drop(labels=cols_to_drop, axis=1, inplace=True)

    # Second merge, i.e. step 3 in script overview
    merge2_df = merge_df.merge(A_study_series_df, on="studyseriesidk")
    merge2_df.drop(labels="studyseriesidk", axis=1, inplace=True)
    merge2_df["studyinstanceidk"] = merge2_df["studyinstanceidk"].astype("int")
    merge2_df.rename(index=str, columns={"studyinstanceidk": "studyidk"}, inplace=True)

    final_df = merge2_df
    io_views.save_to_db(final_df, "instances_unique_master_list")
    logger.info("New table created: views.instances_unique_master_list")
