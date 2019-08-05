import pandas
import os
import sys
import json
import psycopg2

from d00_utils.db_utils import dbReadWriteClean, dbReadWriteViews

def define_measurement_names():
    
    ''' Return dict of lists of measurements which define views'''

    meas_dict = {}

    meas_dict["PLAX"] = [
        "Diám raíz Ao",
        "Diám. Ao asc.",
        "Diám TSVI",
        "Dimensión AI",
    ]
    # POTENTIAL_MEASUREMENTS_PARASTERNAL_LONG_AXIS_VIEW = ['Diám TSVD', \
    #      'DVItd', 'DVIts', 'SIVtd', 'PPVItd']
    # Note: Removed 'Diam TSVD' as a measurement which would classify
    # a view as PLAX as Antonio is unsure of this, 2019_09_07
    # This change disqualifies 650 frames from being considered PLAX
    meas_dict["POTENTIAL_PLAX"] = [
        "DVItd",
        "DVIts",
        "SIVtd",
        "PPVItd",
    ]
    meas_dict["A4C"] = [
        "AVItd ap4",
        "VTD(el-ps4)",
        "VTD(MDD-ps4)",
        "VTD 4C",
        "AVIts ap4",
        "VTS(el-ps4)",
        "VTS(MDD-ps4)",
        "VTS 4C",
        "Vol. AI (MOD-sp4)",
    ]
    meas_dict["A2C"] = [
        "AVItd ap2",
        "VTD(el-ps2)",
        "VTD(MDD-ps2)",
        "VTD 2C",
        "AVIts ap2",
        "VTS(el-ps2)",
        "VTS(MDD-ps2)",
        "VTS 2C",
        "Vol. AI (MOD-sp2)",
    ]
    meas_dict["ALL_VIEWS"] = (
        MEASUREMENTS_PARASTERNAL_LONG_AXIS_VIEW
        + POTENTIAL_MEASUREMENTS_PARASTERNAL_LONG_AXIS_VIEW
        + MEASUREMENTS_APICAL_4_CHAMBER_VIEW
        + MEASUREMENTS_APICAL_2_CHAMBER_VIEW
    )

    meas_dict["END_DIASTOLIC"] = [
        "DVItd",
        "SIVtd",
        "PPVItd",
        "AVItd ap4",
        "VTD(el-ps4)",
        "VTD(MDD-ps4)",
        "VTD 4C",
        "AVItd ap2",
        "VTD(el-ps2)",
        "VTD(MDD-ps2)",
        "VTD 2C",
    ]
    meas_dict["END_SYSTOLIC"] = [
        "DVIts",
        "AVIts ap4",
        "VTS(el-ps4)",
        "VTS(MDD-ps4)",
        "VTS 4C",
        "AVIts ap2",
        "VTS(el-ps2)",
        "VTS(MDD-ps2)",
        "VTS 2C",
    ]

    return meas_dict



def filter_by_views():
    """
    Creates many tables:
        views.frames_w_labels: all frames with labels plax, a4c, a2c
        views.instances_w_labels: all instances which are labeled plax, a4c, a2c
            Assumption: if a frame has a view label, other frames within that instance correspond 
                        to the same view. This discludes instances which has >1 frames with 
                        conflicting labels
        views.frames_sorted_by_views_temp: intermediate table; used by other scripts
    """

    io_clean = dbReadWriteClean()
    io_views = dbReadWriteViews()

    measurement_abstract_rpt_df = io_clean.get_table("measurement_abstract_rpt")
    measgraphref_df = io_clean.get_table("a_measgraphref")
    measgraphic_df = io_clean.get_table("a_measgraphic")

    # Filter out unnecessary columns in each table
    measgraphref_df = measgraphref_df[
        ["studyidk", "measabstractnumber", "instanceidk", "indexinmglist"]
    ]
    measgraphic_df = measgraphic_df[["instanceidk", "indexinmglist", "frame"]]
    measurement_abstract_rpt_df = measurement_abstract_rpt_df[
        ["studyidk", "measabstractnumber", "name"]
    ]  

    # Merge individual dataframes into one
    merge_df = measgraphref_df.merge(
        measgraphic_df, on=["instanceidk", "indexinmglist"]
    )
    merge_df = merge_df.merge(
        measurement_abstract_rpt_df, on=["studyidk", "measabstractnumber"]
    )

    meas = define_measurement_names()

    # df containing all frames for which we have measurements
    filter_df = merge_df  # [merge_df.name.isin(meas["ALL_VIEWS"])].copy()

    filter_df["is_end_diastolic"] = filter_df["name"].isin(meas["END_DIASTOLIC"])
    filter_df["is_end_systolic"] = filter_df["name"].isin(meas["END_SYSTOLIC"])

    filter_df["is_plax"] = filter_df["name"].isin(meas["PLAX"])
    filter_df["maybe_plax"] = filter_df["name"].isin(meas["POTENTIAL_PLAX"])
    filter_df["is_a4c"] = filter_df["name"].isin(meas["A4C"])
    filter_df["is_a2c"] = filter_df["name"].isin(meas["A2C"])

    filter_df["view"] = ""
    filter_df.loc[filter_df["is_plax"] == True, "view"] = "plax"
    filter_df.loc[filter_df["maybe_plax"] == True, "view"] = "plax"
    filter_df.loc[filter_df["is_a4c"] == True, "view"] = "a4c"
    filter_df.loc[filter_df["is_a2c"] == True, "view"] = "a2c"

    group_df = filter_df.groupby(["instanceidk", "indexinmglist"]).first()
    group_df = group_df.drop(["measabstractnumber", "name"], axis="columns")

    group_df = group_df.reset_index()
    (
        group_df.reset_index()
        .groupby(["instanceidk", "indexinmglist"])["view"]
        .nunique()
        .eq(1)
        == False
    ).sum()
    (
        group_df.reset_index().groupby("instanceidk")["view"].nunique().eq(1) == False
    ).sum()
    is_instance_multiview = (
        group_df.reset_index().groupby("instanceidk")["view"].nunique().eq(1) == False
    ).reset_index()
    is_instance_multiview = is_instance_multiview.rename(
        index=str, columns={"view": "is_multiview"}
    )
    group_df = group_df.merge(is_instance_multiview, on="instanceidk")

    frames_with_views_df = group_df  # .merge(is_instance_multiview, on='instanceidk')
    frames_with_views_df = frames_with_views_df.drop(
        ["is_plax", "maybe_plax", "is_a4c", "is_a2c"], axis=1
    )

    # Intermediate dataframe saved to db for use by other script
    io_views.save_to_db(frames_with_views_df, 'frames_sorted_by_views_temp')

    # Remove unlabeled instances
    df2 = frames_with_views_df
    labeled_df = df2.drop(df2[(df2["view"] == "")].index)

    # Remove instances with view conflicts
    df3 = labeled_df
    conflict_sets = df3[df3["is_multiview"] == True].groupby("instanceidk")
    conflict_list = []
    for instance in list(conflict_sets.instanceidk):
        conflict_list.append(instance[0])  # get instanceidk for multidimn list
    frames_without_conflicts_df = df3[~df3["instanceidk"].isin(conflict_list)]
    labels_by_frame_df = frames_without_conflicts_df.drop("is_multiview", axis=1)

    # Remove unlabeled instances, save to database
    # df2 = frames_without_conflicts_df
    # labels_by_frame_df = df2.drop(df2[(df2['view']=='')].index)
    io_views.save_to_db(labels_by_frame_df, "frames_w_labels")

    # Group all frames of same instance, drop frame-specific columns
    agg_functions = {"view": "first", "studyidk": "first"}
    labels_by_inst_df = labels_by_frame_df.groupby(["instanceidk"]).agg(agg_functions)
    labels_by_inst_df = labels_by_inst_df.reset_index()

    # Filter out instances with naming conflicts per master table
    all_inst_df = io_views.get_table("instances_unique_master_list")
    inst_fair_game = list(set(all_inst_df["instanceidk"].tolist()))
    labels_by_inst_df = labels_by_inst_df[
        labels_by_inst_df["instanceidk"].isin(inst_fair_game)
    ]

    # Filter out instances from old machines
    new_machines_df = io_views.get_table("machines_new_bmi")
    studies_new_machines = list(set(new_machines_df["studyidk"].tolist()))
    lab_inst_new_df = labels_by_inst_df[
        labels_by_inst_df["studyidk"].isin(studies_new_machines)
    ]

    # Merge with views.instances_unique_master_list table to get the following columns:
    # sopinstanceuid, instancefilename
    master_df = io_views.get_table("instances_unique_master_list")
    merge_df = master_df.merge(lab_inst_new_df, on="instanceidk")
    merge_df = merge_df[merge_df["studyidk_x"] == merge_df["studyidk_y"]]
    merge_df["studyidk"] = merge_df["studyidk_x"]
    merge_df.drop(labels=["studyidk_x", "studyidk_y"], axis=1, inplace=True)

    io_views.save_to_db(merge_df, "instances_w_labels")
