# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:26:54 2019

@author: court
"""
from src.d00_utils.db_utils import dbReadWriteClean, dbReadWriteViews
import pandas as pd
import numpy as np


def create_seg_view():
    """
    Produces a table of labelled volume mask segmentations
    The following steps are performend:
        1. Import a_modvolume, a_measgraphref, a_measgraphic, 
            measurement_abstract_rpt
        2. Create df by merging a_measgraphref_df, measurement_abstract_rpt_df
            on 'studyidk' and 'measabstractnumber'
        3. Create one hot encoded columns for measurements of interest based 
        on measurement names
        4. Create one shot encode columns for segment names based on one 
        hot encoded columns for measurements
        5. Drop one hot encoded columns for measurements of interest
        6. Cut the dataframe to relevant columns as drop duplicates
        7. merge with a_measgraphic_df for frame number
        8. Cut the dataframe to relevant columns as drop duplicates
        9. merge with a_modvolume
        10. Drop instance id 1789286 and 3121715 due to problems linking and 
        conflicts.  Drop unnessecary row_id from orginal tables (causes 
        problems in dbWrite)
        11. write to db

    
    :param: 
    :return df: the merge dataframe containing the segment labels
    """

    io_clean = dbReadWriteClean()
    io_views = dbReadWriteViews()

    # 1. Import a_modvolume, a_measgraphref, a_measgraphic,
    #        measurement_abstract_rpt
    a_modvolume_df = io_clean.get_table("a_modvolume")
    a_measgraphref_df = io_clean.get_table("a_measgraphref")
    a_measgraphic_df = io_clean.get_table("a_measgraphic")
    measurement_abstract_rpt_df = io_clean.get_table("measurement_abstract_rpt")

    # 2. merge a_measgraphref_df, measurement_abstract_rpt_df
    df = pd.merge(
        a_measgraphref_df,
        measurement_abstract_rpt_df,
        how="left",
        on=["studyidk", "measabstractnumber"],
    )
    del a_measgraphref_df
    del measurement_abstract_rpt_df

    # 3. Create one hot encoded columns for measurements of interest based
    #        on measurement names
    df["AVItd ap4"] = pd.np.where(df.name.str.contains("AVItd ap4"), True, False)
    df["DVItd ap4"] = pd.np.where(df.name.str.contains("DVItd ap4"), True, False)
    df["VTD(el-ps4)"] = pd.np.where(df.name.str.contains("VTD\(el-ps4\)"), True, False)
    df["VTD(MDD-ps4)"] = pd.np.where(
        df.name.str.contains("VTD\(MDD-ps4\)"), True, False
    )

    df["AVIts ap4"] = pd.np.where(df.name.str.contains("AVIts ap4"), True, False)
    df["DVIts ap4"] = pd.np.where(df.name.str.contains("DVIts ap4"), True, False)
    df["VTS(el-ps4)"] = pd.np.where(df.name.str.contains("VTS\(el-ps4\)"), True, False)
    df["VTS(MDD-ps4)"] = pd.np.where(
        df.name.str.contains("VTS\(MDD-ps4\)"), True, False
    )

    df["AVItd ap2"] = pd.np.where(df.name.str.contains("AVItd ap2"), True, False)
    df["DVItd ap2"] = pd.np.where(df.name.str.contains("DVItd ap2"), True, False)
    df["VTD(el-ps2)"] = pd.np.where(df.name.str.contains("VTD\(el-ps2\)"), True, False)
    df["VTD(MDD-ps2)"] = pd.np.where(
        df.name.str.contains("VTD\(MDD-ps2\)"), True, False
    )

    df["AVIts ap2"] = pd.np.where(df.name.str.contains("AVIts ap2"), True, False)
    df["DVIts ap2"] = pd.np.where(df.name.str.contains("DVIts ap2"), True, False)
    df["VTS(el-ps2)"] = pd.np.where(df.name.str.contains("VTS\(el-ps2\)"), True, False)
    df["VTS(MDD-ps2)"] = pd.np.where(
        df.name.str.contains("VTS\(MDD-ps2\)"), True, False
    )

    df["Vol. AI (MOD-sp4)"] = pd.np.where(
        df.name.str.contains("Vol. AI \(MOD-sp4\)"), True, False
    )
    df["Vol. AI (MOD-sp2)"] = pd.np.where(
        df.name.str.contains("Vol. AI \(MOD-sp2\)"), True, False
    )

    # 4. Create one shot encode columns for segment names based on one
    #        hot encoded columns for measurements
    df["A4C_Ven_ED"] = np.where(
        (
            (df["AVItd ap4"] == True)
            | (df["DVItd ap4"] == True)
            | (df["VTD(el-ps4)"] == True)
            | (df["VTD(MDD-ps4)"] == True)
        ),
        True,
        False,
    )

    df["A4C_Ven_ES"] = np.where(
        (
            (df["AVIts ap4"] == True)
            | (df["DVIts ap4"] == True)
            | (df["VTS(el-ps4)"] == True)
            | (df["VTS(MDD-ps4)"] == True)
        ),
        True,
        False,
    )

    df["A2C_Ven_ED"] = np.where(
        (
            (df["AVItd ap2"] == True)
            | (df["DVItd ap2"] == True)
            | (df["VTD(el-ps2)"] == True)
            | (df["VTD(MDD-ps2)"] == True)
        ),
        True,
        False,
    )

    df["A2C_Ven_ES"] = np.where(
        (
            (df["AVIts ap2"] == True)
            | (df["DVIts ap2"] == True)
            | (df["VTS(el-ps2)"] == True)
            | (df["VTS(MDD-ps2)"] == True)
        ),
        True,
        False,
    )

    df["A4C_Atr_ES"] = np.where((df["Vol. AI (MOD-sp4)"] == True), True, False)
    df["A2C_Atr_ES"] = np.where((df["Vol. AI (MOD-sp2)"] == True), True, False)

    # 5. Drop one hot encoded columns for measurements of interest
    df.drop(
        [
            "AVItd ap4",
            "DVItd ap4",
            "VTD(el-ps4)",
            "VTD(MDD-ps4)",
            "AVIts ap4",
            "DVIts ap4",
            "VTS(el-ps4)",
            "VTS(MDD-ps4)",
            "AVItd ap2",
            "DVItd ap2",
            "VTD(el-ps2)",
            "VTD(MDD-ps2)",
            "AVIts ap2",
            "DVIts ap2",
            "VTS(el-ps2)",
            "VTS(MDD-ps2)",
            "Vol. AI (MOD-sp4)",
            "Vol. AI (MOD-sp2)",
        ],
        axis=1,
        inplace=True,
    )

    # 6. Cut the dataframe to relevant columns as drop duplicates
    df_1 = df[
        [
            "studyidk",
            "instanceidk",
            "indexinmglist",
            "A4C_Ven_ED",
            "A4C_Ven_ES",
            "A2C_Ven_ED",
            "A2C_Ven_ES",
            "A4C_Atr_ES",
            "A2C_Atr_ES",
        ]
    ].copy()
    del df
    df_2 = df_1.drop_duplicates()
    del df_1

    #  7. merge with a_measgraphic_df for frame number
    df_3 = pd.merge(
        df_2, a_measgraphic_df, how="left", on=["instanceidk", "indexinmglist"]
    )
    del df_2
    del a_measgraphic_df

    # 8. Cut the dataframe to relevant columns as drop duplicates
    df_4 = df_3[
        [
            "studyidk",
            "instanceidk",
            "indexinmglist",
            "A4C_Ven_ED",
            "A4C_Ven_ES",
            "A2C_Ven_ED",
            "A2C_Ven_ES",
            "A4C_Atr_ES",
            "A2C_Atr_ES",
            "frame",
        ]
    ].copy()
    del df_3

    # 9. merge with a_modvolume
    df_5 = pd.merge(
        a_modvolume_df, df_4, how="left", on=["instanceidk", "indexinmglist"]
    )
    del df_4
    del a_modvolume_df

    # 10. Drop instance id 1789286 and 3121715 due to problems linking and
    #        conflicts
    #   Drop unnessecary row_id from orginal tables (causes problems in dbWrite)
    df_6 = df_5.drop(
        df_5[(df_5.instanceidk == 1789286) | (df_5.instanceidk == 3121715)].index
    )
    del df_5

    df_6.columns = map(str.lower, df_6.columns)
    
    # write output to db
    io_views.save_to_db(df_6, "chords_by_volume_mask")

    # New transformations added later - may need to revise scripts to make more
    # efficient :see git issue
    # 11. melt the table to a single field for the views
    df_7 = pd.melt(
        df_6,
        id_vars=[
            "instanceidk",
            "indexinmglist",
            "chordsequence",
            "chordtype",
            "x1coordinate",
            "y1coordinate",
            "chordlength",
            "x2coordinate",
            "y2coordinate",
            "interchorddistance",
            "studyidk",
            "frame",
        ],
        value_vars=[
            "a4c_ven_ed",
            "a4c_ven_es",
            "a2c_ven_ed",
            "a2c_ven_es",
            "a4c_atr_es",
            "a2c_atr_es",
        ],
        var_name="view_name",
        value_name="view_exists",
    )
    del df_6

    # 12. drop chord sequence and remove duplicates
    df_8 = df_7.drop(
        [
            "chordsequence",
            "chordtype",
            "x1coordinate",
            "y1coordinate",
            "chordlength",
            "x2coordinate",
            "y2coordinate",
            "interchorddistance",
        ],
        axis=1,
    )
    df_9 = df_8.drop_duplicates()
    del df_7
    del df_8

    # 13. create detailed columns
    # create view column
    df_9["view_only"] = df_9["view_name"].apply(lambda x: str(x)[:3])
    # create ventricle column
    df_9["ventricle_only"] = df_9["view_name"].apply(lambda x: str(x)[4:7])
    # create cycle column
    df_9["cycle_only"] = df_9["view_name"].apply(lambda x: str(x)[8:10])

    # 14. # reset index
    df_9.reset_index()

    # write output to db
    io_views.save_to_db(df_9, "frames_by_volume_mask")
