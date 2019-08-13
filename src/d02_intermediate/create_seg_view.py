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
    
    #3. #create a dictionary to link measurements to views

    measurement_2_view_dict = {'AVItd ap4' :'A4C', 'DVItd ap4' : 'A4C', 'VTD(el-ps4)': 'A4C', 'VTD(MDD-ps4)': 'A4C', 
                               'AVIts ap4': 'A4C', 'DVIts ap4':'A4C', 'VTS(el-ps4)':'A4C', 'VTS(MDD-ps4)':'A4C', 
                               'AVItd ap2':'A2C', 'DVItd ap2':'A2C', 'VTD(el-ps2)':'A2C', 'VTD(MDD-ps2)':'A2C', 
                               'AVIts ap2':'A2C', 'DVIts ap2':'A2C', 'VTS(el-ps2)': 'A2C', 'VTS(MDD-ps2)':'A2C', 
                               'Vol. AI (MOD-sp4)':'A4C', 'Vol. AI (MOD-sp2)':'A2C'}
    
    measurement_2_chamber_dict = {'AVItd ap4' :'lv', 'DVItd ap4' : 'lv', 'VTD(el-ps4)': 'lv', 'VTD(MDD-ps4)': 'lv', 
                               'AVIts ap4': 'lv', 'DVIts ap4':'lv', 'VTS(el-ps4)':'lv', 'VTS(MDD-ps4)':'lv', 
                               'AVItd ap2':'lv', 'DVItd ap2':'lv', 'VTD(el-ps2)':'lv', 'VTD(MDD-ps2)':'lv', 
                               'AVIts ap2':'lv', 'DVIts ap2':'lv', 'VTS(el-ps2)': 'lv', 'VTS(MDD-ps2)':'lv', 
                               'Vol. AI (MOD-sp4)':'la', 'Vol. AI (MOD-sp2)':'la'}
    
    measurement_2_cardio_moment_dict = {'AVItd ap4' :'ED', 'DVItd ap4' : 'ED', 'VTD(el-ps4)': 'ED', 'VTD(MDD-ps4)': 'ED', 
                               'AVIts ap4': 'ES', 'DVIts ap4':'ES', 'VTS(el-ps4)':'ES', 'VTS(MDD-ps4)':'ES', 
                               'AVItd ap2':'ED', 'DVItd ap2':'ED', 'VTD(el-ps2)':'ED', 'VTD(MDD-ps2)':'ED', 
                               'AVIts ap2':'ES', 'DVIts ap2':'ES', 'VTS(el-ps2)': 'ES', 'VTS(MDD-ps2)':'ES', 
                               'Vol. AI (MOD-sp4)':'ES', 'Vol. AI (MOD-sp2)':'ES'}
    
    #4. # Add these dictionaries to the df_2
    df_2 = df
    df_2['view'] = df['name'].map(measurement_2_view_dict)
    df_2['chamber'] = df['name'].map(measurement_2_chamber_dict)
    df_2['cardio_moment'] = df['name'].map(measurement_2_cardio_moment_dict)
    
    del df
    
    #5# Cut out the rows without view, chamber and cadio_moment
    df_3 = df_2[pd.notna(df_2['view']) == True]
    del df_2
    
    #6.  dropping -1 instances
    df_4 = df_3[df_3['instanceidk'] != -1]
    
    #7. Only colums we need &drop duplicates
    df_5 = df_4[['studyidk', 'indexinmglist', 'instanceidk', 'frame','view',
       'chamber', 'cardio_moment']].copy()
    df_6 = df_5.drop_duplicates()
    
    del df_4
    del df_5

    # 8. Drop instance id 1789286 and 3121715 due to problems linking and
    #        conflicts
    #   Drop unnessecary row_id from orginal tables (causes problems in dbWrite)
    df_7 = df_6.drop(
        df_6[(df_6.instanceidk == 1789286) | (df_6.instanceidk == 3121715)].index
    )
    del df_6
    
    # 9. Saving to frames_by_volume_mask
    df_7.columns = map(str.lower, df_7.columns)
    io_views.save_to_db(df_7, "frames_by_volume_mask")

    # 9. merge with a_modvolume
    df_8 = pd.merge(
        a_modvolume_df, df_7, how="left", on=["instanceidk", "indexinmglist"]
    )
    del df_7
    del a_modvolume_df
    
    io_views.save_to_db(df_8, "chords_by_volume_mask")
    