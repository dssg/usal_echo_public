import pandas
import os
import sys
import json
import psycopg2

from d00_utils.db_utils import dbReadWriteClean, dbReadWriteViews

def get_connection():
    '''
    Currently this function is unused
    Establish connection to psql database
    Requirement: .psql_credentials.json in root directory
    '''
    filename = os.path.expanduser('~') + '/.psql_credentials.json'
    with open(filename) as f:
        conf = json.load(f)
        
        connection_string = "postgresql://{}:{}@{}/{}".format(
        conf["user"], conf["psswd"], conf["host"], conf["database"])
    
        conn = sqlalchemy.create_engine(connection_string, pool_pre_ping=True)
        
        return conn

def filter_by_views():
    '''
    Input: from postgres db schema 'clean', the following tables:
        measurement_abstract_rpt, measgraphref, measgraphic
    Joins tables and filters out only frames that have view labels

    Output: to postgres db schema 'views', the table 'frames_sorted_by_views'
    Outputs to db schema 'frames_sorted_by_views table with the following attributes:
        is_end_diastolic
        is_end_systolic
        view
        is_multiview

    '''

    io_clean = dbReadWriteClean()
    io_views = dbReadWriteViews()

    measurement_abstract_rpt_df = io_clean.get_table('measurement_abstract_rpt')
    measgraphref_df = io_clean.get_table('a_measgraphref')
    measgraphic_df = io_clean.get_table('a_measgraphic')

    # Filter out unnecessary columns in each table
    measgraphref_df = measgraphref_df[['studyidk', \
        'measabstractnumber', 'instanceidk', 'indexinmglist']]
    measgraphic_df = measgraphic_df[['instanceidk', \
        'indexinmglist', 'frame']]
    measurement_abstract_rpt_df = measurement_abstract_rpt_df[[\
        'studyidk', 'measabstractnumber', 'name']] #here

    # Merge individual dataframes into one
    merge_df = measgraphref_df.merge(measgraphic_df, on=[\
        'instanceidk', 'indexinmglist'])
    merge_df = merge_df.merge(measurement_abstract_rpt_df, on=[\
        'studyidk', 'measabstractnumber'])


    # Define measurement names
    MEASUREMENTS_PARASTERNAL_LONG_AXIS_VIEW = ['Diám raíz Ao', \
        'Diám. Ao asc.', 'Diám TSVI', 'Dimensión AI']
    # POTENTIAL_MEASUREMENTS_PARASTERNAL_LONG_AXIS_VIEW = ['Diám TSVD', \
    #      'DVItd', 'DVIts', 'SIVtd', 'PPVItd']
    # Note: Removed 'Diam TSVD' as a measurement which would classify
    # a view as PLAX as Antonio is unsure of this, 2019_09_07
    # This change disqualifies 650 frames from being considered PLAX
    POTENTIAL_MEASUREMENTS_PARASTERNAL_LONG_AXIS_VIEW = [ \
         'DVItd', 'DVIts', 'SIVtd', 'PPVItd']
    MEASUREMENTS_APICAL_4_CHAMBER_VIEW = ['AVItd ap4', 'VTD(el-ps4)', \
         'VTD(MDD-ps4)', 'VTD 4C', 'AVIts ap4', 'VTS(el-ps4)', \
         'VTS(MDD-ps4)', 'VTS 4C', 'Vol. AI (MOD-sp4)']
    MEASUREMENTS_APICAL_2_CHAMBER_VIEW = ['AVItd ap2', 'VTD(el-ps2)', \
        'VTD(MDD-ps2)', 'VTD 2C', 'AVIts ap2', 'VTS(el-ps2)', \
        'VTS(MDD-ps2)', 'VTS 2C', 'Vol. AI (MOD-sp2)']
    ALL_MEASUREMENTS = MEASUREMENTS_PARASTERNAL_LONG_AXIS_VIEW + \
        POTENTIAL_MEASUREMENTS_PARASTERNAL_LONG_AXIS_VIEW + \
        MEASUREMENTS_APICAL_4_CHAMBER_VIEW + MEASUREMENTS_APICAL_2_CHAMBER_VIEW

    MEASUREMENTS_END_DIASTOLIC = ['DVItd', 'SIVtd', 'PPVItd', \
    'AVItd ap4', 'VTD(el-ps4)', 'VTD(MDD-ps4)', 'VTD 4C', 'AVItd ap2', \
    'VTD(el-ps2)', 'VTD(MDD-ps2)', 'VTD 2C']
    MEASUREMENTS_END_SYSTOLIC = ['DVIts', 'AVIts ap4', 'VTS(el-ps4)', \
        'VTS(MDD-ps4)', 'VTS 4C', 'AVIts ap2', 'VTS(el-ps2)', \
        'VTS(MDD-ps2)', 'VTS 2C']


    # df containing all frames for which we have measurements
    filter_df = merge_df#[merge_df.name.isin(ALL_MEASUREMENTS)].copy()

    filter_df['is_end_diastolic'] = filter_df['name'].isin(MEASUREMENTS_END_DIASTOLIC)
    filter_df['is_end_systolic'] = filter_df['name'].isin(MEASUREMENTS_END_SYSTOLIC)

    filter_df['is_plax'] = filter_df['name'].isin(MEASUREMENTS_PARASTERNAL_LONG_AXIS_VIEW)
    filter_df['maybe_plax'] = filter_df['name'].isin(POTENTIAL_MEASUREMENTS_PARASTERNAL_LONG_AXIS_VIEW)
    filter_df['is_a4c'] = filter_df['name'].isin(MEASUREMENTS_APICAL_4_CHAMBER_VIEW)
    filter_df['is_a2c'] = filter_df['name'].isin(MEASUREMENTS_APICAL_2_CHAMBER_VIEW)

    filter_df['view'] = ''
    filter_df.loc[filter_df['is_plax']==True, 'view'] = 'plax'
    filter_df.loc[filter_df['maybe_plax']==True, 'view'] = 'plax'
    filter_df.loc[filter_df['is_a4c']==True, 'view'] = 'a4c'
    filter_df.loc[filter_df['is_a2c']==True, 'view'] = 'a2c'

    group_df = filter_df.groupby(['instanceidk', 'indexinmglist']).first()
    group_df = group_df.drop(['measabstractnumber', 'name'], axis='columns')

    group_df = group_df.reset_index()
    (group_df.reset_index().groupby(['instanceidk', 'indexinmglist'])['view'].nunique().eq(1)==False).sum()
    (group_df.reset_index().groupby('instanceidk')['view'].nunique().eq(1)==False).sum()
    is_instance_multiview = (group_df.reset_index().groupby('instanceidk')['view'].nunique().eq(1)==False).reset_index()
    is_instance_multiview = is_instance_multiview.rename(index=str, columns={"view": "is_multiview"})
    group_df = group_df.merge(is_instance_multiview, on='instanceidk')


    frames_with_views_df = group_df#.merge(is_instance_multiview, on='instanceidk')
    frames_with_views_df = frames_with_views_df.drop(['is_plax', 'maybe_plax', 'is_a4c', 'is_a2c'], axis=1)
    
    io_views.save_to_db(frames_with_views_df, 'frames_sorted_by_views')
