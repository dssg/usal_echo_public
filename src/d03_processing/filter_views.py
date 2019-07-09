import pandas as pandas
import sys
sys.path.append('../src')

from d01_data.access_database import *
from d03_processing.clean_measurements import *



def get_sqlalchemy_connection():
	filename = os.path.expanduser('~') + '/.psql_credentials.json'
	with open(filename) as f:
		conf = json.load(f)

		connection_string = "postgresql://{}:{}@{}/{}".format(
        conf["user"], conf["psswd"], conf["host"], conf["database"])

    conn = sqlalchemy.create_engine(connection_string)

    return conn



if __name__ == '__main__':
	# path = get_db_config_path()
	# conf = get_db_config(path)
	# conf = get_postgres_credentials()
	conn = get_sqlalchemy_connection(conf)

	# Read tables from postgres db, clean
	measurement_abstract_rpt_df, measgraphref_df, measgraphic_df = \
		read_measurement_tables(conn)
	measurement_abstract_rpt_df, measgraphref_df, measgraphic_df = \
		clean_measurement_datatframes(measurement_abstract_rpt_df, \
			measgraphref_df, measgraphic_df)

	# Filter out unnecessary columns in each table
	measgraphref_df = measgraphref_df[['studyidk', \
		'measabstractnumber', 'instanceidk', 'indexinmglist']]
	measgraphic_df = measgraphic_df[['instanceidk', \
		'indexinmglist', 'frame']]
	measurement_abstract_rpt_df = measurement_abstract_rpt_df[[\
		'studyid', 'measabstractnumber', 'name']]
	measurement_abstract_rpt_df = measurement_abstract_rpt_df.rename(\
		index=str, columns={"studyid": "studyidk"})

	# Merge individual dataframes into one
	merge_df = measgraphref_df.merge(measgraphic_df, on=[\
		'instanceidk', 'indexinmglist'])
	merge_df = merge_df.merge(measurement_abstract_rpt_df, on=[\
		'studyidk', 'measabstractnumber'])

	# Define measurement names
	MEASUREMENTS_PARASTERNAL_LONG_AXIS_VIEW = ['Diám raíz Ao', \
		'Diám. Ao asc.', 'Diám TSVI', 'Dimensión AI']
	POTENTIAL_MEASUREMENTS_PARASTERNAL_LONG_AXIS_VIEW = ['Diám TSVD', \
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
	filter_df = merge_df[merge_df.name.isin(ALL_MEASUREMENTS)].copy()

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

group_df = filter_df.groupby(['instanceidk', 'frame']).first()
group_df = group_df.drop(['measabstractnumber', 'name'], axis='columns')

is_instance_multiview = (group_df.reset_index().groupby('instanceidk')['view'].nunique().eq(1)==False).reset_index()
is_instance_multiview = is_instance_multiview.rename(index=str, columns={"view": "is_multiview"})

frames_with_views_df = group_df.merge(is_instance_multiview, on='instanceidk')

