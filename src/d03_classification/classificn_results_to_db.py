import os
from os import listdir
from os.path import isfile, join
import sys
import pandas as pd

from d00_utils.db_utils import dbReadWriteData

def results_txt_to_db():
    '''
    Takes output of text file from view classification model output
    Processes results into a database table _______.______
    '''

    probs_path = '/home/ubuntu/data/d03_classification/probabilities/' #TO DO: soft-code this
    sys.path.append(probs_path)

    text_files = [f for f in listdir(probs_path) if isfile(join(probs_path,f))]
    f = text_files[0] # TO DO: soft-code this, i.e. don't always get first file in directory

    infile = open(probs_path + f)
    infile = infile.readlines()
    infile = [i.rstrip() for i in infile]
    infile = [i.split('\t') for i in infile]

    # Write txt data to dataframe, basic processing
    df = pd.DataFrame(infile)

    df.columns = infile[0]
    df.drop(0, inplace=True)
    df.drop(labels='study', axis=1, inplace=True)

    # Take str column 'image' and make columns 'study', 'instance'
    fn_list = list(df['image'].tolist())
    study_list = []
    inst_list = []
    for fn in fn_list:
        sub = fn.split('_')
        study_list.append(sub[1])
        subsub = sub[2].split('.')
        inst_list.append(subsub[0])

    df['study'] = study_list
    df['instancefilename'] = inst_list

    cols = df.columns.tolist()
    cols_new = cols[-2:] + cols[1:-2] # rearrange cols
    df_new = df[cols_new]

    # Get ground truth labels via views.instances_w_labels table
    io_views = dbReadWriteData()
    io_views.schema = 'views'
    labels_df = io_views.get_table('instances_w_labels')
    labels_df.rename(columns={'filename':'instancefilename'}, inplace=True)

    # Merge tables df_new and labels_df
    merge_df = df_new.merge(labels_df, on='instancefilename')

    # Reorder, drop, rename columns. Convert probabilities cols to numeric
    cols = merge_df.columns.tolist()
    cols_new = cols[1:2] + cols[-2:-1] + cols[2:-4] + cols[-3:-2] + cols[-1:]
    merge_df = merge_df[cols_new]
    cols_to_rename = {"view": "view_true"}
    merge_df.rename(index=str, columns=cols_to_rename, inplace=True)
    cols_all = list(merge_df.columns.values)
    cols_probs = cols_all[2:-2]
    merge_df[cols_probs] = merge_df[cols_probs].apply(pd.to_numeric)

    # average probabilities over ten frames in an instance
    cols_groupby = ['instancefilename', 'studyidk', 'view_true', 'instanceidk']
    agg_df = merge_df.groupby(cols_groupby)[cols_probs].mean()
    agg_df = agg_df.reset_index(drop=False)

    # predicted view is the column with maximum probability
    agg_df['view_pred'] = agg_df[cols_probs].idxmax(axis=1)
    agg_df.rename(columns={'view_pred':'view23_pred'}, inplace=True)

    # reorder columns for visual inspection
    cols = agg_df.columns.tolist()
    cols_new = cols[0:1] + cols[2:3] + cols[-1:] + cols[4:-1] + cols[3:4] + cols[1:2]
    agg_df = agg_df[cols_new]

    # Map b/w view_true, view_pred

    #view_pred_unique = list(set(comp_df['view_pred'].tolist()))

    # Define a loose mapping from 23-class model to 4-class prediction
    # e.g. can map e.g. all a2c --> 'a2c'
    plax_views = ['plax_plax', 'plax_lac', 'plax_far', 'plax_laz']
    a4c_views = ['a4c', 'a4c_laocc', 'a4c_lvocc_s']
    a2c_views = ['a2c_lvocc_s', 'a2c', 'a2c_laocc']
    other_views = ['a3c_laocc', 'rvinf', 'a3c', 'suprasternal', \
                   'psax_az', 'a3c_lvocc_s', 'other', 'psax_avz', \
                   'subcostal', 'a5c']
    maps_m1 = {'plax': plax_views, 
               'a4c': a4c_views,
               'a2c': a2c_views,
               'other': other_views
              }

    # Define a strict mapping from 23-class model to 4-class prediction
    # e.g. only those classified as pure a2c will be used for measurements
    other_views_m2_m3 = other_views + ['plax_lac', 'plax_far', \
                    'plax_laz', 'a4c_laocc', 'a4c_lvocc_s', \
                    'a2c_lvocc_s', 'a2c_laocc']
    maps_m2_m3 = {'plax': ['plax_plax'], 'a2c': ['a2c'], \
                  'a4c': ['a4c'], 'other': other_views_m2_m3}

    corr_df = agg_df
    view23_pred = agg_df['view23_pred'].tolist()
    view4_pred = []
    for pred in view23_pred:
    #     view4_pred.append([key for key, element in maps_m1.items() if pred in element][0])
        view4_pred.append([key for key, element in maps_m2_m3.items() if pred in element][0])
    corr_df['view4_pred'] = view4_pred

    # Add column to compare whether prediction is true or not
    corr_df['correct'] = corr_df['view_true'] == corr_df['view4_pred']

    # Reorder columns for visual inspection
    cols = corr_df.columns.tolist()
    cols_new = cols[0:1] + cols[-1:] + cols[1:2] + cols[-2:-1] + cols[2:3] + cols[3:-2]
    corr_df = corr_df[cols_new]

    io_views.save_to_db(corr_df, 'output_m1_test1000_aug8_copy')

