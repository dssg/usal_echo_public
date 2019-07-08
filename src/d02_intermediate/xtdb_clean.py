# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

from ..d00_utils.db_utils import ReadWriteClean



def clean_measurement_abstract_rpt(df):
    """Clean measurement_abstract_rpt table.

    The following cleaning steps are performend:
        1. strip string columns: `name`, `unitname`
    
    :param df: measurement_abstract_rpt table as dataframe
    :return: cleaned dataframe
    
    """
    for column in ["name", "unitname"]:
        df[column] = df[column].str.strip()
    print("Cleaned measurement_abstract_rpt table.")

    return df


def clean_measgraphref(df):
    """Clean measgraphref table.
    
    The following cleaning steps are performend:
        1. remove rows with empty `instanceidk` column
        2. transform `instanceidk` and `indexinmglist` values to integer datatype
        3. remove rows with negative values in `instanceidk` and `indexinmglist`
        4. drop `srinstanceidk` column           
    
    :param df: measgraphref table as dataframe
    :return: cleaned table as dataframe
    
    """
    df_noempty = df[df["instanceidk"] != ""]
    
    for column in ["instanceidk", "indexinmglist"]:
        df_noempty[column] = df_noempty[column].astype(int)
        
    df_noempty_positive = df_noempty[(df_noempty["instanceidk"] >= 0) &  
                                     (df_noempty["indexinmglist"] >= 0)]

    df_clean = df_noempty_positive.drop("srinstanceidk", axis="columns")
    print("Cleaned measgraphref table.")

    return df_clean


def clean_measgraphic(df):
    """Clean measgraphic table.
    
    The following cleaning steps are performend:
        1. drop the following columns: ["graphictoolidk","longaxisindex",
                                        "measidk","loopidk","instancerecordtype"]
    
    :param df: measgraphic table as pandas dataframe
    :return: cleaned table as dataframe
    
    """
    df_clean = df.drop(columns=["graphictoolidk","longaxisindex","measidk",
                          "loopidk","instancerecordtype"])
    print("Cleaned measgraphic table.")

    return df_clean


def clean_summary_table(df):
    """Clean summary table.
    
    The following cleaning steps are performend:
        1. replace empty values with 1 in columns `age`, `patientweight`, `patientheight`
        2. replace `,` with `.` in columns `age`, `patientweight`, `patientheight`
    
    :param df: 
    :return: cleaned table as dataframe
    
    """
    # clean age, patientweight, patientheight columns
    column_names_to_clean = ['age', 'patientweight', 'patientheight']
    for column in column_names_to_clean:
        df[column] = df[column].replace('', 1) 
        df[column] = df[column].str.replace(',', '.') 
        df[column] = df[column].fillna(1)
    
    if df['age'].dtype != 'int64':
         df['age'] = df['age'].astype('int64')
    if df['patientweight'].dtype != 'float64':
         df['patientweight'] = df['patientweight'].astype('float64')
    if df['patientheight'].dtype != 'float64':
         df['patientheight'] = df['patientheight'].astype('float64')
    
    # remove outliers
    column_names_to_clean = ['age', 'patientweight', 'patientheight']
    for column in column_names_to_clean:
        boxplot = plt.boxplot(df[column])
        outlier_min, outlier_max = [item.get_ydata()[0] for item in boxplot['caps']]
        df[column] = df[column].apply(lambda x: 1 if x > outlier_max else x)
        df[column] = df[column].apply(lambda x: 1 if x < outlier_min else x)

    # create BMI column and clean outliers
    # (formula from https://www.cdc.gov/nccdphp/dnpao/growthcharts/training/bmiage/page5_1.html)
    df['bmi'] = df.apply(lambda x: ((x.patientweight/x.patientheight/x.patientheight)*10000), axis=1)
    boxplot = plt.boxplot(df['bmi']);
    outlier_min, outlier_max = [item.get_ydata()[0] for item in boxplot['caps']]
    df['bmi'] = df['bmi'].apply(lambda x: 1 if x > outlier_max else x)
    df['bmi'] = df['bmi'].apply(lambda x: 1 if x < outlier_min else x)
    
    # clean gender column
    df['gender'] = df['gender'].replace('', 'U')
    
    # clean findingcode column
    df['findingcode'] = df['findingcode'].apply(lambda x: x.split(","))
    
    return df


def clean_tables():
    """
    Produces the clean summary table
    """
    
    tables_to_clean = dict('': pd.DataFrame()),
                           '':'',
                           '':'',
                           '':'')
    
    summary_table_df = clean_summary_table()
    measurement_abstract_rpt_df = clean_measurement_abstract_rpt(measurement_abstract_rpt_df)
    measgraphref_df = clean_measgraphref(measgraphref_df)
    measgraphic_df = clean_measgraphic(measgraphic_df)
    
    return summary_table_df
