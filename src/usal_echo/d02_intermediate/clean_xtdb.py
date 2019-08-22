#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 3 2019

author: dssg 2019 team heartthrobs
description: the clean_xtdb script processes Xcelera_tablas database tables to have 
                a) consistent column headers 
                b) consistent treatment of missing values (either '' or -1)                
                c) consistent column data types
                d) whitespace removed from string columns
"""

import matplotlib.pyplot as plt
import pandas as pd

from usal_echo.d00_utils.db_utils import dbReadWriteRaw, dbReadWriteClean
from usal_echo.d00_utils.log_utils import setup_logging

logger = setup_logging(__name__, __name__)


def _clean_measurement_abstract_rpt(df):
    """Clean measurement_abstract_rpt table.
    
    :param df: measurement_abstract_rpt table as dataframe
    :return: cleaned dataframe
    
    """
    df.rename(columns={"studyid": "studyidk"}, inplace=True)

    for column in ["studyidk", "measabstractnumber"]:
        df[column] = df[column].astype(int)

    for column in ["name", "unitname"]:
        df[column] = df[column].str.strip()
    print("Cleaned measurement_abstract_rpt table.")

    return df


def _clean_measgraphref(df):
    """Clean measgraphref table.      
    
    :param df: measgraphref table as dataframe
    :return: cleaned table as dataframe
    
    """
    df["instanceidk"] = df["instanceidk"].replace("", -1)

    for column in ["studyidk", "measabstractnumber", "instanceidk", "indexinmglist"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(int)

    print("Cleaned measgraphref table.")

    return df


def _clean_measgraphic(df):
    """Clean measgraphic table.
    
    :param df: measgraphic table as pandas dataframe
    :return: cleaned table as dataframe
    
    """
    for column in ["instanceidk", "indexinmglist"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(int)

    print("Cleaned measgraphic table.")

    return df


def _clean_study_summary(df):
    """Clean dm_spain_view_study_summary table.
    
    In addition to processing missing values, data types and column headers, 
    this function also:
        a) adds a 'bmi' column, with outliers removed
        b) replaces missing gender information with 'U'
        c) splits the findingcode column on commas into indiviudal codes
    
    :param df: dm_spain_view_study_summary table as dataframe
    :return: cleaned table as dataframe
    
    """
    # clean age, patientweight, patientheight columns
    for column in ["age", "patientweight", "patientheight"]:
        df[column] = df[column].replace("", -1)

    for column in ["studyidk", "age"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(int)

    for column in ["patientweight", "patientheight"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # remove outliers
    for column in ["age", "patientweight", "patientheight"]:
        boxplot = plt.boxplot(df[column])
        outlier_min, outlier_max = [item.get_ydata()[0] for item in boxplot["caps"]]
        df[column] = df[column].apply(lambda x: 1 if x > outlier_max else x)
        df[column] = df[column].apply(lambda x: 1 if x < outlier_min else x)

    # create BMI column and clean outliers
    # (formula from https://www.cdc.gov/nccdphp/dnpao/growthcharts/training/bmiage/page5_1.html)
    df["bmi"] = df.apply(
        lambda x: ((x.patientweight / x.patientheight / x.patientheight) * 10000),
        axis=1,
    )
    boxplot = plt.boxplot(df["bmi"])
    outlier_min, outlier_max = [item.get_ydata()[0] for item in boxplot["caps"]]
    df["bmi"] = df["bmi"].apply(lambda x: 1 if x > outlier_max else x)
    df["bmi"] = df["bmi"].apply(lambda x: 1 if x < outlier_min else x)

    # clean gender column
    df["gender"] = df["gender"].replace("", "U")

    # clean findingcode column
    df["findingcode"] = df["findingcode"].apply(lambda x: x.split(","))

    return df


def _clean_modvolume(df):
    """Clean modvolume table.
    
    :param df: modvolume table as dataframe
    :return: cleaned table as dataframe
    
    """
    for column in ["instanceidk", "indexinmglist", "chordsequence"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(int)

    for column in ["chordtype"]:
        df[column] = df[column].str.strip()

    return df


def _clean_instance_filename(df):
    """Clean instance_filename table.
    
    :param df: instance_filename table as dataframe
    :return: cleaned table as dataframe
    
    """
    df.rename(columns={"instancedbkey": "instanceidk"}, inplace=True)

    for column in ["instanceidk", "seriesdbkey"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(int)

    for column in ["instancefilename"]:
        df[column] = df[column].str.strip()

    return df


def clean_tables():
    """Transform raw tables and write them to database schema 'clean'.
    
    """
    io_raw = dbReadWriteRaw()
    io_clean = dbReadWriteClean()

    tables_to_clean = {
        "measurement_abstract_rpt": "_clean_measurement_abstract_rpt(tbl)",
        "a_measgraphref": "_clean_measgraphref(tbl)",
        "a_measgraphic": "_clean_measgraphic(tbl)",
        "dm_spain_view_study_summary": "_clean_study_summary(tbl)",
        "a_modvolume": "_clean_modvolume(tbl)",
        "instance_filename": "_clean_instance_filename(tbl)",
    }

    for key, val in tables_to_clean.items():
        tbl = io_raw.get_table(key)
        clean_tbl = eval(val)

        io_clean.save_to_db(clean_tbl, key)
        logger.info("Created table `{}` in schema {}".format(key, io_clean.schema))
