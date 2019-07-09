import pandas as pd


def clean_measurement_abstract_rpt(df):
    """Clean measurement_abstract_rpt table.

    The following cleaning steps are performend:
        1. strip string columns: `name`, `unitname`
    
    :param df: measurement_abstract_rpt table as dataframe
    :return: cleaned dataframe
    
    """
    for column in ["name", "unitname"]:
        df[column] = df[column].str.strip()
        print(f"Stripped string column {column}")

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


def clean_all(measurement_abstract_rpt_df, measgraphref_df, measgraphic_df):
    """
    Return cleaned measurement tables
    
    :param:
    :return: list(measurement_abstract_rpt_df, measgraphref_df, measgraphic_df)
    
    """
    measurement_abstract_rpt_df = clean_measurement_abstract_rpt(measurement_abstract_rpt_df)
    measgraphref_df = clean_measgraphref(measgraphref_df)
    measgraphic_df = clean_measgraphic(measgraphic_df)
    
    return measurement_abstract_rpt_df, measgraphref_df, measgraphic_df
