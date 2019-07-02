import pandas as pd


def strip_string_columns(df, columns):
    """
    Get dataframe with whitespace stripped from string columns.
    
    :param df: dataframe
    :param columns: string columns
    :return: stripped dataframe
    """
    for column in columns:
        assert df[column].dtype == object
        df[column] = df[column].str.strip()

    return df
