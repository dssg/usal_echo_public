import pandas as pd


def get_table_name_to_string_columns():
    """
    Get mapping of table name to list of string columns.
    :return: mapping
    """
    return {"measurement_abstract_rpt": ["name", "unitname"]}


def get_table_name_to_nonempty_string_columns():
    """
    Get mapping of table name to list of string columns that cannot be empty.
    """
    return {"a_measgraphref": ["instanceidk"]}


def get_table_name_to_unknown_columns():
    """
    Get mapping of table name to list of columns with unknown descriptions.
    :return: mapping
    """
    return {
        "a_measgraphref": ["srinstanceidk"],
        "a_measgraphic": [
            "graphictoolidk",
            "longaxisindex",
            "measidk",
            "loopidk",
            "instancerecordtype",
        ],
    }


def get_table_name_to_nonnegative_columns():
    """
    Get mapping of table name to list of columns with columns that cannot be negative.
    :return: mapping
    """
    return {"a_measgraphref": ["instanceidk", "indexinmglist"]}


def get_table_name_to_int_cast_columns():
    """
    Get mapping of table name to list of columns with columns that should be integers.
    :return: mapping
    """
    return {"a_measgraphref": ["instanceidk", "indexinmglist"]}


def strip_string_column(df, column):
    """
    Get dataframe with whitespace stripped from string columns.
    
    :param df: dataframe
    :param column: string column
    :return: stripped dataframe
    """
    assert df[column].dtype == object
    df[column] = df[column].str.strip()
    print(f"Stripped string column {column}")
    return df


def drop_unknown_column(df, column):
    """
    Drop dataframe column with unknown descriptions in Xcelera documentation.
    
    :param df: dataframe
    :param column: unknown column
    :return: updated dataframe
    """
    df = df.drop(column, axis="columns")
    print(f"Dropped unknown column {column}")
    return df


def drop_rows_with_empty_string(df, column):
    """
    Drop dataframe rows with empty strings for specified column.
    
    :param df: dataframe
    :param column: required column
    :return: updatetd dataframe
    """
    start_rows = df.shape[0]
    df = df[df[column] != ""]
    end_rows = df.shape[0]
    print(f"Dropped {start_rows-end_rows} rows with empty strings in column {column}")
    print(f"{end_rows} rows remaining")
    return df


def cast_column_to_int(df, column):
    """
    Cast dataframe column to specified datatype.
    
    :param df: dataframe
    :param column: column
    :param dtype: datatype
    :return: updated dataframe
    """
    df[column] = df[column].astype(int)
    return df


def drop_rows_with_negative_value_in_column(df, column):
    """
    Drop dataframe rows with negative values for specified column.
    
    :param df: dataframe
    :param column: column
    :return: updated dataframe
    """
    start_rows = df.shape[0]
    df = df[df[column] >= 0]
    end_rows = df.shape[0]
    print(f"Dropped {start_rows-end_rows} rows with negative values in column {column}")
    print(f"{end_rows} rows remaining")
    return df
