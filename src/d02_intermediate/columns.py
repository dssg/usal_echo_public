import pandas as pd


def strip_string_column(df, column):
    """
    Get dataframe with whitespace stripped from string columns.
    
    :param df: dataframe
    :param column: string column
    :return: stripped dataframe
    """
    assert df[column].dtype == object
    df[column] = df[column].str.strip()
    return df


def drop_unknown_columns(df, columns):
    """
    Drop dataframe columns with unknown descriptions in Xcelera documentation.
    
    :param df: dataframe
    :param columns: unknown columns
    :return: updated dataframe
    """
    df = df.drop(columns, axis="columns")
    return df


def drop_rows_with_empty_string_in_column(df, column):
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
    return df


def cast_column(df, column, dtype):
    """
    Cast dataframe column to specified datatype.
    
    :param df: dataframe
    :param column: column
    :param dtype: datatype
    :return: updated dataframe
    """
    df[column] = df[column].astype(dtype)
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
    return df


def get_sorted_unique_values(df, column):
    """
    Get a list of all unique column values in sorted order.
    
    :param df: dataframe
    :param column: column
    """
    sorted_unique_values = list(df[column].sort_values().unique())
    print(f"Sorted unique values for column {column}: {sorted_unique_values}")


def count_rows_with_empty_string_in_column(df, column):
    """
    Count dataframe rows with empty strings for specified column.
    
    :param df: dataframe
    :param column: required column
    """
    start_rows = df.shape[0]
    df = df[df[column] != ""]
    end_rows = df.shape[0]
    print(f"Counted {start_rows-end_rows} rows with empty strings in column {column}")
    

def get_value_counts(df, column):
    return df[column].value_counts()

def get_number_of_values_with_count_no_greater_than_threshold(counts, threshold):
    print(f'Number of values with count <= {threshold}: {len(counts[counts<=threshold])}')

def get_value_count_values(df, column):
    return df[column].value_counts().values
