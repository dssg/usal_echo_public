import pandas as pd


def clean_measurement_abstract_rpt_df(df):
    """
    Strip string columns in measurement_abstract_rpt_df
    :return df: cleaned dataframe
    """
    for column in ["name", "unitname"]:
        df[column] = df[column].str.strip()
        print(f"Stripped string column {column}")

    return df


def clean_measgraphref_df(df):
    """
    Drop rows with negative indexes and unknown columns in measgraphref_df
    :return df: cleaned dataframe
    """
    num_start_rows = df.shape[0]
    for column in ["instanceidk"]:
        df = df[df[column] != ""]
        num_end_rows = df.shape[0]
        print(
            f"Dropped {num_start_rows-num_end_rows} rows with empty strings in column {column}, {num_end_rows} rows remaining"
        )

    for column in ["instanceidk", "indexinmglist"]:
        df = df.copy()
        df[column] = df[column].astype(int)
        print(f"Cast column {column} to int")

    for column in ["instanceidk", "indexinmglist"]:
        num_start_rows = df.shape[0]
        df = df[df[column] >= 0]
        num_end_rows = df.shape[0]
        print(
            f"Dropped {num_start_rows-num_end_rows} rows with negative values in column {column}, {num_end_rows} rows remaining"
        )

    for column in ["srinstanceidk"]:
        df = df.drop(column, axis="columns")
        print(f"Dropped unknown column {column}")

    return df


def clean_measgraphic_df(df):
    """
    Drop unknown columns in measgraphic_df
    :return df: cleaned dataframe
    """
    for column in [
        "graphictoolidk",
        "longaxisindex",
        "measidk",
        "loopidk",
        "instancerecordtype",
    ]:
        df = df.drop(column, axis="columns")
        print(f"Dropped unknown column {column}")

    return df


def clean_measurement_datatframes(
    measurement_abstract_rpt_df, measgraphref_df, measgraphic_df
):
    """
    Return cleaned measurement tables
    :return measurement_abstract_rpt_df: study measurement dataframe
    :return measgraphref_df: instance measurement dataframe
    :return measgraphic_df: frame dataframe
    """
    print("Cleaning datatframe measurement_abstract_rpt_df")
    measurement_abstract_rpt_df = clean_measurement_abstract_rpt_df(
        measurement_abstract_rpt_df
    )
    print("Cleaning datatframe measgraphref_df")
    measgraphref_df = clean_measgraphref_df(measgraphref_df)
    print("Cleaning datatframe measgraphic_df")
    measgraphic_df = clean_measgraphic_df(measgraphic_df)
    return measurement_abstract_rpt_df, measgraphref_df, measgraphic_df
