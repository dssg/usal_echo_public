import pandas as pd
import sqlalchemy
import yaml



def read_table(conn, name):
    """
    Get dataframe from SQL table
    
    :param conn: database connection
    :param name: table name
    :return: dataframe of table
    """
    query = f"select * from {name}"
    df = pd.read_sql(query, conn)
    print(f"Table {name} has {df.shape[0]} rows and {df.shape[1]} columns")
    return df
