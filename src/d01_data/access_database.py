import pandas as pd
import sqlalchemy
import yaml


def get_db_config_path():
    """
    Get path to local configuration file with database credentials
    
    :return: path to config file
    """
    # TODO: document in README
    return "../conf/local/db.yaml"


def get_db_config(path):
    """
    Get configuration object with database credentials
    
    :param path: path to config file
    :return: config object
    """
    with open(path) as f:
        conf = yaml.safe_load(f)

    return conf


def get_sqlalchemy_connection(conf):
    """
    Get SQLAlchemy Engine for database specified in configuration object
    
    :param conf: config object
    :return: database connection
    """
    connection_string = "postgresql://{}:{}@{}/{}".format(
        conf["user"], conf["pw"], conf["host"], conf["DB"]
    )
    conn = sqlalchemy.create_engine(connection_string)

    return conn


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
