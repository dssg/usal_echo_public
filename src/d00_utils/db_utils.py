from json import load
from sqlalchemy import create_engine


def _load_json_credentials(filepath):
    """Load json formatted credentials.
    
    :params filepath (str): path to credentials file
    "returns: credentials as dict
    
    """
    with open(filepath) as f:
        credentials = load(f)

    return credentials


def save_to_db(df, db_table, credentials_file):
    """Write dataframe to table in database.
    
    **Requirements
    credentials file formatted as:
        {
        "user":"your_user",
        "host": "your_server.rds.amazonaws.com",
        "database": "your_database",
        "psswd": "your_password"
        }
    
    :params df (pandas.DataFrame): dataframe to save to database
    :params db_table (str): name of database table to write to
    :params credentials_file (str): path to credentials file
    
    """
    credentials = _load_json_credentials(credentials_file)
    connection_str =  "postgresql://{}:{}@{}/{}".format(credentials['user'],
                                                         credentials['psswd'],
                                                         credentials['host'],
                                                         credentials['database'])
    conn = create_engine(connection_str)
    df.to_sql(db_table, conn, if_exists='append', index=False)