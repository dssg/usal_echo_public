import psycopg2

from json import load
from sqlalchemy import create_engine


def _load_db_credentials(filepath):
    with open(filepath) as f:
        credentials = load(f)

    return credentials


def save_to_db(df, db_table, credentials_file):
    credentials = _load_db_credentials(credentials_file)
    connection_str =  "postgresql://{}:{}@{}/{}".format(credentials['user'],
                                                         credentials['psswd'],
                                                         credentials['host'],
                                                         credentials['database'])
    conn = create_engine(connection_str)
    df.to_sql(db_table, conn, if_exists='append', index=False)