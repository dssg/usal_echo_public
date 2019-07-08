from json import load
from sqlalchemy import create_engine
import pandas as pd
import os


def _load_json_credentials(filepath):
    """Load json formatted credentials.
    
    :params filepath (str): path to credentials file
    "returns: credentials as dict
    
    """
    with open(filepath) as f:
        credentials = load(f)

    return credentials


class ReadWriteData:   
    """
    Class for reading and writing data to postgres database.
    
    **Requirements
        credentials file formatted as:
            {
            "user":"your_user",
            "host": "your_server.rds.amazonaws.com",
            "database": "your_database",
            "psswd": "your_password"
            }
            
    :param credentials_file (str): path to credentials file, default="~/.psql_credentials.json"
    :param schema (str): database schema 
            
    """    
    def __init__(self, schema, credentials_file="~/.psql_credentials.json"):
        self.filepath = os.path.expanduser(credentials_file)
        self.schema = schema
        

    def save_to_db(self, df, db_table):
        """Write dataframe to table in database.
        
        :params df (pandas.DataFrame): dataframe to save to database
        :params db_table (str): name of database table to write to
        
        """
        credentials = _load_json_credentials(self.filepath)
        connection_str =  "postgresql://{}:{}@{}/{}".format(credentials['user'],
                                                             credentials['psswd'],
                                                             credentials['host'],
                                                             credentials['database'])
        conn = create_engine(connection_str)
        df.to_sql("{}.{}".format(self.schema, db_table), conn, if_exists='append', index=False)
        
    
    def read_from_db(self, db_table):
        """Read dataframe from table in database
        
        """
    
        credentials = _load_json_credentials(self.filepath)
        connection_str =  "postgresql://{}:{}@{}/{}".format(credentials['user'],
                                                             credentials['psswd'],
                                                             credentials['host'],
                                                             credentials['database'])
        conn = create_engine(connection_str)
        query = ("select * from {};").format(self.schema, db_table)    
        df = pd.read_sql(query, conn)
        
        return df
       
    
class ReadWriteRaw(ReadWriteData):
    
    def __init__(self, credentials_file):
        super(ReadWriteRaw, self).__init__(schema='raw', credentials_file)
    
    
class ReadWriteEncode(ReadWriteData):
    
    def __init__(self, credentials_file):
        super(ReadWriteRaw, self).__init__(schema='encode', credentials_file)


class ReadWriteClean(ReadWriteData):
    
    def __init__(self, credentials_file):
        super(ReadWriteRaw, self).__init__(schema='clean', credentials_file)
    