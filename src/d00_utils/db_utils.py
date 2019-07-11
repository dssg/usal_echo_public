#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 3 2019

@author: wiebket
"""

import pandas as pd
import os
from json import load
from sqlalchemy import create_engine
from sqlalchemy.schema import CreateSchema
from sqlalchemy import inspect
import tempfile

def _load_json_credentials(filepath):
    """Load json formatted credentials.
    
    :params filepath (str): path to credentials file
    "returns: credentials as dict
    
    """
    with open(filepath) as f:
        credentials = load(f)

    return credentials


class dbReadWriteData:   
    """
    Class for reading and writing data to and from postgres database.
    
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
    def __init__(self, schema=None, credentials_file="~/.psql_credentials.json"):
        self.filepath = os.path.expanduser(credentials_file)
        self.schema = schema
        self.credentials = _load_json_credentials(self.filepath)
        self.connection_str =  "postgresql://{}:{}@{}/{}".format(self.credentials['user'],
                                                             self.credentials['psswd'],
                                                             self.credentials['host'],
                                                             self.credentials['database'])
        self.engine = create_engine(self.connection_str, encoding='utf-8')


    def save_to_db(self, df, db_table, if_exists='replace'):
        """Write dataframe to table in database.
        
        :param df (pandas.DataFrame): dataframe to save to database
        :param db_table (str): name of database table to write to
        :param if_exists (str): write action if table exists, default='replace'
        
        """        
        # Create new database table from empty dataframe
        df[:0].to_sql(db_table, self.engine, self.schema, if_exists, index=False)
        
        # Replace `|` so that it can be used as column separator
        for col in df.columns:
            df[col] = df[col].replace('\|',',', regex=True) 
    
        # Save data to temporary file to be able to use it in fast write method `copy_from`
        tmp = tempfile.NamedTemporaryFile()
        df.to_csv(tmp.name, encoding='utf-8', decimal='.', index=False, sep='|')

        connection = self.engine.raw_connection()
        cursor = connection.cursor()           
        
        with open(tmp.name, 'r') as f:
            next(f) # Skip the header row.
            cursor.copy_from(f, '{}.{}'.format(self.schema, db_table), sep='|', size=100000) 
            connection.commit()
            
    
    def get_table(self, db_table):
        """Read table in database as dataframe.
        
        :param db_table (str): name of database table to read
        
        """
        #TODO speed up reading from db         
        df = pd.read_sql_table(db_table, self.engine, self.schema)
        
        return df
    
    
    def list_tables(self):
        """List tables in database.
        
        """
        inspector = inspect(self.engine)
        print(inspector.get_table_names(self.schema))
       
        
    
class dbReadWriteRaw(dbReadWriteData):
    """Subclass for reading and writing data to and from `raw` schema.
    
    """    
    def __init__(self):
        super().__init__(schema='raw')
        if not self.engine.dialect.has_schema(self.engine, self.schema):
            self.engine.execute(CreateSchema(self.schema))

            
            
class dbReadWriteClean(dbReadWriteData):
    """Subclass for reading and writing data to and from `clean` schema.
    
    """    
    def __init__(self):
        super().__init__(schema='clean')
        if not self.engine.dialect.has_schema(self.engine, self.schema):
            self.engine.execute(CreateSchema(self.schema))