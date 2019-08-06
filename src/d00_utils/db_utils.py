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
import gc
import psycopg2


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
        self.connection_str = "postgresql://{}:{}@{}/{}".format(
            self.credentials["user"],
            self.credentials["psswd"],
            self.credentials["host"],
            self.credentials["database"],
        )
        self.engine = create_engine(self.connection_str, encoding="utf-8")
        self.raw_conn = self.engine.raw_connection()
        self.cursor = self.raw_conn.cursor()

    def save_to_db(self, df, db_table, if_exists="replace"):
        """Write dataframe to table in database.
        
        :param df (pandas.DataFrame): dataframe to save to database
        :param db_table (str): name of database table to write to
        :param if_exists (str): write action if table exists, default='replace'
        
        """
        gc.collect()
        # Create new database table from empty dataframe
        df[:0].to_sql(db_table, self.engine, self.schema, if_exists, index=False)

        # Replace `|` so that it can be used as column separator
        for col in df.columns:
            df[col] = df[col].replace("\|", ",", regex=True)

        # Save data to temporary file to be able to use it in fast write method `copy_from`
        tmp = tempfile.NamedTemporaryFile()
        df.to_csv(tmp.name, encoding="utf-8", decimal=".", index=False, sep="|")

        connection = self.engine.raw_connection()
        cursor = connection.cursor()

        with open(tmp.name, "r") as f:
            next(f)  # Skip the header row.
            cursor.copy_from(
                f, "{}.{}".format(self.schema, db_table), sep="|", size=100000, null=""
            )
            connection.commit()

        gc.collect()

        print(
            "Saved table {} to schema {} (mode={})".format(
                db_table, self.schema, if_exists
            )
        )

    def get_table(self, db_table):
        """Read table in database as dataframe.
        
        :param db_table (str): name of database table to read
        
        """
        # Fetch column names
        q = "SELECT * FROM {}.{} LIMIT(0)".format(self.schema, db_table)
        cols = pd.read_sql(q, self.engine).columns.to_list()

        tmp = tempfile.NamedTemporaryFile()
        connection = self.engine.raw_connection()
        cursor = connection.cursor()

        with open(tmp.name, "w") as f:
            cursor.copy_to(
                f, "{}.{}".format(self.schema, db_table), columns=cols, null=""
            )
        connection.commit()

        df = pd.read_csv(tmp.name, sep="\t", names=cols)
        df.fillna("", inplace=True)

        gc.collect()

        return df

    def list_tables(self):
        """List tables in database.
        
        """
        inspector = inspect(self.engine)
        print(inspector.get_table_names(self.schema))


class dbReadWritePublic(dbReadWriteData):
    """
    TODO: delete this class when other schemas properly populated
    Instantiates class for postres I/O to 'public' schema 
    """

    def __init__(self):
        super().__init__(schema="public")
        if not self.engine.dialect.has_schema(self.engine, self.schema):
            self.engine.execute(CreateSchema(self.schema))


class dbReadWriteRaw(dbReadWriteData):
    """
    Instantiates class for postres I/O to 'raw' schema 
    """

    def __init__(self):
        super().__init__(schema="raw")
        if not self.engine.dialect.has_schema(self.engine, self.schema):
            self.engine.execute(CreateSchema(self.schema))


class dbReadWriteClean(dbReadWriteData):
    """
    Instantiates class for postgres I/O to 'clean' schema
    """

    def __init__(self):
        super().__init__(schema="clean")
        if not self.engine.dialect.has_schema(self.engine, self.schema):
            self.engine.execute(CreateSchema(self.schema))


class dbReadWriteViews(dbReadWriteData):
    """
    Instantiates class for postgres I/O to 'view' schema
    """

    def __init__(self):
        super().__init__(schema="views")
        if not self.engine.dialect.has_schema(self.engine, self.schema):
            self.engine.execute(CreateSchema(self.schema))


class dbReadWriteSegmentation(dbReadWriteData):
    """
    Instantiates class for postgres I/O to 'segmentation' schema
    """

    def __init__(self):
        super().__init__(schema="segmentation")
        if not self.engine.dialect.has_schema(self.engine, self.schema):
            self.engine.execute(CreateSchema(self.schema))
        
            
    def save_numpy_array_to_db(self, np_array, table_name):
        binary_data = psycopg2.Binary(np_array)
        sql = "insert into {} values({})".format(table_name, binary_data)
        self.cursor.execute(sql)
        self.raw_conn.commit()
    
    def get_numpy_array_from_db(self, column_name, table_name):
        sql = 'select {} from {}'.format(column_name, table_name)
        self.cursor.execute(sql)
        results = self.cursor.fetchone()[0] #TODO we need to actually retrieve all of them and iterate
        
        return np.reshape(np.frombuffer(results, dtype='Int8'),(384,384))

    
    
