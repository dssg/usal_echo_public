#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 3 2019


@author: wiebket
"""

import pandas as pd
import numpy as np
import os
from json import load
from sqlalchemy import create_engine
from sqlalchemy.schema import CreateSchema
from sqlalchemy import inspect
import tempfile
import gc
import psycopg2

from d00_utils.log_utils import *

logger = setup_logging(__name__, "save_to_db")


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

        logger.info(
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

    def save_prediction_numpy_array_to_db(self, binary_data_array, column_names):
        sql = "insert into {}.{} ({}) values ('{}', '{}', '{}', '{}', {}, {}, {}, '{}', '{}', '{}', '{}', '{}')".format(
            self.schema,
            'predictions',
            ",".join(column_names),
            binary_data_array[0],
            binary_data_array[1],
            binary_data_array[2],
            binary_data_array[3],
            psycopg2.Binary(binary_data_array[4]),
            psycopg2.Binary(binary_data_array[5]),
            psycopg2.Binary(binary_data_array[6]),
            binary_data_array[7],
            binary_data_array[8],
            binary_data_array[9],
            binary_data_array[10],
            binary_data_array[11]
        )
        self.cursor.execute(sql)
        self.raw_conn.commit()

        print("Saved to table {} to schema {} ".format('predictions', self.schema))
        
    def save_ground_truth_numpy_array_to_db(self, binary_data_array, column_names):  
    #column_names = ['ground_truth_id, study_id, instance_id', 'file_name', 
    #'frame', 'chamber', 'view_name' 'numpy_array'
        sql = "insert into {}.{} ({}) values ('{}', '{}', '{}', '{}', '{}', '{}', {})".format(
            self.schema,
            'ground_truths',
            ",".join(column_names),
            binary_data_array[0],
            binary_data_array[1],
            binary_data_array[2],
            binary_data_array[3],
            binary_data_array[4],
            binary_data_array[5],
            psycopg2.Binary(binary_data_array[6])
        )
        self.cursor.execute(sql)
        self.raw_conn.commit()

        print("Saved to table {} to schema {} ".format('ground truth', self.schema))

    def convert_to_np(self, x, frame):
        if frame == 1:
            np_array = np.reshape(np.frombuffer(x, dtype='uint8'), (384,384))
        else:
            np_array = np.reshape(np.frombuffer(x, dtype='Int8'), (frame,384,384))
            
        return np_array
    
    
    def get_segmentation_table(self, db_table):
        """Read table in database as dataframe.
        
        :param db_table (str): name of database table to read
        
        """
        
        q = "SELECT * FROM {}.{}".format(self.schema, db_table)
        df = pd.read_sql(q, self.engine)

        return df
    
    def get_instance_from_segementation_table(self, db_table, instance):
        
        q = "SELECT * FROM {}.{} WHERE instance_id='{}'".format(self.schema, db_table, instance)
        df = pd.read_sql(q, self.engine)

        return df
    
    def save_seg_evaluation_to_db(self, df, column_names, if_exists="append"):
        # Evaluation Table: evaluation_id, instance_id, frame, chamber, study_id, score_type, score_value
        
        # Create new database table from empty dataframe
        #df.to_sql('evaluation', self.engine, self.schema, if_exists, index=False)
        sql = "insert into {}.{} ({}) values ('{}', '{}', '{}', '{}', '{}', '{}')".format(
            self.schema,
            'evaluations',
            ",".join(column_names),
            df[0],
            df[1],
            df[2],
            df[3],
            df[4],
            df[5]
        )
        self.cursor.execute(sql)
        self.raw_conn.commit()

        print(
            "Saved table {} to schema {} (mode={})".format(
                'evaluation', self.schema, if_exists
            )
        )
        
