#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 5 2019

@author: wiebket
"""

import logging
import os
from pathlib import Path
import pandas as pd


log_basedir = os.path.join(Path(__file__).parents[2], "log")


def setup_logging(name, log_file, level=logging.DEBUG):
    """Create a logging object that writes to log_file.
    
    :param name (str): 
    :param log_file (str): log file name
    :param level (int): logging.level, default=logging.DEBUG
    
    """
    os.makedirs(log_basedir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    handler = logging.FileHandler(os.path.join(log_basedir, log_file + ".log"))
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_logs(log_file, level=None):
    """Retrieve and filter log messages from log_file.
    
    :param log_file (str): log file name
    :param level (str): log level for filtering, default=None
    
    """
    df = pd.read_csv(
        os.path.join(log_basedir, log_file + ".log"),
        sep=" - ",
        names=["timestamp", "module", "level", "message"],
        engine="python",
    )
    if level is not None:
        df = df[df["level"] == level]

    return df
