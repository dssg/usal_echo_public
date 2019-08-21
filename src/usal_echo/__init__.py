#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
from pathlib import Path

usr_dir = os.path.join(str(Path.home()),'usr','usal_echo','conf')

with open(os.path.join(usr_dir, "path_parameters.yml")) as f:
    paths = yaml.safe_load(f)

bucket = paths["bucket"]
dcm_dir = os.path.expanduser(paths["dcm_dir"])
img_dir = os.path.expanduser(paths["img_dir"])
segmentation_dir = os.path.expanduser(paths["segmentation_dir"])
model_dir = os.path.expanduser(paths["model_dir"])
classification_model = paths["classification_model"]


from usal_echo.d00_utils.db_utils import *
from usal_echo.d00_utils.s3_utils import download_s3_objects
from usal_echo.d00_utils.log_utils import get_logs
from usal_echo.d01_data.ingestion_dcm import ingest_dcm
from usal_echo.d01_data.ingestion_xtdb import ingest_xtdb
from usal_echo.d02_intermediate.clean_dcm import clean_dcm_meta
from usal_echo.d02_intermediate.clean_xtdb import clean_tables
from usal_echo.d02_intermediate.filter_instances import filter_all
from usal_echo.d02_intermediate.download_dcm import decompress_dcm, s3_download_decomp_dcm, dcmdir_to_jpgs_for_classification
from usal_echo.d03_classification.predict_views import run_classify, agg_probabilities, predict_views
from usal_echo.d03_classification.evaluate_views import evaluate_views
from usal_echo.d04_segmentation.create_seg_view import create_seg_view
from usal_echo.d04_segmentation.segment_view import run_segment
from usal_echo.d04_segmentation.generate_masks import generate_masks
from usal_echo.d04_segmentation.evaluate_masks import evaluate_masks
from usal_echo.d05_measurement.retrieve_meas import retrieve_meas
from usal_echo.d05_measurement.calculate_meas import calculate_meas
from usal_echo.d05_measurement.evaluate_meas import evaluate_meas
from usal_echo.d06_visualisation.confusion_matrix import plot_confusion_matrix
